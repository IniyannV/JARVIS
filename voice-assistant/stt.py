"""
Speech-to-text engine wrapping faster-whisper.

Model choice rationale:
- tiny.en (~39M params): ~0.3-0.5s on Apple Silicon M-series. Sufficient for
  short voice commands (3-10 words). Accuracy is good for common app names and
  system commands.
- base.en (~74M params): ~0.8-1.2s. Better accent robustness. Recommended if
  tiny.en misrecognizes commands frequently — change WHISPER_MODEL in config.py.
- We default to tiny.en for the best latency on voice-command workloads.
"""

import io
import logging
import inspect
import time
import wave
from typing import Optional

import numpy as np

from config import (
    CHANNELS,
    INTENT_CONFIDENCE_THRESHOLD,
    PARTIAL_TRANSCRIPT_INTERVAL,
    SAMPLE_RATE,
    STT_WINDOW_SECONDS,
    VOICE_ACTIVITY_THRESHOLD,
    WHISPER_STREAM_LOGPROB_THRESHOLD,
    WHISPER_STREAM_NO_SPEECH_THRESHOLD,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL,
)

logger = logging.getLogger("voice-assistant.stt")

# Module-level model singleton — loaded once, reused for every transcription.
_model = None


def _get_model():
    """Lazy-load the Whisper model on first call."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model '%s' (device=%s, compute=%s)…",
            WHISPER_MODEL,
            WHISPER_DEVICE,
            WHISPER_COMPUTE_TYPE,
        )
        _model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        logger.info("Whisper model loaded.")
    return _model


def transcribe(audio_bytes: bytes) -> Optional[str]:
    """
    Transcribe a WAV audio buffer to text.

    Args:
        audio_bytes: Raw bytes of a 16-bit PCM WAV file at SAMPLE_RATE Hz, mono.

    Returns:
        Stripped transcript string, or None if nothing was detected.
    """
    if not audio_bytes:
        logger.warning("transcribe() called with empty audio buffer.")
        return None

    model = _get_model()

    # Decode WAV bytes → float32 numpy array in [-1, 1]
    with io.BytesIO(audio_bytes) as buf:
        with wave.open(buf, "rb") as wf:
            raw_frames = wf.readframes(wf.getnframes())
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()

    # Convert to int16 then float32
    audio_np = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0

    # Mix down to mono if needed
    if n_channels > 1:
        audio_np = audio_np.reshape(-1, n_channels).mean(axis=1)

    logger.debug("Transcribing %.2f seconds of audio…", len(audio_np) / SAMPLE_RATE)

    segments, info = model.transcribe(
        audio_np,
        language="en",
        beam_size=1,           # fastest inference path
        vad_filter=True,       # skip silent chunks
        vad_parameters={"min_silence_duration_ms": 300},
    )

    parts = [seg.text.strip() for seg in segments if seg.text.strip()]
    if not parts:
        logger.info("No speech detected in audio.")
        return None

    transcript = " ".join(parts).strip()
    logger.info("Transcript: %s", transcript)
    return transcript


def _transcribe_supported(model, audio_np: np.ndarray, kwargs: dict):
    """
    Call faster-whisper transcribe() but only pass kwargs supported by the
    installed version (API varies across releases).
    """
    try:
        sig = inspect.signature(model.transcribe)
        supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    except (TypeError, ValueError):
        supported = kwargs
    return model.transcribe(audio_np, **supported)


class StreamingSTT:
    """
    Incremental STT over a rolling audio window.

    Intended usage:
      - A non-audio-thread worker calls process_audio_chunk() with float32
        mono audio chunks in [-1, 1].
      - Every PARTIAL_TRANSCRIPT_INTERVAL seconds, it transcribes the latest
        STT_WINDOW_SECONDS of audio and may emit a partial transcript.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        window_seconds: float = STT_WINDOW_SECONDS,
        interval_seconds: float = PARTIAL_TRANSCRIPT_INTERVAL,
    ) -> None:
        self._sample_rate = sample_rate
        self._window_samples = int(sample_rate * window_seconds)
        self._interval_seconds = interval_seconds
        self._buffer = np.zeros((0,), dtype=np.float32)
        self._last_run = 0.0
        self._last_emitted: str = ""
        self._speech_active_until = 0.0

    def process_audio_chunk(self, chunk: np.ndarray, rms: float) -> Optional[str]:
        """
        Ingest one mono float32 chunk and maybe return a partial transcript.

        Returns:
          A partial transcript string when updated (and gated by confidence),
          otherwise None.
        """
        if chunk is None or len(chunk) == 0:
            return None

        now = time.monotonic()
        if rms > VOICE_ACTIVITY_THRESHOLD:
            # Keep STT "hot" briefly after speech energy appears.
            self._speech_active_until = max(self._speech_active_until, now + 0.9)

        # Append and trim rolling window.
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        self._buffer = np.concatenate([self._buffer, chunk], axis=0)
        if self._buffer.size > self._window_samples:
            self._buffer = self._buffer[-self._window_samples :]

        # Rate-limit STT runs.
        if now - self._last_run < self._interval_seconds:
            return None
        self._last_run = now

        # If we've had no speech energy recently, skip to avoid wasted CPU.
        if now > self._speech_active_until:
            return None

        model = _get_model()
        audio_np = self._buffer
        if audio_np.size < int(self._sample_rate * 0.4):
            return None

        try:
            segments, _info = _transcribe_supported(
                model,
                audio_np,
                {
                    "language": "en",
                    "beam_size": 1,
                    "best_of": 1,
                    "temperature": 0.0,
                    "vad_filter": True,
                    "vad_parameters": {"min_silence_duration_ms": 150},
                    # faster-whisper uses `log_prob_threshold` (API varies by version)
                    "log_prob_threshold": WHISPER_STREAM_LOGPROB_THRESHOLD,
                    "no_speech_threshold": WHISPER_STREAM_NO_SPEECH_THRESHOLD,
                    "condition_on_previous_text": False,
                },
            )
        except Exception as exc:
            logger.warning("Streaming transcription failed: %s", exc)
            return None

        texts = []
        logprobs = []
        for seg in segments:
            seg_text = getattr(seg, "text", "").strip()
            if seg_text:
                texts.append(seg_text)
                avg_lp = getattr(seg, "avg_logprob", None)
                if isinstance(avg_lp, (int, float)):
                    logprobs.append(float(avg_lp))

        if not texts:
            return None

        transcript = " ".join(texts).strip()
        if not transcript:
            return None

        # Confidence gate: use avg_logprob when available, else heuristic.
        confidence = None
        if logprobs:
            mean_lp = sum(logprobs) / len(logprobs)
            # Map approx [-2.0, -0.2] -> [0, 1]
            confidence = max(0.0, min(1.0, (mean_lp + 2.0) / 1.8))
        else:
            confidence = 1.0 if len(transcript) >= 6 else 0.5

        if confidence < INTENT_CONFIDENCE_THRESHOLD:
            return None

        # De-duplicate noisy re-emissions.
        if transcript == self._last_emitted:
            return None
        if self._last_emitted and transcript.startswith(self._last_emitted) and len(transcript) - len(self._last_emitted) < 3:
            return None

        self._last_emitted = transcript
        return transcript
