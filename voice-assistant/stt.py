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
import wave
from typing import Optional

import numpy as np

from config import CHANNELS, SAMPLE_RATE, WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL

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
