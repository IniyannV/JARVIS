"""
Microphone capture with silence detection.

Records audio until:
  - The stop_event is set (hotkey toggled OFF), OR
  - SILENCE_DURATION seconds of continuous silence detected (RMS < SILENCE_THRESHOLD), OR
  - MAX_RECORDING_DURATION seconds elapsed (hard cap)

Returns raw WAV bytes ready for the STT engine.
"""

import io
import logging
import threading
import time
import wave
from typing import Optional

import numpy as np
import sounddevice as sd

import state
from config import (
    CHANNELS,
    CHUNK_DURATION,
    MAX_RECORDING_DURATION,
    MUTE_MIC_WHILE_SPEAKING,
    SAMPLE_RATE,
    SILENCE_DURATION,
    SILENCE_THRESHOLD,
    SPEECH_INTERRUPT_CONFIRM_CHUNKS,
    SPEECH_INTERRUPT_GRACE_SEC,
    STREAM_CHUNK_MS,
    VOICE_ACTIVITY_THRESHOLD,
)

logger = logging.getLogger("voice-assistant.audio")


def _rms(chunk: np.ndarray) -> float:
    """Compute root-mean-square energy of an audio chunk."""
    x = chunk.astype(np.float32)
    # Normalize integer PCM to [-1, 1] so SILENCE_THRESHOLD is meaningful.
    if np.issubdtype(chunk.dtype, np.integer):
        max_val = float(np.iinfo(chunk.dtype).max)
        if max_val > 0:
            x = x / max_val
    return float(np.sqrt(np.mean(x ** 2)))


def record(stop_event: threading.Event) -> Optional[bytes]:
    """
    Capture audio from the default microphone until stop_event or silence.

    Args:
        stop_event: Set this to stop recording early (e.g., hotkey toggled OFF).

    Returns:
        WAV-encoded bytes (16-bit PCM, 16 kHz, mono), or None if nothing was captured.
    """
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
    silence_chunks_needed = int(SILENCE_DURATION / CHUNK_DURATION)
    max_chunks = int(MAX_RECORDING_DURATION / CHUNK_DURATION)

    frames: list[np.ndarray] = []
    silence_counter = 0
    speech_detected = False
    stream = None

    logger.info("Audio capture started (SR=%d, chunk=%.2fs).", SAMPLE_RATE, CHUNK_DURATION)

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=chunk_samples,
        )
        stream.start()

        for _ in range(max_chunks):
            if stop_event.is_set():
                logger.info("Audio capture stopped by stop_event.")
                break

            chunk, overflowed = stream.read(chunk_samples)
            if overflowed:
                logger.debug("Audio buffer overflowed — some samples dropped.")

            # chunk shape: (chunk_samples, CHANNELS)
            mono = chunk[:, 0] if chunk.ndim > 1 else chunk
            energy = _rms(mono)

            if energy > SILENCE_THRESHOLD:
                speech_detected = True
                silence_counter = 0
                frames.append(mono.copy())
            else:
                if speech_detected:
                    # Count trailing silence only after speech starts
                    silence_counter += 1
                    frames.append(mono.copy())  # include trailing silence in buffer
                    if silence_counter >= silence_chunks_needed:
                        logger.info(
                            "%.1fs of silence detected — stopping capture.",
                            SILENCE_DURATION,
                        )
                        break
                # Pre-speech silence: capture but don't count toward recording
                # so we get a clean leading edge on the audio

        else:
            logger.warning("Max recording duration reached.")

    finally:
        if stream is not None:
            stream.stop()
            stream.close()
            logger.debug("Microphone stream closed.")

    if not frames:
        logger.info("No audio frames captured.")
        return None

    if not speech_detected:
        logger.info("No speech detected (all silence).")
        return None

    # Concatenate and encode as WAV
    audio_data = np.concatenate(frames, axis=0)
    return _encode_wav(audio_data)


def stream(
    stop_event: threading.Event,
    on_chunk,
    on_pause=None,
    on_voice_start=None,
) -> None:
    """
    Stream mic audio chunks until stop_event is set.

    - Never blocks on downstream work: on_chunk should return quickly.
    - Updates dashboard mic meter.
    - Can suppress mic input while TTS is playing to avoid self-transcription.

    Args:
        stop_event: signal to stop.
        on_chunk: callable(chunk_f32: np.ndarray, rms: float, ts: float) -> None
        on_pause: optional callable() invoked when a brief pause is detected.
        on_voice_start: optional callable() invoked when speech starts after silence.
    """
    chunk_duration = STREAM_CHUNK_MS / 1000.0
    chunk_samples = max(1, int(SAMPLE_RATE * chunk_duration))
    logger.info("Audio streaming started (SR=%d, chunk=%dms).", SAMPLE_RATE, STREAM_CHUNK_MS)

    stream_obj = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=chunk_samples,
    )
    stream_obj.start()

    last_voice_ts = 0.0
    pause_fired = False
    last_speech_log = 0.0
    speech_active = False
    interrupt_candidate_chunks = 0

    try:
        while not stop_event.is_set():
            chunk, overflowed = stream_obj.read(chunk_samples)
            if overflowed:
                logger.debug("Audio buffer overflowed — some samples dropped.")

            mono_i16 = chunk[:, 0] if chunk.ndim > 1 else chunk
            chunk_f32 = mono_i16.astype(np.float32) / 32768.0
            rms = _rms(chunk_f32)
            now = time.monotonic()

            speaker_is_active = state.speaker is not None and state.speaker.is_speaking
            if speaker_is_active and MUTE_MIC_WHILE_SPEAKING:
                speech_active = False
                pause_fired = False
                last_voice_ts = 0.0
                interrupt_candidate_chunks = 0
                if state.dashboard is not None:
                    try:
                        state.dashboard.update_mic_level(0.0)
                    except Exception:
                        pass
                continue

            if state.dashboard is not None:
                try:
                    state.dashboard.update_mic_level(rms)
                except Exception:
                    pass

            # Only interrupt TTS after sustained mic energy, and not immediately
            # after TTS starts, to avoid cutting off on speaker bleed-through.
            if speaker_is_active and rms > (VOICE_ACTIVITY_THRESHOLD * 1.5):
                interrupt_candidate_chunks += 1
            else:
                interrupt_candidate_chunks = 0

            if (
                speaker_is_active
                and state.speaker is not None
                and (now - state.speaker.started_at) >= SPEECH_INTERRUPT_GRACE_SEC
                and interrupt_candidate_chunks >= SPEECH_INTERRUPT_CONFIRM_CHUNKS
            ):
                try:
                    state.speaker.interrupt()
                except Exception:
                    pass
                interrupt_candidate_chunks = 0

            if rms > VOICE_ACTIVITY_THRESHOLD:
                if not speech_active:
                    speech_active = True
                    if on_voice_start is not None and not speaker_is_active:
                        try:
                            on_voice_start()
                        except Exception:
                            pass
                last_voice_ts = now
                pause_fired = False
                if now - last_speech_log >= 1.0:
                    last_speech_log = now
                    logger.debug("Speech energy detected (rms=%.4f).", rms)
            else:
                if last_voice_ts and (now - last_voice_ts) >= 0.6 and not pause_fired:
                    pause_fired = True
                    speech_active = False
                    if on_pause is not None:
                        try:
                            on_pause()
                        except Exception:
                            pass

            try:
                on_chunk(chunk_f32, rms, now)
            except Exception:
                # Never allow downstream failures to stop mic capture.
                pass
    finally:
        try:
            stream_obj.stop()
            stream_obj.close()
        finally:
            if state.dashboard is not None:
                try:
                    state.dashboard.update_mic_level(0.0)
                except Exception:
                    pass
        logger.info("Audio streaming stopped.")


def _encode_wav(audio: np.ndarray) -> bytes:
    """
    Encode a 1-D int16 numpy array as a WAV file in memory.

    Returns raw WAV bytes.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.astype(np.int16).tobytes())
    return buf.getvalue()


def assert_mic_closed(stream: Optional[object]) -> None:
    """
    Assert that the microphone stream is fully closed.

    Called from tests and after recording to verify no resource leak.
    """
    assert stream is None or getattr(stream, "active", False) is False, (
        "Microphone stream must be closed when is_listening is False."
    )
