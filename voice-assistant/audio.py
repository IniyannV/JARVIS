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
import wave
from typing import Optional

import numpy as np
import sounddevice as sd

from config import (
    CHANNELS,
    CHUNK_DURATION,
    MAX_RECORDING_DURATION,
    SAMPLE_RATE,
    SILENCE_DURATION,
    SILENCE_THRESHOLD,
)

logger = logging.getLogger("voice-assistant.audio")


def _rms(chunk: np.ndarray) -> float:
    """Compute root-mean-square energy of an audio chunk."""
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))


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
