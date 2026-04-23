"""
Wake word detection engine for always-on passive listening.

Design goals:
- Low CPU: run inference at a fixed interval on a small rolling window.
- No cloud/services: uses the existing faster-whisper model already required.
- False-positive protection: phrase match + confidence + cooldown.
"""

from __future__ import annotations

import inspect
import re
import time
from dataclasses import dataclass

import numpy as np

from config import (
    PASSIVE_SAMPLE_RATE,
    WAKE_DETECTION_INTERVAL,
    WAKE_WORD,
    WAKE_WORD_COOLDOWN,
    WAKE_WORD_CONFIDENCE_THRESHOLD,
)
from stt import _get_model


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def _transcribe_supported(model, audio_np: np.ndarray, kwargs: dict):
    try:
        sig = inspect.signature(model.transcribe)
        supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    except (TypeError, ValueError):
        supported = kwargs
    return model.transcribe(audio_np, **supported)


@dataclass
class WakeDetection:
    detected: bool
    transcript: str = ""
    confidence: float = 0.0


class WakeWordEngine:
    """
    Lightweight wake word detector.

    Call process_audio_chunk(chunk_f32) frequently; it will internally rate-limit
    to WAKE_DETECTION_INTERVAL and return True only when a wake phrase is detected.
    """

    def __init__(
        self,
        wake_word: str = WAKE_WORD,
        sample_rate: int = PASSIVE_SAMPLE_RATE,
        interval_sec: float = WAKE_DETECTION_INTERVAL,
        cooldown_sec: float = WAKE_WORD_COOLDOWN,
    ) -> None:
        self._wake_word = _normalize(wake_word)
        self._sample_rate = sample_rate
        self._interval_sec = interval_sec
        self._cooldown_sec = cooldown_sec

        self._window_sec = 1.2
        self._window_samples = int(self._sample_rate * self._window_sec)
        self._buffer = np.zeros((0,), dtype=np.float32)

        self._last_run = 0.0
        self._last_trigger = 0.0

    def reset(self) -> None:
        self._buffer = np.zeros((0,), dtype=np.float32)
        self._last_run = 0.0

    def process_audio_chunk(self, chunk_f32: np.ndarray) -> WakeDetection:
        now = time.monotonic()
        if (now - self._last_trigger) < self._cooldown_sec:
            return WakeDetection(False)

        if chunk_f32 is None or chunk_f32.size == 0:
            return WakeDetection(False)

        if chunk_f32.dtype != np.float32:
            chunk_f32 = chunk_f32.astype(np.float32)

        self._buffer = np.concatenate([self._buffer, chunk_f32], axis=0)
        if self._buffer.size > self._window_samples:
            self._buffer = self._buffer[-self._window_samples :]

        if now - self._last_run < self._interval_sec:
            return WakeDetection(False)
        self._last_run = now

        if self._buffer.size < int(self._sample_rate * 0.6):
            return WakeDetection(False)

        model = _get_model()

        try:
            segments, _info = _transcribe_supported(
                model,
                self._buffer,
                {
                    "language": "en",
                    "beam_size": 1,
                    "best_of": 1,
                    "temperature": 0.0,
                    "vad_filter": True,
                    "vad_parameters": {"min_silence_duration_ms": 150},
                    "condition_on_previous_text": False,
                    "without_timestamps": True,
                },
            )
        except Exception:
            return WakeDetection(False)

        texts = []
        logprobs = []
        for seg in segments:
            seg_text = getattr(seg, "text", "").strip()
            if seg_text:
                texts.append(seg_text)
                lp = getattr(seg, "avg_logprob", None)
                if isinstance(lp, (int, float)):
                    logprobs.append(float(lp))

        transcript = _normalize(" ".join(texts))
        if not transcript:
            return WakeDetection(False)

        # Confidence heuristic: mean logprob mapped into [0, 1].
        if logprobs:
            mean_lp = sum(logprobs) / len(logprobs)
            confidence = max(0.0, min(1.0, (mean_lp + 2.0) / 1.8))
        else:
            confidence = 0.6

        # Require the primary wake word, while still allowing "hey jarvis" as
        # a fuzzy fallback.
        tokens = transcript.split()
        has_jarvis = "jarvis" in tokens
        has_hey = "hey" in tokens
        wake = self._wake_word.split()
        primary_wake = wake[0] if wake else "jarvis"
        exact_phrase = primary_wake in tokens
        hey_jarvis_phrase = "hey jarvis" in transcript

        # Fuzzy: hey ... jarvis within 3 tokens.
        close_phrase = False
        if has_hey and has_jarvis:
            hey_idx = [i for i, t in enumerate(tokens) if t == "hey"]
            jar_idx = [i for i, t in enumerate(tokens) if t == "jarvis"]
            close_phrase = any(abs(h - j) <= 3 for h in hey_idx for j in jar_idx)

        detected = has_jarvis and (exact_phrase or hey_jarvis_phrase or close_phrase)

        # Tighten: keep it wake-like (short) to reduce false positives.
        if len(tokens) > 10:
            return WakeDetection(False, transcript=transcript, confidence=confidence)

        if not detected:
            return WakeDetection(False, transcript=transcript, confidence=confidence)

        if confidence < WAKE_WORD_CONFIDENCE_THRESHOLD:
            return WakeDetection(False, transcript=transcript, confidence=confidence)

        self._last_trigger = now
        return WakeDetection(True, transcript=transcript, confidence=confidence)
