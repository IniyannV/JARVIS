"""
Intent engine for streaming partial transcripts.

Goal: detect complete, actionable commands early (before silence) using simple
heuristics, deduplicate repeated triggers, and hand off finalized commands to
the (async) LLM/executor pipeline. Also emits finalized utterances for
conversational responses.
"""

from __future__ import annotations

import enum
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from config import COMMAND_DEDUP_WINDOW_SEC, INTENT_STABLE_FINALIZE_SEC

import llm


_TRIGGERS = {
    "open",
    "launch",
    "switch",
    "focus",
    "search",
    "google",
    "type",
    "press",
    "mute",
    "screenshot",
    "volume",
    "increase",
    "decrease",
    "turn",
    "sleep",
    "restart",
    "shutdown",
    "find",
    "locate",
    "paste",
    "git",
    "commit",
    "push",
    "pull",
    "run",
    "execute",
}

_CONJUNCTIONS = {"and", "then"}

_PREFIX_NOISE = {
    "jarvis",
    "hey",
    "hi",
    "please",
    "could",
    "can",
    "would",
    "you",
    "i",
    "want",
    "to",
    "um",
    "uh",
}


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    return [t for t in _normalize(text).split(" ") if t]

def _strip_prefix_noise(tokens: list[str]) -> list[str]:
    # Drop a short "wake word / politeness" prefix, e.g. "jarvis please".
    i = 0
    while i < min(len(tokens), 5) and tokens[i] in _PREFIX_NOISE:
        i += 1
    tokens = tokens[i:]

    # If a trigger exists soon after, discard everything before it.
    for j in range(min(len(tokens), 6)):
        if llm.starts_with_command_trigger(" ".join(tokens[j:])):
            return tokens[j:]
    return tokens


@dataclass(frozen=True)
class DetectedCommand:
    text: str
    normalized: str
    detected_at: float


class IntentType(enum.Enum):
    COMMAND = "command"
    QUESTION = "question"
    HYBRID = "hybrid"


class IntentEngine:
    """
    Streaming intent detector.

    - process_transcript() is fast and safe to call frequently (e.g. 2–5 Hz).
    - When an early command segment is detected, it calls on_command_segment(text, interaction_id).
    - When an utterance is finalized (pause/stable), it calls on_interaction_final(full_text, interaction_id).
    """

    def __init__(
        self,
        on_command_segment: Callable[[str, int], None],
        on_interaction_final: Callable[[str, int], None],
    ) -> None:
        self._on_command_segment = on_command_segment
        self._on_interaction_final = on_interaction_final
        self._lock = threading.Lock()

        self._last_partial: str = ""
        self._last_update = 0.0

        self._recent_exec: deque[tuple[str, float]] = deque(maxlen=32)  # (normalized, ts)
        self._queued_this_session: set[str] = set()

        # Token cursor into the latest partial transcript; everything before
        # this is considered "consumed" (already queued for execution).
        self._consumed_tokens = 0
        self._finalize_timer: Optional[threading.Timer] = None
        self._interaction_id = 0
        self._last_final_norm = ""
        self._last_final_ts = 0.0

    def reset_session(self) -> None:
        with self._lock:
            self._last_partial = ""
            self._last_update = 0.0
            self._queued_this_session.clear()
            self._consumed_tokens = 0
            self._interaction_id += 1
            if self._finalize_timer is not None:
                self._finalize_timer.cancel()
                self._finalize_timer = None

    def process_transcript(self, text: str) -> None:
        now = time.monotonic()
        if not text or not text.strip():
            return

        with self._lock:
            if text == self._last_partial:
                return
            self._last_partial = text
            self._last_update = now

            tokens = _strip_prefix_noise(_tokenize(text))
            if not tokens:
                return

            # Only consider the unconsumed tail.
            tail = tokens[self._consumed_tokens :]
            if not tail:
                return

            commands, consumed = self._extract_commands(tail)
            if consumed:
                self._consumed_tokens += consumed

            # One-shot stability finalizer: if the transcript stops changing,
            # flush the remaining tail as a command.
            if self._finalize_timer is not None:
                self._finalize_timer.cancel()
            self._finalize_timer = threading.Timer(
                INTENT_STABLE_FINALIZE_SEC, self._finalize_if_stable, args=(text,)
            )
            self._finalize_timer.daemon = True
            self._finalize_timer.start()

        for cmd in commands:
            self._emit_command_if_not_duplicate(cmd, now)

    def _finalize_if_stable(self, snapshot_text: str) -> None:
        now = time.monotonic()

        with self._lock:
            if self._last_partial != snapshot_text:
                return
            if now - self._last_update < INTENT_STABLE_FINALIZE_SEC:
                return

        self._finalize_interaction(reason="stable")

    def notify_pause(self) -> None:
        """
        Hint from audio pipeline that a brief pause occurred.
        Finalizes the current utterance.
        """
        self._finalize_interaction(reason="pause")

    def _finalize_interaction(self, reason: str) -> None:
        now = time.monotonic()
        with self._lock:
            text = (self._last_partial or "").strip()
            if not text:
                return

            norm = _normalize(text)
            if not norm:
                return

            # Avoid repeated finalize spam on the same phrase.
            if norm == self._last_final_norm and (now - self._last_final_ts) < 1.0:
                return
            self._last_final_norm = norm
            self._last_final_ts = now

            interaction_id = self._interaction_id

            # Reset per-utterance state but keep recent exec for dedup.
            self._last_partial = ""
            self._queued_this_session.clear()
            self._consumed_tokens = 0
            self._interaction_id += 1
            if self._finalize_timer is not None:
                self._finalize_timer.cancel()
                self._finalize_timer = None

        # Emit outside lock.
        self._on_interaction_final(text, interaction_id)

    def _emit_command_if_not_duplicate(self, command_text: str, now: float) -> None:
        normalized = _normalize(command_text)
        if not normalized:
            return

        with self._lock:
            # Intra-session dedup.
            if normalized in self._queued_this_session:
                return

            # Time-window dedup across bursts.
            for prev_norm, ts in reversed(self._recent_exec):
                if prev_norm == normalized and (now - ts) <= COMMAND_DEDUP_WINDOW_SEC:
                    return

            self._queued_this_session.add(normalized)
            self._recent_exec.append((normalized, now))
            interaction_id = self._interaction_id

        self._on_command_segment(command_text, interaction_id)

    # ------------------------------------------------------------------
    # Conversational classification/splitting (LLM-based as required)
    # ------------------------------------------------------------------

    def classify_intent(self, text: str) -> IntentType:
        result = llm.classify_intent_async(text).result()
        if isinstance(result, str):
            try:
                return IntentType(result)
            except Exception:
                pass
        return IntentType.COMMAND

    def split_hybrid(self, text: str) -> dict:
        result = llm.split_hybrid_async(text).result()
        if isinstance(result, dict):
            return result
        return {"commands": [], "questions": [text]}

    def _extract_commands(self, tokens: list[str]) -> tuple[list[str], int]:
        """
        Return (commands, tokens_consumed_from_head).

        Heuristic:
          - Split when we see a conjunction and the next word starts a trigger.
          - This detects patterns like:
              "open chrome and then search for youtube"
              "open chrome and search for youtube"
        """
        if not tokens:
            return [], 0

        commands: list[str] = []
        i = 0
        start = 0

        # Only fire early when the tail begins with a known command trigger.
        tail_text = " ".join(tokens)
        if len(tokens) < 3 or not llm.starts_with_command_trigger(tail_text):
            return [], 0

        while i < len(tokens) - 1:
            t = tokens[i]
            nxt = tokens[i + 1]

            # Boundary when conjunction precedes a new trigger.
            if t in _CONJUNCTIONS and nxt in _TRIGGERS:
                seg = tokens[start:i]
                while seg and seg[-1] in _CONJUNCTIONS:
                    seg = seg[:-1]
                seg_text = " ".join(seg).strip()
                if len(seg) >= 3 and llm.has_known_command_trigger(seg_text):
                    commands.append(seg_text)
                    start = i + 1
            i += 1

        # We only consume up to start (i.e., committed commands).
        consumed = start
        return commands, consumed
