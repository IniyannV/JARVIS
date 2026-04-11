"""
Global hotkey listener with debounced toggle logic.

Listens for Option+\\ even when the app is in the background.
Uses pynput.keyboard.GlobalHotKeys which registers a low-level OS hook.

Thread safety:
- is_listening and _last_toggle_time are protected by _lock.
- on_toggle callback is called from the hotkey thread; it must be quick.
"""

import logging
import threading
import time
from typing import Callable, Optional

from pynput import keyboard

from config import HOTKEY_COMBO, HOTKEY_DEBOUNCE_MS
from logger import log_toggle

logger = logging.getLogger("voice-assistant.hotkey")


class HotkeyListener:
    """
    Wraps pynput GlobalHotKeys with debounced toggle logic.

    Args:
        on_toggle: Callback invoked with (is_listening: bool) on each valid toggle.
                   Called from a background thread — must be thread-safe.
    """

    def __init__(self, on_toggle: Callable[[bool], None]) -> None:
        self._on_toggle = on_toggle
        self._lock = threading.Lock()
        self._is_listening = False
        self._last_toggle_time: float = 0.0
        self._hotkey_listener: Optional[keyboard.GlobalHotKeys] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_listening(self) -> bool:
        with self._lock:
            return self._is_listening

    def start(self) -> None:
        """Start the background hotkey listener thread."""
        self._hotkey_listener = keyboard.GlobalHotKeys(
            {HOTKEY_COMBO: self._handle_hotkey}
        )
        self._hotkey_listener.daemon = True
        self._hotkey_listener.start()
        logger.info("Hotkey listener started. Combo: %s", HOTKEY_COMBO)

    def stop(self) -> None:
        """Stop the hotkey listener."""
        if self._hotkey_listener is not None:
            self._hotkey_listener.stop()
            logger.info("Hotkey listener stopped.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle_hotkey(self) -> None:
        """
        Called by pynput on every matching keypress.

        Applies debounce, flips state, and fires the on_toggle callback.
        """
        now = time.monotonic()
        debounce_sec = HOTKEY_DEBOUNCE_MS / 1000.0

        with self._lock:
            if (now - self._last_toggle_time) < debounce_sec:
                logger.debug("Hotkey debounced (%.0fms since last).", (now - self._last_toggle_time) * 1000)
                return

            self._last_toggle_time = now
            self._is_listening = not self._is_listening
            new_state = self._is_listening

        label = "ON" if new_state else "OFF"
        print(f"[HOTKEY] LISTENING {label}")
        logger.info("Toggle → LISTENING %s", label)
        log_toggle(new_state)

        # Fire callback outside the lock to prevent deadlocks in the callback
        try:
            self._on_toggle(new_state)
        except Exception as exc:
            logger.exception("on_toggle callback raised: %s", exc)
