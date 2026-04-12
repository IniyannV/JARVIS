"""
Global hotkey listener with debounced press handling.

Listens for Option+\\ even when the app is in the background.
Uses pynput.keyboard.GlobalHotKeys which registers a low-level OS hook.

Thread safety:
- _last_press_time is protected by _lock.
- on_press callback is called from the hotkey thread; it must be quick.
"""

import logging
import threading
import time
from typing import Callable, Optional

from pynput import keyboard

from config import HOTKEY_COMBO, HOTKEY_DEBOUNCE_MS

logger = logging.getLogger("voice-assistant.hotkey")


class HotkeyListener:
    """
    Wraps pynput GlobalHotKeys with debounced press logic.

    Args:
        on_press: Callback invoked on each valid hotkey press.
                  Called from a background thread — must be thread-safe.
    """

    def __init__(self, on_press: Callable[[], None]) -> None:
        self._on_press = on_press
        self._lock = threading.Lock()
        self._last_press_time: float = 0.0
        self._hotkey_listener: Optional[keyboard.GlobalHotKeys] = None

    def start(self) -> None:
        """Start the background hotkey listener thread."""
        self._hotkey_listener = keyboard.GlobalHotKeys({HOTKEY_COMBO: self._handle_hotkey})
        self._hotkey_listener.daemon = True
        self._hotkey_listener.start()
        logger.info("Hotkey listener started. Combo: %s", HOTKEY_COMBO)

    def stop(self) -> None:
        """Stop the hotkey listener."""
        if self._hotkey_listener is not None:
            self._hotkey_listener.stop()
            logger.info("Hotkey listener stopped.")

    def _handle_hotkey(self) -> None:
        now = time.monotonic()
        debounce_sec = HOTKEY_DEBOUNCE_MS / 1000.0

        with self._lock:
            if (now - self._last_press_time) < debounce_sec:
                logger.debug(
                    "Hotkey debounced (%.0fms since last).",
                    (now - self._last_press_time) * 1000,
                )
                return
            self._last_press_time = now

        logger.info("Hotkey pressed.")
        try:
            self._on_press()
        except Exception as exc:
            logger.exception("on_press callback raised: %s", exc)

