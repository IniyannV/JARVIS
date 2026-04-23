"""
Clipboard history polling and paste injection helpers.
"""

import logging
import subprocess
import time

import pyautogui

from config import CLIPBOARD_HISTORY_MAX
import state

logger = logging.getLogger("voice-assistant.clipboard")


def _append_history(text: str) -> None:
    with state.clipboard_lock:
        state.clipboard_history.append(text)
        if len(state.clipboard_history) > CLIPBOARD_HISTORY_MAX:
            del state.clipboard_history[:-CLIPBOARD_HISTORY_MAX]


def poll_clipboard(interval_sec: float = 1.0) -> None:
    """
    Poll the macOS clipboard and record distinct values in history.
    """
    last_value: str | None = None
    while True:
        try:
            result = subprocess.run(
                ["pbpaste"],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            logger.warning("Clipboard polling unavailable: pbpaste is not installed.")
            return

        if result.returncode == 0:
            current_value = result.stdout
            if current_value != last_value:
                _append_history(current_value)
                last_value = current_value
        else:
            logger.warning("Clipboard polling failed: %s", result.stderr.strip())

        time.sleep(interval_sec)


def get_history() -> list[str]:
    """
    Return a thread-safe snapshot of clipboard history.
    """
    with state.clipboard_lock:
        return list(state.clipboard_history)


def inject(text: str) -> None:
    """
    Copy text into the clipboard and paste it into the focused app.
    """
    try:
        result = subprocess.run(
            ["pbcopy"],
            input=text,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("pbcopy is not installed.") from exc

    if result.returncode != 0:
        error = result.stderr.strip() or "pbcopy failed."
        raise RuntimeError(error)

    time.sleep(0.1)
    pyautogui.hotkey("command", "v")
