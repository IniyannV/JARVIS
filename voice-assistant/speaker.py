"""
Audible spoken feedback using the macOS `say` command.

Speech is fully asynchronous — it never blocks the pipeline.
Only one utterance plays at a time; a new say() call kills the previous one.
"""

import logging
import subprocess
import threading
import time
from typing import Optional

logger = logging.getLogger("voice-assistant.speaker")


class Speaker:
    """
    Thin wrapper around the macOS `say` CLI.

    Usage:
        speaker = Speaker()
        speaker.say("JARVIS is online")
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._speaking = False
        self._started_at = 0.0

    def say(self, text: str) -> None:
        """
        Speak text asynchronously. Kills any currently playing speech first.

        Runs in a daemon thread so the caller is never blocked.
        """
        if not text or not text.strip():
            return
        thread = threading.Thread(target=self._speak, args=(text,), daemon=True)
        thread.start()

    def interrupt(self) -> None:
        """Immediately stop any ongoing speech."""
        with self._lock:
            proc = self._proc
            self._proc = None
            self._speaking = False
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            proc = self._proc
            speaking = self._speaking and proc is not None and proc.poll() is None
        return speaking

    @property
    def started_at(self) -> float:
        with self._lock:
            return self._started_at

    def _speak(self, text: str) -> None:
        """Internal: kill previous process, then launch a new one."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.kill()
                    self._proc.wait()
                except OSError:
                    pass
            try:
                self._proc = subprocess.Popen(
                    ["say", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._speaking = True
                self._started_at = time.monotonic()
            except FileNotFoundError:
                logger.error("`say` command not found — is this macOS?")
                return

        logger.debug("Speaking: %s", text)
        # Wait for it to finish (still in the daemon thread, not blocking main)
        proc = self._proc
        if proc is not None:
            proc.wait()
        with self._lock:
            if self._proc is proc:
                self._speaking = False


def build_confirmation(action: dict, success: bool) -> str:
    """
    Map a structured action dict to a short spoken confirmation string.

    Args:
        action:  The action dict from the LLM (after execution).
        success: Whether the executor succeeded.

    Returns:
        A short human-friendly string to pass to speaker.say().
    """
    if not success:
        action_type = action.get("action", "unknown")
        if action_type == "unknown":
            return "Sorry, I didn't understand that"
        return f"That didn't work"

    action_type = action.get("action", "unknown")

    if action_type == "open_application":
        return f"Opening {action.get('app_name', 'application')}"

    elif action_type == "switch_window":
        return f"Switching to {action.get('app_name', 'application')}"

    elif action_type == "type_text":
        return "Typing text"

    elif action_type == "press_keys":
        return "Keys pressed"

    elif action_type == "open_url":
        return "Opening URL"

    elif action_type == "web_search":
        query = action.get("query", "")
        return f"Searching for {query}" if query else "Searching"

    elif action_type == "system_action":
        command = action.get("command", "")
        mapping = {
            "volume_up":   "Volume up",
            "volume_down": "Volume down",
            "mute":        "Muted",
            "screenshot":  "Screenshot taken",
            "sleep":       "Going to sleep",
        }
        return mapping.get(command, f"System action: {command}")

    elif action_type == "unknown":
        return "Sorry, I didn't understand that"

    return "Done"
