"""
macOS menu bar application using rumps.

Must run on the main thread (Cocoa requirement).
Exposes update_state() and update_last_command() which are called
thread-safely from the hotkey/orchestrator threads.
"""

import logging
import threading

import rumps

from config import ICON_IDLE, ICON_LISTENING

logger = logging.getLogger("voice-assistant.menubar")


class VoiceAssistantApp(rumps.App):
    """
    Menu bar app with:
    - Dynamic title showing listening state icon
    - Subtitle showing last executed command
    - Quit menu item
    """

    def __init__(self) -> None:
        super().__init__(
            name="Voice Assistant",
            title=ICON_IDLE,
            quit_button="Quit Voice Assistant",
        )
        self._lock = threading.Lock()
        self._toggle_callback = None  # set by controller after init

        # Menu items
        self._state_item = rumps.MenuItem("State: Waiting for 'Hey Jarvis'")
        self._last_cmd_item = rumps.MenuItem("Last command: —")
        self._toggle_btn = rumps.MenuItem("▶ Activate", callback=self._on_toggle_clicked)

        self.menu = [
            self._state_item,
            self._last_cmd_item,
            None,  # separator
            self._toggle_btn,
            None,  # separator
        ]

        logger.info("Menu bar app initialised.")

    # ------------------------------------------------------------------
    # Thread-safe update methods (called from worker threads)
    # ------------------------------------------------------------------

    def update_mode(self, mode: str, hard_mute: bool = False) -> None:
        """Update the state label for passive/active/off."""
        def _update(_):
            # Keep the menu bar icon static to avoid rapid emoji/title updates.
            self.title = ICON_IDLE
            if hard_mute:
                self._state_item.title = "State: Not active"
                self._toggle_btn.title = "▶ Activate"
                return
            if mode == "active":
                self._state_item.title = "State: Listening…"
                self._toggle_btn.title = "⏹ Turn Off"
            else:
                self._state_item.title = "State: Waiting for 'Hey Jarvis'"
                self._toggle_btn.title = "▶ Activate"
        rumps.Timer(_update, 0).start()

    def update_last_command(self, text: str) -> None:
        """Update the last-command menu item. Thread-safe."""
        truncated = text[:60] + ("…" if len(text) > 60 else "")

        def _update(_):
            self._last_cmd_item.title = f"Last command: {truncated}"

        rumps.Timer(_update, 0).start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def set_toggle_callback(self, callback) -> None:
        """Wire up the controller callback so the button can trigger it."""
        self._toggle_callback = callback

    def _on_toggle_clicked(self, _) -> None:
        """Called on main thread when the menu button is clicked."""
        if self._toggle_callback:
            self._toggle_callback()

    @rumps.clicked("Quit Voice Assistant")
    def quit_app(self, _) -> None:
        logger.info("Quit requested from menu bar.")
        rumps.quit_application()
