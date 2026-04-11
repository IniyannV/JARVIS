"""
Entry point for the macOS Voice Assistant.

Startup sequence:
1. Set up logging and ensure log directory exists.
2. Instantiate the menu bar app (must run on main thread).
3. Start the global hotkey listener in a background thread.
4. The menu bar app's run loop blocks the main thread.

When the hotkey fires:
- Toggle ON  → spawn audio_pipeline thread
- Toggle OFF → set stop_event, audio thread exits cleanly
"""

import logging
import threading
from typing import Optional

from logger import setup_logging
from config import LOG_DIR
from pathlib import Path


# ---------------------------------------------------------------------------
# Orchestration pipeline
# ---------------------------------------------------------------------------

def run_pipeline(stop_event: threading.Event, app) -> None:
    """
    Full voice command pipeline: audio → STT → LLM → safety → execute → log.

    Runs in a background thread. Exits when stop_event is set or audio ends.
    """
    import audio
    import stt
    import llm
    import executor
    import safety
    from logger import log_command

    logger = logging.getLogger("voice-assistant.pipeline")
    logger.info("Pipeline started.")

    # Phase 1: Record audio
    wav_bytes = audio.record(stop_event)

    if wav_bytes is None:
        logger.info("No audio captured — pipeline exiting.")
        if app is not None:
            app.update_state(False)
        return

    # Phase 2: Transcribe
    transcript = stt.transcribe(wav_bytes)

    if not transcript:
        logger.info("Empty transcript — pipeline exiting.")
        if app is not None:
            app.update_state(False)
        return

    logger.info("Transcript: %s", transcript)

    # Phase 3: Interpret intent
    action = llm.interpret(transcript)
    logger.info("Action: %s", action)

    # Phase 4: Safety gate
    safe_action = safety.guard(action)
    if safe_action is None:
        log_command(transcript, action.get("action", "unknown"), action, False, "blocked_by_safety")
        if app is not None:
            app.update_last_command(f"[blocked] {transcript}")
            app.update_state(False)
        return

    # Phase 5: Execute
    success, message = executor.execute(safe_action)
    logger.info("Execution result: success=%s, message=%s", success, message)

    # Phase 6: Log
    log_command(
        transcript=transcript,
        action_type=safe_action.get("action", "unknown"),
        action_detail=safe_action,
        success=success,
        error="" if success else message,
    )

    # Phase 7: Update menu bar
    if app is not None:
        display = f"{'✓' if success else '✗'} {transcript}"
        app.update_last_command(display)
        app.update_state(False)

    logger.info("Pipeline complete.")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

class AssistantController:
    """
    Coordinates the hotkey listener, pipeline threads, and menu bar updates.
    """

    def __init__(self, app) -> None:
        self._app = app
        self._lock = threading.Lock()
        self._pipeline_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._logger = logging.getLogger("voice-assistant.controller")

    def on_toggle(self, is_listening: bool) -> None:
        """Called by HotkeyListener on each debounced toggle."""
        if is_listening:
            self._start_listening()
        else:
            self._stop_listening()

    def _start_listening(self) -> None:
        with self._lock:
            if self._pipeline_thread is not None and self._pipeline_thread.is_alive():
                self._logger.warning("Pipeline already running — ignoring duplicate toggle ON.")
                return

            self._stop_event = threading.Event()
            self._pipeline_thread = threading.Thread(
                target=run_pipeline,
                args=(self._stop_event, self._app),
                daemon=True,
                name="pipeline",
            )
            self._pipeline_thread.start()
            self._logger.info("Pipeline thread started.")

        if self._app is not None:
            self._app.update_state(True)

    def _stop_listening(self) -> None:
        with self._lock:
            if self._stop_event is not None:
                self._stop_event.set()
                self._logger.info("Stop event sent to pipeline thread.")

        if self._app is not None:
            self._app.update_state(False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    setup_logging()
    logger = logging.getLogger("voice-assistant.main")
    logger.info("Voice Assistant starting…")

    # Import here so logging is configured before module-level code runs
    from hotkey import HotkeyListener
    from menubar import VoiceAssistantApp

    # Create menu bar app first (will run on main thread)
    app = VoiceAssistantApp()

    # Controller bridges hotkey events ↔ pipeline
    controller = AssistantController(app)

    # Wire menu button → controller
    app.set_toggle_callback(controller.on_toggle)

    # Start hotkey listener in background thread
    hotkey_listener = HotkeyListener(on_toggle=controller.on_toggle)
    hotkey_listener.start()

    logger.info("Hotkey listener running. Press %s to toggle.", "Option+\\")
    print("Voice Assistant running. Press Option+\\ to toggle listening.")
    print("Check the menu bar for status. Logs:", LOG_DIR)

    try:
        # run() blocks — must be called on the main thread
        app.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        hotkey_listener.stop()
        logger.info("Voice Assistant stopped.")


if __name__ == "__main__":
    main()
