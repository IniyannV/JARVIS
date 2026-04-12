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
import queue
import sys
import threading
import time
from typing import Optional

from logger import setup_logging
from config import LOG_DIR
from pathlib import Path
import state


# ---------------------------------------------------------------------------
# Orchestration pipeline
# ---------------------------------------------------------------------------

def run_streaming_pipeline(stop_event: threading.Event, app) -> None:
    """
    Streaming pipeline:
      mic chunks → StreamingSTT (worker) → partial transcripts → IntentEngine →
      command queue → async LLM → safety → executor thread pool

    The mic capture loop must never block on STT/LLM/execution.
    """
    import audio
    import llm
    import safety
    from executor import ExecutorService
    from intent_engine import IntentEngine
    from logger import log_command
    from stt import StreamingSTT

    logger = logging.getLogger("voice-assistant.pipeline")
    logger.info("Streaming pipeline started.")

    audio_q: "queue.Queue[tuple[object, float]]" = queue.Queue(maxsize=64)
    command_q: "queue.Queue[str]" = queue.Queue(maxsize=32)

    state.streaming_stt = StreamingSTT()

    def _on_command(cmd_text: str) -> None:
        logger.info("Detected command: %s", cmd_text)
        if state.dashboard is not None:
            state.dashboard.update_processing()
        try:
            command_q.put_nowait(cmd_text)
        except queue.Full:
            logger.warning("Command queue full; dropping command: %s", cmd_text)

    state.intent_engine = IntentEngine(on_command=_on_command)
    state.intent_engine.reset_session()

    def _stt_worker() -> None:
        while not stop_event.is_set():
            try:
                chunk, rms = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            transcript = state.streaming_stt.process_audio_chunk(chunk, rms) if state.streaming_stt else None
            if transcript:
                logger.debug("Partial transcript: %s", transcript)
                if state.dashboard is not None:
                    state.dashboard.update_transcript(transcript)
                if state.intent_engine is not None:
                    state.intent_engine.process_transcript(transcript)

    def _command_worker() -> None:
        while not stop_event.is_set():
            try:
                cmd_text = command_q.get(timeout=0.2)
            except queue.Empty:
                continue

            # LLM must not block mic capture; it runs here (off the audio thread).
            action = llm.interpret_async(cmd_text).result()
            logger.info("LLM action: %s", action)
            safe_action = safety.guard(action)
            if safe_action is None:
                log_command(cmd_text, action.get("action", "unknown"), action, False, "blocked_by_safety")
                if app is not None:
                    app.update_last_command(f"[blocked] {cmd_text}")
                continue

            exec_service: ExecutorService = state.executor_service  # set in main()
            if exec_service is None:
                continue

            future = exec_service.submit(safe_action, transcript=cmd_text)

            def _log_done(f, _cmd=cmd_text, _act=safe_action):
                try:
                    success, message = f.result()
                except Exception as exc:
                    success, message = False, str(exc)
                log_command(
                    transcript=_cmd,
                    action_type=_act.get("action", "unknown"),
                    action_detail=_act,
                    success=success,
                    error="" if success else message,
                )

            future.add_done_callback(_log_done)

    stt_thread = threading.Thread(target=_stt_worker, daemon=True, name="stt-worker")
    cmd_thread = threading.Thread(target=_command_worker, daemon=True, name="cmd-worker")
    stt_thread.start()
    cmd_thread.start()

    def _on_chunk(chunk_f32, rms: float, _ts: float) -> None:
        try:
            audio_q.put_nowait((chunk_f32, rms))
        except queue.Full:
            pass

    def _on_pause() -> None:
        if state.intent_engine is not None:
            state.intent_engine.notify_pause()

    try:
        audio.stream(stop_event, on_chunk=_on_chunk, on_pause=_on_pause)
    finally:
        stop_event.set()
        if state.intent_engine is not None:
            try:
                state.intent_engine.notify_pause()
            except Exception:
                pass
        state.streaming_stt = None
        state.intent_engine = None
        logger.info("Streaming pipeline stopped.")


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
        if state.dashboard is not None:
            state.dashboard.update_listening_state(is_listening)
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
                target=run_streaming_pipeline,
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

def _check_ollama_status() -> bool:
    """Return True if Ollama is reachable at localhost:11434."""
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def main() -> None:
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    setup_logging()
    logger = logging.getLogger("voice-assistant.main")
    logger.info("Voice Assistant starting…")
    try:
        import config as _config
        logger.info(
            "Runtime loaded: python=%s main=%s config=%s",
            sys.executable,
            __file__,
            getattr(_config, "__file__", "?"),
        )
        logger.info(
            "Config: VOICE_ACTIVITY_THRESHOLD=%s INTENT_STABLE_FINALIZE_SEC=%s STREAM_CHUNK_MS=%s",
            getattr(_config, "VOICE_ACTIVITY_THRESHOLD", "?"),
            getattr(_config, "INTENT_STABLE_FINALIZE_SEC", "?"),
            getattr(_config, "STREAM_CHUNK_MS", "?"),
        )
    except Exception:
        pass

    # Import here so logging is configured before module-level code runs
    try:
        from hotkey import HotkeyListener
        from menubar import VoiceAssistantApp
        from dashboard import Dashboard
        from speaker import Speaker
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or "a required dependency"
        print(f"Missing dependency: {missing}")
        print("Install dependencies with:")
        print("  pip3 install -r requirements.txt")
        print("If you're on Python 3.14 and installs fail, use Python 3.11 as noted in README.md.")
        sys.exit(1)

    # Instantiate speaker immediately (no UI, no thread issues)
    state.speaker = Speaker()

    # Create menu bar app first (will run on main thread)
    app = VoiceAssistantApp()
    state.menubar_app = app

    # Build and store dashboard (must happen on main thread, before run loop)
    state.dashboard = Dashboard()

    # Controller bridges hotkey events ↔ pipeline
    controller = AssistantController(app)

    # Wire menu button → controller
    app.set_toggle_callback(controller.on_toggle)

    # Optional: wire dashboard into the menu bar app (if supported)
    if hasattr(app, "set_dashboard"):
        app.set_dashboard(state.dashboard)

    # Executor service (thread pool) shared across sessions
    from executor import ExecutorService
    state.executor_service = ExecutorService(max_workers=4)

    # Start hotkey listener in background thread
    hotkey_listener = HotkeyListener(on_toggle=controller.on_toggle)
    hotkey_listener.start()

    logger.info("Hotkey listener running. Press %s to toggle.", "Option+\\")
    print("Voice Assistant running. Press Option+\\ to toggle listening.")
    print("Check the menu bar for status. Logs:", LOG_DIR)

    # Startup sequence — runs via a timer so NSApplication is already running
    def _startup(_timer):
        _timer.stop()  # fire once only
        from dashboard import start_drain_timer
        start_drain_timer()
        state.dashboard.show()
        state.dashboard.update_listening_state(False)
        ollama_ok = _check_ollama_status()
        state.dashboard.update_llm_status(ollama_ok)
        state.speaker.say("JARVIS is online")
        logger.info("Startup complete. Ollama online: %s", ollama_ok)

    import rumps
    rumps.Timer(_startup, 0.3).start()

    try:
        # run() blocks — must be called on the main thread
        app.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        hotkey_listener.stop()
        if state.executor_service is not None:
            state.executor_service.shutdown()
        logger.info("Voice Assistant stopped.")


if __name__ == "__main__":
    main()
