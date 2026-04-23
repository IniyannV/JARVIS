"""
Entry point for the macOS Voice Assistant.

Startup sequence:
1. Set up logging and ensure log directory exists.
2. Instantiate the menu bar app (must run on main thread).
3. Start always-on audio + workers (passive wake word).
4. Start the global hotkey listener in a background thread (override).
5. The menu bar app's run loop blocks the main thread.

Modes:
- passive: wake word only ("Jarvis")
- active:  full streaming STT + intent + execution
- hard mute: audio ignored entirely
"""

import logging
import queue
import sys
import threading
import time
from concurrent.futures import Future, wait
from typing import Optional

from logger import setup_logging
from config import ACTIVE_MODE_TIMEOUT, LOG_DIR, VOICE_ACTIVITY_THRESHOLD
from pathlib import Path
import state


# ---------------------------------------------------------------------------
# Orchestration pipeline
# ---------------------------------------------------------------------------

class AlwaysOnOrchestrator:
    """
    Single shared audio stream that routes chunks into:
      - Passive mode: wake word engine only
      - Active mode: streaming STT → intent → async LLM → executor pool
    """

    def __init__(self, app) -> None:
        self._app = app
        self._stop_event = threading.Event()
        self._logger = logging.getLogger("voice-assistant.pipeline")
        self._mode_lock = threading.Lock()

        self._wake_q: "queue.Queue[object]" = queue.Queue(maxsize=256)
        self._stt_q: "queue.Queue[tuple[object, float]]" = queue.Queue(maxsize=128)
        self._command_q: "queue.Queue[tuple[int, str]]" = queue.Queue(maxsize=64)
        self._interaction_q: "queue.Queue[tuple[int, str]]" = queue.Queue(maxsize=32)

        self._threads: list[threading.Thread] = []
        self._ctx_lock = threading.Lock()
        self._ctx: dict[int, dict] = {}

        self._response_epoch_lock = threading.Lock()
        self._response_epoch = 0

    def start(self) -> None:
        import audio
        from intent_engine import IntentEngine
        from stt import StreamingSTT
        from wake_word import WakeWordEngine

        state.streaming_stt = StreamingSTT()
        state.intent_engine = IntentEngine(
            on_command_segment=self._on_command_segment,
            on_interaction_final=self._on_interaction_final,
        )
        state.intent_engine.reset_session()
        state.wake_word_engine = WakeWordEngine()

        # Start workers
        self._threads = [
            threading.Thread(target=self._wake_worker, daemon=True, name="wake-worker"),
            threading.Thread(target=self._stt_worker, daemon=True, name="stt-worker"),
            threading.Thread(target=self._command_worker, daemon=True, name="cmd-worker"),
            threading.Thread(target=self._interaction_worker, daemon=True, name="interaction-worker"),
            threading.Thread(target=self._timeout_worker, daemon=True, name="mode-timeout"),
            threading.Thread(
                target=lambda: audio.stream(
                    self._stop_event,
                    on_chunk=self._on_chunk,
                    on_pause=self._on_pause,
                    on_voice_start=self._on_voice_start,
                ),
                daemon=True,
                name="audio",
            ),
        ]
        for t in self._threads:
            t.start()

        self._set_mode("passive", hard_mute=False, speak=False, flash=False)
        self._logger.info("Always-on pipeline started (passive wake word).")

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    def hotkey_override(self) -> None:
        """
        Hotkey behavior:
          - If active: hard-mute OFF completely
          - Otherwise: force active (bypass wake word)
        """
        with self._mode_lock:
            if state.hard_mute:
                state.hard_mute = False
                self._set_mode("active", hard_mute=False, speak=False, flash=False)
                return
            if state.mode == "active":
                state.hard_mute = True
                self._set_mode("passive", hard_mute=True, speak=False, flash=False)
                return
            self._set_mode("active", hard_mute=False, speak=False, flash=False)

    def _activate_from_wake(self) -> None:
        with self._mode_lock:
            if state.hard_mute:
                return
            self._set_mode("active", hard_mute=False, speak=True, flash=True)

    def _set_mode(self, mode: str, hard_mute: bool, speak: bool, flash: bool) -> None:
        state.mode = mode
        state.hard_mute = hard_mute
        state.last_activity_time = time.monotonic()

        if state.streaming_stt is not None:
            state.streaming_stt.reset()
        if state.intent_engine is not None:
            state.intent_engine.reset_session()
        if state.wake_word_engine is not None and mode == "passive":
            state.wake_word_engine.reset()
        with self._ctx_lock:
            self._ctx.clear()

        if state.dashboard is not None:
            state.dashboard.update_mode(mode, hard_mute=hard_mute)
            if flash and mode == "active":
                state.dashboard.flash_wake()
        if state.menubar_app is not None:
            state.menubar_app.update_mode(mode, hard_mute=hard_mute)

        if speak and state.speaker is not None and mode == "active":
            # If waking up, keep it short.
            state.speaker.say("Yes?")

    # ------------------------------------------------------------------
    # Audio routing
    # ------------------------------------------------------------------

    def _on_chunk(self, chunk_f32, rms: float, ts: float) -> None:
        if state.hard_mute:
            return

        # Keep active mode alive while the user is speaking.
        if state.mode == "active" and rms > VOICE_ACTIVITY_THRESHOLD:
            state.last_activity_time = ts

        if state.mode == "passive":
            # Very light energy gate to keep CPU low, but don't be too strict or
            # we may miss quiet wake-word audio.
            if rms < 0.001:
                return
            try:
                self._wake_q.put_nowait(chunk_f32)
            except queue.Full:
                pass
            return

        # Active mode
        try:
            self._stt_q.put_nowait((chunk_f32, rms))
        except queue.Full:
            pass

    def _on_pause(self) -> None:
        if state.hard_mute:
            return
        if state.mode != "active":
            return
        if state.intent_engine is not None:
            state.intent_engine.notify_pause()

    def _on_voice_start(self) -> None:
        # Cancel pending assistant responses when the user starts speaking again.
        with self._response_epoch_lock:
            self._response_epoch += 1

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    def _wake_worker(self) -> None:
        from wake_word import WakeDetection

        last_debug = 0.0
        while not self._stop_event.is_set():
            try:
                chunk = self._wake_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if state.hard_mute or state.mode != "passive" or state.wake_word_engine is None:
                continue
            det: WakeDetection = state.wake_word_engine.process_audio_chunk(chunk)
            if not det.detected and det.transcript:
                now = time.monotonic()
                if now - last_debug >= 1.0:
                    last_debug = now
                    self._logger.debug(
                        "Wake candidate: transcript=%r conf=%.2f",
                        det.transcript,
                        det.confidence,
                    )
            if det.detected:
                self._logger.info("Wake word detected (conf=%.2f): %s", det.confidence, det.transcript)
                self._activate_from_wake()

    def _stt_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk, rms = self._stt_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if state.hard_mute or state.mode != "active" or state.streaming_stt is None:
                continue
            transcript = state.streaming_stt.process_audio_chunk(chunk, rms)
            if transcript:
                self._logger.debug("Partial transcript: %s", transcript)
                state.last_activity_time = time.monotonic()
                if state.dashboard is not None:
                    state.dashboard.update_transcript(transcript)
                if state.intent_engine is not None:
                    state.intent_engine.process_transcript(transcript)

    def _ensure_ctx(self, interaction_id: int) -> dict:
        with self._ctx_lock:
            ctx = self._ctx.get(interaction_id)
            if ctx is None:
                ctx = {
                    "executed_norm_segments": set(),
                    "actions_taken": [],
                    "results": [],
                    "expected_commands": 0,
                    "command_futures": [],
                    "condition": threading.Condition(),
                }
                self._ctx[interaction_id] = ctx
            return ctx

    def _mark_command_enqueued(self, interaction_id: int) -> None:
        ctx = self._ensure_ctx(interaction_id)
        condition = ctx["condition"]
        with condition:
            ctx["expected_commands"] += 1
            condition.notify_all()

    def _register_command_future(self, interaction_id: int, future: Future) -> None:
        ctx = self._ensure_ctx(interaction_id)
        condition = ctx["condition"]
        with condition:
            ctx["command_futures"].append(future)
            condition.notify_all()

    def _wait_for_command_results(self, interaction_id: int, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        ctx = self._ensure_ctx(interaction_id)
        condition = ctx["condition"]

        with condition:
            while len(ctx["command_futures"]) < ctx["expected_commands"]:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                condition.wait(timeout=remaining)
            futures = list(ctx["command_futures"])

        if not futures:
            return

        remaining = deadline - time.monotonic()
        if remaining > 0:
            wait(futures, timeout=remaining)

    def _on_command_segment(self, cmd_text: str, interaction_id: int) -> None:
        self._logger.info("Detected command segment: %s", cmd_text)
        state.last_activity_time = time.monotonic()
        if state.dashboard is not None:
            state.dashboard.update_processing()

        # Track segment so we can skip it if it appears again in a hybrid split.
        ctx = self._ensure_ctx(interaction_id)
        try:
            import re
            norm = re.sub(r"\\s+", " ", cmd_text.strip().lower())
            ctx["executed_norm_segments"].add(norm)
        except Exception:
            pass

        try:
            self._command_q.put_nowait((interaction_id, cmd_text))
            self._mark_command_enqueued(interaction_id)
        except queue.Full:
            self._logger.warning("Command queue full; dropping segment: %s", cmd_text)

    def _on_interaction_final(self, full_text: str, interaction_id: int) -> None:
        self._logger.info("Final utterance: %s", full_text)
        state.last_activity_time = time.monotonic()
        try:
            self._interaction_q.put_nowait((interaction_id, full_text))
        except queue.Full:
            self._logger.warning("Interaction queue full; dropping utterance.")

    def _command_worker(self) -> None:
        import llm
        import safety
        from logger import log_command

        while not self._stop_event.is_set():
            try:
                interaction_id, cmd_text = self._command_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if state.hard_mute:
                continue

            completion_future: Future = Future()
            self._register_command_future(interaction_id, completion_future)

            action = llm.interpret_async(cmd_text).result()
            if not isinstance(action, dict) or action.get("action") not in llm._VALID_ACTIONS:
                action = {"action": "unknown", "raw_command": cmd_text, "reason": "interpret_worker_error"}
            action = llm._sanity_check(cmd_text, action)
            self._logger.info("LLM action: %s", action)
            safe_action = safety.guard(action)
            if safe_action is None:
                ctx = self._ensure_ctx(interaction_id)
                ctx["results"].append(
                    {
                        "success": False,
                        "message": "blocked_by_safety",
                        "action": action,
                    }
                )
                completion_future.set_result((False, "blocked_by_safety"))
                log_command(cmd_text, action.get("action", "unknown"), action, False, "blocked_by_safety")
                if self._app is not None:
                    self._app.update_last_command(f"[blocked] {cmd_text}")
                continue

            if state.executor_service is None:
                ctx = self._ensure_ctx(interaction_id)
                ctx["results"].append(
                    {
                        "success": False,
                        "message": "executor_unavailable",
                        "action": safe_action,
                    }
                )
                completion_future.set_result((False, "executor_unavailable"))
                continue

            ctx = self._ensure_ctx(interaction_id)
            ctx["actions_taken"].append(safe_action)

            executor_future = state.executor_service.submit(safe_action, transcript=cmd_text)

            def _log_done(f, _cmd=cmd_text, _act=safe_action, _completion=completion_future):
                try:
                    success, message = f.result()
                except Exception as exc:
                    success, message = False, str(exc)
                try:
                    ctx = self._ensure_ctx(interaction_id)
                    ctx["results"].append({"success": success, "message": message, "action": _act})
                except Exception:
                    pass
                if not _completion.done():
                    _completion.set_result((success, message))
                log_command(
                    transcript=_cmd,
                    action_type=_act.get("action", "unknown"),
                    action_detail=_act,
                    success=success,
                    error="" if success else message,
                )

            executor_future.add_done_callback(_log_done)

    def _interaction_worker(self) -> None:
        import llm

        while not self._stop_event.is_set():
            try:
                interaction_id, full_text = self._interaction_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if state.hard_mute:
                continue

            # Snapshot cancellation epoch.
            with self._response_epoch_lock:
                epoch = self._response_epoch

            # Build up commands/questions using LLM classification.
            intent_type = "command"
            if state.intent_engine is not None:
                it = state.intent_engine.classify_intent(full_text)
                intent_type = it.value

            split = {"commands": [], "questions": []}
            if intent_type == "hybrid" and state.intent_engine is not None:
                split = state.intent_engine.split_hybrid(full_text)
            elif intent_type == "command":
                split = {"commands": [full_text], "questions": []}
            else:
                split = {"commands": [], "questions": [full_text]}

            ctx = self._ensure_ctx(interaction_id)
            already = set(ctx.get("executed_norm_segments", set()))

            # Execute any additional commands discovered by hybrid split.
            for cmd in split.get("commands", []):
                norm = " ".join(cmd.strip().lower().split())
                if norm in already:
                    continue
                already.add(norm)
                try:
                    self._command_q.put_nowait((interaction_id, cmd))
                    self._mark_command_enqueued(interaction_id)
                except queue.Full:
                    pass

            self._wait_for_command_results(interaction_id, timeout_sec=5.0)

            # Prepare response context for the assistant.
            history = state.get_history_snapshot()
            response_ctx = {
                "intent_type": intent_type,
                "user_input": full_text,
                "commands": split.get("commands", []),
                "questions": split.get("questions", []),
                "actions_taken": list(ctx.get("actions_taken", [])),
                "results": list(ctx.get("results", [])),
                "conversation_history": history,
            }

            fut = llm.generate_response_async(response_ctx)
            resp = fut.result()
            if isinstance(resp, dict) and resp.get("error"):
                continue
            if not isinstance(resp, str) or not resp.strip():
                continue

            # Drop stale responses if the user started speaking again.
            with self._response_epoch_lock:
                if epoch != self._response_epoch:
                    continue

            if state.dashboard is not None:
                state.dashboard.update_response(resp)
            if state.speaker is not None:
                state.speaker.say(resp)

            state.append_history(
                {
                    "user_input": full_text,
                    "assistant_response": resp,
                    "intent_type": intent_type,
                    "actions": split.get("commands", []),
                }
            )

    def _timeout_worker(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(0.2)
            if state.hard_mute:
                continue
            if state.mode != "active":
                continue
            if (time.monotonic() - state.last_activity_time) > ACTIVE_MODE_TIMEOUT:
                self._logger.info("Active mode timed out; returning to passive.")
                with self._mode_lock:
                    if not state.hard_mute and state.mode == "active":
                        # No speech on timeout.
                        self._set_mode("passive", hard_mute=False, speak=False, flash=False)


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
    orchestrator = AlwaysOnOrchestrator(app)

    # Wire menu button → hotkey-style override
    app.set_toggle_callback(orchestrator.hotkey_override)

    # Optional: wire dashboard into the menu bar app (if supported)
    if hasattr(app, "set_dashboard"):
        app.set_dashboard(state.dashboard)

    # Executor service (thread pool) shared across sessions
    from executor import ExecutorService
    state.executor_service = ExecutorService(max_workers=4)

    # Start hotkey listener in background thread
    hotkey_listener = HotkeyListener(on_press=orchestrator.hotkey_override)
    hotkey_listener.start()

    logger.info("Hotkey listener running. Press %s to toggle.", "Option+\\")
    print("Voice Assistant running. Say 'Jarvis' to activate, or press Option+\\.")
    print("Check the menu bar for status. Logs:", LOG_DIR)

    # Startup sequence — runs via a timer so NSApplication is already running
    def _startup(_timer):
        _timer.stop()  # fire once only
        import clipboard
        from dashboard import start_drain_timer
        start_drain_timer()
        state.dashboard.show()
        state.dashboard.update_mode(state.mode, hard_mute=state.hard_mute)
        ollama_ok = _check_ollama_status()
        state.dashboard.update_llm_status(ollama_ok)
        state.speaker.say("JARVIS is online")
        orchestrator.start()
        threading.Thread(
            target=clipboard.poll_clipboard,
            daemon=True,
            name="clipboard-poller",
        ).start()
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
        orchestrator.stop()
        if state.executor_service is not None:
            state.executor_service.shutdown()
        logger.info("Voice Assistant stopped.")


if __name__ == "__main__":
    main()
