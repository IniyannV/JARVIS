"""
Shared mutable state for JARVIS.

Imported by any module that needs to access the dashboard or speaker
without circular imports. References are set once during main() startup.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dashboard import Dashboard
    from executor import ExecutorService
    from intent_engine import IntentEngine
    from menubar import VoiceAssistantApp
    from speaker import Speaker
    from stt import StreamingSTT
    from wake_word import WakeWordEngine

# Set by main() before the run loop starts
dashboard: Optional["Dashboard"] = None
speaker: Optional["Speaker"] = None
menubar_app: Optional["VoiceAssistantApp"] = None
executor_service: Optional["ExecutorService"] = None

# Mode switching
# - "passive": always-on wake word detection only
# - "active": full streaming STT + intent + execution
mode: str = "passive"
hard_mute: bool = False  # when True, audio is ignored entirely (no wake word)
last_activity_time: float = 0.0

# Set by main()
streaming_stt: Optional["StreamingSTT"] = None
intent_engine: Optional["IntentEngine"] = None
wake_word_engine: Optional["WakeWordEngine"] = None
