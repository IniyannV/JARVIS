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

# Set by main() before the run loop starts
dashboard: Optional["Dashboard"] = None
speaker: Optional["Speaker"] = None
menubar_app: Optional["VoiceAssistantApp"] = None
executor_service: Optional["ExecutorService"] = None

# Set by main() per listening session
streaming_stt: Optional["StreamingSTT"] = None
intent_engine: Optional["IntentEngine"] = None
