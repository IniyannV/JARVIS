"""
Configuration constants for the voice assistant.
No API keys — everything runs locally.
"""

import os
import tempfile

# --- Whisper / STT ---
WHISPER_MODEL = "tiny.en"          # tiny.en: ~39M params, ~0.5s latency on Apple Silicon
                                    # Use "base.en" for better accuracy at ~1s latency
WHISPER_DEVICE = "auto"            # "auto" picks MPS on Apple Silicon, else CPU
WHISPER_COMPUTE_TYPE = "int8"      # int8 is fastest on CPU/MPS for tiny model

# --- Streaming pipeline ---
STREAM_CHUNK_MS = 150              # audio chunk size read from the mic
STT_WINDOW_SECONDS = 2.0           # rolling audio window for partial transcripts
PARTIAL_TRANSCRIPT_INTERVAL = 0.5  # seconds between STT runs
COMMAND_DEDUP_WINDOW_SEC = 2.0     # ignore identical commands within this window
INTENT_CONFIDENCE_THRESHOLD = 0.35  # gate partial transcripts (lower = more eager)
WHISPER_STREAM_NO_SPEECH_THRESHOLD = 0.95
WHISPER_STREAM_LOGPROB_THRESHOLD = -2.0

# --- Ollama / LLM ---
OLLAMA_MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT = 30                # seconds
LLM_RESPONSE_MAX_TOKENS = 60
COMMAND_RESPONSE_MAX_TOKENS = 10
QUESTION_RESPONSE_MAX_TOKENS = 60
RESPONSE_TEMPERATURE = 0.7

# --- Audio Capture ---
SAMPLE_RATE = 16000                # Hz — Whisper expects 16kHz
CHANNELS = 1                       # Mono
CHUNK_DURATION = 0.1               # seconds per audio chunk
SILENCE_THRESHOLD = 0.0003         # RMS below this = silence (lower if mic input is quiet)
SILENCE_DURATION = 2.0             # seconds of silence before auto-stop
MAX_RECORDING_DURATION = 30.0      # hard cap — seconds

# Mic / UI tuning
# If the mic bar barely moves, lower MIC_METER_SATURATION_RMS (e.g. 0.05).
MIC_METER_SATURATION_RMS = 0.12

# Streaming voice activity gate (separate from batch-record silence tuning).
# If partial transcripts/commands aren't triggering, lower this (e.g. 0.003–0.008).
VOICE_ACTIVITY_THRESHOLD = 0.008

# Intent finalization: if the partial transcript stops changing for this long,
# treat the current phrase as a complete command even without a detectable pause.
INTENT_STABLE_FINALIZE_SEC = 0.8

# --- Wake word / passive listening ---
WAKE_WORD = "jarvis"
WAKE_WORD_COOLDOWN = 3.0
ACTIVE_MODE_TIMEOUT = 6.0
PASSIVE_SAMPLE_RATE = 16000
WAKE_DETECTION_INTERVAL = 0.25
WAKE_WORD_CONFIDENCE_THRESHOLD = 0.25

# Speech interruption tuning
SPEECH_INTERRUPT_GRACE_SEC = 0.8
SPEECH_INTERRUPT_CONFIRM_CHUNKS = 3
# When True, suppress mic input entirely during TTS playback so Jarvis
# does not hear or transcribe its own spoken responses.
MUTE_MIC_WHILE_SPEAKING = True

# --- Hotkey ---
HOTKEY_COMBO = "<alt>+\\"          # Option + backslash
HOTKEY_DEBOUNCE_MS = 300           # minimum ms between toggles

# --- Menu Bar ---
ICON_IDLE = "🎙"                   # shown when not listening
ICON_LISTENING = "🔴"              # shown when listening

# --- Logging ---
def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_probe")
        with open(probe, "a", encoding="utf-8"):
            pass
        try:
            os.remove(probe)
        except OSError:
            pass
        return True
    except OSError:
        return False


def _choose_log_dir() -> str:
    """
    Pick a log directory that is writable.

    Prefers `VOICE_ASSISTANT_LOG_DIR` if set, otherwise tries the default
    `~/.voice-assistant`, then falls back to a project-local or temp directory.
    """
    env_dir = os.getenv("VOICE_ASSISTANT_LOG_DIR")
    project_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        env_dir,
        os.path.expanduser("~/.voice-assistant"),
        os.path.join(project_dir, ".voice-assistant"),
        os.path.join(tempfile.gettempdir(), "voice-assistant"),
    ]
    for candidate in candidates:
        if candidate and _is_writable_dir(candidate):
            return candidate
    return os.path.join(tempfile.gettempdir(), "voice-assistant")


LOG_DIR = _choose_log_dir()
COMMAND_LOG_PATH = os.path.join(LOG_DIR, "command_log.jsonl")
APP_LOG_PATH = os.path.join(LOG_DIR, "app.log")

# --- Safety ---
DANGEROUS_ACTIONS = ["sleep", "restart", "shutdown", "delete"]
DANGEROUS_CONFIRM_TIMEOUT = 5      # seconds to wait for confirmation

# --- Dashboard ---
DASHBOARD_WIDTH = 600
DASHBOARD_HEIGHT = 500
DASHBOARD_TITLE = "JARVIS"
MAX_HISTORY_ENTRIES = 100
MIC_METER_UPDATE_HZ = 10
MAX_CONVERSATION_HISTORY = 5

# --- LaunchAgent ---
LAUNCHAGENT_LABEL = "com.voiceassistant.app"
LAUNCHAGENT_PLIST_NAME = f"{LAUNCHAGENT_LABEL}.plist"
