"""
Configuration constants for the voice assistant.
No API keys — everything runs locally.
"""

import os

# --- Whisper / STT ---
WHISPER_MODEL = "tiny.en"          # tiny.en: ~39M params, ~0.5s latency on Apple Silicon
                                    # Use "base.en" for better accuracy at ~1s latency
WHISPER_DEVICE = "auto"            # "auto" picks MPS on Apple Silicon, else CPU
WHISPER_COMPUTE_TYPE = "int8"      # int8 is fastest on CPU/MPS for tiny model

# --- Ollama / LLM ---
OLLAMA_MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT = 30                # seconds

# --- Audio Capture ---
SAMPLE_RATE = 16000                # Hz — Whisper expects 16kHz
CHANNELS = 1                       # Mono
CHUNK_DURATION = 0.1               # seconds per audio chunk
SILENCE_THRESHOLD = 0.03           # RMS below this = silence (raised to cut background noise)
SILENCE_DURATION = 2.0             # seconds of silence before auto-stop
MAX_RECORDING_DURATION = 30.0      # hard cap — seconds

# --- Hotkey ---
HOTKEY_COMBO = "<alt>+\\"          # Option + backslash
HOTKEY_DEBOUNCE_MS = 300           # minimum ms between toggles

# --- Menu Bar ---
ICON_IDLE = "🎙"                   # shown when not listening
ICON_LISTENING = "🔴"              # shown when listening

# --- Logging ---
LOG_DIR = os.path.expanduser("~/.voice-assistant")
COMMAND_LOG_PATH = os.path.join(LOG_DIR, "command_log.jsonl")
APP_LOG_PATH = os.path.join(LOG_DIR, "app.log")

# --- Safety ---
DANGEROUS_ACTIONS = ["sleep", "restart", "shutdown", "delete"]
DANGEROUS_CONFIRM_TIMEOUT = 5      # seconds to wait for confirmation

# --- LaunchAgent ---
LAUNCHAGENT_LABEL = "com.voiceassistant.app"
LAUNCHAGENT_PLIST_NAME = f"{LAUNCHAGENT_LABEL}.plist"
