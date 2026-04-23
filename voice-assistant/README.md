# macOS Voice Command Assistant

A fully local, zero-cost voice command assistant for macOS. Listens for commands
via a hotkey, transcribes with Whisper, interprets intent with a local LLM
(Ollama), and executes macOS actions.

**No internet required after setup. No API keys. No ongoing cost.**

---

## How It Works

```
Jarvis  →  active listening  →  faster-whisper (local)  →  Ollama llama3.2 (local)  →  macOS action
```

1. Say **Jarvis** to activate the assistant
2. Say your command
3. After 2 seconds of silence, the command is processed automatically
4. Press **Option + \\** to force listening or cancel early

---

## Prerequisites

### 1. Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Python 3.11+

```bash
brew install python@3.11
```

### 3. Ollama + LLM model

```bash
brew install ollama
ollama serve &          # start Ollama in background
ollama pull llama3.2   # ~2 GB download — one time only
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

### 4. Python dependencies

```bash
cd voice-assistant
pip3 install -r requirements.txt
```

> **Python 3.14 note:** If `faster-whisper` fails to install (no wheel yet for 3.14),
> use Python 3.11 explicitly:
> ```bash
> brew install python@3.11
> /opt/homebrew/bin/pip3.11 install -r requirements.txt
> # Then run with: /opt/homebrew/bin/python3.11 main.py
> ```

> **Apple Silicon note:** `faster-whisper` automatically uses the Metal backend
> via CTranslate2. No extra configuration needed.

---

## macOS Permissions (Required)

The app needs three system permissions. Grant them **before** running.

### Microphone
`System Settings → Privacy & Security → Microphone`
- Toggle ON for **Terminal** (or your Python environment's app)

### Accessibility (keyboard simulation)
`System Settings → Privacy & Security → Accessibility`
- Click **+** and add **Terminal** (or the Python executable)
- This allows `pyautogui` to simulate keystrokes

### Automation (AppleScript / app control)
`System Settings → Privacy & Security → Automation`
- Terminal → enable **System Events** and **Finder**

> If macOS prompts you for permission when first running, click **Allow**.

---

## Running the App

Make sure Ollama is serving first:
```bash
ollama serve &
```

Then start the assistant:
```bash
cd voice-assistant
python main.py
```

A 🎙 icon appears in the menu bar. The app is now running in the background.

---

## Usage

| Action | How |
|--------|-----|
| Wake the assistant | Say **Jarvis** |
| Start listening | Press **Option + \\** |
| Stop early | Press **Option + \\** again |
| Auto-stop | 2 seconds of silence |
| Quit | Click menu bar icon → **Quit Voice Assistant** |

### Example Voice Commands

| Say | What happens |
|-----|-------------|
| "Open Chrome" | Launches Google Chrome |
| "Switch to Terminal" | Brings Terminal to front |
| "Type hello world" | Types text into focused input |
| "Take a screenshot" | Interactive screenshot → ~/Desktop |
| "Turn up the volume" | Volume +10 |
| "Turn down the volume" | Volume −10 |
| "Mute" | Mutes audio output |
| "Press command space" | Triggers Spotlight |
| "Put the computer to sleep" | Prompts confirmation, then sleeps |

---

## Auto-Start on Login

```bash
python install_launchagent.py
```

This installs a LaunchAgent that starts the assistant automatically at login.

To uninstall auto-start:
```bash
python install_launchagent.py --remove
```

---

## Logs

All activity is logged to `~/.voice-assistant/` (override with `VOICE_ASSISTANT_LOG_DIR`; if the default isn’t writable, the app falls back to a local `.voice-assistant/` folder):

| File | Contents |
|------|----------|
| `app.log` | Full application log with timestamps |
| `command_log.jsonl` | One JSON record per executed command |
| `launchagent_stdout.log` | stdout when running as LaunchAgent |
| `launchagent_stderr.log` | stderr when running as LaunchAgent |

View recent commands:
```bash
tail -f ~/.voice-assistant/command_log.jsonl | python -m json.tool
```

---

## Configuration

Edit `config.py` to tune behaviour:

```python
WHISPER_MODEL = "tiny.en"     # Change to "base.en" for better accuracy
SILENCE_THRESHOLD = 0.01      # Batch mode only; not used for streaming intent
VOICE_ACTIVITY_THRESHOLD = 0.008  # Streaming mode: lower if mic is quiet
SILENCE_DURATION = 2.0        # Seconds of silence before auto-stop
HOTKEY_DEBOUNCE_MS = 300      # Minimum ms between toggles
OLLAMA_MODEL = "llama3.2"     # Change to any Ollama model you have pulled
```

**Whisper model tradeoffs on Apple Silicon:**

| Model | Size | Latency | Accuracy |
|-------|------|---------|----------|
| `tiny.en` | 39M | ~0.3s | Good for short commands |
| `base.en` | 74M | ~0.8s | Better accent/noise tolerance |
| `small.en` | 244M | ~2s | Near-human for most speech |

`tiny.en` is the default — optimal for 3-10 word voice commands.

---

## Architecture

```
main.py
├── hotkey.py        → pynput GlobalHotKeys, debounced toggle
├── audio.py         → sounddevice capture, RMS silence detection
├── stt.py           → faster-whisper transcription
├── llm.py           → Ollama HTTP client, JSON parsing + retry
├── executor.py      → AppleScript + pyautogui action dispatcher
├── safety.py        → dangerous action filter + confirmation prompt
├── logger.py        → JSONL command log + app logging setup
├── menubar.py       → rumps menu bar (main thread)
└── config.py        → all tunable constants
```

---

## Troubleshooting

**"Ollama request failed"**
→ Ensure `ollama serve` is running: `curl http://localhost:11434/api/tags`

**Hotkey doesn't work**
→ Grant Accessibility permission to Terminal in System Settings

**Microphone not capturing**
→ Grant Microphone permission to Terminal in System Settings

**App crashes with `NSInternalInconsistencyException`**
→ `main.py` must be run directly (not via `python -m`). The rumps app must run on the main thread.

**`pyautogui` fails to type**
→ Grant Accessibility permission. Some apps (Terminal itself) block synthetic input.

**Commands misrecognised**
→ Speak **Jarvis** clearly for wake-up, or switch to `base.en` in `config.py` for better STT accuracy.

**LLM returns non-JSON**
→ The retry logic handles this. If persistent, try `ollama pull llama3.2` to re-download the model.
