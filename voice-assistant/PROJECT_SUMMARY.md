# JARVIS Project Summary

## Overview

JARVIS is a local macOS voice assistant built in Python 3.11. It combines:

- always-on passive listening with wake-word detection
- active streaming speech-to-text using `faster-whisper`
- local LLM processing through Ollama
- real-time macOS action execution
- a native dashboard window and menu bar presence
- spoken assistant replies using macOS `say`

The project is designed to run fully locally after setup, with no cloud APIs or paid services.

## Current Runtime Model

JARVIS uses a two-mode listening architecture:

- `passive` mode
  - continuously monitors microphone audio for the wake phrase `"hey jarvis"`
  - uses a lightweight wake-word pipeline with a small rolling audio buffer
  - keeps CPU usage lower than full always-on transcription

- `active` mode
  - streams live audio into incremental STT
  - extracts command segments early while the user is still speaking
  - finalizes full utterances on pauses or transcript stability
  - classifies intent as `command`, `question`, or `hybrid`
  - executes actions immediately and generates a spoken response in parallel

There is also a `hard_mute` state used by the hotkey override to fully silence the assistant.

## Main Capabilities

- Wake word activation with `"Hey Jarvis"`
- Hotkey override via `Option + \`
- Passive and active listening modes
- Streaming transcription with rolling audio windows
- Early command detection before full utterance completion
- LLM-based intent classification:
  - `command`
  - `question`
  - `hybrid`
- Hybrid splitting into executable commands and answerable questions
- Concurrent execution and response generation
- Spoken natural-language responses
- Lightweight conversation history for follow-up questions
- Dashboard visualization for mode, transcript, actions, response, and mic level
- macOS automation:
  - open/switch apps
  - open URLs
  - perform web searches
  - type text
  - press key combinations
  - control volume
  - take screenshots
  - sleep system

## High-Level Architecture

### Core flow

1. `audio.py` captures one shared microphone stream.
2. `main.py` routes audio based on mode:
   - passive -> `wake_word.py`
   - active -> `stt.py`
3. `stt.py` emits partial transcripts from rolling windows.
4. `intent_engine.py`:
   - emits early command segments
   - finalizes full utterances
   - classifies intent
   - splits hybrid inputs
5. `llm.py`:
   - converts commands into structured actions
   - classifies conversational intent
   - generates natural spoken responses
6. `executor.py` executes macOS actions in a thread pool.
7. `speaker.py` speaks the assistant response asynchronously.
8. `dashboard.py` and `menubar.py` reflect status live.

### Concurrency model

The app is intentionally non-blocking:

- microphone capture runs continuously on its own thread
- wake detection, STT, command execution, and response generation each use separate queues/workers
- LLM work is asynchronous
- action execution is thread-pooled
- speech output is asynchronous and interruptible

## Key Files

- `main.py`
  - orchestrates passive/active modes
  - manages audio routing and worker lifecycles
  - coordinates wake word, STT, intent, LLM, execution, and UI updates

- `audio.py`
  - owns the shared microphone stream
  - computes RMS mic energy
  - triggers pause and voice-start callbacks

- `wake_word.py`
  - implements low-cost wake-word detection using short-buffer Whisper inference

- `stt.py`
  - provides the `StreamingSTT` engine for incremental transcription

- `intent_engine.py`
  - tracks partial utterances
  - emits early command segments
  - finalizes interactions
  - classifies conversational intent and splits hybrid requests

- `llm.py`
  - handles Ollama requests
  - parses JSON command actions
  - classifies utterances
  - generates assistant replies

- `executor.py`
  - runs macOS actions using AppleScript, `open`, `pyautogui`, and system tools

- `speaker.py`
  - speaks assistant replies with macOS `say`
  - supports interruption

- `dashboard.py`
  - native AppKit dashboard window
  - displays mode, transcript, action, assistant response, history, and mic level

- `menubar.py`
  - provides the menu bar UI and state indicator

- `state.py`
  - stores global runtime references and lightweight conversation history

- `config.py`
  - centralizes thresholds, model settings, wake-word tuning, and UI/runtime constants

## External Dependencies

- Python 3.11
- `faster-whisper`
- `sounddevice`
- `numpy`
- `requests`
- `pynput`
- `pyautogui`
- `rumps`
- PyObjC frameworks for AppKit/Cocoa integration
- Ollama with a local model such as `llama3.2`

## User Experience

Typical flow:

1. JARVIS starts in passive mode.
2. User says: `"Hey Jarvis"`
3. JARVIS switches to active mode and replies: `"Yes?"`
4. User speaks a command, question, or hybrid request.
5. Commands start executing immediately when possible.
6. JARVIS speaks a natural response once the reply is ready.
7. After inactivity, JARVIS returns to passive mode automatically.

## Current Configuration Highlights

Important runtime knobs in `config.py` include:

- `VOICE_ACTIVITY_THRESHOLD`
- `WAKE_WORD`
- `WAKE_WORD_COOLDOWN`
- `ACTIVE_MODE_TIMEOUT`
- `WAKE_DETECTION_INTERVAL`
- `LLM_RESPONSE_MAX_TOKENS`
- `RESPONSE_TEMPERATURE`
- `INTENT_STABLE_FINALIZE_SEC`

These values control sensitivity, responsiveness, wake behavior, and response style.

## Notes

- The existing `README.md` still describes an older hotkey-first workflow and does not fully reflect the current wake-word and conversational architecture.
- The project has evolved into a conversational assistant rather than a simple command runner.
- The system is optimized for local-first execution and low-latency interaction on macOS.
