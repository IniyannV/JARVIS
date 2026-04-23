"""
Ollama HTTP client with JSON parsing and retry logic.

Sends the voice transcript to a local Ollama instance and parses a structured
action object from the response. Retries once with a stricter prompt on parse
failure. Falls back to {"action": "unknown"} on second failure.
"""

import json
import logging
import queue
import re
import threading
from typing import Any

import requests
from concurrent.futures import Future

from config import (
    COMMAND_RESPONSE_MAX_TOKENS,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
    QUESTION_RESPONSE_MAX_TOKENS,
    RESPONSE_TEMPERATURE,
)

logger = logging.getLogger("voice-assistant.llm")

_SYSTEM_ACTION_COMMANDS = {"volume_up", "volume_down", "mute", "screenshot", "sleep"}
_GIT_COMMANDS = {
    "status",
    "add_all",
    "commit",
    "push",
    "pull",
    "log",
    "diff",
    "branch",
    "checkout",
    "stash",
    "stash_pop",
}

_OPEN_APP_TRIGGERS = ("open", "launch", "start")
_SWITCH_WINDOW_TRIGGERS = ("switch to", "focus", "go to", "bring")
_OPEN_URL_TRIGGERS = ("go to", "visit", "open website", "open site")
_OPEN_URL_HINTS = (
    "website",
    "site",
    "url",
    "dot com",
    "dot io",
    "dot ai",
    "youtube",
    "github",
    "reddit",
    "gmail",
    "netflix",
    "twitter",
    "x.com",
)
_WEB_SEARCH_TRIGGERS = ("search", "search for", "google", "look up", "find online")
_TYPE_TEXT_TRIGGERS = ("type", "write", "enter")
_PRESS_KEYS_TRIGGERS = ("press", "hit", "shortcut", "key", "keys")
_FIND_FILE_TRIGGERS = ("find file", "find my", "find", "locate file", "locate")
_OPEN_RECENT_TRIGGERS = ("open recent", "open latest", "open my latest", "open my recent")
_OPEN_RECENT_HINTS = ("recent", "latest", "newest", "last")
_REVEAL_TRIGGERS = ("reveal", "show in finder", "reveal in finder", "show me")
_PASTE_CLIP_TRIGGERS = ("paste clip", "paste clipboard", "paste my last", "paste second")
_PASTE_CLIP_HINTS = ("clip", "clipboard", "copied", "copy")
_GIT_TRIGGERS = (
    "git",
    "commit",
    "push",
    "pull",
    "checkout",
    "branch",
    "stash",
    "diff",
    "log",
    "add all",
)
_RUN_SCRIPT_TRIGGERS = ("run", "execute", "start dev", "start the dev")
_RUN_SCRIPT_HINTS = (
    "pytest",
    "npm",
    "pnpm",
    "yarn",
    "make",
    "python",
    "uv",
    "pip",
    "server",
    "build",
    "test",
    "dev",
)
_VOLUME_UP_TRIGGERS = ("volume up", "turn up", "louder", "increase volume", "raise volume", "sound up")
_VOLUME_DOWN_TRIGGERS = (
    "volume down",
    "turn down",
    "quieter",
    "decrease volume",
    "lower volume",
    "sound down",
)
_MUTE_TRIGGERS = ("mute", "silent", "silence", "turn off sound", "mute sound")
_SCREENSHOT_TRIGGERS = ("screenshot", "screen shot", "take a screenshot", "capture screen")
_SLEEP_TRIGGERS = ("sleep", "put computer to sleep", "sleep the computer", "sleep my mac")
_COMMAND_START_TRIGGERS = {
    "open",
    "launch",
    "switch",
    "search",
    "google",
    "type",
    "press",
    "mute",
    "volume",
    "find",
    "locate",
    "paste",
    "git",
    "commit",
    "push",
    "pull",
    "run",
    "execute",
    "take",
    "capture",
}
_QUESTION_PREFIXES = ("what", "why", "how", "when", "where", "who", "which")


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.:/+_-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    padded = f" {text} "
    return any(f" {phrase} " in padded for phrase in phrases)


def _starts_with_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(text.startswith(phrase) for phrase in phrases)


def _has_any_token(text: str, tokens: tuple[str, ...]) -> bool:
    padded = f" {text} "
    return any(f" {token} " in padded for token in tokens)


def _is_open_url_text(text: str) -> bool:
    return (
        _contains_any_phrase(text, _OPEN_URL_TRIGGERS)
        or ("open " in f"{text} " and (_has_any_token(text, _OPEN_URL_HINTS) or "." in text))
    )


def _is_open_application_text(text: str) -> bool:
    return (
        _starts_with_any_phrase(text, _OPEN_APP_TRIGGERS)
        and not _is_open_url_text(text)
        and not _is_run_script_text(text)
        and not _is_git_text(text)
    )


def _is_git_text(text: str) -> bool:
    return _contains_any_phrase(text, _GIT_TRIGGERS)


def _is_run_script_text(text: str) -> bool:
    return (
        _contains_any_phrase(text, _RUN_SCRIPT_TRIGGERS)
        and (_has_any_token(text, _RUN_SCRIPT_HINTS) or _is_git_text(text))
    )


def has_known_command_trigger(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(
        (
            _is_open_application_text(normalized),
            _contains_any_phrase(normalized, _SWITCH_WINDOW_TRIGGERS),
            _is_open_url_text(normalized),
            _contains_any_phrase(normalized, _WEB_SEARCH_TRIGGERS),
            _contains_any_phrase(normalized, _TYPE_TEXT_TRIGGERS),
            _contains_any_phrase(normalized, _PRESS_KEYS_TRIGGERS),
            _contains_any_phrase(normalized, _FIND_FILE_TRIGGERS),
            ("open" in normalized.split() and _has_any_token(normalized, _OPEN_RECENT_HINTS)),
            _contains_any_phrase(normalized, _REVEAL_TRIGGERS),
            ("paste" in normalized.split() and _has_any_token(normalized, _PASTE_CLIP_HINTS)),
            _is_git_text(normalized),
            _is_run_script_text(normalized),
            _contains_any_phrase(normalized, _VOLUME_UP_TRIGGERS),
            _contains_any_phrase(normalized, _VOLUME_DOWN_TRIGGERS),
            _contains_any_phrase(normalized, _MUTE_TRIGGERS),
            _contains_any_phrase(normalized, _SCREENSHOT_TRIGGERS),
            _contains_any_phrase(normalized, _SLEEP_TRIGGERS),
        )
    )


def starts_with_command_trigger(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    first = normalized.split()[0]
    return first in _COMMAND_START_TRIGGERS

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a macOS voice command interpreter. Your ONLY job is to convert a voice \
command into a single JSON object. You must respond with RAW JSON only — no \
markdown, no backticks, no explanation, no text before or after the JSON object.

Respond with EXACTLY one of these schemas:

{"action": "open_application", "app_name": "<string>"}
{"action": "switch_window",    "app_name": "<string>"}
{"action": "open_url",         "url": "<full URL with https://>"}
{"action": "web_search",       "query": "<search query string>"}
{"action": "type_text",        "text": "<string>"}
{"action": "press_keys",       "keys": ["<key1>", "<key2>"]}
{"action": "system_action",    "command": "<volume_up|volume_down|mute|screenshot|sleep>"}
{"action": "find_file",        "query": "<file name query>"}
{"action": "open_recent",      "query": "<file name pattern>"}
{"action": "reveal_in_finder", "query": "<file path or name>"}
{"action": "paste_clip",       "index": <integer>}
{"action": "git_command",      "command": "<status|add_all|commit|push|pull|log|diff|branch|checkout|stash|stash_pop>", "message": "<optional commit message>", "branch": "<optional branch>", "repo_path": "<optional path>"}
{"action": "run_script",       "script": "<allowlisted command>", "args": "<argument string>"}
{"action": "unknown",          "raw_command": "<original text>", "reason": "<why>"}

Examples:
"open chrome"                -> {"action": "open_application", "app_name": "Google Chrome"}
"switch to terminal"         -> {"action": "switch_window", "app_name": "Terminal"}
"open youtube"               -> {"action": "open_url", "url": "https://www.youtube.com"}
"go to github"               -> {"action": "open_url", "url": "https://www.github.com"}
"open twitter"               -> {"action": "open_url", "url": "https://www.twitter.com"}
"open netflix"               -> {"action": "open_url", "url": "https://www.netflix.com"}
"search for python tutorials" -> {"action": "web_search", "query": "python tutorials"}
"google how to make pasta"   -> {"action": "web_search", "query": "how to make pasta"}
"search cat videos"          -> {"action": "web_search", "query": "cat videos"}
"type hello world"           -> {"action": "type_text", "text": "hello world"}
"press command space"        -> {"action": "press_keys", "keys": ["command", "space"]}
"take a screenshot"          -> {"action": "system_action", "command": "screenshot"}
"turn up the volume"         -> {"action": "system_action", "command": "volume_up"}
"turn down the volume"       -> {"action": "system_action", "command": "volume_down"}
"mute"                       -> {"action": "system_action", "command": "mute"}
"put computer to sleep"      -> {"action": "system_action", "command": "sleep"}
"find my resume"             -> {"action": "find_file", "query": "resume"}
"open my latest project"     -> {"action": "open_recent", "query": "project"}
"show me config.py"          -> {"action": "reveal_in_finder", "query": "config.py"}
"paste my last clipboard"    -> {"action": "paste_clip", "index": 1}
"paste second to last clip"  -> {"action": "paste_clip", "index": 2}
"git status"                 -> {"action": "git_command", "command": "status"}
"commit with message fix login bug" -> {"action": "git_command", "command": "commit", "message": "fix login bug"}
"push to origin"             -> {"action": "git_command", "command": "push"}
"what branch am I on"        -> {"action": "git_command", "command": "branch"}
"pull latest"                -> {"action": "git_command", "command": "pull"}
"run pytest"                 -> {"action": "run_script", "script": "pytest", "args": ""}
"start the dev server"       -> {"action": "run_script", "script": "npm", "args": "run dev"}
"run make build"             -> {"action": "run_script", "script": "make", "args": "build"}
"check the weather"          -> {"action": "unknown", "raw_command": "check the weather", "reason": "no_matching_action"}
"what time is it"            -> {"action": "unknown", "raw_command": "what time is it", "reason": "no_matching_action"}
"set a timer for five minutes" -> {"action": "unknown", "raw_command": "set a timer for five minutes", "reason": "no_matching_action"}
"what's the forecast today"  -> {"action": "unknown", "raw_command": "what's the forecast today", "reason": "no_matching_action"}

Rules:
- NEVER map a command to an action unless it is a close semantic match. If unsure, return unknown. Never guess.
- Use unknown aggressively. If the transcript does not contain clear trigger vocabulary for one action below, return unknown.
- Volume actions are special: volume_up, volume_down, and mute must ONLY be used when the transcript explicitly mentions volume, louder, quieter, mute, or sound. Never infer volume from unrelated input.
- Trigger rules:
  - open_application: only for "open", "launch", or "start" followed by an application.
  - switch_window: only for "switch to", "focus", "go to", or "bring" an app/window.
  - open_url: only for explicit site/navigation language like "go to", "visit", "open website", "open site", or "open" plus a clear website/domain.
  - web_search: only for "search", "search for", "google", "look up", or "find online".
  - type_text: only for "type", "write", or "enter" text.
  - press_keys: only for "press", "hit", "shortcut", "key", or "keys".
  - system_action screenshot: only for "screenshot", "screen shot", "take a screenshot", or "capture screen".
  - system_action sleep: only for explicit sleep phrases like "sleep", "put computer to sleep", or "sleep the computer".
  - find_file: only for "find" or "locate" a file/document.
  - open_recent: only for opening a recent/latest/newest file.
  - reveal_in_finder: only for "reveal", "show in finder", "reveal in finder", or "show me" a file.
  - paste_clip: only for "paste" with clip/clipboard/copied-item wording.
  - git_command: only for explicit git verbs like git/status/commit/push/pull/branch/checkout/stash/diff/log.
  - run_script: only when the user explicitly asks to run or execute a terminal/dev command.
- Use the full, proper application name (e.g., "Google Chrome" not "chrome").
- Use open_url for specific websites (YouTube, GitHub, Reddit, Gmail, etc.).
- For web_search, the query should be what would go into a Google search bar.
- For press_keys, use lowercase pyautogui key names (e.g., "command", "shift", "space").
- Use find_file to open a named file result and reveal_in_finder when the user wants to see the file in Finder.
- Use open_recent when the user asks for the latest or most recent file matching a name pattern.
- Use paste_clip for requests to paste an earlier clipboard item; index 1 means most recent.
- Use git_command for supported git verbs; use "message" for commit text and "branch" for checkout targets.
- Use run_script only when the user explicitly asks to run a terminal/dev command.
- If the command is unclear, unsupported, or only resembles a nearby action, use "unknown".
- NEVER output anything other than the JSON object.\
"""

_STRICT_SYSTEM_PROMPT = """\
CRITICAL INSTRUCTION: Output ONLY a raw JSON object. No words, no punctuation \
outside the JSON, no markdown, no backticks, no newlines before or after. \
Your entire response must be parseable by json.loads(). \
Use the same action schemas as the main prompt. If the transcript is not a \
clear match for one allowed action, return {"action":"unknown",...}. Never guess.\
"""


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

def _call_ollama(system_prompt: str, user_text: str) -> str:
    """
    Call Ollama's /api/chat endpoint and return the assistant's message content.

    Raises:
        requests.RequestException on network errors.
        ValueError if the response structure is unexpected.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]

def _call_ollama_with_options(system_prompt: str, user_text: str, *, options: dict | None = None) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    }
    if options:
        payload["options"] = options
    response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    """
    Attempt to extract a JSON object from raw LLM output.

    Strips surrounding whitespace and common markdown fences before parsing.

    Raises:
        json.JSONDecodeError if no valid JSON object is found.
    """
    cleaned = raw.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove opening fence (```json or ```)
        lines = lines[1:] if len(lines) > 1 else lines
        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Find first '{' and last '}' to extract just the object
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise json.JSONDecodeError("No JSON object found", cleaned, 0)

    return json.loads(cleaned[start : end + 1])


_VALID_ACTIONS = {
    "open_application",
    "switch_window",
    "open_url",
    "web_search",
    "type_text",
    "press_keys",
    "system_action",
    "find_file",
    "open_recent",
    "reveal_in_finder",
    "paste_clip",
    "git_command",
    "run_script",
    "unknown",
}


def _validate_action(obj: dict) -> dict:
    """
    Validate that the parsed object has a known 'action' field.

    Returns the object unchanged if valid, raises ValueError otherwise.
    """
    action = obj.get("action")
    if action not in _VALID_ACTIONS:
        raise ValueError(f"Unknown action type: {action!r}")
    if action == "system_action" and obj.get("command") not in _SYSTEM_ACTION_COMMANDS:
        raise ValueError(f"Unknown system action command: {obj.get('command')!r}")
    if action == "git_command" and obj.get("command") not in _GIT_COMMANDS:
        raise ValueError(f"Unknown git command: {obj.get('command')!r}")
    return obj


def _unknown_action(transcript: str, reason: str) -> dict[str, str]:
    return {"action": "unknown", "raw_command": transcript, "reason": reason}


def _sanity_check(transcript: str, action: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_text(transcript)
    action_type = action.get("action", "unknown")

    if action_type == "unknown":
        return action

    def fail() -> dict[str, Any]:
        logger.warning("Sanity check rejected action %s for transcript %r", action, transcript)
        return _unknown_action(transcript, "sanity_check_failed")

    if not normalized:
        return fail()

    if action_type == "open_application":
        if not action.get("app_name") or not _is_open_application_text(normalized):
            return fail()
        return action

    if action_type == "switch_window":
        if (
            not action.get("app_name")
            or not _contains_any_phrase(normalized, _SWITCH_WINDOW_TRIGGERS)
            or _is_open_url_text(normalized)
        ):
            return fail()
        return action

    if action_type == "open_url":
        if not action.get("url") or not _is_open_url_text(normalized):
            return fail()
        return action

    if action_type == "web_search":
        if not action.get("query") or not _contains_any_phrase(normalized, _WEB_SEARCH_TRIGGERS):
            return fail()
        return action

    if action_type == "type_text":
        if not action.get("text") or not _contains_any_phrase(normalized, _TYPE_TEXT_TRIGGERS):
            return fail()
        return action

    if action_type == "press_keys":
        keys = action.get("keys")
        if not isinstance(keys, list) or not keys or not _contains_any_phrase(normalized, _PRESS_KEYS_TRIGGERS):
            return fail()
        return action

    if action_type == "find_file":
        if not action.get("query") or not _contains_any_phrase(normalized, _FIND_FILE_TRIGGERS):
            return fail()
        return action

    if action_type == "open_recent":
        if not action.get("query") or "open" not in normalized.split() or not _has_any_token(normalized, _OPEN_RECENT_HINTS):
            return fail()
        return action

    if action_type == "reveal_in_finder":
        if not action.get("query") or not _contains_any_phrase(normalized, _REVEAL_TRIGGERS):
            return fail()
        return action

    if action_type == "paste_clip":
        index = action.get("index")
        if not isinstance(index, int) or index < 1:
            return fail()
        if "paste" not in normalized.split() or not _has_any_token(normalized, _PASTE_CLIP_HINTS):
            return fail()
        return action

    if action_type == "git_command":
        if not action.get("command") or not _is_git_text(normalized):
            return fail()
        return action

    if action_type == "run_script":
        if not action.get("script") or not _is_run_script_text(normalized):
            return fail()
        return action

    if action_type == "system_action":
        command = action.get("command")
        if command == "volume_up" and _contains_any_phrase(normalized, _VOLUME_UP_TRIGGERS):
            return action
        if command == "volume_down" and _contains_any_phrase(normalized, _VOLUME_DOWN_TRIGGERS):
            return action
        if command == "mute" and _contains_any_phrase(normalized, _MUTE_TRIGGERS):
            return action
        if command == "screenshot" and _contains_any_phrase(normalized, _SCREENSHOT_TRIGGERS):
            return action
        if command == "sleep" and _contains_any_phrase(normalized, _SLEEP_TRIGGERS):
            return action
        return fail()

    return fail()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def interpret(transcript: str) -> dict[str, Any]:
    """
    Interpret a voice transcript and return a structured action dict.

    Pipeline:
    1. Call Ollama with the standard system prompt.
    2. Parse and validate the JSON response.
    3. On parse failure, retry with the strict prompt.
    4. On second failure, return an "unknown" fallback action.

    Args:
        transcript: The STT-produced text to interpret.

    Returns:
        A dict with at minimum an "action" key.
    """
    if not transcript or not transcript.strip():
        return _unknown_action("", "empty_transcript")

    # --- Attempt 1 ---
    try:
        logger.debug("Sending to Ollama (attempt 1): %s", transcript)
        raw = _call_ollama(_SYSTEM_PROMPT, transcript)
        logger.debug("Ollama raw response: %s", raw)
        action = _validate_action(_parse_json(raw))
        logger.info("Parsed action: %s", action)
        return _sanity_check(transcript, action)
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning("Attempt 1 parse failed (%s). Retrying with strict prompt…", exc)
    except requests.RequestException as exc:
        logger.error("Ollama request failed: %s", exc)
        return _unknown_action(transcript, f"ollama_error: {exc}")

    # --- Attempt 2 (strict prompt) ---
    try:
        combined_prompt = (
            f"{_STRICT_SYSTEM_PROMPT}\n\n"
            f"Voice command: {transcript}\n\n"
            "Respond with ONLY the JSON object:"
        )
        logger.debug("Sending to Ollama (attempt 2, strict)…")
        raw = _call_ollama(_STRICT_SYSTEM_PROMPT, combined_prompt)
        logger.debug("Ollama strict raw response: %s", raw)
        action = _validate_action(_parse_json(raw))
        logger.info("Parsed action (retry): %s", action)
        return _sanity_check(transcript, action)
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.error("Attempt 2 parse failed (%s). Falling back to unknown.", exc)
    except requests.RequestException as exc:
        logger.error("Ollama request failed on retry: %s", exc)

    # --- Fallback ---
    return _unknown_action(transcript, "parse_failure")

_INTENT_SYSTEM_PROMPT = """\
You are an intent classifier for a macOS voice assistant.

Classify the user's input into exactly one intent type:
- command: user wants you to do an action on the computer
- question: user wants an explanation/answer
- hybrid: user wants both actions and an answer

Output ONLY raw JSON:
{"intent_type": "command" | "question" | "hybrid"}

Guidelines:
- Imperative requests that ask the assistant to do one thing are command, even if they are short.
- Simple actions like "open chrome", "switch to terminal", "increase volume", "take a screenshot" are always command.
- "what is recursion", "why is my plant dying" -> question
- Hybrid means the user clearly asks for both an action and an answer in the same utterance.
- "open chrome and what is recursion" -> hybrid
- "search for python tutorials" -> command
- "open chrome" -> command
"""

_HYBRID_SPLIT_SYSTEM_PROMPT = """\
You split a single user utterance into actionable commands and questions.

Output ONLY raw JSON:
{"commands": ["..."], "questions": ["..."]}

Rules:
- commands should be short, imperative phrases appropriate for a voice assistant
- questions should be standalone questions the assistant can answer
- If an action is "search", treat the search itself as a command and extract the concept
  the user is asking about as a question when appropriate.
"""

_RESPONSE_SYSTEM_PROMPT = """\
You are JARVIS, a macOS conversational assistant.

You are given JSON context about:
- the user's input
- intent type
- actions that were initiated
- any available results
- short conversation history

For command intents: respond in 1-3 words only (e.g. "Done.", "Opening now.", "Got it."). No explanation.
For question or hybrid intents: respond in 1-2 sentences max. Be direct and conversational. No filler phrases, no preamble.
Never start a response with "Sure", "Of course", "Certainly", or similar affirmations.
Responses will be spoken aloud - keep them natural and concise.
- Your response must only describe what you actually did. Never describe an action you did not take.
- Ground every command response in `actions_taken` and `results`.
- If `actions_taken` is empty, or every result failed, explicitly acknowledge failure or that nothing was done.
- If any result succeeded, refer only to the successful action/result that actually happened.

Respond with plain text only (no JSON).
"""


def _response_max_tokens_for_intent(intent_type: str) -> int:
    if intent_type == "command":
        return COMMAND_RESPONSE_MAX_TOKENS
    return QUESTION_RESPONSE_MAX_TOKENS


def _grounded_command_response(context: dict[str, Any]) -> str:
    actions_taken = context.get("actions_taken")
    results = context.get("results")
    actions = [a for a in actions_taken if isinstance(a, dict)] if isinstance(actions_taken, list) else []
    result_items = [r for r in results if isinstance(r, dict)] if isinstance(results, list) else []
    successes = [r for r in result_items if r.get("success")]

    if not actions or not successes:
        return "I couldn't do that."

    message = str(successes[-1].get("message") or "").strip()
    if message:
        return message.rstrip(".")

    action = successes[-1].get("action")
    if not isinstance(action, dict):
        return "Done."

    action_type = action.get("action")
    if action_type == "open_application":
        return f"Opened {action.get('app_name', 'it')}".strip()
    if action_type == "switch_window":
        return f"Switched to {action.get('app_name', 'it')}".strip()
    if action_type == "open_url":
        return "Opened site."
    if action_type == "web_search":
        return "Search opened."
    if action_type == "type_text":
        return "Typed it."
    if action_type == "press_keys":
        return "Shortcut pressed."
    if action_type == "system_action":
        return str(action.get("command") or "Done").replace("_", " ").strip().capitalize()
    return "Done."


def classify_intent(text: str) -> str:
    if not text or not text.strip():
        return "command"
    normalized = _normalize_text(text)
    if normalized and has_known_command_trigger(normalized):
        if not text.strip().endswith("?") and not normalized.startswith(_QUESTION_PREFIXES):
            return "command"
    raw = _call_ollama(_INTENT_SYSTEM_PROMPT, text.strip())
    try:
        obj = _parse_json(raw)
        it = (obj.get("intent_type") or "").strip().lower()
        if it in {"command", "question", "hybrid"}:
            return it
    except Exception:
        pass
    # Fallback: assume command (safe: leads to action parsing) unless it looks like a question.
    t = text.strip().lower()
    if t.endswith("?") or t.startswith(("what", "why", "how", "when", "where", "who")):
        return "question"
    return "command"


def split_hybrid(text: str) -> dict[str, list[str]]:
    raw = _call_ollama(_HYBRID_SPLIT_SYSTEM_PROMPT, text.strip())
    obj = _parse_json(raw)
    commands = obj.get("commands") if isinstance(obj.get("commands"), list) else []
    questions = obj.get("questions") if isinstance(obj.get("questions"), list) else []
    commands = [str(c).strip() for c in commands if str(c).strip()]
    questions = [str(q).strip() for q in questions if str(q).strip()]
    return {"commands": commands, "questions": questions}


def generate_response(context: dict) -> str:
    """
    Generate a natural spoken response for the assistant.

    Context keys:
      intent_type, user_input, actions_taken, results, conversation_history
    """
    intent_type = str(context.get("intent_type") or "").strip().lower()
    if intent_type == "command":
        return _grounded_command_response(context)
    options = {
        "temperature": RESPONSE_TEMPERATURE,
        "num_predict": _response_max_tokens_for_intent(intent_type),
    }
    user_text = json.dumps(context, ensure_ascii=False)
    raw = _call_ollama_with_options(_RESPONSE_SYSTEM_PROMPT, user_text, options=options)
    return raw.strip()


class LLMWorker:
    """
    Background worker for non-blocking LLM calls.

    Use submit(transcript) -> Future[action_dict].
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue[tuple[str, object, Future]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="llm-worker")
        self._thread.start()

    def submit(self, task: str, payload: object) -> Future:
        fut: Future = Future()
        self._queue.put((task, payload, fut))
        return fut

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(("", "", Future()))
        except queue.Full:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            task, payload, fut = self._queue.get()
            if self._stop.is_set():
                break
            if fut.cancelled():
                continue
            try:
                if task == "interpret":
                    fut.set_result(interpret(str(payload)))
                elif task == "classify_intent":
                    fut.set_result(classify_intent(str(payload)))
                elif task == "split_hybrid":
                    fut.set_result(split_hybrid(str(payload)))
                elif task == "generate_response":
                    fut.set_result(generate_response(payload if isinstance(payload, dict) else {}))
                else:
                    fut.set_result({"action": "unknown", "raw_command": str(payload), "reason": "unknown_llm_task"})
            except Exception as exc:
                fut.set_result({"error": f"llm_worker_error: {exc}"})


_worker_singleton: LLMWorker | None = None


def get_worker() -> LLMWorker:
    global _worker_singleton
    if _worker_singleton is None:
        _worker_singleton = LLMWorker()
    return _worker_singleton


def interpret_async(transcript: str) -> Future:
    """Submit transcript for LLM interpretation without blocking the caller."""
    return get_worker().submit("interpret", transcript)


def classify_intent_async(text: str) -> Future:
    return get_worker().submit("classify_intent", text)


def split_hybrid_async(text: str) -> Future:
    return get_worker().submit("split_hybrid", text)


def generate_response_async(context: dict) -> Future:
    return get_worker().submit("generate_response", context)
