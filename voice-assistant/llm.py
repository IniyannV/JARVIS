"""
Ollama HTTP client with JSON parsing and retry logic.

Sends the voice transcript to a local Ollama instance and parses a structured
action object from the response. Retries once with a stricter prompt on parse
failure. Falls back to {"action": "unknown"} on second failure.
"""

import json
import logging
import queue
import threading
from typing import Any

import requests
from concurrent.futures import Future

from config import OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_URL

logger = logging.getLogger("voice-assistant.llm")

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

Rules:
- Use open_url for specific websites (YouTube, GitHub, Reddit, Gmail, etc.).
- Use web_search when the user says "search for", "google", "look up", or asks a question.
- For web_search, the query should be what would go into a Google search bar.
- Use the full, proper application name (e.g., "Google Chrome" not "chrome").
- For press_keys, use lowercase pyautogui key names (e.g., "command", "shift", "space").
- If the command is unclear or does not map to any action, use "unknown".
- NEVER output anything other than the JSON object.\
"""

_STRICT_SYSTEM_PROMPT = """\
CRITICAL INSTRUCTION: Output ONLY a raw JSON object. No words, no punctuation \
outside the JSON, no markdown, no backticks, no newlines before or after. \
Your entire response must be parseable by json.loads(). \
Convert this voice command to the appropriate JSON action schema.\
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
    return obj


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
        return {"action": "unknown", "raw_command": "", "reason": "empty_transcript"}

    # --- Attempt 1 ---
    try:
        logger.debug("Sending to Ollama (attempt 1): %s", transcript)
        raw = _call_ollama(_SYSTEM_PROMPT, transcript)
        logger.debug("Ollama raw response: %s", raw)
        action = _validate_action(_parse_json(raw))
        logger.info("Parsed action: %s", action)
        return action
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning("Attempt 1 parse failed (%s). Retrying with strict prompt…", exc)
    except requests.RequestException as exc:
        logger.error("Ollama request failed: %s", exc)
        return {
            "action": "unknown",
            "raw_command": transcript,
            "reason": f"ollama_error: {exc}",
        }

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
        return action
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.error("Attempt 2 parse failed (%s). Falling back to unknown.", exc)
    except requests.RequestException as exc:
        logger.error("Ollama request failed on retry: %s", exc)

    # --- Fallback ---
    return {"action": "unknown", "raw_command": transcript, "reason": "parse_failure"}


class LLMWorker:
    """
    Background worker for non-blocking LLM calls.

    Use submit(transcript) -> Future[action_dict].
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue[tuple[str, Future]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="llm-worker")
        self._thread.start()

    def submit(self, transcript: str) -> Future:
        fut: Future = Future()
        self._queue.put((transcript, fut))
        return fut

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(("", Future()))
        except queue.Full:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            transcript, fut = self._queue.get()
            if self._stop.is_set():
                break
            if fut.cancelled():
                continue
            try:
                fut.set_result(interpret(transcript))
            except Exception as exc:
                fut.set_result(
                    {"action": "unknown", "raw_command": transcript, "reason": f"llm_worker_error: {exc}"}
                )


_worker_singleton: LLMWorker | None = None


def get_worker() -> LLMWorker:
    global _worker_singleton
    if _worker_singleton is None:
        _worker_singleton = LLMWorker()
    return _worker_singleton


def interpret_async(transcript: str) -> Future:
    """Submit transcript for LLM interpretation without blocking the caller."""
    return get_worker().submit(transcript)
