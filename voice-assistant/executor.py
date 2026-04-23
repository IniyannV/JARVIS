"""
Action executor — routes structured LLM action dicts to macOS automation.

Each public execute_* function returns (success: bool, message: str).
The top-level execute() dispatcher handles routing.
"""

import logging
import os
import shlex
import subprocess
import time
from datetime import datetime
from typing import Tuple

import pyautogui
from concurrent.futures import Future, ThreadPoolExecutor

from config import TERMINAL_SCRIPT_ALLOWLIST
import state

logger = logging.getLogger("voice-assistant.executor")

# pyautogui safety — disable the fail-safe corner (top-left) to avoid
# accidental interruption during automated typing.
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.05


# ---------------------------------------------------------------------------
# AppleScript helpers
# ---------------------------------------------------------------------------

def _sanitize_applescript_string(s: str) -> str:
    """
    Escape double-quotes in a string to prevent AppleScript injection.

    AppleScript strings are delimited by double-quotes; a literal quote inside
    the string must be escaped as \" in the shell-embedded script.
    """
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _run_applescript(script: str) -> Tuple[bool, str]:
    """Run an AppleScript expression via osascript. Returns (success, output)."""
    command_ok, result, error = _run_subprocess(["osascript", "-e", script])
    if not command_ok or result is None:
        logger.error("AppleScript unavailable: %s", error)
        return False, error
    if result.returncode != 0:
        logger.error("AppleScript error: %s", result.stderr.strip())
        return False, result.stderr.strip()
    return True, result.stdout.strip()


def _run_subprocess(
    command: list[str],
    *,
    input_text: str | None = None,
    cwd: str | None = None,
    timeout: float | None = None,
) -> Tuple[bool, subprocess.CompletedProcess[str] | None, str]:
    """
    Run a subprocess with consistent capture/error handling.
    """
    try:
        result = subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
        )
    except FileNotFoundError:
        return False, None, f"Required tool not installed: {command[0]}"
    except subprocess.TimeoutExpired:
        return False, None, f"Command timed out: {' '.join(command)}"
    return True, result, ""


def _result_error_message(result: subprocess.CompletedProcess[str], fallback: str) -> str:
    return result.stderr.strip() or result.stdout.strip() or fallback


def _truncate_output(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _find_file_matches(query: str) -> Tuple[bool, list[str], str]:
    query = query.strip()
    if not query:
        return False, [], "No file query provided"

    command_ok, result, error = _run_subprocess(["mdfind", "-name", query])
    if not command_ok or result is None:
        return False, [], error
    if result.returncode != 0:
        return False, [], _result_error_message(result, f"Failed to search for: {query}")

    matches = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return True, matches, ""


def _resolve_file_path(query: str) -> Tuple[bool, str]:
    candidate = os.path.abspath(os.path.expanduser(query.strip()))
    if query.strip() and os.path.exists(candidate):
        return True, candidate

    success, matches, error = _find_file_matches(query)
    if not success:
        return False, error
    if not matches:
        return False, f"No file found for: {query}"
    return True, matches[0]


def _open_path(path: str, reveal: bool = False) -> Tuple[bool, str]:
    command = ["open", "-R", path] if reveal else ["open", path]
    command_ok, result, error = _run_subprocess(command)
    if not command_ok or result is None:
        return False, error
    if result.returncode != 0:
        action = "reveal" if reveal else "open"
        return False, _result_error_message(result, f"Failed to {action} {path}")
    return True, ""


def _last_used_sort_key(path: str) -> tuple[float, float]:
    last_used_ts = 0.0
    command_ok, result, _ = _run_subprocess(["mdls", "-raw", "-name", "kMDItemLastUsedDate", path])
    if command_ok and result is not None and result.returncode == 0:
        raw_date = result.stdout.strip()
        if raw_date and raw_date != "(null)":
            try:
                last_used_ts = datetime.strptime(raw_date, "%Y-%m-%d %H:%M:%S %z").timestamp()
            except ValueError:
                last_used_ts = 0.0

    try:
        modified_ts = os.path.getmtime(path)
    except OSError:
        modified_ts = 0.0

    return last_used_ts, modified_ts


def _fallback_open_recent_path(query: str) -> Tuple[bool, str]:
    success, matches, error = _find_file_matches(query)
    if not success:
        return False, error
    if not matches:
        return False, f"No file found for: {query}"

    ranked_matches = sorted(matches[:50], key=_last_used_sort_key, reverse=True)
    return True, ranked_matches[0]


def _finalize_new_executor(action_detail: dict, success: bool, message: str) -> Tuple[bool, str]:
    """
    Use the shared JSONL logger when an executor is invoked outside the main pipeline.
    """
    if state.executor_service is None:
        from logger import log_command

        log_command(
            transcript="",
            action_type=action_detail.get("action", "unknown"),
            action_detail=action_detail,
            success=success,
            error="" if success else message,
        )
    return success, message


# ---------------------------------------------------------------------------
# Individual executors
# ---------------------------------------------------------------------------

def execute_open_application(app_name: str) -> Tuple[bool, str]:
    """Activate (launch or bring to front) a macOS application by name."""
    safe_name = _sanitize_applescript_string(app_name)
    script = f'tell application "{safe_name}" to activate'
    success, output = _run_applescript(script)
    if success:
        logger.info("Opened application: %s", app_name)
        return True, f"Opened {app_name}"
    logger.warning("Failed to open %s: %s", app_name, output)
    return False, output


def execute_switch_window(app_name: str) -> Tuple[bool, str]:
    """
    Bring an already-running application to the foreground.

    If the app is not running, falls back to launching it.
    """
    safe_name = _sanitize_applescript_string(app_name)

    # Check if the application is running
    check_script = (
        f'tell application "System Events" to '
        f'(name of processes) contains "{safe_name}"'
    )
    _, is_running_str = _run_applescript(check_script)

    if is_running_str.lower() == "true":
        script = f'tell application "{safe_name}" to activate'
        success, output = _run_applescript(script)
        if success:
            logger.info("Switched to window: %s", app_name)
            return True, f"Switched to {app_name}"
        return False, output

    # App not running — launch it
    logger.info("%s not running, launching instead.", app_name)
    return execute_open_application(app_name)


def execute_open_url(url: str) -> Tuple[bool, str]:
    """Open a URL in the default browser using macOS `open` command."""
    if not url:
        return False, "No URL provided"

    # Ensure the URL has a scheme
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    command_ok, result, error = _run_subprocess(["open", url])
    if not command_ok or result is None:
        logger.warning("Failed to open URL %s: %s", url, error)
        return False, error
    if result.returncode == 0:
        logger.info("Opened URL: %s", url)
        return True, f"Opened {url}"
    logger.error("Failed to open URL %s: %s", url, result.stderr)
    return False, result.stderr.strip()


def execute_web_search(query: str) -> Tuple[bool, str]:
    """Open a Google search for the given query in the default browser."""
    if not query:
        return False, "No search query provided"

    import urllib.parse
    encoded = urllib.parse.quote_plus(query)
    url = f"https://www.google.com/search?q={encoded}"
    command_ok, result, error = _run_subprocess(["open", url])
    if not command_ok or result is None:
        logger.warning("Failed web search for %s: %s", query, error)
        return False, error
    if result.returncode == 0:
        logger.info("Web search: %s", query)
        return True, f"Searched: {query}"
    return False, result.stderr.strip()


def execute_type_text(text: str) -> Tuple[bool, str]:
    """
    Type text into the currently focused input using pyautogui.

    Handles ASCII via typewrite (fast, interval-based) and falls back to
    clipboard-paste for unicode characters that typewrite cannot handle.
    """
    if not text:
        return False, "Empty text"

    # Detect if the text contains non-ASCII characters
    try:
        text.encode("ascii")
        is_ascii = True
    except UnicodeEncodeError:
        is_ascii = False

    if is_ascii:
        # typewrite works reliably for ASCII
        pyautogui.typewrite(text, interval=0.03)
    else:
        # For unicode, use clipboard to paste
        command_ok, result, error = _run_subprocess(["pbcopy"], input_text=text)
        if not command_ok or result is None:
            return False, error
        if result.returncode != 0:
            return False, _result_error_message(result, "pbcopy failed")
        time.sleep(0.1)
        pyautogui.hotkey("command", "v")

    logger.info("Typed text: %s", text[:80])
    return True, f"Typed: {text[:40]}"


def execute_press_keys(keys: list) -> Tuple[bool, str]:
    """
    Press a keyboard shortcut using pyautogui.hotkey.

    Keys should be pyautogui-compatible names: 'command', 'shift', 'space', etc.
    """
    if not keys:
        return False, "No keys specified"

    # Normalize key names
    normalized = [k.lower().strip() for k in keys]
    pyautogui.hotkey(*normalized)
    logger.info("Pressed keys: %s", "+".join(normalized))
    return True, f"Pressed {'+'.join(normalized)}"


def execute_system_action(command: str) -> Tuple[bool, str]:
    """Execute a predefined system-level command."""
    command = command.lower().strip()

    if command == "volume_up":
        script = (
            "set volume output volume "
            "(output volume of (get volume settings) + 10)"
        )
        success, out = _run_applescript(script)
        if success:
            return True, "Volume increased"
        return False, out

    elif command == "volume_down":
        script = (
            "set volume output volume "
            "(output volume of (get volume settings) - 10)"
        )
        success, out = _run_applescript(script)
        if success:
            return True, "Volume decreased"
        return False, out

    elif command == "mute":
        script = "set volume with output muted"
        success, out = _run_applescript(script)
        if success:
            return True, "Muted"
        return False, out

    elif command == "screenshot":
        path = os.path.expanduser("~/Desktop/screenshot.png")
        # -i = interactive crosshair selection; remove -i for full-screen
        command_ok, result, error = _run_subprocess(["screencapture", "-i", path])
        if not command_ok or result is None:
            return False, error
        if result.returncode == 0:
            logger.info("Screenshot saved to %s", path)
            return True, f"Screenshot saved to {path}"
        return False, result.stderr.strip()

    elif command == "sleep":
        # Safety layer handles confirmation before this point
        script = 'tell application "System Events" to sleep'
        success, out = _run_applescript(script)
        if success:
            return True, "System going to sleep"
        return False, out

    else:
        logger.warning("Unknown system command: %s", command)
        return False, f"Unknown system command: {command}"


def execute_unknown(raw_command: str, reason: str) -> Tuple[bool, str]:
    """Log and surface an unrecognised or unparseable command."""
    logger.warning("Unknown command — raw: %r, reason: %s", raw_command, reason)
    return False, f"Could not execute: {raw_command!r} ({reason})"


def execute_find_file(query: str) -> Tuple[bool, str]:
    """Find a file by name and open the top match."""
    action_detail = {"action": "find_file", "query": query}
    success, path_or_error = _resolve_file_path(query)
    if not success:
        logger.warning("Failed to find file for %s: %s", query, path_or_error)
        return _finalize_new_executor(action_detail, False, path_or_error)

    opened, open_error = _open_path(path_or_error)
    if not opened:
        logger.warning("Failed to open file %s: %s", path_or_error, open_error)
        return _finalize_new_executor(action_detail, False, open_error)

    logger.info("Found and opened file for %s: %s", query, path_or_error)
    return _finalize_new_executor(action_detail, True, f"Opened {path_or_error}")


def execute_open_recent(query: str) -> Tuple[bool, str]:
    """Open the most recently modified Finder match, with Spotlight fallback."""
    action_detail = {"action": "open_recent", "query": query}
    if not query.strip():
        logger.warning("Failed to open recent file: empty query")
        return _finalize_new_executor(action_detail, False, "No file query provided")

    safe_query = _sanitize_applescript_string(query.strip())
    script = f"""
tell application "Finder"
    set matchedFiles to (every file of entire contents of startup disk whose name contains "{safe_query}")
    if (count of matchedFiles) is 0 then
        return ""
    end if
    set newestFile to item 1 of matchedFiles
    set newestDate to modification date of newestFile
    repeat with currentFile in matchedFiles
        if (modification date of currentFile) > newestDate then
            set newestFile to currentFile
            set newestDate to modification date of currentFile
        end if
    end repeat
    return POSIX path of (newestFile as alias)
end tell
""".strip()

    success, applescript_output = _run_applescript(script)
    path = applescript_output.strip() if success else ""
    if not path:
        fallback_success, path_or_error = _fallback_open_recent_path(query)
        if not fallback_success:
            logger.warning("Failed to open recent file for %s: %s", query, path_or_error)
            return _finalize_new_executor(action_detail, False, path_or_error)
        path = path_or_error

    opened, open_error = _open_path(path)
    if not opened:
        logger.warning("Failed to open recent file %s: %s", path, open_error)
        return _finalize_new_executor(action_detail, False, open_error)

    logger.info("Opened recent file for %s: %s", query, path)
    return _finalize_new_executor(action_detail, True, f"Opened {path}")


def execute_reveal_in_finder(query: str) -> Tuple[bool, str]:
    """Reveal a file in Finder by path or by Spotlight name lookup."""
    action_detail = {"action": "reveal_in_finder", "query": query}
    success, path_or_error = _resolve_file_path(query)
    if not success:
        logger.warning("Failed to reveal file for %s: %s", query, path_or_error)
        return _finalize_new_executor(action_detail, False, path_or_error)

    revealed, reveal_error = _open_path(path_or_error, reveal=True)
    if not revealed:
        logger.warning("Failed to reveal file %s: %s", path_or_error, reveal_error)
        return _finalize_new_executor(action_detail, False, reveal_error)

    logger.info("Revealed file for %s: %s", query, path_or_error)
    return _finalize_new_executor(action_detail, True, f"Revealed {path_or_error}")


def execute_paste_clip(index: int) -> Tuple[bool, str]:
    """Paste an item from clipboard history, using 1-indexed history lookup."""
    action_detail = {"action": "paste_clip", "index": index}
    try:
        clip_index = int(index)
    except (TypeError, ValueError):
        logger.warning("Failed to paste clipboard item: invalid index %r", index)
        return _finalize_new_executor(action_detail, False, f"Invalid clipboard index: {index}")

    if clip_index < 1:
        logger.warning("Failed to paste clipboard item: index must be >= 1")
        return _finalize_new_executor(action_detail, False, "Clipboard index must be at least 1")

    import clipboard

    history = clipboard.get_history()
    if clip_index > len(history):
        logger.warning(
            "Failed to paste clipboard item: index %s out of range (history=%s)",
            clip_index,
            len(history),
        )
        return _finalize_new_executor(
            action_detail,
            False,
            f"Clipboard history has only {len(history)} item(s)",
        )

    text = history[-clip_index]
    try:
        clipboard.inject(text)
    except Exception as exc:
        logger.warning("Failed to paste clipboard item %s: %s", clip_index, exc)
        return _finalize_new_executor(action_detail, False, str(exc))

    logger.info("Pasted clipboard item %s", clip_index)
    return _finalize_new_executor(action_detail, True, f"Pasted clipboard item {clip_index}")


def execute_git_command(
    command: str,
    message: str = "",
    branch: str = "",
    repo_path: str | None = None,
) -> Tuple[bool, str]:
    """Execute a supported git command in the detected repository."""
    from git_executor import execute_git

    git_command = command.strip().lower()
    action_detail = {
        "action": "git_command",
        "command": git_command,
        "message": message,
        "branch": branch,
        "repo_path": repo_path,
    }
    git_args: list[str] = []
    if git_command == "commit":
        git_args = [message]
    elif git_command == "checkout":
        git_args = [branch]

    success, output = execute_git(git_command, git_args, repo_path)
    if success:
        logger.info("Executed git command %s: %s", git_command, output)
    else:
        logger.warning("Failed git command %s: %s", git_command, output)
    return _finalize_new_executor(action_detail, success, output)


def execute_run_script(script: str, args: str) -> Tuple[bool, str]:
    """Run an allowlisted terminal command with arguments."""
    action_detail = {"action": "run_script", "script": script, "args": args}
    if not script.strip():
        logger.warning("Failed to run script: empty command")
        return _finalize_new_executor(action_detail, False, "No script provided")

    try:
        script_parts = shlex.split(script)
    except ValueError as exc:
        logger.warning("Failed to parse script %r: %s", script, exc)
        return _finalize_new_executor(action_detail, False, str(exc))

    if not script_parts:
        logger.warning("Failed to run script: no command found in %r", script)
        return _finalize_new_executor(action_detail, False, "No script provided")

    base_command = os.path.basename(script_parts[0])
    if base_command not in TERMINAL_SCRIPT_ALLOWLIST:
        logger.warning("Rejected non-allowlisted command: %s", base_command)
        return _finalize_new_executor(action_detail, False, f"Command not in allowlist: {base_command}")

    try:
        arg_parts = shlex.split(args)
    except ValueError as exc:
        logger.warning("Failed to parse args for %s: %s", script, exc)
        return _finalize_new_executor(action_detail, False, str(exc))

    command_ok, result, error = _run_subprocess(
        script_parts + arg_parts,
        timeout=30,
    )
    if not command_ok or result is None:
        logger.warning("Failed to run script %s: %s", base_command, error)
        return _finalize_new_executor(action_detail, False, error)
    if result.returncode != 0:
        failure = _truncate_output(
            _result_error_message(result, f"{base_command} failed"),
            400,
        )
        logger.warning("Script failed %s: %s", base_command, failure)
        return _finalize_new_executor(action_detail, False, failure)

    success_message = _truncate_output(
        result.stdout.strip() or result.stderr.strip() or f"{base_command} completed",
        400,
    )
    logger.info("Ran script %s: %s", base_command, success_message)
    return _finalize_new_executor(action_detail, True, success_message)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def execute(action: dict) -> Tuple[bool, str]:
    """
    Route a structured action dict to the appropriate executor.

    Returns (success, human-readable message).
    """
    action_type = action.get("action", "unknown")

    if action_type == "open_application":
        return execute_open_application(action.get("app_name", ""))

    elif action_type == "switch_window":
        return execute_switch_window(action.get("app_name", ""))

    elif action_type == "open_url":
        return execute_open_url(action.get("url", ""))

    elif action_type == "web_search":
        return execute_web_search(action.get("query", ""))

    elif action_type == "type_text":
        return execute_type_text(action.get("text", ""))

    elif action_type == "press_keys":
        return execute_press_keys(action.get("keys", []))

    elif action_type == "system_action":
        return execute_system_action(action.get("command", ""))

    elif action_type == "find_file":
        return execute_find_file(action.get("query", ""))

    elif action_type == "open_recent":
        return execute_open_recent(action.get("query", ""))

    elif action_type == "reveal_in_finder":
        return execute_reveal_in_finder(action.get("query", ""))

    elif action_type == "paste_clip":
        return execute_paste_clip(action.get("index", 1))

    elif action_type == "git_command":
        return execute_git_command(
            action.get("command", ""),
            message=action.get("message", ""),
            branch=action.get("branch", ""),
            repo_path=action.get("repo_path"),
        )

    elif action_type == "run_script":
        return execute_run_script(
            action.get("script", ""),
            action.get("args", ""),
        )

    elif action_type == "unknown":
        return execute_unknown(
            action.get("raw_command", ""),
            action.get("reason", "unknown"),
        )

    else:
        logger.error("Unrecognized action type: %s", action_type)
        return False, f"Unrecognized action: {action_type}"


class ExecutorService:
    """
    Thread-pool backed action executor.

    submit(action, transcript) returns a Future[(success, message)] and also
    performs best-effort side effects (dashboard + speech) when the action
    completes.
    """

    def __init__(self, max_workers: int = 4) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="exec")

    def submit(self, action: dict, transcript: str = "") -> Future:
        future: Future = self._pool.submit(execute, action)

        def _done(f: Future) -> None:
            try:
                success, message = f.result()
            except Exception as exc:
                success, message = False, str(exc)

            # Dashboard update
            if state.dashboard is not None:
                try:
                    state.dashboard.update_action(message, success)
                except Exception:
                    pass

            # Menu bar last command
            if hasattr(state, "menubar_app") and state.menubar_app is not None:
                try:
                    label = f"{'✓' if success else '✗'} {transcript or message}"
                    state.menubar_app.update_last_command(label)
                except Exception:
                    pass

        future.add_done_callback(_done)
        return future

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
