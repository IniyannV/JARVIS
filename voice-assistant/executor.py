"""
Action executor — routes structured LLM action dicts to macOS automation.

Each public execute_* function returns (success: bool, message: str).
The top-level execute() dispatcher handles routing.
"""

import logging
import os
import subprocess
import time
from typing import Tuple

import pyautogui

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
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("AppleScript error: %s", result.stderr.strip())
        return False, result.stderr.strip()
    return True, result.stdout.strip()


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

    result = subprocess.run(["open", url], capture_output=True, text=True)
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
    result = subprocess.run(["open", url], capture_output=True, text=True)
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
        import subprocess as _sp
        proc = _sp.run(
            ["pbcopy"],
            input=text.encode("utf-8"),
            check=True,
        )
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
        result = subprocess.run(
            ["screencapture", "-i", path],
            capture_output=True,
        )
        if result.returncode == 0:
            logger.info("Screenshot saved to %s", path)
            return True, f"Screenshot saved to {path}"
        return False, result.stderr.decode().strip()

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

    elif action_type == "unknown":
        return execute_unknown(
            action.get("raw_command", ""),
            action.get("reason", "unknown"),
        )

    else:
        logger.error("Unrecognized action type: %s", action_type)
        return False, f"Unrecognized action: {action_type}"
