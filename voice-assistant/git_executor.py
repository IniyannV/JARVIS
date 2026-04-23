"""
Git command execution helpers for voice-driven workflows.
"""

import logging
import os
import subprocess

logger = logging.getLogger("voice-assistant.git")

_MAX_OUTPUT_CHARS = 300


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _run_osascript(script: str) -> str:
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        logger.warning("osascript is not installed; using current working directory.")
        return ""

    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _detect_repo_path(repo_path: str | None) -> str:
    if repo_path:
        candidate = os.path.abspath(os.path.expanduser(repo_path))
        if os.path.isdir(candidate):
            return candidate

    frontmost_path = _run_osascript(
        """
tell application "System Events"
    set frontApp to name of first application process whose frontmost is true
end tell
if frontApp is "Terminal" then
    tell application "Terminal" to get custom title of front window
else if frontApp is "iTerm2" then
    tell application "iTerm2" to get name of current session of current window
else if frontApp is "iTerm" then
    tell application "iTerm" to get name of current session of current window
else
    return ""
end if
""".strip()
    )
    if frontmost_path:
        candidate = os.path.abspath(os.path.expanduser(frontmost_path))
        if os.path.isdir(candidate):
            return candidate

    return os.getcwd()


def execute_git(command: str, args: list[str], repo_path: str | None) -> tuple[bool, str]:
    """
    Execute a supported git command in the detected repository path.
    """
    command = command.strip().lower()
    cwd = _detect_repo_path(repo_path)

    base_commands: dict[str, list[str]] = {
        "status": ["git", "status", "--short"],
        "add_all": ["git", "add", "-A"],
        "push": ["git", "push"],
        "pull": ["git", "pull"],
        "log": ["git", "log", "--oneline", "-5"],
        "diff": ["git", "diff", "--stat"],
        "branch": ["git", "branch", "--show-current"],
        "stash": ["git", "stash"],
        "stash_pop": ["git", "stash", "pop"],
    }

    if command == "commit":
        message = args[0].strip() if args else ""
        if not message:
            return False, "Commit message is required."
        git_cmd = ["git", "commit", "-m", message]
    elif command == "checkout":
        branch = args[0].strip() if args else ""
        if not branch:
            return False, "Branch name is required."
        git_cmd = ["git", "checkout", branch]
    else:
        git_cmd = base_commands.get(command)
        if git_cmd is None:
            return False, f"Unsupported git command: {command}"

    try:
        result = subprocess.run(
            git_cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
    except FileNotFoundError:
        return False, "git is not installed."

    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or f"git {command} failed."
        return False, _truncate(error)

    output = result.stdout.strip() or result.stderr.strip() or f"git {command} completed."
    return True, _truncate(output)
