"""
Safety layer: dangerous-action filter and confirmation prompt.
"""

import logging
import queue
import threading
from typing import Optional

from config import DANGEROUS_ACTIONS, DANGEROUS_CONFIRM_TIMEOUT

logger = logging.getLogger("voice-assistant.safety")


def is_dangerous(action: dict) -> bool:
    """
    Return True if the action requires user confirmation before execution.

    Checks the 'command' field for system_action types and the action type itself.
    """
    action_type = action.get("action", "")
    command = action.get("command", "")

    if action_type == "system_action" and command in DANGEROUS_ACTIONS:
        return True

    if action_type in DANGEROUS_ACTIONS:
        return True

    return False


def _read_confirmation(result_queue: queue.Queue) -> None:
    """
    Block-read one line from stdin and push 'y'/'n' to the queue.
    Runs in a daemon thread so the main logic can time out.
    """
    try:
        answer = input().strip().lower()
        result_queue.put(answer)
    except EOFError:
        result_queue.put("n")


def confirm_dangerous_action(action: dict) -> bool:
    """
    Print a warning and wait up to DANGEROUS_CONFIRM_TIMEOUT seconds for the user
    to press Enter after typing 'y'.

    Returns True if confirmed, False otherwise.
    """
    command = action.get("command", action.get("action", "unknown"))
    print(
        f"\n[SAFETY] Dangerous action requested: '{command}'\n"
        f"Type 'y' and press Enter within {DANGEROUS_CONFIRM_TIMEOUT}s to confirm: ",
        end="",
        flush=True,
    )

    result_queue: queue.Queue = queue.Queue()
    reader = threading.Thread(target=_read_confirmation, args=(result_queue,), daemon=True)
    reader.start()

    try:
        answer = result_queue.get(timeout=DANGEROUS_CONFIRM_TIMEOUT)
    except queue.Empty:
        print("\n[SAFETY] Confirmation timed out — action cancelled.")
        logger.warning("Dangerous action '%s' timed out waiting for confirmation.", command)
        return False

    if answer == "y":
        logger.info("Dangerous action '%s' confirmed by user.", command)
        return True

    print("[SAFETY] Action cancelled.")
    logger.info("Dangerous action '%s' rejected by user.", command)
    return False


def guard(action: dict) -> Optional[dict]:
    """
    Gate an action through the safety layer.

    Returns the action unchanged if safe or confirmed, None if blocked.
    """
    if not is_dangerous(action):
        return action

    if confirm_dangerous_action(action):
        return action

    return None
