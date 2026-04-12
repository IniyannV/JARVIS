"""
JSONL command logger and application log setup.
Writes every executed command to ~/.voice-assistant/command_log.jsonl
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import APP_LOG_PATH, COMMAND_LOG_PATH, LOG_DIR
import state


def setup_logging() -> logging.Logger:
    """Configure root logger to write to file and stdout."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_fmt = "%Y-%m-%dT%H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.insert(0, logging.FileHandler(APP_LOG_PATH))
    except OSError:
        pass

    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        datefmt=date_fmt,
        handlers=handlers,
    )
    # Keep upstream libraries quieter; our app logs at DEBUG.
    logging.getLogger("faster_whisper").setLevel(logging.INFO)
    return logging.getLogger("voice-assistant")


def log_command(
    transcript: str,
    action_type: str,
    action_detail: dict,
    success: bool,
    error: str = "",
) -> None:
    """
    Append one structured record to the JSONL command log.

    Args:
        transcript:    The raw STT transcript.
        action_type:   The action field from the parsed LLM response.
        action_detail: Full action dict as returned by the LLM.
        success:       Whether the executor succeeded.
        error:         Optional error message on failure.
    """
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "transcript": transcript,
        "action_type": action_type,
        "action_detail": action_detail,
        "success": success,
        "error": error,
    }

    try:
        with open(COMMAND_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as e:
        logging.getLogger("voice-assistant").warning(
            "Failed to write command log to %s: %s", COMMAND_LOG_PATH, e
        )

    # Push to dashboard history
    if state.dashboard is not None:
        action_label = action_detail.get("action", action_type)
        result_label = "✓" if success else "✗"
        state.dashboard.add_history_entry(
            record["timestamp"],
            transcript,
            f"{result_label} {action_label}",
        )


def log_toggle(state: bool) -> None:
    """Log hotkey toggle events with a timestamp."""
    logger = logging.getLogger("voice-assistant.hotkey")
    label = "ON" if state else "OFF"
    logger.info("Listening toggled %s at %s", label, datetime.now(timezone.utc).isoformat())
