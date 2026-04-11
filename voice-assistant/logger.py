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


def setup_logging() -> logging.Logger:
    """Configure root logger to write to file and stdout."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_fmt = "%Y-%m-%dT%H:%M:%S"

    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        datefmt=date_fmt,
        handlers=[
            logging.FileHandler(APP_LOG_PATH),
            logging.StreamHandler(sys.stdout),
        ],
    )
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

    with open(COMMAND_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def log_toggle(state: bool) -> None:
    """Log hotkey toggle events with a timestamp."""
    logger = logging.getLogger("voice-assistant.hotkey")
    label = "ON" if state else "OFF"
    logger.info("Listening toggled %s at %s", label, datetime.now(timezone.utc).isoformat())
