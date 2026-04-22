"""
logger.py — Standardised logging setup for the pipeline.
"""

import logging
import sys
from typing import Optional


_LOG_FORMAT = "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def get_logger(
    name: str, level: Optional[int] = None
) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (typically __name__).
        level: Optional log level override.

    Returns:
        Configured Logger.
    """
    global _configured
    if not _configured:
        _setup_root_logger()
        _configured = True

    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def _setup_root_logger() -> None:
    """Configure the root logger with console output."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Avoid duplicate handlers on reimport
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        root.addHandler(handler)
