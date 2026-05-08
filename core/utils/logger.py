"""
core/utils/logger.py

Centralized logging setup for the pipeline.

All modules obtain their logger via get_logger(__name__).
Log level is driven by LOG_LEVEL in .env via AppConfig.

Design rules:
- one logger per module, named by __name__
- single StreamHandler to stdout — no file sinks, no rotation
- format includes timestamp, level, module name, message
- LOG_LEVEL from config drives verbosity at runtime
- no noisy debug spam — agents log entry, exit, and errors only
- logger is safe to call before pipeline state exists
"""

import logging
import sys
from typing import Optional


# -----------------------------------------------
# Format
# -----------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# -----------------------------------------------
# Root pipeline logger name
# -----------------------------------------------

PIPELINE_LOGGER = "meeting_analyzer"


# -----------------------------------------------
# Internal state
# -----------------------------------------------

_configured: bool = False


# -----------------------------------------------
# Setup
# -----------------------------------------------

def _configure_root_logger(level: int) -> None:
    """
    Configure the root pipeline logger once.
    Idempotent — safe to call multiple times.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger(PIPELINE_LOGGER)
    root.setLevel(level)

    # Avoid duplicate handlers if called multiple times in tests
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(handler)

    # Suppress propagation to the Python root logger to avoid duplicate output
    root.propagate = False
    _configured = True


# -----------------------------------------------
# Public API
# -----------------------------------------------

def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a named child logger under the pipeline root.

    Call this at module level in every agent and utility:
        logger = get_logger(__name__)

    Args:
        name  : Typically __name__ of the calling module.
        level : Optional override. Falls back to config LOG_LEVEL.
                Accepts: "DEBUG", "INFO", "WARNING", "ERROR".

    Returns:
        A configured logging.Logger instance.
    """
    resolved_level = _resolve_level(level)
    _configure_root_logger(resolved_level)

    # Child loggers inherit the root handler and level
    logger = logging.getLogger(f"{PIPELINE_LOGGER}.{name}")
    return logger


def _resolve_level(override: Optional[str]) -> int:
    """
    Determine the log level integer.

    Priority: explicit override → config LOG_LEVEL → INFO fallback.
    Config import is deferred inside this function so logger.py
    remains importable even before .env is configured (e.g. in tests).
    """
    if override:
        return _parse_level(override)

    try:
        from core.config import get_config
        config = get_config()
        return _parse_level(config.log_level)
    except Exception:
        # Config not yet available — safe fallback during early init
        return logging.INFO


def _parse_level(level_str: str) -> int:
    """Convert a level string to a logging int. Defaults to INFO on unknown input."""
    return getattr(logging, level_str.upper(), logging.INFO)