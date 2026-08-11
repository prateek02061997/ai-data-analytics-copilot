"""Application logging configuration."""

from __future__ import annotations

import logging
import sys

from src.config.settings import SettingsError, get_settings


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging() -> None:
    """Configure root logging once for the application."""
    try:
        level_name = get_settings().log_level
    except SettingsError:
        level_name = "INFO"

    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=False,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)