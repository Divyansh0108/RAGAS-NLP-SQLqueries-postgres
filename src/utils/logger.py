"""
Centralized logging configuration using Loguru.
Provides structured logging with rotation, retention, and level control.
"""

import sys
from pathlib import Path

from loguru import logger

from src.config import get_settings

# ── Configuration ─────────────────────────────────────────────────────────────
settings = get_settings()

# Remove default handler
logger.remove()

# ── Console Handler ───────────────────────────────────────────────────────────
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.log_level,
    colorize=True,
)

# ── File Handler ──────────────────────────────────────────────────────────────
log_file_path = settings.project_root / settings.log_file
log_file_path.parent.mkdir(parents=True, exist_ok=True)

logger.add(
    log_file_path,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level=settings.log_level,
    rotation=settings.log_rotation,
    retention=settings.log_retention,
    compression="zip",
    enqueue=True,  # Thread-safe logging
)


def get_logger(name: str):
    """
    Get a logger instance for a specific module.

    Args:
        name: Module name (usually __name__)

    Returns:
        Loguru logger bound to the module name
    """
    return logger.bind(name=name)
