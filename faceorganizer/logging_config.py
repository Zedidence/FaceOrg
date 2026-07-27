"""Centralized logging configuration for FaceOrganizer."""

import logging
import sys
from pathlib import Path

_file_handler: logging.FileHandler | None = None


def setup_logging(verbosity: int = 0, log_dir: Path | None = None) -> None:
    """Configure logging for the application.

    Args:
        verbosity: Console verbosity. 0 = WARNING, 1 = INFO, 2+ = DEBUG.
        log_dir: Directory for the log file. If provided, a file logger at
                 DEBUG level is always created at ``<log_dir>/faceorganizer.log``.
    """
    global _file_handler

    console_level = logging.WARNING
    if verbosity == 1:
        console_level = logging.INFO
    elif verbosity >= 2:
        console_level = logging.DEBUG

    root = logging.getLogger("faceorganizer")
    root.setLevel(logging.DEBUG)  # allow all levels; handlers filter
    root.handlers.clear()

    # Console handler — respects verbosity flag
    console_fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(console_fmt)
    root.addHandler(console)

    # File handler — always DEBUG so the full trace is available later
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "faceorganizer.log"

        file_fmt = logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        _file_handler.setLevel(logging.DEBUG)
        _file_handler.setFormatter(file_fmt)
        root.addHandler(_file_handler)

        root.debug("Log file: %s", log_path)

    # Quiet noisy third-party loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger scoped under 'faceorganizer'."""
    return logging.getLogger(f"faceorganizer.{name}")
