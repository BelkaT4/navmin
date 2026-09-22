"""Application logging bootstrap for NavMin."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import TextIO

_LOGGER_NAME = "navmin"
_HANDLER_NAME = "navmin-default"
_SESSION_HANDLER_NAME = "navmin-session-file"
_CONFIG_LOCK = Lock()
_FORMAT = "%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s"
SESSION_LOG_MAX_BYTES = 10 * 1024 * 1024
SESSION_LOG_BACKUP_COUNT = 5


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure one reusable stderr/stream handler for the NavMin logger tree."""
    logger = logging.getLogger(_LOGGER_NAME)

    with _CONFIG_LOCK:
        handler = next(
            (item for item in logger.handlers if item.get_name() == _HANDLER_NAME),
            None,
        )
        if handler is None:
            handler = logging.StreamHandler(stream)
            handler.set_name(_HANDLER_NAME)
            handler.setFormatter(logging.Formatter(_FORMAT))
            logger.addHandler(handler)
        elif stream is not None and isinstance(handler, logging.StreamHandler):
            handler.setStream(stream)

        logger.setLevel(level)
        handler.setLevel(level)
        logger.propagate = False

    return logger


@contextmanager
def session_file_logging(
    path: Path,
    *,
    level: int,
    max_bytes: int = SESSION_LOG_MAX_BYTES,
    backup_count: int = SESSION_LOG_BACKUP_COUNT,
) -> Iterator[Path]:
    """Attach one bounded session file while retaining the INFO console handler."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")
    if backup_count < 1:
        raise ValueError("backup_count must be >= 1")

    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(_LOGGER_NAME)
    with _CONFIG_LOCK:
        previous_level = logger.level
        previous_propagate = logger.propagate
        previous_console = next(
            (item for item in logger.handlers if item.get_name() == _HANDLER_NAME),
            None,
        )
        previous_console_level = (
            previous_console.level if previous_console is not None else None
        )

    logger = configure_logging(logging.INFO)
    handler = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.set_name(_SESSION_HANDLER_NAME)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT))

    with _CONFIG_LOCK:
        logger.setLevel(min(logging.INFO, level))
        logger.addHandler(handler)
    try:
        yield path
    finally:
        with _CONFIG_LOCK:
            logger.removeHandler(handler)
            if previous_console is None:
                current_console = next(
                    (
                        item
                        for item in logger.handlers
                        if item.get_name() == _HANDLER_NAME
                    ),
                    None,
                )
                if current_console is not None:
                    logger.removeHandler(current_console)
                    current_console.close()
            elif previous_console_level is not None:
                previous_console.setLevel(previous_console_level)
            logger.setLevel(previous_level)
            logger.propagate = previous_propagate
        handler.close()


__all__ = [
    "SESSION_LOG_BACKUP_COUNT",
    "SESSION_LOG_MAX_BYTES",
    "configure_logging",
    "session_file_logging",
]
