"""Application logging bootstrap for NavMin."""

from __future__ import annotations

import logging
from threading import Lock
from typing import TextIO

_LOGGER_NAME = "navmin"
_HANDLER_NAME = "navmin-default"
_CONFIG_LOCK = Lock()
_FORMAT = "%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s"


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
