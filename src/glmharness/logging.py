"""Structured logging for the harness.

All diagnostic output goes to **stderr**; stdout is reserved for the final
answer so the harness composes with pipes. Two formats:

- ``text``: human-readable ``LEVEL name: message`` lines.
- ``json``: one JSON object per record with ``ts``, ``level``, ``logger``,
  ``message`` and any structured extras — ship it to any log aggregator.

The ``key=value``-style extras pattern lets call sites attach structured
fields without building dicts by hand.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

LOGGER_NAME = "glmharness"

_FORMATS = ("text", "json")


class JsonFormatter(logging.Formatter):
    """One JSON object per record; extras are flattened into the object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "message",
                "module",
                "msecs",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "taskName",
                "thread",
                "threadName",
            }:
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                value = repr(value)
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=repr)


def configure_logging(fmt: str = "text", level: str = "INFO") -> logging.Logger:
    """Configure the harness logger. Returns the shared logger instance."""
    if fmt not in _FORMATS:
        raise ValueError(f"unknown log format: {fmt!r} (expected one of {_FORMATS})")
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False
    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
    logger.addHandler(handler)
    return logger


def get_logger() -> logging.Logger:
    """The shared harness logger (configures defaults on first use)."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        configure_logging()
    return logger


def elapsed() -> float:
    """Wall-clock seconds since process start (monotonic); for span logs."""
    return time.monotonic()
