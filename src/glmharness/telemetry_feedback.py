"""Telemetry feedback: structured feedback events that fail silently.

Telemetry is part of the feature: every emitting call site must go through
``record_feedback`` so the event vocabulary stays closed. Emission never
throws -- a telemetry failure is silent by contract.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .logging import get_logger

FEEDBACK_KINDS: frozenset[str] = frozenset({
    "bug",
    "gap",
    "ambiguity",
    "feature_request",
    "improvement",
})

Listener = Callable[[str, dict[str, Any]], Any]
_listeners: list[Listener] = []


def on_feedback(listener: Listener) -> Callable[[], None]:
    """Register a feedback listener; returns its disposer."""
    _listeners.append(listener)

    def dispose() -> None:
        try:
            _listeners.remove(listener)
        except ValueError:
            pass

    return dispose


def record_feedback(kind: str, message: str, **detail: Any) -> None:
    """Emit a structured feedback event. Fails silently, always."""
    try:
        get_logger().info("feedback", extra={"kind": kind, "message": message, **detail})
        for listener in list(_listeners):
            listener(kind, {"message": message, **detail})
    except Exception:
        pass
