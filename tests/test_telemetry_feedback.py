"""Tests for the telemetry feedback module and its wiring."""

from __future__ import annotations

import asyncio

import pytest

from glmharness import AgentLoop, SessionLog, ToolRegistry
from glmharness.telemetry_feedback import (
    FEEDBACK_KINDS,
    on_feedback,
    record_feedback,
)


def test_record_feedback_dispatches_to_listeners() -> None:
    received: list[tuple[str, dict]] = []
    dispose = on_feedback(lambda kind, data: received.append((kind, data)))
    try:
        record_feedback("bug", "something broke", turn_status="failed")
    finally:
        dispose()
    assert received == [("bug", {"message": "something broke", "turn_status": "failed"})]


def test_dispose_removes_listener() -> None:
    received: list[tuple[str, dict]] = []
    dispose = on_feedback(lambda kind, data: received.append((kind, data)))
    dispose()
    record_feedback("gap", "after dispose")
    assert received == []


def test_dispose_is_idempotent() -> None:
    dispose = on_feedback(lambda kind, data: None)
    dispose()
    dispose()  # must not raise


def test_record_feedback_never_raises() -> None:
    def exploding_listener(kind: str, data: dict) -> None:
        raise RuntimeError("listener boom")

    dispose = on_feedback(exploding_listener)
    try:
        record_feedback("bug", "listener exploded")
    finally:
        dispose()


def test_feedback_kinds_vocabulary_is_closed() -> None:
    assert FEEDBACK_KINDS == {
        "bug",
        "gap",
        "ambiguity",
        "feature_request",
        "improvement",
    }


def test_loop_failure_records_bug_feedback(ctx) -> None:
    """Turn failure routes a structured bug event through record_feedback."""
    received: list[tuple[str, dict]] = []
    dispose = on_feedback(lambda kind, data: received.append((kind, data)))
    try:
        sessions = SessionLog()
        registry = ToolRegistry(ctx)

        class ExplodingLLM:
            async def stream(self, messages, tools=None):  # type: ignore[no-untyped-def]
                raise RuntimeError("provider exploded")
                yield ""  # pragma: no cover

        loop = AgentLoop(ctx, ExplodingLLM(), sessions, registry)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="provider exploded"):
            asyncio.run(loop.run("hello"))
    finally:
        dispose()
    assert received and received[-1][0] == "bug"
    assert "RuntimeError" in received[-1][1]["message"]
