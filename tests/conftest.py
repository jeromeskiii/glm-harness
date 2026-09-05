"""Shared pytest fixtures.

Putting fixtures here lets every test file pick them up without rerunning
the wiring. ``bus``, ``ctx``, ``log``, and ``tools`` cover the kernel and
the most-used services; tests that need a different composition build their
own (e.g. an AgentLoop instance).
"""

from __future__ import annotations

import pytest

from glmharness import (
    Context,
    EventBus,
    SessionLog,
    Tool,
    ToolRegistry,
)


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def ctx() -> Context:
    return Context()


@pytest.fixture
def log() -> SessionLog:
    return SessionLog()


@pytest.fixture
def tools(ctx: Context) -> ToolRegistry:
    return ToolRegistry(ctx)


@pytest.fixture
def echo_tool() -> Tool:
    return Tool(
        name="echo",
        description="echo back the args",
        schema={"type": "object", "properties": {}},
        handler=lambda args: args,
    )
