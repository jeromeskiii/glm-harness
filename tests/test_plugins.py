"""Tests for shipped plugins: base composition and the safety allowlist."""

from __future__ import annotations

from glmharness import Context, SafetyPlugin, Tool, ToolRegistry
from glmharness.plugins import BasePlugin
from glmharness.session import SessionLog


async def test_safety_plugin_denies_tools_not_on_allowlist(ctx: Context) -> None:
    log = SessionLog()
    ctx.provide("sessions", log)
    SafetyPlugin(("echo",)).apply(ctx)
    registry = ToolRegistry(ctx)
    registry.register(Tool("echo", "echo", {"type": "object"}, lambda a: a))
    registry.register(Tool("secret", "secret", {"type": "object"}, lambda a: a))
    denied = await registry.execute("secret", {})
    assert denied["ok"] is False
    assert denied["error"] == "DENIED_BY_POLICY"
    allowed = await registry.execute("echo", {"x": 1})
    assert allowed["ok"] is True


async def test_safety_plugin_empty_allowlist_is_noop(ctx: Context) -> None:
    log = SessionLog()
    ctx.provide("sessions", log)
    SafetyPlugin(()).apply(ctx)
    registry = ToolRegistry(ctx)
    registry.register(Tool("echo", "echo", {"type": "object"}, lambda a: a))
    result = await registry.execute("echo", {})
    assert result["ok"] is True


def test_base_plugin_provides_sessions_and_tools(ctx: Context) -> None:
    BasePlugin().apply(ctx)
    assert "sessions" in ctx.services
    assert "tools" in ctx.services
