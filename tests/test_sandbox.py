"""Unit tests for SandboxPlugin execution policies and approval gates."""

from __future__ import annotations

import sys
from typing import Any

from glmharness import Context, SessionLog, Tool, ToolRegistry
from glmharness.sandbox import SandboxPlugin


async def test_sandbox_allow_mode(ctx: Context, log: SessionLog) -> None:
    ctx.provide("sessions", log)
    registry = ToolRegistry(ctx)
    registry.register(Tool("write_file", "write", {"type": "object"}, lambda a: "ok"))
    registry.register(Tool("read_file", "read", {"type": "object"}, lambda a: "content"))

    SandboxPlugin(mode="allow").apply(ctx)

    res_write = await registry.execute("write_file", {"path": "foo"})
    assert res_write["ok"] is True
    res_read = await registry.execute("read_file", {"path": "foo"})
    assert res_read["ok"] is True


async def test_sandbox_deny_mode(ctx: Context, log: SessionLog) -> None:
    ctx.provide("sessions", log)
    registry = ToolRegistry(ctx)
    registry.register(Tool("write_file", "write", {"type": "object"}, lambda a: "ok"))
    registry.register(Tool("read_file", "read", {"type": "object"}, lambda a: "content"))

    SandboxPlugin(mode="deny").apply(ctx)

    # Mutating tool denied
    res_write = await registry.execute("write_file", {"path": "foo"})
    assert res_write["ok"] is False
    assert res_write["error"] == "SANDBOX_DENIED"

    # Non-mutating tool allowed
    res_read = await registry.execute("read_file", {"path": "foo"})
    assert res_read["ok"] is True


async def test_sandbox_ask_mode_with_handler(ctx: Context, log: SessionLog) -> None:
    ctx.provide("sessions", log)
    registry = ToolRegistry(ctx)
    registry.register(Tool("bash", "bash", {"type": "object"}, lambda a: "cmd_run"))

    allowed_calls: list[str] = []

    def custom_approver(name: str, args: dict[str, Any]) -> bool:
        allowed_calls.append(name)
        return args.get("command") == "allowed_cmd"

    SandboxPlugin(mode="ask", approval_handler=custom_approver).apply(ctx)

    # Rejection
    res_rejected = await registry.execute("bash", {"command": "evil_cmd"})
    assert res_rejected["ok"] is False
    assert res_rejected["error"] == "ACTION_REJECTED_BY_OPERATOR"

    # Approval
    res_approved = await registry.execute("bash", {"command": "allowed_cmd"})
    assert res_approved["ok"] is True
    assert "bash" in allowed_calls


async def test_sandbox_ask_mode_non_interactive_denies(
    ctx: Context, log: SessionLog, monkeypatch: Any
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    ctx.provide("sessions", log)
    registry = ToolRegistry(ctx)
    registry.register(Tool("bash", "bash", {"type": "object"}, lambda a: "cmd_run"))

    SandboxPlugin(mode="ask").apply(ctx)

    res = await registry.execute("bash", {"command": "ls"})
    assert res["ok"] is False
    assert res["error"] == "ACTION_REJECTED_BY_OPERATOR"


def test_default_mutating_set_covers_github_tools() -> None:
    """Drift guard: every GitHub mutating tool must be sandbox-gated by default."""
    from glmharness.github import GITHUB_MUTATING_TOOLS

    assert GITHUB_MUTATING_TOOLS <= SandboxPlugin().mutating_tools
