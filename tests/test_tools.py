"""Tests for the tool registry: schema export, pipeline, and timeouts."""

from __future__ import annotations

import asyncio
import time

import pytest

from glmharness import Tool, ToolRegistry


async def test_schemas_follow_openai_function_shape(ctx, echo_tool) -> None:
    registry = ToolRegistry(ctx)
    registry.register(echo_tool)
    schemas = registry.schemas()
    assert schemas == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "echo back the args",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


async def test_unknown_tool_returns_unknown_error(ctx, log) -> None:
    ctx.provide("sessions", log)
    registry = ToolRegistry(ctx)
    result = await registry.execute("nope", {})
    assert result["error"] == "UNKNOWN_TOOL"
    assert any(e.type == "tool/result" for e in log.events)


async def test_policy_denied_returns_error(ctx, log) -> None:
    ctx.provide("sessions", log)
    registry = ToolRegistry(ctx)
    denied = Tool("x", "x", {"type": "object"}, lambda a: a, allowed=False)
    registry.register(denied)
    result = await registry.execute("x", {})
    assert result["error"] == "DENIED_BY_POLICY"


async def test_pre_execute_waterfall_can_deny(ctx, log) -> None:
    ctx.provide("sessions", log)

    async def deny(call, next_):
        return {"denied": True, "error": "BLOCKED"}

    ctx.on("tools/pre-execute", deny)
    registry = ToolRegistry(ctx)
    registry.register(Tool("x", "x", {"type": "object"}, lambda a: a))
    result = await registry.execute("x", {})
    assert result["error"] == "BLOCKED"


async def test_post_execute_waterfall_can_rewrite(ctx, echo_tool) -> None:
    async def rewrite(payload, next_):
        payload["result"] = {"tagged": payload["result"], "by": "test"}
        return payload

    ctx.on("tools/post-execute", rewrite)
    registry = ToolRegistry(ctx)
    registry.register(echo_tool)
    result = await registry.execute("echo", {"x": 1})
    assert "tagged" in result["content"]


async def test_handler_exception_is_recorded(ctx, log) -> None:
    ctx.provide("sessions", log)

    def boom(args):
        raise RuntimeError("kaboom")

    registry = ToolRegistry(ctx)
    registry.register(Tool("bad", "bad", {"type": "object"}, boom))
    result = await registry.execute("bad", {})
    assert result["ok"] is False
    assert result["error"] == "RuntimeError"
    assert "kaboom" in result["content"]


async def test_tool_timeout_enforced(ctx) -> None:
    async def slow(args):
        await asyncio.sleep(0.5)

    registry = ToolRegistry(ctx, tool_timeout_s=0.05)
    registry.register(Tool("slow", "slow", {"type": "object"}, slow))
    started = time.monotonic()
    result = await registry.execute("slow", {})
    elapsed = time.monotonic() - started
    assert result["error"] == "TOOL_TIMEOUT"
    assert elapsed < 0.4


async def test_async_handler_is_awaited(ctx) -> None:
    async def echo(args):
        return {"ok": True, **args}

    registry = ToolRegistry(ctx)
    registry.register(Tool("echo", "echo", {"type": "object"}, echo))
    result = await registry.execute("echo", {"a": 1})
    assert '"a": 1' in result["content"]


def test_register_rejects_duplicates(ctx) -> None:
    registry = ToolRegistry(ctx)
    registry.register(Tool("echo", "d", {"type": "object"}, lambda a: a))
    with pytest.raises(RuntimeError, match="duplicate tool"):
        registry.register(Tool("echo", "d", {"type": "object"}, lambda a: a))
