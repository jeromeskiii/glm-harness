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


async def test_invalid_args_fail_closed(ctx, log) -> None:
    ctx.provide("sessions", log)
    registry = ToolRegistry(ctx)
    registry.register(
        Tool(
            "add",
            "add",
            {
                "type": "object",
                "required": ["a", "b"],
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            lambda args: args["a"] + args["b"],
        )
    )
    missing = await registry.execute("add", {"a": 1})
    assert missing["error"] == "INVALID_ARGS"
    assert missing["ok"] is False
    extra = await registry.execute("add", {"a": 1, "b": 2, "c": 3})
    assert extra["error"] == "INVALID_ARGS"
    wrong = await registry.execute("add", {"a": "x", "b": 2})
    assert wrong["error"] == "INVALID_ARGS"
    ok = await registry.execute("add", {"a": 1, "b": 2})
    assert ok["ok"] is True


def test_validate_tool_arguments_types() -> None:
    from glmharness.tools import validate_tool_arguments

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "n": {"type": "number"},
            "flag": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name"],
    }
    assert validate_tool_arguments(schema, {"name": "x", "n": 1.5, "flag": True, "tags": ["a"]}) is None
    assert validate_tool_arguments(schema, {}) is not None
    assert validate_tool_arguments(schema, {"name": 1}) is not None
    assert validate_tool_arguments({}, {"anything": True}) is None
    union = {"type": "object", "properties": {"id": {"type": ["string", "null"]}}}
    assert validate_tool_arguments(union, {"id": None}) is None
    assert validate_tool_arguments(union, {"id": 1}) is not None
    extras = {"type": "object", "additionalProperties": {"type": "integer"}}
    assert validate_tool_arguments(extras, {"a": 1, "b": 2}) is None
    assert validate_tool_arguments(extras, {"a": "nope"}) is not None
    assert validate_tool_arguments({"type": "array", "items": {"type": "string"}}, ["a"]) is None
    assert validate_tool_arguments({"type": "array", "items": {"type": "string"}}, [1]) is not None


async def test_pre_execute_non_dict_fails_loud(ctx) -> None:
    async def bad(_call, _next):
        return "nope"

    ctx.on("tools/pre-execute", bad)
    registry = ToolRegistry(ctx)
    registry.register(Tool("echo", "echo", {"type": "object"}, lambda a: a))
    with pytest.raises(TypeError, match="tools/pre-execute must return a dict"):
        await registry.execute("echo", {})


async def test_rewritten_non_object_arguments_are_invalid(ctx, log) -> None:
    ctx.provide("sessions", log)

    async def smash(call, next_):
        return await next_({**call, "arguments": ["not", "an", "object"]})

    ctx.on("tools/pre-execute", smash)
    registry = ToolRegistry(ctx)
    registry.register(Tool("echo", "echo", {"type": "object"}, lambda a: a))
    result = await registry.execute("echo", {"x": 1})
    assert result["error"] == "INVALID_ARGS"
