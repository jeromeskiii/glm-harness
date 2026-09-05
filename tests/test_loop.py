"""Tests for the agent loop, including retry and timeout behavior."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from glmharness import (
    AgentLoop,
    Context,
    MockLLM,
    ProviderError,
    SessionLog,
    Tool,
    ToolRegistry,
)


class FaultyLLM:
    """Yields ``response`` for the first ``fail_after`` streams, then raises.

    The raise happens **before** any yield — providers fail their stream
    before streaming any tokens when the cause is transport-side.
    """

    def __init__(self, response: str, exc_class: type[Exception], *, fail_after: int):
        self.response = response
        self.exc_class = exc_class
        self.successes_remaining = fail_after
        self.calls = 0

    async def stream(self, messages, tools=None) -> AsyncIterator[str]:
        self.calls += 1
        if self.successes_remaining <= 0:
            raise self.exc_class("transient")
        self.successes_remaining -= 1
        yield self.response


class _OnceLLM:
    """First stream raises ConnectionError; subsequent streams yield ``response``."""

    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    async def stream(self, messages, tools=None) -> AsyncIterator[str]:
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("transient")
        yield self.response


def test_loop_runs_to_completion_on_happy_path() -> None:
    ctx = Context()
    log = SessionLog()
    tools = ToolRegistry(ctx)
    ctx.provide("sessions", log)
    ctx.provide("tools", tools)
    answer = asyncio.run(AgentLoop(ctx, MockLLM("ok"), log, tools).run("ping"))
    assert answer == "ok"
    types = [e.type for e in log.events]
    assert types[0] == "turn/start"
    assert types[-1] == "turn/end"
    assert types.count("request/header") == 1
    assert types.count("request/context") == 1
    assert log.derive_messages()[-1] == {"role": "assistant", "content": "ok"}


def test_loop_records_reconstruction_epoch_only_first_round() -> None:
    ctx = Context()
    log = SessionLog()
    tools = ToolRegistry(ctx)
    ctx.provide("sessions", log)
    ctx.provide("tools", tools)

    class TwoAnswers:
        def __init__(self):
            self.calls = 0

        async def stream(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                yield "no tools here"
            else:
                yield "second"

    asyncio.run(AgentLoop(ctx, TwoAnswers(), log, tools, max_rounds=3).run("x"))
    assert [e.type for e in log.events].count("request/header") == 1
    assert [e.type for e in log.events].count("request/context") == 1


def test_loop_executes_tool_calls_until_stop_answer() -> None:
    ctx = Context()
    log = SessionLog()
    tools = ToolRegistry(ctx)
    ctx.provide("sessions", log)
    ctx.provide("tools", tools)

    seen_calls: list[str] = []

    def echo(args):
        seen_calls.append(args["q"])
        return args

    tools.register(Tool("echo", "echo", {"type": "object"}, echo))

    class Driver:
        def __init__(self):
            self.calls = 0

        async def stream(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                yield '<tool_call>echo<arg_key>q</arg_key><arg_value>"a"</arg_value></tool_call>'
            else:
                yield "done"

    answer = asyncio.run(AgentLoop(ctx, Driver(), log, tools).run("start"))
    assert answer == "done"
    assert seen_calls == ["a"]
    assert [e.type for e in log.events].count("tool/result") == 1


def test_loop_retries_transient_provider_then_succeeds() -> None:
    ctx = Context()
    log = SessionLog()
    tools = ToolRegistry(ctx)
    ctx.provide("sessions", log)
    ctx.provide("tools", tools)
    # First stream raises ConnectionError; subsequent streams yield "hello".
    llm = _OnceLLM("hello")
    config = AgentLoop(
        ctx,
        llm,
        log,
        tools,
        max_retries=3,
        retry_base_delay_s=0.01,
        retry_max_delay_s=0.02,
        retry_jitter=0,
    )
    answer = asyncio.run(config.run("x"))
    assert answer == "hello"
    assert llm.calls == 2
    assert [e.type for e in log.events].count("stream/retry") == 1
    # The recovered response is committed to the append-only log; the discarded
    # attempt's tokens are not (retried attempts never leak into history).
    assert any(e.type == "assistant/chunk" for e in log.events)
    assert log.derive_messages()[-1] == {"role": "assistant", "content": "hello"}


def test_loop_does_not_retry_non_retryable_provider_error() -> None:
    ctx = Context()
    log = SessionLog()
    tools = ToolRegistry(ctx)
    ctx.provide("sessions", log)
    ctx.provide("tools", tools)

    class BadProvider:
        async def stream(self, messages, tools=None):
            raise ProviderError("provider config broken", retryable=False)
            yield ""  # pragma: no cover

    with pytest.raises(ProviderError):
        asyncio.run(AgentLoop(ctx, BadProvider(), log, tools, max_retries=5).run("x"))
    assert not any(e.type == "stream/retry" for e in log.events)


def test_loop_gives_up_after_max_retries() -> None:
    ctx = Context()
    log = SessionLog()
    tools = ToolRegistry(ctx)
    ctx.provide("sessions", log)
    ctx.provide("tools", tools)
    llm = FaultyLLM("never", OSError, fail_after=0)
    with pytest.raises(OSError):
        asyncio.run(AgentLoop(ctx, llm, log, tools, max_retries=2).run("x"))
    types = [e.type for e in log.events]
    assert types.count("stream/retry") == 2
    assert types[-1] == "turn/end"
    assert log.events[-1].data["status"] == "failed"


def test_loop_enforces_request_timeout() -> None:
    ctx = Context()
    log = SessionLog()
    tools = ToolRegistry(ctx)
    ctx.provide("sessions", log)
    ctx.provide("tools", tools)

    class SlowLLM:
        async def stream(self, messages, tools=None):
            await asyncio.sleep(0.5)
            yield "late"

    config = AgentLoop(ctx, SlowLLM(), log, tools, request_timeout_s=0.05, max_retries=0)
    with pytest.raises(TimeoutError):
        asyncio.run(config.run("x"))


def test_loop_cancellation_closes_turn_cancelled() -> None:
    ctx = Context()
    log = SessionLog()
    tools = ToolRegistry(ctx)
    ctx.provide("sessions", log)
    ctx.provide("tools", tools)

    class Slow:
        async def stream(self, messages, tools=None):
            await asyncio.sleep(1)
            yield ""

    async def run_and_cancel():
        agent = AgentLoop(ctx, Slow(), log, tools)
        task = asyncio.create_task(agent.run("x"))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_and_cancel())
    assert log.events[-1].type == "turn/end"
    assert log.events[-1].data["status"] == "cancelled"


def test_loop_max_rounds_reached_records_status() -> None:
    ctx = Context()
    log = SessionLog()
    tools = ToolRegistry(ctx)
    ctx.provide("sessions", log)
    ctx.provide("tools", tools)

    class AlwaysTooling:
        def __init__(self):
            self.calls = 0

        async def stream(self, messages, tools=None):
            self.calls += 1
            yield '<tool_call>echo<arg_key>q</arg_key><arg_value>"x"</arg_value></tool_call>'

    tools.register(Tool("echo", "echo", {"type": "object"}, lambda args: args))
    asyncio.run(AgentLoop(ctx, AlwaysTooling(), log, tools, max_rounds=3).run("x"))
    last_step = next(e for e in reversed(log.events) if e.type == "step/end")
    assert last_step.data["status"] == "max_rounds_reached"
