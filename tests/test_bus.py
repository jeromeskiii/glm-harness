"""Tests for the event bus dispatch modes."""

from __future__ import annotations

import asyncio

import pytest

from glmharness import EventBus


async def test_emit_dispatches_and_returns_none(bus: EventBus) -> None:
    seen: list[str] = []
    bus.on("ev", lambda payload: seen.append(payload["v"]))
    assert await bus.dispatch("ev", "emit", {"v": "a"}) is None
    await asyncio.sleep(0)  # let scheduled coroutines run
    assert seen == ["a"]


async def test_emit_schedules_async_listeners(bus: EventBus) -> None:
    seen: list[str] = []

    async def listener(payload):
        await asyncio.sleep(0)
        seen.append(payload["v"])

    bus.on("ev", listener)
    await bus.dispatch("ev", "emit", {"v": "scheduled"})
    await asyncio.sleep(0.01)
    assert seen == ["scheduled"]


async def test_parallel_runs_concurrently(bus: EventBus) -> None:
    started: list[int] = []
    finished: list[int] = []

    async def slow_one(payload):
        started.append(1)
        await asyncio.sleep(0.01)
        finished.append(1)

    async def fast_one(payload):
        started.append(2)
        finished.append(2)

    bus.on("ev", slow_one)
    bus.on("ev", fast_one)
    await bus.dispatch("ev", "parallel", {})
    # Both started; both finished; parallel returned None.
    assert sorted(started) == [1, 2]
    assert sorted(finished) == [1, 2]


async def test_serial_returns_last_result(bus: EventBus) -> None:
    bus.on("ev", lambda payload: 1)
    bus.on("ev", lambda payload: 2)
    bus.on("ev", lambda payload: 3)
    assert await bus.dispatch("ev", "serial", {}) == 3


async def test_waterfall_rewrites_payload(bus: EventBus) -> None:
    async def step(payload, next_):
        return await next_({**payload, "a": True})

    async def step2(payload, next_):
        return await next_({**payload, "b": True})

    bus.on("ev", step)
    bus.on("ev", step2)
    result = await bus.dispatch("ev", "waterfall", {"seed": True})
    assert result == {"seed": True, "a": True, "b": True}


async def test_waterfall_short_circuit_skips_remaining(bus: EventBus) -> None:
    def deny(payload, next_):
        return {"denied": True, "reason": "policy"}

    def would_run(payload, next_):
        raise AssertionError("downstream must not run when short-circuited")

    bus.on("ev", deny)
    bus.on("ev", would_run)
    assert await bus.dispatch("ev", "waterfall", {}) == {"denied": True, "reason": "policy"}


async def test_off_unsubscribes(bus: EventBus) -> None:
    seen: list[int] = []
    off = bus.on("ev", lambda payload: seen.append(payload))
    off()
    await bus.dispatch("ev", "emit", 1)
    await asyncio.sleep(0)
    assert seen == []


async def test_unknown_mode_raises(bus: EventBus) -> None:
    with pytest.raises(ValueError, match="unknown dispatch mode"):
        await bus.dispatch("ev", "banana", {})
