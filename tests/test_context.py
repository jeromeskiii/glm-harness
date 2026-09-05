"""Tests for Context and PluginLoader."""

from __future__ import annotations

import pytest

from glmharness import BasePlugin, Context, PluginLoader, SessionLog, ToolRegistry


async def test_provide_rejects_duplicates() -> None:
    ctx = Context()
    ctx.provide("x", 1)
    with pytest.raises(RuntimeError, match="duplicate service provider"):
        ctx.provide("x", 2)
    await ctx.close()


async def test_get_missing_service_raises_with_key() -> None:
    ctx = Context()
    with pytest.raises(RuntimeError, match=r"ctx\.nothing"):
        ctx.get("nothing")
    await ctx.close()


async def test_context_owned_disposers_run_in_lifo_order() -> None:
    ctx = Context()
    order: list[str] = []

    def add(label: str) -> None:
        ctx.effect(lambda: order.append(label))

    add("first")
    add("second")
    add("third")
    await ctx.close()
    assert order == ["third", "second", "first"]


async def test_context_on_returns_disposer_owned_by_ctx() -> None:
    ctx = Context()
    calls: list[str] = []

    async def listener(payload):
        calls.append(str(payload))

    off = ctx.on("x", listener)
    await ctx.events.dispatch("x", "emit", "before")
    await __asleep()
    off_again = off()
    await ctx.events.dispatch("x", "emit", "after")
    await __asleep()
    assert calls == ["before"]
    assert off_again is None
    await ctx.close()


async def __asleep() -> None:
    import asyncio

    await asyncio.sleep(0)


async def test_loader_mounts_in_order() -> None:
    ctx = Context()
    sessions = SessionLog()
    tools = ToolRegistry(ctx)
    loader = PluginLoader(ctx)
    await loader.mount([BasePlugin(sessions, tools)])
    assert loader.loaded == ["harness-base"]
    assert ctx.get("sessions") is sessions
    assert ctx.get("tools") is tools
    await ctx.close()


async def test_loader_fail_loud_closes_context() -> None:
    ctx = Context()
    finished: list[int] = []

    class Tidy:
        id = "tidy"

        def apply(self, current_ctx):
            current_ctx.effect(lambda: finished.append(1))

    class Broken:
        id = "broken"

        def apply(self, current_ctx):
            Tidy().apply(current_ctx)  # broken plugin also registered Tidy
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await PluginLoader(ctx).mount([Tidy(), Broken()])
    # After the boot failure ctx.close() runs every disposer LIFO. Both
    # Tidies registered their disposer; both effects must fire.
    assert finished == [1, 1]
