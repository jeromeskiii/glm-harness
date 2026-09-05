"""Context and plugin loader: the other two kernel primitives.

``Context`` is the typed service bag; plugins find each other by ``ctx.get``
never by import. ``PluginLoader`` mounts an ordered plugin tree and boots
fail-loud: a plugin that raises during ``apply`` closes the context (unwinding
all effects LIFO) and propagates — a half-mounted harness is not a harness.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from .bus import EventBus
from .logging import get_logger


class Context:
    def __init__(self) -> None:
        self.events = EventBus()
        self.services: dict[str, Any] = {}
        self._disposers: list[Callable[[], Any]] = []

    def provide(self, key: str, service: Any) -> None:
        if key in self.services:
            raise RuntimeError(f"duplicate service provider: {key}")
        self.services[key] = service

    def get(self, key: str) -> Any:
        try:
            return self.services[key]
        except KeyError as exc:
            raise RuntimeError(f"missing service: ctx.{key}") from exc

    def effect(self, disposer: Callable[[], Any]) -> None:
        """Register cleanup that runs (LIFO) when the context closes."""
        self._disposers.append(disposer)

    def on(self, event: str, listener: Callable[..., Any], *, prepend: bool = False) -> Callable[[], None]:
        """Register a bus listener whose teardown is owned by this context."""
        disposer = self.events.on(event, listener, prepend=prepend)
        self.effect(disposer)
        return disposer

    async def close(self) -> None:
        for disposer in reversed(self._disposers):
            result = disposer()
            if asyncio.iscoroutine(result):
                await result
        self._disposers.clear()
        get_logger().info("context closed", extra={"disposers": "ok"})


@runtime_checkable
class Plugin(Protocol):
    id: str

    def apply(self, ctx: Context) -> Any: ...


class PluginLoader:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.loaded: list[str] = []

    async def mount(self, plugins: list[Plugin]) -> None:
        """Mount plugins in order. Any boot failure closes the context."""
        try:
            for plugin in plugins:
                result = plugin.apply(self.ctx)
                if asyncio.iscoroutine(result):
                    await result
                self.loaded.append(plugin.id)
        except Exception:
            await self.ctx.close()
            raise
        get_logger().info("plugins mounted", extra={"plugins": ",".join(self.loaded)})
