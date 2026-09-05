"""The typed event bus: one of the three kernel primitives.

Four dispatch modes, each with a contract:

- ``emit``: fire-and-forget; coroutine listeners are scheduled, not awaited.
- ``parallel``: concurrent observers; results discarded; one listener raising
  propagates (fail-loud).
- ``serial``: ordered side effects; each listener is awaited in registration
  order; the final result is returned.
- ``waterfall``: around-middleware; each listener receives ``(payload, next)``
  and may rewrite the payload or short-circuit by not calling ``next()``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

Listener = Callable[..., Any]
#: off() handles returned by ``on``.
Off = Callable[[], None]


class EventBus:
    MODES = frozenset({"emit", "waterfall", "parallel", "serial"})

    def __init__(self) -> None:
        self._listeners: dict[str, list[tuple[int, Listener]]] = {}

    def on(self, event: str, listener: Listener, *, prepend: bool = False) -> Off:
        listeners = self._listeners.setdefault(event, [])
        item = (0 if prepend else len(listeners) + 1, listener)
        listeners.insert(0, item) if prepend else listeners.append(item)

        def off() -> None:
            if item in listeners:
                listeners.remove(item)

        return off

    def listener_count(self, event: str) -> int:
        return len(self._listeners.get(event, []))

    async def dispatch(self, event: str, mode: str, payload: Any) -> Any:
        if mode not in self.MODES:
            raise ValueError(f"unknown dispatch mode: {mode!r}")
        listeners = list(self._listeners.get(event, []))
        if mode == "emit":
            for _, listener in listeners:
                result = listener(payload)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            return None
        if mode == "parallel":
            await asyncio.gather(*(self._call(listener, payload) for _, listener in listeners))
            return None
        if mode == "serial":
            result: Any = None
            for _, listener in listeners:
                result = await self._call(listener, payload)
            return result
        return await self._run_waterfall(listeners, 0, payload)

    async def _run_waterfall(
        self, listeners: list[tuple[int, Listener]], index: int, value: Any
    ) -> Any:
        if index == len(listeners):
            return value
        called = False

        async def next_(updated: Any = value) -> Any:
            nonlocal called
            called = True
            return await self._run_waterfall(listeners, index + 1, updated)

        # Convention: a listener that continues calls ``return next_(...)``;
        # a listener that short-circuits returns without calling ``next_``.
        # Either way its return value is the final waterfall result.
        return await self._call(listeners[index][1], value, next_)

    @staticmethod
    async def _call(listener: Listener, *args: Any) -> Any:
        result = listener(*args)
        return await result if asyncio.iscoroutine(result) else result
