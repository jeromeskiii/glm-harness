"""Plugins shipped with the harness.

``BasePlugin`` mounts the session log and tool registry into the context —
the same composition the CLI uses. ``SafetyPlugin`` is the production
allowlist gate: when configured it denies any tool not named in the list
via ``tools/pre-execute``, so the handler never runs. Plugins are how
operators extend the harness without touching the kernel.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from .context import Context
from .session import SessionLog
from .tools import ToolRegistry


class BasePlugin:
    id = "harness-base"

    def __init__(
        self, sessions: SessionLog | None = None, tools: ToolRegistry | None = None
    ):
        self.sessions = sessions
        self.tools = tools

    def apply(self, ctx: Context) -> None:
        if "sessions" not in ctx.services:
            ctx.provide("sessions", self.sessions if self.sessions is not None else SessionLog())
        if "tools" not in ctx.services:
            ctx.provide("tools", self.tools if self.tools is not None else ToolRegistry(ctx))


class SafetyPlugin:
    """Deny tools that are not on an explicit allowlist.

    An empty allowlist is a no-op so the plugin is safe to mount always.
    """

    id = "harness-safety"

    def __init__(self, allowlist: Collection[str] = ()):
        self.allowlist = frozenset(allowlist)

    def apply(self, ctx: Context) -> None:
        if not self.allowlist:
            return
        allowed = self.allowlist

        async def gate(call: dict[str, Any], next_: Any) -> dict[str, Any]:
            name = call.get("name")
            if not isinstance(name, str) or name not in allowed:
                return {
                    **call,
                    "denied": True,
                    "error": "DENIED_BY_POLICY",
                }
            return await next_(call)

        ctx.on("tools/pre-execute", gate)
