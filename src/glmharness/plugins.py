"""Plugins shipped with the harness.

``BasePlugin`` mounts the session log and tool registry into the context —
the same composition the CLI uses. Plugins are how operators extend the
harness without touching the kernel.
"""

from __future__ import annotations

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
