"""Sandbox and approval plane plugin.

Enforces execution policies (allow, deny, ask) on mutating tools
(e.g., write_file, edit_file, bash) at the ``tools/pre-execute`` waterfall.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from .context import Context
from .github import GITHUB_MUTATING_TOOLS

SandboxMode = Literal["allow", "deny", "ask"]
_MUTATING_TOOLS = frozenset({
    "write_file",
    "edit_file",
    "bash",
}) | GITHUB_MUTATING_TOOLS

ApprovalHandler = Callable[[str, dict[str, Any]], Awaitable[bool] | bool]


class SandboxPlugin:
    """Guards mutating tool executions using allow, deny, or interactive approval."""

    id = "sandbox"

    def __init__(
        self,
        mode: SandboxMode = "deny",
        approval_handler: ApprovalHandler | None = None,
        mutating_tools: set[str] | frozenset[str] | None = None,
    ):
        self.mode: SandboxMode = mode
        self.approval_handler = approval_handler
        self.mutating_tools = frozenset(mutating_tools or _MUTATING_TOOLS)

    def apply(self, ctx: Context) -> None:
        async def _gate(call: dict[str, Any], next_fn: Any) -> Any:
            tool_name = str(call.get("name", ""))
            if tool_name not in self.mutating_tools:
                return await next_fn(call)

            if self.mode == "allow":
                return await next_fn(call)

            if self.mode == "deny":
                return {**call, "denied": True, "error": "SANDBOX_DENIED"}

            if self.mode == "ask":
                approved = await self._check_approval(tool_name, call.get("arguments", {}))
                if not approved:
                    return {**call, "denied": True, "error": "ACTION_REJECTED_BY_OPERATOR"}
                return await next_fn(call)

            return await next_fn(call)

        ctx.on("tools/pre-execute", _gate)

    async def _check_approval(self, name: str, arguments: dict[str, Any]) -> bool:
        if self.approval_handler is not None:
            res = self.approval_handler(name, arguments)
            if inspect.isawaitable(res):
                return bool(await res)
            return bool(res)

        # Fallback to TTY prompt
        if not sys.stdin.isatty():
            return False

        sys.stderr.write(f"\n[sandbox-approval] Request to execute '{name}': {arguments}\nApprove? [y/N]: ")
        sys.stderr.flush()
        try:
            choice = sys.stdin.readline().strip().lower()
            return choice in ("y", "yes")
        except Exception:
            return False
