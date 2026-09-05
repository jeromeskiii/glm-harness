"""Tools: schema registration, guarded execution, and tool-call parsing.

Every call flows through ``tools/pre-execute`` (waterfall, may rewrite or
deny) -> policy check -> handler -> ``tools/post-execute`` (waterfall, may
rewrite) -> a durable ``tool/result`` fact. Every stage fails closed: an
unknown tool, a denied tool, a handler exception, or a timeout all produce
an ``ok: false`` result the model can read as a fact.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .context import Context
from .logging import get_logger

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_ARGS_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse model-generated ``<tool_call>name<arg_key>…`` blocks.

    Values are JSON-decoded when possible (numbers, objects, lists, bools,
    strings); anything undecodable stays a string. Returns a list of
    ``{"name": str, "arguments": dict}``.
    """
    calls: list[dict[str, Any]] = []
    for block in _TOOL_CALL_RE.findall(text):
        block = block.strip()
        if not block:
            continue
        if "<arg_key>" in block:
            name, rest = block.split("<arg_key>", 1)
            name = name.strip()
            rest = "<arg_key>" + rest
        else:
            name, rest = block.strip(), ""
        arguments: dict[str, Any] = {}
        for key_match, value_match in _ARGS_RE.findall(rest):
            key = key_match.strip()
            raw_value = value_match.strip()
            try:
                value = json.loads(raw_value)
            except json.JSONDecodeError:
                value = raw_value
            arguments[key] = value
        if name:
            calls.append({"name": name, "arguments": arguments})
    return calls


@dataclass
class Tool:
    """A tool the model may call. ``allowed`` is the static policy gate."""

    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[Any] | Any]
    allowed: bool = True


class ToolRegistry:
    def __init__(self, ctx: Context, tool_timeout_s: float = 30.0):
        self.ctx = ctx
        self.tools: dict[str, Tool] = {}
        self.tool_timeout_s = tool_timeout_s

    def register(self, tool: Tool) -> Callable[[], None]:
        if tool.name in self.tools:
            raise RuntimeError(f"duplicate tool: {tool.name}")
        self.tools[tool.name] = tool

        def unregister() -> None:
            self.tools.pop(tool.name, None)

        return unregister

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema,
                },
            }
            for tool in self.tools.values()
        ]

    def _finish(self, call: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
        sessions = self.ctx.services.get("sessions")
        if sessions is not None:
            sessions.append(
                "tool/result",
                {
                    "call_id": call["id"],
                    "name": call["name"],
                    "ok": response.get("ok", False),
                    "content": response.get("content", response.get("error", "")),
                },
            )
        return response

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        call: dict[str, Any] = {
            "name": name,
            "arguments": copy.deepcopy(arguments),
            "id": str(uuid.uuid4()),
        }
        dispatched = await self.ctx.events.dispatch("tools/pre-execute", "waterfall", call)
        # A waterfall listener may short-circuit by returning a partial
        # object (``{"denied": True, "error": "..."}``). Downstream finishers
        # need ``id`` and ``name`` to record the ``tool/result`` fact, so we
        # merge the dispatcher's output on top of the original call.
        if not isinstance(dispatched, dict):
            raise TypeError(f"tools/pre-execute must return a dict, got {type(dispatched)}")
        call = {**call, **dispatched}
        if call.get("denied"):
            return self._finish(call, {"ok": False, "error": call.get("error", "DENIED_BY_POLICY")})
        tool = self.tools.get(call["name"])
        if tool is None:
            return self._finish(call, {"ok": False, "error": "UNKNOWN_TOOL"})
        if not tool.allowed:
            return self._finish(call, {"ok": False, "error": "DENIED_BY_POLICY"})
        started = asyncio.get_running_loop().time()
        try:
            if self.tool_timeout_s > 0:
                result = await asyncio.wait_for(
                    self._invoke(tool, call["arguments"]), timeout=self.tool_timeout_s
                )
            else:
                result = await self._invoke(tool, call["arguments"])
        except TimeoutError:
            get_logger().warning(
                "tool timed out",
                extra={"tool": name, "timeout_s": self.tool_timeout_s},
            )
            return self._finish(call, {"ok": False, "error": "TOOL_TIMEOUT"})
        except Exception as exc:
            get_logger().warning(
                "tool failed",
                extra={"tool": name, "error": type(exc).__name__},
            )
            return self._finish(
                call, {"ok": False, "error": type(exc).__name__, "content": str(exc)}
            )
        post = await self.ctx.events.dispatch(
            "tools/post-execute", "waterfall", {"call": call, "result": result}
        )
        elapsed = asyncio.get_running_loop().time() - started
        get_logger().info("tool executed", extra={"tool": name, "elapsed_s": round(elapsed, 3)})
        return self._finish(call, {"ok": True, "content": json.dumps(post["result"], ensure_ascii=False)})

    @staticmethod
    async def _invoke(tool: Tool, arguments: dict[str, Any]) -> Any:
        result = tool.handler(copy.deepcopy(arguments))
        return await result if asyncio.iscoroutine(result) else result
