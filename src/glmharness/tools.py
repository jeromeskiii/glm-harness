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
from collections.abc import Callable
from typing import Any, cast

from .artifacts.flow_constants import (
    POST_EXECUTE,
    PRE_EXECUTE,
)
from .artifacts.flow_types import Tool
from .context import Context
from .logging import get_logger

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_ARGS_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
)


def validate_tool_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    """Return a human error or ``None`` if ``arguments`` satisfy ``schema``.

    JSON Schema subset: ``type``, ``required``, ``properties``,
    ``additionalProperties``, ``items``. Unknown keywords are ignored so a
    richer schema still fails closed on the fields we understand.
    """
    if not schema:
        return None
    return _check_value(schema, arguments, path="$")


def _as_schema(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return cast(dict[str, Any], value)


def _check_value(schema: dict[str, Any], value: Any, path: str) -> str | None:
    expected = schema.get("type")
    if expected is not None:
        if not _type_matches(expected, value):
            return f"{path}: expected {expected}, got {type(value).__name__}"
    if expected == "object" or (expected is None and isinstance(value, dict) and "properties" in schema):
        if not isinstance(value, dict):
            return f"{path}: expected object"
        mapping = cast(dict[str, Any], value)
        required_raw = schema.get("required")
        if isinstance(required_raw, list):
            for key in cast(list[object], required_raw):
                if key not in mapping:
                    return f"{path}: missing required property {key!r}"
        properties = _as_schema(schema.get("properties") or {})
        if properties is not None:
            for key, subschema in properties.items():
                nested = _as_schema(subschema)
                if key in mapping and nested is not None:
                    err = _check_value(nested, mapping[key], f"{path}.{key}")
                    if err is not None:
                        return err
        additional = schema.get("additionalProperties", True)
        if additional is False and properties is not None:
            extra = [str(key) for key in mapping if key not in properties]
            if extra:
                return f"{path}: unexpected properties {extra}"
        additional_schema = _as_schema(additional)
        if additional_schema is not None:
            for key, item in mapping.items():
                if properties is not None and key in properties:
                    continue
                err = _check_value(additional_schema, item, f"{path}.{key}")
                if err is not None:
                    return err
    if expected == "array" and isinstance(value, list):
        items = _as_schema(schema.get("items"))
        if items is not None:
            sequence = cast(list[Any], value)
            for index, item in enumerate(sequence):
                err = _check_value(items, item, f"{path}[{index}]")
                if err is not None:
                    return err
    return None


def _type_matches(expected: object, value: Any) -> bool:
    types: list[object] = (
        cast(list[object], expected) if isinstance(expected, list) else [expected]
    )
    for item in types:
        if item == "object" and isinstance(value, dict):
            return True
        if item == "array" and isinstance(value, list):
            return True
        if item == "string" and isinstance(value, str):
            return True
        if item == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if item == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if item == "boolean" and isinstance(value, bool):
            return True
        if item == "null" and value is None:
            return True
    return False


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

    async def _dispatch_with_timeout(self, event: str, mode: str, payload: Any) -> Any:
        """Dispatch a bus event with a hard wall-clock bound.

        A stuck listener (e.g. a custom plugin, a runaway regex, or a slow
        downstream dependency) must not be able to wedge the tool loop. The
        bound is the configured ``tool_timeout_s``; 0 means no bound (use
        only when a peer has explicitly opted in).
        """
        coro = self.ctx.events.dispatch(event, mode, payload)
        if self.tool_timeout_s <= 0:
            return await coro
        return await asyncio.wait_for(coro, timeout=self.tool_timeout_s)

    async def execute(
        self, name: str, arguments: dict[str, Any], *, call_id: str | None = None
    ) -> dict[str, Any]:
        call: dict[str, Any] = {
            "name": name,
            "arguments": copy.deepcopy(arguments),
            "id": call_id or str(uuid.uuid4()),
        }
        try:
            dispatched = await self._dispatch_with_timeout(
                PRE_EXECUTE, "waterfall", call
            )
        except TimeoutError:
            get_logger().warning(
                "tool pre-execute timed out",
                extra={"tool": name, "timeout_s": self.tool_timeout_s},
            )
            return self._finish(call, {"ok": False, "error": "PRE_EXECUTE_TIMEOUT"})
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
        raw_arguments = call.get("arguments", {})
        if not isinstance(raw_arguments, dict):
            return self._finish(
                call,
                {"ok": False, "error": "INVALID_ARGS", "content": "arguments must be an object"},
            )
        call_args = cast(dict[str, Any], raw_arguments)
        schema_error = validate_tool_arguments(tool.schema, call_args)
        if schema_error is not None:
            return self._finish(
                call, {"ok": False, "error": "INVALID_ARGS", "content": schema_error}
            )
        started = asyncio.get_running_loop().time()
        try:
            if self.tool_timeout_s > 0:
                result = await asyncio.wait_for(
                    self._invoke(tool, call_args), timeout=self.tool_timeout_s
                )
            else:
                result = await self._invoke(tool, call_args)
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
        try:
            post = await self._dispatch_with_timeout(
                POST_EXECUTE,
                "waterfall",
                {"call": call, "result": result},
            )
            if not isinstance(post, dict) or "result" not in post:
                raise TypeError("tools/post-execute must return a dict containing result")
            serialized = json.dumps(post["result"], ensure_ascii=False)
        except TimeoutError:
            get_logger().warning(
                "tool post-execute timed out",
                extra={"tool": name, "timeout_s": self.tool_timeout_s},
            )
            return self._finish(call, {"ok": False, "error": "POST_EXECUTE_TIMEOUT"})
        except Exception as exc:
            get_logger().warning(
                "tool result serialization failed",
                extra={"tool": name, "error": type(exc).__name__},
            )
            return self._finish(call, {"ok": False, "error": type(exc).__name__, "content": str(exc)})
        elapsed = asyncio.get_running_loop().time() - started
        get_logger().info("tool executed", extra={"tool": name, "elapsed_s": round(elapsed, 3)})
        return self._finish(call, {"ok": True, "content": serialized})

    @staticmethod
    async def _invoke(tool: Tool, arguments: dict[str, Any]) -> Any:
        result = tool.handler(copy.deepcopy(arguments))
        return await result if asyncio.iscoroutine(result) else result
