"""Standard built-in tools for filesystem and subprocess execution.

Provides ``read_file``, ``write_file``, ``edit_file``, ``list_dir``, and ``bash``
guarded by workspace boundary checks.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .context import Context
from .tools import Tool, ToolRegistry

_MAX_OUTPUT_BYTES = 50_000


def resolve_safe_path(base: Path, relative_or_absolute: str | Path) -> Path:
    """Resolve a path and ensure it does not escape ``base``.

    Raises:
        PermissionError: if the resolved path is outside ``base``.
    """
    base_resolved = base.resolve()
    target_path = Path(relative_or_absolute)
    if target_path.is_absolute():
        resolved = target_path.resolve()
    else:
        resolved = (base_resolved / target_path).resolve()

    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise PermissionError(f"access denied: '{relative_or_absolute}' escapes workspace boundary")
    return resolved


def make_read_file_tool(workspace: Path) -> Tool:
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative or absolute file path within workspace"},
            "offset": {"type": "integer", "description": "Starting line number (0-indexed, default 0)"},
            "limit": {"type": "integer", "description": "Maximum number of lines to return (default 2000)"},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        target = resolve_safe_path(workspace, str(args["path"]))
        if not target.is_file():
            raise FileNotFoundError(f"file not found: {args['path']}")
        offset = int(args.get("offset", 0))
        limit = int(args.get("limit", 2000))
        content = target.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)
        sliced = lines[offset : offset + limit]
        truncated = (offset + len(sliced)) < total_lines
        return {
            "path": str(target.relative_to(workspace.resolve())),
            "content": "".join(sliced),
            "lines": len(sliced),
            "total_lines": total_lines,
            "truncated": truncated,
        }

    return Tool(
        name="read_file",
        description="Read lines from a file within the workspace boundary.",
        schema=schema,
        handler=handler,
    )


def make_write_file_tool(workspace: Path) -> Tool:
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path within workspace"},
            "content": {"type": "string", "description": "Full file content to write"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        target = resolve_safe_path(workspace, str(args["path"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        content = str(args["content"])
        target.write_text(content, encoding="utf-8")
        return {
            "path": str(target.relative_to(workspace.resolve())),
            "bytes_written": len(content.encode("utf-8")),
        }

    return Tool(
        name="write_file",
        description="Create or overwrite a file within the workspace boundary.",
        schema=schema,
        handler=handler,
    )


def make_edit_file_tool(workspace: Path) -> Tool:
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path within workspace"},
            "old_string": {"type": "string", "description": "Exact text to find and replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
        },
        "required": ["path", "old_string", "new_string"],
        "additionalProperties": False,
    }

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        target = resolve_safe_path(workspace, str(args["path"]))
        if not target.is_file():
            raise FileNotFoundError(f"file not found: {args['path']}")
        content = target.read_text(encoding="utf-8")
        old_string = str(args["old_string"])
        new_string = str(args["new_string"])
        count = content.count(old_string)
        if count == 0:
            raise ValueError(f"target text not found in {args['path']}")
        if count > 1:
            raise ValueError(f"target text matches {count} times in {args['path']}; must be unique")
        updated = content.replace(old_string, new_string, 1)
        target.write_text(updated, encoding="utf-8")
        return {
            "path": str(target.relative_to(workspace.resolve())),
            "replacements": 1,
        }

    return Tool(
        name="edit_file",
        description="Replace a unique string occurrence in a file within the workspace.",
        schema=schema,
        handler=handler,
    )


def make_list_dir_tool(workspace: Path) -> Tool:
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path within workspace (default: '.')"},
        },
        "additionalProperties": False,
    }

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        target_str = str(args.get("path", "."))
        target = resolve_safe_path(workspace, target_str)
        if not target.is_dir():
            raise NotADirectoryError(f"not a directory: {target_str}")
        entries: list[dict[str, Any]] = []
        for entry in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            entries.append(
                {
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if entry.is_file() else None,
                }
            )
        rel = "." if target == workspace.resolve() else str(target.relative_to(workspace.resolve()))
        return {
            "path": rel,
            "entries": entries,
        }

    return Tool(
        name="list_dir",
        description="List contents of a directory within the workspace boundary.",
        schema=schema,
        handler=handler,
    )


def make_bash_tool(workspace: Path) -> Tool:
    schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout_s": {"type": "number", "description": "Execution timeout in seconds (default: 30)"},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    cwd_str = str(workspace.resolve())

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        command = str(args["command"])
        timeout_s = float(args.get("timeout_s", 30.0))
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd_str,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s if timeout_s > 0 else None
            )
        except TimeoutError:
            timed_out = True
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            stdout_bytes, stderr_bytes = b"", b"execution timed out"

        stdout = stdout_bytes.decode("utf-8", errors="replace")[:_MAX_OUTPUT_BYTES]
        stderr = stderr_bytes.decode("utf-8", errors="replace")[:_MAX_OUTPUT_BYTES]
        return {
            "exit_code": proc.returncode if not timed_out else -1,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": timed_out,
        }

    return Tool(
        name="bash",
        description="Execute a shell command inside the workspace directory.",
        schema=schema,
        handler=handler,
    )


class BuiltinToolsPlugin:
    """Mounts the standard tool battery (read, write, edit, list, bash) into ToolRegistry."""

    id = "builtin-tools"

    def __init__(self, workspace: Path | None = None, enable_bash: bool = True):
        self.workspace = (workspace or Path.cwd()).resolve()
        self.enable_bash = enable_bash

    def apply(self, ctx: Context) -> None:
        ctx.provide("workspace", self.workspace)
        tools: ToolRegistry = ctx.get("tools")
        tools.register(make_read_file_tool(self.workspace))
        tools.register(make_write_file_tool(self.workspace))
        tools.register(make_edit_file_tool(self.workspace))
        tools.register(make_list_dir_tool(self.workspace))
        if self.enable_bash:
            tools.register(make_bash_tool(self.workspace))
