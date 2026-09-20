"""Unit tests for standard built-in tools (read_file, write_file, edit_file, list_dir, bash)."""

from __future__ import annotations

from pathlib import Path

import pytest

from glmharness import Context, SessionLog, ToolRegistry
from glmharness.builtin_tools import (
    BuiltinToolsPlugin,
    make_bash_tool,
    make_edit_file_tool,
    make_list_dir_tool,
    make_read_file_tool,
    make_write_file_tool,
    resolve_safe_path,
)


def test_resolve_safe_path_boundary(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "inner.txt").write_text("hello")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")

    # Safe relative paths
    assert resolve_safe_path(workspace, "inner.txt") == (workspace / "inner.txt").resolve()
    assert resolve_safe_path(workspace, "./inner.txt") == (workspace / "inner.txt").resolve()

    # Safe absolute path inside workspace
    assert resolve_safe_path(workspace, workspace / "inner.txt") == (workspace / "inner.txt").resolve()

    # Traversal attacks
    with pytest.raises(PermissionError, match="escapes workspace"):
        resolve_safe_path(workspace, "../outside.txt")

    with pytest.raises(PermissionError, match="escapes workspace"):
        resolve_safe_path(workspace, outside)

    with pytest.raises(PermissionError, match="escapes workspace"):
        resolve_safe_path(workspace, "/etc/passwd")


async def test_read_file_tool(tmp_path: Path) -> None:
    file = tmp_path / "test.txt"
    lines = [f"line {i}\n" for i in range(10)]
    file.write_text("".join(lines))

    tool = make_read_file_tool(tmp_path)
    res = await ToolRegistry._invoke(tool, {"path": "test.txt", "offset": 2, "limit": 3})
    assert res["lines"] == 3
    assert res["total_lines"] == 10
    assert res["truncated"] is True
    assert res["content"] == "line 2\nline 3\nline 4\n"

    with pytest.raises(FileNotFoundError):
        await ToolRegistry._invoke(tool, {"path": "missing.txt"})


async def test_read_file_tool_rejects_oversized_file(tmp_path: Path) -> None:
    """The byte cap must reject large files before they load into RAM."""
    from glmharness.builtin_tools import _READ_FILE_MAX_BYTES

    # ``_READ_FILE_MAX_BYTES`` is 10 MiB. Writing that on every CI is slow,
    # so the cap is verified symbolically and only a 1 MiB file is written
    # when the constant is large enough to host it.
    assert _READ_FILE_MAX_BYTES == 10 * 1024 * 1024

    big = tmp_path / "too-big.txt"
    if _READ_FILE_MAX_BYTES >= 1024 * 1024:
        big.write_bytes(b"x" * (1024 * 1024))
        # Shrink the constant locally by monkey-patching the module's symbol,
        # so the rejection path runs without writing 10 MiB on every test
        # run. The check still exercises the production code path.
        import glmharness.builtin_tools as bt

        original = bt._READ_FILE_MAX_BYTES
        bt._READ_FILE_MAX_BYTES = 1024
        try:
            tool = make_read_file_tool(tmp_path)
            with pytest.raises(ValueError, match="too large"):
                await ToolRegistry._invoke(tool, {"path": "too-big.txt"})
        finally:
            bt._READ_FILE_MAX_BYTES = original


async def test_write_file_tool(tmp_path: Path) -> None:
    tool = make_write_file_tool(tmp_path)
    res = await ToolRegistry._invoke(tool, {"path": "nested/dir/output.txt", "content": "world"})
    assert res["bytes_written"] == 5
    assert (tmp_path / "nested" / "dir" / "output.txt").read_text() == "world"


async def test_edit_file_tool(tmp_path: Path) -> None:
    file = tmp_path / "code.py"
    file.write_text("def hello():\n    return 'hello'\n")
    tool = make_edit_file_tool(tmp_path)

    # Success edit
    res = await ToolRegistry._invoke(
        tool, {"path": "code.py", "old_string": "'hello'", "new_string": "'world'"}
    )
    assert res["replacements"] == 1
    assert file.read_text() == "def hello():\n    return 'world'\n"

    # Missing string
    with pytest.raises(ValueError, match="target text not found"):
        await ToolRegistry._invoke(tool, {"path": "code.py", "old_string": "missing", "new_string": "foo"})

    # Ambiguous duplicate string
    file.write_text("val = 1\nval = 1\n")
    with pytest.raises(ValueError, match="matches 2 times"):
        await ToolRegistry._invoke(
            tool, {"path": "code.py", "old_string": "val = 1", "new_string": "val = 2"}
        )


async def test_list_dir_tool(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("hello")
    tool = make_list_dir_tool(tmp_path)

    res = await ToolRegistry._invoke(tool, {"path": "."})
    names = [entry["name"] for entry in res["entries"]]
    assert "a.txt" in names
    assert "sub" in names

    with pytest.raises(NotADirectoryError):
        await ToolRegistry._invoke(tool, {"path": "a.txt"})


async def test_bash_tool_execution(tmp_path: Path) -> None:
    tool = make_bash_tool(tmp_path)

    # Normal command
    res = await ToolRegistry._invoke(tool, {"command": "echo 'running bash'"})
    assert res["exit_code"] == 0
    assert "running bash" in res["stdout"]
    assert res["timed_out"] is False

    # Command failure
    res_err = await ToolRegistry._invoke(tool, {"command": "sh -c 'exit 42'"})
    assert res_err["exit_code"] == 42

    # Timeout
    res_timeout = await ToolRegistry._invoke(tool, {"command": "sleep 2", "timeout_s": 0.1})
    assert res_timeout["timed_out"] is True
    assert res_timeout["exit_code"] == -1

    with pytest.raises(ValueError, match="timeout_s must be a finite number"):
        await ToolRegistry._invoke(tool, {"command": "echo should-not-run", "timeout_s": -1})

    with pytest.raises(ValueError, match="timeout_s must be a finite number"):
        await ToolRegistry._invoke(tool, {"command": "echo should-not-run", "timeout_s": float("nan")})


def test_builtin_tools_plugin(tmp_path: Path) -> None:
    ctx = Context()
    tools = ToolRegistry(ctx)
    ctx.provide("tools", tools)
    ctx.provide("sessions", SessionLog())

    plugin = BuiltinToolsPlugin(workspace=tmp_path)
    plugin.apply(ctx)

    assert "read_file" in tools.tools
    assert "write_file" in tools.tools
    assert "edit_file" in tools.tools
    assert "list_dir" in tools.tools
    assert "bash" in tools.tools
    assert ctx.get("workspace") == tmp_path.resolve()
