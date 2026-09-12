"""Unit tests for workspace search tools (find_files and grep_search)."""

from __future__ import annotations

from pathlib import Path

import pytest

from glmharness import Context, SessionLog, ToolRegistry
from glmharness.builtin_tools import (
    BuiltinToolsPlugin,
    make_find_files_tool,
    make_grep_search_tool,
)


@pytest.fixture
def workspace_with_files(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()

    # Directory layout
    (ws / "src").mkdir()
    (ws / "src" / "main.py").write_text("def run_app():\n    print('starting application')\n    return 0\n")
    (ws / "src" / "utils.py").write_text("def helper_fn():\n    # APP_HELPER\n    return 'help'\n")

    (ws / "docs").mkdir()
    (ws / "docs" / "guide.md").write_text("# User Guide\nSee run_app documentation.\n")

    (ws / "config.json").write_text('{"app": "glm-harness", "version": 1}\n')

    # Ignored directories
    (ws / ".git").mkdir()
    (ws / ".git" / "config").write_text("git internal run_app")
    (ws / "__pycache__").mkdir()
    (ws / "__pycache__" / "main.cpython-314.pyc").write_bytes(b"\x00\x01\x02\x03run_app")

    return ws


async def test_find_files_all_and_pattern(workspace_with_files: Path) -> None:
    tool = make_find_files_tool(workspace_with_files)

    # Find all files (default pattern="*")
    res = await ToolRegistry._invoke(tool, {"path": "."})
    paths = [entry["path"] for entry in res["entries"]]
    assert "src/main.py" in paths
    assert "src/utils.py" in paths
    assert "docs/guide.md" in paths
    assert "config.json" in paths
    # Ignored dirs should not be included
    assert not any(p.startswith(".git") for p in paths)
    assert not any(p.startswith("__pycache__") for p in paths)

    # Glob pattern for Python files
    res_py = await ToolRegistry._invoke(tool, {"pattern": "*.py"})
    py_paths = [entry["path"] for entry in res_py["entries"]]
    assert sorted(py_paths) == ["src/main.py", "src/utils.py"]


async def test_find_files_filtering_and_scoping(workspace_with_files: Path) -> None:
    tool = make_find_files_tool(workspace_with_files)

    # Filter directories only
    res_dirs = await ToolRegistry._invoke(tool, {"file_type": "dir"})
    dir_paths = [entry["path"] for entry in res_dirs["entries"]]
    assert "src" in dir_paths
    assert "docs" in dir_paths
    assert "src/main.py" not in dir_paths

    # Scope to subdirectory
    res_sub = await ToolRegistry._invoke(tool, {"path": "docs"})
    doc_paths = [entry["path"] for entry in res_sub["entries"]]
    assert doc_paths == ["docs/guide.md"]

    # Traversal security check
    with pytest.raises(PermissionError, match="escapes workspace"):
        await ToolRegistry._invoke(tool, {"path": "../outside"})


async def test_find_files_max_results_truncation(workspace_with_files: Path) -> None:
    tool = make_find_files_tool(workspace_with_files)
    res = await ToolRegistry._invoke(tool, {"max_results": 2})
    assert len(res["entries"]) == 2
    assert res["truncated"] is True
    assert res["total_matches"] > 2


async def test_grep_search_literal_and_case(workspace_with_files: Path) -> None:
    tool = make_grep_search_tool(workspace_with_files)

    # Exact case match
    res = await ToolRegistry._invoke(tool, {"query": "run_app"})
    assert res["total_matches"] == 2
    paths = [m["path"] for m in res["matches"]]
    assert "src/main.py" in paths
    assert "docs/guide.md" in paths

    # Case insensitive
    res_ci = await ToolRegistry._invoke(tool, {"query": "app_helper", "case_sensitive": False})
    assert res_ci["total_matches"] == 1
    assert res_ci["matches"][0]["path"] == "src/utils.py"
    assert res_ci["matches"][0]["line_number"] == 2
    assert "# APP_HELPER" in res_ci["matches"][0]["line"]


async def test_grep_search_regex_and_include_pattern(workspace_with_files: Path) -> None:
    tool = make_grep_search_tool(workspace_with_files)

    # Regex search
    res_regex = await ToolRegistry._invoke(
        tool, {"query": r"def\s+\w+\(\):", "is_regex": True}
    )
    assert res_regex["total_matches"] == 2
    for match in res_regex["matches"]:
        assert match["path"].startswith("src/")

    # Include pattern filter (only docs)
    res_docs = await ToolRegistry._invoke(
        tool, {"query": "run_app", "include_pattern": "*.md"}
    )
    assert res_docs["total_matches"] == 1
    assert res_docs["matches"][0]["path"] == "docs/guide.md"


async def test_grep_search_single_file_and_boundary(workspace_with_files: Path) -> None:
    tool = make_grep_search_tool(workspace_with_files)

    # Search targeted to a single file
    res = await ToolRegistry._invoke(tool, {"query": "run_app", "path": "src/main.py"})
    assert res["total_matches"] == 1
    assert res["matches"][0]["path"] == "src/main.py"

    # Traversal security check
    with pytest.raises(PermissionError, match="escapes workspace"):
        await ToolRegistry._invoke(tool, {"query": "test", "path": "../outside"})


async def test_search_tools_mounted_in_builtin_plugin(workspace_with_files: Path) -> None:
    ctx = Context()
    tools = ToolRegistry(ctx)
    ctx.provide("tools", tools)
    ctx.provide("sessions", SessionLog())

    plugin = BuiltinToolsPlugin(workspace=workspace_with_files)
    plugin.apply(ctx)

    assert "find_files" in tools.tools
    assert "grep_search" in tools.tools
