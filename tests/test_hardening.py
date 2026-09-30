"""Reliability hardening tests.

Covers the adversarial-review fixes: atomic file writes (crash safety via
fsync fault injection), write-size caps, bounded server session cache, the
whole-turn deadline for ``agent/send``, and identity constants wired through
``identity.brand``.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
from typing import Any

from glmharness import HarnessConfig
from glmharness.builtin_tools import (
    _MAX_WRITE_BYTES,
    make_edit_file_tool,
    make_write_file_tool,
)
from glmharness.identity.brand import MODEL_VENDOR
from glmharness.server import ProtocolServer

# ---------------------------------------------------------------------------
# helpers (same harness pattern as tests/test_server.py)
# ---------------------------------------------------------------------------


def _make_server_io(messages: list[dict[str, Any]]) -> tuple[io.StringIO, io.StringIO]:
    raw_lines = "\n".join(json.dumps(msg) for msg in messages) + "\n"
    return io.StringIO(raw_lines), io.StringIO()


def _read_responses(writer: io.StringIO) -> list[dict[str, Any]]:
    writer.seek(0)
    return [json.loads(line) for line in writer if line.strip()]


def _authed_server(
    config: HarnessConfig, reqs: list[dict[str, Any]]
) -> tuple[ProtocolServer, list[dict[str, Any]]]:
    """Build a server with its per-instance auto-token injected into the
    first ``initialize`` request (mirrors tests/test_server.py)."""
    reader, writer = _make_server_io(reqs)
    server = ProtocolServer(config, reader=reader, writer=writer)
    token = server._auto_token
    assert token is not None
    injected: list[dict[str, Any]] = []
    for msg in reqs:
        if msg.get("method") == "initialize" and "authToken" not in (msg.get("params") or {}):
            params = dict(msg.get("params") or {})
            params["authToken"] = token
            injected.append({**msg, "params": params})
        else:
            injected.append(msg)
    server.reader = io.StringIO("\n".join(json.dumps(m) for m in injected) + "\n")
    return server, injected


# ---------------------------------------------------------------------------
# atomic writes
# ---------------------------------------------------------------------------


def test_write_file_atomic_success(tmp_path: Path) -> None:
    tool = make_write_file_tool(tmp_path)
    res = tool.handler({"path": "a/b.txt", "content": "hello"})
    assert res["bytes_written"] == 5
    assert (tmp_path / "a" / "b.txt").read_text() == "hello"
    # No temp files left behind.
    assert [p.name for p in (tmp_path / "a").iterdir()] == ["b.txt"]


def test_write_file_survives_fsync_failure(tmp_path: Path, monkeypatch: Any) -> None:
    """Fault injection: if fsync fails mid-write, the original file is intact
    and no temp file is left behind."""
    target = tmp_path / "f.txt"
    target.write_text("original")
    tool = make_write_file_tool(tmp_path)

    def _boom(fd: int) -> None:
        raise OSError("disk on fire")

    monkeypatch.setattr(os, "fsync", _boom)
    try:
        tool.handler({"path": "f.txt", "content": "replaced"})
        raise AssertionError("expected OSError from fsync fault injection")
    except OSError:
        pass
    assert target.read_text() == "original"
    assert [p.name for p in tmp_path.iterdir()] == ["f.txt"]


def test_edit_file_atomic_success(tmp_path: Path) -> None:
    target = tmp_path / "code.py"
    target.write_text("value = 1\n")
    tool = make_edit_file_tool(tmp_path)
    tool.handler({"path": "code.py", "old_string": "value = 1", "new_string": "value = 2"})
    assert target.read_text() == "value = 2\n"
    assert [p.name for p in tmp_path.iterdir()] == ["code.py"]


def test_edit_file_survives_fsync_failure(tmp_path: Path, monkeypatch: Any) -> None:
    target = tmp_path / "code.py"
    target.write_text("keep = 1\n")
    tool = make_edit_file_tool(tmp_path)

    def _boom(fd: int) -> None:
        raise OSError("disk on fire")

    monkeypatch.setattr(os, "fsync", _boom)
    try:
        tool.handler({"path": "code.py", "old_string": "keep = 1", "new_string": "keep = 2"})
        raise AssertionError("expected OSError from fsync fault injection")
    except OSError:
        pass
    assert target.read_text() == "keep = 1\n"
    assert [p.name for p in tmp_path.iterdir()] == ["code.py"]


# ---------------------------------------------------------------------------
# write-size cap (mirrors the read cap)
# ---------------------------------------------------------------------------


def test_write_file_rejects_oversized_content(tmp_path: Path) -> None:
    tool = make_write_file_tool(tmp_path)
    oversized = "x" * (_MAX_WRITE_BYTES + 1)
    try:
        tool.handler({"path": "big.bin", "content": oversized})
        raise AssertionError("expected ValueError for oversized write")
    except ValueError as exc:
        assert "too large" in str(exc)
    assert not (tmp_path / "big.bin").exists()


def test_edit_file_rejects_oversized_replacement(tmp_path: Path) -> None:
    tool = make_edit_file_tool(tmp_path)
    (tmp_path / "f.txt").write_text("needle\n")
    big_new = "y" * (_MAX_WRITE_BYTES + 1)
    try:
        tool.handler({"path": "f.txt", "old_string": "needle", "new_string": big_new})
        raise AssertionError("expected ValueError for oversized edit")
    except ValueError as exc:
        assert "too large" in str(exc)
    assert (tmp_path / "f.txt").read_text() == "needle\n"


# ---------------------------------------------------------------------------
# bounded server session cache
# ---------------------------------------------------------------------------


async def test_server_evicts_oldest_session_beyond_max(tmp_path: Path) -> None:
    config = HarnessConfig(mock="ok", workspace_dir=tmp_path, max_sessions=3)
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {"sessionId": "a"}},
        {"jsonrpc": "2.0", "id": 3, "method": "session/new", "params": {"sessionId": "b"}},
        {"jsonrpc": "2.0", "id": 4, "method": "session/new", "params": {"sessionId": "c"}},
        {"jsonrpc": "2.0", "id": 5, "method": "session/list", "params": {}},
        {"jsonrpc": "2.0", "id": 6, "method": "shutdown", "params": {}},
    ]
    server, reqs = _authed_server(config, reqs)
    code = await server.serve()
    assert code == 0
    resps = _read_responses(server.writer)
    # "a" was evicted; "default" is never evicted.
    assert resps[4]["result"] == ["default", "b", "c"]


async def test_server_tiny_cap_keeps_default_and_newest(tmp_path: Path) -> None:
    """At max_sessions=1 the just-registered session is never evicted (the
    RPC response must not advertise a session that was instantly dropped)
    and ``default`` is never evicted."""
    config = HarnessConfig(mock="ok", workspace_dir=tmp_path, max_sessions=1)
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {"sessionId": "a"}},
        {"jsonrpc": "2.0", "id": 3, "method": "session/list", "params": {}},
        {"jsonrpc": "2.0", "id": 4, "method": "shutdown", "params": {}},
    ]
    server, reqs = _authed_server(config, reqs)
    await server.serve()
    resps = _read_responses(server.writer)
    assert resps[2]["result"] == ["default", "a"]


# ---------------------------------------------------------------------------
# whole-turn deadline for agent/send
# ---------------------------------------------------------------------------


class _HangingAgent:
    """AgentLoop stand-in whose run() never completes on its own."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def run(self, text: str) -> str:
        await asyncio.sleep(30)
        raise AssertionError("should have been cancelled by the turn deadline")


async def test_agent_send_turn_timeout(tmp_path: Path, monkeypatch: Any) -> None:
    import glmharness.server as server_module

    monkeypatch.setattr(server_module, "AgentLoop", _HangingAgent)
    config = HarnessConfig(mock="ok", workspace_dir=tmp_path, turn_timeout_s=0.05)
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "agent/send",
            "params": {"sessionId": "t", "text": "hello"},
        },
        {"jsonrpc": "2.0", "id": 3, "method": "shutdown", "params": {}},
    ]
    server, reqs = _authed_server(config, reqs)
    code = await server.serve()
    assert code == 0
    resps = _read_responses(server.writer)
    err = resps[1]["error"]
    assert err["code"] == -32008
    assert "timed out" in err["message"]
    assert err["data"]["sessionId"] == "t"
    # The log stays coherent: the turn closed with a durable failed marker.
    events = [(e.type, e.data) for e in server.sessions["t"].events]
    assert ("turn/end", {"status": "failed", "error": "TURN_TIMEOUT"}) in events


# ---------------------------------------------------------------------------
# identity constants wired through brand.py
# ---------------------------------------------------------------------------


async def test_initialize_reports_brand_vendor(tmp_path: Path) -> None:
    config = HarnessConfig(mock="ok", workspace_dir=tmp_path)
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}},
    ]
    server, reqs = _authed_server(config, reqs)
    await server.serve()
    resps = _read_responses(server.writer)
    assert resps[0]["result"]["runtimeInfo"]["vendor"] == MODEL_VENDOR
