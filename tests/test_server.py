"""Unit tests for JSON-RPC 2.0 stdio protocol server."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from glmharness import HarnessConfig
from glmharness.server import ProtocolServer


def _make_server_io(messages: list[dict[str, Any]]) -> tuple[io.StringIO, io.StringIO]:
    raw_lines = "\n".join(json.dumps(msg) for msg in messages) + "\n"
    reader = io.StringIO(raw_lines)
    writer = io.StringIO()
    return reader, writer


def _read_responses(writer: io.StringIO) -> list[dict[str, Any]]:
    writer.seek(0)
    lines = [line.strip() for line in writer if line.strip()]
    return [json.loads(line) for line in lines]


def _authed_server(
    config: HarnessConfig, reqs: list[dict[str, Any]]
) -> tuple[ProtocolServer, list[dict[str, Any]]]:
    """Build a server whose stdin is pre-loaded with the same instance's
    auto-token attached to the first ``initialize`` request.

    The auto-token is per-instance, so the token the request stream uses
    must come from this same instance: we build the server first, capture
    the token, inject it into the request list, and only then materialize
    the reader.
    """
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
    # Replace the server's empty reader with the real, authenticated stream.
    server.reader = io.StringIO("\n".join(json.dumps(m) for m in injected) + "\n")
    return server, injected


async def test_server_initialize_and_ping(tmp_path: Path) -> None:
    config = HarnessConfig(mock="ok", workspace_dir=tmp_path)
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "shutdown", "params": {}},
    ]
    server, reqs = _authed_server(config, reqs)
    code = await server.serve()
    assert code == 0

    resps = _read_responses(server.writer)
    assert len(resps) == 3

    init_res = resps[0]["result"]
    assert init_res["abiVersion"] == 2
    assert init_res["runtimeInfo"]["name"] == "glm-5.3-flash"
    assert init_res["runtimeCapabilities"]["tools.native"] is True
    # Auto-token counts as auth.
    assert init_res["runtimeCapabilities"]["auth"] is True
    # Mutating tools are denied by default over RPC.
    assert init_res["runtimeCapabilities"]["rpcAllowMutating"] is False

    ping_res = resps[1]["result"]
    assert ping_res["ok"] is True

    assert resps[2]["result"]["ok"] is True


async def test_server_sessions_and_tools(tmp_path: Path) -> None:
    config = HarnessConfig(mock="ok", workspace_dir=tmp_path, sandbox_mode="allow")
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "session/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "session/new", "params": {"sessionId": "custom-sess"}},
        {"jsonrpc": "2.0", "id": 4, "method": "session/list", "params": {}},
        {"jsonrpc": "2.0", "id": 5, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/execute",
            "params": {"name": "write_file", "arguments": {"path": "test.txt", "content": "hello"}},
        },
        {"jsonrpc": "2.0", "id": 7, "method": "shutdown", "params": {}},
    ]
    server, reqs = _authed_server(config, reqs)
    await server.serve()

    resps = _read_responses(server.writer)
    assert resps[1]["result"] == ["default"]
    assert resps[2]["result"]["sessionId"] == "custom-sess"
    assert "custom-sess" in resps[3]["result"]
    tool_names = [t["function"]["name"] for t in resps[4]["result"]]
    assert "write_file" in tool_names
    # Mutating tool blocked by RPC gate even when sandbox allows it.
    assert resps[5]["error"]["code"] == -32004
    assert "mutating" in resps[5]["error"]["message"]


async def test_server_mutating_tools_blocked_without_opt_in(tmp_path: Path) -> None:
    config = HarnessConfig(mock="ok", workspace_dir=tmp_path)
    for tool_name in ("bash", "write_file", "edit_file", "github_write_file"):
        reqs = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/execute",
                "params": {
                    "name": tool_name,
                    "arguments": {
                        "command": "echo",
                        "path": "x",
                        "old_string": "a",
                        "new_string": "b",
                    },
                },
            },
            {"jsonrpc": "2.0", "id": 3, "method": "shutdown", "params": {}},
        ]
        server, reqs = _authed_server(config, reqs)
        await server.serve()
        resps = _read_responses(server.writer)
        assert resps[1]["error"]["code"] == -32004, f"{tool_name} should be blocked"
        assert tool_name in resps[1]["error"]["message"]


async def test_server_mutating_tools_allowed_when_opt_in(tmp_path: Path) -> None:
    config = HarnessConfig(
        mock="ok", workspace_dir=tmp_path, sandbox_mode="allow", rpc_allow_mutating=True
    )
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/execute",
            "params": {"name": "write_file", "arguments": {"path": "x.txt", "content": "ok"}},
        },
        {"jsonrpc": "2.0", "id": 3, "method": "shutdown", "params": {}},
    ]
    server, reqs = _authed_server(config, reqs)
    await server.serve()
    resps = _read_responses(server.writer)
    assert resps[1]["result"]["ok"] is True
    assert (tmp_path / "x.txt").read_text() == "ok"


async def test_server_session_import_logpath_must_be_under_state_dir(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"type":"user/message","data":{"content":"hi"},"ts":1.0}\n')
    config = HarnessConfig(
        mock="ok",
        workspace_dir=tmp_path,
        state_dir=state_dir,
    )

    # Absolute path: rejected
    reqs_abs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "session/import",
         "params": {"logPath": str(outside)}},
        {"jsonrpc": "2.0", "id": 3, "method": "shutdown", "params": {}},
    ]
    server, _ = _authed_server(config, reqs_abs)
    await server.serve()
    resps = _read_responses(server.writer)
    assert resps[1]["error"]["code"] == -32003

    # Path traversal: rejected
    reqs_trav = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "session/import",
         "params": {"logPath": "../outside.jsonl"}},
        {"jsonrpc": "2.0", "id": 3, "method": "shutdown", "params": {}},
    ]
    server, _ = _authed_server(config, reqs_trav)
    await server.serve()
    resps = _read_responses(server.writer)
    assert resps[1]["error"]["code"] == -32003

    # Relative path under state_dir: accepted
    inside = state_dir / "good.jsonl"
    inside.write_text('{"type":"user/message","data":{"content":"hi"},"ts":1.0}\n')
    reqs_ok = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "session/import",
         "params": {"logPath": "good.jsonl"}},
        {"jsonrpc": "2.0", "id": 3, "method": "shutdown", "params": {}},
    ]
    server, _ = _authed_server(config, reqs_ok)
    await server.serve()
    resps = _read_responses(server.writer)
    assert resps[1]["result"]["imported"] == 1


async def test_server_no_state_dir_blocks_logpath(tmp_path: Path) -> None:
    config = HarnessConfig(mock="ok", workspace_dir=tmp_path)
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "session/import",
         "params": {"logPath": "any.jsonl"}},
        {"jsonrpc": "2.0", "id": 3, "method": "shutdown", "params": {}},
    ]
    server, _ = _authed_server(config, reqs)
    await server.serve()
    resps = _read_responses(server.writer)
    assert resps[1]["error"]["code"] == -32003


async def test_server_agent_send(tmp_path: Path) -> None:
    config = HarnessConfig(mock="I am GLM assistant.", workspace_dir=tmp_path)
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "agent/send", "params": {"text": "hello"}},
        {"jsonrpc": "2.0", "id": 3, "method": "shutdown", "params": {}},
    ]
    server, _ = _authed_server(config, reqs)
    await server.serve()

    resps = _read_responses(server.writer)
    agent_res = resps[1]["result"]
    assert agent_res["ok"] is True
    assert agent_res["answer"] == "I am GLM assistant."
    assert agent_res["sessionId"] == "default"


async def test_server_error_handling(tmp_path: Path) -> None:
    """Mixed-input error handling with auto-token auth.

    Pre-auth calls reject; after ``initialize`` (with the auto-token) the
    server processes invalid frames, unknown methods, and missing params
    with the documented JSON-RPC error codes.
    """
    config = HarnessConfig(mock="ok", workspace_dir=tmp_path)
    # Authenticated reqs (helper injects the auto-token on initialize).
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "unknown_method"},
        {"jsonrpc": "2.0", "id": 3, "method": "agent/send", "params": {}},
        {"jsonrpc": "2.0", "id": 4, "method": "shutdown"},
    ]
    server, _ = _authed_server(config, reqs)
    # Prepend two error frames (parse error + invalid request) to the
    # authenticated stream so the test exercises the full code path.
    real_reader = server.reader
    real_reader.seek(0)
    real_str = real_reader.read()
    head = "not json\n" + json.dumps({"jsonrpc": "1.0", "id": 99, "method": "ping"}) + "\n"
    server.reader = io.StringIO(head + real_str)
    await server.serve()

    resps = _read_responses(server.writer)
    assert resps[0]["error"]["code"] == -32700  # Parse error
    assert resps[1]["error"]["code"] == -32600  # Invalid Request
    assert resps[2]["result"]["abiVersion"] == 2  # authorized initialize
    assert resps[3]["error"]["code"] == -32601  # Method not found
    assert resps[4]["error"]["code"] == -32602  # Invalid params
    assert resps[5]["result"]["ok"] is True


async def test_server_rpc_token_blocks_pre_initialize(tmp_path: Path) -> None:
    """When ``rpc_token`` is set, every call before ``initialize`` is rejected."""
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "initialize",
            "params": {"authToken": "secret"},
        },
        {"jsonrpc": "2.0", "id": 3, "method": "ping", "params": {}},
        {"jsonrpc": "2.0", "id": 4, "method": "shutdown", "params": {}},
    ]
    reader, writer = _make_server_io(reqs)
    config = HarnessConfig(mock="ok", workspace_dir=tmp_path, rpc_token="secret")
    server = ProtocolServer(config, reader=reader, writer=writer)
    await server.serve()
    resps = _read_responses(writer)
    assert resps[0]["error"]["code"] == -32001
    assert resps[1]["result"]["abiVersion"] == 2
    assert resps[1]["result"]["runtimeCapabilities"]["auth"] is True
    assert resps[2]["result"]["ok"] is True
    assert resps[3]["result"]["ok"] is True


async def test_server_rpc_token_rejects_wrong_token(tmp_path: Path) -> None:
    """initialize with the wrong token fails closed."""
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"authToken": "wrong"}},
    ]
    reader, writer = _make_server_io(reqs)
    config = HarnessConfig(mock="ok", workspace_dir=tmp_path, rpc_token="secret")
    server = ProtocolServer(config, reader=reader, writer=writer)
    await server.serve()
    resps = _read_responses(writer)
    assert resps[0]["error"]["code"] == -32001
    assert "authToken" in resps[0]["error"]["message"]


async def test_server_auto_token_required_by_default(tmp_path: Path) -> None:
    """When neither ``rpc_token`` is set nor ``rpc_auto_token`` disabled, the
    server auto-generates a token and rejects every call before the token
    is presented."""
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}},
    ]
    reader, writer = _make_server_io(reqs)
    config = HarnessConfig(mock="ok", workspace_dir=tmp_path)
    server = ProtocolServer(config, reader=reader, writer=writer)
    await server.serve()
    resps = _read_responses(writer)
    assert resps[0]["error"]["code"] == -32001
    assert "send initialize first" in resps[0]["error"]["message"]


async def test_server_explicit_no_auth(tmp_path: Path) -> None:
    """``rpc_auto_token=False`` and unset ``rpc_token`` keeps the legacy
    unauthenticated behavior; this is for same-UID IDE integrations."""
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}},
    ]
    reader, writer = _make_server_io(reqs)
    config = HarnessConfig(mock="ok", workspace_dir=tmp_path, rpc_auto_token=False)
    server = ProtocolServer(config, reader=reader, writer=writer)
    assert server._auto_token is None
    await server.serve()
    resps = _read_responses(writer)
    assert resps[0]["result"]["ok"] is True
