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


async def test_server_initialize_and_ping(tmp_path: Path) -> None:
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "shutdown", "params": {}},
    ]
    reader, writer = _make_server_io(reqs)
    config = HarnessConfig(mock="ok", workspace_dir=tmp_path)
    server = ProtocolServer(config, reader=reader, writer=writer)
    code = await server.serve()
    assert code == 0

    resps = _read_responses(writer)
    assert len(resps) == 3

    # initialize
    init_res = resps[0]["result"]
    assert init_res["abiVersion"] == 2
    assert init_res["runtimeInfo"]["name"] == "glm-5.3-flash"
    assert init_res["runtimeCapabilities"]["tools.native"] is True

    # ping
    ping_res = resps[1]["result"]
    assert ping_res["ok"] is True

    # shutdown
    assert resps[2]["result"]["ok"] is True


async def test_server_sessions_and_tools(tmp_path: Path) -> None:
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
    reader, writer = _make_server_io(reqs)
    config = HarnessConfig(mock="ok", workspace_dir=tmp_path)
    server = ProtocolServer(config, reader=reader, writer=writer)
    await server.serve()

    resps = _read_responses(writer)
    assert resps[1]["result"] == ["default"]
    assert resps[2]["result"]["sessionId"] == "custom-sess"
    assert "custom-sess" in resps[3]["result"]
    tool_names = [t["function"]["name"] for t in resps[4]["result"]]
    assert "write_file" in tool_names
    assert resps[5]["result"]["ok"] is True
    assert (tmp_path / "test.txt").read_text() == "hello"


async def test_server_agent_send(tmp_path: Path) -> None:
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "agent/send", "params": {"text": "hello"}},
        {"jsonrpc": "2.0", "id": 3, "method": "shutdown", "params": {}},
    ]
    reader, writer = _make_server_io(reqs)
    config = HarnessConfig(mock="I am GLM assistant.", workspace_dir=tmp_path)
    server = ProtocolServer(config, reader=reader, writer=writer)
    await server.serve()

    resps = _read_responses(writer)
    agent_res = resps[1]["result"]
    assert agent_res["ok"] is True
    assert agent_res["answer"] == "I am GLM assistant."
    assert agent_res["sessionId"] == "default"


async def test_server_error_handling(tmp_path: Path) -> None:
    raw_lines = (
        "not json\n"
        '{"jsonrpc": "1.0", "id": 1, "method": "ping"}\n'
        '{"jsonrpc": "2.0", "id": 2, "method": "unknown_method"}\n'
        '{"jsonrpc": "2.0", "id": 3, "method": "agent/send", "params": {}}\n'
        '{"jsonrpc": "2.0", "id": 4, "method": "shutdown"}\n'
    )
    reader = io.StringIO(raw_lines)
    writer = io.StringIO()
    config = HarnessConfig(mock="ok", workspace_dir=tmp_path)
    server = ProtocolServer(config, reader=reader, writer=writer)
    await server.serve()

    resps = _read_responses(writer)
    assert resps[0]["error"]["code"] == -32700  # Parse error
    assert resps[1]["error"]["code"] == -32600  # Invalid Request
    assert resps[2]["error"]["code"] == -32601  # Method not found
    assert resps[3]["error"]["code"] == -32602  # Invalid params
    assert resps[4]["result"]["ok"] is True
