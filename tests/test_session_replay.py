"""Tests for multi-turn history reconstruction and replay from logs or projections."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from glmharness.cli import main
from glmharness.config import HarnessConfig
from glmharness.server import ProtocolServer
from glmharness.session import SessionLog


def test_import_projection_populates_session() -> None:
    log = SessionLog()
    projection = [
        {"role": "user", "content": "hello from previous session"},
        {"role": "assistant", "content": "hi there! how can I help?"},
        {"role": "tool", "content": "tool execution result: 42"},
    ]
    imported = log.import_projection(projection)
    assert imported == 3
    assert len(log.events) == 3

    derived = log.derive_messages()
    assert derived == [
        {"role": "user", "content": "hello from previous session"},
        {"role": "assistant", "content": "hi there! how can I help?"},
        {"role": "tool", "content": "tool execution result: 42"},
    ]


def test_import_log_from_jsonl(tmp_path: Path) -> None:
    session_file = tmp_path / "foreign_session.jsonl"
    events = [
        {"type": "user/message", "data": {"content": "turn 1 user"}, "ts": 100.0},
        {"type": "assistant/message", "data": {"content": "turn 1 assistant"}, "ts": 101.0},
        {"kind": "user/message", "payload": {"text": "turn 2 user (dmh format)"}, "ts": 102.0},
    ]
    with session_file.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")

    log = SessionLog()
    imported = log.import_log(session_file)
    assert imported == 3

    derived = log.derive_messages()
    assert len(derived) == 3
    assert derived[0]["content"] == "turn 1 user"
    assert derived[1]["content"] == "turn 1 assistant"
    assert derived[2]["content"] == "turn 2 user (dmh format)"


@pytest.mark.asyncio
async def test_server_initialize_advertises_replay_and_compaction() -> None:
    cfg = HarnessConfig(mock="mock-reply")
    writer = io.StringIO()
    server = ProtocolServer(cfg, writer=writer)
    await server.initialize_runtime()

    frame = {"jsonrpc": "2.0", "id": "1", "method": "initialize"}
    await server.handle_request(frame)

    output = server.writer.getvalue()
    resp = json.loads(output.strip())
    caps = resp["result"]["runtimeCapabilities"]
    assert caps["replay.from_log"] is True
    assert caps["compaction"] is True


@pytest.mark.asyncio
async def test_server_session_import_and_compact_rpc() -> None:
    cfg = HarnessConfig(mock="mock-reply")
    writer = io.StringIO()
    server = ProtocolServer(cfg, writer=writer)
    await server.initialize_runtime()

    # Import projection via session/import
    import_frame = {
        "jsonrpc": "2.0",
        "id": "2",
        "method": "session/import",
        "params": {
            "sessionId": "switched-session",
            "projection": [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second question"},
                {"role": "assistant", "content": "second answer"},
                {"role": "user", "content": "third question"},
                {"role": "assistant", "content": "third answer"},
            ],
        },
    }
    await server.handle_request(import_frame)

    session_log = server.sessions["switched-session"]
    assert len(session_log.derive_messages()) == 6

    # Call session/compact
    compact_frame = {
        "jsonrpc": "2.0",
        "id": "3",
        "method": "session/compact",
        "params": {
            "sessionId": "switched-session",
            "threshold": 4,
            "keepRounds": 2,
        },
    }
    await server.handle_request(compact_frame)

    # Output should contain successful compaction
    messages_after = session_log.derive_messages()
    assert "[compacted history]" in messages_after[0]["content"]


def test_cli_accepts_replay_log_and_projection(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log_file = tmp_path / "replay.jsonl"
    log_file.write_text(
        json.dumps({"type": "user/message", "data": {"content": "prior knowledge"}, "ts": 1.0}) + "\n",
        encoding="utf-8",
    )

    proj_json = json.dumps([{"role": "assistant", "content": "prior assistant"}])

    code = main([
        "--mock",
        "reply",
        "--replay-log",
        str(log_file),
        "--projection",
        proj_json,
        "new question",
    ])
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "reply"
