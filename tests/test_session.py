"""Tests for the append-only session log and corruption handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glmharness import SessionEvent, SessionLog
from glmharness.errors import SessionCorruptError


def test_append_returns_event_with_timestamp() -> None:
    log = SessionLog()
    event = log.append("turn/start", {})
    assert isinstance(event, SessionEvent)
    assert event.type == "turn/start"
    assert event.data == {}


def test_derive_messages_projects_surface_only() -> None:
    log = SessionLog()
    log.append("user/message", {"content": "hi"})
    log.append("assistant/chunk", {"content": "tok"})
    log.append("assistant/message", {"content": "answer"})
    log.append("tool/result", {"content": "{}"})
    assert log.derive_messages() == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "answer"},
        {"role": "tool", "content": "{}"},
    ]


def test_writes_one_jsonl_line_with_fsync(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    log = SessionLog(path=path)
    log.append("user/message", {"content": "hi"})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["type"] == "user/message"
    assert row["data"] == {"content": "hi"}


def test_reload_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    log = SessionLog(path=path)
    log.append("user/message", {"content": "hi"})
    log.append("assistant/message", {"content": "answer"})
    restored = SessionLog(path=path)
    assert restored.derive_messages() == log.derive_messages()


def test_corrupt_skip_drops_lines_and_increments_count(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": "user/message", "data": {"content": "ok"}, "ts": 1.0}),
                "not-json",
                json.dumps({"type": "assistant/message", "data": {"content": "hi"}, "ts": 2.0}),
            ]
        )
        + "\n"
    )
    log = SessionLog(path=path, corrupt_policy="skip")
    assert log.corrupt_lines == 1
    assert [e.type for e in log.events] == ["user/message", "assistant/message"]


def test_corrupt_fail_raises(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text("not-json\n")
    with pytest.raises(SessionCorruptError):
        SessionLog(path=path, corrupt_policy="fail")


def test_corrupt_rename_moves_file_aside(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text(
        json.dumps({"type": "user/message", "data": {"content": "ok"}, "ts": 1.0}) + "\n"
        "garbage\n"
    )
    log = SessionLog(path=path, corrupt_policy="rename")
    siblings = list(tmp_path.iterdir())
    assert len(siblings) == 1
    assert siblings[0].name.startswith("s.jsonl.corrupt-")
    assert log.derive_messages() == [{"role": "user", "content": "ok"}]


def test_rename_policy_appends_resume_after_corrupt(tmp_path: Path) -> None:
    """After a corrupt file is moved aside, subsequent ``append`` calls must
    write to a fresh log file at the original path (the renamed file is
    preserved for inspection). This is the contract operators depend on to
    never silently lose live history after a malformed prefix."""
    path = tmp_path / "live.jsonl"
    path.write_text(
        json.dumps({"type": "user/message", "data": {"content": "before"}, "ts": 1.0})
        + "\n"
        "not-json\n"
    )
    log = SessionLog(path=path, corrupt_policy="rename")
    # The good prefix is in memory.
    assert log.derive_messages() == [{"role": "user", "content": "before"}]
    # Subsequent append must go to the original path (not the renamed one).
    log.append("assistant/message", {"content": "after"})
    text = path.read_text(encoding="utf-8")
    assert '"after"' in text
    # The renamed file is intact and contains only the corrupt prefix.
    renamed = next(tmp_path.iterdir())
    assert renamed.name.startswith("live.jsonl.corrupt-")
    assert "not-json" in renamed.read_text(encoding="utf-8")
    # The "before" line is in the renamed file too (we moved the entire
    # original JSONL to the side, so the good prefix is preserved there).
    assert "before" in renamed.read_text(encoding="utf-8")
