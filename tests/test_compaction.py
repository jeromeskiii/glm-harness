"""Tests for session history token/message compaction."""

from __future__ import annotations

import pytest

from glmharness.compaction import CompactionPlugin, compact_session, estimate_tokens
from glmharness.context import Context, PluginLoader
from glmharness.session import SessionLog


def test_estimate_tokens() -> None:
    assert estimate_tokens("hello") == 1
    assert estimate_tokens("a" * 40) == 10


def test_compact_session_disabled_when_threshold_zero() -> None:
    log = SessionLog()
    log.append("user/message", {"content": "hello"})
    res = compact_session(log, threshold=0)
    assert res["compacted"] is False
    assert res["dropped_count"] == 0


def test_compact_session_noop_under_threshold() -> None:
    log = SessionLog()
    log.append("user/message", {"content": "short"})
    log.append("assistant/message", {"content": "brief"})
    res = compact_session(log, threshold=1000)
    assert res["compacted"] is False
    assert len(log.derive_messages()) == 2


def test_compact_session_summarize_strategy() -> None:
    log = SessionLog()
    log.append("user/message", {"content": "Turn 1: Initial query"})
    log.append("assistant/message", {"content": "Turn 1: Answer 1"})
    log.append("user/message", {"content": "Turn 2: Follow-up question"})
    log.append("assistant/message", {"content": "Turn 2: Answer 2"})
    log.append("user/message", {"content": "Turn 3: What about X?"})
    log.append("assistant/message", {"content": "Turn 3: Answer about X"})
    log.append("user/message", {"content": "Turn 4: Final query"})

    # Threshold < 100 means count-based (metric is surface event count = 7)
    res = compact_session(log, threshold=4, keep_rounds=2, strategy="summarize")
    assert res["compacted"] is True
    assert res["dropped_count"] == 4  # turns 1 (assistant) through turn 3 (assistant)
    assert "Compacted conversation history" in str(res["summary"])

    messages = log.derive_messages()
    # Should have: [compacted history], and the 2 kept messages (turn 3 answer, turn 4 final query)
    assert messages[0]["role"] == "user"
    assert "[compacted history]" in messages[0]["content"]
    assert any("Final query" in m["content"] for m in messages)


def test_compact_session_truncate_strategy() -> None:
    log = SessionLog()
    for i in range(10):
        log.append("user/message", {"content": f"User msg {i}"})
        log.append("assistant/message", {"content": f"Assistant msg {i}"})

    res = compact_session(log, threshold=5, keep_rounds=4, strategy="truncate")
    assert res["compacted"] is True
    assert "Compacted 15 prior interaction turns" in str(res["summary"])

    messages = log.derive_messages()
    assert messages[0]["role"] == "user"
    assert "[compacted history]" in messages[0]["content"]
    assert len(messages) == 5  # summary + 4 kept messages


@pytest.mark.asyncio
async def test_compaction_plugin_triggers_on_agent_request() -> None:
    ctx = Context()
    sessions = SessionLog()
    for i in range(8):
        sessions.append("user/message", {"content": f"msg {i}"})
    ctx.services["sessions"] = sessions

    plugin = CompactionPlugin(threshold=4, keep_rounds=2)
    loader = PluginLoader(ctx)
    await loader.mount([plugin])

    # Dispatch agent/request event
    await ctx.events.dispatch("agent/request", "waterfall", {"messages": []})

    # Compaction event should have been logged
    compacted_events = [e for e in sessions.events if e.type == "session/compacted"]
    assert len(compacted_events) == 1
    assert compacted_events[0].data["dropped_count"] > 0
