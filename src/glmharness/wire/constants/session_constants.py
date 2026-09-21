"""Session wire constants: event vocabulary for the session log.

The session log is the wire format for model-visible state, so its event
names and policy vocabulary live here, not scattered across modules.
"""

from __future__ import annotations

from typing import Literal

SessionEventType = Literal["user/message", "assistant/message", "tool/result"]

SESSION_SURFACE: frozenset[str] = frozenset({
    "user/message",
    "assistant/message",
    "tool/result",
})

CORRUPT_POLICIES: frozenset[str] = frozenset({"skip", "rename", "fail"})
DEFAULT_CORRUPT_POLICY = "skip"
