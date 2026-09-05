"""The session log: append-only JSONL with durability and corruption policy.

The log is the source of truth for everything model-visible
("model-visible means logged"). Appends are single-line writes followed by
``fsync``, so a crash can never interleave two records — worst case is one
truncated final line, which the loader detects.

On load, corrupt lines are handled per the configured policy:

- ``skip``: drop the corrupt line, log a warning, continue (default).
- ``rename``: move the corrupt file aside as ``<name>.corrupt-<ts>`` and
  start a fresh log carrying the parsed-good prefix.
- ``fail``: raise :class:`SessionCorruptError` — never run on suspect history.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from .errors import SessionCorruptError
from .logging import get_logger


@dataclass(frozen=True)
class SessionEvent:
    type: str
    data: dict[str, Any]
    ts: float


@dataclass
class SessionLog:
    """An append-only event log, optionally persisted to a JSONL file."""

    path: Path | None = None
    corrupt_policy: str = "skip"
    events: list[SessionEvent] = field(default_factory=list, init=False)  # type: ignore[assignment]
    corrupt_lines: int = field(default=0, init=False)  # type: ignore[assignment]

    SURFACE = frozenset({"user/message", "assistant/message", "tool/result"})

    def __post_init__(self) -> None:
        if self.path and self.path.exists():
            self._load(self.path)

    def _load(self, path: Path) -> None:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            event = self._parse_line(line, line_no)
            if event is not None:
                self.events.append(event)
        if self.corrupt_lines and self.corrupt_policy == "rename":
            renamed = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
            path.rename(renamed)
            get_logger().warning(
                "corrupt session renamed",
                extra={"original": str(renamed), "kept_events": len(self.events)},
            )

    def _parse_line(self, line: str, line_no: int) -> SessionEvent | None:
        try:
            parsed: object = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError("session row must be a JSON object")
            row_dict: dict[str, Any] = cast(dict[str, Any], parsed)
            event_type = row_dict["type"]
            data = row_dict["data"]
            ts = row_dict["ts"]
            if not isinstance(event_type, str) or not isinstance(data, dict):
                raise ValueError("bad record shape")
            if not isinstance(ts, (int, float)):
                raise ValueError("bad timestamp")
            data_dict: dict[str, Any] = cast(dict[str, Any], data)
            return SessionEvent(event_type, data_dict, float(ts))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.corrupt_lines += 1
            message = f"session line {line_no} is corrupt ({type(exc).__name__})"
            if self.corrupt_policy == "fail":
                raise SessionCorruptError(message) from exc
            get_logger().warning(
                "session line skipped",
                extra={"line": line_no, "policy": self.corrupt_policy},
            )
            return None

    def append(self, event_type: str, data: dict[str, Any]) -> SessionEvent:
        event = SessionEvent(event_type, copy.deepcopy(data), time.time())
        self.events.append(event)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps({"type": event.type, "data": event.data, "ts": event.ts})
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        return event

    def derive_messages(self) -> list[dict[str, str]]:
        """Project surface events into provider-visible chat messages."""
        messages: list[dict[str, str]] = []
        for event in self.events:
            if event.type == "user/message":
                messages.append({"role": "user", "content": event.data["content"]})
            elif event.type == "assistant/message":
                messages.append({"role": "assistant", "content": event.data["content"]})
            elif event.type == "tool/result":
                messages.append({"role": "tool", "content": event.data["content"]})
        return messages

    def tail(self, count: int = 10) -> list[SessionEvent]:
        return self.events[-count:]
