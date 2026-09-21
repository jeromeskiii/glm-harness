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
from .wire.constants.session_constants import (
    DEFAULT_CORRUPT_POLICY,
    SESSION_SURFACE,
)


@dataclass(frozen=True)
class SessionEvent:
    type: str
    data: dict[str, Any]
    ts: float


@dataclass
class SessionLog:
    """An append-only event log, optionally persisted to a JSONL file."""

    path: Path | None = None
    corrupt_policy: str = DEFAULT_CORRUPT_POLICY
    events: list[SessionEvent] = field(default_factory=list, init=False)  # type: ignore[assignment]
    corrupt_lines: int = field(default=0, init=False)  # type: ignore[assignment]

    SURFACE = SESSION_SURFACE

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

    def import_projection(self, projection: list[dict[str, Any]]) -> int:
        """Import pre-turn or pre-switch message projection into the session log."""
        imported = 0
        for item in projection:
            role = str(item.get("role") or "").lower()
            content = str(item.get("content") or item.get("text") or "")
            if not role:
                continue

            event_type = f"{role}/message"
            if role == "tool":
                event_type = "tool/result"
            self.append(event_type, {"content": content})
            imported += 1
        return imported

    def import_log(self, source: Path | str | list[dict[str, Any]]) -> int:
        """Import events from an external session file or event array."""
        records: list[dict[str, Any]] = []
        if isinstance(source, (str, Path)):
            src_path = Path(source)
            if not src_path.exists():
                return 0
            lines = src_path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    parsed: object = json.loads(line_str)
                    if isinstance(parsed, dict):
                        records.append(cast(dict[str, Any], parsed))
                except json.JSONDecodeError:
                    continue
        else:
            records = list(source)

        imported = 0
        for rec in records:
            # Format 1: glmharness record {"type": ..., "data": ...}
            if "type" in rec and isinstance(rec.get("data"), dict):
                evt_type = str(rec["type"])
                evt_data = cast(dict[str, Any], rec["data"])
                self.append(evt_type, evt_data)
                imported += 1
            # Format 2: DMH host record {"kind": ..., "payload": ...}
            elif "kind" in rec and isinstance(rec.get("payload"), dict):
                kind = str(rec["kind"])
                payload = cast(dict[str, Any], rec["payload"])
                content = str(payload.get("content") or payload.get("text") or "")
                if kind in ("user/message", "assistant/message", "tool/result"):
                    self.append(kind, {"content": content})
                    imported += 1
            # Format 3: Projection item {"role": ..., "content": ...}
            elif "role" in rec:
                imported += self.import_projection([rec])
        return imported

    def derive_messages(self) -> list[dict[str, Any]]:
        """Project surface events into provider-visible chat messages with compaction support."""
        watermark_idx = -1
        summary_text: str | None = None

        for _idx, event in enumerate(self.events):
            if event.type in ("session/compacted", "compacted"):
                upto = event.data.get("upto")
                if isinstance(upto, int):
                    watermark_idx = max(watermark_idx, upto)
                    s = event.data.get("summary")
                    if isinstance(s, str):
                        summary_text = s

        messages: list[dict[str, str]] = []
        if watermark_idx >= 0 and summary_text:
            messages.append({"role": "user", "content": f"[compacted history] {summary_text}"})

        for idx, event in enumerate(self.events):
            if idx < watermark_idx:
                continue
            if event.type in ("user/message", "system/message"):
                role = "system" if event.type == "system/message" else "user"
                messages.append({"role": role, "content": str(event.data.get("content", ""))})
            elif event.type == "assistant/message":
                messages.append({"role": "assistant", "content": str(event.data.get("content", ""))})
            elif event.type == "tool/result":
                message: dict[str, Any] = {
                    "role": "tool",
                    "content": str(event.data.get("content", "")),
                }
                call_id = event.data.get("call_id")
                if isinstance(call_id, str) and call_id:
                    message["tool_call_id"] = call_id
                messages.append(message)
        return messages

    def tail(self, count: int = 10) -> list[SessionEvent]:
        return self.events[-count:]

