"""Session token and message compaction for long-running agent workflows."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .context import Context
from .logging import get_logger
from .session import SessionLog


def estimate_tokens(text: str) -> int:
    """Heuristic token estimation (~4 characters per token)."""
    return max(1, len(text) // 4)


def compact_session(
    session_log: SessionLog,
    *,
    threshold: int = 4000,
    keep_rounds: int = 4,
    strategy: str = "summarize",
) -> dict[str, Any]:
    """Compact older events in the session log when history breaches threshold.

    Parameters
    ----------
    session_log:
        Target session log.
    threshold:
        Token budget (or message count if threshold < 100). If 0 or negative,
        compaction is disabled.
    keep_rounds:
        Number of recent interaction messages to retain uncompacted.
    strategy:
        Compaction strategy ('summarize' or 'truncate').

    Returns
    -------
    Dictionary describing compaction results.
    """
    if threshold <= 0:
        return {"compacted": False, "dropped_count": 0, "summary": None, "upto": None}

    latest_watermark = max(
        (int(ev.data["upto"]) for ev in session_log.events
         if ev.type in ("session/compacted", "compacted") and isinstance(ev.data.get("upto"), int)),
        default=-1,
    )
    # Identify only surface events that have not already been compacted.
    surface_events: list[tuple[int, str, str]] = []
    total_tokens = 0
    for idx, ev in enumerate(session_log.events):
        if idx > latest_watermark and ev.type in (
            "user/message",
            "assistant/message",
            "tool/result",
            "system/message",
        ):
            content = str(ev.data.get("content", ""))
            tokens = estimate_tokens(content)
            total_tokens += tokens
            surface_events.append((idx, ev.type, content))

    # Determine if threshold is breached (either token budget or message count if < 100)
    limit = threshold
    metric = total_tokens if threshold >= 100 else len(surface_events)

    if metric <= limit:
        return {"compacted": False, "dropped_count": 0, "summary": None, "upto": None}

    # Ensure there are enough events to compact beyond the retained window
    if len(surface_events) <= keep_rounds + 1:
        return {"compacted": False, "dropped_count": 0, "summary": None, "upto": None}

    # We keep the first prompt (surface_events[0]) and the latest keep_rounds events
    to_compact = surface_events[1:-keep_rounds]
    if not to_compact:
        return {"compacted": False, "dropped_count": 0, "summary": None, "upto": None}

    upto_idx = surface_events[-keep_rounds][0]
    dropped_count = len(to_compact)

    if strategy == "truncate":
        summary = f"Compacted {dropped_count} prior interaction turns."
    else:
        # Build concise structured summary
        snippets: list[str] = []
        for _, ev_type, content in to_compact:
            speaker = ev_type.split("/")[0]
            clean = " ".join(content.split())
            if len(clean) > 80:
                clean = clean[:77] + "..."
            snippets.append(f"{speaker}: {clean}")
        summary = "Compacted conversation history: " + " | ".join(snippets)

    session_log.append(
        "session/compacted",
        {
            "upto": upto_idx,
            "dropped_count": dropped_count,
            "summary": summary,
            "strategy": strategy,
        },
    )

    get_logger().info(
        "session compacted",
        extra={
            "upto": upto_idx,
            "dropped_count": dropped_count,
            "token_estimate": total_tokens,
            "threshold": threshold,
        },
    )

    return {
        "compacted": True,
        "dropped_count": dropped_count,
        "summary": summary,
        "upto": upto_idx,
    }


class CompactionPlugin:
    """Plugin enforcing automated session compaction before model requests."""

    id = "harness-compaction"

    def __init__(
        self,
        threshold: int = 0,
        keep_rounds: int = 4,
        strategy: str = "summarize",
    ) -> None:
        self.threshold = threshold
        self.keep_rounds = keep_rounds
        self.strategy = strategy
        self.ctx: Context | None = None

    def apply(self, ctx: Context) -> None:
        self.ctx = ctx
        ctx.on("agent/request", self._on_agent_request)

    async def _on_agent_request(
        self,
        payload: dict[str, Any],
        next_: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        if self.threshold > 0 and self.ctx is not None:
            sessions_svc: object = self.ctx.services.get("sessions")
            if isinstance(sessions_svc, SessionLog):
                res = compact_session(
                    sessions_svc,
                    threshold=self.threshold,
                    keep_rounds=self.keep_rounds,
                    strategy=self.strategy,
                )
                if res.get("compacted"):
                    payload["messages"] = sessions_svc.derive_messages()
        if next_ is not None:
            return await next_(payload)
        return payload
