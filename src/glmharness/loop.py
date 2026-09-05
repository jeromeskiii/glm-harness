"""The agent loop: one-shot prompt -> tool-calling rounds -> final answer.

Contract (matches the H1 invariant "model-visible means logged"):

- every provider request is logged on the first round as ``request/header``
  and ``request/context`` (the reconstruction epoch);
- streamed chunks are buffered per attempt and committed to the log as
  ``assistant/chunk`` only when an attempt **completes** — a retried attempt
  never pollutes the append-only log with discarded tokens;
- a ``tool_calls`` answer does not close the turn: tools execute and the
  loop continues until the model stops or ``max_rounds`` is exhausted;
- failure and cancellation both close the turn with durable status markers.
"""

from __future__ import annotations

import asyncio
import copy
import uuid
from contextlib import nullcontext
from typing import Any

from .context import Context
from .errors import ProviderError
from .llm import LLM
from .logging import get_logger
from .session import SessionLog
from .tools import ToolRegistry, parse_tool_calls

#: exception types worth replaying (the request is side-effect free)
_RETRYABLE = (ConnectionError, TimeoutError, OSError)


class AgentLoop:
    def __init__(
        self,
        ctx: Context,
        llm: LLM,
        sessions: SessionLog,
        tools: ToolRegistry,
        *,
        max_rounds: int = 12,
        request_timeout_s: float = 0.0,
        max_retries: int = 2,
        retry_base_delay_s: float = 1.0,
        retry_max_delay_s: float = 30.0,
        retry_jitter: float = 0.25,
    ):
        self.ctx = ctx
        self.llm = llm
        self.sessions = sessions
        self.tools = tools
        self.max_rounds = max_rounds
        self.request_timeout_s = request_timeout_s
        self.max_retries = max_retries
        self.retry_base_delay_s = retry_base_delay_s
        self.retry_max_delay_s = retry_max_delay_s
        self.retry_jitter = retry_jitter

    async def run(self, prompt: str) -> str:
        self.sessions.append("turn/start", {})
        self.sessions.append("user/message", {"content": prompt})
        logger = get_logger()
        try:
            final_answer = ""
            for round_idx in range(self.max_rounds):
                messages = self.sessions.derive_messages()
                tools_schemas = self.tools.schemas()
                request: dict[str, Any] = {"messages": messages, "tools": tools_schemas}
                request = await self.ctx.events.dispatch("agent/request", "waterfall", request)

                if round_idx == 0:
                    # Reconstruction epoch: record exactly what is sent to
                    # the provider on the first turn.
                    request_id = str(uuid.uuid4())
                    self.sessions.append(
                        "request/header",
                        {"id": request_id, "message_count": len(request["messages"])},
                    )
                    self.sessions.append(
                        "request/context",
                        {"id": request_id, "request": copy.deepcopy(request)},
                    )

                self.sessions.append("step/start", {"round": round_idx + 1})
                answer = await self._stream_text(request["messages"], request.get("tools"))
                self.sessions.append("assistant/message", {"content": answer})

                tool_calls = parse_tool_calls(answer)
                if not tool_calls:
                    final_answer = answer
                    self.sessions.append(
                        "step/end", {"status": "completed", "rounds": round_idx + 1}
                    )
                    break

                for call in tool_calls:
                    await self.tools.execute(call["name"], call.get("arguments", {}))
                self.sessions.append(
                    "step/end",
                    {"status": "tools_executed", "rounds": round_idx + 1, "tools": len(tool_calls)},
                )
            else:
                final_answer = answer
                self.sessions.append(
                    "step/end", {"status": "max_rounds_reached", "rounds": self.max_rounds}
                )

            self.sessions.append("turn/end", {"status": "completed"})
            await self.ctx.events.dispatch(
                "session/flush", "serial", {"events": len(self.sessions.events)}
            )
            logger.info("turn completed", extra={"rounds": round_idx + 1})
            return final_answer
        except asyncio.CancelledError:
            self.sessions.append("interrupted", {"reason": "cancelled"})
            self.sessions.append("turn/end", {"status": "cancelled"})
            logger.warning("turn cancelled")
            raise
        except Exception as exc:
            self.sessions.append("step/end", {"status": "failed", "error": type(exc).__name__})
            self.sessions.append("turn/end", {"status": "failed"})
            logger.error("turn failed", extra={"error": type(exc).__name__})
            raise

    async def _stream_text(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None
    ) -> str:
        """Stream one provider request with retry; commit chunks on success."""
        logger = get_logger()
        for attempt in range(1, self.max_retries + 2):
            buffer: list[str] = []
            timeout_ctx = (
                asyncio.timeout(self.request_timeout_s)
                if self.request_timeout_s > 0
                else nullcontext()
            )
            try:
                async with timeout_ctx:
                    async for chunk in self.llm.stream(messages, tools):
                        buffer.append(chunk)
            except _RETRYABLE as exc:
                if attempt > self.max_retries:
                    raise
                await self._backoff(attempt, exc, logger)
                continue
            except ProviderError as exc:
                if not exc.retryable or attempt > self.max_retries:
                    raise
                await self._backoff(attempt, exc, logger)
                continue
            # Attempt completed: commit chunks to the append-only log.
            for chunk in buffer:
                self.sessions.append("assistant/chunk", {"content": chunk})
            return "".join(buffer)
        raise RuntimeError("unreachable: retry loop exhausted without a result")

    async def _backoff(self, attempt: int, exc: Exception, logger: Any) -> None:
        delay = self._retry_delay(attempt)
        self.sessions.append(
            "stream/retry",
            {"attempt": attempt, "error": type(exc).__name__, "delay_s": round(delay, 3)},
        )
        logger.warning(
            "stream failed; retrying",
            extra={"attempt": attempt, "error": type(exc).__name__, "delay_s": round(delay, 3)},
        )
        await asyncio.sleep(delay)

    def _retry_delay(self, attempt: int) -> float:
        """Exponential backoff with deterministic jitter."""
        delay = min(self.retry_base_delay_s * (2 ** (attempt - 1)), self.retry_max_delay_s)
        if self.retry_jitter:
            x = (attempt * 1103515245 + 12345) % (2**31)
            delay *= 1 + self.retry_jitter * (2 * x / (2**31) - 1)
        return delay
