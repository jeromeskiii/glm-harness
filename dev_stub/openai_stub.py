"""OpenAI-compatible dev stub for `glm-harness --api-base`.

Pure stdlib. Listens on 127.0.0.1:8777 (override with `--host`/`--port`).
Mimics the surface `OpenAICompatibleGLM` consumes:

- `GET  /v1/models`              — list models
- `POST /v1/chat/completions`    — streaming SSE or JSON

Behaviour:
- If `tools` present in the payload and the last assistant turn has not
  emitted a tool call yet, emit one `list_dir` tool call with
  `path="."`; otherwise emit a short text reply that quotes the last
  user message.
- SSE chunks use `data: <json>` lines, terminated by `data: [DONE]`.
- All requests are logged to stderr with a one-line summary.

This is dev-only. Do not expose past loopback. Do not run in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODEL_NAME = "GLM-5.3-Flash-stub"


def _sse(delta: dict[str, Any], finish_reason: str | None = None) -> str:
    chunk = {"choices": [{"delta": delta, "index": 0, "finish_reason": finish_reason}]}
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _sse_done() -> bytes:
    return b"data: [DONE]\n\n"


def _text_chunks(text: str) -> list[str]:
    # Stream the reply in ~24-char chunks so the consumer sees SSE ticks.
    out: list[str] = []
    step = 24
    for i in range(0, len(text), step):
        out.append(_sse({"content": text[i : i + step]}))
    out.append(_sse({}, finish_reason="stop"))
    return out


def _tool_call_chunks(name: str, arguments: dict[str, Any], call_id: str = "call_stub_1") -> list[str]:
    """Stream a single tool call as the GLM adapter expects to accumulate."""
    args_str = json.dumps(arguments, ensure_ascii=False)
    return [
        _sse(
            {
                "role": "assistant",
                "tool_calls": [
                    {"index": 0, "id": call_id, "function": {"name": name, "arguments": args_str}}
                ],
            }
        ),
        _sse({}, finish_reason="tool_calls"),
    ]


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                return "\n".join(p for p in parts if p)
    return ""


def _last_assistant_had_tool(messages: list[dict[str, Any]]) -> bool:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return bool(m.get("tool_calls"))
        if m.get("role") == "tool":
            return True
    return False


def _reply(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
    if tools and not _last_assistant_had_tool(messages):
        return "TOOL_CALL"  # sentinel — caller emits tool-call chunks
    user = _last_user_text(messages).strip() or "(empty prompt)"
    short = user.replace("\n", " ")[:160]
    return f"[stub] received: {short}"


class Handler(BaseHTTPRequestHandler):
    server_version = "OpenAIStub/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        sys.stderr.write(f"[stub] {self.address_string()} {fmt % args}\n")
        sys.stderr.flush()

    def _write_json(self, status: int, body: Any) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _write_sse(self, chunks: list[str]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(_sse_done())
        self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/v1/models"):
            self._write_json(
                200,
                {
                    "object": "list",
                    "data": [{"id": MODEL_NAME, "object": "model", "created": int(time.time())}],
                },
            )
            return
        if self.path in ("/", "/healthz"):
            self._write_json(200, {"ok": True, "model": MODEL_NAME})
            return
        self._write_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith("/v1/chat/completions"):
            self._write_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except json.JSONDecodeError:
            self._write_json(400, {"error": "invalid json"})
            return

        messages = body.get("messages") or []
        tools = body.get("tools")
        stream = bool(body.get("stream", False))

        reply_text = _reply(messages, tools)

        if reply_text == "TOOL_CALL":
            chunks = _tool_call_chunks("list_dir", {"path": "."})
        else:
            chunks = _text_chunks(reply_text)

        if stream:
            self._write_sse(chunks)
            return

        # Non-streaming fallback for any consumer that opts out of SSE.
        if reply_text == "TOOL_CALL":
            self._write_json(
                200,
                {
                    "id": "stub-1",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": MODEL_NAME,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_stub_1",
                                        "type": "function",
                                        "function": {
                                            "name": "list_dir",
                                            "arguments": json.dumps({"path": "."}),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
            )
            return

        self._write_json(
            200,
            {
                "id": "stub-1",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": reply_text},
                    }
                ],
            },
        )


def main() -> int:
    p = argparse.ArgumentParser(description="OpenAI-compatible dev stub for glm-harness.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8777)
    args = p.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    sys.stderr.write(f"[stub] listening on http://{args.host}:{args.port}/v1\n")
    sys.stderr.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("[stub] shutting down\n")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())