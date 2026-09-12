"""End-to-end checks for the remote OpenAI-compatible path over a real socket.

The supported real-model path is ``glm-harness --api-base <url>``. ``Phase 8``
verified it against an ad-hoc local stub that was never committed, so the path
had no reproducible coverage: the adapter unit tests in ``test_llm_adapters``
inject a fake opener and never touch a socket, a thread, or the HTTP carrier.

These tests stand the stub up in-process and drive the full CLI — config
merge, plugin mount, skill discovery, agent loop, and tool dispatch — through
an actual TCP connection. No model weights and no external service needed.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from glmharness.cli import main


class _StubState:
    """Scripted SSE responses plus a record of the requests that consumed them."""

    def __init__(self, script: list[list[dict[str, object]]]) -> None:
        self.script = script
        self.requests: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def next_chunks(self, request: dict[str, object]) -> list[dict[str, object]]:
        with self._lock:
            index = len(self.requests)
            self.requests.append(request)
        return self.script[index] if index < len(self.script) else []


class _StubHandler(BaseHTTPRequestHandler):
    """Speaks just enough of the OpenAI streaming protocol for one turn."""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        parsed: object = json.loads(raw or b"{}")
        request = parsed if isinstance(parsed, dict) else {}
        chunks = self.server.stub.next_chunks(request)  # type: ignore[attr-defined]

        payload = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        payload += "data: [DONE]\n\n"
        body = payload.encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Silence the stdlib access log; the test asserts on state instead."""


class _StubEndpoint:
    """A localhost OpenAI-compatible endpoint replaying a scripted response list."""

    def __init__(self, script: list[list[dict[str, object]]]) -> None:
        self.state = _StubState(script)
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self._server.stub = self.state  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def api_base(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> _StubEndpoint:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _text_chunks(text: str) -> list[dict[str, object]]:
    mid = len(text) // 2
    return [
        {"choices": [{"delta": {"content": text[:mid]}}]},
        {"choices": [{"delta": {"content": text[mid:]}}]},
    ]


def test_remote_path_streams_answer_over_real_socket(capsys, tmp_path) -> None:
    """A one-shot turn reaches the endpoint and its streamed text hits stdout."""
    with _StubEndpoint([_text_chunks("Hello from the stub")]) as endpoint:
        code = main(["--api-base", endpoint.api_base, "--workspace", str(tmp_path), "say hello"])

    captured = capsys.readouterr()
    assert code == 0
    assert "Hello from the stub" in captured.out

    sent = endpoint.state.requests[0]
    assert sent["stream"] is True
    assert sent["model"] == "GLM-5.3-Flash"
    messages = sent["messages"]
    assert isinstance(messages, list)
    assert any(isinstance(message, dict) and message.get("role") == "user" for message in messages)
    tools = sent["tools"]
    assert isinstance(tools, list) and tools, "the tool battery must be advertised to the endpoint"


def test_remote_path_executes_tool_call_and_completes(capsys, tmp_path) -> None:
    """A tool call assembled from streamed deltas runs, and its result is fed back."""
    (tmp_path / "alpha.txt").write_text("hello from alpha\n", encoding="utf-8")
    script: list[list[dict[str, object]]] = [
        [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "list_dir"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"path"'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ': "."}'}}]}}]},
        ],
        _text_chunks("done: saw the workspace"),
    ]

    with _StubEndpoint(script) as endpoint:
        code = main(["--api-base", endpoint.api_base, "--workspace", str(tmp_path), "list the workspace"])

    captured = capsys.readouterr()
    assert code == 0
    assert "done: saw the workspace" in captured.out

    assert len(endpoint.state.requests) == 2, "the loop must call back once per tool round"
    followup = json.dumps(endpoint.state.requests[1])
    assert "alpha.txt" in followup, "the tool result must be sent back to the endpoint"
    assert "not a directory" not in followup, "the tool call must have executed cleanly"
