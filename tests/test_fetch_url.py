"""Unit tests for the fetch_url tool."""

from __future__ import annotations

import io
from email.message import EmailMessage
from typing import Any
from urllib.error import HTTPError
from urllib.response import addinfourl

import pytest

from glmharness import Context, SessionLog, ToolRegistry
from glmharness.builtin_tools import (
    BuiltinToolsPlugin,
    clean_html_to_markdown,
    make_fetch_url_tool,
)


def _make_mock_response(body: bytes, content_type: str = "text/html", status: int = 200) -> Any:
    msg = EmailMessage()
    msg["Content-Type"] = content_type
    fp = io.BytesIO(body)
    return addinfourl(fp, msg, "https://example.com/docs", code=status)


def test_clean_html_to_markdown() -> None:
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>API Reference</title><style>body { color: red; }</style></head>
    <body>
        <nav><a href="/home">Home</a></nav>
        <h1>Welcome to the API</h1>
        <p>This is a <b>fast</b> and <i>efficient</i> tool.</p>
        <pre><code>def get_data(): return 42</code></pre>
        <script>alert("evil");</script>
    </body>
    </html>
    """
    title, text = clean_html_to_markdown(html)
    assert title == "API Reference"
    assert "Welcome to the API" in text
    assert "This is a fast and efficient tool." in text
    assert "def get_data(): return 42" in text
    assert "alert" not in text
    assert "color: red" not in text


async def test_fetch_url_html_success(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_html = (
        b"<html><head><title>Test Page</title></head>"
        b"<body><h1>Hello World</h1><p>Documentation text here.</p></body></html>"
    )

    def mock_urlopen(req, timeout=15):
        return _make_mock_response(sample_html, content_type="text/html; charset=utf-8")

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    tool = make_fetch_url_tool()
    res = await ToolRegistry._invoke(tool, {"url": "https://example.com/docs"})
    assert res["status_code"] == 200
    assert res["title"] == "Test Page"
    assert "Hello World" in res["content"]
    assert "Documentation text here." in res["content"]
    assert res["truncated"] is False


async def test_fetch_url_json_success(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_json = b'{"status": "ok", "items": [1, 2, 3]}'

    def mock_urlopen(req, timeout=15):
        return _make_mock_response(sample_json, content_type="application/json")

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    tool = make_fetch_url_tool()
    res = await ToolRegistry._invoke(tool, {"url": "https://api.example.com/v1/data"})
    assert res["status_code"] == 200
    assert res["title"] is None
    assert '"status": "ok"' in res["content"]


async def test_fetch_url_rejects_unsupported_schemes() -> None:
    tool = make_fetch_url_tool()

    with pytest.raises(ValueError, match="unsupported URL scheme"):
        await ToolRegistry._invoke(tool, {"url": "file:///etc/passwd"})

    with pytest.raises(ValueError, match="unsupported URL scheme"):
        await ToolRegistry._invoke(tool, {"url": "ftp://ftp.example.com/data"})


async def test_fetch_url_ssrf_protection() -> None:
    tool = make_fetch_url_tool()

    with pytest.raises(PermissionError, match="blocked target address"):
        await ToolRegistry._invoke(tool, {"url": "http://127.0.0.1:8000/secret"})

    with pytest.raises(PermissionError, match="blocked target address"):
        await ToolRegistry._invoke(tool, {"url": "http://localhost:8080/admin"})

    with pytest.raises(PermissionError, match="blocked target address"):
        await ToolRegistry._invoke(tool, {"url": "http://169.254.169.254/latest/meta-data"})


async def test_fetch_url_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_urlopen(req, timeout=15):
        raise HTTPError("https://example.com/404", 404, "Not Found", {}, io.BytesIO(b"Not Found"))

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    tool = make_fetch_url_tool()
    res = await ToolRegistry._invoke(tool, {"url": "https://example.com/missing"})
    assert res["status_code"] == 404
    assert "Not Found" in res["content"]


async def test_fetch_url_mounted_in_plugin(tmp_path) -> None:
    ctx = Context()
    tools = ToolRegistry(ctx)
    ctx.provide("tools", tools)
    ctx.provide("sessions", SessionLog())

    plugin = BuiltinToolsPlugin(workspace=tmp_path)
    plugin.apply(ctx)

    assert "fetch_url" in tools.tools
