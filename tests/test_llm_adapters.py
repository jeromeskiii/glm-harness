"""Tests for LLM adapters without loading real model weights."""

from __future__ import annotations

import builtins as _bi
import json
import sys

import pytest

from glmharness import MockLLM
from glmharness.errors import ConfigError


async def test_mock_llm_yields_response() -> None:
    pieces = []
    async for chunk in MockLLM("hello world").stream([]):
        pieces.append(chunk)
    assert "".join(pieces) == "hello world"


def test_transformers_glm_load_raises_when_transformers_missing(tmp_path, monkeypatch) -> None:
    """If transformers is unimportable, the adapter surfaces a ConfigError."""
    from glmharness import TransformersGLM

    adapter = TransformersGLM(tmp_path, "max", 8)
    real_import = _bi.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("forced missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_bi, "__import__", fake_import)
    sys.modules.pop("transformers", None)
    with pytest.raises(ConfigError, match="requires the 'inference' extra"):
        adapter._load()


def test_transformers_glm_load_is_idempotent_when_model_already_loaded(tmp_path, monkeypatch) -> None:
    """A second ``_load`` call is a no-op once ``model`` is populated."""
    from glmharness import TransformersGLM

    adapter = TransformersGLM(tmp_path, "max", 8)
    adapter.model = object()  # type: ignore[assignment]
    adapter.tokenizer = object()  # type: ignore[assignment]
    # No import attempted; the function returns immediately.
    assert adapter._load() is None


def test_transformers_glm_load_uses_multimodal_stack(tmp_path, monkeypatch) -> None:
    """GLM-5.3-Flash is an image-text-to-text model: load via the I2T stack."""
    import types as _types

    from glmharness import TransformersGLM

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a.b": "model-00001-of-00001.safetensors"}})
    )
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"")
    calls: list[tuple[str, ...]] = []

    class _FakeTokenizer:
        pass

    class _FakeProcessor:
        tokenizer = _FakeTokenizer()

        @classmethod
        def from_pretrained(cls, path, trust_remote_code=False):
            calls.append(("processor", str(path)))
            return cls()

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, path, device_map=None, trust_remote_code=False, torch_dtype=None):
            calls.append(("model", str(path), device_map))
            return cls()

    fake_transformers = _types.ModuleType("transformers")
    fake_transformers.AutoProcessor = _FakeProcessor
    fake_transformers.AutoModelForImageTextToText = _FakeModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    adapter = TransformersGLM(tmp_path, "max", 8)
    adapter._load()

    assert adapter.processor is not None
    assert adapter.tokenizer is adapter.processor.tokenizer
    assert ("processor", str(tmp_path)) in calls
    assert ("model", str(tmp_path), "auto") in calls


def test_transformers_glm_load_fails_fast_without_weights(tmp_path, monkeypatch) -> None:
    """A config-only snapshot raises an actionable ConfigError before loading."""
    import types as _types

    from glmharness import TransformersGLM

    fake_transformers = _types.ModuleType("transformers")
    fake_transformers.AutoProcessor = object()
    fake_transformers.AutoModelForImageTextToText = object()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    adapter = TransformersGLM(tmp_path, "max", 8)
    with pytest.raises(ConfigError, match="snapshot has no model weights"):
        adapter._load()


class _MockResponse:
    def __init__(self, lines: list[str]):
        self.lines = [line.encode("utf-8") for line in lines]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self):
        return iter(self.lines)


async def test_openai_glm_streams_text() -> None:
    from glmharness import OpenAICompatibleGLM

    lines = [
        'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
        'data: {"choices": [{"delta": {"content": " world"}}]}\n',
        "data: [DONE]\n",
    ]

    def fake_opener(req, timeout):
        return _MockResponse(lines)

    adapter = OpenAICompatibleGLM(opener=fake_opener)
    chunks: list[str] = []
    async for chunk in adapter.stream([{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    assert "".join(chunks) == "Hello world"


async def test_openai_glm_zero_timeout_uses_unset_socket_timeout() -> None:
    from glmharness import OpenAICompatibleGLM

    observed: list[object] = []

    def fake_opener(req, timeout):
        observed.append(timeout)
        return _MockResponse(["data: [DONE]\n"])

    adapter = OpenAICompatibleGLM(opener=fake_opener, timeout_s=0)
    async for _ in adapter.stream([{"role": "user", "content": "hi"}]):
        pass

    assert observed == [None]


async def test_openai_glm_streams_and_formats_tool_calls() -> None:
    from glmharness import OpenAICompatibleGLM, parse_tool_calls

    lines = [
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "read_file"}}]}}]}\n',
        (
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, '
            '"function": {"arguments": "{\\"path\\": "}}]}}]}\n'
        ),
        (
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, '
            '"function": {"arguments": "\\"foo.py\\"}"}}]}}]}\n'
        ),
        "data: [DONE]\n",
    ]

    def fake_opener(req, timeout):
        return _MockResponse(lines)

    adapter = OpenAICompatibleGLM(opener=fake_opener)
    chunks: list[str] = []
    async for chunk in adapter.stream([{"role": "user", "content": "read foo"}]):
        chunks.append(chunk)

    full_output = "".join(chunks)
    assert "<tool_call>read_file" in full_output
    calls = parse_tool_calls(full_output)
    assert len(calls) == 1
    assert calls[0]["name"] == "read_file"
    assert calls[0]["arguments"] == {"path": "foo.py"}


async def test_openai_glm_http_error() -> None:
    import urllib.error

    from glmharness import OpenAICompatibleGLM, ProviderError

    def fake_opener(req, timeout):
        raise urllib.error.HTTPError("http://test", 429, "Too Many Requests", {}, None)  # type: ignore[arg-type]

    adapter = OpenAICompatibleGLM(opener=fake_opener)
    with pytest.raises(ProviderError) as exc_info:
        async for _ in adapter.stream([{"role": "user", "content": "hi"}]):
            pass
    assert exc_info.value.retryable is True

    def fake_opener_401(req, timeout):
        raise urllib.error.HTTPError("http://test", 401, "Unauthorized", {}, None)  # type: ignore[arg-type]

    adapter_401 = OpenAICompatibleGLM(opener=fake_opener_401)
    with pytest.raises(ProviderError) as exc_info_401:
        async for _ in adapter_401.stream([{"role": "user", "content": "hi"}]):
            pass
    assert exc_info_401.value.retryable is False
