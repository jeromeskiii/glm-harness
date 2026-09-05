"""Tests for LLM adapters without loading real model weights."""

from __future__ import annotations

import builtins as _bi
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
