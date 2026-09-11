"""Tests for the embeddings service and semantic tools (no real model load)."""

from __future__ import annotations

import math
import sys
import types as _types

import pytest

from glmharness import (
    EmbeddingProvider,
    EmbeddingsPlugin,
    cosine_similarity,
    make_embed_text_tool,
    make_semantic_rank_tool,
    make_semantic_similarity_tool,
)
from glmharness.context import Context
from glmharness.errors import ConfigError
from glmharness.tools import ToolRegistry


class _Vec(list):
    """Minimal numpy-array stand-in: the provider calls ``.tolist()``."""

    def tolist(self) -> list[float]:
        return list(self)


def _char_vector(text: str) -> list[float]:
    """Deterministic normalized 4-dim vector from a string's characters."""
    v = [0.0, 0.0, 0.0, 0.0]
    for i, ch in enumerate(text):
        v[i % 4] += float(ord(ch))
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class _FakeSentenceTransformer:
    def __init__(self, path: str):
        self.path = path

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        return [_Vec(_char_vector(t)) for t in texts]


def _install_fake_sentence_transformers(monkeypatch) -> None:
    fake = _types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)


def _snapshot(tmp_path):
    (tmp_path / "modules.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"")
    return tmp_path


def test_cosine_math() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([3, 4], [3, 4]) == pytest.approx(1.0)
    assert cosine_similarity([1, 1], [1, -1]) == pytest.approx(0.0)


def test_cosine_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="empty vector"):
        cosine_similarity([], [])


def test_provider_requires_snapshot(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not a directory"):
        EmbeddingProvider(tmp_path / "missing").require_snapshot()
    with pytest.raises(ConfigError, match="incomplete"):
        EmbeddingProvider(tmp_path).require_snapshot()
    _snapshot(tmp_path)
    EmbeddingProvider(tmp_path).require_snapshot()


def test_provider_passes_hub_ids_through() -> None:
    EmbeddingProvider("sentence-transformers/all-MiniLM-L6-v2").require_snapshot()


def test_provider_missing_extra_raises_actionable_error(tmp_path, monkeypatch) -> None:
    import builtins as _bi

    _snapshot(tmp_path)
    real_import = _bi.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("forced missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_bi, "__import__", fake_import)
    sys.modules.pop("sentence_transformers", None)
    provider = EmbeddingProvider(tmp_path)
    with pytest.raises(ConfigError, match="embeddings' extra"):
        provider.embed(["hello"])


def test_provider_embed_similarity_rank(tmp_path, monkeypatch) -> None:
    _install_fake_sentence_transformers(monkeypatch)
    _snapshot(tmp_path)
    provider = EmbeddingProvider(tmp_path)

    vectors = provider.embed(["hello", "world"])
    assert len(vectors) == 2
    assert all(len(v) == 4 for v in vectors)
    assert all(sum(x * x for x in v) == pytest.approx(1.0, rel=1e-6) for v in vectors)

    score = provider.similarity("hello", "hello")
    assert score == pytest.approx(1.0, rel=1e-6)
    assert -1.0 <= provider.similarity("hello", "zzzz") <= 1.0

    ranked = provider.rank("cat", ["cat food", "quantum physics", "cat"])
    assert len(ranked) == 3
    scores = [item["score"] for item in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_embed_tool_handler_and_limit(tmp_path, monkeypatch) -> None:
    _install_fake_sentence_transformers(monkeypatch)
    _snapshot(tmp_path)
    tool = make_embed_text_tool(EmbeddingProvider(tmp_path))

    result = tool.handler({"texts": ["a", "b"]})
    assert result["dim"] == 4
    assert result["count"] == 2
    assert len(result["embeddings"]) == 2

    with pytest.raises(ValueError, match="too many texts"):
        tool.handler({"texts": ["x"] * 33})


def test_similarity_tool_handler(tmp_path, monkeypatch) -> None:
    _install_fake_sentence_transformers(monkeypatch)
    _snapshot(tmp_path)
    tool = make_semantic_similarity_tool(EmbeddingProvider(tmp_path))

    result = tool.handler({"text_a": "same", "text_b": "same"})
    assert result["similarity"] == pytest.approx(1.0, rel=1e-6)


def test_rank_tool_handler_and_limit(tmp_path, monkeypatch) -> None:
    _install_fake_sentence_transformers(monkeypatch)
    _snapshot(tmp_path)
    tool = make_semantic_rank_tool(EmbeddingProvider(tmp_path))

    result = tool.handler({"query": "q", "candidates": ["a", "b", "c"]})
    assert len(result["ranked"]) == 3
    scores = [item["score"] for item in result["ranked"]]
    assert scores == sorted(scores, reverse=True)

    with pytest.raises(ValueError, match="too many candidates"):
        tool.handler({"query": "q", "candidates": ["x"] * 65})


def test_embeddings_plugin_mounts_tools_and_service(tmp_path, monkeypatch) -> None:
    _install_fake_sentence_transformers(monkeypatch)
    _snapshot(tmp_path)
    ctx = Context()
    registry = ToolRegistry(ctx)
    ctx.provide("tools", registry)

    EmbeddingsPlugin(EmbeddingProvider(tmp_path)).apply(ctx)

    assert ctx.services.get("embeddings") is not None
    names = set(registry.tools)
    assert {"embed_text", "semantic_similarity", "semantic_rank"} <= names
