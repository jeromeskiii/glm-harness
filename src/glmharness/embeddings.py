"""Embedding service: a local sentence-transformers snapshot behind semantic tools.

``EmbeddingProvider`` lazily loads a local snapshot (e.g. ``all-MiniLM-L6-v2``)
through the optional ``[embeddings]`` extra. ``EmbeddingsPlugin`` mounts three
read-only tools — ``embed_text``, ``semantic_similarity``, and ``semantic_rank``
— and registers the provider in the context services so carriers (REPL, JSON-RPC
host) can reach it directly.

The model load is deliberately lazy: mock runs and remote-endpoint runs never
pay the import or load cost unless a semantic tool is actually called.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .context import Context
from .errors import ConfigError
from .tools import Tool, ToolRegistry

_MAX_TEXTS = 32
_MAX_CANDIDATES = 64


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors, numpy-free."""
    if len(a) != len(b):
        raise ValueError(f"vector dimension mismatch: {len(a)} != {len(b)}")
    if not a or not b:
        raise ValueError("empty vector")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingProvider:
    """Lazy loader over a local sentence-transformers snapshot.

    ``model_path`` must be a directory containing the sentence-transformers
    artifacts (``modules.json`` plus ``model.safetensors`` or
    ``pytorch_model.bin``), or a hub model id. Loading is deferred until the
    first call so the harness stays light when semantic tools are unused.
    """

    def __init__(self, model_path: Path | str):
        self.model_path = Path(model_path)
        self.model: Any = None

    def require_snapshot(self) -> None:
        """Fail fast when a local path lacks a loadable snapshot.

        Non-absolute paths that don't exist are treated as hub model ids and
        passed through to sentence-transformers for resolution.
        """
        if not self.model_path.is_dir():
            if not self.model_path.is_absolute() and not self.model_path.exists():
                return
            raise ConfigError(
                f"embedding model path is not a directory: {self.model_path}"
            )
        has_weights = (self.model_path / "model.safetensors").exists() or (
            self.model_path / "pytorch_model.bin"
        ).exists()
        has_config = (self.model_path / "modules.json").exists() or (
            self.model_path / "config_sentence_transformers.json"
        ).exists()
        if has_weights and has_config:
            return
        raise ConfigError(
            f"embedding snapshot is incomplete: {self.model_path} — expected "
            "modules.json (or config_sentence_transformers.json) plus "
            "model.safetensors (or pytorch_model.bin)"
        )

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            # sentence-transformers is an optional ``[embeddings]`` extra. We
            # type-check without it; the runtime import is what the user gets
            # when they install the optional dep.
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigError(
                "embeddings require the 'embeddings' extra: "
                "pip install 'glmharness[embeddings]'"
            ) from exc
        self.require_snapshot()
        self.model = SentenceTransformer(str(self.model_path))  # type: ignore[reportUnknownMemberType]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` into normalized vectors (``model.encode``)."""
        self._load()
        if not texts:
            raise ValueError("no texts to embed")
        vectors = self.model.encode(  # type: ignore[union-attr]
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return [vec.tolist() for vec in vectors]  # type: ignore[union-attr]

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts."""
        a, b = self.embed([text_a, text_b])
        return cosine_similarity(a, b)

    def rank(self, query: str, candidates: Sequence[str]) -> list[dict[str, Any]]:
        """Rank ``candidates`` against ``query``, best first."""
        if not candidates:
            raise ValueError("no candidates to rank")
        query_vec = self.embed([query])[0]
        ranked = [
            {
                "text": candidate,
                "score": round(cosine_similarity(query_vec, self.embed([candidate])[0]), 4),
            }
            for candidate in candidates
        ]
        ranked.sort(key=lambda item: item["score"], reverse=True)  # type: ignore[arg-type, return-value]
        return ranked


def make_embed_text_tool(provider: EmbeddingProvider) -> Tool:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "texts": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    f"Texts to embed (1-{_MAX_TEXTS} items). Returns one normalized "
                    "vector per text."
                ),
            },
        },
        "required": ["texts"],
        "additionalProperties": False,
    }

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        texts = [str(item) for item in args["texts"]]
        if len(texts) > _MAX_TEXTS:
            raise ValueError(f"too many texts: {len(texts)} > {_MAX_TEXTS}")
        vectors = provider.embed(texts)
        return {"dim": len(vectors[0]), "count": len(vectors), "embeddings": vectors}

    return Tool(
        name="embed_text",
        description=(
            "Embed texts into dense vectors using the local sentence-transformers "
            "model (all-MiniLM-L6-v2, 384-dim, normalized)."
        ),
        schema=schema,
        handler=handler,
    )


def make_semantic_similarity_tool(provider: EmbeddingProvider) -> Tool:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text_a": {"type": "string", "description": "First text to compare."},
            "text_b": {"type": "string", "description": "Second text to compare."},
        },
        "required": ["text_a", "text_b"],
        "additionalProperties": False,
    }

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        score = provider.similarity(str(args["text_a"]), str(args["text_b"]))
        return {"similarity": round(score, 4)}

    return Tool(
        name="semantic_similarity",
        description=(
            "Compute cosine similarity (0-1) between two texts using local "
            "embeddings."
        ),
        schema=schema,
        handler=handler,
    )


def make_semantic_rank_tool(provider: EmbeddingProvider) -> Tool:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The query text to rank against."},
            "candidates": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    f"Candidate texts to rank (1-{_MAX_CANDIDATES} items). Returns "
                    "them best-first with similarity scores."
                ),
            },
        },
        "required": ["query", "candidates"],
        "additionalProperties": False,
    }

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        candidates = [str(item) for item in args["candidates"]]
        if len(candidates) > _MAX_CANDIDATES:
            raise ValueError(f"too many candidates: {len(candidates)} > {_MAX_CANDIDATES}")
        return {"query": args["query"], "ranked": provider.rank(str(args["query"]), candidates)}

    return Tool(
        name="semantic_rank",
        description=(
            "Rank candidate texts by semantic similarity to a query using local "
            "embeddings; best match first."
        ),
        schema=schema,
        handler=handler,
    )


class EmbeddingsPlugin:
    """Mounts the semantic tool battery and registers the embedding provider."""

    id = "embeddings"

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    def apply(self, ctx: Context) -> None:
        ctx.provide("embeddings", self.provider)
        tools: ToolRegistry = ctx.get("tools")
        tools.register(make_embed_text_tool(self.provider))
        tools.register(make_semantic_similarity_tool(self.provider))
        tools.register(make_semantic_rank_tool(self.provider))
