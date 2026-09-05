"""Opt-in integration check for ``TransformersGLM``'s interface.

Run with ``GLMH_RUN_INTEGRATION=1`` (and the ``[inference]`` extra
installed). This does **not** load the full model — the vendored
snapshot's tokenizer needs additional native deps (sentencepiece /
tiktoken) that aren't in the project extra. We only assert that the
adapter's surface is consistent with the version of ``transformers``
the operator has installed, which catches breakage from signature
changes between major versions.
"""

from __future__ import annotations

import os

import pytest

_RUN = os.environ.get("GLMH_RUN_INTEGRATION") == "1"

try:
    import transformers  # noqa: F401

    _HAVE = True
except ImportError:
    _HAVE = False

pytestmark = pytest.mark.skipif(
    not _RUN or not _HAVE,
    reason="integration test; set GLMH_RUN_INTEGRATION=1 with [inference] installed",
)


def test_transformers_glm_class_shape() -> None:
    """``TransformersGLM`` exposes the attributes the harness relies on."""
    from glmharness import TransformersGLM

    assert callable(TransformersGLM.__init__)
    assert callable(TransformersGLM.stream)
    assert callable(TransformersGLM._load)
    assert callable(TransformersGLM._maybe_clear_memory)


def test_transformers_text_streamer_is_importable() -> None:
    """The ``TextIteratorStreamer`` import resolves when transformers is present."""
    from transformers import TextIteratorStreamer  # type: ignore[import-not-found]

    assert callable(TextIteratorStreamer)
