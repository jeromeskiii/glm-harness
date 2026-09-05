"""LLM seam: the provider protocol plus adapters.

``MockLLM`` serves deterministic responses for tests and offline smoke runs.
``TransformersGLM`` adapts a local GLM-5.3-Flash snapshot. Prompt assembly,
logging, retries, and timeouts are harness-owned; the adapter only streams.

The thread bridge is deliberately simple and race-free:

- the generation thread runs ``model.generate`` under ``inference_mode``;
- a feeder thread drains ``TextIteratorStreamer`` and posts tokens to an
  asyncio queue;
- both threads share an error box that is written **before** the stream ends,
  so the consumer can always learn about a generation failure after the
  ``done`` sentinel, regardless of thread interleaving.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from threading import Thread
from typing import Any, Protocol

from .errors import ConfigError, ProviderError
from .logging import get_logger


class LLM(Protocol):
    def stream(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None
    ) -> AsyncIterator[str]: ...


class MockLLM:
    """Deterministic single-response provider. ``stream`` yields once."""

    def __init__(self, response: str):
        self.response = response

    async def stream(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[str]:
        yield self.response


class TransformersGLM:
    """Thin adapter over a local GLM snapshot with real streaming.

    ``trust_remote_code`` stays ``True`` because the GLM-5 series ships its
    chat template as remote code; operators pin their snapshot by path.
    """

    def __init__(
        self,
        model_path: Path,
        reasoning_effort: str = "max",
        max_new_tokens: int = 8192,
    ):
        self.model_path = model_path
        self.reasoning_effort = reasoning_effort
        self.max_new_tokens = max_new_tokens
        self.model: Any = None
        self.tokenizer: Any = None

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            # transformers is an optional ``[inference]`` extra. We type-check
            # without it; the runtime import is what the user gets when they
            # install the optional dep. Each downstream use is annotated
            # with the appropriate type-ignore comment below.
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigError(
                "local mode requires the 'inference' extra: "
                "pip install 'glmharness[inference]'"
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[reportUnknownMemberType]
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(  # type: ignore[reportUnknownMemberType]
            self.model_path, device_map="auto", trust_remote_code=True, torch_dtype="auto"
        )

    def _maybe_clear_memory(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    async def stream(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[str]:
        self._load()
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
            clear_thinking=True,
            reasoning_effort=self.reasoning_effort,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        from transformers import TextIteratorStreamer  # type: ignore[import-not-found]

        streamer: TextIteratorStreamer = TextIteratorStreamer(  # type: ignore[reportUnknownMemberType]
            self.tokenizer, skip_prompt=True, skip_special_tokens=True  # type: ignore[reportUnknownMemberType]
        )
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": self.max_new_tokens,
        }

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | BaseException | None] = asyncio.Queue()
        # Written before the stream ends; safe for the consumer to check
        # after the done sentinel (both writer threads share this box).
        box: dict[str, BaseException | None] = {"error": None}

        def post(item: str | BaseException | None) -> None:
            asyncio.run_coroutine_threadsafe(queue.put(item), loop)

        def _generate() -> None:
            try:
                import torch

                with torch.inference_mode():
                    self.model.generate(**generation_kwargs)
            except Exception as exc:
                box["error"] = exc
                post(exc)
            finally:
                post(None)

        generation_thread = Thread(target=_generate, daemon=True, name="glm-generate")
        generation_thread.start()

        def _feed() -> None:
            try:
                # TextIteratorStreamer.__iter__ yields str; the transformers
                # stub leaves the element type as Any, so both the for-loop
                # variable and the post() argument need type-ignore.
                for text in streamer:  # type: ignore[reportUnknownVariableType, reportUnknownArgumentType]
                    post(text)  # type: ignore[arg-type]
            except Exception as exc:
                box["error"] = exc
                post(exc)
            finally:
                post(None)

        feeder_thread = Thread(target=_feed, daemon=True, name="glm-feed")
        feeder_thread.start()

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
            if box["error"] is not None:
                raise ProviderError(
                    f"generation failed: {type(box['error']).__name__}: {box['error']}"
                ) from box["error"]
        finally:
            generation_thread.join(timeout=30)
            feeder_thread.join(timeout=5)
            if generation_thread.is_alive() or feeder_thread.is_alive():
                get_logger().warning(
                    "generation threads still running after join timeout",
                    extra={
                        "generation_alive": generation_thread.is_alive(),
                        "feeder_alive": feeder_thread.is_alive(),
                    },
                )
            self._maybe_clear_memory()
