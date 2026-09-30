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
import json
import urllib.error
import urllib.request
from collections.abc import AsyncIterator
from pathlib import Path
from threading import Thread
from typing import Any, Protocol, cast

from .errors import ConfigError, ProviderError
from .identity.brand import CLI_ENTRY
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


def _format_openai_tool_calls(calls: list[dict[str, Any]]) -> str:
    """Translate accumulated OpenAI tool-call objects into GLM <tool_call> XML blocks."""
    blocks: list[str] = []
    for call in calls:
        func_raw = call.get("function")
        func = cast(dict[str, Any], func_raw) if isinstance(func_raw, dict) else {}
        name = str(func.get("name", ""))
        raw_args = func.get("arguments", "{}")
        try:
            parsed_args: object = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            parsed_args = {}
        parts = [f"<tool_call>{name}"]
        if isinstance(parsed_args, dict):
            args_dict = cast(dict[str, Any], parsed_args)
            for k, v in args_dict.items():
                val_str = json.dumps(v, ensure_ascii=False)
                parts.append(f"<arg_key>{k}</arg_key><arg_value>{val_str}</arg_value>")
        parts.append("</tool_call>")
        blocks.append("".join(parts))
    return "\n".join(blocks)


class OpenAICompatibleGLM:
    """Remote OpenAI-compatible streaming LLM adapter (vLLM, SGLang, Ollama, API).

    Zero external dependencies; uses urllib.request in a background feeder thread.
    """

    def __init__(
        self,
        api_base: str = "http://127.0.0.1:8000/v1",
        api_key: str | None = None,
        model: str = "GLM-5.3-Flash",
        max_new_tokens: int = 8192,
        temperature: float = 0.7,
        timeout_s: float = 300.0,
        opener: Any = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._opener = opener or urllib.request.urlopen

    async def stream(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[str]:
        url = f"{self.api_base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | BaseException | None] = asyncio.Queue()
        box: dict[str, BaseException | None] = {"error": None}

        def post(item: str | BaseException | None) -> None:
            asyncio.run_coroutine_threadsafe(queue.put(item), loop)

        def _fetch() -> None:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            accumulated_tools: dict[int, dict[str, Any]] = {}
            try:
                # ``0`` disables the application-level request deadline. The
                # urllib contract uses ``None`` for an unset socket timeout;
                # passing 0 raises ValueError instead of disabling it.
                open_timeout = self.timeout_s if self.timeout_s > 0 else None
                with self._opener(req, timeout=open_timeout) as resp:
                    for raw_line in resp:
                        raw_str = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
                        line = raw_str.strip()
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk_raw: object = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(chunk_raw, dict):
                            continue
                        chunk_dict = cast(dict[str, Any], chunk_raw)
                        choices_raw = chunk_dict.get("choices")
                        if not isinstance(choices_raw, list):
                            continue
                        for choice_item in cast(list[object], choices_raw):
                            if not isinstance(choice_item, dict):
                                continue
                            choice = cast(dict[str, Any], choice_item)
                            delta_raw = choice.get("delta")
                            if not isinstance(delta_raw, dict):
                                continue
                            delta = cast(dict[str, Any], delta_raw)
                            content = delta.get("content")
                            if isinstance(content, str) and content:
                                post(content)
                            tool_calls_raw = delta.get("tool_calls")
                            if isinstance(tool_calls_raw, list):
                                for tc_item in cast(list[object], tool_calls_raw):
                                    if not isinstance(tc_item, dict):
                                        continue
                                    tc = cast(dict[str, Any], tc_item)
                                    idx = int(tc.get("index", 0))
                                    if idx not in accumulated_tools:
                                        accumulated_tools[idx] = {
                                            "id": str(tc.get("id", "")),
                                            "function": {"name": "", "arguments": ""},
                                        }
                                    fn_raw = tc.get("function")
                                    if isinstance(fn_raw, dict):
                                        fn = cast(dict[str, Any], fn_raw)
                                        fn_name = fn.get("name")
                                        fn_args = fn.get("arguments")
                                        target_fn = cast(
                                            dict[str, str],
                                            accumulated_tools[idx]["function"],
                                        )
                                        if isinstance(fn_name, str):
                                            target_fn["name"] += fn_name
                                        if isinstance(fn_args, str):
                                            target_fn["arguments"] += fn_args
                if accumulated_tools:
                    tool_list = [accumulated_tools[k] for k in sorted(accumulated_tools.keys())]
                    formatted = _format_openai_tool_calls(tool_list)
                    if formatted:
                        post(formatted)
            except urllib.error.HTTPError as exc:
                retryable = exc.code in (429, 500, 502, 503, 504)
                err = ProviderError(f"API HTTP {exc.code}: {exc.reason}", retryable=retryable)
                box["error"] = err
                post(err)
            except urllib.error.URLError as exc:
                err = ProviderError(f"API connection error: {exc.reason}", retryable=True)
                box["error"] = err
                post(err)
            except Exception as exc:
                err = ProviderError(f"API request failed: {exc}", retryable=True)
                box["error"] = err
                post(err)
            finally:
                post(None)

        feeder = Thread(target=_fetch, daemon=True, name="openai-glm-stream")
        feeder.start()

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
            if box["error"] is not None:
                raise box["error"]
        finally:
            feeder.join(timeout=5)


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
        self.processor: Any = None
        self.model: Any = None
        self.tokenizer: Any = None

    def _require_shards(self) -> None:
        """Fail fast with an actionable error when the snapshot has no weights.

        The vendored repo is a config-only snapshot. The full FP8 checkpoint is
        ~330 GB on disk and dequantizes to ~660 GB at load, so operator-class
        laptops must use a served endpoint instead.
        """
        index = self.model_path / "model.safetensors.index.json"
        try:
            data = json.loads(index.read_text())
            shards = sorted({self.model_path / name for name in data.get("weight_map", {}).values()})
        except (OSError, ValueError):
            shards = sorted(self.model_path.glob("*.safetensors"))
        if shards and all(p.exists() for p in shards):
            return
        if shards:
            missing = [p.name for p in shards if not p.exists()]
            detail = f"{len(missing)} missing shard(s), e.g. {missing[0]}"
        else:
            detail = "no safetensors shards found"
        raise ConfigError(
            f"snapshot has no model weights: {detail}. The full FP8 checkpoint "
            "needs ~330 GB disk and ~660 GB RAM once dequantized; run against a "
            f"served endpoint instead: {CLI_ENTRY} --api-base http://HOST:PORT/v1 'prompt'"
        )

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            # transformers is an optional ``[inference]`` extra. We type-check
            # without it; the runtime import is what the user gets when they
            # install the optional dep. Each downstream use is annotated
            # with the appropriate type-ignore comment below.
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForImageTextToText,
                AutoProcessor,
            )
        except ImportError as exc:
            raise ConfigError(
                "local mode requires the 'inference' extra: "
                "pip install 'glmharness[inference]'"
            ) from exc
        self._require_shards()
        self.processor = AutoProcessor.from_pretrained(  # type: ignore[reportUnknownMemberType]
            self.model_path, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer  # type: ignore[reportUnknownMemberType]
        self.model = AutoModelForImageTextToText.from_pretrained(  # type: ignore[reportUnknownMemberType]
            self.model_path, device_map="auto", trust_remote_code=True, torch_dtype="auto"
        )

    def _maybe_clear_memory(self) -> None:
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():  # type: ignore[reportUnknownMemberType]
                torch.cuda.empty_cache()  # type: ignore[reportUnknownMemberType]
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[reportUnknownMemberType]
                torch.mps.empty_cache()  # type: ignore[reportUnknownMemberType]
        except Exception:
            pass

    async def stream(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[str]:
        self._load()
        prompt = self.processor.apply_chat_template(  # type: ignore[union-attr]
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
        generation_kwargs: dict[str, Any] = {
            **inputs,  # type: ignore[arg-type]
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
                import torch  # type: ignore[import-not-found]

                with torch.inference_mode():  # type: ignore[reportUnknownMemberType]
                    self.model.generate(**generation_kwargs)  # type: ignore[union-attr]
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
