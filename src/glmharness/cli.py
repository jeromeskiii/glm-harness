"""Command-line entry point.

``glm-harness`` runs a one-shot prompt end-to-end. Configuration layers:

1. built-in defaults,
2. ``GLMH_*`` environment variables (see :mod:`glmharness.config`),
3. CLI flags (override envs).

Diagnostics go to stderr; only the final answer is written to stdout so the
harness composes with pipes.

Exit codes:

- 0: success
- 2: configuration error
- 3: provider failure
- 4: tool/pipeline failure
- 130: cancelled (Ctrl-C)
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import signal
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import HarnessConfig
from .context import Context, PluginLoader
from .errors import ConfigError, HarnessError, ProviderError
from .llm import MockLLM, TransformersGLM
from .logging import configure_logging, get_logger
from .loop import AgentLoop
from .plugins import BasePlugin
from .session import SessionLog
from .tools import ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glm-harness",
        description="Run GLM-5.3-Flash through the H1 plugin harness.",
    )
    parser.add_argument("prompt", nargs="?", help="one-shot user prompt (else prompts)")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="path to a GLM-5.3-Flash snapshot (default: $GLMH_MODEL_PATH or repo root)",
    )
    parser.add_argument(
        "--session",
        type=Path,
        default=None,
        help="append-only JSONL session log",
    )
    parser.add_argument(
        "--mock",
        default=None,
        help="use a deterministic response instead of loading the model",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "high", "max"),
        default=None,
        help="reasoning budget; default 'max'",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=None,
        help="per-request timeout in seconds (0 disables, default 0)",
    )
    parser.add_argument("--tool-timeout-s", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument(
        "--corrupt-policy",
        choices=("skip", "rename", "fail"),
        default=None,
        help="how to treat a corrupt session log on load",
    )
    parser.add_argument("--log-format", choices=("text", "json"), default=None)
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


_CLI_FIELD_FOR = {
    "model_path": "model_path",
    "session": "session_path",
    "mock": "mock",
    "reasoning_effort": "reasoning_effort",
    "max_new_tokens": "max_new_tokens",
    "max_rounds": "max_rounds",
    "request_timeout_s": "request_timeout_s",
    "tool_timeout_s": "tool_timeout_s",
    "max_retries": "max_retries",
    "corrupt_policy": "corrupt_policy",
    "log_format": "log_format",
    "log_level": "log_level",
}


def _merge_config(args: argparse.Namespace) -> HarnessConfig:
    """Apply CLI overrides on top of :class:`HarnessConfig.from_env` results."""
    config = HarnessConfig.from_env()
    overrides: dict[str, object] = {}
    for flag, target in _CLI_FIELD_FOR.items():
        value = getattr(args, flag, None)
        if value is not None:
            overrides[target] = value
    if args.prompt is not None:
        overrides["prompt"] = args.prompt
    # ``dataclasses.replace`` keeps the static-type contract: every key is
    # validated against the field declaration rather than routed through
    # ``Any`` like ``HarnessConfig(**dict)`` would.
    merged = dataclasses.replace(config, **overrides)
    merged.validate()
    return merged


async def run(config: HarnessConfig) -> int:
    """Execute one turn under ``config``. Returns a process exit code."""
    configure_logging(fmt=config.log_format, level=config.log_level)
    logger = get_logger()
    for key in config.unknown_env_keys():
        logger.warning("unknown env var; ignoring", extra={"env": key})

    if not config.prompt:
        config.prompt = input("you> ")

    if config.mock is None and config.model_path is None:
        raise ConfigError(
            "either --mock or a model path (--model-path / GLMH_MODEL_PATH) is required"
        )

    ctx = Context()
    sessions = SessionLog(path=config.session_path, corrupt_policy=config.corrupt_policy)
    tools = ToolRegistry(ctx, tool_timeout_s=config.tool_timeout_s)
    if config.mock is not None:
        llm: object = MockLLM(config.mock)
    else:
        assert config.model_path is not None
        llm = TransformersGLM(
            config.model_path,
            reasoning_effort=config.reasoning_effort,
            max_new_tokens=config.max_new_tokens,
        )

    loop = asyncio.get_running_loop()
    cancelled = asyncio.Event()

    def _on_signal(signame: str) -> None:
        if not cancelled.is_set():
            cancelled.set()
            logger.warning("signal received; shutting down", extra={"signal": signame})

    for signame in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(getattr(signal, signame), _on_signal, signame)
        except (NotImplementedError, RuntimeError):
            pass

    try:
        loader = PluginLoader(ctx)
        await loader.mount([BasePlugin(sessions, tools)])
        agent = AgentLoop(
            ctx,
            llm,  # type: ignore[arg-type]
            ctx.get("sessions"),
            ctx.get("tools"),
            max_rounds=config.max_rounds,
            request_timeout_s=config.request_timeout_s,
            max_retries=config.max_retries,
            retry_base_delay_s=config.retry_base_delay_s,
            retry_max_delay_s=config.retry_max_delay_s,
            retry_jitter=config.retry_jitter,
        )

        async def _cancel_when_set() -> None:
            await cancelled.wait()

        cancel_task = asyncio.create_task(_cancel_when_set())
        run_task = asyncio.create_task(agent.run(config.prompt))
        done, _ = await asyncio.wait(
            {run_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if run_task in done and not run_task.cancelled() and run_task.exception() is None:
            sys.stdout.write(run_task.result() + "\n")
            sys.stdout.flush()
            return 0
        if cancel_task in done:
            run_task.cancel()
            try:
                await run_task
            except (asyncio.CancelledError, HarnessError):
                pass
            return 130
        exc = run_task.exception()
        if isinstance(exc, ConfigError):
            return 2
        if isinstance(exc, ProviderError):
            return 3
        return 4
    finally:
        await ctx.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _merge_config(args)
        return asyncio.run(run(config))
    except ConfigError as exc:
        sys.stderr.write(f"config error: {exc}\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        return 130


def entry() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entry()
