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
import json
import os
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from . import __version__
from .builtin_tools import BuiltinToolsPlugin
from .compaction import CompactionPlugin
from .config import (
    HarnessConfig,
    looks_like_snapshot,
    resolve_embed_model_path,
    resolve_model_path,
)
from .context import Context, PluginLoader
from .embeddings import EmbeddingProvider, EmbeddingsPlugin
from .errors import ConfigError, HarnessError, ProviderError
from .github import GitHubOptions, GitHubPlugin, GitHubProvider
from .identity.brand import CLI_ENTRY, ENV_PREFIX
from .llm import MockLLM, OpenAICompatibleGLM, TransformersGLM
from .logging import configure_logging, get_logger
from .loop import AgentLoop
from .plugins import BasePlugin, SafetyPlugin
from .repl import run_repl
from .sandbox import SandboxPlugin
from .server import run_server
from .session import SessionLog
from .skills import RiskLevel, SkillsPlugin
from .stop_slop import StopSlopEngine
from .tools import ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=CLI_ENTRY,
        description="Run GLM-5.3-Flash through the H1 plugin harness.",
    )
    parser.add_argument("prompt", nargs="?", help="one-shot user prompt (else prompts)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=f"sampling temperature for remote endpoints (default: ${ENV_PREFIX}TEMPERATURE or 0.7)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=f"path to a GLM-5.3-Flash snapshot (default: ${ENV_PREFIX}MODEL_PATH or repo root)",
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
        help="per-request timeout in seconds (default 300; 0 disables)",
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
    parser.add_argument(
        "--tool-allowlist",
        default=None,
        help="comma-separated tool names the model may call (default: all registered)",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="workspace root directory for filesystem and bash tools (default: cwd)",
    )
    parser.add_argument(
        "--sandbox",
        choices=("allow", "deny", "ask"),
        default=None,
        help="sandbox policy for mutating tools (allow, deny, ask; default: deny)",
    )
    parser.add_argument(
        "--enable-bash",
        dest="enable_bash",
        action="store_true",
        default=None,
        help=(
            "register the bash tool (default: enabled). The bash tool runs "
            "arbitrary shell commands inside the workspace at the operator's "
            "UID — pass --no-enable-bash to omit it."
        ),
    )
    parser.add_argument(
        "--no-enable-bash",
        dest="enable_bash",
        action="store_false",
        help="do not register the bash tool (overrides --enable-bash).",
    )
    parser.add_argument(
        "--no-enable-fetch-url",
        dest="enable_fetch_url",
        action="store_false",
        default=None,
        help="do not register the fetch_url tool (default: enabled).",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="base URL for OpenAI-compatible endpoint (e.g. http://127.0.0.1:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key / bearer token for remote endpoint",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model identifier for remote endpoint (default: GLM-5.3-Flash)",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="validate config, snapshot, and optional inference extra; do not run a turn",
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="start interactive terminal REPL session",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="start JSON-RPC 2.0 stdio protocol server",
    )
    parser.add_argument(
        "--compaction-threshold",
        type=int,
        default=None,
        help="token or turn threshold triggering automated compaction (0 to disable)",
    )
    parser.add_argument(
        "--compaction-keep-rounds",
        type=int,
        default=None,
        help="number of recent interaction rounds to retain uncompacted (default: 4)",
    )
    parser.add_argument(
        "--compaction-strategy",
        choices=("summarize", "truncate"),
        default=None,
        help="compaction strategy (summarize or truncate)",
    )
    parser.add_argument(
        "--replay-log",
        type=Path,
        default=None,
        help="path to external session JSONL log to replay/reconstruct history from",
    )
    parser.add_argument(
        "--projection",
        default=None,
        help="JSON array of messages [{'role': ..., 'content': ...}] to replay from",
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=None,
        help="path to directory containing external SKILL.md bundles",
    )
    parser.add_argument(
        "--skill-gate-tools",
        action="store_true",
        default=None,
        help="restrict tool execution to tools unlocked by actively triggered skills",
    )
    parser.add_argument(
        "--task-risk",
        choices=("low", "medium", "high", "critical"),
        default=None,
        help="risk boundary for task execution (default: low)",
    )
    parser.add_argument(
        "--slop-analyze",
        type=str,
        default=None,
        help="analyze input prose for AI patterns and output dimensional scores",
    )
    parser.add_argument(
        "--slop-rewrite",
        type=str,
        default=None,
        help="rewrite input prose to eliminate AI patterns, throat-clearing, and jargon",
    )
    parser.add_argument(
        "--slop-rules",
        nargs="?",
        const="all",
        default=None,
        help="display stop-slop rules, optionally filtered by category",
    )
    parser.add_argument(
        "--slop-examples",
        action="store_true",
        help="display stop-slop canonical before/after transformations",
    )
    parser.add_argument(
        "--embed-model-path",
        type=Path,
        default=None,
        help=(
            "path to a sentence-transformers snapshot (e.g. all-MiniLM-L6-v2); "
            f"default: ${ENV_PREFIX}EMBED_MODEL_PATH, then ~/all-MiniLM-L6-v2 when present"
        ),
    )
    parser.add_argument(
        "--rpc-token",
        type=str,
        default=None,
        help=(
            "bearer token required by the stdio JSON-RPC server's initialize "
            f"RPC; default: ${ENV_PREFIX}RPC_TOKEN. When unset, a per-process token is "
            "auto-generated and printed to stderr exactly once (set "
            "--rpc-auto-token=false to disable)."
        ),
    )
    parser.add_argument(
        "--rpc-auto-token",
        choices=("true", "false"),
        default=None,
        help=(
            "whether to auto-generate an RPC token when --rpc-token is unset "
            "(default: true). Set false to run --serve unauthenticated."
        ),
    )
    parser.add_argument(
        "--rpc-allow-mutating",
        action="store_true",
        default=None,
        help=(
            "allow mutating tool calls (bash, write_file, edit_file, "
            "github_write_file, github_create_*) over the JSON-RPC server. "
            "Default: false (denied)."
        ),
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help=(
            "root directory under which the JSON-RPC server may read "
            "session/import.logPath files. Relative paths only; absent this "
            "setting, logPath imports are refused."
        ),
    )
    parser.add_argument(
        "--embed",
        type=str,
        default=None,
        help="embed a text and print the vector as JSON, then exit",
    )
    parser.add_argument(
        "--similarity",
        nargs=2,
        metavar=("TEXT_A", "TEXT_B"),
        default=None,
        help="print cosine similarity between two texts, then exit",
    )
    parser.add_argument(
        "--github",
        type=str,
        default=None,
        help="scope GitHub tools to repo (owner/name format, e.g. octocat/Hello-World)",
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default=None,
        help=(
            "GitHub personal access token / bearer token "
            f"(defaults to $GITHUB_TOKEN or ${ENV_PREFIX}GITHUB_TOKEN)"
        ),
    )
    parser.add_argument(
        "--github-api-base",
        type=str,
        default=None,
        help="GitHub REST API base URL (default: https://api.github.com)",
    )
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
    "workspace": "workspace_dir",
    "sandbox": "sandbox_mode",
    "enable_bash": "enable_bash",
    "enable_fetch_url": "enable_fetch_url",
    "api_base": "api_base",
    "api_key": "api_key",
    "model": "model_name",
    "temperature": "temperature",
    "compaction_threshold": "compaction_threshold",
    "compaction_keep_rounds": "compaction_keep_rounds",
    "compaction_strategy": "compaction_strategy",
    "replay_log": "replay_log",
    "projection": "projection",
    "skills_dir": "skills_dir",
    "skill_gate_tools": "skill_gate_tools",
    "task_risk": "task_risk",
    "embed_model_path": "embed_model_path",
    "rpc_token": "rpc_token",
    "rpc_auto_token": "rpc_auto_token",
    "rpc_allow_mutating": "rpc_allow_mutating",
    "state_dir": "state_dir",
    "github": "github_repo",
    "github_token": "github_token",
    "github_api_base": "github_api_base",
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
    if args.tool_allowlist is not None:
        overrides["tool_allowlist"] = tuple(
            part.strip() for part in str(args.tool_allowlist).split(",") if part.strip()
        )
    # ``--rpc-auto-token`` is a tri-state choice (true|false|none). Coerce
    # only when the operator passed it explicitly; otherwise inherit the
    # config default (``True``).
    rpc_auto = getattr(args, "rpc_auto_token", None)
    if rpc_auto is not None:
        overrides["rpc_auto_token"] = rpc_auto == "true"
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
        if not sys.stdin.isatty():
            raise ConfigError("prompt is required (pass it as an argument)")
        try:
            config.prompt = input("you> ")
        except EOFError as exc:
            raise ConfigError("prompt is required (stdin closed)") from exc
    if not str(config.prompt).strip():
        raise ConfigError("prompt is empty")

    resolve_model_path(config)

    ctx = Context()
    sessions = SessionLog(path=config.session_path, corrupt_policy=config.corrupt_policy)
    if config.replay_log is not None:
        sessions.import_log(config.replay_log)
    if config.projection is not None:
        try:
            proj_data: object = json.loads(config.projection)
            if isinstance(proj_data, list):
                sessions.import_projection(cast(list[dict[str, Any]], proj_data))
        except json.JSONDecodeError:
            pass
    tools = ToolRegistry(ctx, tool_timeout_s=config.tool_timeout_s)
    if config.mock is not None:
        llm: object = MockLLM(config.mock)
    elif config.api_base is not None:
        llm = OpenAICompatibleGLM(
            api_base=config.api_base,
            api_key=config.api_key,
            model=config.model_name,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            timeout_s=config.request_timeout_s,
        )
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
        plugins: list[Any] = [
            BasePlugin(sessions, tools),
            BuiltinToolsPlugin(
                config.workspace_dir,
                enable_bash=config.enable_bash,
                enable_fetch_url=config.enable_fetch_url,
            ),
            SandboxPlugin(mode=config.sandbox_mode),  # type: ignore[arg-type]
            SafetyPlugin(config.tool_allowlist),
            CompactionPlugin(
                threshold=config.compaction_threshold,
                keep_rounds=config.compaction_keep_rounds,
                strategy=config.compaction_strategy,
            ),
            SkillsPlugin(
                skills_dir=config.skills_dir,
                gate_tools=config.skill_gate_tools,
                default_risk=RiskLevel.from_str(config.task_risk),
            ),
        ]
        if config.enable_bash:
            sys.stderr.write(
                "[glmharness] WARNING: the 'bash' tool is enabled — it executes "
                "arbitrary shell commands inside the workspace at your UID. "
                "Pass --no-enable-bash to disable it.\n"
            )
            sys.stderr.flush()
        embed_path = resolve_embed_model_path(config)
        if embed_path is not None:
            plugins.append(EmbeddingsPlugin(EmbeddingProvider(embed_path)))
        gh_token = config.github_token or os.environ.get("GITHUB_TOKEN")
        if config.github_repo or gh_token:
            owner, repo = None, None
            if config.github_repo and "/" in config.github_repo:
                parts = config.github_repo.split("/", 1)
                owner, repo = parts[0], parts[1]
            gh_opts = GitHubOptions(
                token=gh_token,
                api_base=config.github_api_base or "https://api.github.com",
                owner=owner,
                repo=repo,
            )
            plugins.append(GitHubPlugin(GitHubProvider(gh_opts)))
        await loader.mount(plugins)
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
        if isinstance(exc, HarnessError):
            logger.error("%s", exc)
        if isinstance(exc, ConfigError):
            return 2
        if isinstance(exc, ProviderError):
            return 3
        return 4
    finally:
        await ctx.close()


def doctor(config: HarnessConfig) -> int:
    """Print a pre-flight report. Exit 0 if the harness can run a turn."""
    configure_logging(fmt=config.log_format, level=config.log_level)
    rows: list[tuple[str, str, bool]] = []
    rows.append(("python", sys.version.split()[0], True))
    rows.append(("glmharness", __version__, True))
    try:
        config.validate()
        rows.append(("config", "ok", True))
    except ConfigError as exc:
        rows.append(("config", str(exc), False))
    if config.mock is not None:
        rows.append(("provider", f"mock ({config.mock!r})", True))
    elif config.api_base is not None:
        rows.append(
            (
                "provider",
                f"openai-compatible ({config.api_base}, model={config.model_name})",
                True,
            )
        )
    else:
        try:
            resolve_model_path(config)
        except ConfigError as exc:
            rows.append(("snapshot", str(exc), False))
        else:
            snapshot = config.model_path or Path.cwd()
            ok = looks_like_snapshot(snapshot)
            detail = str(snapshot.resolve())
            if not ok:
                detail = f"{snapshot} is missing config.json or tokenizer_config.json"
            rows.append(("snapshot", detail, ok))
        try:
            import transformers  # type: ignore[import-not-found]

            rows.append(("transformers", getattr(transformers, "__version__", "ok"), True))
        except ImportError:
            rows.append(
                ("transformers", "missing — pip install 'glmharness[inference]'", False)
            )
    if config.tool_allowlist:
        rows.append(("tool_allowlist", ",".join(config.tool_allowlist), True))
    embed_path = resolve_embed_model_path(config)
    if embed_path is not None:
        try:
            from glmharness.embeddings import EmbeddingProvider

            EmbeddingProvider(embed_path).require_snapshot()
        except ConfigError as exc:
            rows.append(("embeddings", str(exc), False))
        else:
            try:
                import sentence_transformers  # type: ignore[import-not-found]

                version = getattr(sentence_transformers, "__version__", "ok")
                rows.append(
                    ("embeddings", f"{embed_path.resolve()} (sentence-transformers {version})", True)
                )
            except ImportError:
                rows.append(
                    ("embeddings", f"{embed_path.resolve()} — missing 'glmharness[embeddings]'", False)
                )
    else:
        rows.append(
            ("embeddings", f"disabled (set {ENV_PREFIX}EMBED_MODEL_PATH or --embed-model-path)", True)
        )
    ws = (config.workspace_dir or Path.cwd()).resolve()
    rows.append(("workspace", str(ws), ws.is_dir()))
    rows.append(("sandbox", config.sandbox_mode, True))
    gh_token = config.github_token or os.environ.get("GITHUB_TOKEN")
    if config.github_repo or gh_token:
        scope = config.github_repo or "<unscoped>"
        auth_status = "authenticated" if gh_token else "unauthenticated"
        rows.append(("github", f"{scope} ({auth_status})", True))
    failed = False
    for name, detail, ok in rows:
        mark = "ok" if ok else "FAIL"
        sys.stdout.write(f"{mark:4}  {name}: {detail}\n")
        failed = failed or not ok
    if config.request_timeout_s <= 0:
        sys.stdout.write(
            "WARN  request_timeout: disabled (0); provider calls have no wall-clock bound\n"
        )
    return 2 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.slop_analyze is not None:
            engine = StopSlopEngine()
            report = engine.analyze(args.slop_analyze)
            sys.stdout.write(json.dumps(report, indent=2) + "\n")
            return 0 if report["passes_threshold"] else 1
        if args.slop_rewrite is not None:
            engine = StopSlopEngine()
            res = engine.rewrite(args.slop_rewrite)
            sys.stdout.write(res["rewritten"] + "\n")
            return 0
        if args.slop_rules is not None:
            engine = StopSlopEngine()
            cat = None if args.slop_rules == "all" else args.slop_rules
            rules = engine.get_rules(cat)
            sys.stdout.write(json.dumps(rules, indent=2) + "\n")
            return 0
        if args.slop_examples:
            engine = StopSlopEngine()
            examples = engine.get_examples()
            sys.stdout.write(json.dumps(examples, indent=2) + "\n")
            return 0
        if args.embed is not None or args.similarity is not None:
            embed_config = HarnessConfig()
            if args.embed_model_path is not None:
                embed_config.embed_model_path = args.embed_model_path
            embed_path = resolve_embed_model_path(embed_config)
            if embed_path is None:
                raise ConfigError(
                    f"no embedding model: set --embed-model-path (or {ENV_PREFIX}EMBED_MODEL_PATH)"
                )
            provider = EmbeddingProvider(embed_path)
            if args.embed is not None:
                vector = provider.embed([args.embed])[0]
                sys.stdout.write(json.dumps({"dim": len(vector), "vector": vector}) + "\n")
                return 0
            text_a, text_b = args.similarity
            score = provider.similarity(text_a, text_b)
            sys.stdout.write(json.dumps({"similarity": round(score, 4)}) + "\n")
            return 0

        config = _merge_config(args)
        if args.doctor:
            return doctor(config)
        if args.serve:
            return asyncio.run(run_server(config))
        if args.repl or (not args.prompt and sys.stdin.isatty()):
            return asyncio.run(run_repl(config))
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
