"""Harness configuration: dataclass + env-var loading + validation.

Every knob is settable three ways, in increasing precedence:

1. defaults in :class:`HarnessConfig`,
2. ``GLMH_*`` environment variables (``GLMH_MAX_ROUNDS=20``),
3. explicit CLI flags.

Validation is centralized in :meth:`HarnessConfig.validate` so a bad value
fails at startup with a clear message — never mid-turn.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ConfigError
from .identity.brand import ENV_PREFIX

_ENV_PREFIX = ENV_PREFIX

#: env var name -> (attribute, converter, allowed values)
_ENV_MAP: dict[str, tuple[str, str, tuple[str, ...] | None]] = {
    "MODEL_PATH": ("model_path", "path", None),
    "SESSION": ("session_path", "path", None),
    "MOCK": ("mock", "str", None),
    "REASONING_EFFORT": ("reasoning_effort", "str", ("low", "high", "max")),
    "MAX_NEW_TOKENS": ("max_new_tokens", "int", None),
    "MAX_ROUNDS": ("max_rounds", "int", None),
    "TOOL_TIMEOUT_S": ("tool_timeout_s", "float", None),
    "REQUEST_TIMEOUT_S": ("request_timeout_s", "float", None),
    "TURN_TIMEOUT_S": ("turn_timeout_s", "float", None),
    "MAX_SESSIONS": ("max_sessions", "int", None),
    "MAX_RETRIES": ("max_retries", "int", None),
    "RETRY_BASE_DELAY_S": ("retry_base_delay_s", "float", None),
    "RETRY_MAX_DELAY_S": ("retry_max_delay_s", "float", None),
    "RETRY_JITTER": ("retry_jitter", "float", None),
    "LOG_FORMAT": ("log_format", "str", ("text", "json")),
    "LOG_LEVEL": ("log_level", "str", None),
    "CORRUPT_POLICY": ("corrupt_policy", "str", ("skip", "rename", "fail")),
    "TOOL_ALLOWLIST": ("tool_allowlist", "csv", None),
    "WORKSPACE": ("workspace_dir", "path", None),
    "SANDBOX": ("sandbox_mode", "str", ("allow", "deny", "ask")),
    "API_BASE": ("api_base", "str", None),
    "API_KEY": ("api_key", "str", None),
    "MODEL": ("model_name", "str", None),
    "TEMPERATURE": ("temperature", "float", None),
    "COMPACTION_THRESHOLD": ("compaction_threshold", "int", None),
    "COMPACTION_KEEP_ROUNDS": ("compaction_keep_rounds", "int", None),
    "COMPACTION_STRATEGY": ("compaction_strategy", "str", ("summarize", "truncate")),
    "REPLAY_LOG": ("replay_log", "path", None),
    "PROJECTION": ("projection", "str", None),
    "SKILLS_DIR": ("skills_dir", "path", None),
    "SKILL_GATE_TOOLS": ("skill_gate_tools", "bool", None),
    "TASK_RISK": ("task_risk", "str", ("low", "medium", "high", "critical")),
    "EMBED_MODEL_PATH": ("embed_model_path", "path", None),
    "RPC_TOKEN": ("rpc_token", "str", None),
    "RPC_ALLOW_MUTATING": ("rpc_allow_mutating", "bool", None),
    "RPC_AUTO_TOKEN": ("rpc_auto_token", "bool", None),
    "STATE_DIR": ("state_dir", "path", None),
    "GITHUB_TOKEN": ("github_token", "str", None),
    "GITHUB_REPO": ("github_repo", "str", None),
    "GITHUB_API_BASE": ("github_api_base", "str", None),
}


@dataclass
class HarnessConfig:
    """All runtime knobs with production-safe defaults."""

    # model
    model_path: Path | None = None
    mock: str | None = None
    reasoning_effort: str = "max"
    max_new_tokens: int = 8192

    # remote OpenAI-compatible endpoint
    api_base: str | None = None
    api_key: str | None = None
    model_name: str = "GLM-5.3-Flash"
    # sampling temperature for remote endpoints (local TransformersGLM uses its
    # own reasoning-budget sampling); lower values stabilize tool-calling.
    temperature: float = 0.7

    # session
    session_path: Path | None = None
    corrupt_policy: str = "skip"

    # loop
    max_rounds: int = 12
    request_timeout_s: float = 300.0
    tool_timeout_s: float = 30.0
    # Whole-turn deadline for server ``agent/send`` (covers all rounds and
    # retries). 3600s = max_rounds x request_timeout_s, the legitimate worst
    # case; a turn without a budget could otherwise hold the server ~forever
    # if a provider wedges outside every per-request timeout. 0 disables.
    turn_timeout_s: float = 3600.0

    # retry (exponential backoff with jitter)
    max_retries: int = 2
    retry_base_delay_s: float = 1.0
    retry_max_delay_s: float = 30.0
    retry_jitter: float = 0.25

    # observability
    log_format: str = "text"
    log_level: str = "INFO"

    # safety: empty means every registered tool is eligible
    tool_allowlist: tuple[str, ...] = ()

    # sandbox & workspace
    workspace_dir: Path | None = None
    # Mutating tools are denied unless the operator explicitly opts in.
    sandbox_mode: str = "deny"
    # Built-in tool toggles (CLI surface). Default to True to keep the
    # legacy behavior; the CLI prints a stderr warning whenever bash is on
    # so the operator knows the harness can run shell commands.
    enable_bash: bool = True
    enable_fetch_url: bool = True

    # compaction & replay
    compaction_threshold: int = 0
    compaction_keep_rounds: int = 4
    compaction_strategy: str = "summarize"
    replay_log: Path | None = None
    projection: str | None = None

    # skills & dynamic triggers
    skills_dir: Path | None = None
    skill_gate_tools: bool = False
    task_risk: str = "low"

    # embeddings (optional; semantic tools mount only when a model resolves)
    embed_model_path: Path | None = None

    # stdio JSON-RPC server auth: when ``rpc_token`` is unset and ``rpc_auto_token``
    # is true (default), the server generates a random per-process token and prints
    # it to stderr exactly once. Set ``rpc_token`` to share a token with a peer;
    # set ``rpc_auto_token=False`` only when the peer is trusted-by-same-UID
    # (e.g. an IDE extension launching the server itself). Stdio is loopback,
    # but other local processes with the same UID can still write to the fd.
    rpc_token: str | None = None
    rpc_auto_token: bool = True
    # Allow mutating tool calls (bash, write_file, edit_file, github_write_file,
    # github_create_*, github_add_issue_comment) over the RPC. Default False
    # so a compromised peer cannot escalate to local code execution or remote
    # commit/push. Set True only when the peer is fully trusted.
    rpc_allow_mutating: bool = False
    # Root directory under which session/import RPC may read ``logPath`` files.
    # Defaults to None which means only relative paths are accepted.
    state_dir: Path | None = None

    # Cap on in-memory sessions held by the protocol server. When exceeded,
    # the oldest non-default session is evicted (with a warning) — server
    # memory must be bounded for months-long stdio hosting. Only ``default``
    # is persisted to ``session_path``; dynamic sessions are RAM-only.
    max_sessions: int = 256

    # GitHub seam: when token or repo is configured, GitHubPlugin mounts tools
    github_token: str | None = None
    github_repo: str | None = None
    github_api_base: str | None = None

    # runtime (not from env)
    prompt: str = ""
    # Populated by :meth:`from_env`; defaults to empty so direct construction
    # works without it. The factory returns a fresh dict per instance so the
    # field annotation stays accurate without a mutable default.
    unknown_env: dict[str, str] = field(default_factory=dict, repr=False)  # type: ignore[assignment]

    @classmethod
    def from_env(cls) -> HarnessConfig:
        """Build a config from ``GLMH_*`` environment variables."""
        # Build the config with explicit default values, then layer env-var
        # overrides onto each known field individually so the static type
        # stays accurate (no ``cls(**dict[str, object])`` round-trip).
        config = cls()
        for name, (attr, kind, allowed) in _ENV_MAP.items():
            raw = os.environ.get(_ENV_PREFIX + name)
            if raw is None and name.startswith("GITHUB_"):
                # Conventional bare names (GITHUB_TOKEN, GITHUB_REPO,
                # GITHUB_API_BASE) are honored alongside GLMH_GITHUB_*.
                raw = os.environ.get(name)
            if raw is None or raw == "":
                continue
            try:
                if kind == "int":
                    value: int | float | Path | str | tuple[str, ...] | bool = int(raw)
                elif kind == "float":
                    value = float(raw)
                elif kind == "path":
                    value = Path(raw)
                elif kind == "csv":
                    value = tuple(part.strip() for part in raw.split(",") if part.strip())
                elif kind == "bool":
                    normalized = raw.strip().lower()
                    if normalized in ("1", "true", "yes", "on"):
                        value = True
                    elif normalized in ("0", "false", "no", "off"):
                        value = False
                    else:
                        raise ConfigError(f"{_ENV_PREFIX}{name} must be bool: {raw!r}")
                else:
                    value = raw
            except ValueError as exc:
                raise ConfigError(f"{_ENV_PREFIX}{name} must be {kind}: {raw!r}") from exc
            if allowed is not None and value not in allowed:
                raise ConfigError(
                    f"{_ENV_PREFIX}{name} must be one of {allowed}: {raw!r}"
                )
            setattr(config, attr, value)
        for name in os.environ:
            if name.startswith(_ENV_PREFIX) and name[len(_ENV_PREFIX) :] not in _ENV_MAP:
                config.unknown_env[name] = os.environ[name]
        config.validate()
        return config

    def validate(self) -> None:
        if self.reasoning_effort not in ("low", "high", "max"):
            raise ConfigError(
                f"reasoning_effort must be low|high|max: {self.reasoning_effort!r}"
            )
        if self.max_new_tokens <= 0:
            raise ConfigError("max_new_tokens must be positive")
        if self.max_rounds < 1:
            raise ConfigError("max_rounds must be >= 1")
        if self.max_retries < 0:
            raise ConfigError("max_retries must be >= 0")
        if self.log_format not in ("text", "json"):
            raise ConfigError(f"log_format must be text|json: {self.log_format!r}")
        if self.corrupt_policy not in ("skip", "rename", "fail"):
            raise ConfigError(
                f"corrupt_policy must be skip|rename|fail: {self.corrupt_policy!r}"
            )
        if self.request_timeout_s < 0 or self.tool_timeout_s < 0:
            raise ConfigError("request_timeout_s and tool_timeout_s must be >= 0")
        if self.turn_timeout_s < 0:
            raise ConfigError("turn_timeout_s must be >= 0")
        if self.max_sessions < 1:
            raise ConfigError("max_sessions must be >= 1")
        if not 0 <= self.retry_jitter < 1:
            raise ConfigError("retry_jitter must be in [0, 1)")
        if self.retry_base_delay_s <= 0 or self.retry_max_delay_s < self.retry_base_delay_s:
            raise ConfigError("retry_base_delay_s must be > 0 and <= retry_max_delay_s")
        if not (0.0 <= self.temperature <= 2.0):
            raise ConfigError(f"temperature must be within [0.0, 2.0], got {self.temperature!r}")
        if self.model_path is not None and not self.model_path.is_dir():
            raise ConfigError(f"model path is not a directory: {self.model_path}")
        if self.sandbox_mode not in ("allow", "deny", "ask"):
            raise ConfigError(f"sandbox_mode must be allow|deny|ask: {self.sandbox_mode!r}")
        if self.workspace_dir is not None and not self.workspace_dir.is_dir():
            raise ConfigError(f"workspace path is not a directory: {self.workspace_dir}")
        if self.task_risk not in ("low", "medium", "high", "critical"):
            raise ConfigError(f"task_risk must be low|medium|high|critical: {self.task_risk!r}")
        if self.skills_dir is not None and not self.skills_dir.is_dir():
            raise ConfigError(f"skills path is not a directory: {self.skills_dir}")
        if self.embed_model_path is not None and not self.embed_model_path.is_dir():
            raise ConfigError(f"embedding model path is not a directory: {self.embed_model_path}")
        if self.github_repo is not None and "/" not in self.github_repo:
            raise ConfigError(f"github_repo must be in 'owner/name' format, got {self.github_repo!r}")
        if self.state_dir is not None and not self.state_dir.is_dir():
            raise ConfigError(f"state_dir path is not a directory: {self.state_dir}")

    def retry_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter for the given 1-based attempt.

        Jitter uses the OS CSPRNG (``secrets``) so concurrent retriers do not
        wake at the same instant; a deterministic LCG would amplify retry
        storms against the same provider.
        """
        delay = min(self.retry_base_delay_s * (2 ** (attempt - 1)), self.retry_max_delay_s)
        if self.retry_jitter:
            spread = 2 * secrets.randbelow(2**31) / (2**31) - 1
            delay *= 1 + self.retry_jitter * spread
        return delay

    def unknown_env_keys(self) -> list[str]:
        """``GLMH_*`` variables that map to nothing (typo guard)."""
        return sorted(self.unknown_env)


def looks_like_snapshot(path: Path) -> bool:
    """True when ``path`` looks like a GLM-5.3-Flash transformers snapshot."""
    return (path / "config.json").is_file() and (path / "tokenizer_config.json").is_file()


def resolve_model_path(config: HarnessConfig, *, cwd: Path | None = None) -> HarnessConfig:
    """Fill ``model_path`` from cwd when it is a snapshot and nothing else is set.

    Mock runs and remote API runs never need a snapshot. An explicit ``model_path`` wins. Otherwise
    a checkout of this repository (cwd containing ``config.json`` +
    ``tokenizer_config.json``) is the documented default.
    """
    if config.mock is not None or config.api_base is not None or config.model_path is not None:
        return config
    here = cwd if cwd is not None else Path.cwd()
    if looks_like_snapshot(here):
        config.model_path = here
        return config
    raise ConfigError(
        f"either --mock, --api-base ({ENV_PREFIX}API_BASE), or a model path "
        f"(--model-path / {ENV_PREFIX}MODEL_PATH) is required (cwd is not a GLM snapshot)"
    )


def resolve_embed_model_path(config: HarnessConfig, *, cwd: Path | None = None) -> Path | None:
    """Resolve the embedding model directory, or ``None`` to disable the tools.

    An explicit ``embed_model_path`` (env or flag) wins. When unset, a local
    ``all-MiniLM-L6-v2`` snapshot in the home directory is picked up as a
    documented convenience; anything else disables the semantic tool battery.
    """
    if config.embed_model_path is not None:
        return config.embed_model_path
    home_snapshot = Path.home() / "all-MiniLM-L6-v2"
    if home_snapshot.is_dir():
        return home_snapshot
    here = cwd if cwd is not None else Path.cwd()
    local_snapshot = here / "all-MiniLM-L6-v2"
    if local_snapshot.is_dir():
        return local_snapshot
    return None


def generate_rpc_token() -> str:
    """Generate a per-process bearer token for the JSON-RPC stdio server.

    Uses ``secrets.token_urlsafe`` so an unprivileged observer of the
    surrounding process state cannot predict it. The CLI prints this to
    stderr exactly once when ``--serve`` is invoked without an explicit
    ``rpc_token``.
    """
    return secrets.token_urlsafe(24)
