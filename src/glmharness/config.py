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
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ConfigError

_ENV_PREFIX = "GLMH_"

#: env var name -> (attribute, converter, allowed values)
_ENV_MAP: dict[str, tuple[str, str, tuple[str, ...] | None]] = {
    "MODEL_PATH": ("model_path", "path", None),
    "SESSION": ("session_path", "path", None),
    "REASONING_EFFORT": ("reasoning_effort", "str", ("low", "high", "max")),
    "MAX_NEW_TOKENS": ("max_new_tokens", "int", None),
    "MAX_ROUNDS": ("max_rounds", "int", None),
    "TOOL_TIMEOUT_S": ("tool_timeout_s", "float", None),
    "REQUEST_TIMEOUT_S": ("request_timeout_s", "float", None),
    "MAX_RETRIES": ("max_retries", "int", None),
    "RETRY_BASE_DELAY_S": ("retry_base_delay_s", "float", None),
    "RETRY_MAX_DELAY_S": ("retry_max_delay_s", "float", None),
    "RETRY_JITTER": ("retry_jitter", "float", None),
    "LOG_FORMAT": ("log_format", "str", ("text", "json")),
    "LOG_LEVEL": ("log_level", "str", None),
    "CORRUPT_POLICY": ("corrupt_policy", "str", ("skip", "rename", "fail")),
}


@dataclass
class HarnessConfig:
    """All runtime knobs with production-safe defaults."""

    # model
    model_path: Path | None = None
    mock: str | None = None
    reasoning_effort: str = "max"
    max_new_tokens: int = 8192

    # session
    session_path: Path | None = None
    corrupt_policy: str = "skip"

    # loop
    max_rounds: int = 12
    request_timeout_s: float = 0.0
    tool_timeout_s: float = 30.0

    # retry (exponential backoff with jitter)
    max_retries: int = 2
    retry_base_delay_s: float = 1.0
    retry_max_delay_s: float = 30.0
    retry_jitter: float = 0.25

    # observability
    log_format: str = "text"
    log_level: str = "INFO"

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
            if raw is None or raw == "":
                continue
            try:
                if kind == "int":
                    value: int | float | Path | str = int(raw)
                elif kind == "float":
                    value = float(raw)
                elif kind == "path":
                    value = Path(raw)
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
            raise ConfigError("timeouts must be >= 0 (0 disables them)")
        if not 0 <= self.retry_jitter < 1:
            raise ConfigError("retry_jitter must be in [0, 1)")
        if self.retry_base_delay_s <= 0 or self.retry_max_delay_s < self.retry_base_delay_s:
            raise ConfigError("retry_base_delay_s must be > 0 and <= retry_max_delay_s")
        if self.model_path is not None and not self.model_path.is_dir():
            raise ConfigError(f"model path is not a directory: {self.model_path}")

    def retry_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter for the given 1-based attempt."""
        delay = min(self.retry_base_delay_s * (2 ** (attempt - 1)), self.retry_max_delay_s)
        if self.retry_jitter:
            delay *= 1 + self.retry_jitter * (2 * _hash_fraction(attempt) - 1)
        return delay

    def unknown_env_keys(self) -> list[str]:
        """``GLMH_*`` variables that map to nothing (typo guard)."""
        return sorted(self.unknown_env)


def _hash_fraction(seed: int) -> float:
    """Deterministic pseudo-random fraction in [0, 1) — stable for tests."""
    x = (seed * 1103515245 + 12345) % (2**31)
    return x / (2**31)
