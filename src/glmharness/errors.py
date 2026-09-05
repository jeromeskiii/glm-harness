"""Typed error hierarchy for the harness.

Wire-facing and log-facing errors must be distinguishable: the session log
records ``type(exc).__name__`` for failed steps, so error classes carry the
diagnostic vocabulary. Provider errors additionally declare whether a retry
is safe — a retried stream must be side-effect free, which only holds before
any tool result was committed.
"""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for every error the harness raises deliberately."""


class ConfigError(HarnessError):
    """Invalid configuration (bad env var, bad CLI flag, missing runtime dep)."""


class SessionCorruptError(HarnessError):
    """A session JSONL file failed to parse under the configured policy."""


class ProviderError(HarnessError):
    """The model provider failed a request.

    ``retryable`` marks failures where replaying the request is safe:
    transient transport errors, timeouts, and provider-side 5xx/429
    responses. Local generation errors (OOM, dtype issues) are *not*
    retryable — they will not fix themselves by replay.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ProviderTimeout(ProviderError):
    """The provider exceeded the configured per-request timeout."""

    def __init__(self, timeout_s: float) -> None:
        super().__init__(f"provider exceeded request timeout of {timeout_s:g}s", retryable=True)
        self.timeout_s = timeout_s


class ProviderUnavailable(ProviderError):
    """The provider could not be reached or loaded transiently."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class ToolError(HarnessError):
    """A tool raised unexpectedly outside the guarded execution pipeline."""
