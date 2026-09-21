"""Verdict constants: the closed vocabulary of execution outcomes."""

from __future__ import annotations

VERDICT_OK = "ok"
VERDICT_ERROR = "error"

SANDBOX_MODES: frozenset[str] = frozenset({"allow", "deny", "ask"})
TASK_RISKS: frozenset[str] = frozenset({"low", "medium", "high", "critical"})
REASONING_EFFORTS: frozenset[str] = frozenset({"low", "high", "max"})
