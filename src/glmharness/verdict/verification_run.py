"""Verification run: one full drive of the verification battery.

A run collects consequences, tool results and judgments, then folds them
into a single verdict. It is the harness-side analogue of Reticle's
``verification-run``: a run without a judgment has no result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .consequence import Consequence, judge
from .verified_constants import VERDICT_ERROR, VERDICT_OK


@dataclass
class VerificationRun:
    """Collects judged steps and folds them into one verdict."""

    steps: list[tuple[Consequence, str]] = field(
        default_factory=lambda: list[tuple[Consequence, str]]()
    )

    def record(self, consequence: Consequence, result: dict[str, Any]) -> str:
        verdict = judge(consequence, result)
        self.steps.append((consequence, verdict))
        return verdict

    def verdict(self) -> str:
        if not self.steps:
            return "unknown"
        if any(v == VERDICT_ERROR for _, v in self.steps):
            return VERDICT_ERROR
        if any(v == "unknown" for _, v in self.steps):
            return "unknown"
        return VERDICT_OK
