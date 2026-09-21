"""Verdict layer: consequences, verification runs, and the verdict vocabulary."""

from .consequence import Consequence, judge
from .verification_run import VerificationRun
from .verified_constants import (
    REASONING_EFFORTS,
    SANDBOX_MODES,
    TASK_RISKS,
    VERDICT_ERROR,
    VERDICT_OK,
)

__all__ = [
    "REASONING_EFFORTS",
    "SANDBOX_MODES",
    "TASK_RISKS",
    "VERDICT_ERROR",
    "VERDICT_OK",
    "Consequence",
    "VerificationRun",
    "judge",
]
