"""Artifacts flow-step tool: the step vocabulary for a tool call's journey.

A step is one named stage of the tool lifecycle; these helpers build the
step records used by journaling and tests, so stage names exist in exactly
one place.
"""

from __future__ import annotations

from typing import Any

STAGE_PRE_EXECUTE = "pre-execute"
STAGE_POST_EXECUTE = "post-execute"


def step_step_call(step: str, call: dict[str, Any]) -> dict[str, Any]:
    """Build a step record for a tool call."""
    return {"step": step, "call": call}


def step_step_event(step: str, event: str) -> dict[str, Any]:
    """Build a step record for an event, without a call payload."""
    return {"step": step, "event": event}
