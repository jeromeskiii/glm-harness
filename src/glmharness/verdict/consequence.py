"""Consequence: the named-before-acted claim that a verdict checks.

Every mutating action records the consequence it expects BEFORE executing.
The result is then judged against that record, which is what turns an
outcome into a verdict rather than a story.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Consequence:
    """What a step claims will happen, named before the step runs."""

    tool: str
    expect: str
    arguments: dict[str, Any]


def judge(consequence: Consequence, result: object) -> str:
    """Return ok / error / unknown for the result against the consequence."""
    if not isinstance(result, Mapping):
        return "unknown"
    res_map: Mapping[str, Any] = result  # type: ignore[assignment]
    ok_val = res_map.get("ok")
    if ok_val is True:
        return "ok"
    if ok_val is False:
        return "error"
    return "unknown"
