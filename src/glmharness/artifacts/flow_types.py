"""Artifacts flow types: the dataclasses that flow-step artifacts are made of."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Tool:
    """A tool the model may call. ``allowed`` is the static policy gate."""

    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[Any] | Any]
    allowed: bool = True
