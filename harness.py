"""Compatibility shim — the real package lives at :mod:`glmharness`.

This module re-exports every public name the legacy single-file script
exposed so existing imports and ``python3 harness.py ...`` keep working.
New code should import from the package directly:

    from glmharness import AgentLoop, Context, MockLLM, SessionLog
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from glmharness import *  # noqa: F401,F403
from glmharness import EventBus, PluginLoader  # explicit re-export
from glmharness.llm import TransformersGLM  # not pulled by ``import *``

if __name__ == "__main__":
    from glmharness.cli import main

    sys.exit(main())
