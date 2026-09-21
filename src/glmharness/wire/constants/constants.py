"""Wire constants: the vocabulary that crosses kernel boundaries.

Mirrors Reticle's ``core/src/wire/constants`` — every domain string that
crosses a boundary lives here as a named constant. Nothing else in the
package may inline one of these strings.
"""

from __future__ import annotations

BUS_MODES: frozenset[str] = frozenset({"emit", "waterfall", "parallel", "serial"})

LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

SERVER_CAPABILITIES: dict[str, bool] = {
    "streaming": True,
    "replay.from_log": True,
    "tools.native": True,
    "tools.code": True,
    "compaction": True,
    "approval.ask": True,
    "skills": True,
    "subagents.native": False,
    "plan": False,
    "goals": False,
    "mcp.client": False,
    "prompt.image": False,
    "fs.world": True,
    "sandbox.world": True,
}
