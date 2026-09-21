"""Edit epoch: the session's monotonic revision counter.

Every durable mutation of the session log (append, compact, corrupt-file
rename) bumps the epoch. Consumers compare epochs to detect stale reads
instead of comparing timestamps.
"""

from __future__ import annotations

GENESIS_EPOCH = 0


def next_epoch(current: int) -> int:
    """Monotonic successor of ``current``."""
    if current < GENESIS_EPOCH:
        raise ValueError(f"epoch below genesis: {current}")
    return current + 1
