"""Document identity: what makes a snapshot directory identifiable.

A snapshot is a directory with config.json + tokenizer_config.json (and,
optionally, processor files). Resolution lives in one place so --doctor,
the CLI and the LLM adapters all agree.
"""

from __future__ import annotations

from pathlib import Path

REQUIRED_SNAPSHOT_FILES: tuple[str, ...] = (
    "config.json",
    "tokenizer_config.json",
)

SNAPSHOT_SENTINEL = "model.safetensors.index.json"


def is_snapshot_dir(path: Path) -> bool:
    """A directory is a snapshot when every required file is present."""
    return all((path / name).is_file() for name in REQUIRED_SNAPSHOT_FILES)


def has_weights(path: Path) -> bool:
    """The sentinel index marks a complete (or at least sharded) snapshot."""
    return (path / SNAPSHOT_SENTINEL).is_file()
