"""Identity layer: brand, document identity, and edit epochs."""

from .brand import (
    CLI_ENTRY,
    ENV_PREFIX,
    HARNESS_NAME,
    HARNESS_TITLE,
    HARNESS_VERSION,
    LOG_NAMESPACE,
)
from .document_identity import (
    REQUIRED_SNAPSHOT_FILES,
    SNAPSHOT_SENTINEL,
    has_weights,
    is_snapshot_dir,
)
from .edit_epoch import GENESIS_EPOCH, next_epoch

__all__ = [
    "CLI_ENTRY",
    "ENV_PREFIX",
    "GENESIS_EPOCH",
    "HARNESS_NAME",
    "HARNESS_TITLE",
    "HARNESS_VERSION",
    "LOG_NAMESPACE",
    "REQUIRED_SNAPSHOT_FILES",
    "SNAPSHOT_SENTINEL",
    "has_weights",
    "is_snapshot_dir",
    "next_epoch",
]
