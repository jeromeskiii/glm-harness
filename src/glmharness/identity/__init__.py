"""Identity layer: brand, document identity, and edit epochs."""

from .brand import (
    CLI_ENTRY,
    ENV_PREFIX,
    HARNESS_AUTHOR,
    HARNESS_HF,
    HARNESS_NAME,
    HARNESS_REPO,
    HARNESS_TITLE,
    HARNESS_VERSION,
    LOG_NAMESPACE,
    MODEL_VENDOR,
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
    "HARNESS_AUTHOR",
    "HARNESS_HF",
    "HARNESS_NAME",
    "HARNESS_REPO",
    "HARNESS_TITLE",
    "HARNESS_VERSION",
    "LOG_NAMESPACE",
    "MODEL_VENDOR",
    "REQUIRED_SNAPSHOT_FILES",
    "SNAPSHOT_SENTINEL",
    "has_weights",
    "is_snapshot_dir",
    "next_epoch",
]
