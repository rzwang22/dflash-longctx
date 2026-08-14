"""CPU-safe utilities for Phase 1A DFlash provenance and observability."""

from .materialize import (
    DRAFT_MODEL_ID,
    DRAFT_REVISION,
    load_materialization,
    materialize_draft,
    verify_materialization,
)
from .source import locate_callable, sha256_file

__all__ = [
    "DRAFT_MODEL_ID",
    "DRAFT_REVISION",
    "load_materialization",
    "locate_callable",
    "materialize_draft",
    "sha256_file",
    "verify_materialization",
]
