"""Framework-lazy Phase 2D.2A conditional-remask helpers."""

from .refinement import (
    CANDIDATE_MERGE_MODES,
    VERIFICATION_MODES,
    build_conditional_remask_inputs,
    build_refined_tree,
    select_refinement_prefixes,
    validate_refinement_options,
)
from .robustness import (
    calibration_sha256,
    classify_guard_event,
    validate_nr_calibration,
)

__all__ = [
    "CANDIDATE_MERGE_MODES",
    "VERIFICATION_MODES",
    "build_conditional_remask_inputs",
    "build_refined_tree",
    "calibration_sha256",
    "classify_guard_event",
    "select_refinement_prefixes",
    "validate_nr_calibration",
    "validate_refinement_options",
]
