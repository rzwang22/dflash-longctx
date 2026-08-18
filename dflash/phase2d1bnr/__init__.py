"""Phase 2D.1B-NR numerical-robustness diagnostics."""

from .aggregation import (
    aggregate_cohort_cases,
    aggregate_exact_reproducibility,
    build_margin_trigger_sweep,
)
from .decision import build_research_summary, choose_nr_decision
from .margins import extract_round_margin_events
from .parse import classify_subrun_artifacts

__all__ = [
    "aggregate_cohort_cases",
    "aggregate_exact_reproducibility",
    "build_margin_trigger_sweep",
    "build_research_summary",
    "choose_nr_decision",
    "classify_subrun_artifacts",
    "extract_round_margin_events",
]
