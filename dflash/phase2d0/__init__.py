"""CPU-only Phase 2D.0 dependency-opportunity characterization."""

from .aggregation import aggregate_gap_events, build_research_summary
from .analysis import analyze_captures, classify_gap

__all__ = [
    "aggregate_gap_events",
    "analyze_captures",
    "build_research_summary",
    "classify_gap",
]
