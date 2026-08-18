"""CPU-only Phase 2D.1A fixed-budget selector realization."""

from .events import build_selector_events
from .selectors import build_cross_validation_folds, evaluate_selectors

__all__ = [
    "build_cross_validation_folds",
    "build_selector_events",
    "evaluate_selectors",
]
