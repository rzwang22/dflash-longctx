"""Model-lazy mechanics/independent cohort inventory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dflash.phase1b.workload import build_workload_inventory


def selector_source_id(task: str, source_sample_id: str) -> str:
    return f"{task}::{source_sample_id}"


def build_selector_inventory(
    prepared: Sequence[dict[str, Any]],
    *,
    frozen: dict[str, Any],
    task: str,
    nominal_lengths: Sequence[int],
    requested_sample_count: int,
    exclude_training_sources: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    training = set(frozen["training_source_ids"])
    before = build_workload_inventory(
        prepared,
        task=task,
        nominal_lengths=nominal_lengths,
        requested_sample_count=requested_sample_count,
    )
    filtered = (
        [
            row
            for row in prepared
            if selector_source_id(task, row["source_sample_id"]) not in training
        ]
        if exclude_training_sources
        else list(prepared)
    )
    after = build_workload_inventory(
        filtered,
        task=task,
        nominal_lengths=nominal_lengths,
        requested_sample_count=requested_sample_count,
    )
    by_length = {}
    for length in nominal_lengths:
        key = str(int(length))
        before_details = before["eligible_by_prompt_length"][key]
        after_details = after["eligible_by_prompt_length"][key]
        excluded = sorted(
            set(before_details["source_sample_ids"])
            - set(after_details["source_sample_ids"])
        )
        by_length[key] = {
            "total_eligible_sources": len(before_details["source_sample_ids"]),
            "excluded_training_sources": len(excluded) if exclude_training_sources else 0,
            "excluded_training_source_ids": excluded if exclude_training_sources else [],
            "remaining_independent_sources": len(after_details["source_sample_ids"]),
            "remaining_source_ids": after_details["source_sample_ids"],
            "selected_independent_source_ids": after_details["source_sample_ids"][
                :requested_sample_count
            ],
        }
    inventory = {
        **after,
        "mode": "phase2d1b_selector_inventory",
        "evaluation_cohort": (
            "independent_evaluation"
            if exclude_training_sources
            else "in_sample_mechanics"
        ),
        "independent_generalization": bool(exclude_training_sources),
        "selector_config_hash": frozen["config_sha256"],
        "selector_training_source_ids": sorted(training),
        "training_source_exclusion_completed_before_model_loading": True,
        "training_sources_reintroduced": False,
        "selector_source_inventory_by_prompt_length": by_length,
        "independent_32k_evidence_unavailable": bool(
            exclude_training_sources
            and 32768 in nominal_lengths
            and by_length["32768"]["remaining_independent_sources"] == 0
        ),
        "models_loaded": False,
    }
    return filtered, inventory
