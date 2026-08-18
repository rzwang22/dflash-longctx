"""Pre-registered pilot decisions and RQ answers for Phase 2D.1B-NR."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

DEFAULT_DECISION_RULES = {
    "rare_divergence_threshold": 0.02,
    "high_divergence_threshold": 0.10,
    "promising_trigger_rate": 0.05,
    "maximum_candidate_trigger_rate": 0.20,
    "required_divergence_recall": 0.95,
}


def _policy_row(
    rows: Sequence[dict[str, Any]], mode: str, budget: int
) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if row["selector_mode"] == mode and int(row["node_budget"]) == budget
        ),
        None,
    )


def choose_nr_decision(
    observations: Sequence[dict[str, Any]],
    policy_summary: Sequence[dict[str, Any]],
    sweep: Sequence[dict[str, Any]],
    *,
    rules: dict[str, float] | None = None,
) -> dict[str, Any]:
    configured = {**DEFAULT_DECISION_RULES, **(rules or {})}
    top1 = [
        row for row in observations if row.get("selector_mode") == "top1_packed_tree"
    ]
    if any(row.get("outcome_category") == "actual_structural_failure" for row in top1):
        decision = "investigate_structural_correctness"
        reason = "top1_packed_tree_structural_failure_observed"
        best = None
    elif any(
        row.get("outcome_category") == "reference_instability" for row in observations
    ):
        decision = "investigate_reference_instability"
        reason = "original_dflash_reference_instability_observed"
        best = None
    elif any(
        row.get("outcome_category")
        in {"incomplete_correctness_evidence", "infrastructure_failure"}
        for row in observations
    ):
        decision = "collect_more_robustness_evidence"
        reason = "incomplete_or_infrastructure_failed_observation"
        best = None
    else:
        primary = _policy_row(policy_summary, "global_fixed", 16)
        divergence_rate = (
            primary.get("actual_divergent_configuration_case_rate") if primary else None
        )
        candidates = [
            row
            for row in sweep
            if row["experiment"] == "cohort"
            and row["selector_mode"] == "global_fixed"
            and int(row["node_budget"]) == 16
            and row["scope"] == "overall"
            and row.get("divergence_recall") is not None
            and row.get("trigger_rate") is not None
            and row["divergence_recall"] >= configured["required_divergence_recall"]
        ]
        best = (
            min(candidates, key=lambda row: (row["trigger_rate"], row["threshold"]))
            if candidates
            else None
        )
        if divergence_rate is None or divergence_rate == 0:
            decision = "collect_more_robustness_evidence"
            reason = "no_actual_branching_divergence_observed_or_evidence_too_sparse"
        elif (
            divergence_rate <= configured["rare_divergence_threshold"]
            and best is not None
            and best["trigger_rate"] <= configured["promising_trigger_rate"]
        ):
            decision = "characterize_selective_stabilization_cost"
            reason = "rare_margin_localized_branching_divergence"
        elif (
            divergence_rate <= configured["high_divergence_threshold"]
            and best is not None
            and best["trigger_rate"] <= configured["maximum_candidate_trigger_rate"]
        ):
            decision = "characterize_selective_stabilization_cost"
            reason = "moderate_margin_localized_branching_divergence"
        else:
            decision = "reconsider_branching_verifier"
            reason = "frequent_or_not_selectively_margin_localized_divergence"
    return {
        "schema_version": "1.0",
        "decision": decision,
        "reason": reason,
        "decision_rules": configured,
        "primary_branching_configuration": "global_fixed-n16",
        "qualifying_margin_trigger": best,
        "expected_stabilization_cost_risk": (
            "moderate"
            if decision == "characterize_selective_stabilization_cost"
            and reason.startswith("moderate")
            else None
        ),
        "pilot_engineering_guidance_not_population_claim": True,
    }


def build_research_summary(
    exact_summary: dict[str, Any],
    cohort_summary: dict[str, list[dict[str, Any]]],
    sweep: Sequence[dict[str, Any]],
    decision: dict[str, Any],
) -> dict[str, Any]:
    strata = {
        (row["selector_mode"], int(row["node_budget"])): row
        for row in exact_summary["configuration_strata"]
    }
    policies = {
        (row["selector_mode"], int(row["node_budget"])): row
        for row in cohort_summary["policy_divergence_summary"]
    }
    global16 = policies.get(("global_fixed", 16))
    global32 = policies.get(("global_fixed", 32))
    top1 = policies.get(("top1_packed_tree", 16))
    divergence_sweep = [
        row
        for row in sweep
        if row["experiment"] == "cohort"
        and row["selector_mode"] == "global_fixed"
        and row["scope"] == "overall"
        and row.get("first_actual_divergence_round_count", 0)
    ]
    qualifying = decision.get("qualifying_margin_trigger")
    return {
        "schema_version": "1.0",
        "RQ1": {
            "question": "known global-fixed N=16 divergence process reproducibility",
            "answer": strata.get(("global_fixed", 16), {}).get(
                "reproducibility_classification", "insufficient_evidence"
            ),
        },
        "RQ2": {
            "question": "top1 packed-tree N=16 actual-output exactness",
            "exact_case": strata.get(("top1_packed_tree", 16)),
            "cohort": top1,
        },
        "RQ3": {
            "question": "global-fixed N=16 GovReport matched-cohort divergence rate",
            "observation_rate": global16.get("actual_divergence_observation_rate")
            if global16
            else None,
            "unique_configuration_case_rate": global16.get(
                "actual_divergent_configuration_case_rate"
            )
            if global16
            else None,
        },
        "RQ4": {
            "question": "global-fixed divergence change from N=16 to N=32",
            "n16": global16,
            "n32": global32,
        },
        "RQ5": {
            "question": "whether divergence events are concentrated at small margins",
            "divergence_margin_sweep_rows": divergence_sweep,
        },
        "RQ6": {
            "question": "whether a low-margin trigger covers divergence at small trigger rate",
            "best_preregistered_trigger": qualifying,
        },
        "RQ7": {
            "question": "recommended next diagnostic or method-characterization phase",
            "answer": decision["decision"],
            "reason": decision["reason"],
        },
        "no_performance_claims": True,
    }
