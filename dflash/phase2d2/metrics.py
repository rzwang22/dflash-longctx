"""Aggregation and preregistered Phase 2D.2A decision rules."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any

DECISIONS = (
    "proceed_to_lightweight_causal_adapter",
    "proceed_with_conditional_remask",
    "refinement_benefit_but_guard_too_costly",
    "no_dependency_refinement_benefit",
    "numerical_safety_unresolved",
    "insufficient_evidence",
)


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]


def distribution(values: list[float | int | None]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean) if clean else None,
        "median": statistics.median(clean) if clean else None,
        "p90": _p90(clean),
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
    }


def candidate_quality_for_round(round_row: dict[str, Any]) -> dict[str, Any]:
    """Label replay-observed target tokens without feeding labels into selection."""

    phase = round_row["phase2d2"]
    decisions = phase.get("replay_decisions", [])
    events = phase.get("candidate_merge_events", [])
    tree_nodes = phase.get("tree_spec", {}).get("nodes", [])
    nodes_by_index = {
        int(row["node_index"]): row for row in tree_nodes if "node_index" in row
    }
    rows = []
    for decision in decisions:
        position = int(decision["decision_depth"]) + 1
        token = int(decision["top1_token_id"])
        parent = nodes_by_index.get(int(decision.get("parent_node_index", -1)), {})
        parent_path = tuple(int(value) for value in parent.get("path", []))
        matching = [
            row
            for row in events
            if int(row["position_index"]) == position
            and (
                row.get("parent_refined_prefix") is None
                or (
                    position <= len(row["parent_refined_prefix"])
                    or parent_path[: len(row["parent_refined_prefix"])]
                    == tuple(int(value) for value in row["parent_refined_prefix"])
                )
            )
        ]
        base = [
            row
            for row in matching
            if int(row["token_id"]) == token and row.get("base_rank") is not None
        ]
        conditional = [
            row
            for row in matching
            if int(row["token_id"]) == token
            and (
                row.get("parent_refined_prefix") is None
                or position > len(row["parent_refined_prefix"])
            )
            and row.get("conditional_rank") is not None
        ]
        merged = [
            row
            for row in matching
            if int(row["token_id"]) == token
            and (
                row.get("parent_refined_prefix") is None
                or position > len(row["parent_refined_prefix"])
                or row.get("base_rank") is not None
            )
        ]
        accepted = decision.get("accepted_candidate_token") == token
        best_base_rank = min((int(row["base_rank"]) for row in base), default=None)
        best_conditional_rank = min(
            (int(row["conditional_rank"]) for row in conditional), default=None
        )
        rows.append(
            {
                "position_index": position,
                "target_token_id": token,
                "base_top_m_membership": bool(base),
                "conditional_top_r_membership": bool(conditional),
                "merged_membership": bool(merged),
                "selected_tree_membership": bool(accepted),
                "refined_only_target_recovery": bool(conditional and not base),
                "best_base_rank": best_base_rank,
                "best_conditional_rank": best_conditional_rank,
                "target_token_rank_improvement": (
                    best_base_rank - best_conditional_rank
                    if best_base_rank is not None and best_conditional_rank is not None
                    else None
                ),
                "target_label_used_for_runtime_selection": False,
                "parent_tree_path": list(parent_path),
            }
        )
    survival_depth = 0
    for row in rows:
        if not row["selected_tree_membership"]:
            break
        survival_depth += 1
    first_pruning_depth = (
        next(
            (
                int(row["position_index"])
                for row in rows
                if not row["selected_tree_membership"]
            ),
            None,
        )
        if rows
        else None
    )
    return {
        "availability": "available" if decisions else "unavailable",
        "unavailable_reason": None if decisions else "q1_replay_not_executed",
        "positions": rows,
        "base_membership_count": sum(row["base_top_m_membership"] for row in rows),
        "conditional_membership_count": sum(
            row["conditional_top_r_membership"] for row in rows
        ),
        "merged_membership_count": sum(row["merged_membership"] for row in rows),
        "selected_tree_membership_count": sum(
            row["selected_tree_membership"] for row in rows
        ),
        "refined_only_target_recovery_count": sum(
            row["refined_only_target_recovery"] for row in rows
        ),
        "correct_prefix_survival_depth": survival_depth,
        "first_pruning_depth": first_pruning_depth,
        "base_candidate_coverage_rate": (
            sum(row["base_top_m_membership"] for row in rows) / len(rows)
            if rows
            else None
        ),
        "conditional_candidate_coverage_rate": (
            sum(row["conditional_top_r_membership"] for row in rows) / len(rows)
            if rows
            else None
        ),
        "merged_candidate_coverage_rate": (
            sum(row["merged_membership"] for row in rows) / len(rows) if rows else None
        ),
        "selected_tree_coverage_rate_at_fixed_n": (
            sum(row["selected_tree_membership"] for row in rows) / len(rows)
            if rows
            else None
        ),
        "offline_oracle_coverage": None,
        "offline_oracle_coverage_unavailable_reason": (
            "Phase_2B_offline_oracle_artifact_not_supplied"
        ),
    }


def aggregate_results(
    samples: list[dict[str, Any]], rounds: list[dict[str, Any]]
) -> dict[str, Any]:
    groups: dict[tuple[str, int, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"samples": [], "rounds": []}
    )
    for sample in samples:
        key = (
            sample["task"],
            int(sample["nominal_prompt_length"]),
            sample["configuration_id"],
        )
        groups[key]["samples"].append(sample)
    for row in rounds:
        key = (
            row["task"],
            int(row["nominal_prompt_length"]),
            row["configuration_id"],
        )
        groups[key]["rounds"].append(row)
    aggregates = []
    for (task, context, configuration), group in sorted(groups.items()):
        sample_rows = group["samples"]
        round_rows = group["rounds"]
        returned = sum(int(row["generated_token_count"]) for row in sample_rows)
        committed = sum(int(row["committed_token_count"]) for row in round_rows)
        verified = sum(
            int(row["phase2d2"]["actual_verified_node_count"]) for row in round_rows
        )
        accepted = sum(int(row["accepted_draft_token_count"]) for row in round_rows)
        fallback_rounds = [
            row for row in round_rows if row["phase2d2"]["fallback_replay_executed"]
        ]
        formal_sample_medians = [
            statistics.median(
                repeat["cuda_elapsed_ms"] for repeat in row["formal_timing_repeats"]
            )
            for row in sample_rows
        ]
        total_formal_ms = sum(formal_sample_medians)
        aggregates.append(
            {
                "schema_version": "1.0",
                "task": task,
                "nominal_prompt_length": context,
                "configuration_id": configuration,
                "sample_count": len(sample_rows),
                "round_count": len(round_rows),
                "returned_token_count": returned,
                "committed_token_count": committed,
                "accepted_draft_token_count": accepted,
                "verified_node_count": verified,
                "packed_target_forward_count": len(round_rows),
                "formal_target_forward_count": sum(
                    int(row["phase2d2"]["formal_target_forward_count"])
                    for row in round_rows
                ),
                "diagnostic_shadow_target_forward_count": sum(
                    int(row["phase2d2"]["diagnostic_shadow_target_forward_count"])
                    for row in round_rows
                ),
                "accepted_tokens_per_verified_node": accepted / verified
                if verified
                else None,
                "accepted_tokens_per_round": accepted / len(round_rows)
                if round_rows
                else None,
                "committed_tokens_per_round": committed / len(round_rows)
                if round_rows
                else None,
                "generation_cuda_ms_per_returned_token": (
                    total_formal_ms / returned if returned else None
                ),
                "generation_cuda_ms_per_committed_token": (
                    total_formal_ms / committed if committed else None
                ),
                "fallback_round_count": len(fallback_rounds),
                "fallback_rate": len(fallback_rounds) / len(round_rows)
                if round_rows
                else None,
                "guard_false_positive_count": sum(
                    row["phase2d2"]["guard_false_positive"] for row in round_rows
                ),
                "guard_false_negative_count": sum(
                    row["phase2d2"]["guard_false_negative"] for row in round_rows
                ),
                "packed_replay_decision_divergence_count": sum(
                    row["phase2d2"]["packed_numerical_decision_divergence"]
                    for row in round_rows
                ),
                "output_exact_match_rate": (
                    sum(
                        row["correctness"]["target_only_output_ids_equal"]
                        for row in sample_rows
                    )
                    / len(sample_rows)
                    if sample_rows
                    else None
                ),
                "generation_cuda_latency_ms": distribution(
                    [
                        repeat["cuda_elapsed_ms"]
                        for row in sample_rows
                        for repeat in row["formal_timing_repeats"]
                    ]
                ),
                "conditional_remask_cuda_ms": distribution(
                    [
                        row["phase2d2"].get("conditional_remask_cuda_elapsed_ms")
                        for row in round_rows
                    ]
                ),
                "fallback_cuda_ms": distribution(
                    [
                        row["phase2d2"].get("fallback_cuda_elapsed_ms")
                        for row in round_rows
                    ]
                ),
                "total_draft_cuda_ms": distribution(
                    [
                        row["phase2d2"].get("total_draft_cuda_elapsed_ms")
                        for row in round_rows
                    ]
                ),
                "verification_cuda_ms": distribution(
                    [
                        row["phase2d2"].get("packed_verification_cuda_elapsed_ms")
                        for row in round_rows
                    ]
                ),
                "round_cuda_ms": distribution(
                    [row["phase2d2"].get("round_cuda_elapsed_ms") for row in round_rows]
                ),
                "candidate_refined_only_recovery_count": sum(
                    row["candidate_quality"]["refined_only_target_recovery_count"]
                    for row in round_rows
                ),
                "candidate_selected_tree_coverage_rate_at_fixed_n": (
                    sum(
                        row["candidate_quality"]["selected_tree_membership_count"]
                        for row in round_rows
                    )
                    / sum(
                        len(row["candidate_quality"]["positions"]) for row in round_rows
                    )
                    if sum(
                        len(row["candidate_quality"]["positions"]) for row in round_rows
                    )
                    else None
                ),
            }
        )
    return {
        "schema_version": "1.0",
        "status": "ok" if samples else "insufficient_evidence",
        "sample_count": len(samples),
        "round_count": len(rounds),
        "aggregates": aggregates,
        "configuration_distribution": dict(
            Counter(sample["configuration_id"] for sample in samples)
        ),
        "metrics_reported_per_generation_round_returned_and_committed_token": True,
    }


def build_phase2d2_decision(
    *,
    official_checker_passed: bool,
    structural_correctness_passed: bool,
    guard_false_negative_count: int,
    independent_task_count: int,
    stable_coverage_gain: bool | None,
    stable_online_efficiency_gain: bool | None,
    conditional_cost_acceptable: bool | None,
    full_second_pass_too_heavy: bool | None,
) -> dict[str, Any]:
    """Apply the preregistered priority order without manual overrides."""

    if not official_checker_passed or not structural_correctness_passed:
        decision = "numerical_safety_unresolved"
        reason = "official_checker_or_structural_correctness_failed"
    elif guard_false_negative_count:
        decision = "numerical_safety_unresolved"
        reason = "held_out_margin_guard_false_negative_observed"
    elif independent_task_count < 2 or None in {
        stable_coverage_gain,
        stable_online_efficiency_gain,
        conditional_cost_acceptable,
        full_second_pass_too_heavy,
    }:
        decision = "insufficient_evidence"
        reason = "preregistered_stability_or_cost_evidence_is_incomplete"
    elif not stable_coverage_gain and not stable_online_efficiency_gain:
        decision = "no_dependency_refinement_benefit"
        reason = "fixed_budget_coverage_and_online_efficiency_did_not_improve"
    elif stable_coverage_gain and full_second_pass_too_heavy:
        decision = "proceed_to_lightweight_causal_adapter"
        reason = "candidate_gain_observed_but_full_conditional_pass_is_too_heavy"
    elif (
        stable_coverage_gain
        and stable_online_efficiency_gain
        and conditional_cost_acceptable
    ):
        decision = "proceed_with_conditional_remask"
        reason = "stable_fixed_budget_gain_with_acceptable_conditional_and_guard_cost"
    else:
        decision = "refinement_benefit_but_guard_too_costly"
        reason = "refinement_gain_observed_but_conditional_or_fallback_cost_offsets_it"
    return {
        "schema_version": "1.0",
        "decision": decision,
        "reason": reason,
        "manual_override_used": False,
        "preregistered_priority_order_applied": True,
        "inputs": {
            "official_checker_passed": official_checker_passed,
            "structural_correctness_passed": structural_correctness_passed,
            "guard_false_negative_count": int(guard_false_negative_count),
            "independent_task_count": int(independent_task_count),
            "stable_coverage_gain": stable_coverage_gain,
            "stable_online_efficiency_gain": stable_online_efficiency_gain,
            "conditional_cost_acceptable": conditional_cost_acceptable,
            "full_second_pass_too_heavy": full_second_pass_too_heavy,
        },
    }
