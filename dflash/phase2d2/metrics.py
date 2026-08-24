"""Aggregation and preregistered Phase 2D.2A decision rules."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

CANDIDATE_LABEL_AUTHORITY = "original_dflash_reference"
CANDIDATE_LABEL_SOURCE = "original_dflash_absolute_continuation"

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


def _candidate_quality_provenance(
    label_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    evidence = label_evidence or {}
    return {
        "label_authority": evidence.get("label_authority"),
        "label_source": evidence.get("label_source"),
        "target_label_authority": evidence.get("label_authority"),
        "target_label_source": evidence.get("label_source"),
        "target_label_used_for_runtime_selection": bool(
            evidence.get("target_label_used_for_runtime_selection", False)
        ),
        "target_labels_used_for_runtime_selection": bool(
            evidence.get("target_labels_used_for_runtime_selection", False)
        ),
        "semantic_generated_prefix_length": evidence.get(
            "semantic_generated_prefix_length"
        ),
        "semantic_prefix_token_sha256": evidence.get(
            "semantic_prefix_token_sha256"
        ),
        "reference_alignment_available": bool(
            evidence.get("reference_alignment_available", False)
        ),
        "reference_alignment_reason": evidence.get("reference_alignment_reason"),
        "target_label_evidence_available": bool(
            evidence.get(
                "target_label_evidence_available",
                evidence.get("reference_alignment_available", False),
            )
        ),
        "target_label_evidence_unavailable_reason": evidence.get(
            "target_label_evidence_unavailable_reason"
        ),
        "label_evidence_role": "post_hoc_evaluation_only",
        "q1_replay_used_as_target_label_source": False,
        "target_only_used_as_target_label_source": False,
        "fixed_shape_used_as_target_label_authority": False,
    }


def _unavailable_candidate_quality(
    *,
    label_evidence: Mapping[str, Any] | None,
    reason: str,
) -> dict[str, Any]:
    return {
        **_candidate_quality_provenance(label_evidence),
        "availability": "unavailable",
        "unavailable_reason": reason,
        "positions": [],
        "base_membership_count": 0,
        "conditional_membership_count": 0,
        "merged_membership_count": 0,
        "selected_tree_membership_count": 0,
        "refined_only_target_recovery_count": 0,
        "correct_prefix_survival_depth": 0,
        "first_pruning_depth": None,
        "base_candidate_coverage_rate": None,
        "conditional_candidate_coverage_rate": None,
        "merged_candidate_coverage_rate": None,
        "selected_tree_coverage_rate_at_fixed_n": None,
        "offline_oracle_coverage": None,
        "offline_oracle_coverage_unavailable_reason": (
            "Phase_2B_offline_oracle_artifact_not_supplied"
        ),
    }


def candidate_quality_for_round(
    round_row: dict[str, Any],
    *,
    label_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure selected candidates against an explicit post-hoc label source.

    ``replay_decisions`` deliberately remains a legacy diagnostic artifact but is
    never read here.  The caller must establish semantic-prefix alignment and pass
    the resulting Original-DFlash continuation labels explicitly.
    """

    if label_evidence is None:
        return _unavailable_candidate_quality(
            label_evidence=None,
            reason="explicit_label_evidence_not_provided",
        )
    if (
        label_evidence.get("target_label_used_for_runtime_selection") is not False
        or label_evidence.get("target_labels_used_for_runtime_selection") is not False
    ):
        raise ValueError("candidate-quality labels must remain post-hoc only")
    if label_evidence.get("label_authority") != CANDIDATE_LABEL_AUTHORITY:
        return _unavailable_candidate_quality(
            label_evidence=label_evidence,
            reason="unsupported_target_label_authority",
        )
    if label_evidence.get("label_source") != CANDIDATE_LABEL_SOURCE:
        return _unavailable_candidate_quality(
            label_evidence=label_evidence,
            reason="unsupported_target_label_source",
        )
    if not label_evidence.get("reference_alignment_available"):
        return _unavailable_candidate_quality(
            label_evidence=label_evidence,
            reason=(
                label_evidence.get("reference_alignment_reason")
                or "original_dflash_reference_alignment_unavailable"
            ),
        )
    if not label_evidence.get("target_label_evidence_available", True):
        return _unavailable_candidate_quality(
            label_evidence=label_evidence,
            reason=(
                label_evidence.get("target_label_evidence_unavailable_reason")
                or "original_dflash_target_labels_unavailable"
            ),
        )

    labels = label_evidence.get("target_labels")
    if not isinstance(labels, list) or not labels:
        return _unavailable_candidate_quality(
            label_evidence=label_evidence,
            reason="original_dflash_target_labels_unavailable",
        )
    expected_positions = list(range(1, len(labels) + 1))
    observed_positions = [int(row.get("position_index", -1)) for row in labels]
    if observed_positions != expected_positions:
        return _unavailable_candidate_quality(
            label_evidence=label_evidence,
            reason="artifact_identity_inconsistent",
        )

    phase = round_row["phase2d2"]
    events = phase.get("candidate_merge_events", [])
    tree_nodes = phase.get("tree_spec", {}).get("nodes", [])
    selected_tree_paths = {
        tuple(int(value) for value in row.get("path", [])) for row in tree_nodes
    }
    rows = []
    reference_path: tuple[int, ...] = ()
    provenance = _candidate_quality_provenance(label_evidence)
    for label in labels:
        position = int(label["position_index"])
        token = int(label["target_token_id"])
        reference_parent_path = reference_path
        reference_path = (*reference_path, token)
        positional_events = [
            row for row in events if int(row["position_index"]) == position
        ]
        base_events = [row for row in positional_events if row.get("base_rank") is not None]
        conditional_branches = [
            row
            for row in positional_events
            if row.get("parent_refined_prefix") is not None
            and bool(row.get("candidate_active_after_refined_prefix", True))
        ]
        matching_conditional = []
        for event in conditional_branches:
            refined_prefix = tuple(
                int(value) for value in event["parent_refined_prefix"]
            )
            if (
                len(refined_prefix) <= len(reference_parent_path)
                and reference_parent_path[: len(refined_prefix)] == refined_prefix
            ):
                matching_conditional.append(event)
        conditional_set_available = bool(matching_conditional)
        base = [
            row
            for row in base_events
            if int(row["token_id"]) == token and row.get("base_rank") is not None
        ]
        conditional = [
            row
            for row in matching_conditional
            if int(row["token_id"]) == token
            and row.get("conditional_rank") is not None
        ]
        merged = bool(base or conditional)
        selected = reference_path in selected_tree_paths
        best_base_rank = min((int(row["base_rank"]) for row in base), default=None)
        best_conditional_rank = min(
            (int(row["conditional_rank"]) for row in conditional), default=None
        )
        rows.append(
            {
                "position_index": position,
                "absolute_generated_token_offset": int(
                    label["absolute_generated_token_offset"]
                ),
                "target_token_id": token,
                "base_top_m_membership": bool(base),
                "conditional_top_r_membership": bool(conditional),
                "conditional_candidate_set_available": conditional_set_available,
                "conditional_candidate_set_unavailable_reason": (
                    None
                    if conditional_set_available
                    else "reference_parent_prefix_not_selected_for_conditional_refinement"
                ),
                "merged_membership": merged,
                "selected_tree_membership": selected,
                "refined_only_target_recovery": bool(conditional and not base),
                "best_base_rank": best_base_rank,
                "best_conditional_rank": best_conditional_rank,
                "target_token_rank_improvement": (
                    best_base_rank - best_conditional_rank
                    if best_base_rank is not None and best_conditional_rank is not None
                    else None
                ),
                "target_label_used_for_runtime_selection": False,
                "target_label_authority": CANDIDATE_LABEL_AUTHORITY,
                "target_label_source": CANDIDATE_LABEL_SOURCE,
                "semantic_generated_prefix_length": provenance[
                    "semantic_generated_prefix_length"
                ],
                "semantic_prefix_token_sha256": provenance[
                    "semantic_prefix_token_sha256"
                ],
                "reference_alignment_available": True,
                "reference_alignment_reason": None,
                "reference_parent_prefix_token_ids": list(reference_parent_path),
                "reference_prefix_through_position_token_ids": list(reference_path),
                "parent_tree_path": list(reference_parent_path),
                "matching_conditional_refined_prefixes": sorted(
                    {
                        tuple(int(value) for value in row["parent_refined_prefix"])
                        for row in matching_conditional
                    }
                ),
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
        **provenance,
        "availability": "available",
        "unavailable_reason": None,
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
    quality_rows = [row.get("candidate_quality", {}) for row in rounds]
    specified_authorities = {
        row.get("target_label_authority")
        for row in quality_rows
        if row.get("target_label_authority") is not None
    }
    specified_sources = {
        row.get("target_label_source")
        for row in quality_rows
        if row.get("target_label_source") is not None
    }
    unspecified_provenance_count = sum(
        row.get("target_label_authority") is None for row in quality_rows
    )
    if len(specified_authorities) > 1 or (
        specified_authorities and unspecified_provenance_count
    ):
        raise ValueError("mixed candidate-quality target-label authorities")
    if len(specified_sources) > 1 or (specified_sources and unspecified_provenance_count):
        raise ValueError("mixed candidate-quality target-label sources")
    if any(
        row.get("target_label_used_for_runtime_selection") is True
        or row.get("target_labels_used_for_runtime_selection") is True
        for row in quality_rows
    ):
        raise ValueError("candidate-quality target labels entered runtime selection")
    target_label_authority = next(iter(specified_authorities), None)
    target_label_source = next(iter(specified_sources), None)

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
            row
            for row in round_rows
            if row["phase2d2"].get(
                "fallback_executed",
                row["phase2d2"]["fallback_replay_executed"],
            )
        ]
        formal_sample_medians = [
            statistics.median(
                repeat["cuda_elapsed_ms"] for repeat in row["formal_timing_repeats"]
            )
            for row in sample_rows
        ]
        total_formal_ms = sum(formal_sample_medians)
        group_quality = [row["candidate_quality"] for row in round_rows]
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
                "candidate_quality_target_label_authority": target_label_authority,
                "candidate_quality_target_label_source": target_label_source,
                "candidate_quality_available_round_count": sum(
                    row.get("availability") == "available" for row in group_quality
                ),
                "candidate_quality_unavailable_round_count": sum(
                    row.get("availability") != "available" for row in group_quality
                ),
                "candidate_quality_reference_alignment_available_round_count": sum(
                    row.get("reference_alignment_available") is True
                    for row in group_quality
                ),
                "candidate_quality_target_labels_used_for_runtime_selection": False,
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
        "candidate_quality_target_label_authority": target_label_authority,
        "candidate_quality_target_label_source": target_label_source,
        "candidate_quality_mixed_label_authorities": False,
        "candidate_quality_target_labels_used_for_runtime_selection": False,
        "candidate_quality_available_round_count": sum(
            row.get("availability") == "available" for row in quality_rows
        ),
        "candidate_quality_unavailable_round_count": sum(
            row.get("availability") != "available" for row in quality_rows
        ),
        "candidate_quality_reference_alignment_available_round_count": sum(
            row.get("reference_alignment_available") is True for row in quality_rows
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
