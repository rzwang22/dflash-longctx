#!/usr/bin/env python3
"""Audit Phase 2D.2A-H1 node utilization and candidate mechanism evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase2d2.metrics import (
    CANDIDATE_LABEL_AUTHORITY,
    CANDIDATE_LABEL_SOURCE,
)
from scripts.aggregate_phase2d2a_h_known import (
    BASE_FAMILY,
    EXPECTED_CONTEXT_LENGTHS,
    EXPECTED_FAMILIES,
    EXPECTED_NODE_BUDGETS,
    EXPECTED_VERIFICATION_MODE,
    FROZEN_REFINEMENT_FAMILIES,
    KNOWN_SAMPLE_ID,
    KNOWN_TASK,
)
from scripts.check_phase2d2a_h_known_acceptance import check_h1_output

SCHEMA_VERSION = "1.0"
PHASE = "2D.2A-H1.1"

ARTIFACTS = (
    "phase2d2a-h1-budget-utilization.json",
    "phase2d2a-h1-budget-utilization.csv",
    "phase2d2a-h1-mechanism-decomposition.json",
    "phase2d2a-h1-mechanism-decomposition.csv",
    "phase2d2a-h1-reinterpretation.json",
    "phase2d2a-h1-reinterpretation.csv",
)
H2_ARTIFACT = "phase2d2a-h2-preregistered-design.json"

SCHEMAS = {
    "phase2d2a-h1-budget-utilization.json": (
        "phase2d2a-h11-budget-utilization.schema.json"
    ),
    "phase2d2a-h1-mechanism-decomposition.json": (
        "phase2d2a-h11-mechanism-decomposition.schema.json"
    ),
    "phase2d2a-h1-reinterpretation.json": (
        "phase2d2a-h11-reinterpretation.schema.json"
    ),
    H2_ARTIFACT: "phase2d2a-h2-preregistered-design.schema.json",
}


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL: {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_schema(payload: Any, schema_name: str) -> None:
    schema = _json(REPO_ROOT / "docs" / "schemas" / schema_name)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)


def _distribution(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    clean = [float(value) for value in values]
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "min": min(clean),
        "max": max(clean),
    }


def _histogram(values: list[int]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _normalized_histogram(values: list[int]) -> dict[str, float]:
    counts = Counter(str(value) for value in values)
    return (
        {key: count / len(values) for key, count in sorted(counts.items())}
        if values
        else {}
    )


def node_utilization_from_rounds(
    *,
    configured_node_budget: int,
    actual_verified_node_counts: list[int],
    selected_tree_node_counts: list[int],
) -> dict[str, Any]:
    """Compute utilization directly from per-round target verification counts."""

    if configured_node_budget <= 0:
        raise ValueError("configured node budget must be positive")
    if not actual_verified_node_counts:
        raise ValueError("node utilization requires direct round evidence")
    if len(actual_verified_node_counts) != len(selected_tree_node_counts):
        raise ValueError("actual and selected node evidence counts differ")
    if any(value <= 0 for value in actual_verified_node_counts):
        raise ValueError("actual verified node count must be positive")
    if any(value > configured_node_budget for value in actual_verified_node_counts):
        raise ValueError("actual verification exceeds configured node budget")
    total = sum(actual_verified_node_counts)
    round_count = len(actual_verified_node_counts)
    utilization = total / (round_count * configured_node_budget)
    return {
        "configured_node_budget": configured_node_budget,
        "round_count": round_count,
        "actual_verified_node_counts": actual_verified_node_counts,
        "total_actual_verified_node_count": total,
        "actual_verified_nodes_per_round": _distribution(actual_verified_node_counts),
        "actual_verified_node_count_distribution": _histogram(
            actual_verified_node_counts
        ),
        "actual_verified_node_count_normalized_distribution": (
            _normalized_histogram(actual_verified_node_counts)
        ),
        "selected_tree_node_counts": selected_tree_node_counts,
        "selected_tree_node_count_distribution": _histogram(selected_tree_node_counts),
        "selected_tree_equals_actual_verified_every_round": (
            selected_tree_node_counts == actual_verified_node_counts
        ),
        "node_budget_utilization_rate": utilization,
        "utilization_classification": (
            "full_utilization"
            if utilization == 1.0
            else "underutilized"
            if utilization < 1.0
            else "invalid_overutilized"
        ),
        "actual_work_source": "refinement-rounds.phase2d2.actual_verified_node_count",
        "configured_budget_not_substituted_for_actual_work": True,
    }


def classify_actual_work_regime(
    base_audit: dict[str, Any], refined_audit: dict[str, Any]
) -> dict[str, Any]:
    """Classify a pair from direct work distributions, never its N label."""

    base_stats = base_audit["actual_verified_nodes_per_round"]
    refined_stats = refined_audit["actual_verified_nodes_per_round"]
    direct_summary_equal = all(
        base_stats[field] == refined_stats[field]
        for field in ("mean", "median", "min", "max")
    )
    distribution_equal = (
        base_audit["actual_verified_node_count_normalized_distribution"]
        == refined_audit["actual_verified_node_count_normalized_distribution"]
    )
    if direct_summary_equal and distribution_equal:
        regime = "same_actual_verification_work"
    elif refined_stats["mean"] > base_stats["mean"]:
        regime = "budget_cap_expansion"
    else:
        regime = "different_actual_verification_work"
    return {
        "regime": regime,
        "base_actual_verified_nodes_per_round": base_stats,
        "refined_actual_verified_nodes_per_round": refined_stats,
        "direct_summary_equal": direct_summary_equal,
        "normalized_distribution_equal": distribution_equal,
        "configured_node_budget_used_to_infer_regime": False,
        "regime_evidence_source": "direct_actual_verified_node_count",
    }


def marginal_extra_node_yield(
    *,
    base_verified_per_round: float,
    refined_verified_per_round: float,
    base_accepted_per_round: float,
    refined_accepted_per_round: float,
    base_committed_per_round: float,
    refined_committed_per_round: float,
) -> dict[str, Any]:
    delta_verified = refined_verified_per_round - base_verified_per_round
    delta_accepted = refined_accepted_per_round - base_accepted_per_round
    delta_committed = refined_committed_per_round - base_committed_per_round
    available = delta_verified != 0.0
    return {
        "delta_actual_verified_nodes_per_round": delta_verified,
        "delta_accepted_draft_tokens_per_round": delta_accepted,
        "marginal_accepted_tokens_per_additional_verified_node": (
            delta_accepted / delta_verified if available else None
        ),
        "delta_committed_tokens_per_round": delta_committed,
        "marginal_committed_tokens_per_additional_verified_node": (
            delta_committed / delta_verified if available else None
        ),
        "marginal_yield_available": available,
        "marginal_yield_unavailable_reason": (
            None if available else "delta_actual_verified_nodes_per_round_is_zero"
        ),
    }


def mechanism_signal(
    *, selected_tree_coverage_gain: float | None, refined_only_recovery_count: int
) -> dict[str, Any]:
    if refined_only_recovery_count > 0:
        signal = "new_candidate_availability_signal"
        wording = (
            "refined-only Original-DFlash target recovery is present in the "
            "recorded candidate sets"
        )
    elif selected_tree_coverage_gain is not None and selected_tree_coverage_gain > 0:
        signal = "reranking_or_tree_reallocation_signal"
        wording = (
            "consistent with reranking/tree reallocation rather than refined-only "
            "target recovery"
        )
    else:
        signal = "no_observed_candidate_mechanism_gain"
        wording = "no positive selected-tree coverage mechanism signal observed"
    return {
        "mechanism_signal": signal,
        "descriptive_post_hoc_interpretation_only": True,
        "causal_exclusivity_claimed": False,
        "refined_only_recovery_zero_treated_as_failure": False,
        "interpretation": wording,
    }


def same_work_classification(regime_rows: list[dict[str, Any]]) -> dict[str, Any]:
    same_work = [
        row
        for row in regime_rows
        if row["regime"] == "same_actual_verification_work" and row["correctness_valid"]
    ]
    evidence_complete = bool(same_work) and all(
        row["coverage_absolute_gain"] is not None
        and row["accepted_per_verified_absolute_gain"] is not None
        for row in same_work
    )
    positive = [
        row
        for row in same_work
        if row["coverage_absolute_gain"] is not None
        and row["coverage_absolute_gain"] > 0
        and row["accepted_per_verified_absolute_gain"] is not None
        and row["accepted_per_verified_absolute_gain"] > 0
    ]
    if not evidence_complete:
        classification = "insufficient_same_work_evidence"
    elif len(positive) == len(same_work):
        classification = "consistent_positive_same_work"
    elif positive:
        classification = "mixed_same_work"
    else:
        classification = "no_positive_same_work_signal"
    return {
        "primary_same_work_classification": classification,
        "same_work_valid_pair_count": len(same_work),
        "same_work_joint_positive_pair_count": len(positive),
        "same_work_evidence_complete": evidence_complete,
        "significance_threshold_used": False,
    }


def _family_from_manifest(manifest: dict[str, Any]) -> str:
    phase = manifest["phase2d2"]
    if phase["candidate_merge_mode"] == BASE_FAMILY:
        return BASE_FAMILY
    family = phase.get("frozen_refinement_configuration_id")
    if family not in FROZEN_REFINEMENT_FAMILIES:
        raise ValueError(f"unpreregistered raw refinement family: {family!r}")
    return str(family)


def _load_raw_node_audits(run_root: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    audits = {}
    manifests = sorted(run_root.expanduser().resolve().rglob("manifest.json"))
    for manifest_path in manifests:
        root = manifest_path.parent
        manifest = _json(manifest_path)
        samples = _jsonl(root / "samples.jsonl")
        rounds = _jsonl(root / "refinement-rounds.jsonl")
        if len(samples) != 1 or not rounds:
            raise ValueError(f"{root}: H1.1 requires one complete raw cell")
        sample = samples[0]
        family = _family_from_manifest(manifest)
        context = int(sample["nominal_prompt_length"])
        budget = int(manifest["phase2d2"]["node_budget"])
        key = (family, context, budget)
        if key in audits:
            raise ValueError(f"duplicate raw H1 cell: {key}")
        actual = [int(row["phase2d2"]["actual_verified_node_count"]) for row in rounds]
        selected = [int(row["phase2d2"]["selected_tree_node_count"]) for row in rounds]
        audit = node_utilization_from_rounds(
            configured_node_budget=budget,
            actual_verified_node_counts=actual,
            selected_tree_node_counts=selected,
        )
        audits[key] = {
            "schema_version": SCHEMA_VERSION,
            "task": sample["task"],
            "sample_id": sample["sample_id"],
            "source_sample_id": sample["source_sample_id"],
            "nominal_prompt_length": context,
            "actual_prompt_length": int(sample["actual_prompt_length"]),
            "configuration_family": family,
            "configuration_id": sample["configuration_id"],
            "input_directory": str(root),
            **audit,
        }
    expected = {
        (family, context, budget)
        for family in EXPECTED_FAMILIES
        for context in EXPECTED_CONTEXT_LENGTHS
        for budget in EXPECTED_NODE_BUDGETS
    }
    if set(audits) != expected:
        raise ValueError(
            "raw H1 node matrix mismatch; "
            f"missing={sorted(expected - set(audits))}, "
            f"extra={sorted(set(audits) - expected)}"
        )
    return audits


def _index(
    rows: list[dict[str, Any]], *fields: str
) -> dict[tuple[Any, ...], dict[str, Any]]:
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"duplicate artifact identity: {key}")
        result[key] = row
    return result


def _source_metadata(aggregate_dir: Path) -> dict[str, Any]:
    names = (
        "phase2d2a-h-known-manifest.json",
        "phase2d2a-h-known-paired-comparison.json",
        "phase2d2a-h-known-config-summary.json",
        "phase2d2a-h-known-cost-summary.json",
        "phase2d2a-h-known-decision.json",
    )
    return {
        "h1_aggregate_directory": str(aggregate_dir),
        "artifact_sha256": {name: _sha256(aggregate_dir / name) for name in names},
        "existing_h1_artifacts_overwritten": False,
    }


def _cost_decomposition(cells: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for cell in cells:
        if cell["configuration_family"] == BASE_FAMILY:
            continue
        cost = cell["component_cost"]
        conditional = cost["conditional_remask_share_of_profiled_round_time"]
        verifier = cost["fixed_shape_recheck_share_of_profiled_round_time"]
        combined = cost["combined_additional_work_share_of_profiled_round_time"]
        largest = (
            "correctness_preserving_verifier_cost"
            if verifier > conditional
            else "dependency_modeling_cost"
            if conditional > verifier
            else "equal_measured_share"
        )
        rows.append(
            {
                "refined_family": cell["configuration_family"],
                "nominal_prompt_length": cell["nominal_prompt_length"],
                "node_budget": cell["node_budget"],
                "dependency_modeling_cost_share": conditional,
                "correctness_preserving_verifier_cost_share": verifier,
                "combined_additional_work_share": combined,
                "largest_cost_component": largest,
            }
        )
    return {
        "rows": rows,
        "dependency_modeling_cost_share_distribution": _distribution(
            [row["dependency_modeling_cost_share"] for row in rows]
        ),
        "correctness_preserving_verifier_cost_share_distribution": _distribution(
            [row["correctness_preserving_verifier_cost_share"] for row in rows]
        ),
        "combined_additional_work_share_distribution": _distribution(
            [row["combined_additional_work_share"] for row in rows]
        ),
        "largest_cost_component_distribution": dict(
            sorted(Counter(row["largest_cost_component"] for row in rows).items())
        ),
        "dependency_modeling_and_verifier_costs_separated": True,
        "lightweight_causal_adapter_claimed_to_solve_fixed_shape_verifier_overhead": False,
    }


def _h2_preregistered_design(*, direct_regime_confirmed: bool) -> dict[str, Any] | None:
    if not direct_regime_confirmed:
        return None
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "2D.2A-H2-preregistration",
        "status": "preregistered_not_executed",
        "emitted_only_after_direct_h1_node_audit": True,
        "direct_h1_regime_confirmation": True,
        "primary_h2_mechanism_evaluation": {
            "node_budget": 16,
            "regime": "same_actual_verification_work",
        },
        "secondary_h2_budget_expansion_ablation": {
            "node_budget": 32,
            "regime": "budget_cap_expansion",
        },
        "frozen_refinement_families": list(FROZEN_REFINEMENT_FAMILIES),
        "frozen_family_selected_or_removed": False,
        "winner_selected_from_h1": False,
        "verification_mode": EXPECTED_VERIFICATION_MODE,
        "candidate_label_authority": CANDIDATE_LABEL_AUTHORITY,
        "candidate_label_source": CANDIDATE_LABEL_SOURCE,
        "q1_replay_used_as_target_label_source": False,
        "h2_execution_authorized": False,
        "non_executing_design_recommendation_only": True,
    }


def build_h11_artifacts(
    *,
    h1_manifest: dict[str, Any],
    h1_paired: dict[str, Any],
    h1_config: dict[str, Any],
    h1_decision: dict[str, Any],
    raw_audits: dict[tuple[str, int, int], dict[str, Any]],
    source: dict[str, Any],
) -> dict[str, Any]:
    cells = h1_manifest["cells"]
    cell_by_key = _index(
        cells, "configuration_family", "nominal_prompt_length", "node_budget"
    )
    pairs = h1_paired["rows"]
    pair_by_key = _index(
        pairs, "refined_family", "nominal_prompt_length", "node_budget"
    )
    original_config = _index(h1_config["rows"], "refined_family")
    for key, raw in raw_audits.items():
        cell = cell_by_key.get(key)
        if cell is None:
            raise ValueError(f"raw node audit lacks H1 aggregate cell: {key}")
        online = cell["online_efficiency"]
        if raw["round_count"] != online["round_count"]:
            raise ValueError(f"raw/H1 round count mismatch: {key}")
        if (
            raw["total_actual_verified_node_count"]
            != online["verified_target_node_count"]
        ):
            raise ValueError(f"raw/H1 verified-node total mismatch: {key}")

    regime_rows = []
    mechanism_rows = []
    for family in FROZEN_REFINEMENT_FAMILIES:
        for context in EXPECTED_CONTEXT_LENGTHS:
            for budget in EXPECTED_NODE_BUDGETS:
                pair_key = (family, context, budget)
                pair = pair_by_key[pair_key]
                base_audit = raw_audits[(BASE_FAMILY, context, budget)]
                refined_audit = raw_audits[pair_key]
                regime = classify_actual_work_regime(base_audit, refined_audit)
                base_cell = cell_by_key[(BASE_FAMILY, context, budget)]
                refined_cell = cell_by_key[pair_key]
                base_online = base_cell["online_efficiency"]
                refined_online = refined_cell["online_efficiency"]
                marginal = marginal_extra_node_yield(
                    base_verified_per_round=base_audit[
                        "actual_verified_nodes_per_round"
                    ]["mean"],
                    refined_verified_per_round=refined_audit[
                        "actual_verified_nodes_per_round"
                    ]["mean"],
                    base_accepted_per_round=base_online[
                        "accepted_draft_tokens_per_round"
                    ],
                    refined_accepted_per_round=refined_online[
                        "accepted_draft_tokens_per_round"
                    ],
                    base_committed_per_round=base_online["committed_tokens_per_round"],
                    refined_committed_per_round=refined_online[
                        "committed_tokens_per_round"
                    ],
                )
                accepted_gain = pair["online_efficiency"][
                    "accepted_draft_tokens_per_verified_node"
                ]["absolute_gain"]
                coverage_gain = pair["candidate_quality"]["coverage_absolute_gain"]
                regime_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "task": KNOWN_TASK,
                        "sample_id": KNOWN_SAMPLE_ID,
                        "refined_family": family,
                        "nominal_prompt_length": context,
                        "node_budget": budget,
                        "correctness_valid": pair["correctness_valid"],
                        **regime,
                        **marginal,
                        "coverage_absolute_gain": coverage_gain,
                        "accepted_per_verified_absolute_gain": accepted_gain,
                        "accepted_per_verified_relative_gain": pair[
                            "online_efficiency"
                        ]["accepted_draft_tokens_per_verified_node"]["relative_gain"],
                    }
                )
                quality = refined_cell["candidate_quality"]
                signal = mechanism_signal(
                    selected_tree_coverage_gain=coverage_gain,
                    refined_only_recovery_count=int(
                        quality["refined_only_target_recovery_count"]
                    ),
                )
                mechanism_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "task": KNOWN_TASK,
                        "sample_id": KNOWN_SAMPLE_ID,
                        "refined_family": family,
                        "nominal_prompt_length": context,
                        "node_budget": budget,
                        "candidate_quality_available_round_count": quality[
                            "candidate_quality_available_round_count"
                        ],
                        "candidate_quality_unavailable_round_count": quality[
                            "candidate_quality_unavailable_round_count"
                        ],
                        "labelled_reference_position_count": quality[
                            "labelled_reference_position_count"
                        ],
                        "unavailable_reason_distribution": quality[
                            "unavailable_reason_distribution"
                        ],
                        "base_candidate_coverage_rate": quality[
                            "base_candidate_coverage_rate"
                        ],
                        "conditional_candidate_coverage_rate": quality[
                            "conditional_candidate_coverage_rate"
                        ],
                        "merged_candidate_coverage_rate": quality[
                            "merged_candidate_coverage_rate"
                        ],
                        "selected_tree_coverage_rate": quality[
                            "selected_tree_coverage_rate_at_fixed_n"
                        ],
                        "selected_tree_coverage_absolute_gain_vs_base_cell": (
                            coverage_gain
                        ),
                        "refined_only_target_recovery_count": quality[
                            "refined_only_target_recovery_count"
                        ],
                        "refined_only_target_recovery_rate": quality[
                            "refined_only_target_recovery_rate"
                        ],
                        "target_token_rank_improvement_distribution": quality[
                            "target_token_rank_improvement_distribution"
                        ],
                        "target_token_rank_improvement_histogram": quality.get(
                            "target_token_rank_improvement_histogram", {}
                        ),
                        "correct_prefix_survival_depth_distribution": quality[
                            "correct_prefix_survival_depth_distribution"
                        ],
                        "correct_prefix_survival_depth_histogram": quality.get(
                            "correct_prefix_survival_depth_histogram", {}
                        ),
                        "candidate_label_authority": CANDIDATE_LABEL_AUTHORITY,
                        "candidate_label_source": CANDIDATE_LABEL_SOURCE,
                        "q1_replay_used_as_target_label_source": False,
                        "unavailable_rows_excluded_from_denominators": True,
                        **signal,
                    }
                )

    n16_same = all(
        row["regime"] == "same_actual_verification_work"
        for row in regime_rows
        if row["node_budget"] == 16
    )
    n32_expansion = all(
        row["regime"] == "budget_cap_expansion"
        for row in regime_rows
        if row["node_budget"] == 32
    )
    direct_regime_confirmed = n16_same and n32_expansion
    utilization_groups = {}
    for family_kind, families in (
        ("base", (BASE_FAMILY,)),
        ("refined", FROZEN_REFINEMENT_FAMILIES),
    ):
        for budget in EXPECTED_NODE_BUDGETS:
            group = [
                row["node_budget_utilization_rate"]
                for key, row in raw_audits.items()
                if key[0] in families and key[2] == budget
            ]
            utilization_groups[f"{family_kind}_n{budget}"] = _distribution(group)

    budget_artifact = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": "complete",
        "source": source,
        "direct_actual_work_field": "phase2d2.actual_verified_node_count",
        "configured_budget_not_substituted_for_actual_work": True,
        "matrix_cell_count": len(raw_audits),
        "rows": [raw_audits[key] for key in sorted(raw_audits)],
        "utilization_groups": utilization_groups,
        "pair_regimes": regime_rows,
        "all_n16_pairs_same_actual_verification_work": n16_same,
        "all_n32_pairs_budget_cap_expansion": n32_expansion,
        "h1_direct_regime_hypothesis_confirmed": direct_regime_confirmed,
    }
    mechanism_artifact = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": "complete",
        "source": source,
        "candidate_label_authority": CANDIDATE_LABEL_AUTHORITY,
        "candidate_label_source": CANDIDATE_LABEL_SOURCE,
        "q1_replay_used_as_target_label_source": False,
        "unavailable_candidate_rows_excluded_from_denominators": True,
        "causal_exclusivity_claimed": False,
        "rows": mechanism_rows,
        "mechanism_signal_distribution": dict(
            sorted(Counter(row["mechanism_signal"] for row in mechanism_rows).items())
        ),
    }
    config_rows = []
    for family in FROZEN_REFINEMENT_FAMILIES:
        family_regimes = [row for row in regime_rows if row["refined_family"] == family]
        original = original_config[(family,)]["known_sample_stability_classification"]
        config_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "refined_family": family,
                "original_cap_based_h1_classification": original,
                "original_h1_classification_preserved_unchanged": True,
                **same_work_classification(family_regimes),
                "frozen_family_selected_or_removed": False,
            }
        )
    costs = _cost_decomposition(cells)
    latency_rows = []
    for row in pairs:
        context = row["nominal_prompt_length"]
        budget = row["node_budget"]
        base_timing = cell_by_key[(BASE_FAMILY, context, budget)]["formal_timing"]
        refined_timing = cell_by_key[(row["refined_family"], context, budget)][
            "formal_timing"
        ]
        latency_rows.append(
            {
                "refined_family": row["refined_family"],
                "nominal_prompt_length": context,
                "node_budget": budget,
                "base_formal_timing": dict(base_timing),
                "refined_formal_timing": dict(refined_timing),
                "formal_pair_comparison": dict(row["formal_timing"]),
                "nominal_latency_relative_change": row["formal_timing"][
                    "cuda_ms_per_returned_token_relative_delta"
                ],
                "formal_timing_repeat_count": base_timing["repeat_count"],
                "physical_sample_count": 1,
                "robust_speedup_claimed": False,
            }
        )
    reinterpretation = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": "complete",
        "source": source,
        "original_h1_final_phase2d2a_decision": h1_decision["final_phase2d2a_decision"],
        "final_phase2d2a_decision": "insufficient_evidence",
        "original_cap_based_classifications_preserved": True,
        "primary_same_work_classification_is_secondary_descriptive_view": True,
        "winner_selected": False,
        "frozen_refinement_families": list(FROZEN_REFINEMENT_FAMILIES),
        "frozen_family_selected_or_removed": False,
        "configuration_rows": config_rows,
        "regime_rows": regime_rows,
        "cost_decomposition": costs,
        "latency_interpretation": {
            "rows": latency_rows,
            "interpretation_label": "nominal_latency_change",
            "robust_speedup_claimed": False,
            "reason": "one_physical_sample_and_three_formal_repeats",
            "raw_formal_timing_values_preserved_in_h1_source": True,
        },
        "legacy_mixed_known_sample_classification_deleted_or_rewritten": False,
    }
    h2 = _h2_preregistered_design(direct_regime_confirmed=direct_regime_confirmed)
    reinterpretation_csv_rows = []
    config_by_family = {row["refined_family"]: row for row in config_rows}
    cost_by_key = {
        (
            row["refined_family"],
            row["nominal_prompt_length"],
            row["node_budget"],
        ): row
        for row in costs["rows"]
    }
    latency_by_key = {
        (
            row["refined_family"],
            row["nominal_prompt_length"],
            row["node_budget"],
        ): row
        for row in latency_rows
    }
    for row in regime_rows:
        key = (
            row["refined_family"],
            row["nominal_prompt_length"],
            row["node_budget"],
        )
        reinterpretation_csv_rows.append(
            {
                **row,
                **config_by_family[row["refined_family"]],
                **cost_by_key[key],
                **latency_by_key[key],
            }
        )
    return {
        "budget": budget_artifact,
        "mechanism": mechanism_artifact,
        "reinterpretation": reinterpretation,
        "reinterpretation_csv_rows": reinterpretation_csv_rows,
        "h2": h2,
    }


def audit_h1_root(h1_root: Path) -> dict[str, Any]:
    h1_root = h1_root.expanduser().resolve()
    aggregate_dir = h1_root / "aggregate"
    run_root = h1_root / "runs"
    check_h1_output(aggregate_dir)
    manifest = _json(aggregate_dir / "phase2d2a-h-known-manifest.json")
    paired = _json(aggregate_dir / "phase2d2a-h-known-paired-comparison.json")
    config = _json(aggregate_dir / "phase2d2a-h-known-config-summary.json")
    decision = _json(aggregate_dir / "phase2d2a-h-known-decision.json")
    raw = _load_raw_node_audits(run_root)
    return build_h11_artifacts(
        h1_manifest=manifest,
        h1_paired=paired,
        h1_config=config,
        h1_decision=decision,
        raw_audits=raw,
        source=_source_metadata(aggregate_dir),
    )


def _flat(value: Any) -> Any:
    return (
        json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
    )


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row}) or ["schema_version"]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _flat(row.get(field)) for field in fields})
    temporary.replace(path)


def persist_h11(output_dir: Path, artifacts: dict[str, Any]) -> None:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty H1.1 output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "phase2d2a-h1-budget-utilization.json": artifacts["budget"],
        "phase2d2a-h1-mechanism-decomposition.json": artifacts["mechanism"],
        "phase2d2a-h1-reinterpretation.json": artifacts["reinterpretation"],
    }
    if artifacts["h2"] is not None:
        payloads[H2_ARTIFACT] = artifacts["h2"]
    for name, payload in payloads.items():
        _validate_schema(payload, SCHEMAS[name])
        _write_json(output_dir / name, payload)
    _write_csv(
        output_dir / "phase2d2a-h1-budget-utilization.csv",
        artifacts["budget"]["rows"],
    )
    _write_csv(
        output_dir / "phase2d2a-h1-mechanism-decomposition.csv",
        artifacts["mechanism"]["rows"],
    )
    _write_csv(
        output_dir / "phase2d2a-h1-reinterpretation.csv",
        artifacts["reinterpretation_csv_rows"],
    )


def _display(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _print_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    rendered = [[_display(value) for value in row] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in rendered))
        for index, header in enumerate(headers)
    ]
    print(
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    )
    print("  ".join("-" * width for width in widths))
    for row in rendered:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def print_h11_report(artifacts: dict[str, Any]) -> None:
    """Print the requested audit views without changing source H1 artifacts."""

    budget = artifacts["budget"]
    print("\nDIRECT NODE UTILIZATION")
    _print_table(
        (
            "family",
            "context",
            "N",
            "rounds",
            "total",
            "mean",
            "median",
            "min",
            "max",
            "selected_distribution",
            "utilization",
        ),
        [
            (
                row["configuration_family"],
                row["nominal_prompt_length"],
                row["configured_node_budget"],
                row["round_count"],
                row["total_actual_verified_node_count"],
                row["actual_verified_nodes_per_round"]["mean"],
                row["actual_verified_nodes_per_round"]["median"],
                row["actual_verified_nodes_per_round"]["min"],
                row["actual_verified_nodes_per_round"]["max"],
                row["selected_tree_node_count_distribution"],
                row["node_budget_utilization_rate"],
            )
            for row in budget["rows"]
        ],
    )

    print("\nPAIR REGIME AND EXTRA-NODE MARGINAL YIELD")
    _print_table(
        (
            "family",
            "context",
            "N",
            "regime",
            "delta_nodes/round",
            "delta_accepted/round",
            "accepted/additional_node",
            "committed/additional_node",
        ),
        [
            (
                row["refined_family"],
                row["nominal_prompt_length"],
                row["node_budget"],
                row["regime"],
                row["delta_actual_verified_nodes_per_round"],
                row["delta_accepted_draft_tokens_per_round"],
                row["marginal_accepted_tokens_per_additional_verified_node"],
                row["marginal_committed_tokens_per_additional_verified_node"],
            )
            for row in budget["pair_regimes"]
        ],
    )

    print("\nMECHANISM DECOMPOSITION")
    _print_table(
        (
            "family",
            "context",
            "N",
            "base_cov",
            "conditional_cov",
            "merged_cov",
            "selected_cov",
            "refined_only_count",
            "refined_only_rate",
            "mechanism_signal",
        ),
        [
            (
                row["refined_family"],
                row["nominal_prompt_length"],
                row["node_budget"],
                row["base_candidate_coverage_rate"],
                row["conditional_candidate_coverage_rate"],
                row["merged_candidate_coverage_rate"],
                row["selected_tree_coverage_rate"],
                row["refined_only_target_recovery_count"],
                row["refined_only_target_recovery_rate"],
                row["mechanism_signal"],
            )
            for row in artifacts["mechanism"]["rows"]
        ],
    )

    reinterpretation = artifacts["reinterpretation"]
    print("\nORIGINAL AND SAME-WORK CLASSIFICATIONS")
    _print_table(
        ("family", "original_H1", "primary_same_work", "same_work_pairs"),
        [
            (
                row["refined_family"],
                row["original_cap_based_h1_classification"],
                row["primary_same_work_classification"],
                row["same_work_valid_pair_count"],
            )
            for row in reinterpretation["configuration_rows"]
        ],
    )

    cost = reinterpretation["cost_decomposition"]
    print("\nCOST DECOMPOSITION")
    _print_table(
        ("component", "min_share", "median_share", "max_share"),
        [
            (
                label,
                cost[field]["min"],
                cost[field]["median"],
                cost[field]["max"],
            )
            for label, field in (
                ("dependency_modeling", "dependency_modeling_cost_share_distribution"),
                (
                    "correctness_preserving_verifier",
                    "correctness_preserving_verifier_cost_share_distribution",
                ),
                (
                    "combined_additional_work",
                    "combined_additional_work_share_distribution",
                ),
            )
        ],
    )
    print(
        "largest_cost_component_distribution:",
        _display(cost["largest_cost_component_distribution"]),
    )
    print("final_phase2d2a_decision:", reinterpretation["final_phase2d2a_decision"])
    print("robust_speedup_claimed:", False)
    if artifacts["h2"] is None:
        print("H2 preregistration: not emitted; direct regime hypothesis not confirmed")
    else:
        print(
            "H2 preregistration: emitted as non-executing design; "
            "N16 primary same-work, N32 secondary budget-expansion; no winner selected"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h1-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = audit_h1_root(args.h1_root)
    persist_h11(args.output_dir, artifacts)
    print_h11_report(artifacts)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "matrix_cell_count": artifacts["budget"]["matrix_cell_count"],
                "h1_direct_regime_hypothesis_confirmed": artifacts["budget"][
                    "h1_direct_regime_hypothesis_confirmed"
                ],
                "h2_preregistered_design_emitted": artifacts["h2"] is not None,
                "final_phase2d2a_decision": artifacts["reinterpretation"][
                    "final_phase2d2a_decision"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
