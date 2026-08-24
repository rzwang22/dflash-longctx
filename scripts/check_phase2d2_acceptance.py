#!/usr/bin/env python3
"""Validate Phase 2D.2A Original-DFlash/tree and structural invariants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = REPO_ROOT / "docs" / "schemas"
REQUIRED = (
    "manifest.json",
    "environment.json",
    "source-provenance.json",
    "workload-inventory.json",
    "samples.jsonl",
    "refinement-rounds.jsonl",
    "candidate-merge-events.jsonl",
    "refined-tree-rounds.jsonl",
    "refinement-summary.json",
    "refinement-summary.csv",
    "candidate-quality-summary.csv",
    "fixed-budget-comparison.json",
    "fixed-budget-comparison.csv",
    "robust-verification-events.jsonl",
    "verifier-oracle-comparisons.jsonl",
    "verifier-oracle-summary.json",
    "tree-traversal-checks.jsonl",
    "fallback-summary.json",
    "fallback-summary.csv",
    "correctness-summary.json",
    "output-divergence.json",
    "output-divergence.csv",
    "offline-online-comparison.json",
    "offline-online-comparison.csv",
    "timing-summary.json",
    "timing-summary.csv",
    "phase2d2-decision.json",
    "terminal.log",
)
Q1_REPLAY_MODES = {"always_replay", "margin_guarded_replay"}
FIXED_SHAPE_MODES = {
    "always_fixed_shape_recheck",
    "margin_guarded_fixed_shape_recheck",
}
CANDIDATE_LABEL_AUTHORITY = "original_dflash_reference"
CANDIDATE_LABEL_SOURCE = "original_dflash_absolute_continuation"
FROZEN_REFINEMENT_CONFIGURATION_SET = [
    "conditional_rerank-d1-p1-r2-w0.5",
    "union-d2-p2-r2-w0.5",
    "union-d4-p2-r4-w0.5",
]


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL {path.name}:{line_number}") from exc
    return rows


def _validate(payload: Any, schema_name: str) -> None:
    schema = _json(SCHEMA_ROOT / schema_name)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)


def _validate_gold_kv_shape_evidence(row: dict[str, Any]) -> None:
    """Reject absent or internally inconsistent authoritative KV shapes."""

    if row.get("availability") != "available":
        raise ValueError("KV Gold evidence is unavailable")
    required_shape_fields = {
        "candidate_shape",
        "reference_shape",
        "shape_metadata_available",
        "shape_layer_count_matches",
        "layer_shape_matches",
        "all_layer_shapes_match",
    }
    missing_shape_fields = sorted(required_shape_fields - set(row))
    if missing_shape_fields:
        raise ValueError(
            f"KV Gold shape metadata is incomplete: {missing_shape_fields}"
        )
    if not row["shape_metadata_available"]:
        raise ValueError("KV Gold shapes are unavailable")
    candidate = row["candidate_shape"]
    reference = row["reference_shape"]
    reported_matches = row["layer_shape_matches"]
    if not candidate or not reference:
        raise ValueError("KV Gold shape lists are empty")
    if len(candidate) != len(reference) or not row["shape_layer_count_matches"]:
        raise ValueError("KV Gold layer counts differ")
    if len(reported_matches) != len(candidate):
        raise ValueError("KV Gold per-layer shape evidence is incomplete")
    for layer_index, (left, right, reported) in enumerate(
        zip(candidate, reference, reported_matches)
    ):
        if set(left) != {"k", "v"} or set(right) != {"k", "v"}:
            raise ValueError(f"KV Gold layer {layer_index} lacks K/V")
        expected = {
            "layer_index": layer_index,
            "k": left["k"] == right["k"],
            "v": left["v"] == right["v"],
        }
        if reported != expected:
            raise ValueError(
                f"KV Gold layer {layer_index} shape flags disagree"
            )
        if not expected["k"] or not expected["v"]:
            raise ValueError(f"KV Gold layer {layer_index} K/V shapes differ")
    if not row["all_layer_shapes_match"]:
        raise ValueError("KV Gold shapes do not all match")
    if row.get("structural_status") != "pass":
        raise ValueError("KV Gold structural status did not pass")


def _validate_candidate_quality_provenance(
    row: dict[str, Any], *, sample: dict[str, Any]
) -> None:
    quality = row["candidate_quality"]
    _validate(quality, "phase2d2-candidate-quality.schema.json")
    if quality["target_label_authority"] != CANDIDATE_LABEL_AUTHORITY:
        raise ValueError("candidate-quality target-label authority is not Original DFlash")
    if quality["target_label_source"] != CANDIDATE_LABEL_SOURCE:
        raise ValueError("candidate-quality target-label source is not absolute continuation")
    if quality["label_authority"] != quality["target_label_authority"] or quality[
        "label_source"
    ] != quality["target_label_source"]:
        raise ValueError("candidate-quality label provenance aliases disagree")
    if quality["target_label_used_for_runtime_selection"] or quality[
        "target_labels_used_for_runtime_selection"
    ]:
        raise ValueError("candidate-quality target labels entered runtime selection")
    if quality["q1_replay_used_as_target_label_source"] or quality[
        "target_only_used_as_target_label_source"
    ]:
        raise ValueError("an auxiliary numerical control became a target-label source")
    if quality["semantic_generated_prefix_length"] != int(
        row["generated_root_offset"]
    ):
        raise ValueError("candidate-quality semantic prefix offset is inconsistent")
    if quality["availability"] == "unavailable":
        if quality["reference_alignment_available"]:
            if quality["reference_alignment_reason"] is not None:
                raise ValueError("available reference alignment has a failure reason")
        elif not quality["reference_alignment_reason"]:
            raise ValueError("unavailable reference alignment lacks a reason")
        if (
            quality["target_label_evidence_available"]
            or not quality["target_label_evidence_unavailable_reason"]
            or quality["positions"]
        ):
            raise ValueError("unavailable candidate-quality provenance is incomplete")
        return
    if not quality["reference_alignment_available"]:
        raise ValueError("available candidate-quality row lacks reference alignment")
    if not quality["target_label_evidence_available"]:
        raise ValueError("available candidate-quality row lacks target labels")

    reference = sample["original_dflash_reference"]["generated_output_token_ids"]
    prefix = int(row["generated_root_offset"])
    selected_paths = {
        tuple(int(value) for value in node.get("path", []))
        for node in row["phase2d2"]["tree_spec"]["nodes"]
    }
    expected_survival = 0
    first_pruning = None
    for position in quality["positions"]:
        depth = int(position["position_index"])
        offset = int(position["absolute_generated_token_offset"])
        if offset != prefix + depth or offset >= len(reference):
            raise ValueError("candidate-quality absolute target-label offset is invalid")
        if int(position["target_token_id"]) != int(reference[offset]):
            raise ValueError("candidate-quality target token is not Original DFlash")
        expected_parent = [int(value) for value in reference[prefix + 1 : offset]]
        expected_path = [int(value) for value in reference[prefix + 1 : offset + 1]]
        if position["reference_parent_prefix_token_ids"] != expected_parent:
            raise ValueError("candidate-quality reference parent prefix is inconsistent")
        if position["reference_prefix_through_position_token_ids"] != expected_path:
            raise ValueError("candidate-quality reference path is inconsistent")
        expected_selected = tuple(expected_path) in selected_paths
        if position["selected_tree_membership"] is not expected_selected:
            raise ValueError("selected-tree coverage does not follow reference path")
        if expected_selected and first_pruning is None:
            expected_survival += 1
        elif first_pruning is None:
            first_pruning = depth
    if quality["correct_prefix_survival_depth"] != expected_survival:
        raise ValueError("correct-prefix survival depth is inconsistent")
    if quality["first_pruning_depth"] != first_pruning:
        raise ValueError("first pruning depth is inconsistent")


def _same_evidence_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        left.get(field) == right.get(field)
        for field in (
            "task",
            "sample_id",
            "nominal_prompt_length",
            "configuration_id",
        )
    )


def _validate_round_fallback_contract(row: dict[str, Any]) -> None:
    """Validate mode-specific fallback geometry and authoritative cache state."""

    phase = row["phase2d2"]
    mode = phase["verification_mode"]
    fallback_executed = phase.get(
        "fallback_executed", phase["fallback_replay_executed"]
    )
    executor = phase.get(
        "fallback_executor",
        "q1_replay" if phase["fallback_replay_executed"] else "none",
    )
    query_lengths = phase["fallback_query_lengths"]

    if mode in Q1_REPLAY_MODES:
        if fallback_executed:
            if executor != "q1_replay" or not query_lengths:
                raise ValueError("legacy replay fallback executor is inconsistent")
            if any(value != 1 for value in query_lengths):
                raise ValueError("fallback replay used a query length other than one")
            if not phase["fallback_replay_executed"]:
                raise ValueError("legacy replay execution was not preserved")
        if mode == "always_replay" and not fallback_executed:
            raise ValueError("always_replay did not execute authoritative replay")
        return

    if mode not in FIXED_SHAPE_MODES:
        if fallback_executed or executor != "none":
            raise ValueError("packed_native unexpectedly executed a fallback")
        return

    expected_execution = (
        True
        if mode == "always_fixed_shape_recheck"
        else bool(phase["guard_triggered"])
    )
    if fallback_executed != expected_execution:
        raise ValueError("fixed-shape fallback execution disagrees with its mode")
    if phase["fixed_shape_fallback_triggered"] != expected_execution:
        raise ValueError("fixed-shape fallback trigger record is inconsistent")
    if not expected_execution:
        if executor != "none" or phase["final_output_source"] != "packed_native":
            raise ValueError("non-triggered margin-fixed round changed packed commit")
        if row["cache_compaction"].get("committed_cache_source") != "packed_native":
            raise ValueError("non-triggered margin-fixed cache was not packed-native")
        return

    if executor != "fixed_shape_recheck":
        raise ValueError("fixed-shape mode selected the wrong fallback executor")
    if phase["fallback_replay_executed"]:
        raise ValueError("fixed-shape fallback was mislabeled as q1 replay")
    if query_lengths != [16] or phase["fixed_shape_query_lengths"] != [16]:
        raise ValueError("fixed-shape fallback did not use one q_len=16 forward")
    if phase["fallback_forward_count"] != 1 or phase["fixed_shape_forward_count"] != 1:
        raise ValueError("fixed-shape fallback target-forward count is not one")
    prefix = int(row["prefix_cache_length_before_round"])
    committed = int(row["committed_token_count"])
    expected_cropped = prefix + committed
    temporary_lengths = phase["fixed_shape_temporary_kv_layer_lengths"]
    if (
        phase["fixed_shape_temporary_block_length"] != 16
        or not temporary_lengths
        or any(int(length) != prefix + 16 for length in temporary_lengths)
    ):
        raise ValueError("fixed-shape temporary KV is not a full 16-token block")
    if (
        phase["fixed_shape_cropped_committed_kv_length"] != expected_cropped
        or phase["authoritative_committed_cache_length"] != expected_cropped
    ):
        raise ValueError("fixed-shape authoritative cache crop length is wrong")
    expected_tail = 15 - int(row["accepted_draft_token_count"])
    if phase["fixed_shape_mask_tail_discarded_count"] != expected_tail:
        raise ValueError("fixed-shape MASK-tail discard count is wrong")
    if phase["authoritative_committed_target_hidden_length"] != committed:
        raise ValueError("fixed-shape target_hidden is not root plus accepted path")
    if (
        phase["authoritative_next_root_absolute_sequence_position"]
        != expected_cropped
        or phase["authoritative_next_root_transition_slot_count"] != 1
        or phase["authoritative_next_root_matches_output_transition"] is not True
    ):
        raise ValueError("fixed-shape next-root transition is inconsistent")
    if phase["fixed_shape_round_entry_snapshot_isolated"] is not True:
        raise ValueError("fixed-shape round-entry cache isolation was not proven")
    if phase["fixed_shape_snapshot_matches_preverification_cache"] is not True:
        raise ValueError("fixed-shape snapshot does not match pre-verification KV")
    if not phase["packed_temporary_kv_discarded"]:
        raise ValueError("packed temporary KV was not discarded")
    if phase["final_output_source"] != "fixed_shape_recheck":
        raise ValueError("fixed-shape authoritative output provenance is wrong")
    if (
        row["cache_compaction"].get("committed_cache_source")
        != "phase2d2_fixed_shape_recheck"
    ):
        raise ValueError("fixed-shape authoritative cache provenance is wrong")
    if phase["fixed_shape_mask_tail_absent_from_committed_cache"] is not True:
        raise ValueError("fixed-shape MASK-tail KV remained after crop")
    if phase["correction_or_bonus_committed_in_current_round"]:
        raise ValueError("fixed-shape correction/bonus entered current KV")
    if not phase["sibling_kv_removed_from_committed_cache"]:
        raise ValueError("fixed-shape authoritative cache retained sibling KV")
    layer_lengths = phase["authoritative_committed_target_kv_layer_lengths"]
    if not layer_lengths or any(
        int(length) != expected_cropped for length in layer_lengths
    ):
        raise ValueError("fixed-shape target KV layer lengths disagree")


def _causal_gold_kv_skip_evidence(
    row: dict[str, Any],
    *,
    manifest: dict[str, Any],
    samples: list[dict[str, Any]],
    gold_logits: list[dict[str, Any]],
    traversal: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate one exact downstream KV skip after confirmed logit evidence."""

    failures = []
    if row.get("availability") != "unavailable":
        failures.append("availability_is_not_unavailable")
    if row.get("status") != "not_run":
        failures.append("status_is_not_not_run")
    if row.get("execution_status") != "not_run_due_to_prior_failure":
        failures.append("execution_status_is_not_prior_failure")
    if row.get("reason") != "prior_logits_gold_failure":
        failures.append("reason_is_not_prior_logits_gold_failure")
    if row.get("comparison_mode") != "dense_committed_path":
        failures.append("unavailable_check_is_not_downstream_kv_gold")
    if manifest.get("status") != "ok" or manifest.get("error") is not None:
        failures.append("run_has_infrastructure_or_manifest_failure")

    matching_samples = [
        sample for sample in samples if _same_evidence_identity(row, sample)
    ]
    if len(matching_samples) != 1:
        failures.append("sample_correctness_evidence_is_missing_or_ambiguous")
    else:
        correctness = matching_samples[0].get("correctness", {})
        required_exactness = (
            "original_dflash_reference_output_ids_equal",
            "formal_repeat_output_ids_equal",
            "control_profile_output_ids_equal",
            "control_diagnostic_output_ids_equal",
            "actual_execution_correctness_passed",
            "structural_checks_passed",
            "passed",
        )
        failures.extend(
            f"sample_{field}_did_not_pass"
            for field in required_exactness
            if correctness.get(field) is not True
        )

    matching_traversal = [
        item
        for item in traversal
        if _same_evidence_identity(row, item)
        and item.get("round_index") == row.get("round_index")
    ]
    if len(matching_traversal) != 1:
        failures.append("round_traversal_evidence_is_missing_or_ambiguous")
    else:
        traversal_row = matching_traversal[0]
        checks = traversal_row.get("checks", {})
        if traversal_row.get("status") != "pass":
            failures.append("round_traversal_status_did_not_pass")
        for field in (
            "structural_checks_passed",
            "traversal_checks_passed",
            "kv_root_transition_checks_passed",
            "all_checks_passed",
        ):
            if checks.get(field) is not True:
                failures.append(f"round_traversal_{field}_did_not_pass")

    matching_logits = [
        item
        for item in gold_logits
        if _same_evidence_identity(row, item)
        and item.get("round_index") == row.get("round_index")
    ]
    required_modes = set(manifest.get("phase2d2", {}).get("gold_comparison_modes", []))
    observed_modes = {item.get("comparison_mode") for item in matching_logits}
    if not required_modes or not required_modes.issubset(observed_modes):
        failures.append("required_gold_logit_modes_are_missing")
    if not matching_logits:
        failures.append("gold_logit_evidence_is_missing")
    elif any(item.get("availability") != "available" for item in matching_logits):
        failures.append("gold_logit_evidence_is_unavailable")
    elif any(
        item.get("structural_status") != "pass"
        or bool(item.get("structural_issues"))
        for item in matching_logits
    ):
        failures.append("gold_logit_structural_evidence_failed")
    elif not any(item.get("status") == "fail_semantic" for item in matching_logits):
        failures.append("prior_gold_logit_failure_is_not_present")

    return {
        "accepted": not failures,
        "round_index": row.get("round_index"),
        "reason": row.get("reason"),
        "failures": failures,
        "required_gold_logit_modes": sorted(required_modes),
        "observed_gold_logit_modes": sorted(
            value for value in observed_modes if value is not None
        ),
    }


def check_output(output: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED if not (output / name).is_file()]
    if missing:
        raise ValueError(f"missing Phase 2D.2A artifacts: {missing}")
    manifest = _json(output / "manifest.json")
    provenance = _json(output / "source-provenance.json")
    correctness = _json(output / "correctness-summary.json")
    samples = _jsonl(output / "samples.jsonl")
    rounds = _jsonl(output / "refinement-rounds.jsonl")
    merge_events = _jsonl(output / "candidate-merge-events.jsonl")
    tree_rounds = _jsonl(output / "refined-tree-rounds.jsonl")
    robust_events = _jsonl(output / "robust-verification-events.jsonl")
    decision = _json(output / "phase2d2-decision.json")
    output_divergence = _json(output / "output-divergence.json")
    summary = _json(output / "refinement-summary.json")
    offline_online = _json(output / "offline-online-comparison.json")
    gold_logits_path = output / "gold-logit-checks.jsonl"
    gold_logits = _jsonl(gold_logits_path) if gold_logits_path.is_file() else []
    gold_kv_path = output / "gold-kv-checks.jsonl"
    gold_kv = _jsonl(gold_kv_path) if gold_kv_path.is_file() else []
    traversal = _jsonl(output / "tree-traversal-checks.jsonl")
    verifier_oracle = _jsonl(output / "verifier-oracle-comparisons.jsonl")
    verifier_oracle_summary = _json(output / "verifier-oracle-summary.json")
    if manifest.get("status") not in {"ok", "failed", "error"}:
        raise ValueError(
            f"incomplete Phase 2D.2A manifest status: {manifest.get('status')!r}"
        )
    if not samples or not rounds:
        raise ValueError("Phase 2D.2A run has no sample/round evidence")
    _validate(manifest, "phase2d2-manifest.schema.json")
    _validate(correctness, "phase2d2-correctness-summary.schema.json")
    _validate(decision, "phase2d2-decision.schema.json")
    _validate(summary, "phase2d2-summary.schema.json")
    _validate(
        _json(output / "fixed-budget-comparison.json"),
        "phase2d2-fixed-budget-comparison.schema.json",
    )
    _validate(
        _json(output / "fallback-summary.json"),
        "phase2d2-fallback-summary.schema.json",
    )
    _validate(
        output_divergence,
        "phase2d2-output-divergence.schema.json",
    )
    _validate(
        offline_online,
        "phase2d2-offline-online.schema.json",
    )
    _validate(
        _json(output / "timing-summary.json"),
        "phase2d2-timing-summary.schema.json",
    )
    for row in samples:
        _validate(row, "phase2d2-sample.schema.json")
        if (
            row["original_dflash_reference"]["generated_output_token_ids"]
            != row["generated_output_token_ids"]
        ):
            raise ValueError("sample Original-DFlash/refined-tree token IDs differ")
        if not row["correctness"]["passed"]:
            raise ValueError("sample hard correctness record did not pass")
        if row["candidate_merge_mode"] == "base_only" and not row[
            "correctness"
        ].get("base_only_reference_tree_output_ids_equal"):
            raise ValueError(
                "base_only Original-DFlash/tree outputs are not token exact"
            )
        expected_target_warnings = []
        if row["target_only"]["generated_output_token_ids"] != row[
            "original_dflash_reference"
        ]["generated_output_token_ids"]:
            expected_target_warnings.append(
                "target_only_reference_numerical_divergence"
            )
        if (
            row["target_only"]["generated_output_token_ids"]
            != row["generated_output_token_ids"]
        ):
            expected_target_warnings.append("target_only_tree_numerical_divergence")
        if (
            row["correctness"].get("target_only_auxiliary_warnings")
            != expected_target_warnings
        ):
            raise ValueError("target-only auxiliary output status is inconsistent")
    sample_by_identity = {
        (
            row["task"],
            row["sample_id"],
            int(row["nominal_prompt_length"]),
            row["configuration_id"],
        ): row
        for row in samples
    }
    phase_config = manifest["phase2d2"]
    candidate_provenance_present = (
        phase_config.get("candidate_quality_target_label_authority") is not None
    )
    if candidate_provenance_present:
        expected_phase_provenance = {
            "candidate_quality_target_label_authority": CANDIDATE_LABEL_AUTHORITY,
            "candidate_quality_target_label_source": CANDIDATE_LABEL_SOURCE,
            "candidate_quality_target_labels_used_for_runtime_selection": False,
            "candidate_quality_q1_replay_used_as_target_label_source": False,
            "candidate_quality_evidence_role": "post_hoc_evaluation_only",
            "frozen_refinement_configuration_set": (
                FROZEN_REFINEMENT_CONFIGURATION_SET
            ),
            "frozen_refinement_configuration_set_reselected": False,
        }
        for field, expected in expected_phase_provenance.items():
            if phase_config.get(field) != expected:
                raise ValueError(f"manifest candidate-label provenance mismatch: {field}")
        expected_artifact_provenance = {
            "target_label_authority": CANDIDATE_LABEL_AUTHORITY,
            "target_label_source": CANDIDATE_LABEL_SOURCE,
            "mixed_target_label_authorities": False,
            "q1_replay_used_as_target_label_source": False,
            "frozen_refinement_configuration_set_reselected": False,
        }
        for field, expected in expected_artifact_provenance.items():
            if offline_online.get(field) != expected:
                raise ValueError(f"offline candidate-label provenance mismatch: {field}")
        expected_summary_provenance = {
            "candidate_quality_target_label_authority": CANDIDATE_LABEL_AUTHORITY,
            "candidate_quality_target_label_source": CANDIDATE_LABEL_SOURCE,
            "candidate_quality_mixed_label_authorities": False,
            "candidate_quality_target_labels_used_for_runtime_selection": False,
        }
        for field, expected in expected_summary_provenance.items():
            if summary.get(field) != expected:
                raise ValueError(f"summary candidate-label provenance mismatch: {field}")
        if any(
            aggregate.get("candidate_quality_target_label_authority")
            != CANDIDATE_LABEL_AUTHORITY
            or aggregate.get("candidate_quality_target_label_source")
            != CANDIDATE_LABEL_SOURCE
            or aggregate.get(
                "candidate_quality_target_labels_used_for_runtime_selection"
            )
            is not False
            for aggregate in summary.get("aggregates", [])
        ):
            raise ValueError("aggregate candidate-label provenance is mixed")
        if provenance.get("candidate_quality_target_label_authority") != (
            CANDIDATE_LABEL_AUTHORITY
        ) or provenance.get("candidate_quality_target_label_source") != (
            CANDIDATE_LABEL_SOURCE
        ):
            raise ValueError("source provenance lacks Original-DFlash candidate labels")
        if len(offline_online.get("rows", [])) != len(rounds):
            raise ValueError("offline candidate-quality row count is incomplete")
        offline_by_identity = {
            (
                row["task"],
                row["sample_id"],
                int(row["nominal_prompt_length"]),
                row["configuration_id"],
                int(row["round_index"]),
            ): row
            for row in offline_online["rows"]
        }
        for round_row in rounds:
            identity = (
                round_row["task"],
                round_row["sample_id"],
                int(round_row["nominal_prompt_length"]),
                round_row["configuration_id"],
                int(round_row["round_index"]),
            )
            offline_row = offline_by_identity.get(identity)
            if offline_row is None or any(
                offline_row.get(field)
                != round_row["candidate_quality"].get(field)
                for field in (
                    "target_label_authority",
                    "target_label_source",
                    "target_label_used_for_runtime_selection",
                    "semantic_generated_prefix_length",
                    "semantic_prefix_token_sha256",
                    "reference_alignment_available",
                    "reference_alignment_reason",
                    "target_label_evidence_available",
                    "target_label_evidence_unavailable_reason",
                )
            ):
                raise ValueError("offline candidate-quality provenance is inconsistent")

    for row in rounds:
        _validate(row, "phase2d2-round.schema.json")
        phase = row["phase2d2"]
        if phase["initial_parallel_draft_forward_count"] != 1:
            raise ValueError("initial parallel draft forward count is not one")
        if phase["conditional_refinement_forward_count"] > 1:
            raise ValueError("more than one conditional forward occurred in a round")
        if (
            phase["candidate_merge_mode"] == "base_only"
            and phase["conditional_refinement_forward_count"] != 0
        ):
            raise ValueError("base_only control executed a conditional forward")
        if phase["candidate_merge_mode"] == "base_only" and (
            not phase["base_only_conditional_refinement_bypassed"]
            or phase["actual_verification_tree_source"]
            != "phase2c_frozen_selector_seed_tree"
            or phase["refined_tree_replaced_seed_tree"]
        ):
            raise ValueError("base_only did not preserve the Phase 2C seed-tree path")
        if phase["internal_branch_batch_size"] != len(
            phase.get("conditional_input_token_ids", [])
        ):
            raise ValueError("internal branch batch/input-row count mismatch")
        if (
            phase["actual_verified_node_count"]
            > phase["fixed_root_inclusive_node_budget"]
        ):
            raise ValueError("target verification exceeded fixed root-inclusive N")
        if phase["actual_verified_node_count"] != phase["selected_tree_node_count"]:
            raise ValueError("selected and actually verified node counts differ")
        if phase["root_count"] != 1 or not phase["node_budget_invariant_passed"]:
            raise ValueError("root/node-budget invariant failed")
        if phase["temporary_refinement_state_committed"]:
            raise ValueError("temporary refinement state was committed")
        if candidate_provenance_present:
            identity = (
                row["task"],
                row["sample_id"],
                int(row["nominal_prompt_length"]),
                row["configuration_id"],
            )
            sample = sample_by_identity.get(identity)
            if sample is None:
                raise ValueError("candidate-quality round lacks matching sample")
            _validate_candidate_quality_provenance(row, sample=sample)
        _validate_round_fallback_contract(row)
        if phase["correction_or_bonus_committed_in_current_round"]:
            raise ValueError("correction/bonus was committed in the current round")
        if not phase["sibling_kv_removed_from_committed_cache"]:
            raise ValueError("sibling KV remained in the authoritative cache")
        layer_lengths = phase["authoritative_committed_target_kv_layer_lengths"]
        if not layer_lengths or any(
            int(length) != int(phase["authoritative_committed_cache_length"])
            for length in layer_lengths
        ):
            raise ValueError("authoritative target KV layer lengths disagree")
    for row in merge_events:
        _validate(row, "phase2d2-candidate-merge-event.schema.json")
    for row in tree_rounds:
        _validate(row, "phase2d2-refined-tree-round.schema.json")
    for row in robust_events:
        _validate(row, "phase2d2-robust-verification-event.schema.json")
    for row in verifier_oracle:
        _validate(row, "phase2d2-verifier-oracle-comparison.schema.json")
    _validate(
        verifier_oracle_summary,
        "phase2d2-verifier-oracle-summary.schema.json",
    )
    causal_gold_kv_skips = []
    if manifest["phase2d2"].get("gold_checks_enabled"):
        if not gold_kv:
            raise ValueError("enabled Gold checks produced no KV evidence")
        if any(row.get("availability") != "available" for row in gold_logits):
            raise ValueError("Gold logit evidence is unavailable")
        for row in gold_kv:
            _validate(row, "phase2c-gold-kv-check.schema.json")
            if row.get("availability") == "available":
                _validate_gold_kv_shape_evidence(row)
                continue
            causal = _causal_gold_kv_skip_evidence(
                row,
                manifest=manifest,
                samples=samples,
                gold_logits=gold_logits,
                traversal=traversal,
            )
            if not causal["accepted"]:
                raise ValueError(
                    "KV Gold evidence is unavailable without a complete causal "
                    f"downstream-skip proof: {causal['failures']}"
                )
            causal_gold_kv_skips.append(causal)
    required_pairs = (
        ("target_only", "original_dflash_reference"),
        ("target_only", "refined_tree"),
        ("original_dflash_reference", "refined_tree"),
    )
    for left_name, right_name in required_pairs:
        matches = [
            row
            for row in output_divergence["records"]
            if row.get("left_name") == left_name
            and row.get("right_name") == right_name
        ]
        if len(matches) != len(samples):
            raise ValueError(
                f"absolute-token relation {left_name}/{right_name} is incomplete"
            )
        if (
            left_name == "original_dflash_reference"
            and right_name == "refined_tree"
            and any(not row.get("output_ids_equal") for row in matches)
        ):
            raise ValueError(
                "Original-DFlash/refined-tree absolute-token relation diverged"
            )
    if not correctness.get("all_original_dflash_reference_output_ids_equal"):
        raise ValueError("Original-DFlash/refined-tree output token mismatch")
    if not correctness.get("structural_correctness_passed"):
        raise ValueError(
            "structural correctness failed: "
            f"{correctness.get('hard_correctness_failure_reasons', [])}"
        )
    if not correctness.get("official_checker_passed"):
        raise ValueError("Phase 2D.2A hard correctness gate did not pass")
    return {
        "status": "accepted",
        "sample_count": len(samples),
        "round_count": len(rounds),
        "decision": decision["decision"],
        "causal_downstream_gold_kv_skip_count": len(causal_gold_kv_skips),
        "causal_downstream_gold_kv_skips": causal_gold_kv_skips,
        "candidate_quality_label_provenance": (
            "original_dflash_absolute_continuation"
            if candidate_provenance_present
            else "legacy_artifact_without_explicit_label_provenance"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = check_output(args.output_dir.expanduser().resolve())
    except Exception as exc:  # noqa: BLE001 - clean checker failure.
        print(f"REJECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
