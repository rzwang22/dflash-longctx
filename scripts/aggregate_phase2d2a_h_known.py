#!/usr/bin/env python3
"""Aggregate the preregistered Phase 2D.2A-H1 known-sample matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from scripts.check_phase2d2_acceptance import (
    REQUIRED as PHASE2D2_REQUIRED_ARTIFACTS,
)
from scripts.check_phase2d2_acceptance import check_output

SCHEMA_VERSION = "1.0"
PHASE = "2D.2A-H1"
KNOWN_TASK = "gov_report"
KNOWN_SAMPLE_ID = "gov_report-fd287bbfd3bdbdc9"
EXPECTED_CONTEXT_LENGTHS = (1024, 16384)
EXPECTED_NODE_BUDGETS = (16, 32)
BASE_FAMILY = "base_only"
FROZEN_REFINEMENT_FAMILIES = (
    "conditional_rerank-d1-p1-r2-w0.5",
    "union-d2-p2-r2-w0.5",
    "union-d4-p2-r4-w0.5",
)
EXPECTED_FAMILIES = (BASE_FAMILY, *FROZEN_REFINEMENT_FAMILIES)
EXPECTED_VERIFICATION_MODE = "always_fixed_shape_recheck"
EXPECTED_SELECTOR_MODE = "top1_packed_tree"
EXPECTED_DTYPE = "bfloat16"
EXPECTED_BLOCK_SIZE = 16
EXPECTED_MAX_NEW_TOKENS = 128
EXPECTED_TIMING_REPEATS = 3
EXPECTED_WARMUP_RUNS = 1
EXPECTED_BATCH_SIZE = 1

ARTIFACTS = (
    "phase2d2a-h-known-manifest.json",
    "phase2d2a-h-known-paired-comparison.json",
    "phase2d2a-h-known-paired-comparison.csv",
    "phase2d2a-h-known-config-summary.json",
    "phase2d2a-h-known-config-summary.csv",
    "phase2d2a-h-known-cost-summary.json",
    "phase2d2a-h-known-cost-summary.csv",
    "phase2d2a-h-known-decision.json",
)

SCHEMAS = {
    "phase2d2a-h-known-manifest.json": "phase2d2a-h-known-manifest.schema.json",
    "phase2d2a-h-known-paired-comparison.json": (
        "phase2d2a-h-known-paired-comparison.schema.json"
    ),
    "phase2d2a-h-known-config-summary.json": (
        "phase2d2a-h-known-config-summary.schema.json"
    ),
    "phase2d2a-h-known-cost-summary.json": (
        "phase2d2a-h-known-cost-summary.schema.json"
    ),
    "phase2d2a-h-known-decision.json": "phase2d2a-h-known-decision.schema.json",
}

COMMON_PROVENANCE_FIELDS = (
    "repository_commit",
    "target_model_id",
    "target_model_revision",
    "draft_model_id",
    "draft_model_revision",
    "materialization_id",
    "selector_hash",
    "online_selector_mode",
    "verification_mode",
    "dtype",
    "device",
    "block_size",
    "max_new_tokens",
    "timing_repeats",
    "warmup_runs",
    "batch_size",
    "temperature",
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL: {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise TypeError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def _validate_schema(payload: Any, schema_name: str) -> None:
    schema = _read_json(REPO_ROOT / "docs" / "schemas" / schema_name)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    return float(value)


def _optional_number(value: Any, field: str) -> float | None:
    return None if value is None else _number(value, field)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _relative_delta(base: float | None, refined: float | None) -> float | None:
    if base is None or refined is None or float(base) == 0.0:
        return None
    return (float(refined) - float(base)) / float(base)


def _absolute_delta(base: float | None, refined: float | None) -> float | None:
    if base is None or refined is None:
        return None
    return float(refined) - float(base)


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]


def _distribution(values: list[float | int | None]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean) if clean else None,
        "median": statistics.median(clean) if clean else None,
        "p90": _p90(clean),
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
    }


def _histogram(values: list[Any], *, none_label: str | None = None) -> dict[str, int]:
    labels = []
    for value in values:
        if value is None:
            if none_label is not None:
                labels.append(none_label)
            continue
        labels.append(str(value))
    return dict(sorted(Counter(labels).items()))


def _sign(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value > 0:
        return "+"
    if value < 0:
        return "-"
    return "0"


def _latency_sign(delta: float | None) -> str:
    if delta is None:
        return "unavailable"
    if delta < 0:
        return "faster"
    if delta > 0:
        return "slower"
    return "neutral"


def _eligible_value(source: dict[str, Any], field: str, *, eligible: bool) -> Any:
    return source[field] if eligible else None


def _family(manifest: dict[str, Any]) -> str:
    phase = manifest["phase2d2"]
    if phase["candidate_merge_mode"] == BASE_FAMILY:
        if phase.get("frozen_refinement_configuration_id") is not None:
            raise ValueError("base_only cell unexpectedly selects a refinement family")
        return BASE_FAMILY
    family = phase.get("frozen_refinement_configuration_id")
    if family not in FROZEN_REFINEMENT_FAMILIES:
        raise ValueError(f"unpreregistered refinement family: {family!r}")
    return str(family)


def _repository_commit(manifest: dict[str, Any]) -> str | None:
    repository = manifest.get("repository", {})
    return repository.get("commit_sha") or repository.get("commit")


def _provenance(manifest: dict[str, Any]) -> dict[str, Any]:
    config = manifest["configuration"]
    phase = manifest["phase2d2"]
    return {
        "repository_commit": _repository_commit(manifest),
        "target_model_id": config.get("target_model"),
        "target_model_revision": config.get("target_revision"),
        "draft_model_id": config.get("draft_model"),
        "draft_model_revision": config.get("draft_revision"),
        "materialization_id": manifest["materialization"].get("id"),
        "selector_hash": phase.get("frozen_selector_hash"),
        "frozen_refinement_hash": phase.get("frozen_refinement_config_hash"),
        "online_selector_mode": phase.get("online_selector_mode"),
        "verification_mode": phase.get("verification_mode"),
        "dtype": config.get("dtype"),
        "device": config.get("device"),
        "block_size": config.get("block_size"),
        "max_new_tokens": config.get("max_new_tokens"),
        "timing_repeats": config.get("timing_repeats"),
        "warmup_runs": config.get("warmup_runs_per_context_length"),
        "batch_size": config.get("batch_size"),
        "temperature": config.get("temperature"),
    }


def _validate_preregistered_run(
    root: Path,
    manifest: dict[str, Any],
    sample: dict[str, Any],
    family: str,
) -> None:
    phase = manifest["phase2d2"]
    config = manifest["configuration"]
    if manifest.get("phase") != "2D.2A":
        raise ValueError(f"{root}: not a Phase 2D.2A result")
    if phase.get("frozen_refinement_configuration_set_reselected") is not False:
        raise ValueError(f"{root}: frozen refinement set was reselected")
    if tuple(phase.get("frozen_refinement_configuration_set", ())) != (
        FROZEN_REFINEMENT_FAMILIES
    ):
        raise ValueError(f"{root}: frozen refinement set differs from H1")
    if phase.get("candidate_quality_target_label_authority") != (
        CANDIDATE_LABEL_AUTHORITY
    ) or phase.get("candidate_quality_target_label_source") != (
        CANDIDATE_LABEL_SOURCE
    ):
        raise ValueError(f"{root}: candidate label authority is not Original DFlash")
    if phase.get("candidate_quality_q1_replay_used_as_target_label_source") is not False:
        raise ValueError(f"{root}: q1 entered candidate-quality labels")
    if phase.get("candidate_quality_target_labels_used_for_runtime_selection") is not False:
        raise ValueError(f"{root}: post-hoc labels entered runtime selection")
    expected = {
        "online_selector_mode": EXPECTED_SELECTOR_MODE,
        "verification_mode": EXPECTED_VERIFICATION_MODE,
        "dtype": EXPECTED_DTYPE,
        "device": "cuda:0",
        "block_size": EXPECTED_BLOCK_SIZE,
        "max_new_tokens": EXPECTED_MAX_NEW_TOKENS,
        "timing_repeats": EXPECTED_TIMING_REPEATS,
        "warmup_runs_per_context_length": EXPECTED_WARMUP_RUNS,
        "batch_size": EXPECTED_BATCH_SIZE,
        "temperature": 0.0,
    }
    observed = {**config, **phase}
    for field, value in expected.items():
        if observed.get(field) != value:
            raise ValueError(
                f"{root}: {field} must equal {value!r}, got {observed.get(field)!r}"
            )
    if sample.get("task") != KNOWN_TASK or sample.get("sample_id") != KNOWN_SAMPLE_ID:
        raise ValueError(f"{root}: H1 accepts only the preregistered known sample")
    if int(sample.get("nominal_prompt_length", 0)) not in EXPECTED_CONTEXT_LENGTHS:
        raise ValueError(f"{root}: unexpected H1 context length")
    if int(phase.get("node_budget", 0)) not in EXPECTED_NODE_BUDGETS:
        raise ValueError(f"{root}: unexpected H1 node budget")
    if int(sample.get("node_budget", 0)) != int(phase["node_budget"]):
        raise ValueError(f"{root}: sample/manifest node budgets differ")
    if tuple(config.get("context_lengths", ())) != (
        int(sample["nominal_prompt_length"]),
    ):
        raise ValueError(f"{root}: each H1 directory must configure exactly one cell")
    if len(sample.get("formal_timing_repeats", ())) != EXPECTED_TIMING_REPEATS:
        raise ValueError(f"{root}: formal timing repeat evidence is incomplete")
    if sample.get("verification_mode") != EXPECTED_VERIFICATION_MODE:
        raise ValueError(f"{root}: sample verifier is not always-fixed")
    if family == BASE_FAMILY:
        if phase.get("candidate_merge_mode") != BASE_FAMILY:
            raise ValueError(f"{root}: base family did not use base_only")
        if phase.get("frozen_refinement_config_hash") is not None:
            raise ValueError(f"{root}: base control unexpectedly selected frozen refinement")
    elif phase.get("frozen_refinement_config_hash") is None:
        raise ValueError(f"{root}: refined cell lacks frozen refinement hash")


def _candidate_metrics(rounds: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    available = []
    unavailable = []
    positions: list[dict[str, Any]] = []
    for row in rounds:
        quality = row.get("candidate_quality")
        if not isinstance(quality, dict):
            raise TypeError(f"{root}: candidate_quality must be an object")
        if quality.get("target_label_authority") != CANDIDATE_LABEL_AUTHORITY:
            raise ValueError(f"{root}: candidate-quality authority changed")
        if quality.get("target_label_source") != CANDIDATE_LABEL_SOURCE:
            raise ValueError(f"{root}: candidate-quality source changed")
        if quality.get("q1_replay_used_as_target_label_source") is not False:
            raise ValueError(f"{root}: q1 entered candidate-quality metrics")
        if quality.get("target_label_used_for_runtime_selection") is not False:
            raise ValueError(f"{root}: candidate labels entered runtime selection")
        if quality.get("availability") == "available":
            current_positions = quality.get("positions")
            if not isinstance(current_positions, list) or not current_positions:
                raise ValueError(f"{root}: available candidate evidence has no positions")
            available.append(quality)
            positions.extend(current_positions)
        elif quality.get("availability") == "unavailable":
            unavailable.append(quality)
        else:
            raise ValueError(f"{root}: unknown candidate-quality availability")

    labelled = len(positions)
    base_count = sum(bool(row.get("base_top_m_membership")) for row in positions)
    conditional_count = sum(
        bool(row.get("conditional_top_r_membership")) for row in positions
    )
    merged_count = sum(bool(row.get("merged_membership")) for row in positions)
    selected_count = sum(
        bool(row.get("selected_tree_membership")) for row in positions
    )
    recovered_count = sum(
        bool(row.get("refined_only_target_recovery")) for row in positions
    )
    pruning = [row.get("first_pruning_depth") for row in available]
    survival = [row.get("correct_prefix_survival_depth") for row in available]
    rank_improvements = [
        row.get("target_token_rank_improvement") for row in positions
    ]
    return {
        "candidate_quality_total_round_count": len(rounds),
        "candidate_quality_available_round_count": len(available),
        "candidate_quality_unavailable_round_count": len(unavailable),
        "candidate_label_available_round_rate": _safe_ratio(len(available), len(rounds)),
        "labelled_reference_position_count": labelled,
        "unavailable_reason_distribution": dict(
            sorted(Counter(str(row.get("unavailable_reason")) for row in unavailable).items())
        ),
        "selected_tree_coverage_rate_at_fixed_n": _safe_ratio(selected_count, labelled),
        "base_candidate_coverage_rate": _safe_ratio(base_count, labelled),
        "conditional_candidate_coverage_rate": _safe_ratio(
            conditional_count, labelled
        ),
        "merged_candidate_coverage_rate": _safe_ratio(merged_count, labelled),
        "refined_only_target_recovery_count": recovered_count,
        "refined_only_target_recovery_rate": _safe_ratio(recovered_count, labelled),
        "correct_prefix_survival_depth_distribution": _distribution(
            survival
        ),
        "correct_prefix_survival_depth_histogram": _histogram(survival),
        "first_pruning_depth_distribution": _distribution(pruning),
        "first_pruning_depth_histogram": _histogram(
            pruning, none_label="not_pruned_within_labelled_horizon"
        ),
        "first_pruning_not_observed_count": sum(value is None for value in pruning),
        "target_token_rank_improvement_distribution": _distribution(rank_improvements),
        "target_token_rank_improvement_histogram": _histogram(rank_improvements),
        "target_label_authority": CANDIDATE_LABEL_AUTHORITY,
        "target_label_source": CANDIDATE_LABEL_SOURCE,
        "q1_replay_used_as_target_label_source": False,
        "target_labels_used_for_runtime_selection": False,
        "zero_placeholder_rows_excluded_from_denominators": True,
    }


def _online_metrics(sample: dict[str, Any], rounds: list[dict[str, Any]]) -> dict[str, Any]:
    returned = int(sample["generated_token_count"])
    round_count = len(rounds)
    accepted = sum(int(row["accepted_draft_token_count"]) for row in rounds)
    committed = sum(int(row["committed_token_count"]) for row in rounds)
    verified = sum(
        int(row["phase2d2"]["actual_verified_node_count"]) for row in rounds
    )
    return {
        "generated_returned_token_count": returned,
        "round_count": round_count,
        "accepted_draft_token_count": accepted,
        "committed_token_count": committed,
        "verified_target_node_count": verified,
        "accepted_draft_tokens_per_verified_node": _safe_ratio(accepted, verified),
        "accepted_draft_tokens_per_round": _safe_ratio(accepted, round_count),
        "committed_tokens_per_round": _safe_ratio(committed, round_count),
        "verified_target_nodes_per_returned_token": _safe_ratio(verified, returned),
        "rounds_per_returned_token": _safe_ratio(round_count, returned),
    }


def _formal_timing(sample: dict[str, Any]) -> dict[str, Any]:
    repeats = [
        _number(row.get("cuda_elapsed_ms"), "formal_timing_repeats.cuda_elapsed_ms")
        for row in sample["formal_timing_repeats"]
    ]
    returned = int(sample["generated_token_count"])
    median = statistics.median(repeats)
    mean = statistics.fmean(repeats)
    return {
        "formal_timing_source": "unprofiled_control_repeats",
        "repeat_count": len(repeats),
        "cuda_elapsed_ms_repeats": repeats,
        "generation_cuda_median_ms": median,
        "generation_cuda_mean_ms": mean,
        "generation_cuda_p90_ms": _p90(repeats),
        "generation_cuda_min_ms": min(repeats),
        "generation_cuda_max_ms": max(repeats),
        "generation_cuda_range_ms": max(repeats) - min(repeats),
        "generation_cuda_ms_per_returned_token": _safe_ratio(median, returned),
        "returned_tokens_per_second": _safe_ratio(returned * 1000.0, median),
        "diagnostic_or_gold_latency_used": False,
    }


def _component_cost(
    sample: dict[str, Any], rounds: list[dict[str, Any]]
) -> dict[str, Any]:
    components = sample["profiling"]["components"]
    conditional_total = _optional_number(
        components.get("conditional_remask_cuda_elapsed_ms"),
        "profile conditional_remask_cuda_elapsed_ms",
    )
    fixed_total = _optional_number(
        components.get("canonical_recheck_cuda_elapsed_ms"),
        "profile canonical_recheck_cuda_elapsed_ms",
    )
    round_total = _optional_number(
        components.get("round_cuda_elapsed_ms"), "profile round_cuda_elapsed_ms"
    )
    initial_total = _optional_number(
        components.get("draft_hidden_logits_cuda_elapsed_ms"),
        "profile draft_hidden_logits_cuda_elapsed_ms",
    )
    total_draft = _optional_number(
        components.get("total_draft_cuda_elapsed_ms"),
        "profile total_draft_cuda_elapsed_ms",
    )
    packed_total = _optional_number(
        components.get("verification_cuda_elapsed_ms"),
        "profile verification_cuda_elapsed_ms",
    )
    total_verification = _optional_number(
        components.get("total_verification_cuda_elapsed_ms"),
        "profile total_verification_cuda_elapsed_ms",
    )
    snapshot_total = _optional_number(
        components.get("stabilization_cache_snapshot_cuda_elapsed_ms"),
        "profile stabilization_cache_snapshot_cuda_elapsed_ms",
    )
    replacement_total = _optional_number(
        components.get("kv_replacement_cuda_elapsed_ms"),
        "profile kv_replacement_cuda_elapsed_ms",
    )
    conditional_round_values = [
        _optional_number(
            row["phase2d2"].get("conditional_remask_cuda_elapsed_ms"),
            "round conditional_remask_cuda_elapsed_ms",
        )
        for row in rounds
    ]
    fixed_round_values = [
        _optional_number(
            row["phase2d2"].get("fallback_cuda_elapsed_ms"),
            "round fallback_cuda_elapsed_ms",
        )
        for row in rounds
    ]
    formal_forward_count = sum(
        int(row["phase2d2"]["formal_target_forward_count"]) for row in rounds
    )
    combined = (
        conditional_total + fixed_total
        if conditional_total is not None and fixed_total is not None
        else None
    )
    extended_fixed_overhead = (
        (snapshot_total or 0.0) + fixed_total + (replacement_total or 0.0)
        if fixed_total is not None
        else None
    )
    return {
        "component_timing_source": "separate_profile_and_diagnostics_paths",
        "conditional_remask_cuda_median_ms_per_round": _distribution(
            conditional_round_values
        )["median"],
        "conditional_remask_profile_total_cuda_ms": conditional_total,
        "conditional_remask_share_of_profiled_round_time": (
            _safe_ratio(conditional_total, round_total)
            if conditional_total is not None and round_total is not None
            else None
        ),
        "fixed_shape_recheck_cuda_median_ms_per_round": _distribution(
            fixed_round_values
        )["median"],
        "fixed_shape_recheck_profile_total_cuda_ms": fixed_total,
        "fixed_shape_recheck_share_of_profiled_round_time": (
            _safe_ratio(fixed_total, round_total)
            if fixed_total is not None and round_total is not None
            else None
        ),
        "fixed_shape_round_entry_snapshot_profile_total_cuda_ms": snapshot_total,
        "fixed_shape_kv_replacement_profile_total_cuda_ms": replacement_total,
        "extended_fixed_shape_path_profile_total_cuda_ms": extended_fixed_overhead,
        "extended_fixed_shape_path_share_of_profiled_round_time": (
            _safe_ratio(extended_fixed_overhead, round_total)
            if extended_fixed_overhead is not None and round_total is not None
            else None
        ),
        "fallback_timer_excludes_snapshot_and_kv_replacement": True,
        "combined_conditional_and_fixed_profile_total_cuda_ms": combined,
        "combined_additional_work_share_of_profiled_round_time": (
            _safe_ratio(combined, round_total)
            if combined is not None and round_total is not None
            else None
        ),
        "profile_initial_parallel_draft_cuda_ms": initial_total,
        "profile_total_draft_cuda_ms": total_draft,
        "profile_packed_verification_cuda_ms": packed_total,
        "profile_total_verification_cuda_ms": total_verification,
        "profile_round_cuda_ms": round_total,
        "formal_target_forward_count": formal_forward_count,
        "formal_target_forward_count_per_round": _safe_ratio(
            formal_forward_count, len(rounds)
        ),
        "conditional_over_initial_draft_ratio": None,
        "conditional_over_initial_draft_ratio_unavailable_reason": (
            "timer_scopes_are_non_overlapping_but_not_directly_comparable:"
            "conditional_timer_includes_input_cache_and_candidate_work_while_"
            "initial_timer_covers_draft_forward_and_lm_head"
        ),
        "nested_timer_double_counting_avoided": True,
        "combined_sum_contains_only_non_overlapping_conditional_and_fixed_recheck": True,
    }


def _correctness(
    root: Path,
    manifest: dict[str, Any],
    correctness: dict[str, Any],
    sample: dict[str, Any],
) -> dict[str, Any]:
    repeat_stable = all(
        row.get("output_ids_equal_repeat_0") is True
        for row in sample["formal_timing_repeats"]
    )
    checks = {
        "original_dflash_reference_tree_output_exact": bool(
            sample["correctness"].get("original_dflash_reference_output_ids_equal")
            and correctness.get("all_original_dflash_reference_output_ids_equal")
        ),
        "formal_repeat_stable": repeat_stable,
        "profile_formal_output_exact": sample["profiling"].get(
            "output_ids_equal_formal"
        )
        is True,
        "diagnostic_formal_output_exact": sample["diagnostics"].get(
            "output_ids_equal_formal"
        )
        is True,
        "structural_correctness_passed": bool(
            sample["correctness"].get("structural_checks_passed")
            and correctness.get("structural_correctness_passed")
        ),
        "actual_execution_correctness_passed": bool(
            sample["correctness"].get("actual_execution_correctness_passed")
            and correctness.get("actual_execution_correctness_passed")
        ),
    }
    try:
        checker_result = check_output(root)
    except Exception as exc:  # noqa: BLE001 - persist an invalid cell cleanly.
        checker_status = "rejected"
        checker_error = f"{type(exc).__name__}: {exc}"
        checker_result = None
    else:
        checker_status = "accepted"
        checker_error = None
    declared_hard_correctness = all(checks.values())
    if declared_hard_correctness and checker_status != "accepted":
        raise ValueError(
            f"{root}: checker rejected a cell whose hard correctness fields pass: "
            f"{checker_error}"
        )
    return {
        "runner_status": manifest.get("status"),
        "checker_status": checker_status,
        "checker_result": checker_result,
        "checker_error": checker_error,
        **checks,
        "correctness_valid": declared_hard_correctness
        and checker_status == "accepted",
        "hard_correctness_failure_reasons": correctness.get(
            "hard_correctness_failure_reasons", []
        ),
        "target_only_is_auxiliary": True,
        "target_only_output_ids_equal": sample["correctness"].get(
            "target_only_output_ids_equal"
        ),
    }


def _load_cell(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    missing = [name for name in PHASE2D2_REQUIRED_ARTIFACTS if not (root / name).is_file()]
    if missing:
        raise ValueError(f"{root}: incomplete Phase 2D.2A directory: {missing}")
    manifest = _read_json(root / "manifest.json")
    correctness = _read_json(root / "correctness-summary.json")
    offline = _read_json(root / "offline-online-comparison.json")
    timing = _read_json(root / "timing-summary.json")
    samples = _read_jsonl(root / "samples.jsonl")
    rounds = _read_jsonl(root / "refinement-rounds.jsonl")
    _validate_schema(manifest, "phase2d2-manifest.schema.json")
    _validate_schema(correctness, "phase2d2-correctness-summary.schema.json")
    _validate_schema(offline, "phase2d2-offline-online.schema.json")
    _validate_schema(timing, "phase2d2-timing-summary.schema.json")
    if len(samples) != 1:
        raise ValueError(f"{root}: H1 requires exactly one cell/sample per directory")
    sample = samples[0]
    _validate_schema(sample, "phase2d2-sample.schema.json")
    for row in rounds:
        _validate_schema(row, "phase2d2-round.schema.json")
    related_rounds = [
        row
        for row in rounds
        if row.get("task") == sample.get("task")
        and row.get("sample_id") == sample.get("sample_id")
        and int(row.get("nominal_prompt_length", 0))
        == int(sample.get("nominal_prompt_length", 0))
        and row.get("configuration_id") == sample.get("configuration_id")
    ]
    if not related_rounds or len(related_rounds) != len(rounds):
        raise ValueError(f"{root}: round/sample identity is incomplete or mixed")
    family = _family(manifest)
    _validate_preregistered_run(root, manifest, sample, family)
    cell_correctness = _correctness(root, manifest, correctness, sample)
    candidate = _candidate_metrics(related_rounds, root)
    online = _online_metrics(sample, related_rounds)
    formal = _formal_timing(sample)
    formal["generation_cuda_ms_per_committed_token"] = _safe_ratio(
        formal["generation_cuda_median_ms"], online["committed_token_count"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "input_directory": str(root),
        "task": sample["task"],
        "sample_id": sample["sample_id"],
        "source_sample_id": sample["source_sample_id"],
        "source_text_sha256": sample["source_text_sha256"],
        "context_prefix_sha256": sample["context_prefix_sha256"],
        "nominal_prompt_length": int(sample["nominal_prompt_length"]),
        "actual_prompt_length": int(sample["actual_prompt_length"]),
        "node_budget": int(manifest["phase2d2"]["node_budget"]),
        "configuration_family": family,
        "configuration_id": sample["configuration_id"],
        "provenance": _provenance(manifest),
        "correctness": cell_correctness,
        "candidate_quality": candidate,
        "online_efficiency": online,
        "formal_timing": formal,
        "component_cost": _component_cost(sample, related_rounds),
    }


def _validate_matrix(cells: list[dict[str, Any]]) -> None:
    expected = {
        (family, context, budget)
        for family in EXPECTED_FAMILIES
        for context in EXPECTED_CONTEXT_LENGTHS
        for budget in EXPECTED_NODE_BUDGETS
    }
    observed = [
        (
            row["configuration_family"],
            row["nominal_prompt_length"],
            row["node_budget"],
        )
        for row in cells
    ]
    duplicates = [key for key, count in Counter(observed).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate H1 matrix cells: {duplicates}")
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise ValueError(f"H1 matrix inventory mismatch; missing={missing}, extra={extra}")
    if len(cells) != 16:
        raise ValueError(f"H1 requires exactly 16 cells, got {len(cells)}")


def _validate_cross_run_provenance(cells: list[dict[str, Any]]) -> dict[str, Any]:
    first = cells[0]["provenance"]
    for field in COMMON_PROVENANCE_FIELDS:
        values = {json.dumps(row["provenance"].get(field), sort_keys=True) for row in cells}
        if len(values) != 1:
            raise ValueError(f"mixed H1 provenance: {field}")
        if first.get(field) is None:
            raise ValueError(f"missing H1 provenance: {field}")
    expected = {
        "online_selector_mode": EXPECTED_SELECTOR_MODE,
        "verification_mode": EXPECTED_VERIFICATION_MODE,
        "dtype": EXPECTED_DTYPE,
        "device": "cuda:0",
        "block_size": EXPECTED_BLOCK_SIZE,
        "max_new_tokens": EXPECTED_MAX_NEW_TOKENS,
        "timing_repeats": EXPECTED_TIMING_REPEATS,
        "warmup_runs": EXPECTED_WARMUP_RUNS,
        "batch_size": EXPECTED_BATCH_SIZE,
        "temperature": 0.0,
    }
    for field, value in expected.items():
        if first[field] != value:
            raise ValueError(f"H1 provenance {field} is not preregistered")
    refined_hashes = {
        row["provenance"]["frozen_refinement_hash"]
        for row in cells
        if row["configuration_family"] != BASE_FAMILY
    }
    if None in refined_hashes or len(refined_hashes) != 1:
        raise ValueError("refined H1 cells do not share one frozen refinement hash")
    base_hashes = {
        row["provenance"]["frozen_refinement_hash"]
        for row in cells
        if row["configuration_family"] == BASE_FAMILY
    }
    if base_hashes != {None}:
        raise ValueError("base H1 controls unexpectedly carry a refinement hash")
    for context in EXPECTED_CONTEXT_LENGTHS:
        cohort = [row for row in cells if row["nominal_prompt_length"] == context]
        identity_fields = (
            "task",
            "sample_id",
            "source_sample_id",
            "source_text_sha256",
            "context_prefix_sha256",
            "actual_prompt_length",
        )
        for field in identity_fields:
            if len({row[field] for row in cohort}) != 1:
                raise ValueError(
                    f"context {context} does not share one physical sample field: {field}"
                )
    return {
        **{field: first[field] for field in COMMON_PROVENANCE_FIELDS},
        "frozen_refinement_hash": next(iter(refined_hashes)),
        "base_control_refinement_hash": None,
        "frozen_refinement_configuration_set": list(
            FROZEN_REFINEMENT_FAMILIES
        ),
        "frozen_refinement_configuration_set_reselected": False,
    }


def _pair_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (
            row["configuration_family"],
            row["task"],
            row["sample_id"],
            row["nominal_prompt_length"],
            row["node_budget"],
        ): row
        for row in cells
    }
    pairs = []
    for family in FROZEN_REFINEMENT_FAMILIES:
        for context in EXPECTED_CONTEXT_LENGTHS:
            for budget in EXPECTED_NODE_BUDGETS:
                anchor = (KNOWN_TASK, KNOWN_SAMPLE_ID, context, budget)
                base = by_key.get((BASE_FAMILY, *anchor))
                refined = by_key.get((family, *anchor))
                if base is None or refined is None:
                    raise ValueError(f"missing exact same-N/context pair: {family}, {anchor}")
                physical_fields = (
                    "source_sample_id",
                    "source_text_sha256",
                    "context_prefix_sha256",
                    "actual_prompt_length",
                )
                if any(base[field] != refined[field] for field in physical_fields):
                    raise ValueError(f"pair physical sample mismatch: {family}, {anchor}")
                pair_valid = bool(
                    base["correctness"]["correctness_valid"]
                    and refined["correctness"]["correctness_valid"]
                )
                candidate_sufficient = bool(
                    base["candidate_quality"]["labelled_reference_position_count"]
                    and refined["candidate_quality"]["labelled_reference_position_count"]
                )
                base_online = base["online_efficiency"]
                refined_online = refined["online_efficiency"]
                metric_fields = (
                    "accepted_draft_tokens_per_verified_node",
                    "accepted_draft_tokens_per_round",
                    "committed_tokens_per_round",
                    "verified_target_nodes_per_returned_token",
                    "rounds_per_returned_token",
                )
                online_comparison = {}
                for field in metric_fields:
                    base_value = base_online[field] if pair_valid else None
                    refined_value = refined_online[field] if pair_valid else None
                    online_comparison[field] = {
                        "base": base_value,
                        "refined": refined_value,
                        "absolute_gain": _absolute_delta(base_value, refined_value),
                        "relative_gain": _relative_delta(base_value, refined_value),
                    }
                base_coverage = (
                    base["candidate_quality"][
                        "selected_tree_coverage_rate_at_fixed_n"
                    ]
                    if pair_valid and candidate_sufficient
                    else None
                )
                refined_coverage = (
                    refined["candidate_quality"][
                        "selected_tree_coverage_rate_at_fixed_n"
                    ]
                    if pair_valid and candidate_sufficient
                    else None
                )
                coverage_gain = _absolute_delta(base_coverage, refined_coverage)
                base_ms_token = (
                    base["formal_timing"]["generation_cuda_ms_per_returned_token"]
                    if pair_valid
                    else None
                )
                refined_ms_token = (
                    refined["formal_timing"][
                        "generation_cuda_ms_per_returned_token"
                    ]
                    if pair_valid
                    else None
                )
                base_latency = (
                    base["formal_timing"]["generation_cuda_median_ms"]
                    if pair_valid
                    else None
                )
                refined_latency = (
                    refined["formal_timing"]["generation_cuda_median_ms"]
                    if pair_valid
                    else None
                )
                base_throughput = (
                    base["formal_timing"]["returned_tokens_per_second"]
                    if pair_valid
                    else None
                )
                refined_throughput = (
                    refined["formal_timing"]["returned_tokens_per_second"]
                    if pair_valid
                    else None
                )
                accepted_gain = online_comparison[
                    "accepted_draft_tokens_per_verified_node"
                ]["absolute_gain"]
                latency_per_token_delta = _absolute_delta(
                    base_ms_token, refined_ms_token
                )
                pairs.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "task": KNOWN_TASK,
                        "sample_id": KNOWN_SAMPLE_ID,
                        "nominal_prompt_length": context,
                        "node_budget": budget,
                        "pairing_key": {
                            "task": KNOWN_TASK,
                            "sample_id": KNOWN_SAMPLE_ID,
                            "context_length": context,
                            "node_budget": budget,
                        },
                        "base_family": BASE_FAMILY,
                        "refined_family": family,
                        "correctness_valid": pair_valid,
                        "invalid_reason": (
                            None if pair_valid else "base_or_refined_hard_correctness_failed"
                        ),
                        "candidate_benefit_evidence_sufficient": candidate_sufficient,
                        "candidate_quality": {
                            "base_available_round_count": base["candidate_quality"][
                                "candidate_quality_available_round_count"
                            ],
                            "refined_available_round_count": refined[
                                "candidate_quality"
                            ]["candidate_quality_available_round_count"],
                            "base_labelled_reference_position_count": base[
                                "candidate_quality"
                            ]["labelled_reference_position_count"],
                            "refined_labelled_reference_position_count": refined[
                                "candidate_quality"
                            ]["labelled_reference_position_count"],
                            "base_selected_tree_coverage": base_coverage,
                            "refined_selected_tree_coverage": refined_coverage,
                            "coverage_absolute_gain": coverage_gain,
                            "coverage_relative_gain": _relative_delta(
                                base_coverage, refined_coverage
                            ),
                            "refined_only_target_recovery_rate": (
                                refined["candidate_quality"][
                                    "refined_only_target_recovery_rate"
                                ]
                                if pair_valid and candidate_sufficient
                                else None
                            ),
                        },
                        "online_efficiency": online_comparison,
                        "formal_timing": {
                            "base_generation_cuda_median_ms": base_latency,
                            "refined_generation_cuda_median_ms": refined_latency,
                            "generation_cuda_absolute_delta_ms": _absolute_delta(
                                base_latency, refined_latency
                            ),
                            "generation_cuda_relative_delta": _relative_delta(
                                base_latency, refined_latency
                            ),
                            "base_cuda_ms_per_returned_token": base_ms_token,
                            "refined_cuda_ms_per_returned_token": refined_ms_token,
                            "cuda_ms_per_returned_token_absolute_delta": (
                                latency_per_token_delta
                            ),
                            "cuda_ms_per_returned_token_relative_delta": (
                                _relative_delta(base_ms_token, refined_ms_token)
                            ),
                            "base_returned_tokens_per_second": base_throughput,
                            "refined_returned_tokens_per_second": refined_throughput,
                            "throughput_absolute_delta": _absolute_delta(
                                base_throughput, refined_throughput
                            ),
                            "throughput_relative_delta": _relative_delta(
                                base_throughput, refined_throughput
                            ),
                        },
                        "component_cost": {
                            "conditional_remask_cost": refined["component_cost"],
                            "base_fixed_shape_recheck_profile_total_cuda_ms": base[
                                "component_cost"
                            ]["fixed_shape_recheck_profile_total_cuda_ms"],
                            "refined_fixed_shape_recheck_profile_total_cuda_ms": refined[
                                "component_cost"
                            ]["fixed_shape_recheck_profile_total_cuda_ms"],
                        },
                        "sign_matrix": {
                            "coverage_gain": _sign(coverage_gain),
                            "accepted_per_verified_gain": _sign(accepted_gain),
                            "latency": _latency_sign(latency_per_token_delta),
                        },
                    }
                )
    return pairs


def _config_summaries(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for family in FROZEN_REFINEMENT_FAMILIES:
        rows = [row for row in pairs if row["refined_family"] == family]
        if len(rows) != 4:
            raise ValueError(f"{family}: expected four H1 pairs")
        invalid = [row for row in rows if not row["correctness_valid"]]
        insufficient = [
            row for row in rows if not row["candidate_benefit_evidence_sufficient"]
        ]
        joint_positive = [
            row
            for row in rows
            if row["correctness_valid"]
            and row["candidate_quality"]["coverage_absolute_gain"] is not None
            and row["candidate_quality"]["coverage_absolute_gain"] > 0
            and row["online_efficiency"][
                "accepted_draft_tokens_per_verified_node"
            ]["absolute_gain"]
            is not None
            and row["online_efficiency"][
                "accepted_draft_tokens_per_verified_node"
            ]["absolute_gain"]
            > 0
        ]
        if invalid:
            classification = "invalid_due_to_correctness"
        elif insufficient:
            classification = "insufficient_candidate_label_evidence"
        elif len(joint_positive) == 4:
            classification = "consistent_positive_known_sample"
        elif joint_positive:
            classification = "mixed_known_sample"
        else:
            classification = "no_positive_known_sample_signal"
        summaries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "refined_family": family,
                "pair_count": len(rows),
                "valid_pair_count": len(rows) - len(invalid),
                "invalid_pair_count": len(invalid),
                "candidate_evidence_insufficient_pair_count": len(insufficient),
                "joint_positive_pair_count": len(joint_positive),
                "known_sample_stability_classification": classification,
                "classification_is_descriptive_only": True,
                "winner_selected": False,
                "matrix": [
                    {
                        "nominal_prompt_length": row["nominal_prompt_length"],
                        "node_budget": row["node_budget"],
                        "correctness_valid": row["correctness_valid"],
                        "candidate_label_coverage": {
                            "base_available_round_count": row["candidate_quality"][
                                "base_available_round_count"
                            ],
                            "refined_available_round_count": row[
                                "candidate_quality"
                            ]["refined_available_round_count"],
                            "base_labelled_position_count": row["candidate_quality"][
                                "base_labelled_reference_position_count"
                            ],
                            "refined_labelled_position_count": row[
                                "candidate_quality"
                            ]["refined_labelled_reference_position_count"],
                        },
                        "base_selected_tree_coverage": row["candidate_quality"][
                            "base_selected_tree_coverage"
                        ],
                        "refined_selected_tree_coverage": row[
                            "candidate_quality"
                        ]["refined_selected_tree_coverage"],
                        "coverage_absolute_gain": row["candidate_quality"][
                            "coverage_absolute_gain"
                        ],
                        "refined_only_recovery_rate": row["candidate_quality"][
                            "refined_only_target_recovery_rate"
                        ],
                        "base_accepted_per_verified": row["online_efficiency"][
                            "accepted_draft_tokens_per_verified_node"
                        ]["base"],
                        "refined_accepted_per_verified": row["online_efficiency"][
                            "accepted_draft_tokens_per_verified_node"
                        ]["refined"],
                        "accepted_per_verified_absolute_gain": row[
                            "online_efficiency"
                        ]["accepted_draft_tokens_per_verified_node"]["absolute_gain"],
                        "accepted_per_verified_relative_gain": row[
                            "online_efficiency"
                        ]["accepted_draft_tokens_per_verified_node"]["relative_gain"],
                        "base_accepted_per_round": row["online_efficiency"][
                            "accepted_draft_tokens_per_round"
                        ]["base"],
                        "refined_accepted_per_round": row["online_efficiency"][
                            "accepted_draft_tokens_per_round"
                        ]["refined"],
                        "base_cuda_ms_per_returned_token": row["formal_timing"][
                            "base_cuda_ms_per_returned_token"
                        ],
                        "refined_cuda_ms_per_returned_token": row["formal_timing"][
                            "refined_cuda_ms_per_returned_token"
                        ],
                        "latency_relative_change": row["formal_timing"][
                            "cuda_ms_per_returned_token_relative_delta"
                        ],
                        "conditional_remask_cost": row["component_cost"][
                            "conditional_remask_cost"
                        ],
                        "base_fixed_shape_verifier_cost_ms": row["component_cost"][
                            "base_fixed_shape_recheck_profile_total_cuda_ms"
                        ],
                        "refined_fixed_shape_verifier_cost_ms": row[
                            "component_cost"
                        ]["refined_fixed_shape_recheck_profile_total_cuda_ms"],
                        "sign_matrix": row["sign_matrix"],
                    }
                    for row in rows
                ],
            }
        )
    return summaries


def _cost_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        conditional = pair["component_cost"]["conditional_remask_cost"]
        eligible = bool(pair["correctness_valid"])
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "task": pair["task"],
                "sample_id": pair["sample_id"],
                "nominal_prompt_length": pair["nominal_prompt_length"],
                "node_budget": pair["node_budget"],
                "refined_family": pair["refined_family"],
                "correctness_valid": pair["correctness_valid"],
                "eligible_for_cost_conclusions": eligible,
                "dependency_modeling_cost": {
                    "conditional_remask_cuda_median_ms_per_round": _eligible_value(
                        conditional,
                        "conditional_remask_cuda_median_ms_per_round",
                        eligible=eligible,
                    ),
                    "conditional_remask_profile_total_cuda_ms": _eligible_value(
                        conditional,
                        "conditional_remask_profile_total_cuda_ms",
                        eligible=eligible,
                    ),
                    "conditional_remask_share_of_profiled_round_time": _eligible_value(
                        conditional,
                        "conditional_remask_share_of_profiled_round_time",
                        eligible=eligible,
                    ),
                },
                "correctness_preserving_verifier_cost": {
                    "fixed_shape_recheck_cuda_median_ms_per_round": _eligible_value(
                        conditional,
                        "fixed_shape_recheck_cuda_median_ms_per_round",
                        eligible=eligible,
                    ),
                    "fixed_shape_recheck_profile_total_cuda_ms": _eligible_value(
                        conditional,
                        "fixed_shape_recheck_profile_total_cuda_ms",
                        eligible=eligible,
                    ),
                    "fixed_shape_recheck_share_of_profiled_round_time": _eligible_value(
                        conditional,
                        "fixed_shape_recheck_share_of_profiled_round_time",
                        eligible=eligible,
                    ),
                    "round_entry_snapshot_profile_total_cuda_ms": _eligible_value(
                        conditional,
                        "fixed_shape_round_entry_snapshot_profile_total_cuda_ms",
                        eligible=eligible,
                    ),
                    "kv_replacement_profile_total_cuda_ms": _eligible_value(
                        conditional,
                        "fixed_shape_kv_replacement_profile_total_cuda_ms",
                        eligible=eligible,
                    ),
                    "extended_fixed_shape_path_profile_total_cuda_ms": _eligible_value(
                        conditional,
                        "extended_fixed_shape_path_profile_total_cuda_ms",
                        eligible=eligible,
                    ),
                    "extended_fixed_shape_path_share_of_profiled_round_time": _eligible_value(
                        conditional,
                        "extended_fixed_shape_path_share_of_profiled_round_time",
                        eligible=eligible,
                    ),
                    "fallback_timer_excludes_snapshot_and_kv_replacement": True,
                },
                "combined_additional_work": {
                    "profile_total_cuda_ms": _eligible_value(
                        conditional,
                        "combined_conditional_and_fixed_profile_total_cuda_ms",
                        eligible=eligible,
                    ),
                    "share_of_profiled_round_time": _eligible_value(
                        conditional,
                        "combined_additional_work_share_of_profiled_round_time",
                        eligible=eligible,
                    ),
                },
                "conditional_over_initial_draft_ratio": None,
                "conditional_over_initial_draft_ratio_unavailable_reason": conditional[
                    "conditional_over_initial_draft_ratio_unavailable_reason"
                ],
                "nested_timer_double_counting_avoided": True,
                "always_fixed_verifier_cost_attributed_to_conditional_remask": False,
            }
        )
    return rows


def _flat(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


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


def _aggregate_cells(
    cells: list[dict[str, Any]], input_directories: list[Path]
) -> dict[str, Any]:
    _validate_matrix(cells)
    provenance = _validate_cross_run_provenance(cells)
    pairs = _pair_cells(cells)
    config_summaries = _config_summaries(pairs)
    costs = _cost_rows(pairs)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": "complete",
        "input_directory_count": len(input_directories),
        "input_directories": [str(path) for path in input_directories],
        "matrix_expected_cell_count": 16,
        "matrix_observed_cell_count": len(cells),
        "matrix_complete": True,
        "known_task": KNOWN_TASK,
        "known_sample_id": KNOWN_SAMPLE_ID,
        "context_lengths": list(EXPECTED_CONTEXT_LENGTHS),
        "node_budgets": list(EXPECTED_NODE_BUDGETS),
        "configuration_families": list(EXPECTED_FAMILIES),
        "refinement_families": list(FROZEN_REFINEMENT_FAMILIES),
        "provenance": provenance,
        "candidate_quality_target_label_authority": CANDIDATE_LABEL_AUTHORITY,
        "candidate_quality_target_label_source": CANDIDATE_LABEL_SOURCE,
        "q1_replay_used_as_target_label_source": False,
        "target_labels_used_for_runtime_selection": False,
        "invalid_cell_count": sum(
            not row["correctness"]["correctness_valid"] for row in cells
        ),
        "cells": cells,
        "artifacts": list(ARTIFACTS),
    }
    paired = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": "complete",
        "pairing_key_fields": [
            "task",
            "sample_id",
            "context_length",
            "node_budget",
        ],
        "same_n_and_same_context_pairing_required": True,
        "correctness_failed_cells_excluded_from_gains": True,
        "unavailable_candidate_rows_excluded_from_denominators": True,
        "primary_online_metric": "accepted_draft_tokens_per_verified_node",
        "rows": pairs,
    }
    config_summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": "complete",
        "classification_scope": "descriptive_known_sample_only",
        "winner_selected": False,
        "frozen_refinement_configuration_set_reselected": False,
        "rows": config_summaries,
    }
    cost_summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": "complete",
        "timer_semantics_audited": True,
        "formal_latency_source": "unprofiled_control_repeats",
        "component_timing_source": "separate_profile_and_diagnostics_paths",
        "nested_timer_double_counting_avoided": True,
        "dependency_modeling_and_verifier_costs_separated": True,
        "conditional_full_pass_exact_ratio_available": False,
        "conditional_full_pass_exact_ratio_unavailable_reason": (
            "conditional and initial draft timer scopes are non-overlapping but not "
            "directly comparable"
        ),
        "rows": costs,
    }
    decision = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": "complete",
        "final_phase2d2a_decision": "insufficient_evidence",
        "reason": "H1_contains_one_task_and_no_independent_held_out_task_evidence",
        "independent_task_count": 1,
        "h1_is_known_sample_mechanism_and_cost_study": True,
        "h2_held_out_cross_task_evidence_required": True,
        "winner_selected": False,
        "frozen_refinement_configuration_set_reselected": False,
        "legacy_guard_false_negative_count_used_as_h1_gate": False,
        "always_fixed_correctness_criteria": [
            "original_dflash_reference_tree_token_exactness",
            "formal_repeat_stability",
            "profile_formal_equality",
            "diagnostic_formal_equality",
            "structural_and_kv_correctness",
        ],
        "known_sample_classifications": {
            row["refined_family"]: row["known_sample_stability_classification"]
            for row in config_summaries
        },
        "prohibited_final_decisions_not_issued": [
            "proceed_to_lightweight_causal_adapter",
            "proceed_with_conditional_remask",
            "no_dependency_refinement_benefit",
        ],
    }
    return {
        "manifest": manifest,
        "paired": paired,
        "config_summary": config_summary,
        "cost_summary": cost_summary,
        "decision": decision,
    }


def aggregate_h1(result_dirs: list[Path]) -> dict[str, Any]:
    if not result_dirs:
        raise ValueError("at least one Phase 2D.2A H1 result directory is required")
    resolved = [path.expanduser().resolve() for path in result_dirs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate H1 input directories")
    cells = [_load_cell(root) for root in resolved]
    return _aggregate_cells(cells, resolved)


def persist_h1(output_dir: Path, aggregate: dict[str, Any]) -> None:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty H1 output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "phase2d2a-h-known-manifest.json": aggregate["manifest"],
        "phase2d2a-h-known-paired-comparison.json": aggregate["paired"],
        "phase2d2a-h-known-config-summary.json": aggregate["config_summary"],
        "phase2d2a-h-known-cost-summary.json": aggregate["cost_summary"],
        "phase2d2a-h-known-decision.json": aggregate["decision"],
    }
    for name, payload in payloads.items():
        _validate_schema(payload, SCHEMAS[name])
        _write_json(output_dir / name, payload)
    _write_csv(
        output_dir / "phase2d2a-h-known-paired-comparison.csv",
        aggregate["paired"]["rows"],
    )
    _write_csv(
        output_dir / "phase2d2a-h-known-config-summary.csv",
        aggregate["config_summary"]["rows"],
    )
    _write_csv(
        output_dir / "phase2d2a-h-known-cost-summary.csv",
        aggregate["cost_summary"]["rows"],
    )


def _discover_result_directories(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.expanduser().resolve().rglob("manifest.json"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", action="append", type=Path, default=[])
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = list(args.result_dir)
    if args.result_root is not None:
        roots.extend(_discover_result_directories(args.result_root))
    aggregate = aggregate_h1(roots)
    persist_h1(args.output_dir, aggregate)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "cell_count": len(aggregate["manifest"]["cells"]),
                "pair_count": len(aggregate["paired"]["rows"]),
                "final_phase2d2a_decision": aggregate["decision"][
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
