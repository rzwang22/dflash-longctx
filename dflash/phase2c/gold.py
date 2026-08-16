"""Framework-lazy Phase 2C.1 gold diagnostics and classification helpers."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

GOLD_COMPARISON_MODES = ("native", "equal_shape", "same_mask", "math")
GOLD_STATUSES = (
    "pass_exact",
    "pass_semantic_with_numerical_variation",
    "fail_semantic",
    "fail_structural",
    "not_run",
)
DEFAULT_GOLD_THRESHOLDS = {
    "maximum_absolute_error": 0.5,
    "mean_absolute_error": 0.02,
    "relative_error_epsilon": 1.0e-6,
    "layer_localization_p99_error": 0.1,
}


def validate_gold_thresholds(thresholds: dict[str, float]) -> dict[str, float]:
    """Return a named, finite threshold set without silently changing values."""

    required = set(DEFAULT_GOLD_THRESHOLDS)
    if set(thresholds) != required:
        raise ValueError(f"gold thresholds must contain exactly {sorted(required)}")
    result = {name: float(value) for name, value in thresholds.items()}
    if any(not math.isfinite(value) or value <= 0 for value in result.values()):
        raise ValueError("all Phase 2C.1 gold thresholds must be finite and positive")
    return result


def validate_gold_modes(modes: Iterable[str]) -> tuple[str, ...]:
    values = tuple(modes)
    if not values:
        raise ValueError("at least one gold comparison mode is required")
    if len(values) != len(set(values)):
        raise ValueError("gold comparison modes must be unique")
    unknown = sorted(set(values) - set(GOLD_COMPARISON_MODES))
    if unknown:
        raise ValueError(f"unsupported gold comparison modes: {unknown}")
    return values


def expected_node_alignment(
    *,
    node: dict[str, Any],
    persistent_prefix_length: int,
    packed_query_length: int,
) -> dict[str, Any]:
    """Describe the semantic input and prediction represented by one node logit."""

    node_index = int(node["node_index"])
    depth = int(node["depth"])
    path = [int(value) for value in node["root_to_node_token_path"]]
    if len(path) != depth + 1:
        raise ValueError("root-to-node path length must equal depth + one")
    semantic_positions = list(
        range(persistent_prefix_length, persistent_prefix_length + depth + 1)
    )
    return {
        "flattened_node_index": node_index,
        "parent_index": node.get("parent_index"),
        "node_depth": depth,
        "node_token_id": int(node["token_id"]),
        "root_to_node_token_ids": path,
        "persistent_prefix_length": int(persistent_prefix_length),
        "packed_query_length": int(packed_query_length),
        "packed_kv_length": int(persistent_prefix_length + packed_query_length),
        "dense_reference_query_length": len(path),
        "dense_reference_kv_length": int(persistent_prefix_length + len(path)),
        "semantic_position_ids": semantic_positions,
        "packed_position_id": int(persistent_prefix_length + depth),
        "packed_cache_position": int(persistent_prefix_length + node_index),
        "dense_reference_position_ids": semantic_positions,
        "dense_reference_cache_positions": semantic_positions,
        "dense_reference_root_insertions": 1,
        "packed_logits_query_index": node_index,
        "dense_logits_query_index": len(path) - 1,
        "input_token_semantic_position": int(persistent_prefix_length + depth),
        "predicted_token_semantic_position": int(persistent_prefix_length + depth + 1),
        "packed_logits_predicts_semantic_position": int(
            persistent_prefix_length + depth + 1
        ),
        "dense_logits_predicts_semantic_position": int(
            persistent_prefix_length + depth + 1
        ),
        "logits_semantics": "node input predicts the token after that node",
    }


def structural_alignment_issues(record: dict[str, Any]) -> list[str]:
    """Detect off-by-one and semantic/storage-position confusion in a node record."""

    issues = []
    depth = int(record["node_depth"])
    prefix = int(record["persistent_prefix_length"])
    node_index = int(record["flattened_node_index"])
    path = record["root_to_node_token_ids"]
    if len(path) != depth + 1:
        issues.append("root_to_node_path_length")
    if record["packed_logits_query_index"] != node_index:
        issues.append("packed_logits_query_index")
    if record["dense_logits_query_index"] != depth:
        issues.append("dense_logits_query_index")
    if record["packed_position_id"] != prefix + depth:
        issues.append("packed_semantic_position")
    if record["packed_cache_position"] != prefix + node_index:
        issues.append("packed_storage_position")
    if record["dense_reference_position_ids"] != list(
        range(prefix, prefix + depth + 1)
    ):
        issues.append("dense_semantic_positions")
    if record["dense_reference_cache_positions"] != list(
        range(prefix, prefix + depth + 1)
    ):
        issues.append("dense_cache_positions")
    if record["dense_reference_root_insertions"] != 1:
        issues.append("dense_root_insertion_count")
    if record["input_token_semantic_position"] != prefix + depth:
        issues.append("input_token_semantic_position")
    if record["predicted_token_semantic_position"] != prefix + depth + 1:
        issues.append("prediction_off_by_one")
    if record["packed_logits_predicts_semantic_position"] != prefix + depth + 1:
        issues.append("packed_prediction_position")
    if record["dense_logits_predicts_semantic_position"] != prefix + depth + 1:
        issues.append("dense_prediction_position")
    if record["packed_kv_length"] != prefix + record["packed_query_length"]:
        issues.append("packed_kv_length")
    if record["dense_reference_kv_length"] != (
        prefix + record["dense_reference_query_length"]
    ):
        issues.append("dense_kv_length")
    return issues


def validate_visibility_matrix(
    actual: list[list[bool]], expected: list[list[bool]]
) -> list[str]:
    """Identify tree-mask leakage or missing legal visibility."""

    if len(actual) != len(expected) or any(
        len(left) != len(right) for left, right in zip(actual, expected)
    ):
        return ["tree_mask_shape"]
    issues = []
    for query, (left, right) in enumerate(zip(actual, expected)):
        for key, (visible, should_be_visible) in enumerate(zip(left, right)):
            if visible and not should_be_visible:
                issues.append(f"tree_mask_leakage:q{query}:k{key}")
            elif should_be_visible and not visible:
                issues.append(f"tree_mask_missing_visibility:q{query}:k{key}")
    return issues


def _available(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in records if row.get("availability") == "available"]


def summarize_comparison_mode(
    *,
    mode: str,
    backend: str,
    dtype: str,
    nodes: list[dict[str, Any]],
    structural_issues: list[str],
    thresholds: dict[str, float],
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    """Summarize one mode without conflating ordered Top-K and greedy semantics."""

    thresholds = validate_gold_thresholds(thresholds)
    available = _available(nodes)
    if unavailable_reason is not None:
        return {
            "comparison_mode": mode,
            "availability": "unavailable",
            "status": "not_run",
            "backend": backend,
            "dtype": dtype,
            "structural_status": "not_run",
            "semantic_status": "not_run",
            "numerical_status": "not_run",
            "thresholds": thresholds,
            "warning": None,
            "reason": unavailable_reason,
            "compared_node_count": 0,
            "argmax_mismatch_count": 0,
            "verifier_decision_mismatch_count": 0,
            "top8_ordered_mismatch_count": 0,
            "maximum_absolute_error": None,
            "mean_absolute_error": None,
            "p99_absolute_error": None,
            "first_failing_depth": None,
            "first_failing_node": None,
            "first_numerical_threshold_exceeding_depth": None,
            "first_numerical_threshold_exceeding_node": None,
            "first_top8_mismatch_depth": None,
            "first_top8_mismatch_node": None,
            "first_semantic_mismatch_depth": None,
            "first_semantic_mismatch_node": None,
            "first_structural_mismatch_depth": None,
            "first_structural_mismatch_node": None,
            "structural_issues": structural_issues,
        }
    if not available:
        raise ValueError("available comparison mode requires node records")
    structural_failures = list(structural_issues)
    for row in available:
        structural_failures.extend(row.get("structural_issues", []))
    argmax_mismatches = [row for row in available if not row["argmax_equal"]]
    verifier_mismatches = [
        row for row in argmax_mismatches if row.get("verifier_decision_relevant", False)
    ]
    top8_mismatches = [row for row in available if not row["top8_ordered_equal"]]
    numerical_failures = [
        row
        for row in available
        if row["max_abs_error"] > thresholds["maximum_absolute_error"]
        or row["mean_abs_error"] > thresholds["mean_absolute_error"]
    ]

    def first(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        return min(
            rows,
            key=lambda row: (row["node_depth"], row["flattened_node_index"]),
            default=None,
        )

    structural_nodes = [row for row in available if row.get("structural_issues")]
    first_structural = first(structural_nodes)
    first_semantic = first(verifier_mismatches)
    first_numerical = first(numerical_failures)
    first_top8 = first(top8_mismatches)
    first_failure = first(structural_nodes + verifier_mismatches)
    if structural_failures:
        status = "fail_structural"
    elif verifier_mismatches:
        status = "fail_semantic"
    elif numerical_failures or top8_mismatches or argmax_mismatches:
        status = "pass_semantic_with_numerical_variation"
    else:
        status = "pass_exact"
    total_values = sum(int(row["logit_element_count"]) for row in available)
    weighted_mean = (
        sum(row["mean_abs_error"] * row["logit_element_count"] for row in available)
        / total_values
    )
    return {
        "comparison_mode": mode,
        "availability": "available",
        "status": status,
        "backend": backend,
        "dtype": dtype,
        "structural_status": "fail" if structural_failures else "pass",
        "semantic_status": "fail" if verifier_mismatches else "pass",
        "numerical_status": "outside_strict_thresholds"
        if numerical_failures
        else "within_strict_thresholds",
        "thresholds": thresholds,
        "warning": (
            "ordered Top-8 differs without a verifier-decision mismatch"
            if top8_mismatches and not verifier_mismatches
            else None
        ),
        "reason": None,
        "compared_node_count": len(available),
        "argmax_mismatch_count": len(argmax_mismatches),
        "verifier_decision_mismatch_count": len(verifier_mismatches),
        "top8_ordered_mismatch_count": len(top8_mismatches),
        "maximum_absolute_error": max(row["max_abs_error"] for row in available),
        "mean_absolute_error": weighted_mean,
        "p99_absolute_error": max(row["p99_abs_error"] for row in available),
        "first_failing_depth": (first_failure["node_depth"] if first_failure else None),
        "first_failing_node": (
            first_failure["flattened_node_index"] if first_failure else None
        ),
        "first_numerical_threshold_exceeding_depth": (
            first_numerical["node_depth"] if first_numerical else None
        ),
        "first_numerical_threshold_exceeding_node": (
            first_numerical["flattened_node_index"] if first_numerical else None
        ),
        "first_top8_mismatch_depth": (first_top8["node_depth"] if first_top8 else None),
        "first_top8_mismatch_node": (
            first_top8["flattened_node_index"] if first_top8 else None
        ),
        "first_semantic_mismatch_depth": (
            first_semantic["node_depth"] if first_semantic else None
        ),
        "first_semantic_mismatch_node": (
            first_semantic["flattened_node_index"] if first_semantic else None
        ),
        "first_structural_mismatch_depth": (
            first_structural["node_depth"] if first_structural else None
        ),
        "first_structural_mismatch_node": (
            first_structural["flattened_node_index"] if first_structural else None
        ),
        "structural_issues": sorted(set(structural_failures)),
        "has_non_verifier_argmax_mismatch": bool(
            argmax_mismatches and not verifier_mismatches
        ),
        "strict_failure_present": bool(
            structural_failures
            or verifier_mismatches
            or numerical_failures
            or top8_mismatches
        ),
    }


def classify_comparison_matrix(
    mode_summaries: list[dict[str, Any]],
    *,
    output_ids_equal: bool,
    round_semantics_equal: bool,
) -> dict[str, Any]:
    """Classify exactness, backend variation, or hard structural/semantic failure."""

    by_mode = {row["comparison_mode"]: row for row in mode_summaries}
    native = by_mode.get("native")
    available = [row for row in mode_summaries if row["availability"] == "available"]
    structural_failures = [
        row for row in available if row["structural_status"] == "fail"
    ]
    semantic_failures = [
        row for row in available if row["verifier_decision_mismatch_count"] > 0
    ]
    if structural_failures:
        status = "fail_structural"
        failure_class = "structural_mismatch"
        evidence = [row["comparison_mode"] for row in structural_failures]
    elif semantic_failures or not output_ids_equal or not round_semantics_equal:
        status = "fail_semantic"
        failure_class = "semantic_mismatch"
        evidence = [row["comparison_mode"] for row in semantic_failures]
    elif native is None:
        exact_modes = [row for row in available if row["status"] == "pass_exact"]
        if available and len(exact_modes) == len(available):
            status = "pass_exact"
            failure_class = None
            evidence = [row["comparison_mode"] for row in exact_modes]
        else:
            status = "fail_semantic"
            failure_class = "unresolved_gold_mismatch"
            evidence = []
    elif native["availability"] != "available":
        status = "fail_semantic"
        failure_class = "native_comparison_unavailable"
        evidence = []
    elif native["status"] == "pass_exact":
        varied_controls = [
            row
            for row in available
            if row["comparison_mode"] != "native" and row["status"] != "pass_exact"
        ]
        status = (
            "pass_semantic_with_numerical_variation"
            if varied_controls
            else "pass_exact"
        )
        failure_class = None
        evidence = ["native"]
    else:
        controls = [
            row
            for name, row in by_mode.items()
            if name in {"equal_shape", "same_mask", "math"}
            and row["availability"] == "available"
            and row["status"] == "pass_exact"
        ]
        if controls:
            status = "pass_semantic_with_numerical_variation"
            failure_class = "numerical_backend_variation"
            evidence = [row["comparison_mode"] for row in controls]
        else:
            status = "pass_semantic_with_numerical_variation"
            failure_class = "unresolved_numerical_variation"
            evidence = []
    return {
        "status": status,
        "failure_class": failure_class,
        "all_checks_passed": status
        in {"pass_exact", "pass_semantic_with_numerical_variation"},
        "output_ids_equal": bool(output_ids_equal),
        "round_semantics_equal": bool(round_semantics_equal),
        "classification_evidence_modes": evidence,
        "numerical_backend_variation_proven": failure_class
        == "numerical_backend_variation",
    }


def error_distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    clean = sorted(float(value) for value in values)
    if not clean:
        return {"count": 0, "mean": None, "median": None, "p99": None}
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "p99": clean[max(0, math.ceil(0.99 * len(clean)) - 1)],
    }


def build_gold_comparison_summary(
    logits_records: list[dict[str, Any]], kv_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate persisted records, preserving unavailable/not-run explanations."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in logits_records:
        key = (
            record.get("task"),
            record.get("sample_id"),
            record.get("nominal_prompt_length"),
            record.get("configuration_id"),
            record.get("comparison_mode"),
        )
        grouped[key].append(record)
    rows = []
    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        records = grouped[key]
        available = [row for row in records if row["availability"] == "available"]
        summaries = [
            row.get("mode_summary") for row in records if row.get("mode_summary")
        ]
        summary = summaries[-1] if summaries else None
        rows.append(
            {
                "task": key[0],
                "sample_id": key[1],
                "nominal_prompt_length": key[2],
                "configuration_id": key[3],
                "comparison_mode": key[4],
                "availability": records[-1]["availability"],
                "status": summary["status"] if summary else records[-1]["status"],
                "backend": records[-1]["backend"],
                "dtype": records[-1]["dtype"],
                "compared_node_count": len(available),
                "argmax_mismatch_count": sum(
                    not row.get("argmax_equal", True) for row in available
                ),
                "verifier_decision_mismatch_count": sum(
                    row.get("verifier_decision_relevant", False)
                    and not row.get("argmax_equal", True)
                    for row in available
                ),
                "top8_ordered_mismatch_count": sum(
                    not row.get("top8_ordered_equal", True) for row in available
                ),
                "maximum_absolute_error": max(
                    (row["max_abs_error"] for row in available), default=None
                ),
                "mean_absolute_error": (
                    statistics.fmean(row["mean_abs_error"] for row in available)
                    if available
                    else None
                ),
                "p99_absolute_error": max(
                    (row["p99_abs_error"] for row in available), default=None
                ),
                "first_failing_depth": (
                    summary.get("first_failing_depth") if summary else None
                ),
                "first_failing_node": (
                    summary.get("first_failing_node") if summary else None
                ),
                "first_numerical_threshold_exceeding_depth": (
                    summary.get("first_numerical_threshold_exceeding_depth")
                    if summary
                    else None
                ),
                "first_numerical_threshold_exceeding_node": (
                    summary.get("first_numerical_threshold_exceeding_node")
                    if summary
                    else None
                ),
                "first_top8_mismatch_depth": (
                    summary.get("first_top8_mismatch_depth") if summary else None
                ),
                "first_top8_mismatch_node": (
                    summary.get("first_top8_mismatch_node") if summary else None
                ),
                "first_semantic_mismatch_depth": (
                    summary.get("first_semantic_mismatch_depth") if summary else None
                ),
                "first_semantic_mismatch_node": (
                    summary.get("first_semantic_mismatch_node") if summary else None
                ),
                "first_structural_mismatch_depth": (
                    summary.get("first_structural_mismatch_depth") if summary else None
                ),
                "first_structural_mismatch_node": (
                    summary.get("first_structural_mismatch_node") if summary else None
                ),
                "structural_status": (
                    summary.get("structural_status") if summary else "not_run"
                ),
                "semantic_status": (
                    summary.get("semantic_status") if summary else "not_run"
                ),
                "numerical_status": (
                    summary.get("numerical_status") if summary else "not_run"
                ),
                "thresholds": summary.get("thresholds") if summary else None,
                "warning": summary.get("warning") if summary else None,
                "reason": records[-1].get("reason"),
            }
        )
    configuration_classifications = {}
    for record in logits_records:
        if "matrix_status" not in record:
            continue
        key = "|".join(
            str(record.get(field))
            for field in (
                "task",
                "sample_id",
                "nominal_prompt_length",
                "configuration_id",
                "round_index",
            )
        )
        configuration_classifications[key] = {
            "task": record.get("task"),
            "sample_id": record.get("sample_id"),
            "nominal_prompt_length": record.get("nominal_prompt_length"),
            "configuration_id": record.get("configuration_id"),
            "round_index": record.get("round_index"),
            "status": record["matrix_status"],
            "failure_class": record.get("matrix_failure_class"),
            "classification_evidence_modes": record.get(
                "classification_evidence_modes", []
            ),
        }
    classifications = list(configuration_classifications.values())
    summary_status = (
        "not_run"
        if not logits_records
        else "error"
        if (
            any(
                row["status"] in {"fail_semantic", "fail_structural"}
                for row in classifications
            )
            if classifications
            else any(
                row.get("status") in {"fail_semantic", "fail_structural"}
                for row in logits_records
            )
        )
        else "ok"
    )
    return {
        "schema_version": "1.0",
        "status": summary_status,
        "logit_record_count": len(logits_records),
        "kv_record_count": len(kv_records),
        "groups": rows,
        "configuration_classifications": classifications,
        "interpretation": {
            "top8_ordered_mismatch_is_not_automatically_semantic_failure": True,
            "verifier_relevant_argmax_mismatch_is_hard_failure": True,
            "numerical_backend_variation_requires_control_evidence": True,
        },
    }
