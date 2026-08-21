"""Numerical-guard calibration and replay-event semantics for Phase 2D.2A."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

CALIBRATION_SCHEMA_VERSION = "phase2d2a-nr-calibration-v1"
FROZEN_DECISION_MARGIN_THRESHOLD = 0.25
PROVENANCE_FIELDS = (
    "gpu_name",
    "compute_capability",
    "dtype",
    "target_model_revision",
    "draft_model_revision",
    "torch_version",
    "transformers_version",
    "target_attention_implementation",
)
SOURCE_ARTIFACT_FIELDS = (
    "nr_directory",
    "nr_manifest_sha256",
    "margin_trigger_sweep_sha256",
    "nr_acceptance_checker_result",
    "qualifying_rows",
    "cohort_overall_strata",
    "observed_actual_divergence_outcome_counts",
    "observed_actual_divergence_round_count",
    "observed_divergence_rounds_caught_by_trigger",
    "observed_false_negative_count",
    "minimum_preregistered_zero_false_negative_threshold",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def calibration_sha256(calibration_without_hash: dict[str, Any]) -> str:
    value = dict(calibration_without_hash)
    value.pop("calibration_sha256", None)
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def validate_nr_calibration(
    calibration: dict[str, Any], runtime_provenance: dict[str, Any]
) -> dict[str, Any]:
    """Require independent-cohort calibration and exact runtime provenance."""

    required = {
        "schema_version",
        "source_phase",
        "independent_calibration_cohort",
        "decision_margin_threshold",
        "source_artifacts",
        "provenance",
        "evaluation_outputs_used_to_choose_threshold",
        "formal_error_bound_claimed",
        "empirically_validated_guard_only",
        "calibration_sha256",
    }
    missing = sorted(required - set(calibration))
    if missing:
        raise ValueError(f"NR calibration lacks required fields: {missing}")
    if calibration["schema_version"] != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported NR calibration schema version")
    if calibration["source_phase"] != "2D.1B-NR":
        raise ValueError("NR calibration must originate from Phase 2D.1B-NR")
    if not calibration["independent_calibration_cohort"]:
        raise ValueError("evaluation outputs cannot calibrate their own guard")
    if calibration["evaluation_outputs_used_to_choose_threshold"]:
        raise ValueError("evaluation outputs cannot choose the guard threshold")
    if calibration["formal_error_bound_claimed"]:
        raise ValueError("Phase 2D.2A guard has no formal numerical error bound")
    if not calibration["empirically_validated_guard_only"]:
        raise ValueError("NR calibration must identify itself as empirical only")
    threshold = float(calibration["decision_margin_threshold"])
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("calibrated decision margin must be finite and non-negative")
    if threshold != FROZEN_DECISION_MARGIN_THRESHOLD:
        raise ValueError("Phase 2D.2A requires the frozen NR threshold 0.25")
    source_artifacts = calibration["source_artifacts"]
    missing_source = sorted(set(SOURCE_ARTIFACT_FIELDS) - set(source_artifacts))
    if missing_source:
        raise ValueError(
            f"NR calibration source artifacts lack required fields: {missing_source}"
        )
    checker = source_artifacts["nr_acceptance_checker_result"]
    if checker.get("status") != "pass":
        raise ValueError("NR calibration source acceptance checker did not pass")
    if int(source_artifacts["observed_actual_divergence_round_count"]) != 4:
        raise ValueError("NR calibration must contain four observed divergences")
    if int(source_artifacts["observed_divergence_rounds_caught_by_trigger"]) != 4:
        raise ValueError("NR guard did not catch all four observed divergences")
    if int(source_artifacts["observed_false_negative_count"]) != 0:
        raise ValueError("NR calibration contains an observed false negative")
    if (
        float(source_artifacts["minimum_preregistered_zero_false_negative_threshold"])
        != FROZEN_DECISION_MARGIN_THRESHOLD
    ):
        raise ValueError("NR calibration did not freeze the minimum zero-FN threshold")
    if source_artifacts["observed_actual_divergence_outcome_counts"] != {
        "actual_numerical_decision_divergence": 1,
        "actual_unresolved_semantic_divergence": 3,
    }:
        raise ValueError("NR calibration actual-divergence taxonomy changed")
    expected_hash = calibration_sha256(calibration)
    if calibration["calibration_sha256"] != expected_hash:
        raise ValueError("NR calibration SHA-256 mismatch")
    provenance = calibration["provenance"]
    mismatches = {
        field: {
            "calibration": provenance.get(field),
            "runtime": runtime_provenance.get(field),
        }
        for field in PROVENANCE_FIELDS
        if provenance.get(field) != runtime_provenance.get(field)
    }
    if mismatches:
        raise ValueError(f"NR calibration/runtime provenance mismatch: {mismatches}")
    return calibration


def classify_guard_event(
    *,
    guard_triggered: bool,
    replay_executed: bool,
    packed_replay_decision_equal: bool | None,
    packed_replay_output_equal: bool | None,
) -> dict[str, Any]:
    """Classify guard coverage without inventing unavailable replay outcomes."""

    divergence = (
        None
        if packed_replay_decision_equal is None
        else not bool(packed_replay_decision_equal)
    )
    return {
        "guard_triggered": bool(guard_triggered),
        "replay_executed": bool(replay_executed),
        "packed_replay_decision_divergence": divergence,
        "guard_true_positive": bool(guard_triggered and divergence is True),
        "guard_false_positive": bool(guard_triggered and divergence is False),
        "guard_false_negative": bool(not guard_triggered and divergence is True),
        "guard_prevented_output_divergence": bool(
            guard_triggered
            and divergence is True
            and packed_replay_output_equal is False
        ),
        "classification_unavailable_reason": (
            None if divergence is not None else "authoritative_or_shadow_replay_not_run"
        ),
    }
