#!/usr/bin/env python3
"""Shared execution and aggregation helpers for secondary H2.2 N32 evidence."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from scripts.aggregate_phase2d2a_h_known import (
    _candidate_metrics,
    _component_cost,
    _formal_timing,
    _online_metrics,
    _provenance,
    _read_json,
    _read_jsonl,
    _validate_schema,
)
from scripts.audit_phase2d2a_h211_final_route import (
    AUDIT_CSV_NAME as H211_AUDIT_CSV_NAME,
)
from scripts.audit_phase2d2a_h211_final_route import (
    AUDIT_NAME as H211_AUDIT_NAME,
)
from scripts.audit_phase2d2a_h211_final_route import (
    MANIFEST_NAME as H211_MANIFEST_NAME,
)
from scripts.check_phase2d2_acceptance import (
    REQUIRED as PHASE2D2_REQUIRED_ARTIFACTS,
)
from scripts.check_phase2d2_acceptance import check_output
from scripts.check_phase2d2a_h21_acceptance import check_h21_output
from scripts.check_phase2d2a_h211_acceptance import check_h211_output
from scripts.freeze_phase2d2a_h2_cohort import (
    BASE_FAMILY,
    CANDIDATE_LABEL_AUTHORITY,
    CANDIDATE_LABEL_SOURCE,
    FAMILIES,
    FROZEN_REFINEMENT_FAMILIES,
    INDEPENDENT_TASKS,
    SECONDARY_NODE_BUDGET,
    VERIFICATION_MODE,
    _sha256_json,
)
from scripts.phase2d2a_h21_common import (
    BATCH_SIZE,
    BLOCK_SIZE,
    DEVICE,
    DRAFT_MODEL_ID,
    DRAFT_REVISION,
    DTYPE,
    H21_ARTIFACTS,
    MAX_NEW_TOKENS,
    REQUIRED_FREEZE_SHA256,
    SELECTOR_MODE,
    TARGET_MODEL_ID,
    TARGET_REVISION,
    TEMPERATURE,
    TIMING_REPEATS,
    WARMUP_RUNS,
    _candidate_counts,
    _validate_runtime_refinement_identity,
    _validated_frozen_refinement,
    distribution,
    ensure_immutable_freeze,
    matrix_identity,
    read_json,
    sha256_file,
    validate_h21_freeze,
    write_csv,
    write_json,
)
from scripts.phase2d2a_h21_common import (
    load_cell as load_h21_cell,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = REPO_ROOT / "docs" / "schemas"
SCHEMA_VERSION = "1.0"
PHASE = "2D.2A-H2.2"
SECONDARY_MATRIX_NAME = "phase2d2a-h2-secondary-matrix.json"
ACTUAL_WORK_REGIME = "budget_cap_expansion"
EXPECTED_ROW_COUNT = 112
EXPECTED_PAIR_COUNT = 84

ANALYSIS_CONTRACT_NAME = "phase2d2a-h22-analysis-contract.json"
EXECUTION_MANIFEST_NAME = "phase2d2a-h22-execution-manifest.json"
EXECUTION_PLAN_NAME = "phase2d2a-h22-execution-plan.json"
SUBRUN_INDEX_NAME = "phase2d2a-h22-subrun-index.jsonl"
EXTERNAL_STATUS_NAME = "h22-external-status.json"
SENTINEL_GATE_NAME = "phase2d2a-h22-sentinel-gate.json"

AGGREGATE_JSON_NAMES = (
    "phase2d2a-h22-cell-summary.json",
    "phase2d2a-h22-n32-paired-comparison.json",
    "phase2d2a-h22-cross-budget-paired-comparison.json",
    "phase2d2a-h22-incremental-budget-decomposition.json",
    "phase2d2a-h22-task-context-summary.json",
    "phase2d2a-h22-mechanism-summary.json",
    "phase2d2a-h22-cost-summary.json",
    "phase2d2a-h22-secondary-conclusion.json",
)
AGGREGATE_CSV_NAMES = tuple(
    name.replace(".json", ".csv") for name in AGGREGATE_JSON_NAMES[:-1]
)
H22_ARTIFACTS = (
    ANALYSIS_CONTRACT_NAME,
    EXECUTION_MANIFEST_NAME,
    EXECUTION_PLAN_NAME,
    SUBRUN_INDEX_NAME,
    SENTINEL_GATE_NAME,
    *AGGREGATE_JSON_NAMES,
    *AGGREGATE_CSV_NAMES,
)
TASK_ORDER = {"qasper": 0, "hotpotqa": 1, "gov_report": 2}
FAMILY_ORDER = {family: index for index, family in enumerate(FAMILIES)}


def validate_h22_schema(payload: Any) -> None:
    schema = read_json(SCHEMA_ROOT / "phase2d2a-h22-artifact.schema.json")
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)


def validate_h22_external_status(payload: Any) -> None:
    schema = read_json(SCHEMA_ROOT / "phase2d2a-h22-external-status.schema.json")
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)


def validate_h22_freeze(
    freeze_dir: Path,
    supplied_freeze_sha256: str,
    *,
    required_freeze_sha256: str = REQUIRED_FREEZE_SHA256,
) -> dict[str, Any]:
    bundle = validate_h21_freeze(
        freeze_dir,
        supplied_freeze_sha256,
        required_freeze_sha256=required_freeze_sha256,
    )
    secondary = read_json(bundle["freeze_dir"] / SECONDARY_MATRIX_NAME)
    if secondary["row_count"] != EXPECTED_ROW_COUNT or len(secondary["rows"]) != (
        EXPECTED_ROW_COUNT
    ):
        raise ValueError("H2.2 secondary matrix must contain exactly 112 rows")
    if secondary["node_budget"] != SECONDARY_NODE_BUDGET:
        raise ValueError("H2.2 secondary matrix is not N32")
    if secondary["actual_work_regime"] != ACTUAL_WORK_REGIME:
        raise ValueError("H2.2 matrix is not budget-cap expansion")
    if tuple(secondary["frozen_configuration_families"]) != FAMILIES:
        raise ValueError("H2.2 frozen family inventory changed")
    if any(
        row["node_budget"] != SECONDARY_NODE_BUDGET
        or row["verification_mode"] != VERIFICATION_MODE
        or row["online_selector_mode"] != SELECTOR_MODE
        or row["candidate_label_authority"] != CANDIDATE_LABEL_AUTHORITY
        or row["candidate_label_source"] != CANDIDATE_LABEL_SOURCE
        or row["q1_replay_used_as_target_label_source"]
        for row in secondary["rows"]
    ):
        raise ValueError("H2.2 secondary matrix violates frozen semantics")
    bundle["secondary"] = secondary
    bundle["frozen_component_canonical_sha256"] = {
        **bundle["frozen_component_canonical_sha256"],
        "secondary_matrix": _sha256_json(secondary),
    }
    return bundle


def _hash_files(root: Path, names: Iterable[str]) -> dict[str, str]:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise ValueError(f"immutable anchor lacks artifacts: {missing}")
    return {name: sha256_file(root / name) for name in names}


def source_bindings(
    *,
    bundle: dict[str, Any],
    h21_output_root: Path,
    h211_output_dir: Path,
) -> dict[str, Any]:
    h21_output_root = h21_output_root.expanduser().resolve()
    h211_output_dir = h211_output_dir.expanduser().resolve()
    h21_acceptance = check_h21_output(
        h21_output_root,
        bundle["freeze_dir"],
        bundle["freeze_content_sha256"],
        required_freeze_sha256=REQUIRED_FREEZE_SHA256,
    )
    h211_acceptance = check_h211_output(
        h211_output_dir,
        bundle["freeze_dir"],
        bundle["freeze_content_sha256"],
        h21_output_root,
    )
    h21_hashes = _hash_files(h21_output_root, H21_ARTIFACTS)
    h211_names = (H211_AUDIT_NAME, H211_AUDIT_CSV_NAME, H211_MANIFEST_NAME)
    h211_hashes = _hash_files(h211_output_dir, h211_names)
    combined = {"h21": h21_hashes, "h211": h211_hashes}
    return {
        "h21_output_root": str(h21_output_root),
        "h211_output_dir": str(h211_output_dir),
        "h21_acceptance": h21_acceptance,
        "h211_acceptance": h211_acceptance,
        "h21_artifact_file_sha256": h21_hashes,
        "h211_artifact_file_sha256": h211_hashes,
        "h21_anchor_binding_sha256": _sha256_json(h21_hashes),
        "h211_route_binding_sha256": _sha256_json(h211_hashes),
        "source_binding_sha256": _sha256_json(combined),
    }


def execution_id(matrix_row_index: int, row: dict[str, Any]) -> str:
    identity = {
        "matrix_row_index": matrix_row_index,
        "task": row["task"],
        "sample_id": row["sample_id"],
        "nominal_prompt_length": row["nominal_prompt_length"],
        "configuration_family": row["configuration_family"],
        "node_budget": row["node_budget"],
    }
    return f"h22-secondary-{matrix_row_index:03d}-{_sha256_json(identity)[:12]}"


def result_relative_dir(row: dict[str, Any]) -> str:
    return str(
        Path("runs")
        / row["task"]
        / row["sample_id"]
        / str(row["nominal_prompt_length"])
        / row["configuration_family"]
    )


def build_execution_plan(
    bundle: dict[str, Any], bindings: dict[str, Any]
) -> dict[str, Any]:
    rows = []
    for index, frozen in enumerate(bundle["secondary"]["rows"]):
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "matrix_row_index": index,
                "execution_id": execution_id(index, frozen),
                **frozen,
                "block_size": BLOCK_SIZE,
                "temperature": TEMPERATURE,
                "device": DEVICE,
                "target_model_id": TARGET_MODEL_ID,
                "target_model_revision": TARGET_REVISION,
                "draft_model_id": DRAFT_MODEL_ID,
                "draft_model_revision": DRAFT_REVISION,
                "freeze_content_sha256": bundle["freeze_content_sha256"],
                "h21_anchor_binding_sha256": bindings["h21_anchor_binding_sha256"],
                "h211_route_binding_sha256": bindings["h211_route_binding_sha256"],
                "result_relative_dir": result_relative_dir(frozen),
            }
        )
    if (
        len(rows) != EXPECTED_ROW_COUNT
        or len({row["execution_id"] for row in rows}) != EXPECTED_ROW_COUNT
    ):
        raise ValueError("H2.2 plan is not an exact unique 112-row plan")
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "artifact_type": "execution_plan",
        "status": "planned_not_executed",
        "freeze_content_sha256": bundle["freeze_content_sha256"],
        "h21_anchor_binding_sha256": bindings["h21_anchor_binding_sha256"],
        "h211_route_binding_sha256": bindings["h211_route_binding_sha256"],
        "source_secondary_matrix_sha256": bundle["frozen_component_canonical_sha256"][
            "secondary_matrix"
        ],
        "scientific_role": "secondary_budget_expansion_characterization",
        "actual_work_regime": ACTUAL_WORK_REGIME,
        "node_budget": SECONDARY_NODE_BUDGET,
        "row_count": len(rows),
        "configuration_families": list(FAMILIES),
        "execution_grouping_strategy": "one_existing_phase2d2_invocation_per_frozen_row",
        "expected_target_draft_model_load_count": len(rows),
        "model_forward_executed": False,
        "rows": rows,
    }


def validate_plan(
    plan: dict[str, Any], bundle: dict[str, Any], bindings: dict[str, Any]
) -> None:
    validate_h22_schema(plan)
    if plan != build_execution_plan(bundle, bindings):
        raise ValueError("H2.2 execution plan differs from frozen secondary matrix")


def build_analysis_contract(
    bundle: dict[str, Any], bindings: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "artifact_type": "analysis_contract",
        "status": "frozen_not_executed",
        "freeze_content_sha256": bundle["freeze_content_sha256"],
        "h21_anchor_binding_sha256": bindings["h21_anchor_binding_sha256"],
        "h211_route_binding_sha256": bindings["h211_route_binding_sha256"],
        "scientific_role": "secondary_budget_expansion_characterization_not_rescue",
        "configured_node_budget": SECONDARY_NODE_BUDGET,
        "comparison_axes": {
            "within_n32": "combined_refinement_plus_actual_work_difference_when_work_differs",
            "same_family_cross_budget": "n16_family_vs_n32_same_family",
            "paired_incremental_budget_decomposition": (
                "observed_paired_decomposition_not_causal_estimator"
            ),
        },
        "marginal_yield_denominator": "measured_delta_actual_verified_nodes_per_round",
        "zero_actual_node_delta_yield": "unavailable_not_zero",
        "outcome_dependent_thresholds_allowed": False,
        "arbitrary_low_marginal_yield_threshold_defined": False,
        "h22_can_override_h21_primary": False,
        "winner_selection_allowed": False,
        "phase2d2b_or_phase2d3_entry_allowed": False,
        "speedup_claim_allowed": False,
        "model_forward_executed": False,
    }


def build_execution_manifest(
    bundle: dict[str, Any],
    bindings: dict[str, Any],
    plan: dict[str, Any],
    runtime_paths: dict[str, Any],
    *,
    execution_authorized: bool,
    repository_commit: str | None,
    branch: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "artifact_type": "execution_manifest",
        "status": "planned_not_executed",
        "freeze_content_sha256": bundle["freeze_content_sha256"],
        "freeze_artifact_file_sha256": bundle["freeze_artifact_file_sha256"],
        "frozen_component_canonical_sha256": bundle[
            "frozen_component_canonical_sha256"
        ],
        **bindings,
        "h20_artifacts_mutated": False,
        "h21_artifacts_mutated": False,
        "h211_artifacts_mutated": False,
        "execute_frozen_secondary_authorized_by_cli": execution_authorized,
        "node_budget": SECONDARY_NODE_BUDGET,
        "scientific_role": "secondary_budget_expansion_characterization",
        "actual_work_regime": ACTUAL_WORK_REGIME,
        "matrix_expected_row_count": EXPECTED_ROW_COUNT,
        "execution_grouping_strategy": plan["execution_grouping_strategy"],
        "expected_target_draft_model_load_count": plan[
            "expected_target_draft_model_load_count"
        ],
        "repository_commit": repository_commit,
        "branch": branch,
        "runtime_paths": runtime_paths,
        "model_forward_executed": False,
        "completed_cell_count": 0,
        "incomplete_cell_count": EXPECTED_ROW_COUNT,
        "infrastructure_failed_cell_count": 0,
        "scientific_correctness_invalid_cell_count": 0,
        "not_started_cell_count": EXPECTED_ROW_COUNT,
        "winner_selected": False,
        "phase2d2b_entered": False,
        "phase2d3_entered": False,
        "speedup_claimed": False,
        "h22_can_override_h21_primary": False,
        "artifacts": list(H22_ARTIFACTS),
    }


def ensure_immutable_sources(
    manifest: dict[str, Any], bundle: dict[str, Any], bindings: dict[str, Any]
) -> None:
    ensure_immutable_freeze(manifest, bundle)
    for field in (
        "h21_artifact_file_sha256",
        "h211_artifact_file_sha256",
        "h21_anchor_binding_sha256",
        "h211_route_binding_sha256",
        "source_binding_sha256",
    ):
        if manifest[field] != bindings[field]:
            raise ValueError(f"H2.2 immutable source binding changed: {field}")


def _frozen_refinement_for_h22_row(
    output_root: Path, plan_row: dict[str, Any]
) -> dict[str, Any] | None:
    if plan_row["configuration_family"] == BASE_FAMILY:
        return None
    execution_manifest = read_json(output_root / EXECUTION_MANIFEST_NAME)
    configured = execution_manifest["runtime_paths"]["frozen_refinement_config"]
    return _validated_frozen_refinement(Path(configured))


def _validate_result_identity(
    root: Path,
    row: dict[str, Any],
    manifest: dict[str, Any],
    sample: dict[str, Any],
    frozen_refinement: dict[str, Any] | None,
) -> None:
    for field in (
        "task",
        "sample_id",
        "source_sample_id",
        "source_text_sha256",
        "nominal_prompt_length",
        "actual_prompt_length",
        "context_prefix_sha256",
    ):
        if sample.get(field) != row[field]:
            raise ValueError(f"{root}: frozen result identity mismatch: {field}")
    phase = manifest["phase2d2"]
    config = manifest["configuration"]
    expected_runtime = {
        "node_budget": SECONDARY_NODE_BUDGET,
        "verification_mode": VERIFICATION_MODE,
        "online_selector_mode": SELECTOR_MODE,
        "frozen_refinement_configuration_id": row["frozen_refinement_configuration_id"],
    }
    for field, expected in expected_runtime.items():
        if phase.get(field) != expected:
            raise ValueError(f"{root}: frozen runtime field changed: {field}")
    expected_config = {
        "target_model": TARGET_MODEL_ID,
        "target_revision": TARGET_REVISION,
        "draft_model": DRAFT_MODEL_ID,
        "draft_revision": DRAFT_REVISION,
        "context_lengths": [row["nominal_prompt_length"]],
        "max_new_tokens": MAX_NEW_TOKENS,
        "warmup_runs_per_context_length": WARMUP_RUNS,
        "timing_repeats": TIMING_REPEATS,
        "batch_size": BATCH_SIZE,
        "temperature": TEMPERATURE,
        "block_size": BLOCK_SIZE,
        "dtype": DTYPE,
        "device": DEVICE,
    }
    for field, expected in expected_config.items():
        if config.get(field) != expected:
            raise ValueError(f"{root}: frozen configuration changed: {field}")
    _validate_runtime_refinement_identity(root, row, phase, sample, frozen_refinement)
    if phase.get("candidate_quality_target_label_authority") != (
        CANDIDATE_LABEL_AUTHORITY
    ):
        raise ValueError(f"{root}: Original-DFlash label authority changed")
    if phase.get("candidate_quality_target_label_source") != CANDIDATE_LABEL_SOURCE:
        raise ValueError(f"{root}: Original-DFlash label source changed")
    if (
        phase.get("candidate_quality_q1_replay_used_as_target_label_source")
        is not False
    ):
        raise ValueError(f"{root}: q1 entered target labels")
    if phase.get("evidence_role") != "held_out_evaluation":
        raise ValueError(f"{root}: result is not held-out evaluation evidence")


def verification_work(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    actual = [int(row["phase2d2"]["actual_verified_node_count"]) for row in rounds]
    selected = [int(row["phase2d2"]["selected_tree_node_count"]) for row in rounds]
    return {
        "actual_verified_node_count_distribution": distribution(actual),
        "selected_tree_node_count_distribution": distribution(selected),
        "actual_verified_node_counts": actual,
        "selected_tree_node_counts": selected,
        "actual_verified_nodes_per_round": (
            sum(actual) / len(actual) if actual else None
        ),
        "selected_tree_nodes_per_round": (
            sum(selected) / len(selected) if selected else None
        ),
        "round_count": len(actual),
        "actual_work_derived_from_round_artifacts": True,
        "configured_node_budget_not_used_as_actual_work": True,
    }


def load_h22_cell(output_root: Path, plan_row: dict[str, Any]) -> dict[str, Any]:
    root = output_root / plan_row["result_relative_dir"]
    status_path = root / EXTERNAL_STATUS_NAME
    try:
        if not status_path.is_file():
            raise ValueError("missing H2.2 external status")
        external = read_json(status_path)
        validate_h22_external_status(external)
        if external["frozen_matrix_row_identity"] != matrix_identity(plan_row):
            raise ValueError("external status matrix identity mismatch")
        if external["execution_end_status"] != "completed":
            raise ValueError(
                f"external status incomplete: {external['execution_end_status']}"
            )
        missing = [
            name for name in PHASE2D2_REQUIRED_ARTIFACTS if not (root / name).is_file()
        ]
        if missing:
            raise ValueError(f"incomplete Phase2D2 artifacts: {missing}")
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
            raise ValueError("H2.2 requires exactly one sample per cell")
        sample = samples[0]
        _validate_schema(sample, "phase2d2-sample.schema.json")
        for round_row in rounds:
            _validate_schema(round_row, "phase2d2-round.schema.json")
        _validate_result_identity(
            root,
            plan_row,
            manifest,
            sample,
            _frozen_refinement_for_h22_row(output_root, plan_row),
        )
        related = [
            row
            for row in rounds
            if row.get("task") == sample["task"]
            and row.get("sample_id") == sample["sample_id"]
            and int(row.get("nominal_prompt_length", 0))
            == int(sample["nominal_prompt_length"])
            and row.get("configuration_id") == sample["configuration_id"]
        ]
        if not related or len(related) != len(rounds):
            raise ValueError("round/sample identity is incomplete or mixed")
        candidate = _candidate_metrics(related, root)
        candidate.update(_candidate_counts(related))
        checks = {
            "original_dflash_reference_tree_output_exact": bool(
                sample["correctness"].get("original_dflash_reference_output_ids_equal")
                and correctness.get("all_original_dflash_reference_output_ids_equal")
            ),
            "actual_execution_correctness_passed": bool(
                sample["correctness"].get("actual_execution_correctness_passed")
                and correctness.get("actual_execution_correctness_passed")
            ),
            "formal_repeat_stable": all(
                repeat.get("output_ids_equal_repeat_0") is True
                for repeat in sample["formal_timing_repeats"]
            ),
            "profile_formal_output_exact": (
                sample["profiling"].get("output_ids_equal_formal") is True
            ),
            "diagnostic_formal_output_exact": (
                sample["diagnostics"].get("output_ids_equal_formal") is True
            ),
            "structural_and_kv_correctness_passed": bool(
                sample["correctness"].get("structural_checks_passed")
                and correctness.get("structural_correctness_passed")
            ),
            "always_fixed_verifier_provenance": (
                sample.get("verification_mode") == VERIFICATION_MODE
            ),
            "original_dflash_only_target_labels": (
                candidate["target_label_authority"] == CANDIDATE_LABEL_AUTHORITY
                and candidate["target_label_source"] == CANDIDATE_LABEL_SOURCE
                and not candidate["q1_replay_used_as_target_label_source"]
            ),
        }
        try:
            checker_result = check_output(root)
        except Exception as exc:  # noqa: BLE001 - scientific rejection is retained.
            checker_status = "rejected"
            checker_error = f"{type(exc).__name__}: {exc}"
            checker_result = None
        else:
            checker_status = "accepted"
            checker_error = None
        expected_checker_rc = 0 if checker_status == "accepted" else 1
        if external["external_phase2d2_checker_return_code"] != expected_checker_rc:
            raise ValueError(
                "recorded external checker return code is not reproducible"
            )
        correctness_valid = all(checks.values()) and checker_status == "accepted"
        online = _online_metrics(sample, related)
        formal = _formal_timing(sample)
        component = _component_cost(sample, related)
        provenance = _provenance(manifest)
        if external["repository_commit"] != provenance["repository_commit"]:
            raise ValueError("external/result repository commits differ")
        return {
            "schema_version": SCHEMA_VERSION,
            **matrix_identity(plan_row),
            "result_directory": str(root),
            "cell_status": "valid" if correctness_valid else "correctness_invalid",
            "infrastructure_failure_reason": None,
            "correctness": {
                **checks,
                "external_phase2d2_checker_status": checker_status,
                "external_phase2d2_checker_result": checker_result,
                "external_phase2d2_checker_error": checker_error,
                "hard_correctness_failure_reasons": correctness.get(
                    "hard_correctness_failure_reasons", []
                ),
                "correctness_valid": correctness_valid,
            },
            "candidate_quality": candidate,
            "online_efficiency": online,
            "formal_timing": formal,
            "component_cost": component,
            "provenance": provenance,
            "verification_work": verification_work(related),
            "generated_output_token_ids": sample["generated_output_token_ids"],
        }
    except Exception as exc:  # noqa: BLE001 - broken evidence is infrastructure.
        return {
            "schema_version": SCHEMA_VERSION,
            **matrix_identity(plan_row),
            "result_directory": str(root),
            "cell_status": "infrastructure_failed",
            "infrastructure_failure_reason": f"{type(exc).__name__}: {exc}",
            "correctness": None,
            "candidate_quality": None,
            "online_efficiency": None,
            "formal_timing": None,
            "component_cost": None,
            "provenance": None,
            "verification_work": None,
            "generated_output_token_ids": None,
        }


def load_h21_anchor_cells(
    h21_output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    h21_output_root = h21_output_root.expanduser().resolve()
    plan = read_json(h21_output_root / "phase2d2a-h21-execution-plan.json")
    cells = []
    for row in plan["rows"]:
        cell = load_h21_cell(h21_output_root, row)
        if cell["cell_status"] == "infrastructure_failed":
            raise ValueError(
                f"accepted H2.1 anchor cell became invalid: {row['execution_id']}"
            )
        root = h21_output_root / row["result_relative_dir"]
        rounds = _read_jsonl(root / "refinement-rounds.jsonl")
        samples = _read_jsonl(root / "samples.jsonl")
        if len(samples) != 1:
            raise ValueError("accepted H2.1 anchor sample inventory changed")
        cell = {
            **cell,
            "verification_work": verification_work(rounds),
            "generated_output_token_ids": samples[0]["generated_output_token_ids"],
        }
        cells.append(cell)
    return plan, cells


def _safe_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(right) - float(left)


def _rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _metric(cell: dict[str, Any], field: str) -> float | None:
    value = cell["online_efficiency"].get(field)
    return float(value) if value is not None else None


def _coverage(cell: dict[str, Any]) -> float | None:
    value = cell["candidate_quality"].get("selected_tree_coverage_rate_at_fixed_n")
    return float(value) if value is not None else None


def _cell_key(cell: dict[str, Any]) -> tuple[Any, ...]:
    return (
        cell["task"],
        cell["sample_id"],
        int(cell["nominal_prompt_length"]),
        cell["configuration_family"],
    )


def _validate_cross_budget_identity(
    left: dict[str, Any], right: dict[str, Any]
) -> None:
    fields = (
        "task",
        "sample_id",
        "source_sample_id",
        "source_text_sha256",
        "nominal_prompt_length",
        "actual_prompt_length",
        "context_prefix_sha256",
        "configuration_family",
        "frozen_refinement_configuration_id",
    )
    for field in fields:
        if left[field] != right[field]:
            raise ValueError(f"N16/N32 pairing mismatch: {field}")
    if left["node_budget"] != 16 or right["node_budget"] != 32:
        raise ValueError("cross-budget pair is not exact N16-to-N32")


def cross_budget_pairs(
    h21_cells: list[dict[str, Any]], h22_cells: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    left = {_cell_key(cell): cell for cell in h21_cells}
    right = {_cell_key(cell): cell for cell in h22_cells}
    if set(left) != set(right) or len(left) != EXPECTED_ROW_COUNT:
        raise ValueError("N16/N32 same-family pairing inventory differs")
    rows = []
    for key in sorted(
        left,
        key=lambda item: (TASK_ORDER[item[0]], item[1], item[2], FAMILY_ORDER[item[3]]),
    ):
        n16 = left[key]
        n32 = right[key]
        _validate_cross_budget_identity(n16, n32)
        valid = n16["cell_status"] == "valid" and n32["cell_status"] == "valid"
        if valid:
            node_delta = _safe_delta(
                n16["verification_work"]["actual_verified_nodes_per_round"],
                n32["verification_work"]["actual_verified_nodes_per_round"],
            )
            accepted_delta = _safe_delta(
                _metric(n16, "accepted_draft_tokens_per_round"),
                _metric(n32, "accepted_draft_tokens_per_round"),
            )
            committed_delta = _safe_delta(
                _metric(n16, "committed_tokens_per_round"),
                _metric(n32, "committed_tokens_per_round"),
            )
            marginal_reason = (
                None
                if node_delta and node_delta > 0
                else (
                    "zero_actual_verified_node_delta"
                    if node_delta == 0
                    else "non_positive_actual_verified_node_delta"
                )
            )
        else:
            node_delta = accepted_delta = committed_delta = None
            marginal_reason = "correctness_invalid_or_infrastructure_incomplete"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "task": key[0],
                "sample_id": key[1],
                "nominal_prompt_length": key[2],
                "configuration_family": key[3],
                "comparison_valid": valid,
                "n16_cell_status": n16["cell_status"],
                "n32_cell_status": n32["cell_status"],
                "delta_actual_verified_nodes_per_round": node_delta,
                "delta_accepted_draft_tokens_per_round": accepted_delta,
                "delta_committed_tokens_per_round": committed_delta,
                "delta_accepted_tokens_per_verified_node": (
                    _safe_delta(
                        _metric(n16, "accepted_draft_tokens_per_verified_node"),
                        _metric(n32, "accepted_draft_tokens_per_verified_node"),
                    )
                    if valid
                    else None
                ),
                "delta_selected_tree_coverage": (
                    _safe_delta(_coverage(n16), _coverage(n32)) if valid else None
                ),
                "delta_rounds_per_returned_token": (
                    _safe_delta(
                        _metric(n16, "rounds_per_returned_token"),
                        _metric(n32, "rounds_per_returned_token"),
                    )
                    if valid
                    else None
                ),
                "nominal_cuda_ms_per_returned_token_change": (
                    _safe_delta(
                        n16["formal_timing"]["generation_cuda_ms_per_returned_token"],
                        n32["formal_timing"]["generation_cuda_ms_per_returned_token"],
                    )
                    if valid
                    else None
                ),
                "n16_metrics": {
                    "accepted_draft_tokens_per_verified_node": (
                        _metric(n16, "accepted_draft_tokens_per_verified_node")
                        if valid
                        else None
                    ),
                    "accepted_draft_tokens_per_round": (
                        _metric(n16, "accepted_draft_tokens_per_round")
                        if valid
                        else None
                    ),
                    "committed_tokens_per_round": (
                        _metric(n16, "committed_tokens_per_round") if valid else None
                    ),
                    "selected_tree_coverage": _coverage(n16) if valid else None,
                    "rounds_per_returned_token": (
                        _metric(n16, "rounds_per_returned_token") if valid else None
                    ),
                    "nominal_cuda_ms_per_returned_token": (
                        n16["formal_timing"]["generation_cuda_ms_per_returned_token"]
                        if valid
                        else None
                    ),
                },
                "n32_metrics": {
                    "accepted_draft_tokens_per_verified_node": (
                        _metric(n32, "accepted_draft_tokens_per_verified_node")
                        if valid
                        else None
                    ),
                    "accepted_draft_tokens_per_round": (
                        _metric(n32, "accepted_draft_tokens_per_round")
                        if valid
                        else None
                    ),
                    "committed_tokens_per_round": (
                        _metric(n32, "committed_tokens_per_round") if valid else None
                    ),
                    "selected_tree_coverage": _coverage(n32) if valid else None,
                    "rounds_per_returned_token": (
                        _metric(n32, "rounds_per_returned_token") if valid else None
                    ),
                    "nominal_cuda_ms_per_returned_token": (
                        n32["formal_timing"]["generation_cuda_ms_per_returned_token"]
                        if valid
                        else None
                    ),
                },
                "accepted_tokens_per_additional_verified_node": (
                    accepted_delta / node_delta
                    if valid and node_delta is not None and node_delta > 0
                    else None
                ),
                "committed_tokens_per_additional_verified_node": (
                    committed_delta / node_delta
                    if valid and node_delta is not None and node_delta > 0
                    else None
                ),
                "marginal_yield_unavailable_reason": marginal_reason,
                "configured_budget_used_as_denominator": False,
                "n16_actual_verified_node_count_distribution": n16["verification_work"][
                    "actual_verified_node_count_distribution"
                ]
                if valid
                else None,
                "n32_actual_verified_node_count_distribution": n32["verification_work"][
                    "actual_verified_node_count_distribution"
                ]
                if valid
                else None,
                "n16_selected_tree_node_count_distribution": n16["verification_work"][
                    "selected_tree_node_count_distribution"
                ]
                if valid
                else None,
                "n32_selected_tree_node_count_distribution": n32["verification_work"][
                    "selected_tree_node_count_distribution"
                ]
                if valid
                else None,
                "generated_output_ids_equal": (
                    n16["generated_output_token_ids"]
                    == n32["generated_output_token_ids"]
                    if valid
                    else None
                ),
            }
        )
    return rows


def within_n32_pairs(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {_cell_key(cell): cell for cell in cells}
    rows = []
    anchors = sorted(
        {
            (cell["task"], cell["sample_id"], cell["nominal_prompt_length"])
            for cell in cells
        },
        key=lambda item: (TASK_ORDER[item[0]], item[1], item[2]),
    )
    for task, sample_id, context in anchors:
        base = by_key[(task, sample_id, context, BASE_FAMILY)]
        for family in FROZEN_REFINEMENT_FAMILIES:
            refined = by_key[(task, sample_id, context, family)]
            for field in (
                "task",
                "task_role",
                "sample_id",
                "source_sample_id",
                "source_text_sha256",
                "selection_rank",
                "selection_hash",
                "nominal_prompt_length",
                "actual_prompt_length",
                "context_prefix_sha256",
                "node_budget",
                "verification_mode",
                "online_selector_mode",
                "max_new_tokens",
                "warmup_runs",
                "timing_repeats",
                "dtype",
                "freeze_content_sha256",
            ):
                if base[field] != refined[field]:
                    raise ValueError(f"within-budget pairing mismatch: {field}")
            valid = base["cell_status"] == "valid" and refined["cell_status"] == "valid"
            base_work = (
                base["verification_work"]["actual_verified_nodes_per_round"]
                if valid
                else None
            )
            refined_work = (
                refined["verification_work"]["actual_verified_nodes_per_round"]
                if valid
                else None
            )
            work_delta = _safe_delta(base_work, refined_work) if valid else None
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "task": task,
                    "sample_id": sample_id,
                    "nominal_prompt_length": context,
                    "base_family": BASE_FAMILY,
                    "refined_family": family,
                    "comparison_valid": valid,
                    "comparison_axis": "within_n32_system_comparison",
                    "comparison_interpretation": (
                        "combined_refinement_plus_actual_work_difference"
                        if valid and work_delta != 0
                        else "within_n32_equal_observed_work_comparison"
                        if valid
                        else "correctness_invalid_or_infrastructure_incomplete"
                    ),
                    "called_same_work": False,
                    "delta_actual_verified_nodes_per_round": work_delta,
                    "metrics": {
                        field: {
                            "base": _metric(base, field) if valid else None,
                            "refined": _metric(refined, field) if valid else None,
                            "delta": _safe_delta(
                                _metric(base, field), _metric(refined, field)
                            )
                            if valid
                            else None,
                        }
                        for field in (
                            "accepted_draft_tokens_per_verified_node",
                            "accepted_draft_tokens_per_round",
                            "committed_tokens_per_round",
                            "rounds_per_returned_token",
                        )
                    },
                    "actual_verified_nodes_per_round": {
                        "base": base_work,
                        "refined": refined_work,
                        "delta": work_delta,
                    },
                    "selected_tree_coverage": {
                        "base": _coverage(base) if valid else None,
                        "refined": _coverage(refined) if valid else None,
                        "delta": _safe_delta(_coverage(base), _coverage(refined))
                        if valid
                        else None,
                    },
                    "nominal_cuda_ms_per_returned_token": {
                        "base": base["formal_timing"][
                            "generation_cuda_ms_per_returned_token"
                        ]
                        if valid
                        else None,
                        "refined": refined["formal_timing"][
                            "generation_cuda_ms_per_returned_token"
                        ]
                        if valid
                        else None,
                    },
                    "component_cost": (
                        {
                            "base": base["component_cost"],
                            "refined": refined["component_cost"],
                            "delta": {
                                field: _safe_delta(
                                    base["component_cost"].get(field),
                                    refined["component_cost"].get(field),
                                )
                                for field in (
                                    "conditional_remask_profile_total_cuda_ms",
                                    "extended_fixed_shape_path_profile_total_cuda_ms",
                                    "combined_conditional_and_fixed_profile_total_cuda_ms",
                                    "profile_round_cuda_ms",
                                )
                            },
                        }
                        if valid
                        else None
                    ),
                }
            )
    if len(rows) != EXPECTED_PAIR_COUNT:
        raise ValueError("H2.2 within-N32 pair inventory is not 84")
    return rows


def incremental_budget_decomposition(
    h21_within: list[dict[str, Any]], h22_within: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    key = lambda row: (
        row["task"],
        row["sample_id"],
        row["nominal_prompt_length"],
        row["refined_family"],
    )
    left = {key(row): row for row in h21_within}
    right = {key(row): row for row in h22_within}
    if set(left) != set(right) or len(left) != EXPECTED_PAIR_COUNT:
        raise ValueError("paired decomposition inventory differs across budgets")
    rows = []
    for identity in sorted(
        left,
        key=lambda item: (TASK_ORDER[item[0]], item[1], item[2], FAMILY_ORDER[item[3]]),
    ):
        n16 = left[identity]
        n32 = right[identity]
        valid = n16["comparison_valid"] and n32["comparison_valid"]

        def gain(row: dict[str, Any], field: str) -> float | None:
            metric = row["metrics"][field]
            return _safe_delta(metric["base"], metric["refined"])

        accepted_n16 = gain(n16, "accepted_draft_tokens_per_round") if valid else None
        accepted_n32 = gain(n32, "accepted_draft_tokens_per_round") if valid else None
        committed_n16 = gain(n16, "committed_tokens_per_round") if valid else None
        committed_n32 = gain(n32, "committed_tokens_per_round") if valid else None
        work_n16 = n16["delta_actual_verified_nodes_per_round"] if valid else None
        work_n32 = n32["delta_actual_verified_nodes_per_round"] if valid else None
        incremental_work = _safe_delta(work_n16, work_n32) if valid else None
        accepted_contribution = (
            _safe_delta(accepted_n16, accepted_n32) if valid else None
        )
        committed_contribution = (
            _safe_delta(committed_n16, committed_n32) if valid else None
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "task": identity[0],
                "sample_id": identity[1],
                "nominal_prompt_length": identity[2],
                "refined_family": identity[3],
                "comparison_valid": valid,
                "decomposition_name": "paired_incremental_budget_decomposition",
                "causal_estimator_claimed": False,
                "same_work_refinement_gain_n16_accepted_tokens_per_round": accepted_n16,
                "combined_gain_n32_accepted_tokens_per_round": accepted_n32,
                "incremental_budget_contribution_accepted_tokens_per_round": accepted_contribution,
                "same_work_refinement_gain_n16_committed_tokens_per_round": committed_n16,
                "combined_gain_n32_committed_tokens_per_round": committed_n32,
                "incremental_budget_contribution_committed_tokens_per_round": committed_contribution,
                "incremental_actual_work": incremental_work,
                "accepted_incremental_budget_marginal_yield": (
                    accepted_contribution / incremental_work
                    if valid and incremental_work is not None and incremental_work > 0
                    else None
                ),
                "committed_incremental_budget_marginal_yield": (
                    committed_contribution / incremental_work
                    if valid and incremental_work is not None and incremental_work > 0
                    else None
                ),
                "marginal_yield_unavailable_reason": (
                    None
                    if valid and incremental_work is not None and incremental_work > 0
                    else "zero_incremental_actual_work"
                    if valid and incremental_work == 0
                    else "non_positive_incremental_actual_work"
                    if valid
                    else "correctness_invalid_or_infrastructure_incomplete"
                ),
                "selected_tree_coverage_decomposition": {
                    "same_work_gain_n16": n16["selected_tree_coverage"]["delta"]
                    if valid
                    else None,
                    "combined_gain_n32": n32["selected_tree_coverage"]["delta"]
                    if valid
                    else None,
                    "observed_incremental_difference": _safe_delta(
                        n16["selected_tree_coverage"]["delta"],
                        n32["selected_tree_coverage"]["delta"],
                    )
                    if valid
                    else None,
                    "interpretation": "descriptive_observed_paired_decomposition",
                },
            }
        )
    return rows


def _distribution(values: Iterable[float | None]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "values": [],
        }
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "min": min(clean),
        "max": max(clean),
        "values": clean,
    }


def task_context_summaries(
    cells: list[dict[str, Any]], cross_budget: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for family in FAMILIES:
        for task, context in sorted(
            {(cell["task"], cell["nominal_prompt_length"]) for cell in cells},
            key=lambda item: (TASK_ORDER[item[0]], item[1]),
        ):
            group = [
                cell
                for cell in cells
                if cell["configuration_family"] == family
                and cell["task"] == task
                and cell["nominal_prompt_length"] == context
                and cell["cell_status"] == "valid"
            ]
            cross = [
                row
                for row in cross_budget
                if row["configuration_family"] == family
                and row["task"] == task
                and row["nominal_prompt_length"] == context
                and row["comparison_valid"]
            ]
            labelled = sum(
                int(cell["candidate_quality"]["labelled_reference_position_count"])
                for cell in group
            )
            selected = sum(
                int(cell["candidate_quality"]["selected_tree_membership_count"])
                for cell in group
            )
            accepted = sum(
                int(cell["online_efficiency"]["accepted_draft_token_count"])
                for cell in group
            )
            verified = sum(
                int(cell["online_efficiency"]["verified_target_node_count"])
                for cell in group
            )
            delta_nodes = sum(
                float(row["delta_actual_verified_nodes_per_round"] or 0)
                for row in cross
            )
            delta_accepted = sum(
                float(row["delta_accepted_draft_tokens_per_round"] or 0)
                for row in cross
            )
            delta_committed = sum(
                float(row["delta_committed_tokens_per_round"] or 0) for row in cross
            )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "task": task,
                    "task_role": (
                        "independent_held_out_task"
                        if task in INDEPENDENT_TASKS
                        else "same_task_replication_only"
                    ),
                    "nominal_prompt_length": context,
                    "configuration_family": family,
                    "valid_n32_sample_count": len(group),
                    "n32_selected_tree_coverage": _rate(selected, labelled),
                    "n32_accepted_tokens_per_verified_node": _rate(accepted, verified),
                    "n32_candidate_coverage_aggregation": "pooled_membership_over_pooled_labelled_positions",
                    "n32_accepted_verified_aggregation": "sum_accepted_over_sum_actual_verified_nodes",
                    "cross_budget_valid_sample_count": len(cross),
                    "pooled_delta_actual_verified_nodes_per_round": delta_nodes,
                    "pooled_delta_accepted_tokens_per_round": delta_accepted,
                    "pooled_delta_committed_tokens_per_round": delta_committed,
                    "pooled_accepted_tokens_per_additional_verified_node": (
                        delta_accepted / delta_nodes if delta_nodes > 0 else None
                    ),
                    "pooled_committed_tokens_per_additional_verified_node": (
                        delta_committed / delta_nodes if delta_nodes > 0 else None
                    ),
                    "marginal_yield_unavailable_reason": (
                        None
                        if delta_nodes > 0
                        else "non_positive_pooled_actual_work_delta"
                    ),
                    "per_sample_delta_actual_work_distribution": _distribution(
                        row["delta_actual_verified_nodes_per_round"] for row in cross
                    ),
                    "per_sample_n32_selected_tree_coverage_distribution": (
                        _distribution(_coverage(cell) for cell in group)
                    ),
                    "per_sample_n32_accepted_per_verified_distribution": (
                        _distribution(
                            _metric(
                                cell,
                                "accepted_draft_tokens_per_verified_node",
                            )
                            for cell in group
                        )
                    ),
                    "per_sample_accepted_marginal_yield_distribution": _distribution(
                        row["accepted_tokens_per_additional_verified_node"]
                        for row in cross
                    ),
                    "hotpotqa_and_qasper_pooled_together": False,
                }
            )
    return rows


def mechanism_summaries(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for family in FROZEN_REFINEMENT_FAMILIES:
        for task, context in sorted(
            {(cell["task"], cell["nominal_prompt_length"]) for cell in cells},
            key=lambda item: (TASK_ORDER[item[0]], item[1]),
        ):
            group = [
                cell
                for cell in cells
                if cell["configuration_family"] == family
                and cell["task"] == task
                and cell["nominal_prompt_length"] == context
                and cell["cell_status"] == "valid"
            ]
            labelled = sum(
                int(cell["candidate_quality"]["labelled_reference_position_count"])
                for cell in group
            )
            counts = {
                field: sum(int(cell["candidate_quality"][field]) for cell in group)
                for field in (
                    "base_candidate_membership_count",
                    "conditional_candidate_membership_count",
                    "merged_candidate_membership_count",
                    "selected_tree_membership_count",
                    "refined_only_target_recovery_count",
                )
            }
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "task": task,
                    "nominal_prompt_length": context,
                    "refined_family": family,
                    "valid_sample_count": len(group),
                    "labelled_reference_position_count": labelled,
                    "base_candidate_coverage": _rate(
                        counts["base_candidate_membership_count"], labelled
                    ),
                    "conditional_candidate_coverage": _rate(
                        counts["conditional_candidate_membership_count"], labelled
                    ),
                    "merged_candidate_coverage": _rate(
                        counts["merged_candidate_membership_count"], labelled
                    ),
                    "selected_tree_coverage": _rate(
                        counts["selected_tree_membership_count"], labelled
                    ),
                    "refined_only_target_recovery_count": counts[
                        "refined_only_target_recovery_count"
                    ],
                    "refined_only_target_recovery_rate": _rate(
                        counts["refined_only_target_recovery_count"], labelled
                    ),
                    "target_token_rank_improvement_distribution": _distribution(
                        cell["candidate_quality"][
                            "target_token_rank_improvement_distribution"
                        ].get("mean")
                        for cell in group
                    ),
                    "correct_prefix_survival_depth_distribution": _distribution(
                        cell["candidate_quality"][
                            "correct_prefix_survival_depth_distribution"
                        ].get("mean")
                        for cell in group
                    ),
                    "h22_secondary_evidence_only": True,
                    "h21_primary_rewritten": False,
                }
            )
    return rows


def cost_summaries(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for family in FROZEN_REFINEMENT_FAMILIES:
        for task, context in sorted(
            {(cell["task"], cell["nominal_prompt_length"]) for cell in cells},
            key=lambda item: (TASK_ORDER[item[0]], item[1]),
        ):
            group = [
                cell
                for cell in cells
                if cell["configuration_family"] == family
                and cell["task"] == task
                and cell["nominal_prompt_length"] == context
                and cell["cell_status"] == "valid"
            ]
            dependency = [
                cell["component_cost"]["conditional_remask_profile_total_cuda_ms"]
                for cell in group
            ]
            verifier = [
                cell["component_cost"][
                    "extended_fixed_shape_path_profile_total_cuda_ms"
                ]
                for cell in group
            ]
            combined = [
                cell["component_cost"][
                    "combined_conditional_and_fixed_profile_total_cuda_ms"
                ]
                for cell in group
            ]
            comparisons = [
                None
                if dependency_cost is None or verifier_cost is None
                else verifier_cost > dependency_cost
                for dependency_cost, verifier_cost in zip(
                    dependency, verifier, strict=True
                )
            ]
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "task": task,
                    "nominal_prompt_length": context,
                    "refined_family": family,
                    "valid_sample_count": len(group),
                    "dependency_modeling_cost": _distribution(dependency),
                    "correctness_preserving_verifier_cost": _distribution(verifier),
                    "combined_non_overlapping_additional_work": _distribution(combined),
                    "verifier_cost_greater_than_dependency_cost_count": sum(
                        value is True for value in comparisons
                    ),
                    "comparable_sample_count": sum(
                        value is not None for value in comparisons
                    ),
                    "verifier_cost_greater_for_all_comparable_samples": bool(
                        comparisons and all(value is True for value in comparisons)
                    ),
                    "nested_timer_double_counting_avoided": bool(
                        group
                        and all(
                            cell["component_cost"][
                                "nested_timer_double_counting_avoided"
                            ]
                            for cell in group
                        )
                    ),
                    "speedup_claimed": False,
                }
            )
    return rows


def _h21_within_pairs(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Reuse the H2.2 structure while preserving the measured N16 work.
    return within_n32_pairs(cells)


def build_aggregate(
    *,
    bundle: dict[str, Any],
    bindings: dict[str, Any],
    plan: dict[str, Any],
    h21_cells: list[dict[str, Any]],
    h22_cells: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if len(h21_cells) != EXPECTED_ROW_COUNT or len(h22_cells) != EXPECTED_ROW_COUNT:
        raise ValueError("H2.2 aggregation requires exact 112-cell N16/N32 matrices")
    within = within_n32_pairs(h22_cells)
    cross = cross_budget_pairs(h21_cells, h22_cells)
    n16_within = _h21_within_pairs(h21_cells)
    decomposition = incremental_budget_decomposition(n16_within, within)
    task_context = task_context_summaries(h22_cells, cross)
    mechanisms = mechanism_summaries(h22_cells)
    costs = cost_summaries(h22_cells)
    status = Counter(cell["cell_status"] for cell in h22_cells)
    base_sanity = [row for row in cross if row["configuration_family"] == BASE_FAMILY]
    base_cap_not_consumed = all(
        row["comparison_valid"] and row["delta_actual_verified_nodes_per_round"] == 0
        for row in base_sanity
    )
    common = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": "complete",
        "freeze_content_sha256": bundle["freeze_content_sha256"],
        "h21_anchor_binding_sha256": bindings["h21_anchor_binding_sha256"],
        "h211_route_binding_sha256": bindings["h211_route_binding_sha256"],
    }
    artifacts = {
        "cell_summary": {
            **common,
            "artifact_type": "cell_summary",
            "expected_cell_count": EXPECTED_ROW_COUNT,
            "observed_cell_count": len(h22_cells),
            "valid_cell_count": status["valid"],
            "correctness_invalid_cell_count": status["correctness_invalid"],
            "infrastructure_failed_cell_count": status["infrastructure_failed"],
            "correctness_invalid_excluded_from_benefit_aggregation": True,
            "rows": h22_cells,
        },
        "within_n32": {
            **common,
            "artifact_type": "n32_paired_comparison",
            "comparison_axis": "within_n32_system_comparison",
            "same_work_claimed": False,
            "expected_pair_count": EXPECTED_PAIR_COUNT,
            "rows": within,
        },
        "cross_budget": {
            **common,
            "artifact_type": "cross_budget_paired_comparison",
            "comparison_axis": "same_family_n16_to_n32",
            "actual_work_denominator_source": "round_artifacts",
            "configured_budget_used_as_actual_work": False,
            "rows": cross,
        },
        "decomposition": {
            **common,
            "artifact_type": "incremental_budget_decomposition",
            "decomposition_name": "paired_incremental_budget_decomposition",
            "causal_estimator_claimed": False,
            "rows": decomposition,
        },
        "task_context": {
            **common,
            "artifact_type": "task_context_summary",
            "independent_tasks": list(INDEPENDENT_TASKS),
            "govreport_contributes_to_independent_task_interpretation": False,
            "hotpotqa_and_qasper_pooled_together": False,
            "rows": task_context,
        },
        "mechanism": {
            **common,
            "artifact_type": "mechanism_summary",
            "h21_primary_rewritten": False,
            "new_candidate_availability_declared_dominant": False,
            "rows": mechanisms,
        },
        "cost": {
            **common,
            "artifact_type": "cost_summary",
            "nested_timer_double_counting_avoided": True,
            "speedup_claimed": False,
            "rows": costs,
        },
        "conclusion": {
            **common,
            "artifact_type": "secondary_conclusion",
            "scientific_role": "secondary_budget_expansion_characterization",
            "execution_complete": status["infrastructure_failed"] == 0,
            "base_n16_n32_sanity": {
                "base_budget_cap_not_consumed": base_cap_not_consumed,
                "rows": base_sanity,
            },
            "characterizes_extra_actual_work_consumption": True,
            "characterizes_marginal_progress_per_additional_verified_node": True,
            "stable_cross_task_same_work_signal_produced": False,
            "h21_primary_rewritten": False,
            "h22_can_override_h21_primary": False,
            "winner_selected": False,
            "phase2d2b_entered": False,
            "phase2d3_entered": False,
            "speedup_claimed": False,
        },
    }
    for payload in artifacts.values():
        validate_h22_schema(payload)
    return artifacts


def persist_aggregate(output_root: Path, artifacts: dict[str, dict[str, Any]]) -> None:
    mapping = {
        "cell_summary": AGGREGATE_JSON_NAMES[0],
        "within_n32": AGGREGATE_JSON_NAMES[1],
        "cross_budget": AGGREGATE_JSON_NAMES[2],
        "decomposition": AGGREGATE_JSON_NAMES[3],
        "task_context": AGGREGATE_JSON_NAMES[4],
        "mechanism": AGGREGATE_JSON_NAMES[5],
        "cost": AGGREGATE_JSON_NAMES[6],
        "conclusion": AGGREGATE_JSON_NAMES[7],
    }
    for key, name in mapping.items():
        write_json(output_root / name, artifacts[key])
        if name != AGGREGATE_JSON_NAMES[-1]:
            write_csv(
                output_root / name.replace(".json", ".csv"), artifacts[key]["rows"]
            )
