#!/usr/bin/env python3
"""Shared frozen-input and aggregation helpers for Phase 2D.2A-H2.1."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
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
from scripts.check_phase2d2_acceptance import (
    REQUIRED as PHASE2D2_REQUIRED_ARTIFACTS,
)
from scripts.check_phase2d2_acceptance import check_output
from scripts.check_phase2d2a_h2_freeze import check_h2_freeze
from scripts.freeze_phase2d2a_h2_cohort import (
    ARTIFACTS as H20_ARTIFACTS,
)
from scripts.freeze_phase2d2a_h2_cohort import (
    BASE_FAMILY,
    CANDIDATE_LABEL_AUTHORITY,
    CANDIDATE_LABEL_SOURCE,
    FAMILIES,
    FROZEN_REFINEMENT_FAMILIES,
    INDEPENDENT_TASKS,
    KNOWN_TASK,
    PRIMARY_NODE_BUDGET,
    TASK_SAMPLE_COUNTS,
    VERIFICATION_MODE,
    _sha256_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = REPO_ROOT / "docs" / "schemas"
SCHEMA_VERSION = "1.0"
PHASE = "2D.2A-H2.1"

REQUIRED_FREEZE_SHA256 = (
    "d5fdc816b183c19bd5683fb89a3c5e1ace641a15d2ac68da7b6dc0e56a678856"
)
PRIMARY_MATRIX_NAME = "phase2d2a-h2-primary-matrix.json"
COHORT_NAME = "phase2d2a-h2-frozen-cohort.json"
RULES_NAME = "phase2d2a-h2-aggregation-rules.json"
FREEZE_MANIFEST_NAME = "phase2d2a-h2-freeze-manifest.json"

TARGET_MODEL_ID = "Qwen/Qwen3-8B"
TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DRAFT_MODEL_ID = "z-lab/Qwen3-8B-DFlash-b16"
DRAFT_REVISION = "9b41424b7109f9c5413454f481b09a82b85333f4"
SELECTOR_MODE = "top1_packed_tree"
ACTUAL_WORK_REGIME = "same_actual_verification_work"
BLOCK_SIZE = 16
MAX_NEW_TOKENS = 128
WARMUP_RUNS = 1
TIMING_REPEATS = 3
BATCH_SIZE = 1
TEMPERATURE = 0.0
DTYPE = "bfloat16"
DEVICE = "cuda:0"

TASK_ORDER = {"qasper": 0, "hotpotqa": 1, "gov_report": 2}
FAMILY_ORDER = {family: index for index, family in enumerate(FAMILIES)}
EXPECTED_TASK_ROW_COUNTS = {"gov_report": 24, "qasper": 24, "hotpotqa": 64}
EXPECTED_PRIMARY_ROW_COUNT = 112
EXPECTED_PAIR_COUNT = 84
EXPECTED_INDEPENDENT_STRATA = (
    ("qasper", 1024),
    ("qasper", 8192),
    ("hotpotqa", 1024),
    ("hotpotqa", 8192),
)

EXECUTION_MANIFEST_NAME = "phase2d2a-h21-execution-manifest.json"
EXECUTION_PLAN_NAME = "phase2d2a-h21-execution-plan.json"
SUBRUN_INDEX_NAME = "phase2d2a-h21-subrun-index.jsonl"
EXTERNAL_STATUS_NAME = "h21-external-status.json"

AGGREGATE_JSON_ARTIFACTS = (
    "phase2d2a-h21-cell-summary.json",
    "phase2d2a-h21-sample-paired-comparison.json",
    "phase2d2a-h21-task-context-summary.json",
    "phase2d2a-h21-family-summary.json",
    "phase2d2a-h21-mechanism-summary.json",
    "phase2d2a-h21-cost-summary.json",
    "phase2d2a-h21-primary-conclusion.json",
)
AGGREGATE_CSV_ARTIFACTS = tuple(
    name.replace(".json", ".csv") for name in AGGREGATE_JSON_ARTIFACTS[:-1]
)
H21_ARTIFACTS = (
    EXECUTION_MANIFEST_NAME,
    EXECUTION_PLAN_NAME,
    SUBRUN_INDEX_NAME,
    *AGGREGATE_JSON_ARTIFACTS,
    *AGGREGATE_CSV_ARTIFACTS,
)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _flat(value: Any) -> Any:
    return (
        json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row}) or ["schema_version"]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _flat(row.get(field)) for field in fields})
    temporary.replace(path)


def validate_h21_schema(payload: Any) -> None:
    schema = read_json(SCHEMA_ROOT / "phase2d2a-h21-artifact.schema.json")
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)


def validate_external_status_schema(payload: Any) -> None:
    schema = read_json(SCHEMA_ROOT / "phase2d2a-h21-external-status.schema.json")
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _freeze_file_hashes(freeze_dir: Path) -> dict[str, str]:
    return {name: sha256_file(freeze_dir / name) for name in H20_ARTIFACTS}


def validate_h21_freeze(
    freeze_dir: Path,
    supplied_freeze_sha256: str,
    *,
    required_freeze_sha256: str = REQUIRED_FREEZE_SHA256,
) -> dict[str, Any]:
    """Validate all immutable H2.0 inputs before any model-loading code path."""

    freeze_dir = freeze_dir.expanduser().resolve()
    if supplied_freeze_sha256 != required_freeze_sha256:
        raise ValueError(
            "H2.1 requires the exact frozen H2.0 content SHA256; "
            f"expected={required_freeze_sha256}, supplied={supplied_freeze_sha256}"
        )
    checker = check_h2_freeze(freeze_dir)
    manifest = read_json(freeze_dir / FREEZE_MANIFEST_NAME)
    cohort = read_json(freeze_dir / COHORT_NAME)
    primary = read_json(freeze_dir / PRIMARY_MATRIX_NAME)
    rules = read_json(freeze_dir / RULES_NAME)
    if manifest["status"] != "frozen_not_executed":
        raise ValueError("H2.0 freeze is not frozen_not_executed")
    if manifest["freeze_content_sha256"] != supplied_freeze_sha256:
        raise ValueError("H2.0 manifest freeze SHA differs from supplied SHA")
    if checker["freeze_content_sha256"] != supplied_freeze_sha256:
        raise ValueError("H2.0 checker freeze SHA differs from supplied SHA")
    if (
        primary["row_count"] != EXPECTED_PRIMARY_ROW_COUNT
        or len(primary["rows"]) != EXPECTED_PRIMARY_ROW_COUNT
    ):
        raise ValueError("H2.1 primary matrix must contain exactly 112 rows")
    if primary["node_budget"] != PRIMARY_NODE_BUDGET:
        raise ValueError("H2.1 primary matrix is not N16")
    if primary["actual_work_regime"] != ACTUAL_WORK_REGIME:
        raise ValueError("H2.1 primary matrix is not same-actual-verification-work")
    if tuple(primary["frozen_configuration_families"]) != FAMILIES:
        raise ValueError("H2.1 primary matrix family set changed")
    task_counts = Counter(row["task"] for row in primary["rows"])
    if dict(task_counts) != EXPECTED_TASK_ROW_COUNTS:
        raise ValueError(f"H2.1 task row counts changed: {dict(task_counts)}")
    if any(
        row["node_budget"] != 16
        or row["verification_mode"] != VERIFICATION_MODE
        or row["online_selector_mode"] != SELECTOR_MODE
        or row["candidate_label_authority"] != CANDIDATE_LABEL_AUTHORITY
        or row["candidate_label_source"] != CANDIDATE_LABEL_SOURCE
        or row["q1_replay_used_as_target_label_source"]
        for row in primary["rows"]
    ):
        raise ValueError("H2.1 primary matrix violates frozen N16/provenance settings")
    return {
        "freeze_dir": freeze_dir,
        "checker": checker,
        "manifest": manifest,
        "cohort": cohort,
        "primary": primary,
        "rules": rules,
        "freeze_content_sha256": supplied_freeze_sha256,
        "freeze_artifact_file_sha256": _freeze_file_hashes(freeze_dir),
        "frozen_component_canonical_sha256": {
            "frozen_cohort": _sha256_json(cohort),
            "primary_matrix": _sha256_json(primary),
            "aggregation_rules": _sha256_json(rules),
        },
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
    return f"h21-primary-{matrix_row_index:03d}-{_sha256_json(identity)[:12]}"


def result_relative_dir(row: dict[str, Any]) -> str:
    return str(
        Path("runs")
        / row["task"]
        / row["sample_id"]
        / str(row["nominal_prompt_length"])
        / row["configuration_family"]
    )


def build_execution_plan(bundle: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for index, frozen in enumerate(bundle["primary"]["rows"]):
        row = {
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
            "result_relative_dir": result_relative_dir(frozen),
        }
        rows.append(row)
    identities = [row["execution_id"] for row in rows]
    if len(rows) != EXPECTED_PRIMARY_ROW_COUNT or len(set(identities)) != len(rows):
        raise ValueError("H2.1 execution plan is not an exact unique 112-row plan")
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "artifact_type": "execution_plan",
        "status": "planned_not_executed",
        "freeze_content_sha256": bundle["freeze_content_sha256"],
        "source_primary_matrix_sha256": bundle["frozen_component_canonical_sha256"][
            "primary_matrix"
        ],
        "matrix_role": "primary_same_actual_verification_work",
        "actual_work_regime": ACTUAL_WORK_REGIME,
        "node_budget": 16,
        "row_count": len(rows),
        "task_row_counts": dict(sorted(Counter(row["task"] for row in rows).items())),
        "task_context_row_counts": dict(
            sorted(
                Counter(
                    f"{row['task']}:{row['nominal_prompt_length']}" for row in rows
                ).items()
            )
        ),
        "family_row_counts": dict(
            sorted(Counter(row["configuration_family"] for row in rows).items())
        ),
        "configuration_families": list(FAMILIES),
        "refinement_families": list(FROZEN_REFINEMENT_FAMILIES),
        "execution_grouping_strategy": "one_existing_phase2d2_invocation_per_frozen_row",
        "expected_target_draft_model_load_count": len(rows),
        "model_forward_executed": False,
        "rows": rows,
    }


def execution_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        TASK_ORDER[row["task"]],
        int(row["selection_rank"]),
        int(row["nominal_prompt_length"]),
        FAMILY_ORDER[row["configuration_family"]],
    )


def matrix_identity(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "matrix_row_index",
        "execution_id",
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
        "configuration_family",
        "frozen_refinement_configuration_id",
        "node_budget",
        "verification_mode",
        "online_selector_mode",
        "max_new_tokens",
        "warmup_runs",
        "timing_repeats",
        "dtype",
        "freeze_content_sha256",
    )
    return {field: row[field] for field in fields}


def exact_plan_row_matches_frozen(
    plan_row: dict[str, Any], frozen: dict[str, Any]
) -> bool:
    additions = {
        "schema_version",
        "matrix_row_index",
        "execution_id",
        "block_size",
        "temperature",
        "device",
        "target_model_id",
        "target_model_revision",
        "draft_model_id",
        "draft_model_revision",
        "freeze_content_sha256",
        "result_relative_dir",
    }
    observed = {key: value for key, value in plan_row.items() if key not in additions}
    return observed == frozen


def validate_plan(plan: dict[str, Any], bundle: dict[str, Any]) -> None:
    validate_h21_schema(plan)
    expected = build_execution_plan(bundle)
    if plan != expected:
        raise ValueError("H2.1 execution plan differs from the frozen primary matrix")
    if any(
        not exact_plan_row_matches_frozen(row, frozen)
        for row, frozen in zip(plan["rows"], bundle["primary"]["rows"], strict=True)
    ):
        raise ValueError("H2.1 plan row identity differs from frozen primary matrix")


def build_execution_manifest(
    bundle: dict[str, Any],
    *,
    plan: dict[str, Any],
    runtime_paths: dict[str, Any],
    execution_authorized: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "artifact_type": "execution_manifest",
        "status": "planned_not_executed",
        "freeze_content_sha256": bundle["freeze_content_sha256"],
        "freeze_directory": str(bundle["freeze_dir"]),
        "freeze_status": bundle["manifest"]["status"],
        "freeze_artifact_file_sha256": bundle["freeze_artifact_file_sha256"],
        "frozen_component_canonical_sha256": bundle[
            "frozen_component_canonical_sha256"
        ],
        "h20_artifacts_mutated": False,
        "h2_primary_execution_authorized_by_cli": execution_authorized,
        "h2_secondary_execution_authorized": False,
        "node_budget": 16,
        "actual_work_regime": ACTUAL_WORK_REGIME,
        "matrix_expected_row_count": EXPECTED_PRIMARY_ROW_COUNT,
        "configuration_families": list(FAMILIES),
        "execution_grouping_strategy": plan["execution_grouping_strategy"],
        "expected_target_draft_model_load_count": plan[
            "expected_target_draft_model_load_count"
        ],
        "grouping_rationale": (
            "The existing runner accepts one explicit sample ID and one family. "
            "Its correctness/checker scope is an output directory, so combining "
            "contexts could make one context's failure contaminate another cell. "
            "One invocation per frozen row preserves independent cell artifacts."
        ),
        "repository_commit": git_value("rev-parse", "HEAD"),
        "branch": git_value("branch", "--show-current"),
        "runtime_paths": runtime_paths,
        "model_forward_executed": False,
        "completed_cell_count": 0,
        "incomplete_cell_count": EXPECTED_PRIMARY_ROW_COUNT,
        "scientific_correctness_invalid_cell_count": 0,
        "winner_selected": False,
        "final_phase2d2a_branch_entered": False,
        "speedup_claim_authorized": False,
        "artifacts": list(H21_ARTIFACTS),
    }


def ensure_immutable_freeze(manifest: dict[str, Any], bundle: dict[str, Any]) -> None:
    if manifest["freeze_content_sha256"] != bundle["freeze_content_sha256"]:
        raise ValueError("existing H2.1 output root uses a conflicting freeze SHA")
    if manifest["freeze_artifact_file_sha256"] != bundle["freeze_artifact_file_sha256"]:
        raise ValueError("immutable H2.0 artifacts changed after H2.1 planning")
    if (
        manifest["frozen_component_canonical_sha256"]
        != bundle["frozen_component_canonical_sha256"]
    ):
        raise ValueError("immutable H2.0 component content changed")


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _absolute(base: float | None, refined: float | None) -> float | None:
    return None if base is None or refined is None else refined - base


def _relative(base: float | None, refined: float | None) -> float | None:
    return None if base in (None, 0) or refined is None else (refined - base) / base


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]


def distribution(values: Iterable[float | int | None]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean) if clean else None,
        "median": statistics.median(clean) if clean else None,
        "p90": _p90(clean),
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
        "values": clean,
    }


def _sign(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _joint_sign(coverage: float | None, accepted: float | None, valid: bool) -> str:
    if not valid:
        return "invalid"
    if coverage is None or accepted is None:
        return "invalid"
    if coverage > 0 and accepted > 0:
        return "positive_positive"
    if (coverage > 0) != (accepted > 0):
        return "mixed_sign"
    return "non_positive"


def _candidate_counts(rounds: list[dict[str, Any]]) -> dict[str, int]:
    positions = [
        position
        for row in rounds
        if row["candidate_quality"].get("availability") == "available"
        for position in row["candidate_quality"]["positions"]
    ]
    return {
        "base_candidate_membership_count": sum(
            bool(row.get("base_top_m_membership")) for row in positions
        ),
        "conditional_candidate_membership_count": sum(
            bool(row.get("conditional_top_r_membership")) for row in positions
        ),
        "merged_candidate_membership_count": sum(
            bool(row.get("merged_membership")) for row in positions
        ),
        "selected_tree_membership_count": sum(
            bool(row.get("selected_tree_membership")) for row in positions
        ),
    }


def _infrastructure_cell(plan_row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        **matrix_identity(plan_row),
        "result_directory": plan_row["result_relative_dir"],
        "cell_status": "infrastructure_failed",
        "infrastructure_failure_reason": reason,
        "correctness": None,
        "candidate_quality": None,
        "online_efficiency": None,
        "formal_timing": None,
        "component_cost": None,
        "provenance": None,
    }


def _validate_result_identity(
    root: Path,
    row: dict[str, Any],
    manifest: dict[str, Any],
    sample: dict[str, Any],
) -> None:
    expected_sample = {
        "task": row["task"],
        "sample_id": row["sample_id"],
        "source_sample_id": row["source_sample_id"],
        "source_text_sha256": row["source_text_sha256"],
        "context_prefix_sha256": row["context_prefix_sha256"],
        "nominal_prompt_length": row["nominal_prompt_length"],
        "actual_prompt_length": row["actual_prompt_length"],
    }
    for field, value in expected_sample.items():
        if sample.get(field) != value:
            raise ValueError(
                f"{root}: frozen {field} mismatch: {sample.get(field)!r} != {value!r}"
            )
    phase = manifest.get("phase2d2", {})
    config = manifest.get("configuration", {})
    expected_runtime = {
        "node_budget": 16,
        "verification_mode": VERIFICATION_MODE,
        "online_selector_mode": SELECTOR_MODE,
        "frozen_refinement_configuration_id": row["frozen_refinement_configuration_id"],
    }
    for field, value in expected_runtime.items():
        if phase.get(field) != value:
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
    for field, value in expected_config.items():
        if config.get(field) != value:
            raise ValueError(f"{root}: frozen configuration field changed: {field}")
    expected_mode = (
        BASE_FAMILY if row["configuration_family"] == BASE_FAMILY else "refined_tree"
    )
    if phase.get("candidate_merge_mode") != expected_mode:
        raise ValueError(f"{root}: candidate merge mode differs from frozen family")
    if (
        phase.get("candidate_quality_target_label_authority")
        != CANDIDATE_LABEL_AUTHORITY
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


def load_cell(output_root: Path, plan_row: dict[str, Any]) -> dict[str, Any]:
    root = output_root / plan_row["result_relative_dir"]
    status_path = root / EXTERNAL_STATUS_NAME
    if not status_path.is_file():
        return _infrastructure_cell(plan_row, "missing_h21_external_status")
    try:
        external = read_json(status_path)
        validate_external_status_schema(external)
        if external["freeze_content_sha256"] != plan_row["freeze_content_sha256"]:
            raise ValueError("external status freeze SHA mismatch")
        if external["frozen_matrix_row_identity"] != matrix_identity(plan_row):
            raise ValueError("external status matrix identity mismatch")
        if external["execution_end_status"] != "completed":
            raise ValueError(
                f"external status is incomplete: {external['execution_end_status']}"
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
            raise ValueError("H2.1 requires exactly one frozen cell per directory")
        sample = samples[0]
        _validate_schema(sample, "phase2d2-sample.schema.json")
        for round_row in rounds:
            _validate_schema(round_row, "phase2d2-round.schema.json")
        _validate_result_identity(root, plan_row, manifest, sample)
        related = [
            round_row
            for round_row in rounds
            if round_row.get("task") == sample["task"]
            and round_row.get("sample_id") == sample["sample_id"]
            and int(round_row.get("nominal_prompt_length", 0))
            == int(sample["nominal_prompt_length"])
            and round_row.get("configuration_id") == sample["configuration_id"]
        ]
        if not related or len(related) != len(rounds):
            raise ValueError("round/sample identity is incomplete or mixed")
        candidate = _candidate_metrics(related, root)
        candidate.update(_candidate_counts(related))
        online = _online_metrics(sample, related)
        formal = _formal_timing(sample)
        formal["generation_cuda_ms_per_committed_token"] = _ratio(
            formal["generation_cuda_median_ms"], online["committed_token_count"]
        )
        repeat_stable = all(
            repeat.get("output_ids_equal_repeat_0") is True
            for repeat in sample["formal_timing_repeats"]
        )
        checks = {
            "original_dflash_reference_tree_output_exact": bool(
                sample["correctness"].get("original_dflash_reference_output_ids_equal")
                and correctness.get("all_original_dflash_reference_output_ids_equal")
            ),
            "actual_execution_correctness_passed": bool(
                sample["correctness"].get("actual_execution_correctness_passed")
                and correctness.get("actual_execution_correctness_passed")
            ),
            "formal_repeat_stable": repeat_stable,
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
        expected_rc = 0 if checker_status == "accepted" else 1
        if external["external_phase2d2_checker_return_code"] != expected_rc:
            raise ValueError(
                "recorded external checker return code is not reproducible"
            )
        correctness_valid = all(checks.values()) and checker_status == "accepted"
        correctness_payload = {
            **checks,
            "external_phase2d2_checker_status": checker_status,
            "external_phase2d2_checker_result": checker_result,
            "external_phase2d2_checker_error": checker_error,
            "runner_return_code": external["runner_return_code"],
            "hard_correctness_failure_reasons": correctness.get(
                "hard_correctness_failure_reasons", []
            ),
            "correctness_valid": correctness_valid,
            "target_only_is_auxiliary": True,
            "target_only_output_ids_equal": sample["correctness"].get(
                "target_only_output_ids_equal"
            ),
        }
        provenance = _provenance(manifest)
        if external["repository_commit"] != provenance["repository_commit"]:
            raise ValueError("external status/result repository commits differ")
        return {
            "schema_version": SCHEMA_VERSION,
            **matrix_identity(plan_row),
            "result_directory": str(root),
            "cell_status": "valid" if correctness_valid else "correctness_invalid",
            "infrastructure_failure_reason": None,
            "correctness": correctness_payload,
            "candidate_quality": candidate,
            "online_efficiency": online,
            "formal_timing": formal,
            "component_cost": _component_cost(sample, related),
            "provenance": provenance,
        }
    except Exception as exc:  # noqa: BLE001 - classify broken evidence as infra.
        return _infrastructure_cell(plan_row, f"{type(exc).__name__}: {exc}")


def validate_no_extra_outputs(output_root: Path, plan: dict[str, Any]) -> None:
    runs = output_root / "runs"
    if not runs.exists():
        return
    expected = {
        (output_root / row["result_relative_dir"]).resolve() for row in plan["rows"]
    }
    observed = set()
    malformed = []
    for path in runs.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(runs)
        if len(relative.parts) < 5:
            malformed.append(str(path.resolve()))
            continue
        observed.add((runs.joinpath(*relative.parts[:4])).resolve())
    if malformed:
        raise ValueError(f"malformed extra H2.1 run outputs detected: {malformed}")
    extra = sorted(str(path) for path in observed - expected)
    if extra:
        raise ValueError(f"extra non-frozen H2.1 outputs detected: {extra}")


def pair_cells(
    plan: dict[str, Any], cells: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_key = {
        (
            row["task"],
            row["sample_id"],
            row["nominal_prompt_length"],
            row["configuration_family"],
        ): row
        for row in cells
    }
    pairs = []
    anchors = sorted(
        {
            (row["task"], row["sample_id"], row["nominal_prompt_length"])
            for row in plan["rows"]
        },
        key=lambda key: (TASK_ORDER[key[0]], key[1], key[2]),
    )
    for task, sample_id, context in anchors:
        base = by_key[(task, sample_id, context, BASE_FAMILY)]
        for family in FROZEN_REFINEMENT_FAMILIES:
            refined = by_key[(task, sample_id, context, family)]
            identity_fields = (
                "source_sample_id",
                "source_text_sha256",
                "context_prefix_sha256",
                "actual_prompt_length",
                "node_budget",
                "verification_mode",
                "online_selector_mode",
                "max_new_tokens",
                "warmup_runs",
                "timing_repeats",
                "dtype",
                "freeze_content_sha256",
            )
            if any(base[field] != refined[field] for field in identity_fields):
                raise ValueError(
                    f"base/refined frozen pairing mismatch: {task}/{sample_id}/{context}/{family}"
                )
            incomplete = "infrastructure_failed" in {
                base["cell_status"],
                refined["cell_status"],
            }
            correctness_valid = bool(
                not incomplete
                and base["cell_status"] == "valid"
                and refined["cell_status"] == "valid"
            )
            candidate_sufficient = bool(
                correctness_valid
                and base["candidate_quality"]["labelled_reference_position_count"]
                and refined["candidate_quality"]["labelled_reference_position_count"]
            )

            def metric_pair(
                section: str,
                field: str,
                *,
                base_row: dict[str, Any] = base,
                refined_row: dict[str, Any] = refined,
                eligible: bool = correctness_valid,
            ) -> dict[str, Any]:
                base_value = base_row[section][field] if eligible else None
                refined_value = refined_row[section][field] if eligible else None
                return {
                    "base": base_value,
                    "refined": refined_value,
                    "absolute_gain": _absolute(base_value, refined_value),
                    "relative_gain": _relative(base_value, refined_value),
                }

            candidate_fields = (
                "base_candidate_coverage_rate",
                "conditional_candidate_coverage_rate",
                "merged_candidate_coverage_rate",
                "selected_tree_coverage_rate_at_fixed_n",
                "refined_only_target_recovery_rate",
            )
            candidate = {
                "base_labelled_reference_position_count": (
                    base["candidate_quality"]["labelled_reference_position_count"]
                    if correctness_valid
                    else None
                ),
                "refined_labelled_reference_position_count": (
                    refined["candidate_quality"]["labelled_reference_position_count"]
                    if correctness_valid
                    else None
                ),
                "base_available_round_count": (
                    base["candidate_quality"]["candidate_quality_available_round_count"]
                    if correctness_valid
                    else None
                ),
                "refined_available_round_count": (
                    refined["candidate_quality"][
                        "candidate_quality_available_round_count"
                    ]
                    if correctness_valid
                    else None
                ),
                "base_unavailable_reason_distribution": (
                    base["candidate_quality"]["unavailable_reason_distribution"]
                    if correctness_valid
                    else {}
                ),
                "refined_unavailable_reason_distribution": (
                    refined["candidate_quality"]["unavailable_reason_distribution"]
                    if correctness_valid
                    else {}
                ),
                "base_selected_tree_membership_count": (
                    base["candidate_quality"]["selected_tree_membership_count"]
                    if correctness_valid
                    else None
                ),
                "refined_selected_tree_membership_count": (
                    refined["candidate_quality"]["selected_tree_membership_count"]
                    if correctness_valid
                    else None
                ),
                "refined_base_candidate_membership_count": (
                    refined["candidate_quality"]["base_candidate_membership_count"]
                    if correctness_valid
                    else None
                ),
                "refined_conditional_candidate_membership_count": (
                    refined["candidate_quality"][
                        "conditional_candidate_membership_count"
                    ]
                    if correctness_valid
                    else None
                ),
                "refined_merged_candidate_membership_count": (
                    refined["candidate_quality"]["merged_candidate_membership_count"]
                    if correctness_valid
                    else None
                ),
                "refined_only_target_recovery_count": (
                    refined["candidate_quality"]["refined_only_target_recovery_count"]
                    if candidate_sufficient
                    else None
                ),
                "correct_prefix_survival_depth": (
                    refined["candidate_quality"][
                        "correct_prefix_survival_depth_distribution"
                    ]
                    if candidate_sufficient
                    else None
                ),
                "first_pruning_depth": (
                    refined["candidate_quality"]["first_pruning_depth_distribution"]
                    if candidate_sufficient
                    else None
                ),
                "target_token_rank_improvement": (
                    refined["candidate_quality"][
                        "target_token_rank_improvement_distribution"
                    ]
                    if candidate_sufficient
                    else None
                ),
            }
            for field in candidate_fields:
                base_value = (
                    base["candidate_quality"][field] if candidate_sufficient else None
                )
                refined_value = (
                    refined["candidate_quality"][field]
                    if candidate_sufficient
                    else None
                )
                candidate[field] = {
                    "base": base_value,
                    "refined": refined_value,
                    "absolute_gain": _absolute(base_value, refined_value),
                    "relative_gain": _relative(base_value, refined_value),
                }
            online = {
                field: metric_pair("online_efficiency", field)
                for field in (
                    "accepted_draft_token_count",
                    "committed_token_count",
                    "verified_target_node_count",
                    "round_count",
                    "generated_returned_token_count",
                    "accepted_draft_tokens_per_verified_node",
                    "accepted_draft_tokens_per_round",
                    "committed_tokens_per_round",
                    "verified_target_nodes_per_returned_token",
                    "rounds_per_returned_token",
                )
            }
            timing = {
                field: metric_pair("formal_timing", field)
                for field in (
                    "generation_cuda_median_ms",
                    "generation_cuda_ms_per_returned_token",
                    "returned_tokens_per_second",
                )
            }
            timing["base_cuda_elapsed_ms_repeats"] = (
                base["formal_timing"]["cuda_elapsed_ms_repeats"]
                if correctness_valid
                else []
            )
            timing["refined_cuda_elapsed_ms_repeats"] = (
                refined["formal_timing"]["cuda_elapsed_ms_repeats"]
                if correctness_valid
                else []
            )
            coverage_gain = candidate["selected_tree_coverage_rate_at_fixed_n"][
                "absolute_gain"
            ]
            accepted_gain = online["accepted_draft_tokens_per_verified_node"][
                "absolute_gain"
            ]
            pairs.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "task": task,
                    "task_role": base["task_role"],
                    "sample_id": sample_id,
                    "source_sample_id": base["source_sample_id"],
                    "source_text_sha256": base["source_text_sha256"],
                    "context_prefix_sha256": base["context_prefix_sha256"],
                    "nominal_prompt_length": context,
                    "actual_prompt_length": base["actual_prompt_length"],
                    "node_budget": 16,
                    "base_family": BASE_FAMILY,
                    "refined_family": family,
                    "pair_status": (
                        "incomplete_execution"
                        if incomplete
                        else "valid"
                        if correctness_valid
                        else "invalid_due_to_correctness"
                    ),
                    "correctness_valid": correctness_valid,
                    "candidate_evidence_sufficient": candidate_sufficient,
                    "candidate_quality": candidate,
                    "online_efficiency": online,
                    "formal_timing": timing,
                    "component_cost": (
                        refined["component_cost"] if correctness_valid else None
                    ),
                    "signs": {
                        "coverage_gain": _sign(coverage_gain),
                        "accepted_per_verified_gain": _sign(accepted_gain),
                        "joint_signal": _joint_sign(
                            coverage_gain, accepted_gain, correctness_valid
                        ),
                    },
                }
            )
    if len(pairs) != EXPECTED_PAIR_COUNT:
        raise ValueError(f"H2.1 requires exactly 84 paired rows, got {len(pairs)}")
    return pairs


def validate_cell_provenance(cells: list[dict[str, Any]]) -> None:
    complete = [row for row in cells if row["cell_status"] != "infrastructure_failed"]
    if not complete:
        return
    fields = (
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
    first = complete[0]["provenance"]
    for field in fields:
        values = {
            json.dumps(row["provenance"].get(field), sort_keys=True) for row in complete
        }
        if len(values) != 1 or first.get(field) is None:
            raise ValueError(f"mixed or missing H2.1 cell provenance: {field}")
    expected = {
        "target_model_id": TARGET_MODEL_ID,
        "target_model_revision": TARGET_REVISION,
        "draft_model_id": DRAFT_MODEL_ID,
        "draft_model_revision": DRAFT_REVISION,
        "online_selector_mode": SELECTOR_MODE,
        "verification_mode": VERIFICATION_MODE,
        "dtype": DTYPE,
        "device": DEVICE,
        "block_size": BLOCK_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "timing_repeats": TIMING_REPEATS,
        "warmup_runs": WARMUP_RUNS,
        "batch_size": BATCH_SIZE,
        "temperature": TEMPERATURE,
    }
    for field, value in expected.items():
        if first[field] != value:
            raise ValueError(f"H2.1 cell provenance changed: {field}")
    refined_hashes = {
        row["provenance"].get("frozen_refinement_hash")
        for row in complete
        if row["configuration_family"] != BASE_FAMILY
    }
    if None in refined_hashes or len(refined_hashes) > 1:
        raise ValueError("H2.1 refined cells do not share one frozen refinement hash")
    if any(
        row["provenance"].get("frozen_refinement_hash") is not None
        for row in complete
        if row["configuration_family"] == BASE_FAMILY
    ):
        raise ValueError("H2.1 base controls unexpectedly carry refinement provenance")


def _pooled_candidate(
    pairs: list[dict[str, Any]], side: str
) -> tuple[int, int, float | None]:
    labelled = sum(
        int(row["candidate_quality"][f"{side}_labelled_reference_position_count"] or 0)
        for row in pairs
        if row["correctness_valid"]
    )
    selected = sum(
        int(row["candidate_quality"][f"{side}_selected_tree_membership_count"] or 0)
        for row in pairs
        if row["correctness_valid"]
    )
    return labelled, selected, _ratio(selected, labelled)


def _sign_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    values = Counter(row["signs"][field] for row in rows)
    return {
        label: values.get(label, 0)
        for label in (
            "positive",
            "zero",
            "negative",
            "unavailable",
        )
    }


def task_context_summaries(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    contexts = sorted({(row["task"], row["nominal_prompt_length"]) for row in pairs})
    for family in FROZEN_REFINEMENT_FAMILIES:
        for task, context in contexts:
            cohort = [
                row
                for row in pairs
                if row["refined_family"] == family
                and row["task"] == task
                and row["nominal_prompt_length"] == context
            ]
            expected = TASK_SAMPLE_COUNTS[task]
            if len(cohort) != expected:
                raise ValueError(
                    f"unexpected sample count for {task}/{context}/{family}"
                )
            incomplete = [
                row for row in cohort if row["pair_status"] == "incomplete_execution"
            ]
            invalid = [
                row
                for row in cohort
                if row["pair_status"] == "invalid_due_to_correctness"
            ]
            valid = [row for row in cohort if row["correctness_valid"]]
            base_labelled, base_selected, base_coverage = _pooled_candidate(
                valid, "base"
            )
            refined_labelled, refined_selected, refined_coverage = _pooled_candidate(
                valid, "refined"
            )
            candidate_sufficient = base_labelled > 0 and refined_labelled > 0
            coverage_gain = (
                _absolute(base_coverage, refined_coverage)
                if not incomplete and not invalid and candidate_sufficient
                else None
            )
            base_accepted = sum(
                int(row["online_efficiency"]["accepted_draft_token_count"]["base"])
                for row in valid
            )
            refined_accepted = sum(
                int(row["online_efficiency"]["accepted_draft_token_count"]["refined"])
                for row in valid
            )
            base_verified = sum(
                int(row["online_efficiency"]["verified_target_node_count"]["base"])
                for row in valid
            )
            refined_verified = sum(
                int(row["online_efficiency"]["verified_target_node_count"]["refined"])
                for row in valid
            )
            base_apv = _ratio(base_accepted, base_verified)
            refined_apv = _ratio(refined_accepted, refined_verified)
            accepted_gain = (
                _absolute(base_apv, refined_apv)
                if not incomplete and not invalid
                else None
            )
            if incomplete:
                status = "incomplete_execution"
            elif invalid:
                status = "invalid_due_to_correctness"
            elif not candidate_sufficient:
                status = "insufficient_candidate_label_evidence"
            else:
                status = (
                    "joint_positive"
                    if coverage_gain > 0 and accepted_gain > 0
                    else "not_joint_positive"
                )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "task": task,
                    "task_role": cohort[0]["task_role"],
                    "nominal_prompt_length": context,
                    "refined_family": family,
                    "expected_sample_count": expected,
                    "observed_pair_count": len(cohort),
                    "valid_pair_count": len(valid),
                    "correctness_invalid_pair_count": len(invalid),
                    "infrastructure_incomplete_pair_count": len(incomplete),
                    "stratum_status": status,
                    "candidate_coverage": {
                        "aggregation": "pooled_membership_count_over_pooled_labelled_positions",
                        "base_labelled_position_count": base_labelled,
                        "refined_labelled_position_count": refined_labelled,
                        "base_membership_count": base_selected,
                        "refined_membership_count": refined_selected,
                        "base_selected_tree_coverage": base_coverage,
                        "refined_selected_tree_coverage": refined_coverage,
                        "coverage_absolute_gain": coverage_gain,
                        "mean_of_sample_rates_used_as_primary": False,
                    },
                    "accepted_per_verified": {
                        "aggregation": "sum_accepted_draft_tokens_over_sum_actual_verified_nodes",
                        "base_accepted_draft_token_count": base_accepted,
                        "refined_accepted_draft_token_count": refined_accepted,
                        "base_actual_verified_node_count": base_verified,
                        "refined_actual_verified_node_count": refined_verified,
                        "base_rate": base_apv,
                        "refined_rate": refined_apv,
                        "absolute_gain": accepted_gain,
                        "relative_gain": _relative(base_apv, refined_apv),
                        "mean_of_sample_ratios_used_as_primary": False,
                        "per_sample_absolute_gain_distribution": distribution(
                            row["online_efficiency"][
                                "accepted_draft_tokens_per_verified_node"
                            ]["absolute_gain"]
                            for row in valid
                        ),
                    },
                    "sample_heterogeneity": {
                        "coverage_gain_sign_counts": _sign_counts(
                            cohort, "coverage_gain"
                        ),
                        "accepted_per_verified_gain_sign_counts": _sign_counts(
                            cohort, "accepted_per_verified_gain"
                        ),
                        "joint_signal_counts": dict(
                            sorted(
                                Counter(
                                    row["signs"]["joint_signal"] for row in cohort
                                ).items()
                            )
                        ),
                    },
                    "formal_timing": {
                        "interpretation": "nominal_latency_change_only",
                        "base_cuda_ms_per_returned_token_distribution": distribution(
                            row["formal_timing"][
                                "generation_cuda_ms_per_returned_token"
                            ]["base"]
                            for row in valid
                        ),
                        "refined_cuda_ms_per_returned_token_distribution": distribution(
                            row["formal_timing"][
                                "generation_cuda_ms_per_returned_token"
                            ]["refined"]
                            for row in valid
                        ),
                        "relative_change_distribution": distribution(
                            row["formal_timing"][
                                "generation_cuda_ms_per_returned_token"
                            ]["relative_gain"]
                            for row in valid
                        ),
                        "speedup_claimed": False,
                    },
                }
            )
    return rows


def family_summaries(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for family in FROZEN_REFINEMENT_FAMILIES:
        required = [
            row
            for row in task_rows
            if row["refined_family"] == family
            and (row["task"], row["nominal_prompt_length"])
            in EXPECTED_INDEPENDENT_STRATA
        ]
        if len(required) != 4:
            raise ValueError(f"{family}: missing frozen independent stratum")
        statuses = Counter(row["stratum_status"] for row in required)
        if statuses["incomplete_execution"]:
            classification = "incomplete_execution"
        elif statuses["invalid_due_to_correctness"]:
            classification = "invalid_due_to_correctness"
        elif statuses["insufficient_candidate_label_evidence"]:
            classification = "insufficient_candidate_label_evidence"
        elif statuses["joint_positive"] == 4:
            classification = "stable_cross_task_same_work_signal"
        elif statuses["joint_positive"]:
            classification = "mixed_cross_task_same_work_signal"
        else:
            classification = "no_cross_task_same_work_signal"
        replication = [
            row
            for row in task_rows
            if row["refined_family"] == family and row["task"] == KNOWN_TASK
        ]
        replication_statuses = Counter(row["stratum_status"] for row in replication)
        if replication_statuses["incomplete_execution"]:
            replication_class = "incomplete_execution"
        elif replication_statuses["invalid_due_to_correctness"]:
            replication_class = "invalid_due_to_correctness"
        elif replication_statuses["insufficient_candidate_label_evidence"]:
            replication_class = "insufficient_candidate_label_evidence"
        elif replication_statuses["joint_positive"] == len(replication):
            replication_class = "consistent_positive_same_task_replication"
        elif replication_statuses["joint_positive"]:
            replication_class = "mixed_same_task_replication"
        else:
            replication_class = "no_positive_same_task_replication"
        summaries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "refined_family": family,
                "independent_task_count": 2,
                "required_independent_stratum_count": 4,
                "required_strata": [
                    {
                        "task": row["task"],
                        "nominal_prompt_length": row["nominal_prompt_length"],
                        "status": row["stratum_status"],
                    }
                    for row in required
                ],
                "joint_positive_independent_stratum_count": statuses["joint_positive"],
                "cross_task_classification": classification,
                "govreport_contributes_to_cross_task_classification": False,
                "same_task_replication_classification": replication_class,
                "winner_selected": False,
            }
        )
    return summaries


def mechanism_summaries(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    keys = sorted(
        {
            (row["task"], row["nominal_prompt_length"], row["refined_family"])
            for row in pairs
        },
        key=lambda key: (FAMILY_ORDER[key[2]], TASK_ORDER[key[0]], key[1]),
    )
    for task, context, family in keys:
        cohort = [
            row
            for row in pairs
            if (row["task"], row["nominal_prompt_length"], row["refined_family"])
            == (task, context, family)
        ]
        valid = [
            row
            for row in cohort
            if row["correctness_valid"] and row["candidate_evidence_sufficient"]
        ]
        labelled = sum(
            int(row["candidate_quality"]["refined_labelled_reference_position_count"])
            for row in valid
        )

        def pooled(
            count_field: str,
            *,
            evidence: tuple[dict[str, Any], ...] = tuple(valid),
            denominator: int = labelled,
        ) -> tuple[int, float | None]:
            count = sum(
                int(row["candidate_quality"][count_field] or 0) for row in evidence
            )
            return count, _ratio(count, denominator)

        base_count, base_cov = pooled("refined_base_candidate_membership_count")
        conditional_count, conditional_cov = pooled(
            "refined_conditional_candidate_membership_count"
        )
        merged_count, merged_cov = pooled("refined_merged_candidate_membership_count")
        selected_count, selected_cov = pooled("refined_selected_tree_membership_count")
        base_cell_labelled, base_cell_selected, base_cell_cov = _pooled_candidate(
            valid, "base"
        )
        selected_gain = _absolute(base_cell_cov, selected_cov)
        recovery = sum(
            int(row["candidate_quality"]["refined_only_target_recovery_count"] or 0)
            for row in valid
        )
        if not valid:
            signal = "unavailable"
        elif recovery > 0:
            signal = "new_candidate_availability_signal"
        elif selected_gain is not None and selected_gain > 0:
            signal = "reranking_or_tree_reallocation_signal"
        else:
            signal = "no_observed_candidate_mechanism_gain"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "task": task,
                "nominal_prompt_length": context,
                "refined_family": family,
                "valid_sample_count": len(valid),
                "labelled_reference_position_count": labelled,
                "base_candidate_membership_count": base_count,
                "conditional_candidate_membership_count": conditional_count,
                "merged_candidate_membership_count": merged_count,
                "selected_tree_membership_count": selected_count,
                "base_candidate_coverage": base_cov,
                "conditional_candidate_coverage": conditional_cov,
                "merged_candidate_coverage": merged_cov,
                "selected_tree_coverage": selected_cov,
                "base_control_selected_membership_count": base_cell_selected,
                "base_control_labelled_position_count": base_cell_labelled,
                "base_control_selected_tree_coverage": base_cell_cov,
                "selected_tree_coverage_gain": selected_gain,
                "refined_only_target_recovery_count": recovery,
                "refined_only_target_recovery_rate": _ratio(recovery, labelled),
                "target_token_rank_improvement_distribution": distribution(
                    row["candidate_quality"]["target_token_rank_improvement"]["mean"]
                    if isinstance(
                        row["candidate_quality"]["target_token_rank_improvement"], dict
                    )
                    else None
                    for row in valid
                ),
                "correct_prefix_survival_depth_distribution": distribution(
                    row["candidate_quality"]["correct_prefix_survival_depth"]["mean"]
                    if isinstance(
                        row["candidate_quality"]["correct_prefix_survival_depth"], dict
                    )
                    else None
                    for row in valid
                ),
                "mechanism_signal": signal,
                "new_h2_evidence": signal == "new_candidate_availability_signal",
                "h1_refined_only_zero_rewritten": False,
                "causal_exclusivity_claimed": False,
            }
        )
    return rows


def cost_summaries(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    keys = sorted(
        {
            (row["task"], row["nominal_prompt_length"], row["refined_family"])
            for row in pairs
        },
        key=lambda key: (FAMILY_ORDER[key[2]], TASK_ORDER[key[0]], key[1]),
    )
    for task, context, family in keys:
        cohort = [
            row
            for row in pairs
            if (row["task"], row["nominal_prompt_length"], row["refined_family"])
            == (task, context, family)
        ]
        valid = [row for row in cohort if row["correctness_valid"]]

        def costs(
            field: str,
            *,
            evidence: tuple[dict[str, Any], ...] = tuple(valid),
        ) -> dict[str, Any]:
            return distribution(
                row["component_cost"].get(field) if row["component_cost"] else None
                for row in evidence
            )

        dependency = costs("conditional_remask_profile_total_cuda_ms")
        verifier = costs("extended_fixed_shape_path_profile_total_cuda_ms")
        comparisons = []
        for row in valid:
            component = row["component_cost"]
            dep = component.get("conditional_remask_profile_total_cuda_ms")
            ver = component.get("extended_fixed_shape_path_profile_total_cuda_ms")
            comparisons.append(None if dep is None or ver is None else ver > dep)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "task": task,
                "nominal_prompt_length": context,
                "refined_family": family,
                "valid_sample_count": len(valid),
                "dependency_modeling_cost": {
                    "conditional_remask_profile_total_cuda_ms": dependency,
                    "conditional_remask_share_of_profiled_round_time": costs(
                        "conditional_remask_share_of_profiled_round_time"
                    ),
                },
                "correctness_preserving_verifier_cost": {
                    "always_fixed_recheck_profile_total_cuda_ms": costs(
                        "fixed_shape_recheck_profile_total_cuda_ms"
                    ),
                    "round_entry_snapshot_profile_total_cuda_ms": costs(
                        "fixed_shape_round_entry_snapshot_profile_total_cuda_ms"
                    ),
                    "kv_replacement_profile_total_cuda_ms": costs(
                        "fixed_shape_kv_replacement_profile_total_cuda_ms"
                    ),
                    "extended_path_profile_total_cuda_ms": verifier,
                    "extended_path_share_of_profiled_round_time": costs(
                        "extended_fixed_shape_path_share_of_profiled_round_time"
                    ),
                },
                "combined_non_overlapping_additional_work": {
                    "profile_total_cuda_ms": costs(
                        "combined_conditional_and_fixed_profile_total_cuda_ms"
                    ),
                    "share_of_profiled_round_time": costs(
                        "combined_additional_work_share_of_profiled_round_time"
                    ),
                },
                "verifier_cost_greater_than_dependency_cost_count": sum(
                    value is True for value in comparisons
                ),
                "comparable_sample_count": sum(
                    value is not None for value in comparisons
                ),
                "h1_cost_hierarchy_generalizes_for_all_comparable_samples": bool(
                    comparisons and all(value is True for value in comparisons)
                ),
                "nested_timer_double_counting_avoided": True,
                "always_fixed_cost_attributed_to_conditional_remask": False,
            }
        )
    return rows


def build_aggregate(
    *,
    output_root: Path,
    bundle: dict[str, Any],
    plan: dict[str, Any],
    cells: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if cells is None:
        validate_no_extra_outputs(output_root, plan)
        cells = [load_cell(output_root, row) for row in plan["rows"]]
    if len(cells) != EXPECTED_PRIMARY_ROW_COUNT:
        raise ValueError("H2.1 cell summary must retain exactly 112 frozen rows")
    validate_cell_provenance(cells)
    pairs = pair_cells(plan, cells)
    task_context = task_context_summaries(pairs)
    families = family_summaries(task_context)
    mechanisms = mechanism_summaries(pairs)
    costs = cost_summaries(pairs)
    status_counts = Counter(row["cell_status"] for row in cells)
    execution_complete = status_counts["infrastructure_failed"] == 0
    heldout_mechanism_by_family = {
        family: dict(
            sorted(
                Counter(
                    row["mechanism_signal"]
                    for row in mechanisms
                    if row["refined_family"] == family
                    and row["task"] in INDEPENDENT_TASKS
                ).items()
            )
        )
        for family in FROZEN_REFINEMENT_FAMILIES
    }
    heldout_cost_hierarchy_by_family = {
        family: bool(
            relevant
            and all(
                row["h1_cost_hierarchy_generalizes_for_all_comparable_samples"]
                for row in relevant
            )
        )
        for family in FROZEN_REFINEMENT_FAMILIES
        if (
            relevant := [
                row
                for row in costs
                if row["refined_family"] == family and row["task"] in INDEPENDENT_TASKS
            ]
        )
    }
    primary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "artifact_type": "primary_conclusion",
        "status": "complete" if execution_complete else "incomplete_execution",
        "freeze_content_sha256": bundle["freeze_content_sha256"],
        "execution_complete": execution_complete,
        "independent_task_count": 2,
        "independent_tasks": list(INDEPENDENT_TASKS),
        "govreport_contributes_to_independent_task_count": False,
        "question": "Does a stable cross-task same-work dependency-refinement signal exist?",
        "family_cross_task_classifications": {
            row["refined_family"]: row["cross_task_classification"] for row in families
        },
        "stable_cross_task_same_work_signal_exists": any(
            row["cross_task_classification"] == "stable_cross_task_same_work_signal"
            for row in families
        )
        if execution_complete
        else None,
        "mechanism_and_cost_evidence_artifacts": [
            "phase2d2a-h21-mechanism-summary.json",
            "phase2d2a-h21-cost-summary.json",
        ],
        "heldout_mechanism_signal_distribution_by_family": (
            heldout_mechanism_by_family
        ),
        "heldout_verifier_cost_greater_than_dependency_cost_by_family": (
            heldout_cost_hierarchy_by_family
        ),
        "winner_selected": False,
        "final_phase2d2a_branch_entered": False,
        "proceed_to_lightweight_causal_adapter_issued": False,
        "proceed_with_conditional_remask_issued": False,
        "h22_can_override_primary_classification": False,
        "speedup_claimed": False,
        "latency_interpretation": "nominal_latency_change_only",
    }

    def artifact(
        artifact_type: str, rows: list[dict[str, Any]], **extra: Any
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "phase": PHASE,
            "artifact_type": artifact_type,
            "status": "complete" if execution_complete else "incomplete_execution",
            "freeze_content_sha256": bundle["freeze_content_sha256"],
            **extra,
            "rows": rows,
        }

    return {
        "cell_summary": artifact(
            "cell_summary",
            cells,
            expected_cell_count=EXPECTED_PRIMARY_ROW_COUNT,
            observed_cell_count=len(cells),
            valid_cell_count=status_counts["valid"],
            correctness_invalid_cell_count=status_counts["correctness_invalid"],
            infrastructure_failed_cell_count=status_counts["infrastructure_failed"],
            correctness_invalid_cells_excluded_from_benefit_aggregation=True,
        ),
        "paired": artifact(
            "sample_paired_comparison",
            pairs,
            expected_pair_count=EXPECTED_PAIR_COUNT,
            exact_same_sample_context_n16_pairing=True,
            unavailable_candidate_positions_excluded_from_denominators=True,
        ),
        "task_context": artifact(
            "task_context_summary",
            task_context,
            candidate_primary_aggregation="sum_membership_over_sum_labelled_positions",
            online_primary_aggregation="sum_accepted_over_sum_actual_verified_nodes",
            mean_of_sample_ratios_used_as_primary=False,
            independent_tasks=list(INDEPENDENT_TASKS),
            independent_task_count=2,
        ),
        "family": artifact(
            "family_summary",
            families,
            classification_rule="all_four_independent_task_context_strata_joint_positive",
            govreport_contributes_to_cross_task_stability=False,
            winner_selected=False,
        ),
        "mechanism": artifact(
            "mechanism_summary",
            mechanisms,
            h1_scientific_values_copied=False,
            h1_refined_only_zero_rewritten=False,
        ),
        "cost": artifact(
            "cost_summary",
            costs,
            dependency_modeling_and_correctness_preserving_verifier_cost_separated=True,
            nested_timer_double_counting_avoided=True,
            speedup_claimed=False,
        ),
        "primary": primary,
    }


def persist_aggregate(output_root: Path, aggregate: dict[str, Any]) -> None:
    mapping = {
        "phase2d2a-h21-cell-summary.json": aggregate["cell_summary"],
        "phase2d2a-h21-sample-paired-comparison.json": aggregate["paired"],
        "phase2d2a-h21-task-context-summary.json": aggregate["task_context"],
        "phase2d2a-h21-family-summary.json": aggregate["family"],
        "phase2d2a-h21-mechanism-summary.json": aggregate["mechanism"],
        "phase2d2a-h21-cost-summary.json": aggregate["cost"],
        "phase2d2a-h21-primary-conclusion.json": aggregate["primary"],
    }
    for name, payload in mapping.items():
        validate_h21_schema(payload)
        write_json(output_root / name, payload)
        if name != "phase2d2a-h21-primary-conclusion.json":
            write_csv(output_root / name.replace(".json", ".csv"), payload["rows"])
