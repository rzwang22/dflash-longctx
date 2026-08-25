#!/usr/bin/env python3
"""Plan or execute the frozen Phase 2D.2A-H2.1 primary N16 matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_phase2d2_acceptance import REQUIRED as PHASE2D2_REQUIRED_ARTIFACTS
from scripts.phase2d2a_h21_common import (
    BASE_FAMILY,
    BATCH_SIZE,
    BLOCK_SIZE,
    DEVICE,
    DRAFT_MODEL_ID,
    DRAFT_REVISION,
    DTYPE,
    EXECUTION_MANIFEST_NAME,
    EXECUTION_PLAN_NAME,
    EXTERNAL_STATUS_NAME,
    FROZEN_REFINEMENT_FAMILIES,
    MAX_NEW_TOKENS,
    REPO_ROOT,
    REQUIRED_FREEZE_SHA256,
    SELECTOR_MODE,
    SUBRUN_INDEX_NAME,
    TARGET_MODEL_ID,
    TARGET_REVISION,
    TEMPERATURE,
    TIMING_REPEATS,
    VERIFICATION_MODE,
    WARMUP_RUNS,
    build_execution_manifest,
    build_execution_plan,
    ensure_immutable_freeze,
    execution_sort_key,
    git_value,
    matrix_identity,
    read_json,
    read_jsonl,
    validate_external_status_schema,
    validate_h21_freeze,
    validate_h21_schema,
    validate_plan,
    write_json,
    write_jsonl,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", required=True, type=Path)
    parser.add_argument("--freeze-content-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--execute-frozen-primary", action="store_true")
    parser.add_argument(
        "--sentinel-pair-only",
        action="store_true",
        help="Execute one frozen Qasper 1K base/refinement pair only.",
    )
    parser.add_argument("--target-model-path", type=Path)
    parser.add_argument("--materialized-draft-path", type=Path)
    parser.add_argument("--longbench-root", type=Path)
    parser.add_argument("--frozen-selector-config", type=Path)
    parser.add_argument("--frozen-refinement-config", type=Path)
    parser.add_argument("--python-executable", default=sys.executable)
    return parser


def _runtime_paths(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target_model_path": str(args.target_model_path.expanduser().resolve())
        if args.target_model_path
        else None,
        "materialized_draft_path": str(
            args.materialized_draft_path.expanduser().resolve()
        )
        if args.materialized_draft_path
        else None,
        "longbench_root": str(args.longbench_root.expanduser().resolve())
        if args.longbench_root
        else None,
        "frozen_selector_config": str(
            args.frozen_selector_config.expanduser().resolve()
        )
        if args.frozen_selector_config
        else None,
        "frozen_refinement_config": str(
            args.frozen_refinement_config.expanduser().resolve()
        )
        if args.frozen_refinement_config
        else None,
        "python_executable": str(Path(args.python_executable).expanduser().resolve()),
    }


def _validate_runtime_paths(paths: dict[str, Any]) -> None:
    missing_args = [field for field, value in paths.items() if value is None]
    if missing_args:
        raise ValueError(f"execution requires runtime paths: {missing_args}")
    for field in ("target_model_path", "materialized_draft_path", "longbench_root"):
        if not Path(paths[field]).is_dir():
            raise ValueError(
                f"execution path is not a directory: {field}={paths[field]}"
            )
    for field in ("frozen_selector_config", "frozen_refinement_config"):
        if not Path(paths[field]).is_file():
            raise ValueError(f"execution path is not a file: {field}={paths[field]}")
    if not Path(paths["python_executable"]).is_file():
        raise ValueError("Python executable does not exist")


def _prepare_root(
    output_root: Path,
    *,
    bundle: dict[str, Any],
    plan: dict[str, Any],
    runtime_paths: dict[str, Any],
    execution_authorized: bool,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / EXECUTION_PLAN_NAME
    manifest_path = output_root / EXECUTION_MANIFEST_NAME
    if plan_path.is_file():
        existing_plan = read_json(plan_path)
        validate_plan(existing_plan, bundle)
        if existing_plan != plan:
            raise ValueError("existing H2.1 execution plan changed")
    else:
        unexpected = [path for path in output_root.iterdir() if path.name != "runs"]
        runs = output_root / "runs"
        if runs.exists() and any(runs.iterdir()):
            unexpected.append(runs)
        if unexpected:
            raise ValueError(
                f"refusing to initialize non-empty H2.1 output root: {unexpected}"
            )
        validate_h21_schema(plan)
        write_json(plan_path, plan)
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        validate_h21_schema(manifest)
        ensure_immutable_freeze(manifest, bundle)
        if manifest.get("repository_commit") != git_value("rev-parse", "HEAD"):
            raise ValueError(
                "repository commit changed after H2.1 planning; use a new output root"
            )
        old_paths = manifest.get("runtime_paths", {})
        for field, value in runtime_paths.items():
            if old_paths.get(field) not in (None, value) and value is not None:
                raise ValueError(f"runtime path changed on resume: {field}")
            if value is not None:
                old_paths[field] = value
        manifest["runtime_paths"] = old_paths
        manifest["h2_primary_execution_authorized_by_cli"] = bool(
            manifest.get("h2_primary_execution_authorized_by_cli")
            or execution_authorized
        )
    else:
        manifest = build_execution_manifest(
            bundle,
            plan=plan,
            runtime_paths=runtime_paths,
            execution_authorized=execution_authorized,
        )
    validate_h21_schema(manifest)
    write_json(manifest_path, manifest)
    return manifest


def _status_for_row(output_root: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    path = output_root / row["result_relative_dir"] / EXTERNAL_STATUS_NAME
    if not path.is_file():
        return None
    status = read_json(path)
    validate_external_status_schema(status)
    if status["freeze_content_sha256"] != row["freeze_content_sha256"]:
        raise ValueError(f"{row['execution_id']}: conflicting freeze status")
    if status["frozen_matrix_row_identity"] != matrix_identity(row):
        raise ValueError(f"{row['execution_id']}: conflicting matrix identity")
    return status


def _index_rows(output_root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in plan["rows"]:
        status = _status_for_row(output_root, row)
        rows.append(
            {
                "schema_version": "1.0",
                **matrix_identity(row),
                "result_relative_dir": row["result_relative_dir"],
                "execution_end_status": (
                    status["execution_end_status"] if status else "not_started"
                ),
                "runner_return_code": status["runner_return_code"] if status else None,
                "external_phase2d2_checker_return_code": (
                    status["external_phase2d2_checker_return_code"] if status else None
                ),
            }
        )
    return rows


def _refresh_index_and_manifest(
    output_root: Path,
    plan: dict[str, Any],
    manifest: dict[str, Any],
    *,
    attempted: bool,
) -> None:
    rows = _index_rows(output_root, plan)
    write_jsonl(output_root / SUBRUN_INDEX_NAME, rows)
    complete = sum(row["execution_end_status"] == "completed" for row in rows)
    incomplete = sum(
        row["execution_end_status"] in {"incomplete", "blocked_dependency"}
        for row in rows
    )
    not_started = len(rows) - complete - incomplete
    scientific_invalid = sum(
        row["execution_end_status"] == "completed"
        and row["external_phase2d2_checker_return_code"] not in (None, 0)
        for row in rows
    )
    execution_attempted = bool(manifest.get("model_forward_executed") or attempted)
    manifest.update(
        {
            "status": (
                "execution_complete"
                if complete == len(rows)
                else "partial_execution"
                if execution_attempted
                else "planned_not_executed"
            ),
            "model_forward_executed": execution_attempted,
            "completed_cell_count": complete,
            "incomplete_cell_count": incomplete + not_started,
            "scientific_correctness_invalid_cell_count": scientific_invalid,
        }
    )
    validate_h21_schema(manifest)
    write_json(output_root / EXECUTION_MANIFEST_NAME, manifest)


def _runner_command(
    row: dict[str, Any],
    *,
    output_root: Path,
    paths: dict[str, Any],
) -> list[str]:
    result_dir = output_root / row["result_relative_dir"]
    command = [
        paths["python_executable"],
        str(REPO_ROOT / "experiments" / "phase2d2_robust_conditional_refinement.py"),
        "--target-model",
        TARGET_MODEL_ID,
        "--target-revision",
        TARGET_REVISION,
        "--target-model-path",
        paths["target_model_path"],
        "--draft-model",
        DRAFT_MODEL_ID,
        "--draft-revision",
        DRAFT_REVISION,
        "--materialized-draft-path",
        paths["materialized_draft_path"],
        "--longbench-root",
        paths["longbench_root"],
        "--longbench-task",
        row["task"],
        "--sample-policy",
        "matched",
        "--sample-id",
        row["sample_id"],
        "--context-lengths",
        str(row["nominal_prompt_length"]),
        "--pilot-samples",
        "1",
        "--max-new-tokens",
        str(MAX_NEW_TOKENS),
        "--warmup-runs",
        str(WARMUP_RUNS),
        "--timing-repeats",
        str(TIMING_REPEATS),
        "--batch-size",
        str(BATCH_SIZE),
        "--temperature",
        str(TEMPERATURE),
        "--block-size",
        str(BLOCK_SIZE),
        "--dtype",
        DTYPE,
        "--device",
        DEVICE,
        "--frozen-selector-config",
        paths["frozen_selector_config"],
        "--online-selector-mode",
        SELECTOR_MODE,
        "--node-budget",
        "16",
        "--verification-mode",
        VERIFICATION_MODE,
        "--evidence-role",
        "held_out_evaluation",
        "--output-dir",
        str(result_dir),
    ]
    family = row["configuration_family"]
    if family == BASE_FAMILY:
        command.extend(("--candidate-merge-mode", BASE_FAMILY))
    else:
        base_row = {
            **row,
            "configuration_family": BASE_FAMILY,
        }
        command.extend(
            (
                "--frozen-refinement-config",
                paths["frozen_refinement_config"],
                "--frozen-refinement-config-id",
                family,
                "--base-result-dir",
                str(
                    output_root
                    / (
                        Path("runs")
                        / row["task"]
                        / row["sample_id"]
                        / str(row["nominal_prompt_length"])
                        / base_row["configuration_family"]
                    )
                ),
            )
        )
    return command


def _validate_produced_identity(
    result_dir: Path, row: dict[str, Any]
) -> tuple[bool, str | None]:
    try:
        samples = read_jsonl(result_dir / "samples.jsonl")
        if len(samples) != 1:
            raise ValueError("expected one sample row")
        sample = samples[0]
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
                raise ValueError(f"frozen result identity mismatch: {field}")
        manifest = read_json(result_dir / "manifest.json")
        phase = manifest["phase2d2"]
        if (
            phase["node_budget"] != 16
            or phase["verification_mode"] != VERIFICATION_MODE
        ):
            raise ValueError("frozen runtime identity mismatch")
        if (
            phase["frozen_refinement_configuration_id"]
            != row["frozen_refinement_configuration_id"]
        ):
            raise ValueError("frozen refinement family mismatch")
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _base_is_available(output_root: Path, row: dict[str, Any]) -> bool:
    base_dir = (
        output_root
        / "runs"
        / row["task"]
        / row["sample_id"]
        / str(row["nominal_prompt_length"])
        / BASE_FAMILY
    )
    status_path = base_dir / EXTERNAL_STATUS_NAME
    if not status_path.is_file():
        return False
    status = read_json(status_path)
    return status.get("execution_end_status") == "completed" and all(
        (base_dir / name).is_file() for name in PHASE2D2_REQUIRED_ARTIFACTS
    )


def _execute_row(
    output_root: Path,
    row: dict[str, Any],
    paths: dict[str, Any],
) -> str:
    result_dir = output_root / row["result_relative_dir"]
    existing = _status_for_row(output_root, row)
    if existing is not None:
        if existing["execution_end_status"] == "completed":
            print(f"SKIP completed {row['execution_id']}")
            return "skipped_completed"
        print(f"LEAVE incomplete {row['execution_id']}")
        return "left_incomplete"
    if result_dir.exists():
        print(f"LEAVE incomplete directory without status: {result_dir}")
        return "left_incomplete"
    if row["configuration_family"] != BASE_FAMILY and not _base_is_available(
        output_root, row
    ):
        result_dir.mkdir(parents=True, exist_ok=True)
        status = {
            "schema_version": "1.0",
            "phase": "2D.2A-H2.1",
            "freeze_content_sha256": row["freeze_content_sha256"],
            "execution_id": row["execution_id"],
            "frozen_matrix_row_identity": matrix_identity(row),
            "repository_commit": git_value("rev-parse", "HEAD"),
            "execution_start_utc": utc_now(),
            "execution_end_utc": utc_now(),
            "execution_start_status": "blocked_dependency",
            "execution_end_status": "blocked_dependency",
            "runner_return_code": None,
            "external_phase2d2_checker_return_code": None,
            "result_identity_valid": False,
            "result_identity_error": "matching frozen base cell is not complete",
            "command": [],
        }
        validate_external_status_schema(status)
        write_json(result_dir / EXTERNAL_STATUS_NAME, status)
        return "blocked_dependency"
    command = _runner_command(row, output_root=output_root, paths=paths)
    started = utc_now()
    print(f"RUN {row['execution_id']} -> {result_dir}")
    runner = subprocess.run(command, cwd=REPO_ROOT, check=False)
    checker_rc = None
    if result_dir.exists():
        checker = subprocess.run(
            (
                paths["python_executable"],
                str(REPO_ROOT / "scripts" / "check_phase2d2_acceptance.py"),
                str(result_dir),
            ),
            cwd=REPO_ROOT,
            check=False,
        )
        checker_rc = checker.returncode
    result_dir.mkdir(parents=True, exist_ok=True)
    complete_artifacts = all(
        (result_dir / name).is_file() for name in PHASE2D2_REQUIRED_ARTIFACTS
    )
    identity_valid, identity_error = _validate_produced_identity(result_dir, row)
    end_status = "completed" if complete_artifacts and identity_valid else "incomplete"
    status = {
        "schema_version": "1.0",
        "phase": "2D.2A-H2.1",
        "freeze_content_sha256": row["freeze_content_sha256"],
        "execution_id": row["execution_id"],
        "frozen_matrix_row_identity": matrix_identity(row),
        "repository_commit": git_value("rev-parse", "HEAD"),
        "execution_start_utc": started,
        "execution_end_utc": utc_now(),
        "execution_start_status": "started",
        "execution_end_status": end_status,
        "runner_return_code": runner.returncode,
        "external_phase2d2_checker_return_code": checker_rc,
        "result_identity_valid": identity_valid,
        "result_identity_error": identity_error,
        "command": command,
    }
    validate_external_status_schema(status)
    write_json(result_dir / EXTERNAL_STATUS_NAME, status)
    return end_status


def _sentinel_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    qasper = sorted(
        (
            row
            for row in plan["rows"]
            if row["task"] == "qasper" and row["nominal_prompt_length"] == 1024
        ),
        key=execution_sort_key,
    )
    sample_id = qasper[0]["sample_id"]
    families = {BASE_FAMILY, FROZEN_REFINEMENT_FAMILIES[0]}
    return [
        row
        for row in qasper
        if row["sample_id"] == sample_id and row["configuration_family"] in families
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sentinel_pair_only and not args.execute_frozen_primary:
        raise ValueError("--sentinel-pair-only requires --execute-frozen-primary")
    # This complete freeze gate intentionally precedes all runtime-path checks and
    # every subprocess/model-loading path.
    bundle = validate_h21_freeze(
        args.freeze_dir,
        args.freeze_content_sha256,
        required_freeze_sha256=REQUIRED_FREEZE_SHA256,
    )
    plan = build_execution_plan(bundle)
    runtime_paths = _runtime_paths(args)
    output_root = args.output_root.expanduser().resolve()
    manifest = _prepare_root(
        output_root,
        bundle=bundle,
        plan=plan,
        runtime_paths=runtime_paths,
        execution_authorized=args.execute_frozen_primary,
    )
    _refresh_index_and_manifest(output_root, plan, manifest, attempted=False)
    print(json.dumps(plan, indent=2, sort_keys=True))
    print("task_row_counts:", plan["task_row_counts"])
    print("task_context_row_counts:", plan["task_context_row_counts"])
    print("family_row_counts:", plan["family_row_counts"])
    print("expected_model_load_groups:", plan["expected_target_draft_model_load_count"])
    if not args.execute_frozen_primary:
        print("PLAN-ONLY: zero model forwards executed")
        return 0
    _validate_runtime_paths(runtime_paths)
    rows = (
        _sentinel_rows(plan)
        if args.sentinel_pair_only
        else sorted(plan["rows"], key=execution_sort_key)
    )
    for row in rows:
        _execute_row(output_root, row, runtime_paths)
        _refresh_index_and_manifest(output_root, plan, manifest, attempted=True)
    _refresh_index_and_manifest(output_root, plan, manifest, attempted=True)
    print(
        json.dumps(
            {
                "execution_scope": "sentinel_pair"
                if args.sentinel_pair_only
                else "full_primary_matrix",
                "requested_row_count": len(rows),
                "output_root": str(output_root),
                "status": manifest["status"],
                "completed_cell_count": manifest["completed_cell_count"],
                "incomplete_cell_count": manifest["incomplete_cell_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if manifest["incomplete_cell_count"] == 0 or args.sentinel_pair_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
