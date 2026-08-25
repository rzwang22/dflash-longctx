#!/usr/bin/env python3
"""Plan or execute the frozen Phase 2D.2A-H2.1 primary N16 matrix."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_phase2d2_acceptance import (  # noqa: E402
    REQUIRED as PHASE2D2_REQUIRED_ARTIFACTS,
)
from scripts.phase2d2a_h21_common import (  # noqa: E402
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
    SENTINEL_GATE_NAME,
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
    load_cell,
    matrix_identity,
    read_json,
    read_jsonl,
    validate_external_status_schema,
    validate_h21_freeze,
    validate_h21_schema,
    validate_plan,
    write_json,
    write_jsonl,
    _validate_runtime_refinement_identity,
    _validated_frozen_refinement,
)

REQUIRED_CHILD_IMPORTS = ("jsonschema", "torch", "transformers")
INFRASTRUCTURE_OUTCOMES = {
    "blocked_dependency",
    "infrastructure_failed",
    "left_incomplete",
}


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


def _absolute_without_symlink_dereference(value: str | os.PathLike[str]) -> str:
    """Make an executable path absolute without replacing a venv symlink."""

    expanded = os.path.expanduser(os.fspath(value))
    if not os.path.dirname(expanded):
        discovered = shutil.which(expanded)
        if discovered is not None:
            expanded = discovered
    return os.path.abspath(expanded)


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
        # A virtualenv Python is commonly a symlink to a uv-managed base
        # interpreter.  Resolving it disables virtualenv site-packages in the
        # child.  Preserve that entry point while still making it absolute.
        "python_executable": _absolute_without_symlink_dereference(
            args.python_executable
        ),
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


def _probe_child_environment(
    child_python: str,
    required_imports: Iterable[str],
) -> dict[str, Any]:
    imports = tuple(required_imports)
    probe_source = """
import importlib
import json
import platform
import sys

required = json.loads(sys.argv[1])
statuses = {}
for name in required:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        statuses[name] = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "version": None,
        }
    else:
        statuses[name] = {
            "available": True,
            "error": None,
            "version": getattr(module, "__version__", None),
        }
payload = {
    "child_sys_executable": sys.executable,
    "child_python_version": platform.python_version(),
    "required_imports": statuses,
    "required_import_status": all(row["available"] for row in statuses.values()),
}
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["required_import_status"] else 1)
"""
    command = [child_python, "-c", probe_source, json.dumps(imports)]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except Exception as exc:  # noqa: BLE001 - infrastructure evidence is returned.
        return {
            "status": "failed",
            "command": command,
            "return_code": None,
            "child_sys_executable": None,
            "child_python_version": None,
            "required_imports": {
                name: {"available": False, "error": None, "version": None}
                for name in imports
            },
            "required_import_status": False,
            "probe_error": f"{type(exc).__name__}: {exc}",
            "stdout": "",
            "stderr": "",
        }
    stdout = completed.stdout.strip()
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        payload = {
            "child_sys_executable": None,
            "child_python_version": None,
            "required_imports": {
                name: {"available": False, "error": None, "version": None}
                for name in imports
            },
            "required_import_status": False,
            "probe_error": f"invalid child probe JSON: {type(exc).__name__}: {exc}",
        }
    else:
        payload["probe_error"] = None
    probe_passed = bool(
        completed.returncode == 0
        and payload.get("probe_error") is None
        and payload.get("required_import_status") is True
    )
    payload.update(
        {
            "status": "passed" if probe_passed else "failed",
            "command": command,
            "return_code": completed.returncode,
            "stdout": stdout,
            "stderr": completed.stderr.strip(),
        }
    )
    return payload


def _child_environment_preflight(paths: dict[str, Any]) -> dict[str, Any]:
    child = paths["python_executable"]
    child_is_symlink = Path(child).is_symlink()
    payload = _probe_child_environment(child, REQUIRED_CHILD_IMPORTS)
    payload.update(
        {
            "parent_sys_executable": sys.executable,
            "configured_child_executable": child,
            "configured_child_is_symlink": child_is_symlink,
            "configured_child_realpath": os.path.realpath(child),
            "configured_child_abspath": os.path.abspath(child),
            "symlink_preserved": (
                not child_is_symlink
                or os.path.abspath(child) != os.path.realpath(child)
            ),
        }
    )
    if not payload["symlink_preserved"]:
        payload["status"] = "failed"
        payload["probe_error"] = "configured child symlink was dereferenced"
    return payload


def _record_execution_preflight(
    output_root: Path,
    manifest: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    manifest["child_environment_preflight"] = payload
    manifest["child_environment_preflight_status"] = payload["status"]
    if payload["status"] != "passed":
        manifest["status"] = "infrastructure_failure"
        manifest["infrastructure_failure_reason"] = (
            payload.get("probe_error") or "required child import unavailable"
        )
    validate_h21_schema(manifest)
    write_json(output_root / EXECUTION_MANIFEST_NAME, manifest)


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
    canonical: bool = False,
) -> None:
    rows = _index_rows(output_root, plan)
    external_complete = sum(row["execution_end_status"] == "completed" for row in rows)
    external_infrastructure_failed = sum(
        row["execution_end_status"]
        in {"incomplete", "blocked_dependency", "infrastructure_failed"}
        for row in rows
    )
    not_started = len(rows) - external_complete - external_infrastructure_failed
    canonical_infrastructure_failed = 0
    if canonical:
        plan_by_id = {row["execution_id"]: row for row in plan["rows"]}
        scientific_invalid = 0
        for index_row in rows:
            if index_row["execution_end_status"] != "completed":
                index_row["canonical_cell_status"] = None
                index_row["canonical_cell_failure_reason"] = None
                continue
            cell = load_cell(output_root, plan_by_id[index_row["execution_id"]])
            index_row["canonical_cell_status"] = cell["cell_status"]
            index_row["canonical_cell_failure_reason"] = cell.get(
                "infrastructure_failure_reason"
            )
            if cell["cell_status"] == "infrastructure_failed":
                canonical_infrastructure_failed += 1
            elif cell["cell_status"] == "correctness_invalid":
                scientific_invalid += 1
    else:
        scientific_invalid = sum(
            row["execution_end_status"] == "completed"
            and row["external_phase2d2_checker_return_code"] not in (None, 0)
            for row in rows
        )
    complete = external_complete - canonical_infrastructure_failed
    infrastructure_failed = (
        external_infrastructure_failed + canonical_infrastructure_failed
    )
    write_jsonl(output_root / SUBRUN_INDEX_NAME, rows)
    scientific_subprocess_launched = bool(
        manifest.get("scientific_subprocess_launched") or attempted
    )
    model_forward_evidence = bool(
        manifest.get("model_forward_executed")
        or external_complete
        or any(
            (output_root / row["result_relative_dir"] / "samples.jsonl").is_file()
            and (output_root / row["result_relative_dir"] / "samples.jsonl")
            .stat()
            .st_size
            > 0
            for row in plan["rows"]
        )
    )
    manifest.update(
        {
            "status": (
                "execution_complete"
                if complete == len(rows)
                else "partial_execution"
                if scientific_subprocess_launched
                else "planned_not_executed"
            ),
            "scientific_subprocess_launched": scientific_subprocess_launched,
            "model_forward_executed": model_forward_evidence,
            "completed_cell_count": complete,
            "incomplete_cell_count": infrastructure_failed + not_started,
            "infrastructure_failed_cell_count": infrastructure_failed,
            "not_started_cell_count": not_started,
            "scientific_correctness_invalid_cell_count": scientific_invalid,
            "canonical_cell_classification_applied": canonical,
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
    result_dir: Path,
    row: dict[str, Any],
    frozen_refinement_path: str,
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
        frozen_refinement = (
            None
            if row["configuration_family"] == BASE_FAMILY
            else _validated_frozen_refinement(Path(frozen_refinement_path))
        )
        _validate_runtime_refinement_identity(
            result_dir,
            row,
            phase,
            sample,
            frozen_refinement,
        )
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
    try:
        runner = subprocess.run(command, cwd=REPO_ROOT, check=False)
    except Exception as exc:  # noqa: BLE001 - persist infrastructure evidence.
        runner = None
        runner_error = f"{type(exc).__name__}: {exc}"
    else:
        runner_error = None
    checker_rc = None
    checker_error = None
    if runner is not None and result_dir.exists():
        try:
            checker = subprocess.run(
                (
                    paths["python_executable"],
                    str(REPO_ROOT / "scripts" / "check_phase2d2_acceptance.py"),
                    str(result_dir),
                ),
                cwd=REPO_ROOT,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001 - persist infrastructure evidence.
            checker_error = f"{type(exc).__name__}: {exc}"
        else:
            checker_rc = checker.returncode
    result_dir.mkdir(parents=True, exist_ok=True)
    complete_artifacts = all(
        (result_dir / name).is_file() for name in PHASE2D2_REQUIRED_ARTIFACTS
    )
    identity_valid, identity_error = _validate_produced_identity(
        result_dir,
        row,
        paths["frozen_refinement_config"],
    )
    end_status = (
        "completed"
        if complete_artifacts and identity_valid
        else "infrastructure_failed"
    )
    missing_artifacts = [
        name
        for name in PHASE2D2_REQUIRED_ARTIFACTS
        if not (result_dir / name).is_file()
    ]
    infrastructure_failure_reason = None
    if end_status == "infrastructure_failed":
        infrastructure_failure_reason = "; ".join(
            reason
            for reason in (
                runner_error,
                checker_error,
                f"missing artifacts: {missing_artifacts}"
                if missing_artifacts
                else None,
                identity_error,
            )
            if reason
        )
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
        "runner_return_code": runner.returncode if runner is not None else None,
        "external_phase2d2_checker_return_code": checker_rc,
        "result_identity_valid": identity_valid,
        "result_identity_error": identity_error,
        "infrastructure_failure_reason": infrastructure_failure_reason,
        "command": command,
    }
    validate_external_status_schema(status)
    write_json(result_dir / EXTERNAL_STATUS_NAME, status)
    return end_status


def _run_rows_with_circuit_breaker(
    rows: Iterable[dict[str, Any]],
    execute: Callable[[dict[str, Any]], str],
    after_each: Callable[[], None] | None = None,
) -> dict[str, Any]:
    outcomes = []
    for row in rows:
        outcome = execute(row)
        outcomes.append(
            {"execution_id": row["execution_id"], "execution_outcome": outcome}
        )
        if after_each is not None:
            after_each()
        if outcome in INFRASTRUCTURE_OUTCOMES:
            return {
                "triggered": True,
                "failed_execution_id": row["execution_id"],
                "failure_outcome": outcome,
                "scheduled_row_count": len(outcomes),
                "outcomes": outcomes,
            }
    return {
        "triggered": False,
        "failed_execution_id": None,
        "failure_outcome": None,
        "scheduled_row_count": len(outcomes),
        "outcomes": outcomes,
    }


def _sentinel_cell_gate(
    row: dict[str, Any],
    status: dict[str, Any] | None,
    *,
    complete_required_artifacts: bool,
    cell_status: str,
    infrastructure_failure_reason: str | None = None,
) -> dict[str, Any]:
    reasons = []
    if status is None:
        reasons.append("missing_external_status")
    else:
        requirements = {
            "execution_end_status_completed": (
                status.get("execution_end_status") == "completed"
            ),
            "runner_return_code_zero": status.get("runner_return_code") == 0,
            "external_checker_return_code_zero": (
                status.get("external_phase2d2_checker_return_code") == 0
            ),
            "frozen_identity_valid": status.get("result_identity_valid") is True,
        }
        reasons.extend(name for name, passed in requirements.items() if not passed)
    if not complete_required_artifacts:
        reasons.append("incomplete_required_artifacts")
    hard_correctness_accepted = cell_status == "valid"
    if cell_status == "infrastructure_failed":
        reasons.append(
            "cell_status_infrastructure_failed:"
            f"{infrastructure_failure_reason or 'unspecified infrastructure failure'}"
        )
    elif cell_status == "correctness_invalid":
        reasons.append("hard_correctness_not_accepted")
    elif cell_status != "valid":
        reasons.append(f"unknown_canonical_cell_status:{cell_status}")
    return {
        "execution_id": row["execution_id"],
        "execution_end_status": status.get("execution_end_status") if status else None,
        "runner_return_code": status.get("runner_return_code") if status else None,
        "external_phase2d2_checker_return_code": (
            status.get("external_phase2d2_checker_return_code") if status else None
        ),
        "frozen_identity_valid": (
            status.get("result_identity_valid") is True if status else False
        ),
        "complete_required_artifacts": complete_required_artifacts,
        "canonical_cell_status": cell_status,
        "infrastructure_failure_reason": infrastructure_failure_reason,
        "hard_correctness_accepted": hard_correctness_accepted,
        "passed": not reasons,
        "failure_reasons": reasons,
    }


def _evaluate_sentinel_gate(
    output_root: Path,
    rows: list[dict[str, Any]],
    freeze_content_sha256: str,
) -> dict[str, Any]:
    cell_gates = []
    for row in rows:
        try:
            status = _status_for_row(output_root, row)
        except Exception as exc:  # noqa: BLE001 - preserve gate failure evidence.
            status = None
            status_error = f"{type(exc).__name__}: {exc}"
        else:
            status_error = None
        result_dir = output_root / row["result_relative_dir"]
        complete = all(
            (result_dir / name).is_file() for name in PHASE2D2_REQUIRED_ARTIFACTS
        )
        cell = load_cell(output_root, row)
        gate = _sentinel_cell_gate(
            row,
            status,
            complete_required_artifacts=complete,
            cell_status=cell["cell_status"],
            infrastructure_failure_reason=cell.get("infrastructure_failure_reason"),
        )
        if status_error is not None:
            gate["failure_reasons"].append(f"invalid_external_status:{status_error}")
            gate["passed"] = False
        cell_gates.append(gate)
    failure_reasons = [
        f"{gate['execution_id']}:{reason}"
        for gate in cell_gates
        for reason in gate["failure_reasons"]
    ]
    payload = {
        "schema_version": "1.0",
        "phase": "2D.2A-H2.1",
        "artifact_type": "sentinel_gate",
        "status": "complete",
        "freeze_content_sha256": freeze_content_sha256,
        "passed": len(cell_gates) == 2 and not failure_reasons,
        "requested_execution_ids": [row["execution_id"] for row in rows],
        "cell_gates": cell_gates,
        "failure_reasons": failure_reasons,
    }
    validate_h21_schema(payload)
    write_json(output_root / SENTINEL_GATE_NAME, payload)
    return payload


def _sentinel_gate_return_code(payload: dict[str, Any]) -> int:
    return 0 if payload.get("passed") is True else 1


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
    try:
        _validate_runtime_paths(runtime_paths)
    except Exception as exc:  # noqa: BLE001 - persist infrastructure evidence.
        preflight = {
            "status": "failed",
            "parent_sys_executable": sys.executable,
            "configured_child_executable": runtime_paths["python_executable"],
            "configured_child_is_symlink": Path(
                runtime_paths["python_executable"]
            ).is_symlink(),
            "configured_child_realpath": os.path.realpath(
                runtime_paths["python_executable"]
            ),
            "configured_child_abspath": os.path.abspath(
                runtime_paths["python_executable"]
            ),
            "symlink_preserved": (
                os.path.abspath(runtime_paths["python_executable"])
                != os.path.realpath(runtime_paths["python_executable"])
                if Path(runtime_paths["python_executable"]).is_symlink()
                else True
            ),
            "child_sys_executable": None,
            "child_python_version": None,
            "required_imports": {},
            "required_import_status": False,
            "return_code": None,
            "probe_error": f"runtime path validation failed: {type(exc).__name__}: {exc}",
            "command": [],
            "stdout": "",
            "stderr": "",
        }
        _record_execution_preflight(output_root, manifest, preflight)
        print(json.dumps(preflight, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    preflight = _child_environment_preflight(runtime_paths)
    _record_execution_preflight(output_root, manifest, preflight)
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if preflight["status"] != "passed":
        print("H2.1 child-environment preflight failed; no scientific subrun launched")
        return 1
    rows = (
        _sentinel_rows(plan)
        if args.sentinel_pair_only
        else sorted(plan["rows"], key=execution_sort_key)
    )
    circuit = _run_rows_with_circuit_breaker(
        rows,
        lambda row: _execute_row(output_root, row, runtime_paths),
        lambda: _refresh_index_and_manifest(
            output_root, plan, manifest, attempted=True
        ),
    )
    _refresh_index_and_manifest(
        output_root,
        plan,
        manifest,
        attempted=True,
        canonical=True,
    )
    manifest["infrastructure_circuit_breaker"] = circuit
    if circuit["triggered"]:
        manifest["status"] = "infrastructure_failure"
        manifest["infrastructure_failure_reason"] = (
            "scientific scheduling stopped after "
            f"{circuit['failed_execution_id']}:{circuit['failure_outcome']}"
        )
    sentinel_gate = None
    if args.sentinel_pair_only:
        sentinel_gate = _evaluate_sentinel_gate(
            output_root, rows, bundle["freeze_content_sha256"]
        )
        manifest["sentinel_gate"] = sentinel_gate
    validate_h21_schema(manifest)
    write_json(output_root / EXECUTION_MANIFEST_NAME, manifest)
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
                "infrastructure_failed_cell_count": manifest[
                    "infrastructure_failed_cell_count"
                ],
                "scientific_correctness_invalid_cell_count": manifest[
                    "scientific_correctness_invalid_cell_count"
                ],
                "not_started_cell_count": manifest["not_started_cell_count"],
                "infrastructure_circuit_breaker_triggered": circuit["triggered"],
                "sentinel_gate_passed": (
                    sentinel_gate["passed"] if sentinel_gate is not None else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.sentinel_pair_only:
        return _sentinel_gate_return_code(sentinel_gate or {})
    return 0 if manifest["incomplete_cell_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
