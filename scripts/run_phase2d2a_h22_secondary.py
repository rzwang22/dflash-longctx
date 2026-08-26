#!/usr/bin/env python3
"""Plan or execute the frozen secondary Phase 2D.2A-H2.2 N32 matrix."""

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

from scripts.check_phase2d2_acceptance import (
    REQUIRED as PHASE2D2_REQUIRED_ARTIFACTS,
)
from scripts.phase2d2a_h21_common import (
    BASE_FAMILY,
    BATCH_SIZE,
    BLOCK_SIZE,
    DEVICE,
    DRAFT_MODEL_ID,
    DRAFT_REVISION,
    DTYPE,
    FROZEN_REFINEMENT_FAMILIES,
    MAX_NEW_TOKENS,
    SELECTOR_MODE,
    TARGET_MODEL_ID,
    TARGET_REVISION,
    TEMPERATURE,
    TIMING_REPEATS,
    VERIFICATION_MODE,
    WARMUP_RUNS,
    _validated_frozen_refinement,
    execution_sort_key,
    git_value,
    matrix_identity,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from scripts.phase2d2a_h22_common import (
    ANALYSIS_CONTRACT_NAME,
    EXECUTION_MANIFEST_NAME,
    EXECUTION_PLAN_NAME,
    EXPECTED_ROW_COUNT,
    EXTERNAL_STATUS_NAME,
    PHASE,
    SCHEMA_VERSION,
    SENTINEL_GATE_NAME,
    SUBRUN_INDEX_NAME,
    _validate_result_identity,
    build_analysis_contract,
    build_execution_manifest,
    build_execution_plan,
    ensure_immutable_sources,
    load_h22_cell,
    source_bindings,
    validate_h22_external_status,
    validate_h22_freeze,
    validate_h22_schema,
    validate_plan,
)
from scripts.run_phase2d2a_h21_primary import (
    _child_environment_preflight,
    _run_rows_with_circuit_breaker,
    _runtime_paths,
    _sentinel_cell_gate,
    _sentinel_gate_return_code,
    _validate_runtime_paths,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", required=True, type=Path)
    parser.add_argument("--freeze-content-sha256", required=True)
    parser.add_argument("--h21-output-root", required=True, type=Path)
    parser.add_argument("--h211-output-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--execute-frozen-secondary", action="store_true")
    parser.add_argument("--sentinel-pair-only", action="store_true")
    parser.add_argument("--target-model-path", type=Path)
    parser.add_argument("--materialized-draft-path", type=Path)
    parser.add_argument("--longbench-root", type=Path)
    parser.add_argument("--frozen-selector-config", type=Path)
    parser.add_argument("--frozen-refinement-config", type=Path)
    parser.add_argument("--python-executable", default=sys.executable)
    return parser


def _prepare_root(
    output_root: Path,
    *,
    bundle: dict[str, Any],
    bindings: dict[str, Any],
    plan: dict[str, Any],
    contract: dict[str, Any],
    runtime_paths: dict[str, Any],
    execution_authorized: bool,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / EXECUTION_PLAN_NAME
    contract_path = output_root / ANALYSIS_CONTRACT_NAME
    manifest_path = output_root / EXECUTION_MANIFEST_NAME
    if plan_path.is_file():
        existing_plan = read_json(plan_path)
        validate_plan(existing_plan, bundle, bindings)
        if existing_plan != plan:
            raise ValueError("existing H2.2 execution plan changed")
        if read_json(contract_path) != contract:
            raise ValueError("existing H2.2 analysis contract changed")
    else:
        unexpected = [path for path in output_root.iterdir() if path.name != "runs"]
        runs = output_root / "runs"
        if runs.exists() and any(runs.iterdir()):
            unexpected.append(runs)
        if unexpected:
            raise ValueError(f"refusing non-empty H2.2 output root: {unexpected}")
        validate_h22_schema(plan)
        validate_h22_schema(contract)
        write_json(plan_path, plan)
        write_json(contract_path, contract)
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        validate_h22_schema(manifest)
        ensure_immutable_sources(manifest, bundle, bindings)
        if manifest.get("repository_commit") != git_value("rev-parse", "HEAD"):
            raise ValueError("repository commit changed; use a new H2.2 output root")
        old_paths = manifest["runtime_paths"]
        for field, value in runtime_paths.items():
            if old_paths.get(field) not in (None, value) and value is not None:
                raise ValueError(f"runtime path changed on resume: {field}")
            if value is not None:
                old_paths[field] = value
        manifest["execute_frozen_secondary_authorized_by_cli"] = bool(
            manifest["execute_frozen_secondary_authorized_by_cli"]
            or execution_authorized
        )
    else:
        manifest = build_execution_manifest(
            bundle,
            bindings,
            plan,
            runtime_paths,
            execution_authorized=execution_authorized,
            repository_commit=git_value("rev-parse", "HEAD"),
            branch=git_value("branch", "--show-current"),
        )
    validate_h22_schema(manifest)
    write_json(manifest_path, manifest)
    return manifest


def _status_for_row(output_root: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    path = output_root / row["result_relative_dir"] / EXTERNAL_STATUS_NAME
    if not path.is_file():
        return None
    status = read_json(path)
    validate_h22_external_status(status)
    if status["freeze_content_sha256"] != row["freeze_content_sha256"]:
        raise ValueError(f"{row['execution_id']}: conflicting freeze status")
    if status["frozen_matrix_row_identity"] != matrix_identity(row):
        raise ValueError(f"{row['execution_id']}: conflicting matrix identity")
    if status["h21_anchor_binding_sha256"] != row["h21_anchor_binding_sha256"]:
        raise ValueError(f"{row['execution_id']}: conflicting H2.1 anchor")
    if status["h211_route_binding_sha256"] != row["h211_route_binding_sha256"]:
        raise ValueError(f"{row['execution_id']}: conflicting H2.1.1 route")
    return status


def _index_rows(output_root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in plan["rows"]:
        status = _status_for_row(output_root, row)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
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


def _refresh(
    output_root: Path,
    plan: dict[str, Any],
    manifest: dict[str, Any],
    *,
    attempted: bool,
    canonical: bool = False,
) -> None:
    rows = _index_rows(output_root, plan)
    external_complete = sum(row["execution_end_status"] == "completed" for row in rows)
    external_infra = sum(
        row["execution_end_status"] in {"infrastructure_failed", "blocked_dependency"}
        for row in rows
    )
    not_started = len(rows) - external_complete - external_infra
    canonical_infra = 0
    if canonical:
        plan_by_id = {row["execution_id"]: row for row in plan["rows"]}
        scientific_invalid = 0
        for index_row in rows:
            if index_row["execution_end_status"] != "completed":
                index_row["canonical_cell_status"] = None
                index_row["canonical_cell_failure_reason"] = None
                continue
            cell = load_h22_cell(output_root, plan_by_id[index_row["execution_id"]])
            index_row["canonical_cell_status"] = cell["cell_status"]
            index_row["canonical_cell_failure_reason"] = cell.get(
                "infrastructure_failure_reason"
            )
            canonical_infra += cell["cell_status"] == "infrastructure_failed"
            scientific_invalid += cell["cell_status"] == "correctness_invalid"
    else:
        scientific_invalid = sum(
            row["execution_end_status"] == "completed"
            and row["external_phase2d2_checker_return_code"] not in (None, 0)
            for row in rows
        )
    complete = external_complete - canonical_infra
    infrastructure_failed = external_infra + canonical_infra
    write_jsonl(output_root / SUBRUN_INDEX_NAME, rows)
    launched = bool(manifest.get("scientific_subprocess_launched") or attempted)
    model_forward = bool(
        manifest.get("model_forward_executed")
        or attempted
        or external_complete
        or any(
            (output_root / row["result_relative_dir"] / "samples.jsonl").is_file()
            for row in plan["rows"]
        )
    )
    manifest.update(
        {
            "status": (
                "execution_complete"
                if complete == EXPECTED_ROW_COUNT
                else "partial_execution"
                if launched
                else "planned_not_executed"
            ),
            "scientific_subprocess_launched": launched,
            "model_forward_executed": model_forward,
            "completed_cell_count": complete,
            "incomplete_cell_count": infrastructure_failed + not_started,
            "infrastructure_failed_cell_count": infrastructure_failed,
            "scientific_correctness_invalid_cell_count": scientific_invalid,
            "not_started_cell_count": not_started,
            "canonical_cell_classification_applied": canonical,
        }
    )
    validate_h22_schema(manifest)
    write_json(output_root / EXECUTION_MANIFEST_NAME, manifest)


def _runner_command(
    row: dict[str, Any], output_root: Path, paths: dict[str, Any]
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
        "32",
        "--verification-mode",
        VERIFICATION_MODE,
        "--evidence-role",
        "held_out_evaluation",
        "--output-dir",
        str(result_dir),
    ]
    if row["configuration_family"] == BASE_FAMILY:
        command.extend(("--candidate-merge-mode", BASE_FAMILY))
    else:
        base_dir = (
            output_root
            / "runs"
            / row["task"]
            / row["sample_id"]
            / str(row["nominal_prompt_length"])
            / BASE_FAMILY
        )
        command.extend(
            (
                "--frozen-refinement-config",
                paths["frozen_refinement_config"],
                "--frozen-refinement-config-id",
                row["configuration_family"],
                "--base-result-dir",
                str(base_dir),
            )
        )
    return command


def _base_available(output_root: Path, row: dict[str, Any]) -> bool:
    base_dir = (
        output_root
        / "runs"
        / row["task"]
        / row["sample_id"]
        / str(row["nominal_prompt_length"])
        / BASE_FAMILY
    )
    path = base_dir / EXTERNAL_STATUS_NAME
    if not path.is_file():
        return False
    status = read_json(path)
    return status.get("execution_end_status") == "completed" and all(
        (base_dir / name).is_file() for name in PHASE2D2_REQUIRED_ARTIFACTS
    )


def _produced_identity(
    result_dir: Path, row: dict[str, Any], paths: dict[str, Any]
) -> tuple[bool, str | None]:
    try:
        samples = read_jsonl(result_dir / "samples.jsonl")
        if len(samples) != 1:
            raise ValueError("expected one sample row")
        manifest = read_json(result_dir / "manifest.json")
        frozen = (
            None
            if row["configuration_family"] == BASE_FAMILY
            else _validated_frozen_refinement(Path(paths["frozen_refinement_config"]))
        )
        _validate_result_identity(result_dir, row, manifest, samples[0], frozen)
        return True, None
    except Exception as exc:  # noqa: BLE001 - infrastructure evidence.
        return False, f"{type(exc).__name__}: {exc}"


def _write_blocked_status(
    output_root: Path, row: dict[str, Any], bindings: dict[str, Any]
) -> None:
    result_dir = output_root / row["result_relative_dir"]
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "freeze_content_sha256": row["freeze_content_sha256"],
        "h21_anchor_binding_sha256": bindings["h21_anchor_binding_sha256"],
        "h211_route_binding_sha256": bindings["h211_route_binding_sha256"],
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
        "result_identity_error": "matching frozen N32 base cell is not complete",
        "command": [],
    }
    validate_h22_external_status(payload)
    write_json(result_dir / EXTERNAL_STATUS_NAME, payload)


def _execute_row(
    output_root: Path,
    row: dict[str, Any],
    paths: dict[str, Any],
    bindings: dict[str, Any],
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
    if row["configuration_family"] != BASE_FAMILY and not _base_available(
        output_root, row
    ):
        _write_blocked_status(output_root, row, bindings)
        return "blocked_dependency"
    command = _runner_command(row, output_root, paths)
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
        except Exception as exc:  # noqa: BLE001
            checker_error = f"{type(exc).__name__}: {exc}"
        else:
            checker_rc = checker.returncode
    result_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        name
        for name in PHASE2D2_REQUIRED_ARTIFACTS
        if not (result_dir / name).is_file()
    ]
    identity_valid, identity_error = _produced_identity(result_dir, row, paths)
    end_status = (
        "completed" if not missing and identity_valid else "infrastructure_failed"
    )
    reasons = [
        reason
        for reason in (
            runner_error,
            checker_error,
            f"missing artifacts: {missing}" if missing else None,
            identity_error,
        )
        if reason
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "freeze_content_sha256": row["freeze_content_sha256"],
        "h21_anchor_binding_sha256": bindings["h21_anchor_binding_sha256"],
        "h211_route_binding_sha256": bindings["h211_route_binding_sha256"],
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
        "infrastructure_failure_reason": "; ".join(reasons) if reasons else None,
        "command": command,
    }
    validate_h22_external_status(payload)
    write_json(result_dir / EXTERNAL_STATUS_NAME, payload)
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


def _evaluate_sentinel(
    output_root: Path,
    rows: list[dict[str, Any]],
    bundle: dict[str, Any],
    bindings: dict[str, Any],
) -> dict[str, Any]:
    gates = []
    cells = []
    for row in rows:
        status = _status_for_row(output_root, row)
        result_dir = output_root / row["result_relative_dir"]
        cell = load_h22_cell(output_root, row)
        cells.append(cell)
        gate = _sentinel_cell_gate(
            row,
            status,
            complete_required_artifacts=all(
                (result_dir / name).is_file() for name in PHASE2D2_REQUIRED_ARTIFACTS
            ),
            cell_status=cell["cell_status"],
            infrastructure_failure_reason=cell.get("infrastructure_failure_reason"),
        )
        gate["verification_work"] = cell.get("verification_work")
        gate["correctness"] = cell.get("correctness")
        gates.append(gate)
    base = next(cell for cell in cells if cell["configuration_family"] == BASE_FAMILY)
    refined = next(
        cell for cell in cells if cell["configuration_family"] != BASE_FAMILY
    )
    direct_delta = (
        refined["verification_work"]["actual_verified_nodes_per_round"]
        - base["verification_work"]["actual_verified_nodes_per_round"]
        if base["cell_status"] == refined["cell_status"] == "valid"
        else None
    )
    reasons = [
        f"{gate['execution_id']}:{reason}"
        for gate in gates
        for reason in gate["failure_reasons"]
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "artifact_type": "sentinel_gate",
        "status": "complete",
        "freeze_content_sha256": bundle["freeze_content_sha256"],
        "h21_anchor_binding_sha256": bindings["h21_anchor_binding_sha256"],
        "h211_route_binding_sha256": bindings["h211_route_binding_sha256"],
        "h21_primary_anchor_accepted": True,
        "h211_final_route_accepted": True,
        "passed": len(gates) == 2 and not reasons,
        "cell_gates": gates,
        "failure_reasons": reasons,
        "direct_actual_verification_work": {
            "base_n32_actual_verified_node_count_distribution": base.get(
                "verification_work", {}
            ).get("actual_verified_node_count_distribution"),
            "refined_n32_actual_verified_node_count_distribution": refined.get(
                "verification_work", {}
            ).get("actual_verified_node_count_distribution"),
            "delta_actual_verified_nodes_per_round": direct_delta,
            "configured_n_assumed_as_actual_work": False,
        },
    }
    validate_h22_schema(payload)
    write_json(output_root / SENTINEL_GATE_NAME, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sentinel_pair_only and not args.execute_frozen_secondary:
        raise ValueError("--sentinel-pair-only requires --execute-frozen-secondary")
    bundle = validate_h22_freeze(args.freeze_dir, args.freeze_content_sha256)
    bindings = source_bindings(
        bundle=bundle,
        h21_output_root=args.h21_output_root,
        h211_output_dir=args.h211_output_dir,
    )
    plan = build_execution_plan(bundle, bindings)
    contract = build_analysis_contract(bundle, bindings)
    runtime_paths = _runtime_paths(args)
    output_root = args.output_root.expanduser().resolve()
    manifest = _prepare_root(
        output_root,
        bundle=bundle,
        bindings=bindings,
        plan=plan,
        contract=contract,
        runtime_paths=runtime_paths,
        execution_authorized=args.execute_frozen_secondary,
    )
    _refresh(output_root, plan, manifest, attempted=False)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.execute_frozen_secondary:
        print("PLAN-ONLY: zero model forwards executed")
        return 0
    try:
        _validate_runtime_paths(runtime_paths)
    except Exception as exc:  # noqa: BLE001
        preflight = {
            "status": "failed",
            "parent_sys_executable": sys.executable,
            "configured_child_executable": runtime_paths["python_executable"],
            "required_import_status": False,
            "probe_error": f"runtime path validation failed: {type(exc).__name__}: {exc}",
        }
    else:
        preflight = _child_environment_preflight(runtime_paths)
    manifest["child_environment_preflight"] = preflight
    manifest["child_environment_preflight_status"] = preflight["status"]
    if preflight["status"] != "passed":
        manifest["status"] = "infrastructure_failure"
        manifest["infrastructure_failure_reason"] = preflight.get("probe_error")
        write_json(output_root / EXECUTION_MANIFEST_NAME, manifest)
        print(json.dumps(preflight, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    write_json(output_root / EXECUTION_MANIFEST_NAME, manifest)
    print(json.dumps(preflight, indent=2, sort_keys=True))
    rows = (
        _sentinel_rows(plan)
        if args.sentinel_pair_only
        else sorted(plan["rows"], key=execution_sort_key)
    )
    circuit = _run_rows_with_circuit_breaker(
        rows,
        lambda row: _execute_row(output_root, row, runtime_paths, bindings),
        lambda: _refresh(output_root, plan, manifest, attempted=True),
    )
    _refresh(output_root, plan, manifest, attempted=True, canonical=True)
    manifest["infrastructure_circuit_breaker"] = circuit
    if circuit["triggered"]:
        manifest["status"] = "infrastructure_failure"
    sentinel = None
    if args.sentinel_pair_only:
        sentinel = _evaluate_sentinel(output_root, rows, bundle, bindings)
        manifest["sentinel_gate"] = sentinel
    write_json(output_root / EXECUTION_MANIFEST_NAME, manifest)
    summary = {
        "execution_scope": "sentinel_pair"
        if args.sentinel_pair_only
        else "full_secondary_matrix",
        "output_root": str(output_root),
        "completed_cell_count": manifest["completed_cell_count"],
        "infrastructure_failed_cell_count": manifest[
            "infrastructure_failed_cell_count"
        ],
        "scientific_correctness_invalid_cell_count": manifest[
            "scientific_correctness_invalid_cell_count"
        ],
        "not_started_cell_count": manifest["not_started_cell_count"],
        "sentinel_gate_passed": sentinel["passed"] if sentinel else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.sentinel_pair_only:
        return _sentinel_gate_return_code(sentinel or {})
    return 0 if manifest["incomplete_cell_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
