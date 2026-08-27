#!/usr/bin/env python3
"""Validate secondary Phase 2D.2A-H2.2 N32 characterization artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.phase2d2a_h21_common import matrix_identity, read_jsonl
from scripts.phase2d2a_h22_common import (
    AGGREGATE_CSV_NAMES,
    AGGREGATE_JSON_NAMES,
    ANALYSIS_CONTRACT_NAME,
    EXECUTION_MANIFEST_NAME,
    EXECUTION_PLAN_NAME,
    EXPECTED_PAIR_COUNT,
    EXPECTED_ROW_COUNT,
    SENTINEL_GATE_NAME,
    SUBRUN_INDEX_NAME,
    build_aggregate,
    build_analysis_contract,
    ensure_immutable_sources,
    load_h21_anchor_cells,
    load_h22_cell,
    read_json,
    source_bindings,
    validate_h22_freeze,
    validate_h22_schema,
    validate_plan,
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validate_no_extra_outputs(output_root: Path, plan: dict[str, Any]) -> None:
    runs = output_root / "runs"
    if not runs.exists():
        return
    expected = {
        (output_root / row["result_relative_dir"]).resolve() for row in plan["rows"]
    }
    observed = set()
    for path in runs.rglob("*"):
        if path.is_file():
            relative = path.relative_to(runs)
            if len(relative.parts) < 5:
                raise ValueError(f"malformed extra H2.2 output: {path}")
            observed.add((runs.joinpath(*relative.parts[:4])).resolve())
    extra = sorted(map(str, observed - expected))
    if extra:
        raise ValueError(f"extra non-frozen H2.2 outputs: {extra}")


def _verify_equations(artifacts: dict[str, dict[str, Any]]) -> None:
    cross = artifacts["cross_budget"]["rows"]
    for row in cross:
        delta = row["delta_actual_verified_nodes_per_round"]
        accepted = row["delta_accepted_draft_tokens_per_round"]
        committed = row["delta_committed_tokens_per_round"]
        if row["comparison_valid"] and delta is not None and delta > 0:
            if row["accepted_tokens_per_additional_verified_node"] != accepted / delta:
                raise ValueError(
                    "accepted marginal yield does not use actual node delta"
                )
            if (
                row["committed_tokens_per_additional_verified_node"]
                != committed / delta
            ):
                raise ValueError(
                    "committed marginal yield does not use actual node delta"
                )
        elif delta == 0 and (
            row["accepted_tokens_per_additional_verified_node"] is not None
            or row["committed_tokens_per_additional_verified_node"] is not None
        ):
            raise ValueError("zero actual-node delta was converted to a zero yield")
        if row["configured_budget_used_as_denominator"]:
            raise ValueError("configured N entered marginal-yield denominator")
    for row in artifacts["decomposition"]["rows"]:
        if not row["comparison_valid"]:
            continue
        expected_accepted = (
            row["combined_gain_n32_accepted_tokens_per_round"]
            - row["same_work_refinement_gain_n16_accepted_tokens_per_round"]
        )
        expected_committed = (
            row["combined_gain_n32_committed_tokens_per_round"]
            - row["same_work_refinement_gain_n16_committed_tokens_per_round"]
        )
        if (
            row["incremental_budget_contribution_accepted_tokens_per_round"]
            != expected_accepted
            or row["incremental_budget_contribution_committed_tokens_per_round"]
            != expected_committed
        ):
            raise ValueError("paired incremental-budget decomposition formula changed")
        if row["causal_estimator_claimed"]:
            raise ValueError("H2.2 decomposition is incorrectly called causal")
    for row in artifacts["within_n32"]["rows"]:
        if row["delta_actual_verified_nodes_per_round"] not in (None, 0) and (
            row["comparison_interpretation"]
            != "combined_refinement_plus_actual_work_difference"
        ):
            raise ValueError("different-work N32 comparison is not labelled combined")
        if row["called_same_work"]:
            raise ValueError("H2.2 N32 comparison was called same-work")


def check_h22_output(
    *,
    freeze_dir: Path,
    freeze_content_sha256: str,
    h21_output_root: Path,
    h211_output_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    bundle = validate_h22_freeze(freeze_dir, freeze_content_sha256)
    bindings = source_bindings(
        bundle=bundle,
        h21_output_root=h21_output_root,
        h211_output_dir=h211_output_dir,
    )
    required = (
        ANALYSIS_CONTRACT_NAME,
        EXECUTION_MANIFEST_NAME,
        EXECUTION_PLAN_NAME,
        SUBRUN_INDEX_NAME,
        SENTINEL_GATE_NAME,
        *AGGREGATE_JSON_NAMES,
        *AGGREGATE_CSV_NAMES,
    )
    missing = [name for name in required if not (output_root / name).is_file()]
    if missing:
        raise ValueError(f"missing H2.2 artifacts: {missing}")
    plan = read_json(output_root / EXECUTION_PLAN_NAME)
    manifest = read_json(output_root / EXECUTION_MANIFEST_NAME)
    contract = read_json(output_root / ANALYSIS_CONTRACT_NAME)
    sentinel = read_json(output_root / SENTINEL_GATE_NAME)
    validate_plan(plan, bundle, bindings)
    validate_h22_schema(manifest)
    validate_h22_schema(contract)
    validate_h22_schema(sentinel)
    ensure_immutable_sources(manifest, bundle, bindings)
    if contract != build_analysis_contract(bundle, bindings):
        raise ValueError("H2.2 analysis contract changed after execution")
    if plan["row_count"] != EXPECTED_ROW_COUNT or any(
        row["node_budget"] != 32 for row in plan["rows"]
    ):
        raise ValueError("H2.2 plan is not exact N32 secondary matrix")
    index = read_jsonl(output_root / SUBRUN_INDEX_NAME)
    if len(index) != EXPECTED_ROW_COUNT:
        raise ValueError("H2.2 subrun index does not retain 112 rows")
    plan_by_id = {row["execution_id"]: row for row in plan["rows"]}
    if {row["execution_id"] for row in index} != set(plan_by_id):
        raise ValueError("H2.2 subrun index differs from frozen plan")
    for row in index:
        expected = matrix_identity(plan_by_id[row["execution_id"]])
        if any(row[field] != value for field, value in expected.items()):
            raise ValueError("H2.2 subrun index identity changed")
    _validate_no_extra_outputs(output_root, plan)
    _, h21_cells = load_h21_anchor_cells(h21_output_root)
    h22_cells = [load_h22_cell(output_root, row) for row in plan["rows"]]
    expected = build_aggregate(
        bundle=bundle,
        bindings=bindings,
        plan=plan,
        h21_cells=h21_cells,
        h22_cells=h22_cells,
    )
    names = {
        "cell_summary": AGGREGATE_JSON_NAMES[0],
        "within_n32": AGGREGATE_JSON_NAMES[1],
        "cross_budget": AGGREGATE_JSON_NAMES[2],
        "decomposition": AGGREGATE_JSON_NAMES[3],
        "task_context": AGGREGATE_JSON_NAMES[4],
        "mechanism": AGGREGATE_JSON_NAMES[5],
        "cost": AGGREGATE_JSON_NAMES[6],
        "conclusion": AGGREGATE_JSON_NAMES[7],
    }
    persisted = {}
    for key, name in names.items():
        observed = read_json(output_root / name)
        validate_h22_schema(observed)
        if _canonical(observed) != _canonical(expected[key]):
            raise ValueError(f"persisted H2.2 artifact differs: {name}")
        persisted[key] = observed
    for name in AGGREGATE_CSV_NAMES:
        with (output_root / name).open(encoding="utf-8", newline="") as stream:
            if not list(csv.DictReader(stream)):
                raise ValueError(f"empty H2.2 CSV artifact: {name}")
    cells = persisted["cell_summary"]
    if cells["observed_cell_count"] != EXPECTED_ROW_COUNT:
        raise ValueError("H2.2 cell inventory changed")
    if cells["infrastructure_failed_cell_count"]:
        raise ValueError("H2.2 execution is infrastructure-incomplete")
    for cell in cells["rows"]:
        if cell["cell_status"] != "valid":
            continue
        work = cell["verification_work"]
        counts = work["actual_verified_node_counts"]
        measured = sum(counts) / len(counts) if counts else None
        if (
            work["actual_work_derived_from_round_artifacts"] is not True
            or work["configured_node_budget_not_used_as_actual_work"] is not True
            or work["actual_verified_nodes_per_round"] != measured
        ):
            raise ValueError(
                "actual verification work is not derived from round artifacts"
            )
    if len(persisted["within_n32"]["rows"]) != EXPECTED_PAIR_COUNT:
        raise ValueError("H2.2 N32 pair inventory differs from 84")
    if len(persisted["cross_budget"]["rows"]) != EXPECTED_ROW_COUNT:
        raise ValueError("H2.2 N16/N32 pair inventory differs from 112")
    _verify_equations(persisted)
    conclusion = persisted["conclusion"]
    if (
        conclusion["stable_cross_task_same_work_signal_produced"]
        or conclusion["h21_primary_rewritten"]
        or conclusion["h22_can_override_h21_primary"]
        or conclusion["winner_selected"]
        or conclusion["phase2d2b_entered"]
        or conclusion["phase2d3_entered"]
        or conclusion["speedup_claimed"]
    ):
        raise ValueError("H2.2 issued a prohibited primary/route claim")
    if any(
        row["task"] == "gov_report" and row["task_role"] != "same_task_replication_only"
        for row in persisted["task_context"]["rows"]
    ):
        raise ValueError("GovReport entered independent-task interpretation")
    return {
        "status": "accepted",
        "phase": "2D.2A-H2.2",
        "freeze_content_sha256": freeze_content_sha256,
        "cell_count": cells["observed_cell_count"],
        "valid_cell_count": cells["valid_cell_count"],
        "correctness_invalid_cell_count": cells["correctness_invalid_cell_count"],
        "infrastructure_failed_cell_count": cells["infrastructure_failed_cell_count"],
        "within_n32_pair_count": len(persisted["within_n32"]["rows"]),
        "cross_budget_pair_count": len(persisted["cross_budget"]["rows"]),
        "h22_can_override_h21_primary": False,
        "winner_selected": False,
        "speedup_claimed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", required=True, type=Path)
    parser.add_argument("--freeze-content-sha256", required=True)
    parser.add_argument("--h21-output-root", required=True, type=Path)
    parser.add_argument("--h211-output-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = check_h22_output(
            freeze_dir=args.freeze_dir,
            freeze_content_sha256=args.freeze_content_sha256,
            h21_output_root=args.h21_output_root,
            h211_output_dir=args.h211_output_dir,
            output_root=args.output_root,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"REJECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
