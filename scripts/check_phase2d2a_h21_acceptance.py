#!/usr/bin/env python3
"""Validate a Phase 2D.2A-H2.1 primary held-out evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.phase2d2a_h21_common import (
    AGGREGATE_JSON_ARTIFACTS,
    BASE_FAMILY,
    EXECUTION_MANIFEST_NAME,
    EXECUTION_PLAN_NAME,
    EXPECTED_INDEPENDENT_STRATA,
    EXPECTED_PAIR_COUNT,
    EXPECTED_PRIMARY_ROW_COUNT,
    FAMILIES,
    FROZEN_REFINEMENT_FAMILIES,
    REQUIRED_FREEZE_SHA256,
    SUBRUN_INDEX_NAME,
    build_aggregate,
    ensure_immutable_freeze,
    matrix_identity,
    read_json,
    read_jsonl,
    validate_h21_freeze,
    validate_h21_schema,
    validate_no_extra_outputs,
    validate_plan,
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _exactly_equal(
    observed: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    if _canonical(observed) != _canonical(expected):
        raise ValueError(f"persisted H2.1 {label} differs from recomputed evidence")


def check_h21_output(
    output_root: Path,
    freeze_dir: Path,
    freeze_content_sha256: str,
    *,
    required_freeze_sha256: str = REQUIRED_FREEZE_SHA256,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    bundle = validate_h21_freeze(
        freeze_dir,
        freeze_content_sha256,
        required_freeze_sha256=required_freeze_sha256,
    )
    required = [
        EXECUTION_MANIFEST_NAME,
        EXECUTION_PLAN_NAME,
        SUBRUN_INDEX_NAME,
        *AGGREGATE_JSON_ARTIFACTS,
    ]
    missing = [name for name in required if not (output_root / name).is_file()]
    if missing:
        raise ValueError(f"missing Phase 2D.2A-H2.1 artifacts: {missing}")
    manifest = read_json(output_root / EXECUTION_MANIFEST_NAME)
    plan = read_json(output_root / EXECUTION_PLAN_NAME)
    validate_h21_schema(manifest)
    validate_plan(plan, bundle)
    ensure_immutable_freeze(manifest, bundle)
    if manifest["h20_artifacts_mutated"]:
        raise ValueError("H2.1 claims to mutate H2.0")
    if manifest["h2_secondary_execution_authorized"]:
        raise ValueError("H2.1 authorized H2.2/N32")
    if plan["row_count"] != EXPECTED_PRIMARY_ROW_COUNT:
        raise ValueError("H2.1 plan is not exactly 112 rows")
    identities = [row["execution_id"] for row in plan["rows"]]
    if any(count != 1 for count in Counter(identities).values()):
        raise ValueError("H2.1 plan contains duplicate execution rows")
    if any(
        row["node_budget"] != 16
        or row["verification_mode"] != "always_fixed_shape_recheck"
        or row["candidate_label_authority"] != "original_dflash_reference"
        or row["candidate_label_source"] != "original_dflash_absolute_continuation"
        or row["q1_replay_used_as_target_label_source"]
        for row in plan["rows"]
    ):
        raise ValueError("H2.1 plan violates frozen verifier/label semantics")
    if tuple(plan["configuration_families"]) != FAMILIES:
        raise ValueError("H2.1 removed, added, or reselected a family")
    index = read_jsonl(output_root / SUBRUN_INDEX_NAME)
    if len(index) != EXPECTED_PRIMARY_ROW_COUNT:
        raise ValueError("H2.1 subrun index does not retain exactly 112 rows")
    plan_by_id = {row["execution_id"]: row for row in plan["rows"]}
    if {row["execution_id"] for row in index} != set(plan_by_id):
        raise ValueError("H2.1 subrun index differs from frozen plan")
    for row in index:
        expected = matrix_identity(plan_by_id[row["execution_id"]])
        if any(row[field] != value for field, value in expected.items()):
            raise ValueError("H2.1 subrun index identity differs from frozen plan")
    validate_no_extra_outputs(output_root, plan)
    expected_aggregate = build_aggregate(
        output_root=output_root,
        bundle=bundle,
        plan=plan,
    )
    names = {
        "phase2d2a-h21-cell-summary.json": "cell_summary",
        "phase2d2a-h21-sample-paired-comparison.json": "paired",
        "phase2d2a-h21-task-context-summary.json": "task_context",
        "phase2d2a-h21-family-summary.json": "family",
        "phase2d2a-h21-mechanism-summary.json": "mechanism",
        "phase2d2a-h21-cost-summary.json": "cost",
        "phase2d2a-h21-primary-conclusion.json": "primary",
    }
    persisted = {}
    for name, key in names.items():
        payload = read_json(output_root / name)
        validate_h21_schema(payload)
        _exactly_equal(payload, expected_aggregate[key], name)
        persisted[key] = payload
    cells = persisted["cell_summary"]
    pairs = persisted["paired"]
    task_rows = persisted["task_context"]["rows"]
    family_rows = persisted["family"]["rows"]
    primary = persisted["primary"]
    if not primary["execution_complete"] or cells["infrastructure_failed_cell_count"]:
        raise ValueError("H2.1 primary matrix is infrastructure-incomplete")
    if cells["observed_cell_count"] != EXPECTED_PRIMARY_ROW_COUNT:
        raise ValueError("H2.1 cell summary lost frozen rows")
    if (
        pairs["expected_pair_count"] != EXPECTED_PAIR_COUNT
        or len(pairs["rows"]) != EXPECTED_PAIR_COUNT
    ):
        raise ValueError("H2.1 paired summary is not the exact 84-row product")
    for row in pairs["rows"]:
        if (
            row["base_family"] != BASE_FAMILY
            or row["refined_family"] not in FROZEN_REFINEMENT_FAMILIES
        ):
            raise ValueError("H2.1 base/refined pairing changed")
        if not row["correctness_valid"]:
            candidate = row["candidate_quality"]
            if any(
                candidate[field]["absolute_gain"] is not None
                for field in (
                    "selected_tree_coverage_rate_at_fixed_n",
                    "base_candidate_coverage_rate",
                    "conditional_candidate_coverage_rate",
                    "merged_candidate_coverage_rate",
                )
            ):
                raise ValueError(
                    "correctness-invalid pair contributes candidate benefit"
                )
            if any(
                metric["absolute_gain"] is not None
                for metric in row["online_efficiency"].values()
            ):
                raise ValueError("correctness-invalid pair contributes online benefit")
    for row in task_rows:
        candidate = row["candidate_coverage"]
        accepted = row["accepted_per_verified"]
        if candidate["mean_of_sample_rates_used_as_primary"]:
            raise ValueError("H2.1 replaced pooled candidate metric with mean rates")
        if accepted["mean_of_sample_ratios_used_as_primary"]:
            raise ValueError("H2.1 replaced pooled online metric with mean ratios")
        expected_base = (
            candidate["base_membership_count"]
            / candidate["base_labelled_position_count"]
            if candidate["base_labelled_position_count"]
            else None
        )
        expected_refined = (
            candidate["refined_membership_count"]
            / candidate["refined_labelled_position_count"]
            if candidate["refined_labelled_position_count"]
            else None
        )
        if (
            candidate["base_selected_tree_coverage"] != expected_base
            or candidate["refined_selected_tree_coverage"] != expected_refined
        ):
            raise ValueError("H2.1 candidate coverage is not pooled from counts")
        expected_apv = (
            accepted["refined_accepted_draft_token_count"]
            / accepted["refined_actual_verified_node_count"]
            if accepted["refined_actual_verified_node_count"]
            else None
        )
        if accepted["refined_rate"] != expected_apv:
            raise ValueError("H2.1 accepted/verified is not pooled from counts")
        if "sample_heterogeneity" not in row:
            raise ValueError("H2.1 loses sample-level heterogeneity")
    if {
        (row["task"], row["nominal_prompt_length"])
        for row in task_rows
        if row["task"] in {"qasper", "hotpotqa"}
    } != set(EXPECTED_INDEPENDENT_STRATA):
        raise ValueError("Qasper/HotpotQA independent strata changed")
    if any(
        row["govreport_contributes_to_cross_task_classification"] for row in family_rows
    ):
        raise ValueError("GovReport entered cross-task stability")
    if len(family_rows) != 3 or any(row["winner_selected"] for row in family_rows):
        raise ValueError("H2.1 selected a family winner")
    if (
        primary["independent_task_count"] != 2
        or primary["govreport_contributes_to_independent_task_count"]
    ):
        raise ValueError("H2.1 independent task accounting changed")
    if primary["winner_selected"] or primary["final_phase2d2a_branch_entered"]:
        raise ValueError("H2.1 automatically entered a final Phase2D2A branch")
    if (
        primary["proceed_to_lightweight_causal_adapter_issued"]
        or primary["proceed_with_conditional_remask_issued"]
    ):
        raise ValueError("H2.1 issued a prohibited next-phase decision")
    return {
        "status": "accepted",
        "phase": "2D.2A-H2.1",
        "freeze_content_sha256": freeze_content_sha256,
        "execution_complete": primary["execution_complete"],
        "cell_count": cells["observed_cell_count"],
        "valid_cell_count": cells["valid_cell_count"],
        "correctness_invalid_cell_count": cells["correctness_invalid_cell_count"],
        "infrastructure_failed_cell_count": cells["infrastructure_failed_cell_count"],
        "pair_count": len(pairs["rows"]),
        "independent_task_count": 2,
        "family_cross_task_classifications": primary[
            "family_cross_task_classifications"
        ],
        "winner_selected": False,
        "final_phase2d2a_branch_entered": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", required=True, type=Path)
    parser.add_argument("--freeze-content-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = check_h21_output(
            args.output_root, args.freeze_dir, args.freeze_content_sha256
        )
    except Exception as exc:  # noqa: BLE001 - clean acceptance rejection.
        print(f"REJECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
