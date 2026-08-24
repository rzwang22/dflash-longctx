#!/usr/bin/env python3
"""Validate a Phase 2D.2A-H1 known-sample aggregate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

if __package__:
    from scripts.aggregate_phase2d2a_h_known import (
        ARTIFACTS,
        EXPECTED_CONTEXT_LENGTHS,
        EXPECTED_FAMILIES,
        EXPECTED_NODE_BUDGETS,
        FROZEN_REFINEMENT_FAMILIES,
        SCHEMAS,
    )
else:
    from aggregate_phase2d2a_h_known import (
        ARTIFACTS,
        EXPECTED_CONTEXT_LENGTHS,
        EXPECTED_FAMILIES,
        EXPECTED_NODE_BUDGETS,
        FROZEN_REFINEMENT_FAMILIES,
        SCHEMAS,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = REPO_ROOT / "docs" / "schemas"


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc


def _validate(payload: Any, schema_name: str) -> None:
    schema = _json(SCHEMA_ROOT / schema_name)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)


def check_h1_output(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    missing = [name for name in ARTIFACTS if not (output_dir / name).is_file()]
    if missing:
        raise ValueError(f"missing Phase 2D.2A-H1 artifacts: {missing}")
    payloads = {
        name: _json(output_dir / name)
        for name in SCHEMAS
    }
    for name, schema in SCHEMAS.items():
        _validate(payloads[name], schema)
    manifest = payloads["phase2d2a-h-known-manifest.json"]
    paired = payloads["phase2d2a-h-known-paired-comparison.json"]
    config = payloads["phase2d2a-h-known-config-summary.json"]
    costs = payloads["phase2d2a-h-known-cost-summary.json"]
    decision = payloads["phase2d2a-h-known-decision.json"]

    expected_cells = {
        (family, context, budget)
        for family in EXPECTED_FAMILIES
        for context in EXPECTED_CONTEXT_LENGTHS
        for budget in EXPECTED_NODE_BUDGETS
    }
    observed_cells = [
        (
            row["configuration_family"],
            int(row["nominal_prompt_length"]),
            int(row["node_budget"]),
        )
        for row in manifest["cells"]
    ]
    if set(observed_cells) != expected_cells or any(
        count != 1 for count in Counter(observed_cells).values()
    ):
        raise ValueError("H1 manifest does not contain one exact 16-cell matrix")

    expected_pairs = {
        (family, context, budget)
        for family in FROZEN_REFINEMENT_FAMILIES
        for context in EXPECTED_CONTEXT_LENGTHS
        for budget in EXPECTED_NODE_BUDGETS
    }
    observed_pairs = [
        (
            row["refined_family"],
            int(row["nominal_prompt_length"]),
            int(row["node_budget"]),
        )
        for row in paired["rows"]
    ]
    if set(observed_pairs) != expected_pairs or any(
        count != 1 for count in Counter(observed_pairs).values()
    ):
        raise ValueError("H1 paired artifact is not the exact same-N/context matrix")
    for row in paired["rows"]:
        key = row["pairing_key"]
        if int(key["context_length"]) != int(row["nominal_prompt_length"]):
            raise ValueError("H1 pair crosses context lengths")
        if int(key["node_budget"]) != int(row["node_budget"]):
            raise ValueError("H1 pair crosses node budgets")
        if row["correctness_valid"]:
            continue
        candidate = row["candidate_quality"]
        if any(
            candidate[field] is not None
            for field in (
                "base_selected_tree_coverage",
                "refined_selected_tree_coverage",
                "coverage_absolute_gain",
                "coverage_relative_gain",
                "refined_only_target_recovery_rate",
            )
        ):
            raise ValueError("invalid H1 pair contributes candidate benefit metrics")
        for metric in row["online_efficiency"].values():
            if any(value is not None for value in metric.values()):
                raise ValueError("invalid H1 pair contributes online benefit metrics")
        if any(
            value is not None
            for field, value in row["formal_timing"].items()
            if field != "formal_timing_source"
        ):
            raise ValueError("invalid H1 pair contributes formal cost metrics")

    if {row["refined_family"] for row in config["rows"]} != set(
        FROZEN_REFINEMENT_FAMILIES
    ):
        raise ValueError("H1 config summary reselected the frozen family set")
    if {(
        row["refined_family"],
        int(row["nominal_prompt_length"]),
        int(row["node_budget"]),
    ) for row in costs["rows"]} != expected_pairs:
        raise ValueError("H1 cost rows do not match the paired matrix")
    for row in costs["rows"]:
        if row["eligible_for_cost_conclusions"] != row["correctness_valid"]:
            raise ValueError("H1 cost eligibility contradicts correctness")
        if row["correctness_valid"]:
            continue
        values = [
            *row["dependency_modeling_cost"].values(),
            *(
                value
                for field, value in row[
                    "correctness_preserving_verifier_cost"
                ].items()
                if field != "fallback_timer_excludes_snapshot_and_kv_replacement"
            ),
            *row["combined_additional_work"].values(),
        ]
        if any(value is not None for value in values):
            raise ValueError("invalid H1 cell contributes to cost conclusions")
    if decision["final_phase2d2a_decision"] != "insufficient_evidence":
        raise ValueError("one-task H1 issued a prohibited final Phase 2D.2A decision")
    if decision["independent_task_count"] != 1 or decision["winner_selected"]:
        raise ValueError("H1 decision overstates one-task evidence")
    return {
        "status": "accepted",
        "phase": "2D.2A-H1",
        "cell_count": len(manifest["cells"]),
        "valid_cell_count": sum(
            row["correctness"]["correctness_valid"] for row in manifest["cells"]
        ),
        "invalid_cell_count": manifest["invalid_cell_count"],
        "pair_count": len(paired["rows"]),
        "final_phase2d2a_decision": decision["final_phase2d2a_decision"],
        "candidate_quality_label_authority": manifest[
            "candidate_quality_target_label_authority"
        ],
        "q1_replay_used_as_target_label_source": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = check_h1_output(args.output_dir)
    except Exception as exc:  # noqa: BLE001 - clean acceptance failure.
        print(f"REJECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
