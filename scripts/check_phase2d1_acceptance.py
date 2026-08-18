#!/usr/bin/env python3
"""Validate Phase 2D.1A artifacts without loading model frameworks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase2d1.events import (
    FORBIDDEN_RUNTIME_FIELD_FRAGMENTS,
    RUNTIME_FEATURE_NAMES,
)

if __package__:
    from scripts.phase2d1_selector_realization import (
        OUTPUT_FILENAMES,
        SCHEMA_DIR,
    )
else:
    from phase2d1_selector_realization import OUTPUT_FILENAMES, SCHEMA_DIR

CSV_FILES = (
    "fixed-policy-summary.csv",
    "policy-winner-composition.csv",
    "policy-overlap.csv",
    "policy-portfolio-frontier.csv",
    "global-fixed-evaluation.csv",
    "context-rule-evaluation.csv",
    "decision-stump-evaluation.csv",
    "selector-headroom.csv",
    "selector-regret.csv",
    "source-cluster-bootstrap.csv",
    "context-selector-summary.csv",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@cache
def _validator(name: str) -> Draft202012Validator:
    schema = _read_json(SCHEMA_DIR / name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate(value: Any, name: str) -> None:
    _validator(name).validate(value)


def _check(checks: list[dict[str, Any]], name: str, passed: bool) -> None:
    checks.append({"check": name, "passed": bool(passed)})


def check_output(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    checks: list[dict[str, Any]] = []
    missing = [name for name in OUTPUT_FILENAMES if not (output_dir / name).is_file()]
    _check(checks, "all_artifacts_exist", not missing)
    if missing:
        return {
            "phase": "2D.1A",
            "passed": False,
            "missing_artifacts": missing,
            "checks": checks,
        }
    events = _read_jsonl(output_dir / "selector-events.jsonl")
    fixed = _read_json(output_dir / "fixed-policy-summary.json")
    folds = _read_json(output_dir / "cross-validation-folds.json")
    research = _read_json(output_dir / "research-question-summary.json")
    decision = _read_json(output_dir / "selector-decision.json")
    manifest = _read_json(output_dir / "analysis-manifest.json")
    for event in events:
        _validate(event, "phase2d1-selector-event.schema.json")
    _validate(fixed, "phase2d1-fixed-policy-summary.schema.json")
    _validate(folds, "phase2d1-cross-validation-folds.schema.json")
    _validate(research, "phase2d1-research-question-summary.schema.json")
    _validate(decision, "phase2d1-selector-decision.schema.json")
    _validate(manifest, "phase2d1-analysis-manifest.schema.json")
    _check(checks, "json_schema_validation", True)

    _check(
        checks,
        "coverage_oracle_invariant",
        all(
            event["top1_coverage_depth"]
            <= event["policy_oracle_coverage_depth"]
            <= event["prefix_closed_target_oracle_coverage_depth"]
            <= event["candidate_availability_prefix"]
            for event in events
        ),
    )
    _check(
        checks,
        "runtime_features_are_target_blind",
        all(
            tuple(event["runtime_feature_names"]) == RUNTIME_FEATURE_NAMES
            and event["runtime_features_use_target_information"] is False
            and not any(
                fragment in name
                for name in event["runtime_features"]
                for fragment in FORBIDDEN_RUNTIME_FIELD_FRAGMENTS
            )
            for event in events
        ),
    )
    _check(
        checks,
        "node_budget_root_inclusive",
        all(
            event["root_included_in_node_budget"]
            and all(
                policy["actual_constructed_node_count"] <= policy["node_budget_cap"]
                for policy in event["policy_configurations"]
            )
            for event in events
        ),
    )
    _check(
        checks,
        "n32_top1_actual_nodes_not_misreported",
        all(
            event["top1_actual_constructed_node_count"] < 32
            for event in events
            if event["node_budget_cap"] == 32
        ),
    )
    by_budget = {
        budget: {
            event["physical_round_id"]
            for event in events
            if event["node_budget_cap"] == budget
        }
        for budget in (16, 32)
    }
    _check(
        checks,
        "node_budget_strata_align_by_physical_round",
        by_budget[16] == by_budget[32],
    )

    leakage = False
    prediction_completeness = True
    training_only = True
    for filename in (
        "global-fixed-evaluation.csv",
        "context-rule-evaluation.csv",
        "decision-stump-evaluation.csv",
    ):
        rows = _read_csv(output_dir / filename)
        for scheme in ("leave-one-source-out", "leave-one-task-out"):
            prediction_completeness &= sum(
                row["evaluation_scheme"] == scheme for row in rows
            ) == len(events)
        for row in rows:
            training = set(json.loads(row["training_source_ids"]))
            held_out = set(json.loads(row["held_out_source_ids"]))
            if training & held_out or row["held_out_source_id"] in training:
                leakage = True
            training_only &= (
                row["selector_parameters_selected_from_training_only"] == "True"
                and row["runtime_features_use_target_information"] == "False"
            )
    _check(checks, "no_source_leakage", not leakage)
    _check(checks, "each_event_predicted_once_per_scheme", prediction_completeness)
    _check(checks, "selector_parameters_training_only", training_only)
    _check(
        checks,
        "fold_document_reports_no_leakage",
        folds["source_leakage_detected"] is False,
    )
    loso_folds = [
        fold
        for fold in folds["folds"]
        if fold["evaluation_scheme"] == "leave-one-source-out"
    ]
    _check(
        checks,
        "loso_membership_complete_and_disjoint",
        len(loso_folds) == folds["source_cluster_count"]
        and sorted(
            source for fold in loso_folds for source in fold["held_out_source_ids"]
        )
        == sorted(folds["source_cluster_ids"])
        and all(
            not set(fold["training_source_ids"]) & set(fold["held_out_source_ids"])
            for fold in folds["folds"]
        ),
    )

    portfolio_rows = _read_csv(output_dir / "policy-portfolio-frontier.csv")
    _check(
        checks,
        "portfolio_is_explicit_non_deployable_oracle",
        bool(portfolio_rows)
        and all(
            row["portfolio_is_oracle"] == "True"
            and row["portfolio_is_deployable"] == "False"
            for row in portfolio_rows
        ),
    )
    csv_ok = True
    for filename in CSV_FILES:
        rows = _read_csv(output_dir / filename)
        if not rows or any(row.get("schema_version") != "1.0" for row in rows):
            csv_ok = False
    _check(checks, "all_csv_schema_versions", csv_ok)
    _check(
        checks,
        "frameworks_not_loaded",
        not any(manifest["framework_modules_loaded"].values()),
    )
    _check(
        checks,
        "input_hashes_recorded",
        bool(manifest["capture_inputs"])
        and bool(manifest["phase2d0_input"]["artifact_sha256"])
        and bool(manifest["phase2d0_input"]["directory_sha256"]),
    )
    _check(
        checks,
        "decision_gate_provenance",
        decision["decision_rules"] == manifest["decision_rules"]
        and decision["decision"] == research["decision"] == manifest["decision"],
    )
    _check(checks, "inputs_read_only", manifest["input_artifacts_mutated"] is False)
    _check(
        checks,
        "offline_only",
        all(
            event["actual_online_acceptance_derived"] is False
            and event["latency_or_speedup_derived"] is False
            for event in events
        ),
    )
    return {
        "phase": "2D.1A",
        "output_dir": str(output_dir),
        "passed": all(check["passed"] for check in checks),
        "missing_artifacts": [],
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = check_output(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
