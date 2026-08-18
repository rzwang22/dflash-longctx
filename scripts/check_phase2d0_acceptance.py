#!/usr/bin/env python3
"""Validate Phase 2D.0 artifacts without loading model frameworks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

if __package__:
    from scripts.phase2d0_dependency_opportunity import (
        AVAILABILITY_FIELDS,
        CONTEXT_GAP_FIELDS,
        DEPENDENCY_FIELDS,
        FRONTIER_FIELDS,
        GAP_FIELDS,
        MINIMUM_FIELDS,
        OUTPUT_FILENAMES,
        SCHEMA_DIR,
        SIGNAL_FIELDS,
    )
else:
    from phase2d0_dependency_opportunity import (
        AVAILABILITY_FIELDS,
        CONTEXT_GAP_FIELDS,
        DEPENDENCY_FIELDS,
        FRONTIER_FIELDS,
        GAP_FIELDS,
        MINIMUM_FIELDS,
        OUTPUT_FILENAMES,
        SCHEMA_DIR,
        SIGNAL_FIELDS,
    )


CSV_FILENAMES = (
    "gap-decomposition.csv",
    "gap-decomposition-v2.csv",
    "context-gap-composition.csv",
    "context-gap-composition-v2.csv",
    "availability-by-depth.csv",
    "static-survival-frontier.csv",
    "minimum-survival-budget.csv",
    "dependency-opportunity.csv",
    "signal-hypotheses.csv",
)
CSV_FIELDS = {
    "gap-decomposition.csv": GAP_FIELDS,
    "gap-decomposition-v2.csv": GAP_FIELDS,
    "context-gap-composition.csv": CONTEXT_GAP_FIELDS,
    "context-gap-composition-v2.csv": CONTEXT_GAP_FIELDS,
    "availability-by-depth.csv": AVAILABILITY_FIELDS,
    "static-survival-frontier.csv": FRONTIER_FIELDS,
    "minimum-survival-budget.csv": MINIMUM_FIELDS,
    "dependency-opportunity.csv": DEPENDENCY_FIELDS,
    "signal-hypotheses.csv": SIGNAL_FIELDS,
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@cache
def _validator(schema_name: str) -> Draft202012Validator:
    schema = _read_json(SCHEMA_DIR / schema_name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate(value: Any, schema_name: str) -> None:
    _validator(schema_name).validate(value)


def _check(checks: list[dict[str, Any]], name: str, passed: bool) -> None:
    checks.append({"check": name, "passed": bool(passed)})


def check_output(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    checks: list[dict[str, Any]] = []
    missing = [name for name in OUTPUT_FILENAMES if not (output_dir / name).is_file()]
    _check(checks, "all_artifacts_exist", not missing)
    if missing:
        return {
            "phase": "2D.0",
            "output_dir": str(output_dir),
            "passed": False,
            "missing_artifacts": missing,
            "checks": checks,
        }

    events = _read_jsonl(output_dir / "opportunity-events.jsonl")
    signals = _read_jsonl(output_dir / "signal-labels.jsonl")
    gap = _read_json(output_dir / "gap-decomposition.json")
    gap_v2 = _read_json(output_dir / "gap-decomposition-v2.json")
    rq = _read_json(output_dir / "research-question-summary.json")
    rq_v2 = _read_json(output_dir / "research-question-summary-v2.json")
    manifest = _read_json(output_dir / "analysis-manifest.json")
    for event in events:
        _validate(event, "phase2d0-opportunity-event.schema.json")
    for signal in signals:
        _validate(signal, "phase2d0-signal-label.schema.json")
    _validate(gap, "phase2d0-gap-decomposition.schema.json")
    _validate(gap_v2, "phase2d0-gap-decomposition.schema.json")
    _validate(rq, "phase2d0-research-question-summary.schema.json")
    _validate(rq_v2, "phase2d0-research-question-summary.schema.json")
    _validate(manifest, "phase2d0-analysis-manifest.schema.json")
    _check(checks, "json_schema_validation", True)
    _check(checks, "v2_json_aliases_match", gap == gap_v2 and rq == rq_v2)
    _check(checks, "event_count", len(events) == gap["event_count"])
    _check(
        checks,
        "manifest_event_count",
        len(events) == manifest["event_count"],
    )
    _check(checks, "signal_rows_present", bool(signals))
    _check(
        checks,
        "frameworks_not_loaded",
        not any(manifest["framework_modules_loaded"].values()),
    )
    _check(
        checks,
        "capture_preflight_preceded_output",
        manifest["capture_preflight_completed_before_output_creation"] is True,
    )
    _check(
        checks,
        "offline_only_interpretation",
        all(
            event["actual_online_acceptance_derived"] is False
            and event["latency_or_speedup_derived"] is False
            for event in events
        ),
    )

    csv_versions_ok = True
    csv_fields_ok = True
    for filename in CSV_FILENAMES:
        with (output_dir / filename).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            if not set(CSV_FIELDS[filename]) <= set(reader.fieldnames or []):
                csv_fields_ok = False
            expected_version = (
                "2.0"
                if filename.startswith(("gap-decomposition", "context-gap-composition"))
                else "1.0"
            )
            if rows and any(
                row.get("schema_version") != expected_version for row in rows
            ):
                csv_versions_ok = False
    _check(checks, "csv_schema_versions", csv_versions_ok)
    _check(checks, "all_csv_schema_fields", csv_fields_ok)

    composition_ok = True
    denominator_ok = True
    no_ambiguous_fraction = True
    seen = set()
    for group in gap["task_context"]:
        key = tuple(
            group[field] for field in ("task", "context_length", "top_m", "node_budget")
        )
        if key in seen:
            continue
        seen.add(key)
        matching = [
            row
            for row in gap["task_context"]
            if tuple(
                row[field]
                for field in ("task", "context_length", "top_m", "node_budget")
            )
            == key
        ]
        eligible = [
            row for row in matching if row["gap_class"] != "right_censored_or_terminal"
        ]
        censored = next(
            row for row in matching if row["gap_class"] == "right_censored_or_terminal"
        )
        composition_ok &= (
            sum(row["gap_class_round_count"] for row in matching)
            == group["raw_round_count"]
            and math.isclose(
                sum(row["fraction_over_all_rounds"] for row in matching), 1.0
            )
            and (
                group["eligible_round_count"] == 0
                or math.isclose(
                    sum(row["fraction_among_eligible_rounds"] for row in eligible),
                    1.0,
                )
            )
            and censored["fraction_among_eligible_rounds"] is None
            and bool(censored["fraction_among_eligible_rounds_unavailable_reason"])
        )
        denominator_ok &= all(
            row["fraction_denominator_name"]
            == (
                "all_configuration_rounds"
                if row["gap_class"] == "right_censored_or_terminal"
                else "eligible_configuration_rounds"
            )
            and row["fraction_denominator_count"]
            == (
                row["raw_round_count"]
                if row["gap_class"] == "right_censored_or_terminal"
                else row["eligible_round_count"]
            )
            for row in matching
        )
        no_ambiguous_fraction &= all("fraction" not in row for row in matching)
    _check(checks, "all_strata_fraction_contract", composition_ok)
    _check(checks, "explicit_fraction_denominators", denominator_ok)
    _check(checks, "ambiguous_naked_fraction_absent", no_ambiguous_fraction)
    _check(
        checks,
        "node_budgets_are_configuration_strata",
        gap["node_budgets_are_separate_configuration_strata"] is True
        and gap["cross_node_budget_counts_are_configuration_round_counts"] is True,
    )
    expected_censoring = {
        (event["task"], event["node_budget"])
        for event in events
        if event["top_m"] == 16
    }
    reported_censoring = {
        (row["task"], row["node_budget"])
        for row in rq["censoring_by_task_and_node_budget"]
    }
    _check(
        checks,
        "task_budget_censoring_reported_separately",
        expected_censoring == reported_censoring,
    )

    return {
        "phase": "2D.0",
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
