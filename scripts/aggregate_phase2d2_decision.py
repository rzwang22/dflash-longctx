#!/usr/bin/env python3
"""Aggregate held-out Phase 2D.2A task runs into the preregistered decision."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase2d2.metrics import build_phase2d2_decision
from scripts.check_phase2d2_acceptance import check_output


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_positive_by_task(
    values: dict[str, list[float]], *, minimum_tasks: int = 2
) -> bool | None:
    if len(values) < minimum_tasks or any(not rows for rows in values.values()):
        return None
    task_means = [statistics.fmean(rows) for rows in values.values()]
    return bool(
        all(value > 0.0 for value in task_means) and statistics.fmean(task_means) > 0.0
    )


def aggregate_result_directories(roots: list[Path]) -> dict[str, Any]:
    if not roots:
        raise ValueError("at least one Phase 2D.2A result directory is required")
    records = []
    for root in roots:
        checker = check_output(root)
        manifest = _read(root / "manifest.json")
        if manifest["phase2d2"].get("evidence_role") != "held_out_evaluation":
            raise ValueError("cross-task decision accepts held-out results only")
        if manifest["phase2d2"]["candidate_merge_mode"] == "base_only":
            raise ValueError(
                "cross-task decision inputs must be refined configurations"
            )
        records.append(
            {
                "root": str(root.expanduser().resolve()),
                "checker": checker,
                "manifest": manifest,
                "correctness": _read(root / "correctness-summary.json"),
                "comparison": _read(root / "fixed-budget-comparison.json"),
                "fallback": _read(root / "fallback-summary.json"),
                "timing": _read(root / "timing-summary.json"),
            }
        )
    frozen_hashes = {
        row["manifest"]["phase2d2"].get("frozen_refinement_config_hash")
        for row in records
    }
    family_ids = {
        row["manifest"]["phase2d2"].get("frozen_refinement_configuration_id")
        for row in records
    }
    if None in frozen_hashes or len(frozen_hashes) != 1:
        raise ValueError("decision inputs do not share one frozen refinement hash")
    if None in family_ids or len(family_ids) != 1:
        raise ValueError("decision inputs do not share one refinement family")

    coverage: dict[str, list[float]] = defaultdict(list)
    efficiency: dict[str, list[float]] = defaultdict(list)
    conditional_total_ratios = []
    conditional_initial_draft_ratios = []
    tasks = set()
    budgets = set()
    guard_false_negative_count = 0
    for record in records:
        phase = record["manifest"]["phase2d2"]
        task = record["manifest"]["workload"]["task"]
        tasks.add(task)
        budgets.add(int(phase["node_budget"]))
        guard_false_negative_count += int(
            record["fallback"]["guard_false_negative_count"]
        )
        for row in record["comparison"].get("rows", []):
            coverage_gain = row.get(
                "selected_tree_target_coverage_rate_at_fixed_n_gain"
            )
            efficiency_gain = row.get("accepted_tokens_per_verified_node_gain")
            if coverage_gain is not None:
                coverage[task].append(float(coverage_gain))
            if efficiency_gain is not None:
                efficiency[task].append(float(efficiency_gain))
        for row in record["timing"].get("rows", []):
            conditional = row.get("profile_conditional_remask_cuda_elapsed_ms")
            total = row.get("profile_total_cuda_elapsed_ms")
            initial = row.get("profile_initial_parallel_draft_cuda_elapsed_ms")
            fallback = row.get("profile_fallback_cuda_elapsed_ms")
            if conditional is not None and total:
                conditional_total_ratios.append(
                    (float(conditional) + float(fallback or 0.0)) / float(total)
                )
            if conditional is not None and initial:
                conditional_initial_draft_ratios.append(
                    float(conditional) / float(initial)
                )

    stable_coverage = _stable_positive_by_task(coverage)
    stable_efficiency = _stable_positive_by_task(efficiency)
    conditional_cost_ratio_median = (
        statistics.median(conditional_total_ratios)
        if conditional_total_ratios
        else None
    )
    conditional_over_initial_draft_median = (
        statistics.median(conditional_initial_draft_ratios)
        if conditional_initial_draft_ratios
        else None
    )
    conditional_cost_acceptable = (
        conditional_cost_ratio_median <= 0.25
        if conditional_cost_ratio_median is not None
        else None
    )
    full_second_pass_too_heavy = (
        conditional_over_initial_draft_median >= 1.0
        if conditional_over_initial_draft_median is not None
        else None
    )
    official_checker_passed = all(
        row["correctness"]["official_checker_passed"] for row in records
    )
    structural_correctness_passed = all(
        row["correctness"]["structural_correctness_passed"] for row in records
    )
    decision = build_phase2d2_decision(
        official_checker_passed=official_checker_passed,
        structural_correctness_passed=structural_correctness_passed,
        guard_false_negative_count=guard_false_negative_count,
        independent_task_count=len(tasks),
        stable_coverage_gain=stable_coverage,
        stable_online_efficiency_gain=stable_efficiency,
        conditional_cost_acceptable=conditional_cost_acceptable,
        full_second_pass_too_heavy=full_second_pass_too_heavy,
    )
    decision["cross_task_aggregation"] = {
        "input_directories": [
            {
                "path": row["root"],
                "checker": row["checker"],
                "task": row["manifest"]["workload"]["task"],
                "node_budget": row["manifest"]["phase2d2"]["node_budget"],
                "repository_commit": row["manifest"]["repository"].get("commit_sha"),
            }
            for row in records
        ],
        "frozen_refinement_config_hash": next(iter(frozen_hashes)),
        "frozen_refinement_configuration_id": next(iter(family_ids)),
        "tasks": sorted(tasks),
        "node_budgets": sorted(budgets),
        "stability_rule": (
            "at least two tasks; every task mean gain > 0 and cross-task macro mean > 0"
        ),
        "conditional_cost_rule": (
            "median (conditional-remask + authoritative fallback) / profiled total <= 0.25"
        ),
        "full_second_pass_rule": (
            "median conditional-remask / initial parallel draft >= 1.0"
        ),
        "coverage_gain_by_task": dict(coverage),
        "efficiency_gain_by_task": dict(efficiency),
        "conditional_cost_ratio_median": conditional_cost_ratio_median,
        "conditional_over_initial_draft_median": (
            conditional_over_initial_draft_median
        ),
        "smoke_or_single_task_manual_promotion_allowed": False,
    }
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    decision = aggregate_result_directories(args.result_dir)
    if args.output.exists():
        raise ValueError(f"refusing to overwrite decision artifact: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
