#!/usr/bin/env python3
"""Run CPU-only Phase 2D.1A fixed-budget selector realization."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from statistics import fmean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsonschema import Draft202012Validator

from dflash.phase2b1.validation import validate_capture_directories
from dflash.phase2d1.aggregation import (
    add_context_censoring,
    build_decision,
    fixed_policy_summaries,
    policy_overlap,
    selector_summary_rows,
    source_cluster_bootstrap,
    winner_composition,
)
from dflash.phase2d1.events import build_selector_events
from dflash.phase2d1.preflight import validate_phase2d0_directory
from dflash.phase2d1.selectors import (
    EVALUATION_SCHEMES,
    SELECTOR_FAMILIES,
    build_cross_validation_folds,
    evaluate_policy_portfolios,
    evaluate_selectors,
)

PHASE2D0_BASELINE_SHA = "26b451dbe55737aeabdccb51871d562208694320"
SCHEMA_DIR = REPO_ROOT / "docs/schemas"
OUTPUT_FILENAMES = (
    "selector-events.jsonl",
    "fixed-policy-summary.json",
    "fixed-policy-summary.csv",
    "policy-winner-composition.csv",
    "policy-overlap.csv",
    "policy-portfolio-frontier.csv",
    "cross-validation-folds.json",
    "global-fixed-evaluation.csv",
    "context-rule-evaluation.csv",
    "decision-stump-evaluation.csv",
    "selector-headroom.csv",
    "selector-regret.csv",
    "source-cluster-bootstrap.csv",
    "context-selector-summary.csv",
    "research-question-summary.json",
    "selector-decision.json",
    "analysis-manifest.json",
    "terminal.log",
)


def _integer_csv(value: str, *, minimum: int) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or tuple(sorted(set(parsed))) != parsed or parsed[0] < minimum:
        raise argparse.ArgumentTypeError(
            f"values must be unique, increasing, and at least {minimum}"
        )
    return parsed


def _choices_csv(value: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(parsed) - set(allowed)
    if not parsed or unknown or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError(
            f"values must be unique members of {','.join(allowed)}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", action="append", required=True, type=Path)
    parser.add_argument("--phase2d0-dir", required=True, type=Path)
    parser.add_argument("--top-m", type=int, default=16)
    parser.add_argument(
        "--node-budgets",
        type=lambda value: _integer_csv(value, minimum=16),
        default=(16, 32),
    )
    parser.add_argument(
        "--evaluation",
        type=lambda value: _choices_csv(value, EVALUATION_SCHEMES),
        default=EVALUATION_SCHEMES,
    )
    parser.add_argument(
        "--selector-families",
        type=lambda value: _choices_csv(value, SELECTOR_FAMILIES),
        default=SELECTOR_FAMILIES,
    )
    parser.add_argument(
        "--portfolio-sizes",
        type=lambda value: _integer_csv(value, minimum=1),
        default=(1, 2, 3),
    )
    parser.add_argument("--minimum-leaf-support", type=int, default=50)
    parser.add_argument("--minimum-leaf-source-support", type=int, default=2)
    parser.add_argument("--stump-max-thresholds-per-feature", type=int, default=64)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20260818)
    parser.add_argument("--minimum-stable-clusters", type=int, default=8)
    parser.add_argument("--minimum-source-document-count", type=int, default=8)
    parser.add_argument("--minimum-task-count", type=int, default=2)
    parser.add_argument("--minimum-eligible-physical-rounds", type=int, default=500)
    parser.add_argument("--minimum-headroom-capture-ratio", type=float, default=0.25)
    parser.add_argument("--minimum-held-out-gain-over-top1", type=float, default=0.10)
    parser.add_argument(
        "--minimum-global-fixed-selection-stability", type=float, default=0.70
    )
    parser.add_argument(
        "--minimum-additional-gain-over-best-fixed", type=float, default=0.05
    )
    parser.add_argument(
        "--maximum-supported-stratum-regression", type=float, default=0.10
    )
    parser.add_argument("--minimum-supported-stratum-rounds", type=int, default=30)
    parser.add_argument(
        "--require-stable-clustered-ci",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--expected-phase2d0-decision", default="prioritize_selector")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase2d1-selector-realization"),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.top_m != 16:
        raise ValueError("Phase 2D.1A core analysis requires --top-m 16")
    if args.node_budgets != (16, 32):
        raise ValueError("Phase 2D.1A requires separate N=16 and N=32 strata")
    if set(args.evaluation) != set(EVALUATION_SCHEMES):
        raise ValueError("both source-out and task-out evaluation are required")
    if set(args.selector_families) != set(SELECTOR_FAMILIES):
        raise ValueError("all three selector families are required for RQ1-RQ7")
    if args.portfolio_sizes != (1, 2, 3):
        raise ValueError("Phase 2D.1A portfolio sizes must be 1,2,3")
    positive = (
        "minimum_leaf_support",
        "minimum_leaf_source_support",
        "stump_max_thresholds_per_feature",
        "bootstrap_repeats",
        "minimum_stable_clusters",
        "minimum_source_document_count",
        "minimum_task_count",
        "minimum_eligible_physical_rounds",
        "minimum_supported_stratum_rounds",
    )
    for field in positive:
        if getattr(args, field) < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    for field in (
        "minimum_headroom_capture_ratio",
        "minimum_held_out_gain_over_top1",
        "minimum_global_fixed_selection_stability",
        "minimum_additional_gain_over_best_fixed",
        "maximum_supported_stratum_regression",
    ):
        if getattr(args, field) < 0:
            raise ValueError(f"--{field.replace('_', '-')} must be non-negative")
    if args.minimum_global_fixed_selection_stability > 1:
        raise ValueError("--minimum-global-fixed-selection-stability must be <= 1")
    return args


def _git(*arguments: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@cache
def _validator(name: str) -> Draft202012Validator:
    schema = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate(value: Any, schema_name: str) -> None:
    _validator(schema_name).validate(value)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["schema_version"]
    fields.extend(
        sorted({key for row in rows for key in row if key != "schema_version"})
    )
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field), sort_keys=True)
                    if isinstance(row.get(field), (dict, list))
                    else row.get(field)
                    for field in fields
                }
            )
    temporary.replace(path)


class _Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _ensure_nonoverlap(args: argparse.Namespace) -> None:
    output = args.output_dir.expanduser().resolve()
    protected = [args.phase2d0_dir, *args.capture_dir]
    for raw in protected:
        path = raw.expanduser().resolve()
        if output == path or path in output.parents or output in path.parents:
            raise ValueError("output directory must not overlap immutable inputs")
    if output.exists():
        raise ValueError("Phase 2D.1A output directory already exists")


def _candidate_degradation_present(rq: dict[str, Any]) -> bool:
    trends = rq["rq1"]["task_specific_context_trends"]
    supported = {}
    for task in ("gov_report", "hotpotqa"):
        values = [
            row["availability_failure_fraction_change"]
            for row in trends
            if row["task"] == task
            and row["availability_failure_fraction_change"] is not None
        ]
        supported[task] = bool(values) and fmean(values) > 0
    return all(supported.values())


def _thresholds(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "minimum_source_document_count": args.minimum_source_document_count,
        "minimum_task_count": args.minimum_task_count,
        "minimum_eligible_physical_rounds": args.minimum_eligible_physical_rounds,
        "minimum_headroom_capture_ratio": args.minimum_headroom_capture_ratio,
        "minimum_held_out_gain_over_top1": args.minimum_held_out_gain_over_top1,
        "minimum_global_fixed_selection_stability": (
            args.minimum_global_fixed_selection_stability
        ),
        "minimum_additional_gain_over_best_fixed": (
            args.minimum_additional_gain_over_best_fixed
        ),
        "maximum_supported_stratum_regression": (
            args.maximum_supported_stratum_regression
        ),
        "minimum_supported_stratum_rounds": args.minimum_supported_stratum_rounds,
        "minimum_leaf_support": args.minimum_leaf_support,
        "minimum_leaf_source_support": args.minimum_leaf_source_support,
        "require_stable_clustered_ci": args.require_stable_clustered_ci,
        "threshold_semantics": "preregistered engineering go/no-go gates, not statistical significance theorems",
    }


def _manifest(
    args: argparse.Namespace,
    *,
    command: str,
    validated_capture: dict[str, Any],
    phase2d0: dict[str, Any],
    events: list[dict[str, Any]],
    folds: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    raw_physical = len(
        {
            (
                row["task"],
                row["sample_id"],
                row["nominal_prompt_length"],
                row["generation_repeat"],
                row["round_index"],
            )
            for row in validated_capture["rounds"]
        }
    )
    eligible_physical = len({row["physical_round_id"] for row in events})
    return {
        "schema_version": "1.0",
        "phase": "2D.1A",
        "status": "ok",
        "mode": "cpu_only_offline_fixed_budget_selector_realization",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase2d0_baseline_sha": PHASE2D0_BASELINE_SHA,
        "repository": {
            "commit_sha": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "command": command,
        "capture_inputs": validated_capture["inputs"],
        "phase2d0_input": {
            "directory": phase2d0["directory"],
            "directory_sha256": phase2d0["directory_sha256"],
            "artifact_sha256": phase2d0["artifact_sha256"],
            "decision": phase2d0["research_summary"]["decision"],
            "schema_version": phase2d0["manifest"]["schema_version"],
        },
        "configuration": {
            "top_m": args.top_m,
            "node_budgets": list(args.node_budgets),
            "node_budget_includes_root": True,
            "evaluation": list(args.evaluation),
            "selector_families": list(args.selector_families),
            "portfolio_sizes": list(args.portfolio_sizes),
            "minimum_leaf_support": args.minimum_leaf_support,
            "minimum_leaf_source_support": args.minimum_leaf_source_support,
            "stump_max_thresholds_per_feature": (args.stump_max_thresholds_per_feature),
            "bootstrap_repeats": args.bootstrap_repeats,
            "random_seed": args.random_seed,
            "minimum_stable_clusters": args.minimum_stable_clusters,
        },
        "decision_rules": decision["decision_rules"],
        "fold_semantics": {
            "primary": "leave-one-source-document-out",
            "supplemental": "leave-one-task-out",
            "round_level_random_split_used": False,
            "same_source_all_contexts_and_rounds_in_one_fold": True,
            "source_leakage_detected": False,
            "source_cluster_count": folds["source_cluster_count"],
        },
        "count_semantics": {
            "raw_physical_round_count": raw_physical,
            "eligible_physical_round_count": eligible_physical,
            "eligible_configuration_round_count_over_fixed_node_budgets": len(events),
            "n16_and_n32_are_separate_configuration_records": True,
            "configuration_records_are_not_independent_physical_rounds": True,
        },
        "framework_modules_loaded": {
            "torch": "torch" in sys.modules,
            "transformers": "transformers" in sys.modules,
            "datasets": "datasets" in sys.modules,
        },
        "input_artifacts_mutated": False,
        "instrumented_custom_code_modified": False,
        "materialization_required": False,
        "formal_timing_contaminated": False,
        "decision": decision["decision"],
        "artifacts": list(OUTPUT_FILENAMES),
        "forbidden_interpretations": [
            "actual accepted tokens",
            "round reduction",
            "online latency improvement",
            "hardware speedup",
            "portfolio oracle as a deployable selector",
            "policy oracle as a target-blind online selector",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(effective_argv)
    _ensure_nonoverlap(args)

    validated_capture = validate_capture_directories(
        args.capture_dir, schema_dir=SCHEMA_DIR
    )
    phase2d0 = validate_phase2d0_directory(
        args.phase2d0_dir,
        validated_capture=validated_capture,
        schema_dir=SCHEMA_DIR,
        top_m=args.top_m,
        node_budgets=args.node_budgets,
    )
    if phase2d0["research_summary"]["decision"] != args.expected_phase2d0_decision:
        raise ValueError(
            "Phase 2D.0 decision mismatch: expected "
            f"{args.expected_phase2d0_decision!r}, got "
            f"{phase2d0['research_summary']['decision']!r}"
        )
    events = build_selector_events(phase2d0["events"])
    folds = build_cross_validation_folds(events, args.evaluation)
    predictions = evaluate_selectors(
        events,
        folds,
        selector_families=args.selector_families,
        minimum_leaf_support=args.minimum_leaf_support,
        minimum_leaf_source_support=args.minimum_leaf_source_support,
        maximum_thresholds_per_feature=args.stump_max_thresholds_per_feature,
    )
    all_predictions = [row for rows in predictions.values() for row in rows]
    portfolios = evaluate_policy_portfolios(
        events, folds, portfolio_sizes=args.portfolio_sizes
    )
    fixed = fixed_policy_summaries(events)
    winners = winner_composition(events)
    overlaps = policy_overlap(events)
    selector_rows = selector_summary_rows(events, predictions)
    bootstrap_rows = source_cluster_bootstrap(
        all_predictions,
        repeats=args.bootstrap_repeats,
        seed=args.random_seed,
        minimum_stable_clusters=args.minimum_stable_clusters,
    )
    context_rows = add_context_censoring(selector_rows, phase2d0["gap"])
    physical_count = len({event["physical_round_id"] for event in events})
    decision, research = build_decision(
        selector_rows,
        bootstrap_rows,
        all_predictions,
        winners,
        portfolios,
        thresholds=_thresholds(args),
        source_count=folds["source_cluster_count"],
        task_count=folds["task_count"],
        eligible_physical_round_count=physical_count,
        candidate_availability_degradation_present=_candidate_degradation_present(
            phase2d0["research_summary"]
        ),
    )
    command = shlex.join(
        [sys.executable, str(Path(__file__).resolve()), *effective_argv]
    )
    manifest = _manifest(
        args,
        command=command,
        validated_capture=validated_capture,
        phase2d0=phase2d0,
        events=events,
        folds=folds,
        decision=decision,
    )

    for event in events:
        _validate(event, "phase2d1-selector-event.schema.json")
    _validate(fixed, "phase2d1-fixed-policy-summary.schema.json")
    _validate(folds, "phase2d1-cross-validation-folds.schema.json")
    _validate(research, "phase2d1-research-question-summary.schema.json")
    _validate(decision, "phase2d1-selector-decision.schema.json")
    _validate(manifest, "phase2d1-analysis-manifest.schema.json")
    if any(manifest["framework_modules_loaded"].values()):
        raise RuntimeError("Phase 2D.1A unexpectedly loaded a model framework")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with (output_dir / "terminal.log").open("w", encoding="utf-8") as log:
        sys.stdout = _Tee(original_stdout, log)
        sys.stderr = _Tee(original_stderr, log)
        try:
            _write_jsonl(output_dir / "selector-events.jsonl", events)
            _write_json(output_dir / "fixed-policy-summary.json", fixed)
            _write_csv(output_dir / "fixed-policy-summary.csv", fixed["rows"])
            _write_csv(output_dir / "policy-winner-composition.csv", winners)
            _write_csv(output_dir / "policy-overlap.csv", overlaps)
            _write_csv(output_dir / "policy-portfolio-frontier.csv", portfolios)
            _write_json(output_dir / "cross-validation-folds.json", folds)
            _write_csv(
                output_dir / "global-fixed-evaluation.csv",
                predictions["global_fixed"],
            )
            _write_csv(
                output_dir / "context-rule-evaluation.csv",
                predictions["context_rule"],
            )
            _write_csv(
                output_dir / "decision-stump-evaluation.csv",
                predictions["decision_stump"],
            )
            _write_csv(output_dir / "selector-headroom.csv", selector_rows)
            _write_csv(output_dir / "selector-regret.csv", selector_rows)
            _write_csv(output_dir / "source-cluster-bootstrap.csv", bootstrap_rows)
            _write_csv(output_dir / "context-selector-summary.csv", context_rows)
            _write_json(output_dir / "research-question-summary.json", research)
            _write_json(output_dir / "selector-decision.json", decision)
            _write_json(output_dir / "analysis-manifest.json", manifest)
            print(
                f"Validated {len(args.capture_dir)} immutable captures and "
                f"{phase2d0['directory']} before creating output."
            )
            print(
                f"Analyzed {physical_count} eligible physical rounds separately "
                f"at N={list(args.node_budgets)} into {len(events)} "
                "configuration-round records."
            )
            print(f"Phase 2D.1A decision: {decision['decision']}")
            print(
                "CPU-only offline coverage analysis complete; no model framework, "
                "CUDA, model, or dataset was loaded."
            )
            return 0
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
