#!/usr/bin/env python3
"""Run CPU-only Phase 2D.0 dependency-opportunity characterization."""

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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsonschema import Draft202012Validator

from dflash.phase2b1.validation import validate_capture_directories
from dflash.phase2d0.aggregation import (
    aggregate_frontier_rows,
    aggregate_gap_events,
    availability_by_depth,
    build_research_summary,
    dependency_rows,
)
from dflash.phase2d0.analysis import (
    DEFAULT_BEAM_WIDTHS,
    DEFAULT_CONFIDENCE_THRESHOLDS,
    GAP_CLASS_PRIORITY,
    SUPPORTED_POLICIES,
    analyze_captures,
)
from dflash.phase2d0.signals import (
    build_signal_rows,
    evaluate_signal_hypotheses,
)

BASELINE_SHA = "1411c27e086cacd62425f329dcf7432cedd74654"
SCHEMA_DIR = REPO_ROOT / "docs/schemas"
OUTPUT_FILENAMES = (
    "opportunity-events.jsonl",
    "gap-decomposition.json",
    "gap-decomposition.csv",
    "context-gap-composition.csv",
    "availability-by-depth.csv",
    "static-survival-frontier.csv",
    "minimum-survival-budget.csv",
    "dependency-opportunity.csv",
    "signal-labels.jsonl",
    "signal-hypotheses.csv",
    "research-question-summary.json",
    "analysis-manifest.json",
    "terminal.log",
)


def _integer_csv(
    value: str, *, minimum: int, maximum: int | None = None
) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or tuple(sorted(set(parsed))) != parsed or parsed[0] < minimum:
        raise argparse.ArgumentTypeError(
            f"values must be unique, increasing, and at least {minimum}"
        )
    if maximum is not None and parsed[-1] > maximum:
        raise argparse.ArgumentTypeError(f"values must not exceed {maximum}")
    return parsed


def _top_ms(value: str) -> tuple[int, ...]:
    parsed = _integer_csv(value, minimum=2, maximum=16)
    if not set(parsed) <= {2, 4, 8, 16}:
        raise argparse.ArgumentTypeError("Top-M values must come from 2,4,8,16")
    return parsed


def _node_budgets(value: str) -> tuple[int, ...]:
    return _integer_csv(value, minimum=16)


def _policies(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(parsed) - set(SUPPORTED_POLICIES)
    if not parsed or unknown:
        raise argparse.ArgumentTypeError(f"unsupported policies: {sorted(unknown)}")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("policies must not be repeated")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-dir",
        action="append",
        type=Path,
        required=True,
        help="Immutable Phase 2B capture directory; repeat for each task.",
    )
    parser.add_argument("--top-ms", type=_top_ms, default=(2, 4, 8, 16))
    parser.add_argument("--node-budgets", type=_node_budgets, default=(16, 32))
    parser.add_argument("--survival-budget-cap", type=int, default=256)
    parser.add_argument("--policies", type=_policies, default=SUPPORTED_POLICIES)
    parser.add_argument("--rank-cap", type=int, default=100_000)
    parser.add_argument("--budget-failure-multiplier", type=float, default=2.0)
    parser.add_argument("--rank-inflation-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase2d0-dependency-opportunity"),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if 16 not in args.node_budgets or 32 not in args.node_budgets:
        raise ValueError("--node-budgets must include core budgets 16 and 32")
    if args.survival_budget_cap < max(args.node_budgets):
        raise ValueError("--survival-budget-cap must cover all core node budgets")
    if args.rank_cap < 1:
        raise ValueError("--rank-cap must be positive")
    if args.budget_failure_multiplier <= 1:
        raise ValueError("--budget-failure-multiplier must be greater than one")
    if args.rank_inflation_multiplier <= 1:
        raise ValueError("--rank-inflation-multiplier must be greater than one")
    if "top1_chain" not in args.policies:
        raise ValueError("--policies must include top1_chain")
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


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
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


GAP_FIELDS = (
    "schema_version",
    "scope",
    "task",
    "context_length",
    "top_m",
    "node_budget",
    "gap_class",
    "gap_class_round_count",
    "round_count",
    "eligible_round_count",
    "terminal_or_censored_excluded_count",
    "task_count",
    "fraction",
    "micro_fraction",
    "macro_fraction",
    "fraction_denominator",
    "eligible_fraction_denominator",
)
CONTEXT_GAP_FIELDS = (
    "schema_version",
    "task",
    "context_length",
    "top_m",
    "node_budget",
    "gap_class",
    "gap_class_round_count",
    "round_count",
    "eligible_round_count",
    "terminal_or_censored_excluded_count",
    "fraction",
    "micro_fraction",
    "macro_fraction",
)
AVAILABILITY_FIELDS = (
    "schema_version",
    "task",
    "context_length",
    "top_m",
    "depth",
    "observable_round_count",
    "target_token_available_count",
    "target_token_available_fraction",
    "target_prefix_available_count",
    "target_prefix_available_fraction",
    "right_censored_round_count",
)
FRONTIER_FIELDS = (
    "schema_version",
    "task",
    "context_length",
    "top_m",
    "policy",
    "policy_variant",
    "node_budget",
    "round_count",
    "mean_verified_nodes",
    "mean_coverage_depth",
    "coverage_depth_per_verified_node",
    "mean_coverage_gain_per_added_node",
    "mean_coverage_gain_per_added_node_unavailable_reason",
    "interpretation",
)
MINIMUM_FIELDS = (
    "schema_version",
    "task",
    "sample_id",
    "source_sample_id",
    "nominal_prompt_length",
    "generation_repeat",
    "round_index",
    "top_m",
    "core_node_budget",
    "policy",
    "policy_variant",
    "target_depth",
    "minimum_survival_budget",
    "minimum_survival_budget_unavailable_reason",
    "minimum_budget_censored",
    "survival_budget_cap",
    "root_included_in_budget",
)
DEPENDENCY_FIELDS = (
    "schema_version",
    "task",
    "sample_id",
    "source_sample_id",
    "nominal_prompt_length",
    "generation_repeat",
    "round_index",
    "top_m",
    "node_budget",
    "gap_class",
    "critical_depth",
    "candidate_availability_prefix",
    "best_static_coverage_depth",
    "oracle_coverage_depth",
    "oracle_minus_static_coverage",
    "oracle_required_nodes",
    "oracle_required_nodes_unavailable_reason",
    "best_static_required_nodes",
    "best_static_required_nodes_unavailable_reason",
    "static_oracle_budget_ratio",
    "static_oracle_budget_ratio_unavailable_reason",
    "target_prefix_score_rank",
    "target_prefix_score_rank_lower_bound",
    "target_prefix_score_rank_censored",
    "first_pruning_depth",
    "first_pruning_depth_unavailable_reason",
    "shallow_target_prefix_entered_tree",
    "correct_path_pruned_at_depth",
    "correct_path_pruned_at_depth_unavailable_reason",
    "significant_rank_inflation",
    "significant_budget_inflation",
)
SIGNAL_FIELDS = (
    "schema_version",
    "scope",
    "group",
    "label",
    "feature",
    "single_feature_only",
    "risk_score_direction",
    "observation_count",
    "sample_count",
    "positive_count",
    "negative_count",
    "positive_rate",
    "auroc",
    "auprc",
    "unavailable_reason",
    "calibration_or_binned_precision",
    "result_unstable",
    "instability_reasons",
    "training_performed",
    "refinement_gain_predicted",
)


def _manifest(
    args: argparse.Namespace,
    *,
    inputs: list[dict[str, Any]],
    command: str,
    event_count: int,
    raw_round_count: int,
    rq_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "phase": "2D.0",
        "status": "ok",
        "mode": "cpu_only_offline_dependency_opportunity_characterization",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_commit_sha": BASELINE_SHA,
        "command": command,
        "repository": {
            "commit_sha": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "inputs": inputs,
        "capture_preflight_completed_before_output_creation": True,
        "input_capture_mutated": False,
        "configuration": {
            "block_size": 16,
            "draft_candidate_positions": 15,
            "captured_top_k_minimum": 16,
            "batch_size": 1,
            "temperature": 0,
            "score_semantics": "normalized log probability from log_softmax",
            "top_ms": list(args.top_ms),
            "node_budgets": list(args.node_budgets),
            "survival_budget_cap": args.survival_budget_cap,
            "policies": list(args.policies),
            "confidence_thresholds": list(DEFAULT_CONFIDENCE_THRESHOLDS),
            "beam_widths": list(DEFAULT_BEAM_WIDTHS),
            "rank_cap": args.rank_cap,
            "budget_failure_multiplier": args.budget_failure_multiplier,
            "rank_inflation_multiplier": args.rank_inflation_multiplier,
            "root_included_in_node_budget": True,
        },
        "classification_priority": list(GAP_CLASS_PRIORITY),
        "classification_rules": {
            "right_censored_or_terminal": "terminal_round or reference_right_censored",
            "no_failure_within_observed_horizon": "a0 reaches the complete observed target horizon",
            "candidate_availability_failure": "the target token at a0+1 is absent from positional Top-M",
            "selection_failure": "another tested target-blind static policy covers a0+1 at the same root-inclusive budget",
            "budget_failure": "all tested policies fail at the core budget but one covers a0+1 by at most budget_failure_multiplier times that budget",
            "dependency_realization_opportunity": "candidate and oracle cover a0+1, all tested static policies fail, modest budget does not resolve it, and score-rank or required-budget inflation crosses the recorded threshold",
            "possible_dependency_opportunity": "candidate and oracle cover a0+1 and all tested policies fail, but conservative inflation evidence is incomplete",
            "unresolved": "no preceding mutually exclusive condition applies",
        },
        "target_information_usage": {
            "deployable_features": "captured draft-side values and static structure only",
            "offline_labels_only": [
                "target token rank and membership",
                "accepted draft depth",
                "oracle path and coverage",
                "gap class",
            ],
            "target_used_by_static_policy": False,
            "target_used_by_oracle": True,
        },
        "signal_semantics": {
            "top_m_entropy": "renormalized entropy of captured Top-M mass, not full-vocabulary entropy",
            "training_performed": False,
            "refinement_gain_label_defined": False,
            "early_critical_depth_threshold": 4,
        },
        "decision_rules": rq_summary["rq6"]["decision_rules"],
        "decision": rq_summary["decision"],
        "framework_modules_loaded": {
            "torch": "torch" in sys.modules,
            "transformers": "transformers" in sys.modules,
            "datasets": "datasets" in sys.modules,
        },
        "formal_timing_contaminated": False,
        "instrumented_custom_code_modified": False,
        "materialization_required": False,
        "event_count": event_count,
        "raw_round_count": raw_round_count,
        "artifacts": list(OUTPUT_FILENAMES),
        "forbidden_interpretations": [
            "actual online acceptance",
            "round reduction",
            "latency or speedup",
            "hardware efficiency",
            "dependency gap dominance before real capture analysis",
            "refinement gain or causal-refiner effectiveness",
        ],
    }


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


def _ensure_output_is_separate(output_dir: Path, capture_dirs: list[Path]) -> None:
    output = output_dir.expanduser().resolve()
    for raw_capture in capture_dirs:
        capture = raw_capture.expanduser().resolve()
        if output == capture or capture in output.parents or output in capture.parents:
            raise ValueError(
                "--output-dir must be outside every immutable capture directory"
            )


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(effective_argv)
    _ensure_output_is_separate(args.output_dir, args.capture_dir)

    # Full capture validation and all analysis happen before output creation.
    validated = validate_capture_directories(args.capture_dir, schema_dir=SCHEMA_DIR)
    analysis = analyze_captures(
        validated["rounds"],
        top_ms=args.top_ms,
        node_budgets=args.node_budgets,
        survival_budget_cap=args.survival_budget_cap,
        policies=args.policies,
        rank_cap=args.rank_cap,
        budget_failure_multiplier=args.budget_failure_multiplier,
        rank_inflation_multiplier=args.rank_inflation_multiplier,
    )
    events = analysis["events"]
    gap = aggregate_gap_events(events)
    context_rows = gap["task_context"]
    gap_rows = [*context_rows, *gap["task_level"], *gap["cross_task_macro"]]
    availability_rows = availability_by_depth(events)
    frontier_rows = aggregate_frontier_rows(analysis["frontier_rows"])
    minimum_rows = analysis["minimum_budget_rows"]
    dependency = dependency_rows(events)
    signal_rows = build_signal_rows(events)
    signal_results = evaluate_signal_hypotheses(signal_rows)
    rq_summary = build_research_summary(events, signal_results)
    command = shlex.join(
        [sys.executable, str(Path(__file__).resolve()), *effective_argv]
    )
    manifest = _manifest(
        args,
        inputs=validated["inputs"],
        command=command,
        event_count=len(events),
        raw_round_count=gap["raw_round_count"],
        rq_summary=rq_summary,
    )

    for event in events:
        _validate(event, "phase2d0-opportunity-event.schema.json")
    for row in signal_rows:
        _validate(row, "phase2d0-signal-label.schema.json")
    _validate(gap, "phase2d0-gap-decomposition.schema.json")
    _validate(rq_summary, "phase2d0-research-question-summary.schema.json")
    _validate(manifest, "phase2d0-analysis-manifest.schema.json")
    if any(manifest["framework_modules_loaded"].values()):
        raise RuntimeError("CPU-only analysis unexpectedly loaded a model framework")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with (output_dir / "terminal.log").open("w", encoding="utf-8") as log:
        sys.stdout = _Tee(original_stdout, log)
        sys.stderr = _Tee(original_stderr, log)
        try:
            _write_jsonl(output_dir / "opportunity-events.jsonl", events)
            _write_json(output_dir / "gap-decomposition.json", gap)
            _write_csv(output_dir / "gap-decomposition.csv", gap_rows, GAP_FIELDS)
            _write_csv(
                output_dir / "context-gap-composition.csv",
                context_rows,
                CONTEXT_GAP_FIELDS,
            )
            _write_csv(
                output_dir / "availability-by-depth.csv",
                availability_rows,
                AVAILABILITY_FIELDS,
            )
            _write_csv(
                output_dir / "static-survival-frontier.csv",
                frontier_rows,
                FRONTIER_FIELDS,
            )
            _write_csv(
                output_dir / "minimum-survival-budget.csv",
                minimum_rows,
                MINIMUM_FIELDS,
            )
            _write_csv(
                output_dir / "dependency-opportunity.csv",
                dependency,
                DEPENDENCY_FIELDS,
            )
            _write_jsonl(output_dir / "signal-labels.jsonl", signal_rows)
            _write_csv(
                output_dir / "signal-hypotheses.csv",
                signal_results,
                SIGNAL_FIELDS,
            )
            _write_json(output_dir / "research-question-summary.json", rq_summary)
            _write_json(output_dir / "analysis-manifest.json", manifest)
            print(
                f"Validated {len(validated['inputs'])} immutable capture directories "
                f"and analyzed {gap['raw_round_count']} rounds into {len(events)} "
                "configuration events."
            )
            print(f"Deterministic Phase 2D.1 decision: {rq_summary['decision']}")
            print(
                "CPU-only analysis complete; Torch, Transformers, datasets, CUDA, and models were not loaded."
            )
            return 0
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
