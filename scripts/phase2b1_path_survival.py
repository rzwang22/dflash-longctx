#!/usr/bin/env python3
"""Analyze target-path survival from immutable Phase 2B captures on CPU."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsonschema import Draft202012Validator

from dflash.phase2b1.analysis import (
    aggregate_events,
    analyze_capture_rounds,
    budget_frontier_rows,
    long_context_rows,
    round_event_rows,
)
from dflash.phase2b1.validation import validate_capture_directories

SCHEMA_DIR = REPO_ROOT / "docs/schemas"
OUTPUT_FILENAMES = (
    "path-survival-events.jsonl",
    "path-survival-summary.json",
    "path-survival-summary.csv",
    "depth-survival.csv",
    "pruning-reasons.csv",
    "minimum-budget.csv",
    "confidence-diagnostics.json",
    "confidence-diagnostics.csv",
    "budget-frontier.csv",
    "long-context-summary.csv",
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


def _budgets(value: str) -> tuple[int, ...]:
    return _integer_csv(value, minimum=16)


def _widths(value: str) -> tuple[int, ...]:
    return _integer_csv(value, minimum=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-dir",
        action="append",
        type=Path,
        required=True,
        help="Immutable Phase 2B capture directory; repeat for multiple tasks.",
    )
    parser.add_argument("--node-budgets", type=_budgets, default=(16, 32, 64, 128, 256))
    parser.add_argument("--beam-widths", type=_widths, default=(2, 4, 8, 16, 32))
    parser.add_argument("--rank-cap", type=int, default=100_000)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/phase2b1-path-survival")
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.rank_cap < 1:
        raise ValueError("--rank-cap must be positive")
    if 16 not in args.node_budgets:
        raise ValueError("--node-budgets must include the root-inclusive budget 16")
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
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def _schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def _validate_output(instance: Any, schema_name: str) -> None:
    schema = _schema(schema_name)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(instance)


def _summary_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for group in summary["groups"]:
        rows.append(
            {
                "task": group["task"],
                "context_length": group["context_length"],
                "policy": group["policy"],
                "policy_variant": group["policy_variant"],
                "policy_class": group["policy_class"],
                "node_budget": group["requested_total_tree_node_budget"],
                "sample_count": group["sample_count"],
                "round_count": group["round_count"],
                "full_depth_eligible_round_count": group[
                    "full_depth_eligible_round_count"
                ],
                "right_censored_round_count": group["right_censored_round_count"],
                "covered_depth_mean": group["covered_target_prefix_depth"]["mean"],
                "covered_depth_median": group["covered_target_prefix_depth"]["median"],
                "covered_depth_p90": group["covered_target_prefix_depth"]["p90"],
                "baseline_accepted_depth_mean": group["baseline_accepted_draft_depth"][
                    "mean"
                ],
                "observable_target_horizon_mean": group["observable_target_horizon"][
                    "mean"
                ],
                "absolute_extra_covered_tokens": group[
                    "extra_covered_target_tokens_vs_top1"
                ]["sum"],
                "extra_depth_mean": group["extra_covered_target_tokens_vs_top1"][
                    "mean"
                ],
                "extra_depth_median": group["extra_covered_target_tokens_vs_top1"][
                    "median"
                ],
                "extra_depth_p90": group["extra_covered_target_tokens_vs_top1"]["p90"],
                "positive_improvement_fraction": group["round_micro"][
                    "positive_improvement_fraction"
                ],
                "first_rejection_recovery_fraction": group["round_micro"][
                    "first_rejection_recovery_fraction"
                ],
                "candidate_unavailable_fraction": group["round_micro"][
                    "candidate_unavailable_fraction"
                ],
                "candidate_available_but_pruned_fraction": group["round_micro"][
                    "candidate_available_but_pruned_fraction"
                ],
                "score_rank_censored_fraction": group["round_micro"][
                    "score_rank_censored_fraction"
                ],
                "first_candidate_unavailable_depth_mean": group[
                    "first_candidate_unavailable_depth"
                ]["mean"],
                "first_pruned_depth_mean": group["first_pruned_depth"]["mean"],
                "off_top1_decision_count_mean": group["off_top1_decision_count"][
                    "mean"
                ],
                "longest_consecutive_off_top1_span_mean": group[
                    "longest_consecutive_off_top1_span"
                ]["mean"],
                "target_rank_mean_across_rounds": group["target_rank_mean_per_round"][
                    "mean"
                ],
                "target_rank_maximum_p90": group["target_rank_maximum_per_round"][
                    "p90"
                ],
                "target_rank_recall_at_2_depth_micro": group["target_rank_recall"][
                    "at_2"
                ]["depth_micro"],
                "target_rank_recall_at_4_depth_micro": group["target_rank_recall"][
                    "at_4"
                ]["depth_micro"],
                "target_rank_recall_at_8_depth_micro": group["target_rank_recall"][
                    "at_8"
                ]["depth_micro"],
                "target_rank_recall_at_16_depth_micro": group["target_rank_recall"][
                    "at_16"
                ]["depth_micro"],
                "minimum_survival_budget_at_observable_horizon_median": group[
                    "minimum_survival_budget_at_observable_horizon"
                ]["median"],
                "minimum_survival_budget_censored_count": group[
                    "minimum_survival_budget_at_observable_horizon"
                ]["censored_above_maximum_count"],
                "sample_macro_covered_depth_mean": group["sample_macro"][
                    "covered_target_prefix_depth_mean"
                ],
                "sample_macro_extra_depth_mean": group["sample_macro"][
                    "extra_covered_target_tokens_vs_top1_mean"
                ],
                "sample_macro_positive_improvement_fraction": group["sample_macro"][
                    "positive_improvement_fraction"
                ],
                "sample_macro_first_rejection_recovery_fraction": group["sample_macro"][
                    "first_rejection_recovery_fraction"
                ],
                "oracle_improvable_round_count": group["focus_subset"][
                    "oracle_improvable_round_count"
                ],
                "oracle_improvable_but_policy_no_improvement_count": group[
                    "focus_subset"
                ]["oracle_improvable_but_policy_no_improvement_count"],
            }
        )
    return rows


def _depth_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        for depth in event["depth_events"]:
            grouped[
                (
                    event["task"],
                    event["nominal_prompt_length"],
                    event["policy"],
                    event["policy_variant"],
                    event["requested_total_tree_node_budget"],
                    depth["depth"],
                )
            ].append(depth)
    rows = []
    for key, values in sorted(grouped.items()):
        task, context, policy, variant, budget, depth = key
        rows.append(
            {
                "task": task,
                "context_length": context,
                "policy": policy,
                "policy_variant": variant,
                "node_budget": budget,
                "depth": depth,
                "eligible_round_count": len(values),
                "target_token_available_fraction": mean(
                    value["target_token_in_top_k"] for value in values
                ),
                "target_prefix_survival_fraction": mean(
                    value["target_prefix_survived"] for value in values
                ),
                "score_rank_censored_fraction": mean(
                    value["target_prefix_score_rank_censored"] for value in values
                ),
            }
        )
    return rows


def _pruning_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[Any, ...], int] = defaultdict(int)
    totals: dict[tuple[Any, ...], int] = defaultdict(int)
    for event in events:
        base = (
            event["task"],
            event["nominal_prompt_length"],
            event["policy"],
            event["policy_variant"],
            event["requested_total_tree_node_budget"],
        )
        for depth in event["depth_events"]:
            totals[base] += 1
            counts[(*base, depth["pruning_reason"])] += 1
    return [
        {
            "task": key[0],
            "context_length": key[1],
            "policy": key[2],
            "policy_variant": key[3],
            "node_budget": key[4],
            "pruning_reason": key[5],
            "depth_event_count": count,
            "depth_event_fraction": count / totals[key[:5]],
        }
        for key, count in sorted(counts.items())
    ]


def _minimum_budget_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for event in events:
        identity = (
            event["task"],
            event["nominal_prompt_length"],
            event["sample_id"],
            event["generation_repeat"],
            event["round_index"],
            event["policy_variant"],
        )
        if identity in seen:
            continue
        seen.add(identity)
        for depth in event["depth_events"]:
            rows.append(
                {
                    "task": event["task"],
                    "context_length": event["nominal_prompt_length"],
                    "sample_id": event["sample_id"],
                    "generation_repeat": event["generation_repeat"],
                    "round_index": event["round_index"],
                    "policy": event["policy"],
                    "policy_variant": event["policy_variant"],
                    "depth": depth["depth"],
                    "minimum_total_tree_node_budget_to_preserve_target_prefix": depth[
                        "minimum_total_tree_node_budget_to_preserve_target_prefix"
                    ],
                    "minimum_budget_label": depth["minimum_budget_label"],
                    "minimum_budget_censored": depth["minimum_budget_censored"],
                    "root_included_in_budget": True,
                }
            )
    return rows


def _confidence_csv_rows(confidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in row.items()
            if key != "calibration_by_risk_decile"
        }
        | {"calibration_by_risk_decile_json": row["calibration_by_risk_decile"]}
        for row in confidence["results"]
    ]


def _manifest(
    args: argparse.Namespace,
    inputs: list[dict[str, Any]],
    *,
    command: str,
    event_count: int,
    policy_replay_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "phase": "2B.1",
        "status": "ok",
        "mode": "cpu_only_offline_target_path_survival_analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "repository": {
            "commit_sha": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "inputs": inputs,
        "configuration": {
            "captured_top_k_minimum": 16,
            "rank_cap": args.rank_cap,
            "rank_cap_semantics": "if exact rank exceeds cap, exact rank is null and lower bound is cap+1",
            "node_budgets": list(args.node_budgets),
            "beam_widths": list(args.beam_widths),
            "confidence_margin_thresholds": [0.25, 0.5, 1.0],
            "root_included_in_total_node_budget": True,
            "maximum_draft_depth": 15,
            "scoring_rule": "mean cumulative captured draft log_probability; at equal depth equivalent to cumulative score",
            "tie_breaking": "higher normalized score, higher cumulative score, shorter depth, lexicographically ascending token IDs",
            "censoring_rule": "reference depth below 15 is right-censored, must end at the final verified output boundary, does not imply a terminal round, and is excluded from unobserved depth checkpoints",
        },
        "policy_classes": {
            "top1_chain": {"deployable": True, "uses_target": False},
            "score_beam": {"deployable": True, "uses_target": False},
            "confidence_gated": {"deployable": True, "uses_target": False},
            "layerwise_beam": {
                "deployable": True,
                "uses_target": False,
                "diagnostic_only": True,
            },
            "oracle_prefix_closed": {
                "deployable": False,
                "uses_target": True,
                "role": "structural upper bound only",
            },
        },
        "target_information_usage": {
            "used_for": [
                "post-hoc membership and pruning labels",
                "target-prefix score-rank query",
                "minimum survival budget evaluation after target-blind replay",
                "oracle upper bound",
                "post-hoc target-vs-Top1 score gap",
            ],
            "not_used_for": [
                "deployable tree construction",
                "draft-confidence predictor features",
                "policy tuning or training",
            ],
        },
        "forbidden_interpretations": [
            "actual tree accepted length",
            "actual tree verification latency",
            "end-to-end speedup",
            "verification round reduction",
            "tree mask or KV sharing benefit",
            "sequential refinement effect",
            "learned selector effect",
        ],
        "event_count": event_count,
        "policy_replay_count": policy_replay_count,
        "artifacts": list(OUTPUT_FILENAMES),
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


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(effective_argv)
    # Strict preflight is deliberately before output-dir creation.
    validated = validate_capture_directories(args.capture_dir, schema_dir=SCHEMA_DIR)
    analysis = analyze_capture_rounds(
        validated["rounds"],
        node_budgets=args.node_budgets,
        beam_widths=args.beam_widths,
        rank_cap=args.rank_cap,
    )
    summary = aggregate_events(analysis["events"])
    round_events = round_event_rows(analysis["events"])
    frontier = budget_frontier_rows(summary)
    long_summary = long_context_rows(summary)
    command = shlex.join(
        [sys.executable, str(Path(__file__).resolve()), *effective_argv]
    )
    manifest = _manifest(
        args,
        validated["inputs"],
        command=command,
        event_count=len(round_events),
        policy_replay_count=len(analysis["events"]),
    )

    for event in round_events:
        _validate_output(event, "phase2b1-path-survival-event.schema.json")
    _validate_output(summary, "phase2b1-path-survival-summary.schema.json")
    _validate_output(
        analysis["confidence"], "phase2b1-confidence-diagnostics.schema.json"
    )
    _validate_output(manifest, "phase2b1-analysis-manifest.schema.json")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with (output_dir / "terminal.log").open("w", encoding="utf-8") as log:
        sys.stdout = _Tee(original_stdout, log)
        sys.stderr = _Tee(original_stderr, log)
        try:
            _write_jsonl(output_dir / "path-survival-events.jsonl", round_events)
            _write_json(output_dir / "path-survival-summary.json", summary)
            _write_csv(
                output_dir / "path-survival-summary.csv", _summary_csv_rows(summary)
            )
            _write_csv(
                output_dir / "depth-survival.csv", _depth_rows(analysis["events"])
            )
            _write_csv(
                output_dir / "pruning-reasons.csv", _pruning_rows(analysis["events"])
            )
            _write_csv(
                output_dir / "minimum-budget.csv",
                _minimum_budget_rows(analysis["events"]),
            )
            _write_json(
                output_dir / "confidence-diagnostics.json", analysis["confidence"]
            )
            _write_csv(
                output_dir / "confidence-diagnostics.csv",
                _confidence_csv_rows(analysis["confidence"]),
            )
            _write_csv(output_dir / "budget-frontier.csv", frontier)
            _write_csv(output_dir / "long-context-summary.csv", long_summary)
            _write_json(output_dir / "analysis-manifest.json", manifest)
            print(
                f"Validated {len(validated['inputs'])} capture directories and "
                f"analyzed {len(validated['rounds'])} rounds into "
                f"{len(analysis['events'])} policy-budget events."
            )
            print("CPU-only analysis complete; no model framework or CUDA was loaded.")
            return 0
        except Exception as exc:  # noqa: BLE001 - keep a useful terminal log.
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
            return 1
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
