#!/usr/bin/env python3
"""Replay bounded candidate-tree policies from a Phase 2B capture on CPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase2b.capture import validate_capture_round
from dflash.phase2b.metrics import DEPTHS, aggregate_tree_events
from dflash.phase2b.tree import (
    confidence_gated_tree,
    oracle_prefix_closed_tree,
    policy_event,
    score_beam_tree,
    top1_chain_tree,
)
from experiments.phase0_smoke import TeeStream
from experiments.phase1_observability import _atomic_write_jsonl
from scripts.collect_environment import write_json

POLICIES = ("top1_chain", "score_beam", "confidence_gated", "oracle_prefix_closed")
SUMMARY_FIELDS = (
    "task",
    "context_length",
    "requested_total_tree_node_budget",
    "policy",
    "policy_variant",
    "policy_class",
    "sample_count",
    "round_count",
    "terminal_round_count",
    "right_censored_round_count",
    "full_depth_eligible_round_count",
    "covered_depth_mean",
    "covered_depth_median",
    "covered_depth_p90",
    "extra_depth_mean",
    "extra_depth_median",
    "extra_depth_p90",
    "positive_improvement_fraction",
    "first_rejection_recovery_fraction",
    "full_depth_coverage_fraction",
    "deployable_oracle_gap_mean",
    "oracle_improvable_but_deployable_not_fraction",
)
DEPTH_FIELDS = (
    "task",
    "context_length",
    "requested_total_tree_node_budget",
    "policy",
    "policy_variant",
    "depth",
    "eligible_round_count",
    "covered_round_count",
    "round_micro_recall",
)


def _integer_csv(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("values must be comma-separated integers") from exc
    if not parsed or tuple(sorted(set(parsed))) != parsed or parsed[0] < 16:
        raise argparse.ArgumentTypeError(
            "node budgets must be unique, increasing, and at least 16"
        )
    return parsed


def _float_csv(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated numbers") from exc
    if not parsed or tuple(sorted(set(parsed))) != parsed or parsed[0] < 0:
        raise argparse.ArgumentTypeError(
            "thresholds must be unique, increasing, and non-negative"
        )
    return parsed


def _policy_csv(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(parsed) - set(POLICIES)
    if not parsed or unknown:
        raise argparse.ArgumentTypeError(f"unknown policies: {sorted(unknown)}")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("policies must not be repeated")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-rounds", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase2b-replay"))
    parser.add_argument("--node-budgets", type=_integer_csv, default=(16, 32, 64))
    parser.add_argument("--policies", type=_policy_csv, default=POLICIES)
    parser.add_argument(
        "--confidence-thresholds", type=_float_csv, default=(0.25, 0.5, 1.0)
    )
    parser.add_argument("--confidence-branch-width", type=int, default=4)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.confidence_branch_width < 2 or args.confidence_branch_width > 16:
        raise ValueError("--confidence-branch-width must be in [2, 16]")
    return args


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_snapshot() -> dict[str, Any]:
    def git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    status = git("status", "--porcelain")
    return {
        "commit_sha": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def replay_captures(
    captures: list[dict[str, Any]],
    *,
    budgets: tuple[int, ...],
    policies: tuple[str, ...],
    confidence_thresholds: tuple[float, ...],
    confidence_branch_width: int,
) -> list[dict[str, Any]]:
    events = []
    for capture in captures:
        validate_capture_round(capture)
        reference = capture["reference_target_continuation_token_ids"]
        positions = capture["positions"]
        for budget in budgets:
            oracle_nodes = oracle_prefix_closed_tree(
                positions, reference, budget=budget
            )
            oracle = policy_event(
                capture=capture,
                nodes=oracle_nodes,
                policy="oracle_prefix_closed",
                policy_class="oracle",
                policy_variant="target_informed_upper_bound",
                requested_budget=budget,
                policy_parameters={"objective": "maximum observable target-prefix coverage"},
            )
            deployable = []
            if "top1_chain" in policies:
                deployable.append(
                    policy_event(
                        capture=capture,
                        nodes=top1_chain_tree(positions, budget=budget),
                        policy="top1_chain",
                        policy_class="deployable",
                        policy_variant="exact_baseline_chain",
                        requested_budget=budget,
                        policy_parameters={"actual_total_node_count": 16},
                    )
                )
            if "score_beam" in policies:
                deployable.append(
                    policy_event(
                        capture=capture,
                        nodes=score_beam_tree(positions, budget=budget),
                        policy="score_beam",
                        policy_class="deployable",
                        policy_variant="top1_backbone_length_normalized_best_first",
                        requested_budget=budget,
                        policy_parameters={
                            "score": "mean cumulative draft log_probability",
                            "tie_break": "cumulative_score_then_depth_then_token_ids",
                        },
                    )
                )
            if "confidence_gated" in policies:
                for threshold in confidence_thresholds:
                    deployable.append(
                        policy_event(
                            capture=capture,
                            nodes=confidence_gated_tree(
                                positions,
                                budget=budget,
                                margin_threshold=threshold,
                                branch_width=confidence_branch_width,
                            ),
                            policy="confidence_gated",
                            policy_class="deployable",
                            policy_variant=f"top1_top2_margin_le_{threshold:g}",
                            requested_budget=budget,
                            policy_parameters={
                                "margin_threshold": threshold,
                                "branch_width": confidence_branch_width,
                                "confidence_source": "draft Top1-Top2 log_probability margin",
                                "allocation_score": (
                                    "mean cumulative draft log_probability"
                                ),
                                "tie_break": (
                                    "cumulative_score_then_depth_then_token_ids"
                                ),
                            },
                        )
                    )
            for event in deployable:
                gap = oracle["covered_target_prefix_depth"] - event[
                    "covered_target_prefix_depth"
                ]
                if gap < 0:
                    raise RuntimeError("oracle coverage fell below a deployable policy")
                event["deployable_oracle_coverage_gap"] = gap
                event["oracle_improvable_but_deployable_not"] = bool(
                    oracle["extra_covered_depth_vs_top1"] > 0
                    and event["extra_covered_depth_vs_top1"] <= 0
                )
            if "oracle_prefix_closed" in policies:
                oracle["deployable_oracle_coverage_gap"] = 0
                events.append(oracle)
            events.extend(deployable)
    return events


def _summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for group in summary["by_task_context_budget_policy"].values():
        rows.append(
            {
                "task": group["task"],
                "context_length": group["nominal_prompt_length"],
                "requested_total_tree_node_budget": group[
                    "requested_total_tree_node_budget"
                ],
                "policy": group["policy"],
                "policy_variant": group["policy_variant"],
                "policy_class": group["policy_class"],
                "sample_count": group["sample_count"],
                "round_count": group["round_count"],
                "terminal_round_count": group["terminal_round_count"],
                "right_censored_round_count": group["right_censored_round_count"],
                "full_depth_eligible_round_count": group[
                    "full_depth_eligible_round_count"
                ],
                "covered_depth_mean": group["covered_target_prefix_depth"]["mean"],
                "covered_depth_median": group["covered_target_prefix_depth"]["median"],
                "covered_depth_p90": group["covered_target_prefix_depth"]["p90"],
                "extra_depth_mean": group["extra_covered_depth_vs_top1"]["mean"],
                "extra_depth_median": group["extra_covered_depth_vs_top1"]["median"],
                "extra_depth_p90": group["extra_covered_depth_vs_top1"]["p90"],
                "positive_improvement_fraction": group["round_micro"][
                    "positive_improvement_fraction"
                ],
                "first_rejection_recovery_fraction": group["round_micro"][
                    "first_rejection_recovery_fraction"
                ],
                "full_depth_coverage_fraction": group["round_micro"][
                    "full_depth_coverage_fraction"
                ],
                "deployable_oracle_gap_mean": group[
                    "deployable_oracle_coverage_gap"
                ]["mean"],
                "oracle_improvable_but_deployable_not_fraction": group[
                    "round_micro"
                ]["oracle_improvable_but_deployable_not_fraction"],
            }
        )
    return rows


def _depth_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for group in summary["by_task_context_budget_policy"].values():
        for depth in DEPTHS:
            metric = group["depth_recall"][str(depth)]
            rows.append(
                {
                    "task": group["task"],
                    "context_length": group["nominal_prompt_length"],
                    "requested_total_tree_node_budget": group[
                        "requested_total_tree_node_budget"
                    ],
                    "policy": group["policy"],
                    "policy_variant": group["policy_variant"],
                    "depth": depth,
                    **metric,
                }
            )
    return rows


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def policy_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "node_budget_semantics": "maximum total nodes including the root",
        "candidate_node_budgets": {
            str(budget): budget - 1 for budget in args.node_budgets
        },
        "maximum_depth": 15,
        "maximum_per_position_candidates": 16,
        "prefix_closed": True,
        "unique_prefix_nodes": True,
        "static_parallel_logit_candidates": True,
        "cartesian_product_materialized": False,
        "policies": {
            "top1_chain": {"class": "deployable", "uses_target": False},
            "score_beam": {
                "class": "deployable",
                "uses_target": False,
                "score": "mean cumulative draft log_probability",
                "top1_backbone": True,
            },
            "confidence_gated": {
                "class": "deployable",
                "uses_target": False,
                "thresholds": list(args.confidence_thresholds),
                "branch_width": args.confidence_branch_width,
                "allocation_score": "mean cumulative draft log_probability",
                "tie_break": "cumulative_score_then_depth_then_token_ids",
            },
            "oracle_prefix_closed": {
                "class": "oracle",
                "uses_target": True,
                "objective": "maximum observable target-prefix coverage",
            },
        },
        "forbidden_interpretations": [
            "actual tree acceptance",
            "latency",
            "speedup",
            "round reduction",
            "free KV, memory, or compute",
            "lossless sampling",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(effective_argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "terminal.log"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log_path.open("w", encoding="utf-8") as log_stream:
        sys.stdout = TeeStream(original_stdout, log_stream)
        sys.stderr = TeeStream(original_stderr, log_stream)
        try:
            capture_path = args.capture_rounds.expanduser().resolve()
            captures = _read_jsonl(capture_path)
            if not captures:
                raise ValueError("capture-rounds JSONL is empty")
            events = replay_captures(
                captures,
                budgets=args.node_budgets,
                policies=args.policies,
                confidence_thresholds=args.confidence_thresholds,
                confidence_branch_width=args.confidence_branch_width,
            )
            summary = aggregate_tree_events(events)
            policy_config = policy_configuration(args)
            manifest_path = (
                args.capture_manifest.expanduser().resolve()
                if args.capture_manifest
                else capture_path.parent / "manifest.json"
            )
            manifest = {
                "schema_version": "1.0",
                "phase": "2B",
                "status": "ok",
                "mode": "offline_cpu_replay",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": shlex.join(
                    [sys.executable, str(Path(__file__).resolve()), *effective_argv]
                ),
                "repository": _repository_snapshot(),
                "capture_rounds": str(capture_path),
                "capture_rounds_sha256": _sha256(capture_path),
                "capture_manifest": str(manifest_path) if manifest_path.is_file() else None,
                "capture_manifest_sha256": (
                    _sha256(manifest_path) if manifest_path.is_file() else None
                ),
                "capture_round_count": len(captures),
                "tree_event_count": len(events),
                "node_budgets": list(args.node_budgets),
                "policies": list(args.policies),
                "confidence_thresholds": list(args.confidence_thresholds),
                "confidence_branch_width": args.confidence_branch_width,
                "root_counts_toward_budget": True,
                "maximum_draft_depth": 15,
                "target_label_source": "final lossless greedy DFlash verified output",
                "right_censoring_rule": (
                    "fewer than 15 observable continuation tokens after round root"
                ),
                "artifacts": [
                    "replay-manifest.json",
                    "tree-policy-events.jsonl",
                    "tree-summary.json",
                    "tree-summary.csv",
                    "tree-depth-coverage.csv",
                    "tree-policy-config.json",
                    "terminal.log",
                ],
            }
            _atomic_write_jsonl(output_dir / "tree-policy-events.jsonl", events)
            write_json(output_dir / "tree-summary.json", summary)
            _write_csv(output_dir / "tree-summary.csv", SUMMARY_FIELDS, _summary_rows(summary))
            _write_csv(output_dir / "tree-depth-coverage.csv", DEPTH_FIELDS, _depth_rows(summary))
            write_json(output_dir / "tree-policy-config.json", policy_config)
            write_json(output_dir / "replay-manifest.json", manifest)
            print(f"Replayed {len(captures)} rounds into {len(events)} policy events")
            return 0
        except Exception as exc:  # noqa: BLE001 - CLI reports a persisted failure.
            print(f"ERROR: {exc}", file=sys.stderr)
            traceback.print_exc()
            return 1
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
