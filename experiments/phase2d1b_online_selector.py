#!/usr/bin/env python3
"""Run Phase 2D.1B online fixed-budget selector validation on one LongBench task."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import statistics
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase1b.workload import build_longbench_workloads
from dflash.phase2b1.validation import sha256_file, validate_capture_directories
from dflash.phase2d1.runtime import validate_frozen_selector
from dflash.phase2d1b.alignment import build_offline_online_alignment
from dflash.phase2d1b.cohort import build_selector_inventory
from dflash.phase2d1b.metrics import (
    aggregate_selector_results,
    build_decision,
    selector_round_record,
)
from experiments.phase0_smoke import TeeStream
from experiments.phase1b_longctx import (
    _load_tokenizer_only,
    _prepare_workload,
)
from experiments.phase1b_longctx import (
    build_parser as build_phase1b_parser,
)
from experiments.phase1b_longctx import (
    validate_args as validate_phase1b_args,
)
from scripts.collect_environment import collect_environment

SELECTOR_MODES = (
    "reference_dflash",
    "top1_packed_tree",
    "global_fixed",
    "context_rule",
    "decision_stump",
)
ROOT_ARTIFACTS = (
    "manifest.json",
    "workload-inventory.json",
    "frozen-selector.json",
    "selector-decisions.jsonl",
    "selector-rounds.jsonl",
    "samples.jsonl",
    "selector-summary.json",
    "selector-summary.csv",
    "selector-comparison.json",
    "selector-comparison.csv",
    "cross-run-comparison.json",
    "cross-run-comparison.csv",
    "correctness-summary.json",
    "offline-online-selector-alignment.jsonl",
    "offline-online-selector-summary.json",
    "offline-online-selector-summary.csv",
    "environment.json",
    "source-provenance.json",
    "terminal.log",
)
PHASE2C_DIAGNOSTIC_ARTIFACTS = (
    "gold-logit-checks.jsonl",
    "gold-kv-checks.jsonl",
    "tree-traversal-checks.jsonl",
    "round-differential.jsonl",
    "output-divergence.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_phase1b_parser()
    parser.description = __doc__
    parser.add_argument("--frozen-selector-config", required=True, type=Path)
    parser.add_argument(
        "--online-selector-mode",
        action="append",
        choices=SELECTOR_MODES,
        dest="online_selector_modes",
    )
    parser.add_argument(
        "--selector-node-budget",
        action="append",
        type=int,
        choices=(16, 32),
        dest="selector_node_budgets",
    )
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--phase2d1-dir", type=Path, default=None)
    parser.add_argument("--capture-dir", action="append", type=Path, default=[])
    parser.add_argument("--exclude-selector-training-sources", action="store_true")
    parser.add_argument("--enable-gold-checks", action="store_true")
    parser.add_argument("--gold-max-rounds", type=int, default=32)
    parser.add_argument(
        "--decision-stabilization",
        choices=("off", "shadow", "always", "margin"),
        default="shadow",
    )
    parser.add_argument("--decision-margin-threshold", type=float, default=0.1)
    parser.add_argument("--minimum-accepted-gain", type=float, default=0.0)
    parser.add_argument(
        "--minimum-latency-improvement-fraction", type=float, default=0.0
    )
    parser.add_argument(
        "--stump-minimum-additional-improvement-fraction",
        type=float,
        default=0.01,
    )
    parser.set_defaults(output_dir=Path("outputs/phase2d1b-online-selector"))
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = validate_phase1b_args(build_parser().parse_args(argv))
    args.online_selector_modes = tuple(
        args.online_selector_modes or SELECTOR_MODES
    )
    if len(args.online_selector_modes) != len(set(args.online_selector_modes)):
        raise ValueError("online selector modes must be unique")
    args.selector_node_budgets = tuple(args.selector_node_budgets or (16, 32))
    if len(args.selector_node_budgets) != len(set(args.selector_node_budgets)):
        raise ValueError("selector node budgets must be unique")
    if args.sample_id is not None and args.sample_policy != "matched":
        raise ValueError("--sample-id requires --sample-policy matched")
    if args.gold_max_rounds < 1:
        raise ValueError("--gold-max-rounds must be positive")
    if (args.phase2d1_dir is None) != (not args.capture_dir):
        raise ValueError(
            "offline/online alignment requires both --phase2d1-dir and --capture-dir"
        )
    if args.decision_stabilization in {"always", "margin"}:
        raise ValueError(
            "Phase 2D.1B forbids online canonical-recheck fallback; use off or shadow"
        )
    for field in (
        "minimum_accepted_gain",
        "minimum_latency_improvement_fraction",
        "stump_minimum_additional_improvement_fraction",
    ):
        if getattr(args, field) < 0:
            raise ValueError(f"--{field.replace('_', '-')} must be non-negative")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
    fields = sorted({key for row in rows for key in row}) or ["status"]
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


def _append_option(command: list[str], name: str, value: Any) -> None:
    if value is not None:
        command.extend((name, str(value)))


def _common_phase2c_command(args: argparse.Namespace, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "experiments/phase2c_online_tree.py"),
        "--phase2d1b-actual-execution-correctness",
    ]
    for name, value in (
        ("--target-model", args.target_model),
        ("--target-revision", args.target_revision),
        ("--target-model-path", args.target_model_path),
        ("--draft-model", args.draft_model),
        ("--draft-revision", args.draft_revision),
        ("--materialized-draft-path", args.materialized_draft_path),
        ("--longbench-root", args.longbench_root),
        ("--longbench-task", args.longbench_task),
        ("--dataset-jsonl", args.dataset_jsonl),
        ("--govreport-jsonl", args.govreport_jsonl),
        ("--govreport-dataset-path", args.govreport_dataset_path),
        ("--govreport-dataset-cache", args.govreport_dataset_cache),
        ("--dataset-name", args.dataset_name),
        ("--dataset-config", args.dataset_config),
        ("--dataset-split", args.dataset_split),
        ("--context-field", args.context_field),
        ("--text-field", args.text_field),
        ("--query-field", args.query_field),
        ("--answers-field", args.answers_field),
        ("--id-field", args.id_field),
        ("--sample-policy", args.sample_policy),
        ("--seed", args.seed),
        ("--context-lengths", ",".join(map(str, args.context_lengths))),
        ("--pilot-samples", args.pilot_samples),
        ("--timing-repeats", args.timing_repeats),
        ("--warmup-runs", args.warmup_runs),
        ("--max-new-tokens", args.max_new_tokens),
        ("--device", args.device),
        ("--dtype", args.dtype),
        ("--sample-id", args.sample_id),
        ("--decision-stabilization", args.decision_stabilization),
        ("--decision-margin-threshold", args.decision_margin_threshold),
        ("--gold-max-rounds", args.gold_max_rounds),
        ("--frozen-selector-config", args.frozen_selector_config),
        ("--output-dir", output),
    ):
        _append_option(command, name, value)
    if args.exclude_selector_training_sources:
        command.append("--exclude-selector-training-sources")
    if args.enable_gold_checks:
        command.extend(
            (
                "--enable-gold-checks",
                "--gold-comparison-mode",
                "native",
                "--gold-comparison-mode",
                "same_mask",
                "--gold-comparison-mode",
                "math",
            )
        )
    return command


def _subruns(args: argparse.Namespace) -> list[dict[str, Any]]:
    result = []
    if "top1_packed_tree" in args.online_selector_modes or (
        args.online_selector_modes == ("reference_dflash",)
    ):
        result.append({"mode": "top1_packed_tree", "budget": 16})
    for mode in ("global_fixed", "context_rule", "decision_stump"):
        if mode in args.online_selector_modes:
            result.extend(
                {"mode": mode, "budget": budget}
                for budget in args.selector_node_budgets
            )
    if not result:
        raise ValueError("reference_dflash needs at least one tree carrier run")
    return result


def _preflight(
    args: argparse.Namespace, frozen: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    tokenizer, _, _ = _load_tokenizer_only(args)
    prepared, _, _, _ = _prepare_workload(args, tokenizer, build_cohort=False)
    filtered, inventory = build_selector_inventory(
        prepared,
        frozen=frozen,
        task=args.longbench_task,
        nominal_lengths=args.context_lengths,
        requested_sample_count=args.pilot_samples,
        exclude_training_sources=args.exclude_selector_training_sources,
    )
    if args.inspect_workload_only:
        inventory["selected_cohort"] = None
        inventory["selected_sample_ids"] = sorted(
            {
                source_id
                for details in inventory[
                    "selector_source_inventory_by_prompt_length"
                ].values()
                for source_id in details["selected_independent_source_ids"]
            }
        )
        inventory["models_loaded"] = False
        return inventory, [], []
    workloads, cohort = build_longbench_workloads(
        filtered,
        tokenizer=tokenizer,
        task=args.longbench_task,
        nominal_lengths=args.context_lengths,
        sample_count=args.pilot_samples,
        sample_policy=args.sample_policy,
        max_new_tokens=args.max_new_tokens,
        model_position_limit=max(args.context_lengths) + args.max_new_tokens,
        sample_id=args.sample_id,
    )
    inventory["selected_cohort"] = cohort
    inventory["selected_sample_ids"] = sorted({row["sample_id"] for row in workloads})
    inventory["models_loaded"] = False
    offline_events: list[dict[str, Any]] = []
    capture_rounds: list[dict[str, Any]] = []
    if args.phase2d1_dir is not None:
        phase2d1_dir = args.phase2d1_dir.expanduser().resolve()
        for name, expected in frozen["phase2d1_artifact_hashes"].items():
            path = phase2d1_dir / name
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"Phase 2D.1A alignment artifact hash mismatch: {name}")
        offline_events = _read_jsonl(phase2d1_dir / "selector-events.jsonl")
        validated = validate_capture_directories(
            args.capture_dir, schema_dir=REPO_ROOT / "docs" / "schemas"
        )
        actual_hashes = {
            row["task"]: row["capture_directory_sha256"]
            for row in validated["inputs"]
        }
        if actual_hashes != frozen["capture_hashes"]:
            raise ValueError("alignment capture hashes do not match frozen selector")
        capture_rounds = validated["rounds"]
    return inventory, offline_events, capture_rounds


def _reference_sample(tree_sample: dict[str, Any]) -> dict[str, Any]:
    repeats = tree_sample["reference"]["control_repeats"]
    walls = [float(row["cpu_wall_elapsed_ms"]) for row in repeats]
    generated = int(tree_sample["reference"]["generated_token_count"])
    return {
        "schema_version": "1.0",
        "task": tree_sample["task"],
        "sample_id": tree_sample["sample_id"],
        "source_sample_id": tree_sample["source_sample_id"],
        "nominal_prompt_length": tree_sample["nominal_prompt_length"],
        "actual_prompt_length": tree_sample["actual_prompt_length"],
        "selector_mode": "reference_dflash",
        "node_budget_cap": 16,
        "generated_token_count": generated,
        "generated_output_token_ids_sha256": tree_sample["reference"][
            "generated_output_token_ids_sha256"
        ],
        "control_repeats": repeats,
        "formal_generation_wall_latency_median_ms": statistics.median(walls),
        "formal_latency_per_returned_token_ms": statistics.median(walls)
        / generated,
        "formal_throughput_tokens_per_second": generated
        / (statistics.median(walls) / 1000.0),
        "accepted_draft_token_lengths": tree_sample["reference"][
            "accepted_draft_token_lengths"
        ],
        "round_count": tree_sample["reference"]["round_count"],
        "rounds_per_generation": tree_sample["reference"]["round_count"],
        "round_records": tree_sample["reference"]["round_records"],
        "peak_allocated_bytes": max(
            row["peak_allocated_bytes"]
            for row in tree_sample["reference"]["profiling_repeats"]
        ),
        "peak_reserved_bytes": max(
            row["peak_reserved_bytes"]
            for row in tree_sample["reference"]["profiling_repeats"]
        ),
        "formal_latency_includes_selector": False,
    }


def _tree_sample(row: dict[str, Any], *, mode: str, budget: int) -> dict[str, Any]:
    repeats = row["control"]["repeats"]
    walls = [float(value["cpu_wall_elapsed_ms"]) for value in repeats]
    generated = int(row["generated_token_count"])
    return {
        "schema_version": "1.0",
        "task": row["task"],
        "sample_id": row["sample_id"],
        "source_sample_id": row["source_sample_id"],
        "nominal_prompt_length": row["nominal_prompt_length"],
        "actual_prompt_length": row["actual_prompt_length"],
        "selector_mode": mode,
        "node_budget_cap": budget,
        "generated_token_count": generated,
        "generated_output_token_ids_sha256": row["generated_output_token_ids_sha256"],
        "control_repeats": repeats,
        "formal_generation_wall_latency_median_ms": statistics.median(walls),
        "formal_latency_per_returned_token_ms": statistics.median(walls)
        / generated,
        "formal_throughput_tokens_per_second": generated
        / (statistics.median(walls) / 1000.0),
        "rounds_per_generation": row["profiling"]["round_count_mean"],
        "peak_allocated_bytes": row["profiling"]["peak_allocated_bytes"],
        "peak_reserved_bytes": row["profiling"]["peak_reserved_bytes"],
        "control_profile_output_ids_equal": row["profiling"][
            "control_profile_output_ids_equal"
        ],
        "reference_output_ids_equal": row["correctness"][
            "original_dflash_reference_tree_output_ids_equal"
        ],
        "reference_output_token_ids_sha256": row["reference"][
            "generated_output_token_ids_sha256"
        ],
        "formal_latency_includes_selector": mode != "top1_packed_tree",
    }


def _reference_rounds(sample: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in sample["reference"]["round_records"]:
        accepted = int(record["accepted_draft_token_count"])
        committed = int(record["committed_token_count"])
        rows.append(
            {
                "schema_version": "1.0",
                "task": sample["task"],
                "sample_id": sample["sample_id"],
                "source_sample_id": sample["source_sample_id"],
                "source_text_sha256": sample["source_text_sha256"],
                "nominal_prompt_length": sample["nominal_prompt_length"],
                "actual_prompt_length": sample["actual_prompt_length"],
                "profiling_repeat": 0,
                "round_index": int(record["round_index"]),
                "generated_root_offset": int(record["context_length_before"])
                - int(sample["actual_prompt_length"]),
                "root_token_id": None,
                "accepted_draft_token_count": accepted,
                "committed_token_count": committed,
                "terminal": bool(record["terminal_round"]),
                "right_censored": bool(record["truncated_token_count"]),
                "truncated_token_count": int(record["truncated_token_count"]),
                "actual_constructed_node_count": 16,
                "node_budget_cap": 16,
                "same_budget_cap": True,
                "same_actual_node_count_as_top1": True,
                "selected_policy_config": "original_dflash_top1_chain",
                "selector_mode": "reference_dflash",
                "selector_feature_value": None,
                "selector_decision": None,
                "fallback_used": False,
                "decision_boundary_nearby": False,
                "round_latency": None,
                "draft_latency": None,
                "selector_latency": 0.0,
                "verification_latency": None,
                "KV_compaction_latency": None,
                "draft_topk_device_to_host_sync_cpu_ms": 0.0,
                "host_to_device_transfer_cpu_ms": 0.0,
                "selector_runtime_included_in_formal_control": True,
                "component_timing_source": "reference_profile_not_flattened",
            }
        )
    return rows


def _empty_alignment() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return [], {
        "schema_version": "1.0",
        "status": "not_requested",
        "alignment_valid_count": 0,
        "alignment_invalid_count": 0,
        "different_round_segmentation_forced_matches": 0,
        "offline_coverage_treated_as_online_acceptance": False,
    }


def _collect_diagnostics(output: Path, subruns: list[dict[str, Any]]) -> None:
    for filename in PHASE2C_DIAGNOSTIC_ARTIFACTS:
        destination = output / filename
        sources = [row["output_dir"] / filename for row in subruns]
        if filename.endswith(".jsonl"):
            destination.write_text(
                "".join(
                    source.read_text(encoding="utf-8")
                    for source in sources
                    if source.is_file()
                ),
                encoding="utf-8",
            )
        else:
            _write_json(
                destination,
                {
                    "schema_version": "1.0",
                    "status": "subrun_index",
                    "subrun_artifacts": [str(source) for source in sources],
                },
            )


def _csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in row.items()
            if not isinstance(value, (dict, list))
        }
        for row in summary["aggregates"]
    ]


def _cross_run_comparison(
    completed: list[dict[str, Any]], samples: list[dict[str, Any]]
) -> dict[str, Any]:
    """Summarize repeat/profile stability without changing Phase 2C semantics."""
    records = []
    for row in samples:
        if row["selector_mode"] == "reference_dflash":
            continue
        records.append(
            {
                "task": row["task"],
                "sample_id": row["sample_id"],
                "nominal_prompt_length": row["nominal_prompt_length"],
                "selector_mode": row["selector_mode"],
                "node_budget_cap": row["node_budget_cap"],
                "reference_output_token_ids_sha256": row[
                    "reference_output_token_ids_sha256"
                ],
                "tree_output_token_ids_sha256": row[
                    "generated_output_token_ids_sha256"
                ],
                "control_profile_output_ids_equal": row[
                    "control_profile_output_ids_equal"
                ],
                "original_dflash_reference_tree_output_ids_equal": row[
                    "reference_output_ids_equal"
                ],
            }
        )
    grouped_hashes: dict[tuple[str, str, int], dict[str, set[str]]] = {}
    for row in records:
        key = (
            row["task"],
            row["sample_id"],
            row["nominal_prompt_length"],
        )
        values = grouped_hashes.setdefault(key, {"reference": set(), "tree": set()})
        values["reference"].add(row["reference_output_token_ids_sha256"])
        values["tree"].add(row["tree_output_token_ids_sha256"])
    return {
        "schema_version": "1.0",
        "status": "complete" if records else "no_tree_records",
        "comparison_scope": (
            "independent_phase2c_subruns_and_control_profile_repeats"
        ),
        "formal_latency_source": "control_only",
        "subrun_count": len(completed),
        "record_count": len(records),
        "reference_output_hash_distribution": dict(
            Counter(row["reference_output_token_ids_sha256"] for row in records)
        ),
        "tree_output_hash_distribution": dict(
            Counter(row["tree_output_token_ids_sha256"] for row in records)
        ),
        "reference_output_stable_across_subruns": all(
            len(values["reference"]) <= 1 for values in grouped_hashes.values()
        ),
        "tree_output_stable_across_subruns": all(
            len(values["tree"]) <= 1 for values in grouped_hashes.values()
        ),
        "all_control_profile_output_ids_equal": all(
            row["control_profile_output_ids_equal"] for row in records
        ),
        "all_original_dflash_reference_tree_output_ids_equal": all(
            row["original_dflash_reference_tree_output_ids_equal"]
            for row in records
        ),
        "records": records,
    }


def _phase2d1b_subrun_correctness(value: dict[str, Any]) -> bool:
    """Require token-exact online behavior even when Phase 2C runs in shadow mode."""
    return bool(
        value["all_checks_passed"]
        and value["actual_execution_correctness_passed"]
        and value["all_required_token_exact_checks_passed"]
        and not value["observed_native_output_divergence"]
    )


def _initialize_semantic_artifacts(output: Path) -> None:
    json_defaults = {
        "selector-summary.json": {"schema_version": "1.0", "status": "started"},
        "selector-comparison.json": {"schema_version": "1.0", "status": "not_run"},
        "cross-run-comparison.json": {
            "schema_version": "1.0",
            "status": "not_run",
        },
        "correctness-summary.json": {
            "schema_version": "1.0",
            "status": "started",
            "all_checks_passed": False,
            "actual_execution_correctness_passed": False,
            "actual_output_exactness_required": True,
            "diagnostic_numerical_variation_observed": False,
            "diagnostic_numerical_variation_reasons": [],
            "hard_failure_classes": [],
        },
        "offline-online-selector-summary.json": {
            "schema_version": "1.0",
            "status": "not_run",
        },
        "output-divergence.json": {"schema_version": "1.0", "status": "not_run"},
    }
    for name, value in json_defaults.items():
        path = output / name
        if not path.exists():
            _write_json(path, value)
    for name in (
        "selector-decisions.jsonl",
        "selector-rounds.jsonl",
        "samples.jsonl",
        "offline-online-selector-alignment.jsonl",
        "gold-logit-checks.jsonl",
        "gold-kv-checks.jsonl",
        "tree-traversal-checks.jsonl",
        "round-differential.jsonl",
    ):
        path = output / name
        if not path.exists():
            path.write_text("", encoding="utf-8")
    for name in (
        "selector-summary.csv",
        "selector-comparison.csv",
        "cross-run-comparison.csv",
        "offline-online-selector-summary.csv",
    ):
        path = output / name
        if not path.exists():
            path.write_text("status\nnot_run\n", encoding="utf-8")
    (output / "terminal.log").touch(exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise ValueError("Phase 2D.1B output directory already exists")
    frozen = validate_frozen_selector(
        json.loads(args.frozen_selector_config.read_text(encoding="utf-8"))
    )
    inventory, offline_events, capture_rounds = _preflight(args, frozen)
    output.mkdir(parents=True)
    environment = collect_environment(
        run_config={"phase": "2D.1B", "task": args.longbench_task},
        run_command=shlex.join([sys.executable, *sys.argv]),
    )
    manifest = {
        "schema_version": "1.0",
        "phase": "2D.1B",
        "status": "inventory_only" if args.inspect_workload_only else "started",
        "selector_config_hash": frozen["config_sha256"],
        "online_selector_modes": list(args.online_selector_modes),
        "selector_node_budgets": list(args.selector_node_budgets),
        "evaluation_cohort": inventory["evaluation_cohort"],
        "independent_generalization": inventory["independent_generalization"],
        "independent_source_ids": (
            sorted(
                {
                    source
                    for row in inventory[
                        "selector_source_inventory_by_prompt_length"
                    ].values()
                    for source in row["selected_independent_source_ids"]
                }
            )
            if inventory["independent_generalization"]
            else []
        ),
        "decision_rules": {
            "minimum_accepted_gain": args.minimum_accepted_gain,
            "minimum_latency_improvement_fraction": args.minimum_latency_improvement_fraction,
            "stump_minimum_additional_improvement_fraction": args.stump_minimum_additional_improvement_fraction,
        },
        "target_blind_contract": {
            "target_logits_read_by_selector": False,
            "target_hidden_read_by_selector": False,
            "accepted_depth_read_by_selector": False,
            "extra_target_forward_added": False,
            "selector_decision_before_target_verification": True,
        },
        "formal_latency_source": "unprofiled_control_cpu_wall_including_selector_and_gpu_sync",
        "component_timing_source": "separate_profiling_path",
        "correctness_gate": {
            "actual_execution_is_semantic_oracle": True,
            "original_dflash_tree_output_exactness_required": True,
            "structural_kv_traversal_failures_are_hard": True,
            "consequence_free_numerical_backend_variation_is_diagnostic": True,
        },
        "subruns": [],
        "artifacts": list(ROOT_ARTIFACTS + PHASE2C_DIAGNOSTIC_ARTIFACTS),
        "error": None,
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "workload-inventory.json", inventory)
    shutil.copyfile(args.frozen_selector_config, output / "frozen-selector.json")
    _write_json(output / "environment.json", environment)
    _write_json(
        output / "source-provenance.json",
        {"schema_version": "1.0", "status": "not_run"},
    )
    _initialize_semantic_artifacts(output)
    if args.inspect_workload_only:
        for name in ROOT_ARTIFACTS:
            path = output / name
            if not path.exists():
                if name.endswith(".jsonl"):
                    path.write_text("", encoding="utf-8")
                elif name.endswith(".csv"):
                    path.write_text("status\nnot_run\n", encoding="utf-8")
                elif name.endswith(".log"):
                    path.write_text("inventory-only: no model materialized\n", encoding="utf-8")
                else:
                    _write_json(path, {"schema_version": "1.0", "status": "not_run"})
        return 0

    exit_code = 1
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with (output / "terminal.log").open("w", encoding="utf-8", buffering=1) as log:
        sys.stdout = TeeStream(original_stdout, log)
        sys.stderr = TeeStream(original_stderr, log)
        try:
            completed = []
            for subrun in _subruns(args):
                child = output / "phase2c-subruns" / f"{subrun['mode']}-n{subrun['budget']}"
                command = _common_phase2c_command(args, child)
                if subrun["mode"] == "top1_packed_tree":
                    command.extend(("--tree-policy", "top1_chain", "--node-budget", "16", "--beam-width", "4"))
                else:
                    command.extend(("--online-selector-mode", subrun["mode"], "--selector-node-budget", str(subrun["budget"])))
                print(f"subrun={subrun['mode']} N={subrun['budget']}")
                print(shlex.join(command))
                result = subprocess.run(
                    command, cwd=REPO_ROOT, text=True, check=False
                )
                if result.returncode:
                    raise RuntimeError(
                        f"Phase 2C subrun failed: mode={subrun['mode']} N={subrun['budget']} rc={result.returncode}"
                    )
                completed.append({**subrun, "output_dir": child})

            samples = []
            rounds = []
            decisions = []
            correctness_records = []
            reference_seen = set()
            provenance = []
            for subrun in completed:
                child_samples = _read_jsonl(subrun["output_dir"] / "samples.jsonl")
                child_rounds = _read_jsonl(subrun["output_dir"] / "tree-rounds.jsonl")
                child_correctness = _read_json(subrun["output_dir"] / "correctness-summary.json")
                correctness_records.append(
                    {
                        "selector_mode": subrun["mode"],
                        "node_budget_cap": subrun["budget"],
                        "all_checks_passed": child_correctness["all_checks_passed"],
                        "all_required_token_exact_checks_passed": child_correctness[
                            "all_required_token_exact_checks_passed"
                        ],
                        "observed_native_output_divergence": child_correctness[
                            "observed_native_output_divergence"
                        ],
                        "actual_execution_correctness_passed": child_correctness[
                            "actual_execution_correctness_passed"
                        ],
                        "actual_output_exactness_required": child_correctness[
                            "actual_output_exactness_required"
                        ],
                        "diagnostic_numerical_variation_observed": child_correctness[
                            "diagnostic_numerical_variation_observed"
                        ],
                        "diagnostic_numerical_variation_reasons": child_correctness.get(
                            "diagnostic_numerical_variation_reasons", []
                        ),
                        "phase2d1b_online_correctness_passed": (
                            _phase2d1b_subrun_correctness(child_correctness)
                        ),
                        "failure_class": child_correctness.get("failure_class"),
                        "hard_failure_class": child_correctness.get(
                            "hard_failure_class"
                        ),
                    }
                )
                provenance.append(_read_json(subrun["output_dir"] / "source-provenance.json"))
                for row in child_samples:
                    key = (row["task"], row["sample_id"], row["nominal_prompt_length"])
                    if "reference_dflash" in args.online_selector_modes and key not in reference_seen:
                        samples.append(_reference_sample(row))
                        rounds.extend(_reference_rounds(row))
                        reference_seen.add(key)
                    if subrun["mode"] in args.online_selector_modes:
                        samples.append(_tree_sample(row, mode=subrun["mode"], budget=subrun["budget"]))
                for row in child_rounds:
                    normalized = selector_round_record(
                        row, mode=subrun["mode"], node_budget=subrun["budget"]
                    )
                    rounds.append(normalized)
                    if row.get("online_selector") is not None:
                        decisions.append(
                            {
                                "schema_version": "1.0",
                                **{key: normalized[key] for key in ("task", "sample_id", "source_sample_id", "nominal_prompt_length", "profiling_repeat", "round_index", "generated_root_offset")},
                                **row["online_selector"],
                            }
                        )
            summary = aggregate_selector_results(samples, rounds)
            phase2c_correctness_passed = all(
                row["phase2d1b_online_correctness_passed"]
                for row in correctness_records
            )
            target_blind_invariants_passed = all(
                not row["selector_decision"]
                or row["selector_decision"][
                    "decision_made_before_target_verification"
                ]
                and not row["selector_decision"][
                    "target_dependent_runtime_input_used"
                ]
                and row["selector_decision"]["target_forward_count_added"] == 0
                for row in rounds
            )
            correctness_passed = (
                phase2c_correctness_passed and target_blind_invariants_passed
            )
            correctness = {
                "schema_version": "1.0",
                "status": "pass" if correctness_passed else "fail",
                "all_checks_passed": correctness_passed,
                "actual_execution_correctness_passed": (
                    phase2c_correctness_passed
                ),
                "actual_output_exactness_required": True,
                "diagnostic_numerical_variation_observed": any(
                    row["diagnostic_numerical_variation_observed"]
                    for row in correctness_records
                ),
                "diagnostic_numerical_variation_reasons": sorted(
                    {
                        reason
                        for row in correctness_records
                        for reason in row[
                            "diagnostic_numerical_variation_reasons"
                        ]
                    }
                ),
                "hard_failure_classes": sorted(
                    {
                        row["hard_failure_class"]
                        for row in correctness_records
                        if row["hard_failure_class"] is not None
                    }
                ),
                "records": correctness_records,
                "phase2c_correctness_passed": phase2c_correctness_passed,
                "target_blind_invariants_passed": target_blind_invariants_passed,
            }
            comparison = build_decision(
                summary,
                correctness_passed=correctness_passed,
                independent_evidence=inventory["independent_generalization"],
                minimum_accepted_gain=args.minimum_accepted_gain,
                minimum_latency_improvement_fraction=args.minimum_latency_improvement_fraction,
                stump_minimum_additional_improvement_fraction=args.stump_minimum_additional_improvement_fraction,
            )
            if offline_events:
                alignment_rows, alignment_summary = build_offline_online_alignment(
                    online_rounds=rounds,
                    offline_events=offline_events,
                    captures=capture_rounds,
                    frozen=frozen,
                )
            else:
                alignment_rows, alignment_summary = _empty_alignment()
            _write_jsonl(output / "selector-decisions.jsonl", decisions)
            _write_jsonl(output / "selector-rounds.jsonl", rounds)
            _write_jsonl(output / "samples.jsonl", samples)
            _write_json(output / "selector-summary.json", summary)
            _write_csv(output / "selector-summary.csv", _csv_rows(summary))
            _write_json(output / "selector-comparison.json", comparison)
            _write_csv(output / "selector-comparison.csv", comparison.get("eligible_candidates", []))
            cross_run = _cross_run_comparison(completed, samples)
            _write_json(output / "cross-run-comparison.json", cross_run)
            _write_csv(output / "cross-run-comparison.csv", cross_run["records"])
            _write_json(output / "correctness-summary.json", correctness)
            _write_jsonl(output / "offline-online-selector-alignment.jsonl", alignment_rows)
            _write_json(output / "offline-online-selector-summary.json", alignment_summary)
            _write_csv(output / "offline-online-selector-summary.csv", [alignment_summary])
            _write_json(output / "source-provenance.json", {"schema_version": "1.0", "status": "ok", "subruns": provenance})
            _collect_diagnostics(output, completed)
            manifest.update(
                {
                    "status": "ok" if correctness_passed else "failed",
                    "subruns": [
                        {**row, "output_dir": str(row["output_dir"])} for row in completed
                    ],
                    "automatic_decision": comparison["decision"],
                    "error": None,
                }
            )
            _write_json(output / "manifest.json", manifest)
            print(f"automatic_decision={comparison['decision']}")
            exit_code = 0 if correctness_passed else 1
        except Exception as exc:  # noqa: BLE001 - preserve failure artifacts.
            manifest["status"] = "error"
            manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
            _write_json(output / "manifest.json", manifest)
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
