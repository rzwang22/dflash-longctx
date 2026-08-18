#!/usr/bin/env python3
"""Run Phase 2D.1B-NR independent-process numerical diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase1.materialize import verify_materialization
from dflash.phase1b.workload import LONG_BENCH_TASKS
from dflash.phase2b1.validation import sha256_file, validate_capture_directories
from dflash.phase2d1.runtime import validate_frozen_selector
from dflash.phase2d1bnr.aggregation import (
    aggregate_cohort_cases,
    aggregate_exact_reproducibility,
    build_margin_trigger_sweep,
)
from dflash.phase2d1bnr.cohort import (
    derive_matched_capture_cohort,
    safe_subrun_directory_name,
    subrun_identity,
)
from dflash.phase2d1bnr.decision import (
    DEFAULT_DECISION_RULES,
    build_research_summary,
    choose_nr_decision,
)
from dflash.phase2d1bnr.margins import extract_round_margin_events
from dflash.phase2d1bnr.parse import (
    ArtifactError,
    build_divergence_event,
    classify_subrun_artifacts,
    infrastructure_failure_observation,
    public_observation,
    read_json,
    read_jsonl,
)
from experiments.phase1b_longctx import parse_context_lengths

BASELINE_SHA = "d60e7db0514053ac580f2bfa9ca8cf7a534c1b49"
DEFAULT_THRESHOLDS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
DEFAULT_MODES = ("top1_packed_tree", "global_fixed")
ARTIFACTS = (
    "nr-manifest.json",
    "exact-case-repeats.jsonl",
    "exact-case-reproducibility-summary.json",
    "cohort-subruns.jsonl",
    "cohort-case-summary.csv",
    "context-divergence-summary.csv",
    "policy-divergence-summary.csv",
    "divergence-events.jsonl",
    "replay-semantic-events.jsonl",
    "round-margin-events.jsonl",
    "margin-trigger-sweep.csv",
    "margin-trigger-sweep.json",
    "research-question-summary.json",
    "nr-decision.json",
    "subrun-index.jsonl",
    "terminal.log",
)


class TeeStream:
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


def _parse_csv_choices(value: str, *, choices: set[str]) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("values must be non-empty and unique")
    invalid = sorted(set(result) - choices)
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported values: {invalid}")
    return result


def _parse_budgets(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid node budget") from exc
    if (
        not result
        or len(set(result)) != len(result)
        or any(v not in {16, 32} for v in result)
    ):
        raise argparse.ArgumentTypeError(
            "node budgets must be unique values from 16,32"
        )
    return result


def _parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid margin threshold") from exc
    if not result or len(set(result)) != len(result) or any(v < 0 for v in result):
        raise argparse.ArgumentTypeError(
            "margin thresholds must be unique and non-negative"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model-path", required=True, type=Path)
    parser.add_argument(
        "--target-revision", default="b968826d9c46dd6066d109eabc6255188de91218"
    )
    parser.add_argument("--materialized-draft-path", required=True, type=Path)
    parser.add_argument(
        "--draft-revision", default="9b41424b7109f9c5413454f481b09a82b85333f4"
    )
    parser.add_argument("--longbench-root", required=True, type=Path)
    parser.add_argument(
        "--longbench-task", choices=LONG_BENCH_TASKS, default="gov_report"
    )
    parser.add_argument("--frozen-selector-config", required=True, type=Path)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase2d1b-nr-numerical-robustness"),
    )
    parser.add_argument("--exact-case-sample-id", default="gov_report-fd287bbfd3bdbdc9")
    parser.add_argument(
        "--exact-case-context-length", type=parse_context_lengths, default=(1024,)
    )
    parser.add_argument("--exact-case-repeats", type=int, default=10)
    parser.add_argument(
        "--cohort-context-lengths",
        type=parse_context_lengths,
        default=(1024, 4096, 8192, 16384, 32768),
    )
    parser.add_argument("--cohort-repeats", type=int, default=1)
    parser.add_argument(
        "--modes",
        type=lambda value: _parse_csv_choices(
            value, choices={"top1_packed_tree", "global_fixed"}
        ),
        default=DEFAULT_MODES,
    )
    parser.add_argument("--node-budgets", type=_parse_budgets, default=(16, 32))
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--margin-thresholds", type=_parse_thresholds, default=DEFAULT_THRESHOLDS
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--run-exact-case", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--run-cohort", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append a disjoint requested study stage to an existing NR root.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if len(args.exact_case_context_length) != 1:
        raise ValueError("--exact-case-context-length requires exactly one length")
    args.exact_case_context_length = args.exact_case_context_length[0]
    if args.exact_case_repeats < 1 or args.cohort_repeats < 1:
        raise ValueError("repeat counts must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.longbench_task != "gov_report":
        raise ValueError("the Phase 2D.1B-NR formal cohort is GovReport only")
    if "top1_packed_tree" in args.modes and 16 not in args.node_budgets:
        raise ValueError("top1_packed_tree requires node budget 16")
    return args


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
    fields = sorted({field for row in rows for field in row}) or ["status"]
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


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _gpu_inventory() -> dict[str, Any]:
    try:
        output = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": str(exc), "devices": []}
    return {"available": bool(output), "devices": output}


def _protected_changes(current_sha: str) -> list[str]:
    changed = set(
        _git("diff", "--name-only", f"{BASELINE_SHA}..{current_sha}").splitlines()
    )
    changed.update(_git("diff", "--name-only").splitlines())
    changed.update(_git("diff", "--cached", "--name-only").splitlines())
    changed.update(_git("ls-files", "--others", "--exclude-standard").splitlines())
    forbidden = (
        "third_party/",
        "dflash/phase2c/",
        "dflash/phase2d1/",
        "dflash/phase2d1b/correctness.py",
    )
    return sorted(path for path in changed if path.startswith(forbidden))


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _configs(args: argparse.Namespace, *, exact: bool) -> list[tuple[str, int]]:
    result = []
    if "top1_packed_tree" in args.modes:
        result.append(("top1_packed_tree", 16))
    if "global_fixed" in args.modes:
        budgets = (16,) if exact else args.node_budgets
        result.extend(("global_fixed", budget) for budget in budgets)
    return result


def _specs(
    args: argparse.Namespace, cohort: list[dict[str, str]]
) -> list[dict[str, Any]]:
    exact_source = next(
        (row for row in cohort if row["sample_id"] == args.exact_case_sample_id),
        None,
    )
    if args.run_exact_case and exact_source is None:
        raise ValueError(
            "exact-case sample is absent from the validated matched capture"
        )
    result = []
    if args.run_exact_case:
        for mode, budget in _configs(args, exact=True):
            for repeat in range(args.exact_case_repeats):
                result.append(
                    subrun_identity(
                        experiment="exact_case",
                        task=args.longbench_task,
                        sample_id=exact_source["sample_id"],
                        source_sample_id=exact_source["source_sample_id"],
                        context_length=args.exact_case_context_length,
                        selector_mode=mode,
                        node_budget=budget,
                        repeat_index=repeat,
                        max_new_tokens=args.max_new_tokens,
                    )
                )
    if args.run_cohort:
        for source in cohort:
            for context in args.cohort_context_lengths:
                for mode, budget in _configs(args, exact=False):
                    for repeat in range(args.cohort_repeats):
                        result.append(
                            subrun_identity(
                                experiment="cohort",
                                task=args.longbench_task,
                                sample_id=source["sample_id"],
                                source_sample_id=source["source_sample_id"],
                                context_length=context,
                                selector_mode=mode,
                                node_budget=budget,
                                repeat_index=repeat,
                                max_new_tokens=args.max_new_tokens,
                            )
                        )
    ids = [row["independent_repeat_observation_id"] for row in result]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate subrun identity generated")
    return result


def _command(
    args: argparse.Namespace, identity: dict[str, Any], output_dir: Path
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "experiments/phase2c_online_tree.py"),
        "--phase2d1b-actual-execution-correctness",
        "--target-model-path",
        str(args.target_model_path),
        "--target-revision",
        args.target_revision,
        "--materialized-draft-path",
        str(args.materialized_draft_path),
        "--draft-revision",
        args.draft_revision,
        "--longbench-root",
        str(args.longbench_root),
        "--longbench-task",
        args.longbench_task,
        "--sample-policy",
        "matched",
        "--sample-id",
        identity["sample_id"],
        "--context-lengths",
        str(identity["nominal_prompt_length"]),
        "--pilot-samples",
        "1",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--warmup-runs",
        "0",
        "--timing-repeats",
        "1",
        "--device",
        args.device,
        "--dtype",
        "bfloat16",
        "--decision-stabilization",
        "shadow",
        "--enable-gold-checks",
        "--gold-comparison-mode",
        "native",
        "--gold-comparison-mode",
        "same_mask",
        "--gold-comparison-mode",
        "math",
        "--gold-max-rounds",
        str(args.max_new_tokens),
        "--frozen-selector-config",
        str(args.frozen_selector_config),
        "--output-dir",
        str(output_dir),
    ]
    if identity["selector_mode"] == "top1_packed_tree":
        command.extend(
            (
                "--tree-policy",
                "top1_chain",
                "--node-budget",
                "16",
                "--beam-width",
                "4",
            )
        )
    else:
        command.extend(
            (
                "--online-selector-mode",
                identity["selector_mode"],
                "--selector-node-budget",
                str(identity["node_budget"]),
            )
        )
    return command


def _manifest(
    args: argparse.Namespace,
    *,
    cohort: list[dict[str, str]],
    capture_validation: dict[str, Any],
    frozen: dict[str, Any],
    materialization: dict[str, Any],
    specs: list[dict[str, Any]],
) -> dict[str, Any]:
    current_sha = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    dirty = bool(_git("status", "--porcelain"))
    working_tree_paths = _git("status", "--porcelain").splitlines()
    dataset_path = args.longbench_root.expanduser().resolve() / "gov_report.jsonl"
    protected = _protected_changes(current_sha)
    if protected:
        raise ValueError(f"protected online implementation changed: {protected}")
    return {
        "schema_version": "1.0",
        "phase": "2D.1B-NR",
        "status": "started",
        "baseline_repo_sha": BASELINE_SHA,
        "repository": {
            "current_sha": current_sha,
            "branch": branch,
            "dirty": dirty,
            "working_tree_status": working_tree_paths,
            "protected_online_paths_changed": protected,
        },
        "target_model": {
            "path": str(args.target_model_path.expanduser().resolve()),
            "revision": args.target_revision,
        },
        "materialized_draft": {
            "path": str(args.materialized_draft_path.expanduser().resolve()),
            "revision": args.draft_revision,
            "materialization_id": materialization.get("materialization_id"),
            "materialization_manifest_sha256": sha256_file(
                args.materialized_draft_path.expanduser().resolve()
                / "materialization.json"
            ),
            "manifest": materialization,
        },
        "frozen_selector": {
            "path": str(args.frozen_selector_config.expanduser().resolve()),
            "config_sha256": frozen["config_sha256"],
            "file_sha256": sha256_file(
                args.frozen_selector_config.expanduser().resolve()
            ),
            "training_source_ids": frozen["training_source_ids"],
        },
        "capture_inputs": capture_validation["inputs"],
        "longbench": {
            "root": str(args.longbench_root.expanduser().resolve()),
            "task": args.longbench_task,
            "jsonl_path": str(dataset_path),
            "jsonl_sha256": sha256_file(dataset_path),
            "matched_capture_cohort": cohort,
        },
        "environment": {
            "python": sys.version,
            "device": args.device,
            "cuda_gpu_inventory": _gpu_inventory(),
        },
        "configuration": {
            "dtype": "bfloat16",
            "target_attention_backend": "sdpa_required_and_verified_from_subrun_manifest",
            "decision_stabilization": "shadow",
            "gold_comparison_modes": ["native", "same_mask", "math"],
            "warmup_runs": 0,
            "timing_repeats": 1,
            "exact_case_repeats": args.exact_case_repeats,
            "cohort_repeats": args.cohort_repeats,
            "exact_case_context_length": args.exact_case_context_length,
            "cohort_context_lengths": list(args.cohort_context_lengths),
            "modes": list(args.modes),
            "node_budgets": list(args.node_budgets),
            "max_new_tokens": args.max_new_tokens,
            "margin_thresholds": list(args.margin_thresholds),
            "run_exact_case": args.run_exact_case,
            "run_cohort": args.run_cohort,
        },
        "count_semantics": {
            "physical_case": "task x source/sample x context",
            "configuration_case": "physical case x selector/config x node budget",
            "independent_repeat_observation": "configuration case x fresh process repeat",
            "repeats_are_not_independent_source_evidence": True,
            "requested_subrun_count": len(specs),
            "requested_physical_case_count": len(
                {row["physical_case_id"] for row in specs}
            ),
            "requested_configuration_case_count": len(
                {row["configuration_case_id"] for row in specs}
            ),
        },
        "decision_rules": DEFAULT_DECISION_RULES,
        "execution_stages": [shlex.join([sys.executable, *sys.argv])],
        "subruns": [],
        "artifacts": list(ARTIFACTS),
        "online_execution_modified": False,
        "selector_modified": False,
        "tree_modified": False,
        "kv_compaction_modified": False,
        "correctness_gate_modified": False,
        "numerical_thresholds_modified": False,
        "materialization_required": False,
        "formal_timing_claims_made": False,
        "no_performance_claims": True,
        "error": None,
    }


def _persist_results(
    output: Path,
    manifest: dict[str, Any],
    observations: list[dict[str, Any]],
    divergence_events: list[dict[str, Any]],
    replay_events: list[dict[str, Any]],
    round_events: list[dict[str, Any]],
    thresholds: tuple[float, ...],
) -> None:
    public = [public_observation(row) for row in observations]
    exact = [row for row in public if row["experiment"] == "exact_case"]
    cohort = [row for row in public if row["experiment"] == "cohort"]
    exact_summary = aggregate_exact_reproducibility(public)
    cohort_summary = aggregate_cohort_cases(public)
    sweep_rows = build_margin_trigger_sweep(round_events, thresholds=thresholds)
    decision = choose_nr_decision(
        public,
        cohort_summary["policy_divergence_summary"],
        sweep_rows,
    )
    research = build_research_summary(
        exact_summary, cohort_summary, sweep_rows, decision
    )
    _write_json(output / "nr-manifest.json", manifest)
    _write_jsonl(output / "subrun-index.jsonl", public)
    _write_jsonl(output / "exact-case-repeats.jsonl", exact)
    _write_json(output / "exact-case-reproducibility-summary.json", exact_summary)
    _write_jsonl(output / "cohort-subruns.jsonl", cohort)
    _write_csv(
        output / "cohort-case-summary.csv",
        cohort_summary["cohort_case_summary"],
    )
    _write_csv(
        output / "context-divergence-summary.csv",
        cohort_summary["context_divergence_summary"],
    )
    _write_csv(
        output / "policy-divergence-summary.csv",
        cohort_summary["policy_divergence_summary"],
    )
    _write_jsonl(output / "divergence-events.jsonl", divergence_events)
    _write_jsonl(output / "replay-semantic-events.jsonl", replay_events)
    _write_jsonl(output / "round-margin-events.jsonl", round_events)
    _write_csv(output / "margin-trigger-sweep.csv", sweep_rows)
    _write_json(
        output / "margin-trigger-sweep.json",
        {
            "schema_version": "1.0",
            "thresholds": list(thresholds),
            "rows": sweep_rows,
        },
    )
    _write_json(output / "research-question-summary.json", research)
    _write_json(output / "nr-decision.json", decision)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and not args.resume:
        raise ValueError(f"output directory already exists: {output}")
    if args.resume and not output.is_dir():
        raise ValueError(f"cannot resume absent output directory: {output}")
    if not args.target_model_path.expanduser().resolve().is_dir():
        raise ValueError(f"target model directory is absent: {args.target_model_path}")
    materialization = verify_materialization(
        args.materialized_draft_path, expected_implementation="instrumented"
    )
    frozen = validate_frozen_selector(
        json.loads(args.frozen_selector_config.read_text(encoding="utf-8"))
    )
    capture_validation = validate_capture_directories(
        [args.capture_dir], schema_dir=REPO_ROOT / "docs" / "schemas"
    )
    capture_hashes = {
        row["task"]: row["capture_directory_sha256"]
        for row in capture_validation["inputs"]
    }
    expected_capture_hash = frozen["capture_hashes"].get(args.longbench_task)
    if capture_hashes.get(args.longbench_task) != expected_capture_hash:
        raise ValueError("capture directory hash does not match frozen selector")
    cohort = derive_matched_capture_cohort(
        capture_validation["rounds"],
        task=args.longbench_task,
        context_lengths=args.cohort_context_lengths,
    )
    if len(cohort) != 3:
        raise ValueError(
            f"formal GovReport matched-3 capture must yield 3 sources, got {len(cohort)}"
        )
    specs = _specs(args, cohort)
    new_manifest = _manifest(
        args,
        cohort=cohort,
        capture_validation=capture_validation,
        frozen=frozen,
        materialization=materialization,
        specs=specs,
    )
    if args.resume:
        manifest = read_json(output / "nr-manifest.json")
        compatibility = {
            "baseline_repo_sha": new_manifest["baseline_repo_sha"],
            "current_sha": new_manifest["repository"]["current_sha"],
            "frozen_selector_hash": new_manifest["frozen_selector"]["config_sha256"],
            "materialization_id": new_manifest["materialized_draft"][
                "materialization_id"
            ],
            "longbench_jsonl_sha256": new_manifest["longbench"]["jsonl_sha256"],
            "margin_thresholds": new_manifest["configuration"]["margin_thresholds"],
            "max_new_tokens": new_manifest["configuration"]["max_new_tokens"],
        }
        existing = {
            "baseline_repo_sha": manifest.get("baseline_repo_sha"),
            "current_sha": manifest.get("repository", {}).get("current_sha"),
            "frozen_selector_hash": manifest.get("frozen_selector", {}).get(
                "config_sha256"
            ),
            "materialization_id": manifest.get("materialized_draft", {}).get(
                "materialization_id"
            ),
            "longbench_jsonl_sha256": manifest.get("longbench", {}).get("jsonl_sha256"),
            "margin_thresholds": manifest.get("configuration", {}).get(
                "margin_thresholds"
            ),
            "max_new_tokens": manifest.get("configuration", {}).get("max_new_tokens"),
        }
        if existing != compatibility:
            raise ValueError(
                f"resume provenance/configuration mismatch: {existing} != {compatibility}"
            )
        manifest["status"] = "started"
        manifest["error"] = None
        manifest.setdefault("execution_stages", []).append(
            shlex.join([sys.executable, *sys.argv])
        )
        manifest["configuration"]["run_exact_case"] = bool(
            manifest["configuration"].get("run_exact_case") or args.run_exact_case
        )
        manifest["configuration"]["run_cohort"] = bool(
            manifest["configuration"].get("run_cohort") or args.run_cohort
        )
        observations = read_jsonl(output / "subrun-index.jsonl")
        divergence_events = read_jsonl(output / "divergence-events.jsonl")
        replay_events = read_jsonl(output / "replay-semantic-events.jsonl")
        round_events = read_jsonl(output / "round-margin-events.jsonl")
    else:
        manifest = new_manifest
        output.mkdir(parents=True)
        observations = []
        divergence_events = []
        replay_events = []
        round_events = []
    raw_root = output / "raw-subruns"
    raw_root.mkdir(exist_ok=args.resume)
    existing_ids = {row["independent_repeat_observation_id"] for row in observations}
    duplicate_ids = existing_ids & {
        row["independent_repeat_observation_id"] for row in specs
    }
    if duplicate_ids:
        raise ValueError(f"resume requested duplicate subruns: {sorted(duplicate_ids)}")
    combined_specs = [
        *[
            {
                key: row[key]
                for key in (
                    "experiment",
                    "task",
                    "sample_id",
                    "source_sample_id",
                    "nominal_prompt_length",
                    "selector_mode",
                    "node_budget",
                    "repeat_index",
                    "max_new_tokens",
                    "physical_case_id",
                    "configuration_case_id",
                    "independent_repeat_observation_id",
                )
            }
            for row in observations
        ],
        *specs,
    ]
    manifest["count_semantics"].update(
        {
            "requested_subrun_count": len(combined_specs),
            "requested_physical_case_count": len(
                {row["physical_case_id"] for row in combined_specs}
            ),
            "requested_configuration_case_count": len(
                {row["configuration_case_id"] for row in combined_specs}
            ),
        }
    )
    exit_code = 1
    original_stdout, original_stderr = sys.stdout, sys.stderr
    log_mode = "a" if args.resume else "w"
    with (output / "terminal.log").open(log_mode, encoding="utf-8", buffering=1) as log:
        sys.stdout = TeeStream(original_stdout, log)
        sys.stderr = TeeStream(original_stderr, log)
        try:
            for index, identity in enumerate(specs, 1):
                subrun_dir = raw_root / safe_subrun_directory_name(identity)
                if subrun_dir.exists():
                    raise ValueError(f"subrun output already exists: {subrun_dir}")
                command = _command(args, identity, subrun_dir)
                manifest["subruns"].append(
                    {
                        **identity,
                        "raw_subrun_dir": str(subrun_dir),
                        "command": command,
                        "command_shell": shlex.join(command),
                        "return_code": None,
                        "outcome_category": None,
                    }
                )
                _write_json(output / "nr-manifest.json", manifest)
                print(
                    f"[{index}/{len(specs)}] {identity['independent_repeat_observation_id']}"
                )
                print(shlex.join(command))
                completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
                try:
                    observation = classify_subrun_artifacts(
                        subrun_dir,
                        identity=identity,
                        return_code=completed.returncode,
                    )
                except (ArtifactError, OSError, TypeError, ValueError) as exc:
                    observation = infrastructure_failure_observation(
                        identity,
                        output_dir=subrun_dir,
                        return_code=completed.returncode,
                        error=exc,
                    )
                observations.append(observation)
                manifest["subruns"][-1]["return_code"] = completed.returncode
                manifest["subruns"][-1]["outcome_category"] = observation[
                    "outcome_category"
                ]
                event = build_divergence_event(observation)
                if event is not None:
                    divergence_events.append(event)
                replay_events.extend(observation.get("_replay_rows", []))
                round_events.extend(extract_round_margin_events(observation))
                _persist_results(
                    output,
                    manifest,
                    observations,
                    divergence_events,
                    replay_events,
                    round_events,
                    args.margin_thresholds,
                )
                print(
                    f"return_code={completed.returncode} "
                    f"outcome={observation['outcome_category']}"
                )
                if observation["outcome_category"] in {
                    "incomplete_correctness_evidence",
                    "infrastructure_failure",
                }:
                    raise RuntimeError(
                        "NR study stopped on incomplete/infrastructure observation: "
                        f"{identity['independent_repeat_observation_id']}"
                    )
            manifest["status"] = "ok"
            manifest["completed_subrun_count"] = len(observations)
            exit_code = 0
        except KeyboardInterrupt:
            manifest["status"] = "incomplete"
            manifest["error"] = {
                "type": "KeyboardInterrupt",
                "message": "interrupted by user",
            }
            exit_code = 130
        except Exception as exc:  # noqa: BLE001 - persist partial diagnostic study.
            manifest["status"] = "incomplete"
            manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
            print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
        finally:
            manifest["completed_subrun_count"] = len(observations)
            _persist_results(
                output,
                manifest,
                observations,
                divergence_events,
                replay_events,
                round_events,
                args.margin_thresholds,
            )
            sys.stdout, sys.stderr = original_stdout, original_stderr
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
