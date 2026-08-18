"""Run Phase 2A target-rank or target-verification diagnostics on LongBench."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase1.backend import attention_backend_snapshot
from dflash.phase2a.rank import aggregate_rank_events
from dflash.phase2a.verification import aggregate_verification_samples
from experiments.phase0_smoke import (
    TeeStream,
    _pretrained_location_kwargs,
    _python_file,
    _require_cuda,
    _torch_dtype,
)
from experiments.phase1_observability import (
    _atomic_write_jsonl,
    _call_generation,
    _load_materialized_draft,
    _loaded_source_record,
    _token_equal,
)
from experiments.phase1b_longctx import (
    _load_tokenizer_only,
    _position_limit,
    _prepare_workload,
    _print_inventory,
    _workload_manifest_rows,
)
from experiments.phase1b_longctx import build_parser as build_phase1b_parser
from experiments.phase1b_longctx import new_manifest as phase1b_new_manifest
from experiments.phase1b_longctx import (
    run_configuration as phase1b_run_configuration,
)
from experiments.phase1b_longctx import validate_args as validate_phase1b_args
from scripts.collect_environment import collect_environment, write_json

SCHEMA_VERSION = "1.0"
RUN_MODES = ("rank", "verification-profile")
RANK_SUMMARY_FIELDS = (
    "task",
    "context_length",
    "sample_count",
    "round_count",
    "rounds_with_rejection_frontier",
    "full_block_accepted_round_count",
    "frontier_top2_recall_micro",
    "frontier_top4_recall_micro",
    "frontier_top8_recall_micro",
    "frontier_top16_recall_micro",
    "frontier_top2_recall_macro",
    "frontier_top4_recall_macro",
    "frontier_top8_recall_macro",
    "frontier_top16_recall_macro",
    "frontier_rank_median",
    "frontier_rank_p90",
    "frontier_rank_p95",
    "frontier_rank_over_16_fraction",
)
VERIFICATION_SUMMARY_FIELDS = (
    "task",
    "context_length",
    "sample_count",
    "verification_cuda_ms",
    "self_attention_module_cuda_ms",
    "mlp_module_cuda_ms",
    "lm_head_cuda_ms",
    "unattributed_cuda_ms",
    "self_attention_over_verification",
    "mlp_over_verification",
    "lm_head_over_verification",
    "draft_cuda_ms",
    "draft_over_round",
    "verification_over_round",
)


def parse_rank_top_ms(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rank Top-M values must be integers") from exc
    if not parsed or tuple(sorted(set(parsed))) != parsed or parsed[0] < 2:
        raise argparse.ArgumentTypeError(
            "rank Top-M values must be unique, increasing, and at least 2"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = build_phase1b_parser()
    parser.description = __doc__
    parser.add_argument("--run-mode", choices=RUN_MODES, default="rank")
    parser.add_argument(
        "--rank-top-ms", type=parse_rank_top_ms, default=(2, 4, 8, 16)
    )
    parser.add_argument("--rank-max-m", type=int, default=16)
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Run exactly one stable LongBench sample ID at every requested length.",
    )
    parser.set_defaults(
        output_dir=Path("outputs/phase2a"),
        timing_repeats=1,
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = validate_phase1b_args(build_parser().parse_args(argv))
    if args.inspect_workload_only:
        raise ValueError("Phase 2A generation modes do not support inventory-only")
    if args.run_mode == "rank" and (
        args.rank_max_m < 16 or args.rank_max_m < max(args.rank_top_ms)
    ):
        raise ValueError("--rank-max-m must be at least 16 and every --rank-top-ms")
    if args.sample_id is not None and args.sample_policy != "matched":
        raise ValueError("--sample-id requires --sample-policy matched")
    return args


def run_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **phase1b_run_configuration(args),
        "phase": "2A",
        "run_mode": args.run_mode,
        "rank_top_ms": list(args.rank_top_ms),
        "rank_max_m": args.rank_max_m,
        "sample_id": args.sample_id,
        "effective_sample_count": 1
        if args.sample_id is not None
        else args.pilot_samples,
        "rank_and_verification_profile_mutually_exclusive": True,
    }


def _sha256_token_ids(token_ids: list[int]) -> str:
    import hashlib

    canonical = json.dumps(token_ids, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(statistics.median(finite)) if finite else None


def _require_token_exact(
    torch_module: Any, control_ids: Any, diagnostic_ids: Any, *, label: str
) -> None:
    if not _token_equal(torch_module, control_ids, diagnostic_ids):
        raise RuntimeError(f"{label} changed output token IDs")


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _rank_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for group in summary["by_task_and_context_length"].values():
        micro = group["first_rejection_frontier_top_m_recall_micro"]
        macro = group["first_rejection_frontier_top_m_recall_macro"]
        rank = group["frontier_rank"]
        rows.append(
            {
                "task": group["task"],
                "context_length": group["context_length"],
                "sample_count": group["sample_count"],
                "round_count": group["round_count"],
                "rounds_with_rejection_frontier": group[
                    "rounds_with_rejection_frontier"
                ],
                "full_block_accepted_round_count": group[
                    "full_block_accepted_round_count"
                ],
                **{
                    f"frontier_top{top_m}_recall_micro": micro[
                        f"top_{top_m}_recall"
                    ]
                    for top_m in (2, 4, 8, 16)
                },
                **{
                    f"frontier_top{top_m}_recall_macro": macro[
                        f"top_{top_m}_recall"
                    ]
                    for top_m in (2, 4, 8, 16)
                },
                "frontier_rank_median": rank["median"],
                "frontier_rank_p90": rank["p90"],
                "frontier_rank_p95": rank["p95"],
                "frontier_rank_over_16_fraction": rank["over_16_fraction"],
            }
        )
    return rows


def _verification_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for group in summary["by_task_and_context_length"].values():
        metrics = group["metrics"]
        rows.append(
            {
                "task": group["task"],
                "context_length": group["context_length"],
                "sample_count": group["sample_count"],
                **{field: metrics[field]["median"] for field in metrics},
            }
        )
    return rows


def _new_manifest(
    args: argparse.Namespace, environment: dict[str, Any], command: str
) -> dict[str, Any]:
    manifest = phase1b_new_manifest(args, environment, command)
    manifest["configuration"] = run_configuration(args)
    manifest["measurement_policy"] = {
        "formal_end_to_end_latency_source": "instrumented_default_control",
        "rank_timing_included_in_formal_latency": False,
        "rank_and_verification_profile_mutually_exclusive": True,
        "output_token_ids_must_match_control": True,
        "cuda_event_resolution": "once at each generation boundary",
        "per_layer_or_per_round_cuda_synchronize": False,
        "module_self_attention_scope": (
            "QKV projections, attention/SDPA path, and output projection"
        ),
        "module_self_attention_is_pure_sdpa": False,
        "pure_sdpa_requires_nsight_or_ncu": True,
    }
    return manifest


def _load_inputs(args: argparse.Namespace, manifest: dict[str, Any], provenance: dict[str, Any]):
    tokenizer, target_source, transformers_version = _load_tokenizer_only(args)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("The target tokenizer does not define eos_token_id")
    prepared, inventory, _, _ = _prepare_workload(args, tokenizer, build_cohort=False)
    if getattr(args, "exclude_selector_training_sources", False):
        from dflash.phase1b.workload import build_workload_inventory

        frozen = getattr(args, "frozen_selector", None)
        if frozen is None:
            raise ValueError("training-source exclusion requires a frozen selector")
        training_ids = set(frozen["training_source_ids"])
        total_source_ids = {
            f"{args.longbench_task}::{row['source_sample_id']}" for row in prepared
        }
        excluded = sorted(total_source_ids & training_ids)
        prepared = [
            row
            for row in prepared
            if f"{args.longbench_task}::{row['source_sample_id']}" not in training_ids
        ]
        inventory = build_workload_inventory(
            prepared,
            task=args.longbench_task,
            nominal_lengths=args.context_lengths,
            requested_sample_count=args.pilot_samples,
        )
        inventory["selector_training_source_exclusion"] = {
            "enabled": True,
            "total_eligible_source_ids_before_length_filter": sorted(total_source_ids),
            "excluded_training_source_ids": excluded,
            "remaining_independent_source_ids": sorted(
                f"{args.longbench_task}::{row['source_sample_id']}" for row in prepared
            ),
            "training_sources_reintroduced": False,
            "completed_before_model_loading": True,
        }
    explicit_matches = (
        [row for row in prepared if row["sample_id"] == args.sample_id]
        if args.sample_id is not None
        else []
    )
    inventory["explicit_sample_selection"] = {
        "requested_sample_id": args.sample_id,
        "enabled": args.sample_id is not None,
        "validated": False,
        "matching_record_count": len(explicit_matches),
    }
    if len(explicit_matches) == 1:
        selected = explicit_matches[0]
        inventory["explicit_sample_selection"].update(
            {
                "source_sample_id": selected["source_sample_id"],
                "actual_max_templated_prompt_length": selected[
                    "source_full_prompt_length"
                ],
                "requested_context_lengths": list(args.context_lengths),
            }
        )
    manifest["workload"]["inventory"] = inventory
    _print_inventory(
        inventory, show_all_sample_ids=args.show_all_eligible_sample_ids
    )
    from dflash.phase1b.workload import build_longbench_workloads

    workloads, cohort = build_longbench_workloads(
        prepared,
        tokenizer=tokenizer,
        task=args.longbench_task,
        nominal_lengths=args.context_lengths,
        sample_count=args.pilot_samples,
        sample_policy=args.sample_policy,
        max_new_tokens=args.max_new_tokens,
        model_position_limit=max(args.context_lengths) + args.max_new_tokens,
        sample_id=args.sample_id,
    )
    if args.sample_id is not None:
        inventory["explicit_sample_selection"]["validated"] = True
    manifest["workload"]["cohort"] = cohort
    manifest["workload"]["sample_ids"] = sorted(
        {row["sample_id"] for row in workloads}
    )
    manifest["workload"]["sample_ids_by_length"] = cohort[
        "sample_ids_by_length"
    ]
    manifest["workload"]["samples"] = _workload_manifest_rows(workloads)
    print(
        f"Workload preflight passed before model loading: task={args.longbench_task} "
        f"policy={args.sample_policy} records={len(workloads)}"
    )

    torch_module = _require_cuda(args)
    try:
        from transformers import AutoModel, AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("Transformers backend is not installed") from exc
    dtype = _torch_dtype(torch_module, args.dtype)
    target = AutoModelForCausalLM.from_pretrained(
        target_source["load_source"],
        trust_remote_code=True,
        dtype=dtype,
        device_map=args.device,
        **_pretrained_location_kwargs(target_source),
    ).eval()
    target_source["class_name"] = (
        f"{target.__class__.__module__}.{target.__class__.__qualname__}"
    )
    target_source["python_file"] = _python_file(target)
    draft, draft_manifest = _load_materialized_draft(
        AutoModel,
        model_path=args.materialized_draft_path,
        implementation="instrumented",
        dtype=dtype,
        device=args.device,
    )
    if int(draft.block_size) != args.block_size:
        raise RuntimeError(
            f"Loaded draft block size {draft.block_size} does not equal "
            f"{args.block_size}"
        )
    provenance.update(_loaded_source_record(draft, draft_manifest))
    provenance["target"] = target_source
    provenance["status"] = "ok"
    position_limit = _position_limit(target, tokenizer)
    for workload in workloads:
        if workload["actual_prompt_length"] + args.max_new_tokens > position_limit:
            raise RuntimeError(
                f"Sample {workload['sample_id']} exceeds model position limit"
            )
    manifest["model_position_limit"] = position_limit
    manifest["attention_backend"] = attention_backend_snapshot(
        torch_module, target=target, draft=draft
    )
    print(f"Transformers: {transformers_version}")
    print(f"GPU: {torch_module.cuda.get_device_name(args.device)}")
    print(f"Instrumented DFlash module: {provenance['actual_loaded_module_path']}")
    return torch_module, tokenizer, target, draft, workloads


def _warmup(
    args: argparse.Namespace,
    torch_module: Any,
    tokenizer: Any,
    target: Any,
    draft: Any,
    workloads: list[dict[str, Any]],
) -> None:
    first_by_length = {}
    for workload in workloads:
        first_by_length.setdefault(workload["nominal_prompt_length"], workload)
    for nominal_length in args.context_lengths:
        input_ids = torch_module.tensor(
            [first_by_length[nominal_length]["input_ids"]],
            dtype=torch_module.long,
            device=args.device,
        )
        for _ in range(args.warmup_runs):
            draft.spec_generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                target=target,
                stop_token_ids=[tokenizer.eos_token_id],
            )
        if args.warmup_runs:
            torch_module.cuda.synchronize(input_ids.device)


def _run_workload(
    args: argparse.Namespace,
    *,
    torch_module: Any,
    tokenizer: Any,
    target: Any,
    draft: Any,
    workload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    input_ids = torch_module.tensor(
        [workload["input_ids"]], dtype=torch_module.long, device=args.device
    )
    if int(input_ids.shape[1]) != workload["actual_prompt_length"]:
        raise RuntimeError("Persisted workload token count changed before generation")
    rank_events: list[dict[str, Any]] = []
    module_events: list[dict[str, Any]] = []
    verification_rounds: list[dict[str, Any]] = []
    control_timings = []
    profile_repeats = []
    reference_ids = None

    for repeat in range(args.timing_repeats):
        control_ids, _, control_timing, _ = _call_generation(
            torch_module=torch_module,
            draft=draft,
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            return_diagnostics=False,
            enable_profiling=False,
        )
        options = (
            {
                "enable_rank_diagnostics": True,
                "rank_top_ms": list(args.rank_top_ms),
                "rank_max_m": args.rank_max_m,
            }
            if args.run_mode == "rank"
            else {"enable_verification_profiling": True}
        )
        observed_ids, diagnostics, observed_timing, _ = _call_generation(
            torch_module=torch_module,
            draft=draft,
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            return_diagnostics=True,
            enable_profiling=args.run_mode == "verification-profile",
            generation_options=options,
        )
        _require_token_exact(
            torch_module,
            control_ids,
            observed_ids,
            label=f"{args.run_mode} diagnostic for {workload['sample_id']}",
        )
        if reference_ids is not None:
            _require_token_exact(
                torch_module,
                reference_ids,
                control_ids,
                label="greedy Phase 2A repeat",
            )
        reference_ids = control_ids
        control_timings.append(
            {
                "repeat": repeat,
                "cuda_elapsed_ms": control_timing["cuda_elapsed_ms"],
                "cpu_wall_elapsed_ms": control_timing["cpu_wall_elapsed_ms"],
                "formal_end_to_end": True,
            }
        )
        common = {
            "task": workload["longbench_task"],
            "sample_id": workload["sample_id"],
            "source_sample_id": workload["source_sample_id"],
            "nominal_prompt_length": workload["nominal_prompt_length"],
            "actual_prompt_length": workload["actual_prompt_length"],
            "diagnostic_run": repeat,
        }
        if args.run_mode == "rank":
            current = diagnostics["rank_diagnostics"]["events"]
            expected = len(diagnostics["rounds"]) * (args.block_size - 1)
            if len(current) != expected:
                raise RuntimeError("Rank events included a bonus or omitted draft positions")
            rank_events.extend({**common, **event} for event in current)
            profile_repeats.append(
                {
                    "repeat": repeat,
                    "formal_end_to_end": False,
                    "timing_reported": False,
                    "reason": "rank instrumentation is excluded from latency",
                }
            )
        else:
            module_profile = diagnostics["target_module_profiling"]
            for stage in ("prefill", "verification"):
                module_events.extend(
                    {**common, **event}
                    for event in module_profile[stage]["records"]
                )
            for round_record in diagnostics["rounds"]:
                module_summary = round_record["target_module_profile"]["summary"]
                verification_rounds.append(
                    {
                        **common,
                        "round_index": round_record["round_index"],
                        "stop_reason": round_record.get("stop_reason"),
                        "truncated_token_count": round_record.get(
                            "truncated_token_count"
                        ),
                        "committed_token_count": round_record.get(
                            "committed_token_count"
                        ),
                        "returned_token_delta": round_record.get(
                            "returned_token_delta"
                        ),
                        **module_summary,
                        "draft_cuda_ms": round_record["timing"][
                            "draft_cuda_elapsed_ms"
                        ],
                        "round_cuda_ms": round_record["timing"][
                            "round_cuda_elapsed_ms"
                        ],
                        "draft_over_round": (
                            round_record["timing"]["draft_cuda_elapsed_ms"]
                            / round_record["timing"]["round_cuda_elapsed_ms"]
                            if round_record["timing"]["round_cuda_elapsed_ms"]
                            else None
                        ),
                        "verification_over_round": (
                            round_record["timing"]["verification_cuda_elapsed_ms"]
                            / round_record["timing"]["round_cuda_elapsed_ms"]
                            if round_record["timing"]["round_cuda_elapsed_ms"]
                            else None
                        ),
                    }
                )
            generation_summary = module_profile["verification"]["summary"]
            draft_ms = sum(
                row["timing"]["draft_cuda_elapsed_ms"] or 0.0
                for row in diagnostics["rounds"]
            )
            round_ms = sum(
                row["timing"]["round_cuda_elapsed_ms"] or 0.0
                for row in diagnostics["rounds"]
            )
            profile_repeats.append(
                {
                    "repeat": repeat,
                    "verification_cuda_ms": generation_summary[
                        "target_verification_cuda_ms"
                    ],
                    "self_attention_module_cuda_ms": generation_summary[
                        "target_self_attention_module_cuda_ms"
                    ],
                    "mlp_module_cuda_ms": generation_summary[
                        "target_mlp_module_cuda_ms"
                    ],
                    "lm_head_cuda_ms": generation_summary[
                        "target_lm_head_cuda_ms"
                    ],
                    "unattributed_cuda_ms": generation_summary[
                        "target_unattributed_cuda_ms"
                    ],
                    "self_attention_over_verification": generation_summary[
                        "target_self_attention_over_verification"
                    ],
                    "mlp_over_verification": generation_summary[
                        "target_mlp_over_verification"
                    ],
                    "lm_head_over_verification": generation_summary[
                        "target_lm_head_over_verification"
                    ],
                    "draft_cuda_ms": draft_ms,
                    "draft_over_round": draft_ms / round_ms if round_ms else None,
                    "verification_over_round": (
                        generation_summary["target_verification_cuda_ms"] / round_ms
                        if round_ms
                        else None
                    ),
                    "partition_reliable": generation_summary["partition_reliable"],
                    "per_layer": generation_summary["per_layer"],
                    "module_call_counts": generation_summary[
                        "module_call_counts"
                    ],
                    "prefill_module_profile": module_profile["prefill"]["summary"],
                    "observed_cpu_wall_elapsed_ms": observed_timing[
                        "cpu_wall_elapsed_ms"
                    ],
                    "formal_end_to_end": False,
                }
            )

    generated_ids = reference_ids[0, input_ids.shape[1] :].tolist()
    sample = {
        "schema_version": SCHEMA_VERSION,
        "run_mode": args.run_mode,
        "task": workload["longbench_task"],
        "sample_id": workload["sample_id"],
        "source_sample_id": workload["source_sample_id"],
        "explicit_sample_id": workload.get("explicit_sample_id"),
        "nominal_prompt_length": workload["nominal_prompt_length"],
        "actual_prompt_length": workload["actual_prompt_length"],
        "generated_token_count": len(generated_ids),
        "generated_output_token_ids": generated_ids,
        "generated_output_token_ids_sha256": _sha256_token_ids(generated_ids),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "control_diagnostic_output_token_ids_equal": True,
        "control_timing": control_timings,
        "diagnostic_repeats": profile_repeats,
    }
    if args.run_mode == "verification-profile":
        for field in VERIFICATION_SUMMARY_FIELDS[3:]:
            sample[field] = _median([row.get(field) for row in profile_repeats])
        sample["partition_reliable"] = all(
            row["partition_reliable"] for row in profile_repeats
        )
    return sample, rank_events, module_events, verification_rounds


def run_phase2a(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    samples: list[dict[str, Any]],
    rank_events: list[dict[str, Any]],
    module_events: list[dict[str, Any]],
    verification_rounds: list[dict[str, Any]],
) -> dict[str, Any]:
    torch_module, tokenizer, target, draft, workloads = _load_inputs(
        args, manifest, provenance
    )
    _warmup(args, torch_module, tokenizer, target, draft, workloads)
    for index, workload in enumerate(workloads, 1):
        print(
            f"[{index}/{len(workloads)}] task={workload['longbench_task']} "
            f"sample={workload['sample_id']} nominal={workload['nominal_prompt_length']}"
        )
        sample, current_rank, current_modules, current_rounds = _run_workload(
            args,
            torch_module=torch_module,
            tokenizer=tokenizer,
            target=target,
            draft=draft,
            workload=workload,
        )
        samples.append(sample)
        rank_events.extend(current_rank)
        module_events.extend(current_modules)
        verification_rounds.extend(current_rounds)
    manifest["status"] = "ok"
    if args.run_mode == "rank":
        summary_top_ms = sorted({*args.rank_top_ms, 2, 4, 8, 16})
        aggregate = aggregate_rank_events(rank_events, top_ms=summary_top_ms)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "run_mode": args.run_mode,
            "event_count": len(rank_events),
            "aggregates": aggregate,
            "error": None,
        }
    aggregate = aggregate_verification_samples(samples, module_events)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "run_mode": args.run_mode,
        "sample_count": len(samples),
        "round_count": len(verification_rounds),
        "aggregates": aggregate,
        "error": None,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    common_paths = {
        "manifest_json": output_dir / "manifest.json",
        "workload_inventory_json": output_dir / "workload-inventory.json",
        "samples_jsonl": output_dir / "samples.jsonl",
        "environment_json": output_dir / "environment.json",
        "source_provenance_json": output_dir / "source-provenance.json",
        "terminal_log": output_dir / "terminal.log",
    }
    mode_paths = (
        {
            "rank_events_jsonl": output_dir / "rank-events.jsonl",
            "rank_summary_json": output_dir / "rank-summary.json",
            "rank_summary_csv": output_dir / "rank-summary.csv",
        }
        if args.run_mode == "rank"
        else {
            "verification_module_events_jsonl": output_dir
            / "verification-module-events.jsonl",
            "verification_profile_rounds_jsonl": output_dir
            / "verification-profile-rounds.jsonl",
            "verification_profile_summary_json": output_dir
            / "verification-profile-summary.json",
            "verification_profile_summary_csv": output_dir
            / "verification-profile-summary.csv",
        }
    )
    paths = {**common_paths, **mode_paths}
    command = shlex.join([sys.executable, *sys.argv])
    environment = collect_environment(
        run_config=run_configuration(args), run_command=command
    )
    manifest = _new_manifest(args, environment, command)
    manifest["artifacts"] = {name: str(path) for name, path in paths.items()}
    provenance = {"schema_version": SCHEMA_VERSION, "status": "started", "error": None}
    samples: list[dict[str, Any]] = []
    rank_events: list[dict[str, Any]] = []
    module_events: list[dict[str, Any]] = []
    verification_rounds: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "started",
        "run_mode": args.run_mode,
        "aggregates": None,
        "error": None,
    }
    if args.run_mode == "rank":
        summary["event_count"] = 0
    else:
        summary["sample_count"] = 0
        summary["round_count"] = 0
    write_json(paths["environment_json"], environment)
    exit_code = 1
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with paths["terminal_log"].open("w", encoding="utf-8", buffering=1) as log:
        sys.stdout = TeeStream(original_stdout, log)
        sys.stderr = TeeStream(original_stderr, log)
        try:
            print(f"Phase 2A output directory: {output_dir}")
            summary = run_phase2a(
                args,
                manifest,
                provenance,
                samples,
                rank_events,
                module_events,
                verification_rounds,
            )
            exit_code = 0
        except KeyboardInterrupt:
            error = {"type": "KeyboardInterrupt", "message": "Interrupted by user"}
            manifest["status"] = provenance["status"] = summary["status"] = "error"
            manifest["error"] = provenance["error"] = summary["error"] = error
            exit_code = 130
            traceback.print_exc()
        except Exception as exc:  # noqa: BLE001 - persist partial Phase 2A artifacts.
            error = {"type": type(exc).__name__, "message": str(exc)}
            manifest["status"] = provenance["status"] = summary["status"] = "error"
            manifest["error"] = provenance["error"] = summary["error"] = error
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
        finally:
            try:
                write_json(paths["manifest_json"], manifest)
                inventory = manifest["workload"].get("inventory") or {
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "task": args.longbench_task,
                    "record_count": 0,
                    "eligible_by_prompt_length": {},
                    "sample_ids_reaching_maximum_length": [],
                    "source_sample_ids_reaching_maximum_length": [],
                    "explicit_sample_selection": {
                        "requested_sample_id": args.sample_id,
                        "enabled": args.sample_id is not None,
                        "validated": False,
                    },
                    "error": manifest["error"],
                }
                write_json(paths["workload_inventory_json"], inventory)
                _atomic_write_jsonl(paths["samples_jsonl"], samples)
                write_json(paths["source_provenance_json"], provenance)
                if args.run_mode == "rank":
                    summary["event_count"] = len(rank_events)
                    _atomic_write_jsonl(paths["rank_events_jsonl"], rank_events)
                    write_json(paths["rank_summary_json"], summary)
                    rows = (
                        _rank_csv_rows(summary["aggregates"])
                        if summary.get("aggregates")
                        else []
                    )
                    _write_csv(paths["rank_summary_csv"], RANK_SUMMARY_FIELDS, rows)
                else:
                    summary["sample_count"] = len(samples)
                    summary["round_count"] = len(verification_rounds)
                    _atomic_write_jsonl(
                        paths["verification_module_events_jsonl"], module_events
                    )
                    _atomic_write_jsonl(
                        paths["verification_profile_rounds_jsonl"],
                        verification_rounds,
                    )
                    write_json(paths["verification_profile_summary_json"], summary)
                    rows = (
                        _verification_csv_rows(summary["aggregates"])
                        if summary.get("aggregates")
                        else []
                    )
                    _write_csv(
                        paths["verification_profile_summary_csv"],
                        VERIFICATION_SUMMARY_FIELDS,
                        rows,
                    )
                for name, path in paths.items():
                    print(f"{name}: {path}")
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
