#!/usr/bin/env python3
"""Run or inventory a Phase 1B.1 LongBench long-context DFlash workload."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import sys
import traceback
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase1.backend import attention_backend_snapshot
from dflash.phase1.materialize import DRAFT_MODEL_ID, DRAFT_REVISION
from dflash.phase1b.metrics import (
    aggregate_sweep,
    distribution,
    summarize_profile_rounds,
)
from dflash.phase1b.workload import (
    DEFAULT_CONTEXT_LENGTHS,
    LONG_BENCH_TASKS,
    build_longbench_workloads,
    build_workload_inventory,
    load_longbench_records,
    prepare_longbench_records,
    prompt_template_metadata,
    resolve_jsonl_path,
)
from experiments.phase0_smoke import (
    TeeStream,
    _pretrained_location_kwargs,
    _python_file,
    _require_cuda,
    _resolve_model_source,
    _torch_dtype,
)
from experiments.phase1_observability import (
    _atomic_write_jsonl,
    _call_generation,
    _enrich_rounds,
    _load_materialized_draft,
    _loaded_source_record,
    _token_equal,
)
from scripts.collect_environment import (
    DEFAULT_TARGET_MODEL,
    DEFAULT_TARGET_REVISION,
    collect_environment,
    write_json,
)

SCHEMA_VERSION = "1.0"

SUMMARY_CSV_FIELDS = (
    "nominal_prompt_length",
    "sample_count",
    "control_total_cuda_median_ms",
    "control_total_cuda_mean_ms",
    "control_total_cuda_p90_ms",
    "generated_tokens_per_second_median",
    "profile_prefill_cuda_median_ms",
    "profile_decode_cuda_median_ms",
    "profile_draft_cuda_median_ms",
    "profile_combined_attention_cuda_median_ms",
    "profile_verification_cuda_median_ms",
    "profile_round_cuda_median_ms",
    "profile_draft_over_round_median",
    "profile_verification_over_round_median",
    "profile_combined_attention_over_draft_median",
    "profile_combined_attention_over_round_median",
    "mean_non_terminal_accepted_draft_tokens",
    "mean_non_terminal_committed_tokens",
    "round_count",
    "acceptance_eligible_round_count",
    "truncated_token_count",
    "micro_raw_all_round_accepted_over_proposed",
    "micro_non_terminal_accepted_over_proposed",
    "macro_non_terminal_acceptance_rate",
    "macro_effective_returned_tokens_per_round",
)


def parse_context_lengths(value: str) -> tuple[int, ...]:
    parsed = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        multiplier = 1024 if item.endswith("k") else 1
        number = item[:-1] if multiplier == 1024 else item
        try:
            parsed.append(int(number) * multiplier)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid context length: {item!r}"
            ) from exc
    if not parsed or tuple(sorted(set(parsed))) != tuple(parsed):
        raise argparse.ArgumentTypeError(
            "Context lengths must be unique, positive, and strictly increasing"
        )
    if parsed[0] <= 0:
        raise argparse.ArgumentTypeError("Context lengths must be positive")
    return tuple(parsed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--target-revision", default=DEFAULT_TARGET_REVISION)
    parser.add_argument("--target-model-path", type=Path, default=None)
    parser.add_argument("--draft-model", default=DRAFT_MODEL_ID)
    parser.add_argument("--draft-revision", default=DRAFT_REVISION)
    parser.add_argument("--materialized-draft-path", type=Path, default=None)
    parser.add_argument("--longbench-root", type=Path, default=None)
    parser.add_argument(
        "--longbench-task", choices=LONG_BENCH_TASKS, default="gov_report"
    )
    parser.add_argument("--dataset-jsonl", type=Path, default=None)
    parser.add_argument("--govreport-jsonl", type=Path, default=None)
    parser.add_argument("--govreport-dataset-path", type=Path, default=None)
    parser.add_argument("--govreport-dataset-cache", type=Path, default=None)
    parser.add_argument("--dataset-name", default="ccdv/govreport-summarization")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--context-field", default=None)
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--query-field", default=None)
    parser.add_argument("--answers-field", default=None)
    parser.add_argument("--id-field", default=None)
    parser.add_argument(
        "--sample-policy", choices=("matched", "per-length"), default="matched"
    )
    parser.add_argument("--inspect-workload-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--context-lengths",
        type=parse_context_lengths,
        default=DEFAULT_CONTEXT_LENGTHS,
        metavar="LENGTHS",
        help="Comma-separated token lengths; K suffix means 1024 (default: 1K,4K,8K,16K,32K).",
    )
    parser.add_argument("--pilot-samples", type=int, default=8)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/phase1b-longctx")
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    direct_jsonl = resolve_jsonl_path(
        task=args.longbench_task,
        longbench_root=args.longbench_root,
        dataset_jsonl=args.dataset_jsonl,
        govreport_jsonl=args.govreport_jsonl,
    )
    source_count = sum(
        (
            direct_jsonl is not None,
            args.govreport_dataset_path is not None,
            args.govreport_dataset_cache is not None,
        )
    )
    if source_count != 1:
        raise ValueError(
            "Choose exactly one LongBench JSONL/root or backward-compatible local "
            "Hugging Face dataset source"
        )
    if args.longbench_task != "gov_report" and (
        args.govreport_dataset_path is not None
        or args.govreport_dataset_cache is not None
    ):
        raise ValueError(
            "--govreport-dataset-path and --govreport-dataset-cache are "
            "backward-compatible GovReport sources only"
        )
    if args.context_field and args.text_field and args.context_field != args.text_field:
        raise ValueError("--context-field and compatibility --text-field disagree")
    if not args.inspect_workload_only and args.materialized_draft_path is None:
        raise ValueError("Generation mode requires --materialized-draft-path")
    if args.pilot_samples <= 0:
        raise ValueError("--pilot-samples must be positive")
    if args.timing_repeats <= 0:
        raise ValueError("--timing-repeats must be positive")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.batch_size != 1 or args.temperature != 0.0 or args.block_size != 16:
        raise ValueError(
            "Phase 1B freezes batch_size=1, temperature=0, and block_size=16"
        )
    if args.draft_model != DRAFT_MODEL_ID or args.draft_revision != DRAFT_REVISION:
        raise ValueError("Phase 1B is restricted to the pinned DFlash draft revision")
    return args


def run_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target_model": args.target_model,
        "target_revision": args.target_revision,
        "target_model_path": str(args.target_model_path)
        if args.target_model_path
        else None,
        "draft_model": args.draft_model,
        "draft_revision": args.draft_revision,
        "materialized_draft_path": (
            str(args.materialized_draft_path) if args.materialized_draft_path else None
        ),
        "dataset_source": {
            "longbench_root": (
                str(args.longbench_root) if args.longbench_root else None
            ),
            "longbench_task": args.longbench_task,
            "dataset_jsonl": str(args.dataset_jsonl) if args.dataset_jsonl else None,
            "jsonl_path": str(args.govreport_jsonl) if args.govreport_jsonl else None,
            "dataset_path": (
                str(args.govreport_dataset_path)
                if args.govreport_dataset_path
                else None
            ),
            "dataset_cache_dir": (
                str(args.govreport_dataset_cache)
                if args.govreport_dataset_cache
                else None
            ),
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "split": args.dataset_split,
            "context_field": args.context_field,
            "text_field": args.text_field,
            "query_field": args.query_field,
            "answers_field": args.answers_field,
            "id_field": args.id_field,
        },
        "sample_policy": args.sample_policy,
        "paired_context_length_comparison": args.sample_policy == "matched",
        "inspect_workload_only": args.inspect_workload_only,
        "context_lengths": list(args.context_lengths),
        "pilot_samples": args.pilot_samples,
        "timing_repeats": args.timing_repeats,
        "warmup_runs_per_context_length": args.warmup_runs,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "block_size": args.block_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "trust_remote_code": True,
        "backend": "transformers",
        "output_dir": str(args.output_dir),
    }


def new_manifest(
    args: argparse.Namespace, environment: dict[str, Any], command: str
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "repository": environment["repository"],
        "configuration": run_configuration(args),
        "workload": {
            "dataset": "LongBench",
            "task": args.longbench_task,
            "split": args.dataset_split,
            "seed": args.seed,
            "prompt_template": prompt_template_metadata(args.longbench_task),
            "sample_policy": args.sample_policy,
            "paired_context_length_comparison": args.sample_policy == "matched",
            "sample_ids": [],
            "sample_ids_by_length": {},
            "cohort": None,
            "inventory": None,
            "samples": [],
            "raw_dataset_in_repository": False,
            "filler_or_repeated_text_used": False,
        },
        "measurement_policy": {
            "formal_end_to_end_latency_source": "instrumented_default_control",
            "component_timing_source": "diagnostics_profiling_path",
            "pair_order": ["instrumented_default_control", "diagnostics_profiling"],
            "output_token_ids_must_match": True,
            "cuda_event_resolution": "once at each generation boundary",
            "per_layer_or_per_round_cuda_synchronize": False,
            "combined_attention_scope": (
                "actual SDPA call over persistent context KV and current block KV"
            ),
            "injected_kv_only_latency_reported": False,
        },
        "accounting_policy": {
            "accepted_draft_tokens": "accepted draft proposals; seed excluded",
            "committed_tokens": "accepted draft proposals plus the seed",
            "raw_all_round_metrics": "retain every executed round",
            "primary_acceptance_scope": "non-terminal, non-truncated rounds",
            "terminal_and_truncation_metrics": "reported separately",
        },
        "model_position_limit": None,
        "attention_backend": None,
        "artifacts": {},
        "error": None,
    }


def new_summary() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "started",
        "sample_length_record_count": 0,
        "profiling_round_record_count": 0,
        "aggregates": None,
        "error": None,
    }


def _resolved_jsonl(args: argparse.Namespace) -> Path | None:
    return resolve_jsonl_path(
        task=args.longbench_task,
        longbench_root=args.longbench_root,
        dataset_jsonl=args.dataset_jsonl,
        govreport_jsonl=args.govreport_jsonl,
    )


def _load_tokenizer_only(args: argparse.Namespace) -> tuple[Any, dict[str, Any], str]:
    """Load only the target tokenizer; never instantiate a model or require CUDA."""

    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Transformers tokenizer support is not installed") from exc

    target_source = _resolve_model_source(
        "target", args.target_model, args.target_revision, args.target_model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        target_source["load_source"],
        trust_remote_code=True,
        **_pretrained_location_kwargs(target_source),
    )
    return tokenizer, target_source, transformers.__version__


def _prepare_workload(
    args: argparse.Namespace, tokenizer: Any, *, build_cohort: bool
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]] | None,
    dict[str, Any] | None,
]:
    """Load, validate, inventory, and optionally select before any model load."""

    records = load_longbench_records(
        jsonl_path=_resolved_jsonl(args),
        dataset_path=args.govreport_dataset_path,
        dataset_cache_dir=args.govreport_dataset_cache,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
    )
    prepared = prepare_longbench_records(
        records,
        tokenizer=tokenizer,
        task=args.longbench_task,
        split=args.dataset_split,
        seed=args.seed,
        context_field=args.context_field,
        text_field=args.text_field,
        query_field=args.query_field,
        answers_field=args.answers_field,
        id_field=args.id_field,
    )
    inventory = build_workload_inventory(
        prepared,
        task=args.longbench_task,
        nominal_lengths=args.context_lengths,
        requested_sample_count=args.pilot_samples,
    )
    inventory.update(
        {
            "status": "ok",
            "mode": (
                "inspect_workload_only"
                if args.inspect_workload_only
                else "generation_preflight"
            ),
            "models_loaded": False,
            "cuda_required": False,
            "error": None,
            "dataset_jsonl": (
                str(_resolved_jsonl(args)) if _resolved_jsonl(args) else None
            ),
            "source_record_count": len(records),
            "skipped_record_count": len(records) - len(prepared),
            "dataset_split": args.dataset_split,
            "requested_context_lengths": list(args.context_lengths),
            "requested_sample_count": args.pilot_samples,
            "field_mapping": {
                "context": args.context_field or args.text_field or "auto: context",
                "query": args.query_field or "auto: input",
                "answers": args.answers_field or "auto: answers",
                "sample_id": args.id_field or "auto: _id/id/hash",
            },
        }
    )
    if not build_cohort:
        return prepared, inventory, None, None

    provisional_position_limit = max(args.context_lengths) + args.max_new_tokens
    workloads, cohort = build_longbench_workloads(
        prepared,
        tokenizer=tokenizer,
        task=args.longbench_task,
        nominal_lengths=args.context_lengths,
        sample_count=args.pilot_samples,
        sample_policy=args.sample_policy,
        max_new_tokens=args.max_new_tokens,
        model_position_limit=provisional_position_limit,
    )
    return prepared, inventory, workloads, cohort


def _print_inventory(inventory: dict[str, Any]) -> None:
    print(f"LongBench task: {inventory['task']}")
    print(
        "Records: "
        f"source={inventory.get('source_record_count', inventory['record_count'])} "
        f"usable={inventory['record_count']} "
        f"skipped={inventory.get('skipped_record_count', 0)}"
    )
    print("Prompt length | Eligible samples")
    print("--------------+-----------------")
    for length, details in inventory["eligible_by_prompt_length"].items():
        print(f"{int(length):>13,} | {details['eligible_sample_count']:>16,}")
    lengths = inventory["document_token_length"]
    print(
        "Document tokens: "
        f"min={lengths['min']} median={lengths['median']} "
        f"p90={lengths['p90']} max={lengths['max']}"
    )
    print("Sample IDs reaching maximum requested length:")
    maximum_ids = inventory["sample_ids_reaching_maximum_length"]
    maximum_source_ids = inventory.get(
        "source_sample_ids_reaching_maximum_length", []
    )
    if maximum_ids:
        for sample_id, source_id in zip(maximum_ids, maximum_source_ids, strict=True):
            print(f"  {sample_id} (source={source_id})")
    else:
        print("  (none)")
    matched = inventory["recommendations"]["matched"]
    print(
        "Recommended matched pilot: "
        f"samples={matched['sample_count']} lengths={matched['context_lengths']}"
    )
    per_length = inventory["recommendations"]["per_length"][
        "sample_count_by_length"
    ]
    print(f"Recommended per-length sample counts: {per_length}")


def _sha256_token_ids(token_ids: Sequence[int]) -> str:
    canonical = json.dumps(list(token_ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _profile_components(diagnostics: dict[str, Any]) -> dict[str, float | None]:
    rounds = diagnostics["rounds"]
    draft = sum(row["timing"]["draft_cuda_elapsed_ms"] or 0.0 for row in rounds)
    combined = sum(
        row["timing"]["combined_attention_cuda_elapsed_ms"] or 0.0 for row in rounds
    )
    verification = sum(
        row["timing"]["verification_cuda_elapsed_ms"] or 0.0 for row in rounds
    )
    decode = sum(row["timing"]["round_cuda_elapsed_ms"] or 0.0 for row in rounds)
    return {
        "prefill_cuda_elapsed_ms": diagnostics["profiling"]["prefill"][
            "cuda_elapsed_ms"
        ],
        "decode_cuda_elapsed_ms": decode,
        "total_cuda_elapsed_ms": diagnostics["profiling"]["total"]["cuda_elapsed_ms"],
        "draft_cuda_elapsed_ms": draft,
        "combined_attention_cuda_elapsed_ms": combined,
        "verification_cuda_elapsed_ms": verification,
        "combined_attention_over_draft_ratio": combined / draft if draft else None,
        "combined_attention_over_round_ratio": combined / decode if decode else None,
        "draft_over_round_ratio": draft / decode if decode else None,
        "verification_over_round_ratio": verification / decode if decode else None,
    }


def _timing_distributions(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "prefill_cuda_elapsed_ms",
        "decode_cuda_elapsed_ms",
        "total_cuda_elapsed_ms",
        "draft_cuda_elapsed_ms",
        "combined_attention_cuda_elapsed_ms",
        "verification_cuda_elapsed_ms",
        "combined_attention_over_draft_ratio",
        "combined_attention_over_round_ratio",
        "draft_over_round_ratio",
        "verification_over_round_ratio",
    )
    return {
        field: distribution(record.get(field) for record in records) for field in fields
    }


def _position_limit(target: Any, tokenizer: Any) -> int:
    configured = getattr(target.config, "max_position_embeddings", None)
    if configured and int(configured) < 10**9:
        return int(configured)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if not tokenizer_limit or int(tokenizer_limit) >= 10**9:
        raise RuntimeError("Could not determine a finite model position limit")
    return int(tokenizer_limit)


def _workload_manifest_rows(
    workloads: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    omitted = {"prompt_text", "input_ids"}
    return [
        {key: value for key, value in row.items() if key not in omitted}
        for row in workloads
    ]


def _write_summary_csv(path: Path, aggregates: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        for length, group in aggregates["by_context_length"].items():
            sample = group["sample_level"]
            control = sample["control_timing.total_cuda_elapsed_ms.median"]
            writer.writerow(
                {
                    "nominal_prompt_length": length,
                    "sample_count": group["sample_count"],
                    "control_total_cuda_median_ms": control["median"],
                    "control_total_cuda_mean_ms": control["mean"],
                    "control_total_cuda_p90_ms": control["p90"],
                    "generated_tokens_per_second_median": sample[
                        "control_timing.generated_tokens_per_second.median"
                    ]["median"],
                    "profile_prefill_cuda_median_ms": sample[
                        "profiling_timing.prefill_cuda_elapsed_ms.median"
                    ]["median"],
                    "profile_decode_cuda_median_ms": sample[
                        "profiling_timing.decode_cuda_elapsed_ms.median"
                    ]["median"],
                    "profile_draft_cuda_median_ms": sample[
                        "profiling_timing.draft_cuda_elapsed_ms.median"
                    ]["median"],
                    "profile_combined_attention_cuda_median_ms": sample[
                        "profiling_timing.combined_attention_cuda_elapsed_ms.median"
                    ]["median"],
                    "profile_verification_cuda_median_ms": sample[
                        "profiling_timing.verification_cuda_elapsed_ms.median"
                    ]["median"],
                    "profile_round_cuda_median_ms": group["round_level"][
                        "round_cuda_elapsed_ms"
                    ]["median"],
                    "profile_draft_over_round_median": sample[
                        "profiling_timing.draft_over_round_ratio.median"
                    ]["median"],
                    "profile_verification_over_round_median": sample[
                        "profiling_timing.verification_over_round_ratio.median"
                    ]["median"],
                    "profile_combined_attention_over_draft_median": sample[
                        "profiling_timing.combined_attention_over_draft_ratio.median"
                    ]["median"],
                    "profile_combined_attention_over_round_median": sample[
                        "profiling_timing.combined_attention_over_round_ratio.median"
                    ]["median"],
                    "mean_non_terminal_accepted_draft_tokens": group["round_level"][
                        "non_terminal_accepted_length"
                    ]["mean"],
                    "mean_non_terminal_committed_tokens": group["round_level"][
                        "non_terminal_committed_token_count"
                    ]["mean"],
                    "round_count": group["micro_average"]["round_count"],
                    "acceptance_eligible_round_count": group["micro_average"][
                        "acceptance_eligible_round_count"
                    ],
                    "truncated_token_count": group["micro_average"][
                        "truncated_token_count"
                    ],
                    "micro_raw_all_round_accepted_over_proposed": group[
                        "micro_average"
                    ]["raw_all_round_accepted_over_proposed"],
                    "micro_non_terminal_accepted_over_proposed": group["micro_average"][
                        "non_terminal_accepted_over_proposed"
                    ],
                    "macro_non_terminal_acceptance_rate": group[
                        "sample_level_macro_average"
                    ]["non_terminal_acceptance_rate"],
                    "macro_effective_returned_tokens_per_round": group[
                        "sample_level_macro_average"
                    ]["effective_returned_tokens_per_round"],
                }
            )
    temporary.replace(path)


def _run_one_workload(
    *,
    args: argparse.Namespace,
    torch_module: Any,
    target: Any,
    draft: Any,
    tokenizer: Any,
    workload: dict[str, Any],
    attention_backend: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    input_ids = torch_module.tensor(
        [workload["input_ids"]], dtype=torch_module.long, device=args.device
    )
    if int(input_ids.shape[1]) != workload["actual_prompt_length"]:
        raise RuntimeError("Persisted workload token count changed before generation")

    control_timings = []
    profile_components = []
    all_rounds = []
    accounting = None
    reference_ids = None
    generated_text = None
    peak_allocated = 0
    peak_reserved = 0
    for repeat in range(args.timing_repeats):
        control_ids, _, control_measurement, _ = _call_generation(
            torch_module=torch_module,
            draft=draft,
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            return_diagnostics=False,
            enable_profiling=False,
        )
        profile_ids, diagnostics, profile_measurement, _ = _call_generation(
            torch_module=torch_module,
            draft=draft,
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            return_diagnostics=True,
            enable_profiling=True,
        )
        if not _token_equal(torch_module, control_ids, profile_ids):
            raise RuntimeError(
                f"Control/profile output mismatch for {workload['sample_id']} "
                f"at {workload['nominal_prompt_length']} tokens, repeat {repeat}"
            )
        if reference_ids is not None and not _token_equal(
            torch_module, reference_ids, control_ids
        ):
            raise RuntimeError(
                "Greedy timing repeats produced different output token IDs"
            )
        reference_ids = control_ids
        generated_count = int(control_ids.shape[1] - input_ids.shape[1])
        generated_text = tokenizer.decode(
            control_ids[0, input_ids.shape[1] :], skip_special_tokens=True
        )
        control_timings.append(
            {
                "repeat": repeat,
                "total_cuda_elapsed_ms": control_measurement["cuda_elapsed_ms"],
                "cpu_wall_elapsed_ms": control_measurement["cpu_wall_elapsed_ms"],
                "generated_tokens_per_second": (
                    generated_count / (control_measurement["cuda_elapsed_ms"] / 1000.0)
                    if control_measurement["cuda_elapsed_ms"]
                    else None
                ),
            }
        )
        components = _profile_components(diagnostics)
        components.update(
            {
                "repeat": repeat,
                "cpu_wall_elapsed_ms": profile_measurement["cpu_wall_elapsed_ms"],
            }
        )
        profile_components.append(components)
        enriched = _enrich_rounds(diagnostics, attention_backend, repeat)
        for row in enriched:
            row.update(
                {
                    "sample_id": workload["sample_id"],
                    "source_sample_id": workload["source_sample_id"],
                    "longbench_task": workload["longbench_task"],
                    "sample_policy": workload["sample_policy"],
                    "nominal_prompt_length": workload["nominal_prompt_length"],
                    "actual_prompt_length": workload["actual_prompt_length"],
                    "timing_source": "diagnostics_profiling_path",
                }
            )
        all_rounds.extend(enriched)
        current_accounting = summarize_profile_rounds(
            enriched, generated_token_count=generated_count
        )
        if accounting is None:
            accounting = current_accounting
        elif (
            accounting["raw_all_round_accepted_length"]
            != current_accounting["raw_all_round_accepted_length"]
        ):
            raise RuntimeError("Greedy profiling repeats changed acceptance accounting")
        memory = diagnostics["profiling"]["memory"]
        peak_allocated = max(peak_allocated, memory["peak_allocated_bytes"])
        peak_reserved = max(peak_reserved, memory["peak_reserved_bytes"])

    generated_ids = reference_ids[0, input_ids.shape[1] :].tolist()
    control_total = distribution(
        row["total_cuda_elapsed_ms"] for row in control_timings
    )
    control_throughput = distribution(
        row["generated_tokens_per_second"] for row in control_timings
    )
    profiling_timing = _timing_distributions(profile_components)
    attention_calls = [call for row in all_rounds for call in row["attention_calls"]]
    sample = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": workload["sample_id"],
        "source_sample_id": workload["source_sample_id"],
        "longbench_task": workload["longbench_task"],
        "sample_policy": workload["sample_policy"],
        "paired_context_length_comparison": workload[
            "paired_context_length_comparison"
        ],
        "dataset_split": workload["dataset_split"],
        "random_seed": workload["random_seed"],
        "prompt_template_id": workload["prompt_template_id"],
        "prompt_template_name": workload["prompt_template_name"],
        "prompt_template_version": workload["prompt_template_version"],
        "prompt_template_sha256": workload["prompt_template_sha256"],
        "source_text_sha256": workload["source_text_sha256"],
        "document_token_length": workload["document_token_length"],
        "fixed_prompt_length": workload["fixed_prompt_length"],
        "query_sha256": workload["query_sha256"],
        "reference_answers": workload["reference_answers"],
        "reference_answers_sha256": workload["reference_answers_sha256"],
        "context_prefix_sha256": workload["context_prefix_sha256"],
        "report_prefix_sha256": workload["report_prefix_sha256"],
        "prompt_sha256": workload["prompt_sha256"],
        "input_token_ids_sha256": workload["input_token_ids_sha256"],
        "nominal_prompt_length": workload["nominal_prompt_length"],
        "actual_prompt_length": workload["actual_prompt_length"],
        "exact_nominal_length": workload["exact_nominal_length"],
        "nominal_prompt_length_shortfall": workload["nominal_prompt_length_shortfall"],
        "max_new_tokens": args.max_new_tokens,
        "generated_token_count": len(generated_ids),
        "generated_output_token_ids": generated_ids,
        "generated_output_token_ids_sha256": _sha256_token_ids(generated_ids),
        "generated_text": generated_text,
        "control_profile_output_token_ids_equal": True,
        "all_repeat_output_token_ids_equal": True,
        "control_timing": {
            "source": "instrumented_default_control",
            "formal_end_to_end": True,
            "repeats": control_timings,
            "total_cuda_elapsed_ms": control_total,
            "cpu_wall_elapsed_ms": distribution(
                row["cpu_wall_elapsed_ms"] for row in control_timings
            ),
            "generated_tokens_per_second": control_throughput,
        },
        "profiling_timing": {
            "source": "diagnostics_profiling_path",
            "formal_end_to_end": False,
            "repeats": profile_components,
            **profiling_timing,
        },
        "accounting": accounting,
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "source": "diagnostics_profiling_path",
        },
        "theoretical_attention_shape": {
            "batch_size": attention_calls[0]["batch_size"] if attention_calls else None,
            "query_length": attention_calls[0]["query_length"]
            if attention_calls
            else None,
            "query_head_count": (
                attention_calls[0]["query_head_count"] if attention_calls else None
            ),
            "key_value_head_count": (
                attention_calls[0]["key_value_head_count"] if attention_calls else None
            ),
            "head_dimension": attention_calls[0]["head_dimension"]
            if attention_calls
            else None,
            "minimum_persistent_context_kv_length": min(
                (call["persistent_context_kv_length"] for call in attention_calls),
                default=None,
            ),
            "maximum_persistent_context_kv_length": max(
                (call["persistent_context_kv_length"] for call in attention_calls),
                default=None,
            ),
            "maximum_effective_kv_length": max(
                (call["effective_kv_length"] for call in attention_calls), default=None
            ),
            "maximum_theoretical_qk_score_elements": max(
                (call["theoretical_qk_score_elements"] for call in attention_calls),
                default=None,
            ),
        },
    }
    return sample, all_rounds


def run_sweep(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    provenance: dict[str, Any],
    samples: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
    *,
    samples_path: Path,
    rounds_path: Path,
) -> None:
    tokenizer, target_source, transformers_version = _load_tokenizer_only(args)
    prepared, inventory, _, _ = _prepare_workload(
        args, tokenizer, build_cohort=False
    )
    manifest["workload"]["inventory"] = inventory
    _print_inventory(inventory)
    provisional_position_limit = max(args.context_lengths) + args.max_new_tokens
    workloads, cohort = build_longbench_workloads(
        prepared,
        tokenizer=tokenizer,
        task=args.longbench_task,
        nominal_lengths=args.context_lengths,
        sample_count=args.pilot_samples,
        sample_policy=args.sample_policy,
        max_new_tokens=args.max_new_tokens,
        model_position_limit=provisional_position_limit,
    )
    manifest["workload"]["cohort"] = cohort
    manifest["workload"]["sample_ids"] = sorted(
        {row["sample_id"] for row in workloads}
    )
    manifest["workload"]["sample_ids_by_length"] = cohort[
        "sample_ids_by_length"
    ]
    manifest["workload"]["samples"] = _workload_manifest_rows(workloads)
    print(
        "Workload preflight passed before model loading: "
        f"policy={args.sample_policy}, records={len(workloads)}"
    )

    # No CUDA or target/draft model is touched until the inventory and cohort
    # checks above have succeeded.
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
            f"Loaded draft block size {draft.block_size} does not equal {args.block_size}"
        )
    provenance.update(_loaded_source_record(draft, draft_manifest))
    provenance["target"] = target_source
    provenance["status"] = "ok"

    position_limit = _position_limit(target, tokenizer)
    for workload in workloads:
        if workload["actual_prompt_length"] + args.max_new_tokens > position_limit:
            raise RuntimeError(
                f"Sample {workload['sample_id']} at nominal length "
                f"{workload['nominal_prompt_length']} exceeds target position limit "
                f"{position_limit} after reserving {args.max_new_tokens} output tokens"
            )
    manifest["model_position_limit"] = position_limit
    manifest["attention_backend"] = attention_backend_snapshot(
        torch_module, target=target, draft=draft
    )
    if not manifest["attention_backend"]["draft"]["calls_pytorch_sdpa"]:
        raise RuntimeError(
            "Phase 1B combined-attention timing requires the draft model's "
            "configured attention implementation to be PyTorch SDPA"
        )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("The target tokenizer does not define eos_token_id")
    print(f"Git commit: {manifest['repository']['commit_sha']}")
    print(f"Transformers: {transformers_version}")
    print(f"GPU: {torch_module.cuda.get_device_name(args.device)}")
    print(f"Model position limit: {position_limit}")
    print(f"Instrumented DFlash module: {provenance['actual_loaded_module_path']}")

    first_by_length = {}
    for workload in workloads:
        first_by_length.setdefault(workload["nominal_prompt_length"], workload)
    for nominal_length in args.context_lengths:
        warmup = first_by_length[nominal_length]
        warmup_ids = torch_module.tensor(
            [warmup["input_ids"]], dtype=torch_module.long, device=args.device
        )
        for _ in range(args.warmup_runs):
            draft.spec_generate(
                input_ids=warmup_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                target=target,
                stop_token_ids=[tokenizer.eos_token_id],
            )
        if args.warmup_runs:
            torch_module.cuda.synchronize(warmup_ids.device)
        print(f"Warmup complete: nominal_prompt_length={nominal_length}")

    for index, workload in enumerate(workloads, 1):
        print(
            f"[{index}/{len(workloads)}] sample={workload['sample_id']} "
            f"nominal={workload['nominal_prompt_length']} "
            f"actual={workload['actual_prompt_length']}"
        )
        sample, sample_rounds = _run_one_workload(
            args=args,
            torch_module=torch_module,
            target=target,
            draft=draft,
            tokenizer=tokenizer,
            workload=workload,
            attention_backend=manifest["attention_backend"],
        )
        samples.append(sample)
        rounds.extend(sample_rounds)
        _atomic_write_jsonl(samples_path, samples)
        _atomic_write_jsonl(rounds_path, rounds)

    summary["aggregates"] = aggregate_sweep(samples, rounds)
    summary["sample_length_record_count"] = len(samples)
    summary["profiling_round_record_count"] = len(rounds)
    summary["status"] = "ok"
    manifest["status"] = "ok"


def validate_artifacts(
    manifest: dict[str, Any], summary: dict[str, Any], samples: Sequence[dict[str, Any]]
) -> None:
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported Phase 1B manifest schema")
    if summary["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported Phase 1B summary schema")
    if summary["status"] == "ok" and not samples:
        raise ValueError("Successful Phase 1B run contains no samples")
    for sample in samples:
        if not sample["control_profile_output_token_ids_equal"]:
            raise ValueError("A sample failed control/profile token equality")
        if (
            sample["actual_prompt_length"] + sample["max_new_tokens"]
            > manifest["model_position_limit"]
        ):
            raise ValueError("A sample exceeds the model position limit")


def run_inventory_only(args: argparse.Namespace) -> dict[str, Any]:
    """Inspect tokenizer-derived eligibility without CUDA or model construction."""

    tokenizer, target_source, transformers_version = _load_tokenizer_only(args)
    _, inventory, _, _ = _prepare_workload(args, tokenizer, build_cohort=False)
    inventory.update(
        {
            "status": "ok",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "inspect_workload_only",
            "tokenizer": {
                "model_id": args.target_model,
                "revision": args.target_revision,
                "source_kind": target_source["source_kind"],
                "load_source": target_source["load_source"],
                "transformers_version": transformers_version,
                "class_name": (
                    f"{tokenizer.__class__.__module__}."
                    f"{tokenizer.__class__.__qualname__}"
                ),
            },
            "models_loaded": False,
            "cuda_required": False,
        }
    )
    _print_inventory(inventory)
    return inventory


def _inventory_main(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "workload-inventory.json"
    environment_path = output_dir / "environment.json"
    terminal_log_path = output_dir / "terminal.log"
    command = shlex.join([sys.executable, *sys.argv])
    environment = collect_environment(
        run_config=run_configuration(args),
        run_command=command,
        import_torch=False,
    )
    write_json(environment_path, environment)
    inventory: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "started",
        "mode": "inspect_workload_only",
        "task": args.longbench_task,
        "record_count": 0,
        "prompt_template": prompt_template_metadata(args.longbench_task),
        "document_token_length": {
            "count": 0,
            "min": None,
            "median": None,
            "p90": None,
            "max": None,
        },
        "eligible_by_prompt_length": {},
        "sample_ids_reaching_maximum_length": [],
        "source_sample_ids_reaching_maximum_length": [],
        "recommendations": {},
        "models_loaded": False,
        "cuda_required": False,
        "error": None,
    }
    exit_code = 1
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with terminal_log_path.open("w", encoding="utf-8", buffering=1) as log:
        sys.stdout = TeeStream(original_stdout, log)
        sys.stderr = TeeStream(original_stderr, log)
        try:
            print(f"Phase 1B.1 inventory output directory: {output_dir}")
            inventory = run_inventory_only(args)
            inventory["error"] = None
            exit_code = 0
        except KeyboardInterrupt:
            inventory["status"] = "error"
            inventory["error"] = {
                "type": "KeyboardInterrupt",
                "message": "Interrupted by user",
            }
            exit_code = 130
            print("ERROR: KeyboardInterrupt: interrupted by user", file=sys.stderr)
            traceback.print_exc()
        except Exception as exc:  # noqa: BLE001 - persist inventory failures.
            inventory["status"] = "error"
            inventory["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
        finally:
            try:
                write_json(inventory_path, inventory)
                print(f"workload_inventory_json: {inventory_path}")
                print(f"environment_json: {environment_path}")
                print(f"terminal_log: {terminal_log_path}")
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.inspect_workload_only:
        return _inventory_main(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest_json": output_dir / "manifest.json",
        "workload_inventory_json": output_dir / "workload-inventory.json",
        "samples_jsonl": output_dir / "samples.jsonl",
        "rounds_jsonl": output_dir / "rounds.jsonl",
        "summary_json": output_dir / "summary.json",
        "summary_csv": output_dir / "summary.csv",
        "environment_json": output_dir / "environment.json",
        "source_provenance_json": output_dir / "source-provenance.json",
        "terminal_log": output_dir / "terminal.log",
    }
    command = shlex.join([sys.executable, *sys.argv])
    environment = collect_environment(
        run_config=run_configuration(args), run_command=command
    )
    manifest = new_manifest(args, environment, command)
    manifest["artifacts"] = {name: str(path) for name, path in paths.items()}
    summary = new_summary()
    provenance: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "started",
        "error": None,
    }
    samples: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    write_json(paths["environment_json"], environment)
    exit_code = 1
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with paths["terminal_log"].open("w", encoding="utf-8", buffering=1) as log:
        sys.stdout = TeeStream(original_stdout, log)
        sys.stderr = TeeStream(original_stderr, log)
        try:
            print(f"Phase 1B output directory: {output_dir}")
            run_sweep(
                args,
                manifest,
                summary,
                provenance,
                samples,
                rounds,
                samples_path=paths["samples_jsonl"],
                rounds_path=paths["rounds_jsonl"],
            )
            exit_code = 0
        except KeyboardInterrupt:
            error = {"type": "KeyboardInterrupt", "message": "Interrupted by user"}
            manifest["status"] = summary["status"] = provenance["status"] = "error"
            manifest["error"] = summary["error"] = provenance["error"] = error
            exit_code = 130
            print("ERROR: KeyboardInterrupt: interrupted by user", file=sys.stderr)
            traceback.print_exc()
        except Exception as exc:  # noqa: BLE001 - persist every runtime failure.
            error = {"type": type(exc).__name__, "message": str(exc)}
            manifest["status"] = summary["status"] = provenance["status"] = "error"
            manifest["error"] = summary["error"] = provenance["error"] = error
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
        finally:
            try:
                if samples:
                    summary["aggregates"] = aggregate_sweep(samples, rounds)
                    summary["sample_length_record_count"] = len(samples)
                    summary["profiling_round_record_count"] = len(rounds)
                validate_artifacts(manifest, summary, samples)
                write_json(paths["manifest_json"], manifest)
                inventory = manifest["workload"].get("inventory") or {
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "mode": "generation_preflight",
                    "task": args.longbench_task,
                    "record_count": 0,
                    "prompt_template": prompt_template_metadata(
                        args.longbench_task
                    ),
                    "document_token_length": {
                        "count": 0,
                        "min": None,
                        "median": None,
                        "p90": None,
                        "max": None,
                    },
                    "eligible_by_prompt_length": {},
                    "sample_ids_reaching_maximum_length": [],
                    "source_sample_ids_reaching_maximum_length": [],
                    "recommendations": {},
                    "models_loaded": False,
                    "cuda_required": False,
                    "error": manifest["error"],
                }
                write_json(paths["workload_inventory_json"], inventory)
                _atomic_write_jsonl(paths["samples_jsonl"], samples)
                _atomic_write_jsonl(paths["rounds_jsonl"], rounds)
                write_json(paths["summary_json"], summary)
                if summary["aggregates"] is not None:
                    _write_summary_csv(paths["summary_csv"], summary["aggregates"])
                else:
                    empty_csv = paths["summary_csv"].with_name(".summary.csv.tmp")
                    empty_csv.write_text("", encoding="utf-8")
                    empty_csv.replace(paths["summary_csv"])
                write_json(paths["source_provenance_json"], provenance)
                for name, path in paths.items():
                    print(f"{name}: {path}")
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
