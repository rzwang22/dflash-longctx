#!/usr/bin/env python3
"""Run the Phase 1B offline GovReport long-context DFlash baseline sweep."""

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
    PROMPT_TEMPLATE_ID,
    build_nested_workloads,
    load_govreport_records,
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
    parser.add_argument("--materialized-draft-path", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--govreport-jsonl", type=Path)
    source.add_argument("--govreport-dataset-path", type=Path)
    source.add_argument("--govreport-dataset-cache", type=Path)
    parser.add_argument("--dataset-name", default="ccdv/govreport-summarization")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--id-field", default=None)
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
        "materialized_draft_path": str(args.materialized_draft_path),
        "dataset_source": {
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
            "text_field": args.text_field,
            "id_field": args.id_field,
        },
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
            "dataset": "GovReport",
            "split": args.dataset_split,
            "seed": args.seed,
            "prompt_template_id": PROMPT_TEMPLATE_ID,
            "sample_ids": [],
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
    fieldnames = [
        "nominal_prompt_length",
        "sample_count",
        "control_total_cuda_median_ms",
        "control_total_cuda_mean_ms",
        "control_total_cuda_p90_ms",
        "generated_tokens_per_second_median",
        "profile_prefill_cuda_median_ms",
        "profile_decode_cuda_median_ms",
        "profile_combined_attention_cuda_median_ms",
        "profile_combined_attention_over_draft_median",
        "profile_combined_attention_over_round_median",
        "micro_raw_all_round_accepted_over_proposed",
        "micro_non_terminal_accepted_over_proposed",
        "macro_non_terminal_acceptance_rate",
        "macro_effective_returned_tokens_per_round",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
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
                    "profile_combined_attention_cuda_median_ms": sample[
                        "profiling_timing.combined_attention_cuda_elapsed_ms.median"
                    ]["median"],
                    "profile_combined_attention_over_draft_median": sample[
                        "profiling_timing.combined_attention_over_draft_ratio.median"
                    ]["median"],
                    "profile_combined_attention_over_round_median": sample[
                        "profiling_timing.combined_attention_over_round_ratio.median"
                    ]["median"],
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
        "dataset_split": workload["dataset_split"],
        "random_seed": workload["random_seed"],
        "prompt_template_id": workload["prompt_template_id"],
        "source_text_sha256": workload["source_text_sha256"],
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
    torch_module = _require_cuda(args)
    try:
        import transformers
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Transformers backend is not installed") from exc

    target_source = _resolve_model_source(
        "target", args.target_model, args.target_revision, args.target_model_path
    )
    dtype = _torch_dtype(torch_module, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        target_source["load_source"],
        trust_remote_code=True,
        **_pretrained_location_kwargs(target_source),
    )
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
    records = load_govreport_records(
        jsonl_path=args.govreport_jsonl,
        dataset_path=args.govreport_dataset_path,
        dataset_cache_dir=args.govreport_dataset_cache,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
    )
    workloads = build_nested_workloads(
        records,
        tokenizer=tokenizer,
        nominal_lengths=args.context_lengths,
        sample_count=args.pilot_samples,
        split=args.dataset_split,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        model_position_limit=position_limit,
        text_field=args.text_field,
        id_field=args.id_field,
    )
    manifest["workload"]["sample_ids"] = sorted({row["sample_id"] for row in workloads})
    manifest["workload"]["samples"] = _workload_manifest_rows(workloads)
    print(f"Git commit: {manifest['repository']['commit_sha']}")
    print(f"Transformers: {transformers.__version__}")
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest_json": output_dir / "manifest.json",
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
