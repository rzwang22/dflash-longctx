"""Capture exact DFlash Top-16 draft candidates for offline Phase 2B replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase2b.capture import build_capture_round_rows
from experiments.phase0_smoke import TeeStream
from experiments.phase1_observability import (
    _atomic_write_jsonl,
    _call_generation,
    _token_equal,
)
from experiments.phase1b_longctx import build_parser as build_phase1b_parser
from experiments.phase1b_longctx import new_manifest as phase1b_new_manifest
from experiments.phase1b_longctx import (
    run_configuration as phase1b_run_configuration,
)
from experiments.phase1b_longctx import validate_args as validate_phase1b_args
from experiments.phase2a_profile import _load_inputs, _warmup
from scripts.collect_environment import collect_environment, write_json

SCHEMA_VERSION = "1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = build_phase1b_parser()
    parser.description = __doc__
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Run exactly one stable LongBench sample ID at every requested length.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=16,
        help="Per-position candidates captured from exact final draft logits (max 16).",
    )
    parser.set_defaults(output_dir=Path("outputs/phase2b-capture"), timing_repeats=1)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = validate_phase1b_args(build_parser().parse_args(argv))
    if args.inspect_workload_only:
        raise ValueError("Phase 2B capture does not support inventory-only mode")
    if args.sample_id is not None and args.sample_policy != "matched":
        raise ValueError("--sample-id requires --sample-policy matched")
    if not 2 <= args.top_k <= 16:
        raise ValueError("--top-k must be between 2 and 16")
    return args


def run_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **phase1b_run_configuration(args),
        "phase": "2B",
        "mode": "exact_draft_candidate_capture",
        "sample_id": args.sample_id,
        "top_k": args.top_k,
        "capture_repeats": args.timing_repeats,
        "control_then_capture": True,
        "control_capture_token_exact_required": True,
        "full_vocabulary_logits_saved": False,
        "target_logits_saved": False,
        "model_hidden_states_saved": False,
        "kv_cache_saved": False,
    }


def _token_ids_sha256(token_ids: list[int]) -> str:
    canonical = json.dumps(token_ids, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _new_manifest(
    args: argparse.Namespace, environment: dict[str, Any], command: str
) -> dict[str, Any]:
    manifest = phase1b_new_manifest(args, environment, command)
    manifest["configuration"] = run_configuration(args)
    manifest["measurement_policy"] = {
        "pair_order": ["instrumented_default_control", "candidate_capture"],
        "output_token_ids_must_match": True,
        "capture_from_exact_final_draft_logits": True,
        "extra_draft_forward_calls": 0,
        "capture_latency_is_formal_end_to_end": False,
        "control_latency_is_reported_for_sanity_only": True,
        "tree_replay_is_offline_cpu_only": True,
    }
    manifest["alignment_policy"] = {
        "block_position_zero": "target-produced seed/root",
        "draft_positions": "1..15",
        "root_output_offset_scope": "generated tokens only; prompt excluded",
        "uncensored_invariant": (
            "LCP(Top1 draft chain, final verified continuation) equals accepted drafts"
        ),
        "right_censored_invariant": (
            "all observable final-output continuation tokens match the accepted prefix"
        ),
        "right_censored_full_depth_excluded": True,
        "immediate_verification_lcp_checked_inside_generation": True,
    }
    manifest["interpretation"] = {
        "offline_feasibility_only": True,
        "real_tree_verification_implemented": False,
        "actual_tree_acceptance_measured": False,
        "latency_or_speedup_claim_supported": False,
    }
    return manifest


def _capture_one(
    args: argparse.Namespace,
    *,
    torch_module: Any,
    tokenizer: Any,
    target: Any,
    draft: Any,
    workload: dict[str, Any],
    provenance: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    input_ids = torch_module.tensor(
        [workload["input_ids"]], dtype=torch_module.long, device=args.device
    )
    if int(input_ids.shape[1]) != int(workload["actual_prompt_length"]):
        raise RuntimeError("persisted workload token count changed before capture")

    repeats = []
    round_rows = []
    reference_ids = None
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
        capture_ids, diagnostics, capture_measurement, _ = _call_generation(
            torch_module=torch_module,
            draft=draft,
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            return_diagnostics=True,
            enable_profiling=False,
            generation_options={
                "enable_tree_capture": True,
                "tree_top_k": args.top_k,
            },
        )
        if not _token_equal(torch_module, control_ids, capture_ids):
            raise RuntimeError(
                "Phase 2B control/capture token mismatch: "
                f"sample={workload['sample_id']}, "
                f"nominal={workload['nominal_prompt_length']}, repeat={repeat}"
            )
        if reference_ids is not None and not _token_equal(
            torch_module, reference_ids, control_ids
        ):
            raise RuntimeError("greedy Phase 2B capture repeats changed output token IDs")
        reference_ids = control_ids
        generated_ids = [
            int(value)
            for value in capture_ids[0, int(input_ids.shape[1]) :].detach().cpu().tolist()
        ]
        metadata = {
            "task": workload["longbench_task"],
            "dataset_split": workload["dataset_split"],
            "sample_id": workload["sample_id"],
            "source_sample_id": workload["source_sample_id"],
            "source_index": int(workload["source_index"]),
            "source_text_sha256": workload["source_text_sha256"],
            "context_prefix_sha256": workload["context_prefix_sha256"],
            "prompt_sha256": workload["prompt_sha256"],
            "input_token_ids_sha256": workload["input_token_ids_sha256"],
            "nominal_prompt_length": int(workload["nominal_prompt_length"]),
            "actual_prompt_length": int(workload["actual_prompt_length"]),
            "generation_repeat": repeat,
            "capture_source_provenance": {
                "repository_custom_code_commit": provenance[
                    "repository_custom_code_commit"
                ],
                "draft_model_id": provenance["base_model_id"],
                "draft_model_revision": provenance["base_model_revision"],
                "materialization_id": provenance["materialization_id"],
                "actual_loaded_module_sha256": provenance[
                    "actual_loaded_module_sha256"
                ],
                "actual_loaded_dependency_sha256": {
                    filename: record["actual_loaded_sha256"]
                    for filename, record in provenance[
                        "actual_loaded_dependencies"
                    ].items()
                },
                "target_model_id": args.target_model,
                "target_model_revision": args.target_revision,
            },
        }
        current_rows = build_capture_round_rows(
            diagnostics=diagnostics,
            final_verified_generated_token_ids=generated_ids,
            metadata=metadata,
        )
        round_rows.extend(current_rows)
        repeats.append(
            {
                "repeat": repeat,
                "control_capture_output_token_ids_equal": True,
                "control_cuda_elapsed_ms_sanity_only": control_measurement[
                    "cuda_elapsed_ms"
                ],
                "capture_cuda_elapsed_ms_non_formal": capture_measurement[
                    "cuda_elapsed_ms"
                ],
                "capture_round_count": len(current_rows),
            }
        )

    generated_ids = [
        int(value)
        for value in reference_ids[0, int(input_ids.shape[1]) :].detach().cpu().tolist()
    ]
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "task": workload["longbench_task"],
            "dataset_split": workload["dataset_split"],
            "sample_id": workload["sample_id"],
            "source_sample_id": workload["source_sample_id"],
            "source_text_sha256": workload["source_text_sha256"],
            "context_prefix_sha256": workload["context_prefix_sha256"],
            "nominal_prompt_length": int(workload["nominal_prompt_length"]),
            "actual_prompt_length": int(workload["actual_prompt_length"]),
            "generated_token_count": len(generated_ids),
            "final_verified_output_token_ids": generated_ids,
            "final_verified_output_token_ids_sha256": _token_ids_sha256(generated_ids),
            "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
            "control_capture_output_token_ids_equal": True,
            "capture_repeats": repeats,
            "full_vocabulary_logits_saved": False,
            "target_logits_saved": False,
            "capture_source_provenance": {
                "repository_custom_code_commit": provenance[
                    "repository_custom_code_commit"
                ],
                "materialization_id": provenance["materialization_id"],
                "actual_loaded_module_sha256": provenance[
                    "actual_loaded_module_sha256"
                ],
            },
        },
        round_rows,
    )


def run_capture(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    samples: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
) -> None:
    torch_module, tokenizer, target, draft, workloads = _load_inputs(
        args, manifest, provenance
    )
    _warmup(args, torch_module, tokenizer, target, draft, workloads)
    for index, workload in enumerate(workloads, 1):
        print(
            f"[{index}/{len(workloads)}] capture task={workload['longbench_task']} "
            f"sample={workload['sample_id']} nominal={workload['nominal_prompt_length']}"
        )
        sample, current_rounds = _capture_one(
            args,
            torch_module=torch_module,
            tokenizer=tokenizer,
            target=target,
            draft=draft,
            workload=workload,
            provenance=provenance,
        )
        samples.append(sample)
        rounds.extend(current_rounds)
    manifest["status"] = "ok"
    manifest["capture"] = {
        "sample_record_count": len(samples),
        "round_record_count": len(rounds),
        "all_control_capture_outputs_token_exact": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest_json": output_dir / "manifest.json",
        "samples_jsonl": output_dir / "samples.jsonl",
        "tree_capture_rounds_jsonl": output_dir / "tree-capture-rounds.jsonl",
        "environment_json": output_dir / "environment.json",
        "source_provenance_json": output_dir / "source-provenance.json",
        "terminal_log": output_dir / "terminal.log",
    }
    command = shlex.join([sys.executable, *sys.argv])
    environment = collect_environment(
        run_config=run_configuration(args), run_command=command
    )
    manifest = _new_manifest(args, environment, command)
    manifest["artifacts"] = {name: str(path) for name, path in paths.items()}
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
            print(f"Phase 2B capture output directory: {output_dir}")
            run_capture(args, manifest, provenance, samples, rounds)
            exit_code = 0
        except KeyboardInterrupt:
            error = {"type": "KeyboardInterrupt", "message": "Interrupted by user"}
            manifest["status"] = provenance["status"] = "error"
            manifest["error"] = provenance["error"] = error
            exit_code = 130
            traceback.print_exc()
        except Exception as exc:  # noqa: BLE001 - preserve partial capture evidence.
            error = {"type": type(exc).__name__, "message": str(exc)}
            manifest["status"] = provenance["status"] = "error"
            manifest["error"] = provenance["error"] = error
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
        finally:
            try:
                write_json(paths["manifest_json"], manifest)
                _atomic_write_jsonl(paths["samples_jsonl"], samples)
                _atomic_write_jsonl(paths["tree_capture_rounds_jsonl"], rounds)
                write_json(paths["source_provenance_json"], provenance)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                if exit_code == 0:
                    exit_code = 1
            sys.stdout, sys.stderr = original_stdout, original_stderr
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
