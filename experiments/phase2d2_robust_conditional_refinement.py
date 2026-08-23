#!/usr/bin/env python3
"""Run Phase 2D.2A numerically robust conditional-remask refinement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import statistics
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase1.materialize import verify_materialization
from dflash.phase1b.workload import build_longbench_workloads
from dflash.phase2c.divergence import output_pair_comparison
from dflash.phase2c.gold import (
    DEFAULT_GOLD_THRESHOLDS,
    GOLD_COMPARISON_MODES,
    build_gold_comparison_summary,
    validate_gold_modes,
    validate_gold_thresholds,
)
from dflash.phase2d1.runtime import validate_frozen_selector
from dflash.phase2d2.freeze import validate_frozen_refinement
from dflash.phase2d2.metrics import (
    aggregate_results,
    build_phase2d2_decision,
    candidate_quality_for_round,
)
from dflash.phase2d2.refinement import (
    CANDIDATE_MERGE_MODES,
    FIXED_SHAPE_MARGIN_THRESHOLD,
    FIXED_SHAPE_VERIFICATION_MODES,
    MARGIN_GUARDED_VERIFICATION_MODES,
    Q1_REPLAY_VERIFICATION_MODES,
    VERIFICATION_MODES,
    validate_refinement_options,
)
from dflash.phase2d2.robustness import validate_nr_calibration
from experiments.phase0_smoke import TeeStream, _pretrained_location_kwargs
from experiments.phase1_observability import _call_generation
from experiments.phase1b_longctx import (
    _load_tokenizer_only,
    _prepare_workload,
)
from experiments.phase1b_longctx import (
    build_parser as build_phase1b_parser,
)
from experiments.phase1b_longctx import (
    new_manifest as phase1b_manifest,
)
from experiments.phase1b_longctx import (
    validate_args as validate_phase1b_args,
)
from experiments.phase2a_profile import _load_inputs
from experiments.phase2c_online_tree import _target_only_greedy
from scripts.check_phase2d2_acceptance import check_output
from scripts.collect_environment import collect_environment

SCHEMA_VERSION = "1.0"
BASELINE_SHA = "1ba29e8ce4a165c03af997c2f3b3b595ab79c1d2"
BRANCH = "codex/phase2d2a-robust-conditional-remask"
BASE_CONTROL_SELECTOR_MODE = "top1_packed_tree"
ROOT_ARTIFACTS = (
    "manifest.json",
    "environment.json",
    "source-provenance.json",
    "workload-inventory.json",
    "samples.jsonl",
    "refinement-rounds.jsonl",
    "candidate-merge-events.jsonl",
    "refined-tree-rounds.jsonl",
    "refinement-summary.json",
    "refinement-summary.csv",
    "candidate-quality-summary.csv",
    "fixed-budget-comparison.json",
    "fixed-budget-comparison.csv",
    "robust-verification-events.jsonl",
    "verifier-oracle-comparisons.jsonl",
    "verifier-oracle-summary.json",
    "tree-traversal-checks.jsonl",
    "fallback-summary.json",
    "fallback-summary.csv",
    "correctness-summary.json",
    "output-divergence.json",
    "output-divergence.csv",
    "offline-online-comparison.json",
    "offline-online-comparison.csv",
    "timing-summary.json",
    "timing-summary.csv",
    "phase2d2-decision.json",
    "terminal.log",
)
GOLD_ARTIFACTS = (
    "gold-logit-checks.jsonl",
    "gold-kv-checks.jsonl",
    "gold-layer-localization.jsonl",
    "gold-comparison-summary.json",
    "gold-comparison-summary.csv",
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_phase1b_parser()
    parser.description = __doc__
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--frozen-selector-config", required=True, type=Path)
    parser.add_argument("--frozen-refinement-config", type=Path, default=None)
    parser.add_argument("--frozen-refinement-config-id", default=None)
    parser.add_argument(
        "--evidence-role",
        choices=("mechanism_analysis", "held_out_evaluation"),
        default="mechanism_analysis",
    )
    parser.add_argument(
        "--online-selector-mode",
        choices=(
            "top1_packed_tree",
            "global_fixed",
            "context_rule",
            "decision_stump",
        ),
        default=BASE_CONTROL_SELECTOR_MODE,
        help=(
            "Frozen-selector mode. The default preserves the validated Phase 2C "
            "Top-1 packed-tree base control; pass another mode explicitly for a "
            "frozen-selector experiment."
        ),
    )
    parser.add_argument("--node-budget", type=int, choices=(16, 32), default=16)
    parser.add_argument(
        "--candidate-merge-mode",
        choices=CANDIDATE_MERGE_MODES,
        default="base_only",
    )
    parser.add_argument("--base-top-m", type=int, default=16)
    parser.add_argument("--conditional-top-r", type=int, choices=(2, 4), default=2)
    parser.add_argument("--refine-depth", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument(
        "--refined-prefix-budget", type=int, choices=(1, 2, 4), default=2
    )
    parser.add_argument("--refinement-weight", type=float, default=0.5)
    parser.add_argument(
        "--verification-mode",
        choices=VERIFICATION_MODES,
        default="always_replay",
    )
    parser.add_argument("--decision-margin-threshold", type=float, default=None)
    parser.add_argument("--nr-calibration-dir", type=Path, default=None)
    parser.add_argument("--enable-gold-checks", action="store_true")
    parser.add_argument(
        "--gold-comparison-mode",
        action="append",
        choices=GOLD_COMPARISON_MODES,
        dest="gold_comparison_modes",
    )
    parser.add_argument("--gold-layer-localization", action="store_true")
    parser.add_argument("--gold-max-rounds", type=int, default=32)
    parser.add_argument("--gold-record-all-nodes", action="store_true")
    parser.add_argument(
        "--enable-verifier-oracle-diagnostics",
        action="store_true",
        help=(
            "Run packed/q1/fixed-shape verifier comparisons on independent "
            "round-entry KV clones in the separate diagnostics generation only."
        ),
    )
    parser.add_argument(
        "--gold-max-abs-threshold",
        type=float,
        default=DEFAULT_GOLD_THRESHOLDS["maximum_absolute_error"],
    )
    parser.add_argument(
        "--gold-mean-abs-threshold",
        type=float,
        default=DEFAULT_GOLD_THRESHOLDS["mean_absolute_error"],
    )
    parser.add_argument(
        "--gold-relative-epsilon",
        type=float,
        default=DEFAULT_GOLD_THRESHOLDS["relative_error_epsilon"],
    )
    parser.add_argument(
        "--gold-layer-threshold",
        type=float,
        default=DEFAULT_GOLD_THRESHOLDS["layer_localization_p99_error"],
    )
    parser.add_argument(
        "--base-result-dir",
        action="append",
        type=Path,
        default=[],
        help="Optional same-N base output(s) for fixed-budget comparison.",
    )
    parser.set_defaults(output_dir=Path("outputs/phase2d2a-robust-conditional-remask"))
    return parser


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = validate_phase1b_args(build_parser().parse_args(argv))
    if args.inspect_workload_only:
        raise ValueError("Phase 2D.2A does not support inventory-only execution")
    if args.sample_id is not None and args.sample_policy != "matched":
        raise ValueError("--sample-id requires --sample-policy matched")
    if args.gold_max_rounds < 1:
        raise ValueError("--gold-max-rounds must be positive")
    if args.gold_layer_localization and not args.enable_gold_checks:
        raise ValueError("--gold-layer-localization requires --enable-gold-checks")
    if args.gold_record_all_nodes and not args.enable_gold_checks:
        raise ValueError("--gold-record-all-nodes requires --enable-gold-checks")
    args.gold_comparison_modes = validate_gold_modes(
        args.gold_comparison_modes or ["native", "same_mask", "math"]
    )
    args.gold_thresholds = validate_gold_thresholds(
        {
            "maximum_absolute_error": args.gold_max_abs_threshold,
            "mean_absolute_error": args.gold_mean_abs_threshold,
            "relative_error_epsilon": args.gold_relative_epsilon,
            "layer_localization_p99_error": args.gold_layer_threshold,
        }
    )
    args.frozen_selector = validate_frozen_selector(
        _load_json(args.frozen_selector_config)
    )
    args.frozen_refinement = None
    if (args.frozen_refinement_config is None) != (
        args.frozen_refinement_config_id is None
    ):
        raise ValueError(
            "--frozen-refinement-config and --frozen-refinement-config-id are paired"
        )
    if args.frozen_refinement_config is not None:
        frozen = validate_frozen_refinement(_load_json(args.frozen_refinement_config))
        matches = [
            row
            for row in frozen.get("configurations", [])
            if row["configuration_id"] == args.frozen_refinement_config_id
        ]
        if len(matches) != 1:
            raise ValueError("frozen refinement configuration ID is not unique")
        selected = matches[0]
        for field in (
            "candidate_merge_mode",
            "base_top_m",
            "conditional_top_r",
            "refine_depth",
            "refined_prefix_budget",
            "refinement_weight",
        ):
            setattr(args, field, selected[field])
        args.frozen_refinement = frozen
    if (
        args.evidence_role == "held_out_evaluation"
        and args.candidate_merge_mode != "base_only"
        and args.frozen_refinement is None
    ):
        raise ValueError(
            "held-out refined evaluation requires a frozen refinement config"
        )
    args.exclude_selector_training_sources = args.evidence_role == "held_out_evaluation"
    if args.verification_mode in MARGIN_GUARDED_VERIFICATION_MODES:
        if args.dtype == "auto":
            raise ValueError(
                f"{args.verification_mode} requires an explicit dtype"
            )
        if args.nr_calibration_dir is None:
            raise ValueError(
                f"{args.verification_mode} requires --nr-calibration-dir"
            )
        calibration_path = args.nr_calibration_dir / "nr-calibration.json"
        if not calibration_path.is_file():
            raise ValueError(f"missing NR calibration artifact: {calibration_path}")
        args.nr_calibration = _load_json(calibration_path)
        calibrated = float(args.nr_calibration["decision_margin_threshold"])
        if (
            args.verification_mode == "margin_guarded_fixed_shape_recheck"
            and calibrated != FIXED_SHAPE_MARGIN_THRESHOLD
        ):
            raise ValueError(
                "margin_guarded_fixed_shape_recheck requires the frozen 0.25 "
                "NR threshold"
            )
        if (
            args.decision_margin_threshold is not None
            and float(args.decision_margin_threshold) != calibrated
        ):
            raise ValueError("CLI margin threshold differs from frozen NR calibration")
        args.decision_margin_threshold = calibrated
    else:
        args.nr_calibration = None
        if args.decision_margin_threshold is None:
            args.decision_margin_threshold = 0.0
    validate_refinement_options(
        enabled=True,
        merge_mode=args.candidate_merge_mode,
        base_top_m=args.base_top_m,
        conditional_top_r=args.conditional_top_r,
        refine_depth=args.refine_depth,
        refined_prefix_budget=args.refined_prefix_budget,
        refinement_weight=args.refinement_weight,
        node_budget=args.node_budget,
        block_size=args.block_size,
        external_batch_size=args.batch_size,
        verification_mode=args.verification_mode,
        calibration_validated=args.nr_calibration is not None,
    )
    return args


def _runtime_calibration_provenance(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "Phase 2D.2A CUDA preflight dependencies are missing"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 2D.2A requires CUDA; no CPU fallback is permitted")
    index = torch.device(args.device).index or 0
    capability = torch.cuda.get_device_capability(index)
    return {
        "gpu_name": torch.cuda.get_device_name(index),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "dtype": args.dtype,
        "target_model_revision": args.target_revision,
        "draft_model_revision": args.draft_revision,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "target_attention_implementation": "sdpa",
    }


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    materialization = verify_materialization(
        args.materialized_draft_path, expected_implementation="instrumented"
    )
    tokenizer, target_source, _ = _load_tokenizer_only(args)
    from transformers import AutoConfig

    target_config = AutoConfig.from_pretrained(
        target_source["load_source"],
        trust_remote_code=True,
        **_pretrained_location_kwargs(target_source),
    )
    position_limits = [
        int(value)
        for value in (
            getattr(target_config, "max_position_embeddings", None),
            getattr(tokenizer, "model_max_length", None),
        )
        if value is not None and 0 < int(value) < 10**9
    ]
    if not position_limits:
        raise RuntimeError(
            "target model position limit is unavailable during preflight"
        )
    model_position_limit = min(position_limits)
    prepared, inventory, _, _ = _prepare_workload(args, tokenizer, build_cohort=False)
    if args.exclude_selector_training_sources:
        from dflash.phase1b.workload import build_workload_inventory

        training_ids = set(args.frozen_selector["training_source_ids"])
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
            "completed_before_model_loading": True,
            "remaining_independent_source_ids": sorted(
                f"{args.longbench_task}::{row['source_sample_id']}" for row in prepared
            ),
            "training_sources_reintroduced": False,
        }
    workloads, cohort = build_longbench_workloads(
        prepared,
        tokenizer=tokenizer,
        task=args.longbench_task,
        nominal_lengths=args.context_lengths,
        sample_count=args.pilot_samples,
        sample_policy=args.sample_policy,
        max_new_tokens=args.max_new_tokens,
        model_position_limit=model_position_limit,
        sample_id=args.sample_id,
    )
    calibration = None
    runtime_provenance = _runtime_calibration_provenance(args)
    if args.nr_calibration is not None:
        calibration = validate_nr_calibration(args.nr_calibration, runtime_provenance)
    return {
        "materialization": materialization,
        "inventory": inventory,
        "cohort": cohort,
        "workload_count": len(workloads),
        "calibration": calibration,
        "runtime_calibration_provenance": runtime_provenance,
        "model_position_limit": model_position_limit,
    }


def _sha256_ids(values: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _generated_ids(tensor: Any, prompt_length: int) -> list[int]:
    return [int(value) for value in tensor[0, prompt_length:].detach().cpu().tolist()]


def _first_divergent_generated_offset(
    left: list[int], right: list[int]
) -> int | None:
    for offset, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return offset
    return min(len(left), len(right)) if len(left) != len(right) else None


def _output_identity_metadata(
    generated: list[int], *, reference: list[int]
) -> dict[str, Any]:
    return {
        "generated_output_token_ids_sha256": _sha256_ids(generated),
        "generated_token_count": len(generated),
        "output_ids_equal_reference": generated == reference,
        "first_divergent_generated_offset_vs_reference": (
            _first_divergent_generated_offset(reference, generated)
        ),
    }


def _attribute_verifier_oracle_comparisons(
    rows: list[dict[str, Any]],
    *,
    prompt_token_ids: list[int],
    original_dflash_generated_token_ids: list[int],
) -> list[dict[str, Any]]:
    attributed = []
    full_reference = [*prompt_token_ids, *original_dflash_generated_token_ids]
    for source in rows:
        row = dict(source)
        prefix = int(row["generated_prefix_length"])
        expected_prefix = full_reference[: len(prompt_token_ids) + prefix]
        expected_hash = _sha256_ids(expected_prefix)
        pairwise = row["pairwise"]
        disagreement_depths = [
            value["first_disagreement_depth"]
            for value in pairwise.values()
            if not value["decision_equal"]
            and value["first_disagreement_depth"] is not None
        ]
        first_depth = min(disagreement_depths) if disagreement_depths else None
        support = {
            "attribution_available": False,
            "attribution_unavailable_reason": None,
            "semantic_prefix_matches_original_dflash": (
                row["semantic_prefix_token_sha256"] == expected_hash
            ),
            "first_verifier_disagreement_depth": first_depth,
            "first_verifier_disagreement_absolute_generated_offset": (
                prefix + first_depth + 1 if first_depth is not None else None
            ),
            "packed_native_matches_original_dflash": None,
            "q1_replay_matches_original_dflash": None,
            "fixed_shape_recheck_matches_original_dflash": None,
        }
        if prefix > len(original_dflash_generated_token_ids):
            support["attribution_unavailable_reason"] = (
                "generated_prefix_beyond_original_dflash_output"
            )
        elif not support["semantic_prefix_matches_original_dflash"]:
            support["attribution_unavailable_reason"] = (
                "semantic_prefix_hash_differs_from_original_dflash"
            )
        else:
            paths = (
                ("packed_native", "packed_native_matches_original_dflash"),
                ("q1_replay", "q1_replay_matches_original_dflash"),
                (
                    "fixed_shape_recheck",
                    "fixed_shape_recheck_matches_original_dflash",
                ),
            )
            sequences = {
                path: row[path]["candidate_next_committed_token_ids"]
                for path, _ in paths
            }
            remaining = original_dflash_generated_token_ids[prefix:]
            if any(len(sequence) > len(remaining) for sequence in sequences.values()):
                support["attribution_unavailable_reason"] = (
                    "original_dflash_continuation_truncated"
                )
            else:
                support["attribution_available"] = True
                for path, field in paths:
                    sequence = sequences[path]
                    support[field] = remaining[: len(sequence)] == sequence
        row["reference_support"] = support
        attributed.append(row)
    return attributed


def _output_relations(
    *,
    target_generated: list[int],
    reference_generated: list[int],
    formal_generated: list[int],
    profile_generated: list[int],
    diagnostic_generated: list[int],
    formal_repeat_output_ids_equal: bool,
) -> dict[str, bool]:
    """Name every control relation instead of inferring the third side."""

    return {
        # Backward-compatible name: target-only versus the formal tree output.
        "target_only_output_ids_equal": target_generated == formal_generated,
        "target_only_original_dflash_reference_output_ids_equal": (
            target_generated == reference_generated
        ),
        "original_dflash_reference_output_ids_equal": (
            reference_generated == formal_generated
        ),
        "formal_repeat_output_ids_equal": formal_repeat_output_ids_equal,
        "control_profile_output_ids_equal": formal_generated == profile_generated,
        "control_diagnostic_output_ids_equal": (
            formal_generated == diagnostic_generated
        ),
    }


def _evaluate_output_contract(relations: dict[str, bool]) -> dict[str, Any]:
    """Apply the frozen Phase 2C actual-execution output oracle."""

    required_relations = (
        "original_dflash_reference_output_ids_equal",
        "formal_repeat_output_ids_equal",
        "control_profile_output_ids_equal",
        "control_diagnostic_output_ids_equal",
    )
    hard_reasons = [name for name in required_relations if not relations[name]]
    target_only_warnings = []
    if not relations["target_only_original_dflash_reference_output_ids_equal"]:
        target_only_warnings.append("target_only_reference_numerical_divergence")
    if not relations["target_only_output_ids_equal"]:
        target_only_warnings.append("target_only_tree_numerical_divergence")
    return {
        "hard_output_relation_names": list(required_relations),
        "hard_output_relations_passed": not hard_reasons,
        "hard_output_failure_reasons": hard_reasons,
        "actual_reference_tree_output_ids_equal": relations[
            "original_dflash_reference_output_ids_equal"
        ],
        "target_only_reference_output_ids_equal": relations[
            "target_only_original_dflash_reference_output_ids_equal"
        ],
        "target_only_tree_output_ids_equal": relations[
            "target_only_output_ids_equal"
        ],
        "target_only_auxiliary_status": (
            target_only_warnings[0] if target_only_warnings else "equal"
        ),
        "target_only_auxiliary_warnings": target_only_warnings,
        "target_only_auxiliary_warning_observed": bool(target_only_warnings),
    }


def _configuration_id(args: argparse.Namespace) -> str:
    return (
        f"{args.candidate_merge_mode}-d{args.refine_depth}-"
        f"p{args.refined_prefix_budget}-r{args.conditional_top_r}-"
        f"n{args.node_budget}-{args.verification_mode}"
    )


def _generation_options(
    args: argparse.Namespace,
    *,
    profiling: bool,
    gold: bool,
    shadow_replay: bool,
    context_length: int,
    verifier_oracle_diagnostics: bool = False,
) -> dict[str, Any]:
    calibration_hash = (
        args.nr_calibration["calibration_sha256"]
        if args.nr_calibration is not None
        else None
    )
    return {
        "enable_online_tree": True,
        "online_tree_policy": "layerwise_beam",
        "online_tree_node_budget": args.node_budget,
        "online_tree_beam_width": 4 if args.node_budget == 16 else 8,
        "online_tree_draft_top_k": args.base_top_m,
        "enable_online_tree_profiling": profiling,
        "enable_online_tree_gold_checks": gold,
        "enable_online_tree_divergence_diagnostics": gold,
        "online_tree_gold_comparison_modes": list(args.gold_comparison_modes),
        "online_tree_gold_layer_localization": args.gold_layer_localization,
        "online_tree_gold_max_rounds": args.gold_max_rounds,
        "online_tree_gold_record_all_nodes": args.gold_record_all_nodes,
        "online_tree_gold_thresholds": args.gold_thresholds,
        "decision_stabilization": "off",
        "decision_margin_threshold": args.decision_margin_threshold,
        "online_selector_mode": args.online_selector_mode,
        "online_selector_config": args.frozen_selector,
        "online_selector_config_hash": args.frozen_selector["config_sha256"],
        "online_selector_context_length": int(context_length),
        "enable_phase2d2_refinement": True,
        "phase2d2_candidate_merge_mode": args.candidate_merge_mode,
        "phase2d2_base_top_m": args.base_top_m,
        "phase2d2_conditional_top_r": args.conditional_top_r,
        "phase2d2_refine_depth": args.refine_depth,
        "phase2d2_refined_prefix_budget": args.refined_prefix_budget,
        "phase2d2_refinement_weight": args.refinement_weight,
        "phase2d2_verification_mode": args.verification_mode,
        "phase2d2_calibration_validated": args.nr_calibration is not None,
        "phase2d2_calibration_hash": calibration_hash,
        "phase2d2_shadow_replay": shadow_replay,
        "enable_phase2d2_verifier_oracle_diagnostics": (
            verifier_oracle_diagnostics
        ),
    }


def _warmup(
    args: argparse.Namespace,
    *,
    torch_module: Any,
    tokenizer: Any,
    target: Any,
    draft: Any,
    workloads: list[dict[str, Any]],
) -> None:
    first = {}
    for workload in workloads:
        first.setdefault(workload["nominal_prompt_length"], workload)
    for length in args.context_lengths:
        input_ids = torch_module.tensor(
            [first[length]["input_ids"]], dtype=torch_module.long, device=args.device
        )
        for _ in range(args.warmup_runs):
            _call_generation(
                torch_module=torch_module,
                draft=draft,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                return_diagnostics=False,
                enable_profiling=False,
                generation_options=_generation_options(
                    args,
                    profiling=False,
                    gold=False,
                    shadow_replay=False,
                    context_length=length,
                ),
            )


def _run_one(
    args: argparse.Namespace,
    *,
    torch_module: Any,
    tokenizer: Any,
    target: Any,
    draft: Any,
    workload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    input_ids = torch_module.tensor(
        [workload["input_ids"]], dtype=torch_module.long, device=args.device
    )
    prompt_length = int(input_ids.shape[1])
    target_start = torch_module.cuda.Event(enable_timing=True)
    target_end = torch_module.cuda.Event(enable_timing=True)
    target_start.record()
    target_only = _target_only_greedy(
        target=target,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    target_end.record()
    target_end.synchronize()
    target_only_ms = float(target_start.elapsed_time(target_end))
    reference, reference_diagnostics, reference_timing, _ = _call_generation(
        torch_module=torch_module,
        draft=draft,
        target=target,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
        return_diagnostics=True,
        enable_profiling=False,
    )
    formal_ids = None
    first_formal_generated = None
    formal_timings = []
    formal_equal = True
    for repeat in range(args.timing_repeats):
        current, _, timing, _ = _call_generation(
            torch_module=torch_module,
            draft=draft,
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            return_diagnostics=False,
            enable_profiling=False,
            generation_options=_generation_options(
                args,
                profiling=False,
                gold=False,
                shadow_replay=False,
                context_length=workload["nominal_prompt_length"],
            ),
        )
        current_generated = _generated_ids(current, prompt_length)
        if formal_ids is None:
            formal_ids = current
            first_formal_generated = current_generated
        formal_equal = formal_equal and bool(torch_module.equal(formal_ids, current))
        formal_timings.append(
            {
                "repeat": repeat,
                **timing,
                **_output_identity_metadata(
                    current_generated,
                    reference=first_formal_generated,
                ),
                "output_ids_equal_repeat_0": (
                    current_generated == first_formal_generated
                ),
                "first_divergent_generated_offset_vs_repeat_0": (
                    _first_divergent_generated_offset(
                        first_formal_generated, current_generated
                    )
                ),
            }
        )
    torch_module.cuda.reset_peak_memory_stats(input_ids.device)
    profile_ids, profile_diagnostics, profile_timing, _ = _call_generation(
        torch_module=torch_module,
        draft=draft,
        target=target,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
        return_diagnostics=True,
        enable_profiling=True,
        generation_options=_generation_options(
            args,
            profiling=True,
            gold=False,
            shadow_replay=False,
            context_length=workload["nominal_prompt_length"],
        ),
    )
    diagnostic_ids, diagnostic, diagnostic_timing, _ = _call_generation(
        torch_module=torch_module,
        draft=draft,
        target=target,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
        return_diagnostics=True,
        enable_profiling=False,
        generation_options=_generation_options(
            args,
            profiling=False,
            gold=args.enable_gold_checks,
            shadow_replay=(
                args.verification_mode not in FIXED_SHAPE_VERIFICATION_MODES
            ),
            context_length=workload["nominal_prompt_length"],
            verifier_oracle_diagnostics=(
                args.enable_verifier_oracle_diagnostics
            ),
        ),
    )
    target_generated = _generated_ids(target_only, prompt_length)
    reference_generated = _generated_ids(reference, prompt_length)
    formal_generated = _generated_ids(formal_ids, prompt_length)
    profile_generated = _generated_ids(profile_ids, prompt_length)
    diagnostic_generated = _generated_ids(diagnostic_ids, prompt_length)
    profile_output_identity = _output_identity_metadata(
        profile_generated, reference=formal_generated
    )
    diagnostic_output_identity = _output_identity_metadata(
        diagnostic_generated, reference=formal_generated
    )
    relations = _output_relations(
        target_generated=target_generated,
        reference_generated=reference_generated,
        formal_generated=formal_generated,
        profile_generated=profile_generated,
        diagnostic_generated=diagnostic_generated,
        formal_repeat_output_ids_equal=formal_equal,
    )
    raw_oracle_comparisons = [
        row["phase2d2"]["verifier_oracle_comparison"]
        for row in diagnostic["rounds"]
        if row.get("phase2d2", {}).get("verifier_oracle_comparison") is not None
    ]
    oracle_comparisons = _attribute_verifier_oracle_comparisons(
        raw_oracle_comparisons,
        prompt_token_ids=[int(value) for value in workload["input_ids"]],
        original_dflash_generated_token_ids=reference_generated,
    )
    oracle_by_round = {row["round_index"]: row for row in oracle_comparisons}
    for row in diagnostic["rounds"]:
        oracle = oracle_by_round.get(row["round_index"])
        if oracle is None:
            continue
        oracle.update(
            {
                "task": workload["longbench_task"],
                "sample_id": workload["sample_id"],
                "source_sample_id": workload["source_sample_id"],
                "nominal_prompt_length": workload["nominal_prompt_length"],
                "actual_prompt_length": workload["actual_prompt_length"],
                "configuration_id": _configuration_id(args),
                "evidence_source": "separate_diagnostics_path",
            }
        )
        phase = row["phase2d2"]
        reference_support = oracle["reference_support"]
        phase["verifier_oracle_comparison"] = oracle
        phase["packed_q1_decision_disagreement"] = not oracle["pairwise"][
            "packed_native_vs_q1_replay"
        ]["decision_equal"]
        phase["packed_fixed_shape_decision_disagreement"] = not oracle[
            "pairwise"
        ]["packed_native_vs_fixed_shape_recheck"]["decision_equal"]
        phase["packed_fixed_shape_first_disagreement_depth"] = oracle[
            "pairwise"
        ]["packed_native_vs_fixed_shape_recheck"]["first_disagreement_depth"]
        phase["guard_triggered_on_packed_q1_disagreement"] = bool(
            phase["guard_triggered"]
            and phase["packed_q1_decision_disagreement"]
        )
        phase["guard_missed_packed_q1_disagreement"] = bool(
            not phase["guard_triggered"]
            and phase["packed_q1_decision_disagreement"]
        )
        phase["original_dflash_reference_attribution_available"] = (
            reference_support["attribution_available"]
        )
        phase["original_dflash_supports_packed_decision"] = reference_support[
            "packed_native_matches_original_dflash"
        ]
        phase["original_dflash_supports_q1_decision"] = reference_support[
            "q1_replay_matches_original_dflash"
        ]
        phase["original_dflash_supports_fixed_shape_decision"] = reference_support[
            "fixed_shape_recheck_matches_original_dflash"
        ]
    structural_reasons = []
    rounds = []
    for row in diagnostic["rounds"]:
        current = dict(row)
        phase = current["phase2d2"]
        if phase["initial_parallel_draft_forward_count"] != 1:
            structural_reasons.append(
                f"round_{row['round_index']}:initial_draft_forward_count"
            )
        if phase["conditional_refinement_forward_count"] > 1:
            structural_reasons.append(
                f"round_{row['round_index']}:conditional_forward_count"
            )
        if (
            args.candidate_merge_mode == "base_only"
            and phase["conditional_refinement_forward_count"]
        ):
            structural_reasons.append(
                f"round_{row['round_index']}:base_control_refined"
            )
        if not phase["node_budget_invariant_passed"]:
            structural_reasons.append(f"round_{row['round_index']}:node_budget")
        if phase["temporary_refinement_state_committed"]:
            structural_reasons.append(
                f"round_{row['round_index']}:temporary_state_commit"
            )
        if phase["fallback_executed"]:
            expected_query_length = (
                1
                if phase["verification_mode"] in Q1_REPLAY_VERIFICATION_MODES
                else 16
            )
            if any(
                value != expected_query_length
                for value in phase["fallback_query_lengths"]
            ):
                structural_reasons.append(
                    f"round_{row['round_index']}:fallback_q_len"
                )
        current.update(
            {
                "task": workload["longbench_task"],
                "sample_id": workload["sample_id"],
                "source_sample_id": workload["source_sample_id"],
                "nominal_prompt_length": workload["nominal_prompt_length"],
                "actual_prompt_length": workload["actual_prompt_length"],
                "configuration_id": _configuration_id(args),
                "component_timing_source": "diagnostics_path",
            }
        )
        current["candidate_quality"] = candidate_quality_for_round(current)
        phase["candidate_refinement_improved_coverage"] = bool(
            current["candidate_quality"]["refined_only_target_recovery_count"]
        )
        phase["candidate_refinement_improved_coverage_unavailable_reason"] = None
        rounds.append(current)
    evidence_common = {
        "schema_version": SCHEMA_VERSION,
        "task": workload["longbench_task"],
        "sample_id": workload["sample_id"],
        "source_sample_id": workload["source_sample_id"],
        "nominal_prompt_length": workload["nominal_prompt_length"],
        "actual_prompt_length": workload["actual_prompt_length"],
        "configuration_id": _configuration_id(args),
        "evidence_source": "separate_diagnostics_path",
    }
    gold_logits = [
        {**evidence_common, **row}
        for row in diagnostic["phase2c"]["packed_logits_gold_checks"]
    ]
    gold_kv = [
        {**evidence_common, **row}
        for row in diagnostic["phase2c"]["kv_compaction_gold_checks"]
    ]
    traversal = [
        {**evidence_common, **row}
        for row in diagnostic["phase2c"]["tree_traversal_checks"]
    ]
    gold_layers = [
        {**evidence_common, **row}
        for row in diagnostic["phase2c"]["gold_layer_localization"]
    ]
    structural_reasons.extend(
        f"gold_logit_round_{row.get('round_index')}:structural"
        for row in gold_logits
        if row.get("structural_status") == "fail" or row.get("structural_issues")
    )
    structural_reasons.extend(
        f"gold_kv_round_{row.get('round_index')}:structural"
        for row in gold_kv
        if row.get("structural_status") == "fail"
        or row.get("status") == "fail_structural"
    )
    structural_reasons.extend(
        f"traversal_round_{row.get('round_index')}:failure"
        for row in traversal
        if row.get("status") == "fail"
        or row.get("checks", {}).get("all_checks_passed") is False
    )
    output_contract = _evaluate_output_contract(relations)
    hard_reasons = list(output_contract["hard_output_failure_reasons"])
    hard_reasons.extend(structural_reasons)
    correctness = {
        "target_only_is_authoritative_greedy_oracle": False,
        "target_only_is_auxiliary_numerical_control": True,
        "target_only_mismatch_is_hard_failure": False,
        "original_dflash_reference_is_recorded_control": True,
        "original_dflash_reference_is_token_exact_oracle": True,
        "base_only_reference_tree_output_exactness_required": (
            args.candidate_merge_mode == "base_only"
        ),
        "base_only_reference_tree_output_ids_equal": (
            relations["original_dflash_reference_output_ids_equal"]
            if args.candidate_merge_mode == "base_only"
            else None
        ),
        **relations,
        **output_contract,
        "structural_checks_passed": not structural_reasons,
        "structural_failure_reasons": structural_reasons,
        "hard_correctness_failure_reasons": hard_reasons,
        "actual_execution_correctness_passed": not hard_reasons,
        "passed": not hard_reasons,
        "gold_status": diagnostic["phase2c"]["gold_overall_status"],
        "gold_numerical_variation_is_diagnostic": True,
    }
    sample = {
        "schema_version": SCHEMA_VERSION,
        "task": workload["longbench_task"],
        "sample_id": workload["sample_id"],
        "source_sample_id": workload["source_sample_id"],
        "source_text_sha256": workload["source_text_sha256"],
        "context_prefix_sha256": workload["context_prefix_sha256"],
        "nominal_prompt_length": workload["nominal_prompt_length"],
        "actual_prompt_length": workload["actual_prompt_length"],
        "configuration_id": _configuration_id(args),
        "candidate_merge_mode": args.candidate_merge_mode,
        "verification_mode": args.verification_mode,
        "node_budget": args.node_budget,
        "generated_token_count": len(formal_generated),
        "generated_output_token_ids": formal_generated,
        "generated_output_token_ids_sha256": _sha256_ids(formal_generated),
        "generated_text": tokenizer.decode(formal_generated, skip_special_tokens=True),
        "target_only": {
            "generated_output_token_ids": target_generated,
            "generated_output_token_ids_sha256": _sha256_ids(target_generated),
            "cuda_elapsed_ms": target_only_ms,
        },
        "original_dflash_reference": {
            "generated_output_token_ids": reference_generated,
            "generated_output_token_ids_sha256": _sha256_ids(reference_generated),
            "cuda_elapsed_ms": reference_timing["cuda_elapsed_ms"],
            "round_count": len(reference_diagnostics["rounds"]),
        },
        "formal_timing_repeats": formal_timings,
        "formal_timing_source": "unprofiled instrumented control path",
        "profiling": {
            "formal_end_to_end": False,
            "measurement": profile_timing,
            "components": profile_diagnostics["profiling"]["components"],
            "memory": profile_diagnostics["profiling"]["memory"],
            "output_ids_equal_formal": relations["control_profile_output_ids_equal"],
            **profile_output_identity,
        },
        "diagnostics": {
            "formal_end_to_end": False,
            "measurement": diagnostic_timing,
            "output_ids_equal_formal": relations["control_diagnostic_output_ids_equal"],
            **diagnostic_output_identity,
        },
        "correctness": correctness,
    }
    pairwise = [
        output_pair_comparison(
            left_name="target_only",
            left_ids=target_generated,
            right_name="original_dflash_reference",
            right_ids=reference_generated,
            prompt_length=prompt_length,
        ),
        output_pair_comparison(
            left_name="target_only",
            left_ids=target_generated,
            right_name="refined_tree",
            right_ids=formal_generated,
            prompt_length=prompt_length,
        ),
        output_pair_comparison(
            left_name="original_dflash_reference",
            left_ids=reference_generated,
            right_name="refined_tree",
            right_ids=formal_generated,
            prompt_length=prompt_length,
        ),
    ]
    for row in pairwise:
        row.update(
            {
                "schema_version": SCHEMA_VERSION,
                "task": workload["longbench_task"],
                "sample_id": workload["sample_id"],
                "nominal_prompt_length": workload["nominal_prompt_length"],
                "configuration_id": _configuration_id(args),
            }
        )
    evidence = {
        "gold_logits": gold_logits,
        "gold_kv": gold_kv,
        "gold_layers": gold_layers,
        "traversal": traversal,
        "verifier_oracle": oracle_comparisons,
        "pairwise": pairwise,
    }
    return sample, rounds, evidence


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _flat(value: Any) -> Any:
    return (
        json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row}) or ["schema_version"]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _flat(row.get(field)) for field in fields})
    temporary.replace(path)


def _comparison(
    args: argparse.Namespace,
    summary: dict[str, Any],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    if not args.base_result_dir:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "unavailable",
            "reason": "no_same_budget_base_result_directory_supplied",
            "same_root_inclusive_node_budget_required": True,
            "rows": [],
        }
    rows = []
    current_identities = {
        (
            row["task"],
            row["sample_id"],
            int(row["nominal_prompt_length"]),
            row["context_prefix_sha256"],
        )
        for row in samples
    }
    for root in args.base_result_dir:
        manifest = _load_json(root / "manifest.json")
        base_budget = int(manifest["phase2d2"]["node_budget"])
        if base_budget != args.node_budget:
            raise ValueError(
                f"fixed-budget comparison rejected N={base_budget} vs N={args.node_budget}"
            )
        if manifest["phase2d2"]["candidate_merge_mode"] != "base_only":
            raise ValueError("fixed-budget comparison baseline is not base_only")
        if (
            manifest["phase2d2"]["frozen_selector_hash"]
            != args.frozen_selector["config_sha256"]
        ):
            raise ValueError("fixed-budget comparison selector hashes differ")
        if (
            manifest["phase2d2"].get("online_selector_mode")
            != args.online_selector_mode
        ):
            raise ValueError("fixed-budget comparison selector modes differ")
        base_samples = _load_jsonl(root / "samples.jsonl")
        base_identities = {
            (
                row["task"],
                row["sample_id"],
                int(row["nominal_prompt_length"]),
                row["context_prefix_sha256"],
            )
            for row in base_samples
        }
        if base_identities != current_identities:
            raise ValueError("fixed-budget comparison cohorts or prefixes differ")
        base = _load_json(root / "refinement-summary.json")
        for current in summary["aggregates"]:
            match = next(
                (
                    row
                    for row in base["aggregates"]
                    if row["task"] == current["task"]
                    and row["nominal_prompt_length"] == current["nominal_prompt_length"]
                ),
                None,
            )
            if match is None:
                continue
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "task": current["task"],
                    "nominal_prompt_length": current["nominal_prompt_length"],
                    "node_budget": args.node_budget,
                    "same_root_inclusive_node_budget": True,
                    "base_configuration_id": match["configuration_id"],
                    "refined_configuration_id": current["configuration_id"],
                    "accepted_tokens_per_verified_node_gain": (
                        current["accepted_tokens_per_verified_node"]
                        - match["accepted_tokens_per_verified_node"]
                    ),
                    "refined_only_target_recovery_gain": (
                        current["candidate_refined_only_recovery_count"]
                        - match["candidate_refined_only_recovery_count"]
                    ),
                    "selected_tree_target_coverage_rate_at_fixed_n_gain": (
                        current["candidate_selected_tree_coverage_rate_at_fixed_n"]
                        - match["candidate_selected_tree_coverage_rate_at_fixed_n"]
                        if current["candidate_selected_tree_coverage_rate_at_fixed_n"]
                        is not None
                        and match["candidate_selected_tree_coverage_rate_at_fixed_n"]
                        is not None
                        else None
                    ),
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if rows else "unavailable",
        "reason": None if rows else "no_matching_task_context_rows",
        "same_root_inclusive_node_budget_required": True,
        "rows": rows,
    }


def _sample_timing_row(
    sample: dict[str, Any], rounds: list[dict[str, Any]]
) -> dict[str, Any]:
    related_rounds = [
        row
        for row in rounds
        if row["task"] == sample["task"]
        and row["sample_id"] == sample["sample_id"]
        and row["nominal_prompt_length"] == sample["nominal_prompt_length"]
        and row["configuration_id"] == sample["configuration_id"]
    ]
    committed = sum(int(row["committed_token_count"]) for row in related_rounds)
    total_ms = statistics.median(
        repeat["cuda_elapsed_ms"] for repeat in sample["formal_timing_repeats"]
    )
    profile = sample["profiling"]["measurement"]["profile_breakdown"]
    components = sample["profiling"]["components"]
    prefill_ms = profile["prefill"]["cuda_elapsed_ms"]
    profile_total_ms = profile["total"]["cuda_elapsed_ms"]
    return {
        "schema_version": SCHEMA_VERSION,
        "task": sample["task"],
        "sample_id": sample["sample_id"],
        "nominal_prompt_length": sample["nominal_prompt_length"],
        "configuration_id": sample["configuration_id"],
        "generation_cuda_latency_median_ms": total_ms,
        "generation_cuda_latency_min_ms": min(
            repeat["cuda_elapsed_ms"] for repeat in sample["formal_timing_repeats"]
        ),
        "generation_cuda_latency_max_ms": max(
            repeat["cuda_elapsed_ms"] for repeat in sample["formal_timing_repeats"]
        ),
        "timing_repeat_count": len(sample["formal_timing_repeats"]),
        "generated_token_count": sample["generated_token_count"],
        "committed_token_count": committed,
        "generated_tokens_per_second": sample["generated_token_count"]
        / (total_ms / 1000.0),
        "generation_cuda_ms_per_returned_token": total_ms
        / sample["generated_token_count"],
        "generation_cuda_ms_per_committed_token": (
            total_ms / committed if committed else None
        ),
        "generation_cuda_ms_per_committed_token_unavailable_reason": (
            None if committed else "no_committed_tokens"
        ),
        "profile_prefill_cuda_elapsed_ms": prefill_ms,
        "profile_decode_cuda_elapsed_ms": profile_total_ms - prefill_ms,
        "profile_total_cuda_elapsed_ms": profile_total_ms,
        "profile_initial_parallel_draft_cuda_elapsed_ms": components[
            "draft_hidden_logits_cuda_elapsed_ms"
        ],
        "profile_conditional_remask_cuda_elapsed_ms": components[
            "conditional_remask_cuda_elapsed_ms"
        ],
        "profile_total_draft_cuda_elapsed_ms": components[
            "total_draft_cuda_elapsed_ms"
        ],
        "profile_packed_verification_cuda_elapsed_ms": components[
            "verification_cuda_elapsed_ms"
        ],
        "profile_total_verification_cuda_elapsed_ms": components[
            "total_verification_cuda_elapsed_ms"
        ],
        "profile_fallback_cuda_elapsed_ms": components[
            "canonical_recheck_cuda_elapsed_ms"
        ],
        "peak_allocated_bytes": sample["profiling"]["memory"]["peak_allocated_bytes"],
        "peak_reserved_bytes": sample["profiling"]["memory"]["peak_reserved_bytes"],
        "target_only_cuda_elapsed_ms": sample["target_only"]["cuda_elapsed_ms"],
        "formal_timing_excludes_gold_diagnostics": True,
        "component_timing_source": "separate_profiling_path",
    }


def _verifier_oracle_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    disagreement_rows = [
        row
        for row in rows
        if any(not relation["decision_equal"] for relation in row["pairwise"].values())
    ]
    first = (
        min(
            disagreement_rows,
            key=lambda row: (
                row["nominal_prompt_length"],
                row["generated_prefix_length"],
                row["round_index"],
            ),
        )
        if disagreement_rows
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic",
        "comparison_round_count": len(rows),
        "decision_disagreement_round_count": len(disagreement_rows),
        "all_q1_query_lengths_are_one": bool(
            rows
            and all(
                row["q1_replay"]["query_lengths"]
                and all(value == 1 for value in row["q1_replay"]["query_lengths"])
                for row in rows
            )
        ),
        "all_fixed_shape_query_lengths_are_block_size_16": bool(
            rows
            and all(
                row["fixed_shape_recheck"]["query_lengths"]
                and all(
                    value == 16
                    for value in row["fixed_shape_recheck"]["query_lengths"]
                )
                for row in rows
            )
        ),
        "all_round_entry_clone_hashes_equal": bool(
            rows
            and all(
                row["round_entry_state"]["all_round_entry_cache_hashes_equal"]
                for row in rows
            )
        ),
        "all_diagnostic_isolation_checks_passed": bool(
            rows
            and all(
                all(row["diagnostic_isolation"].values()) for row in rows
            )
        ),
        "reference_attribution_available_count": sum(
            row["reference_support"]["attribution_available"] for row in rows
        ),
        "first_decision_disagreement": (
            {
                "task": first["task"],
                "sample_id": first["sample_id"],
                "nominal_prompt_length": first["nominal_prompt_length"],
                "round_index": first["round_index"],
                "generated_prefix_length": first["generated_prefix_length"],
                "pairwise": first["pairwise"],
                "reference_support": first["reference_support"],
            }
            if first is not None
            else None
        ),
    }


def _persist(
    output: Path,
    *,
    manifest: dict[str, Any],
    environment: dict[str, Any],
    provenance: dict[str, Any],
    samples: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary = aggregate_results(samples, rounds)
    merge_events = []
    tree_rows = []
    robust_events = []
    for row in rounds:
        common = {
            "schema_version": SCHEMA_VERSION,
            "task": row["task"],
            "sample_id": row["sample_id"],
            "nominal_prompt_length": row["nominal_prompt_length"],
            "configuration_id": row["configuration_id"],
            "round_index": row["round_index"],
        }
        merge_events.extend(
            {**common, **event}
            for event in row["phase2d2"].get("candidate_merge_events", [])
        )
        tree_rows.append({**common, **row["phase2d2"]["tree_spec"]})
        robust_events.append(
            {
                **common,
                **{
                    key: value
                    for key, value in row["phase2d2"].items()
                    if key
                    in {
                        "verification_mode",
                        "packed_tentative_output_token_ids",
                        "replay_authoritative_output_token_ids",
                        "fallback_authoritative_output_token_ids",
                        "first_packed_replay_divergent_decision_depth",
                        "first_packed_replay_divergent_absolute_generated_offset",
                        "packed_verifier_margins",
                        "guard_triggered",
                        "guard_trigger",
                        "authoritative_replay_required_by_mode",
                        "fallback_required_by_mode",
                        "fallback_executed",
                        "fallback_executor",
                        "fallback_forward_count",
                        "fallback_replay_executed",
                        "fallback_replay_token_count",
                        "fallback_query_lengths",
                        "fixed_shape_fallback_triggered",
                        "fixed_shape_fallback_executed",
                        "fixed_shape_forward_count",
                        "fixed_shape_query_lengths",
                        "fixed_shape_temporary_kv_layer_lengths",
                        "fixed_shape_temporary_block_length",
                        "fixed_shape_cropped_committed_kv_length",
                        "fixed_shape_mask_tail_discarded_count",
                        "fixed_shape_round_entry_snapshot_isolated",
                        "fixed_shape_snapshot_unchanged_after_packed",
                        "fixed_shape_snapshot_matches_preverification_cache",
                        "fixed_shape_snapshot_unchanged_after_recheck",
                        "fixed_shape_clone_matches_round_entry",
                        "fixed_shape_clone_storage_independent",
                        "diagnostic_shadow_replay_executed",
                        "diagnostic_shadow_replay_query_lengths",
                        "packed_temporary_kv_discarded",
                        "authoritative_committed_cache_length",
                        "authoritative_committed_target_kv_layer_lengths",
                        "authoritative_committed_target_hidden_length",
                        "authoritative_committed_token_ids",
                        "authoritative_accepted_draft_token_ids",
                        "authoritative_next_root_token_id",
                        "authoritative_next_root_absolute_sequence_position",
                        "authoritative_next_root_transition_slot_count",
                        "authoritative_next_root_matches_output_transition",
                        "authoritative_next_root_kind",
                        "correction_or_bonus_committed_in_current_round",
                        "sibling_kv_removed_from_committed_cache",
                        "fixed_shape_mask_tail_absent_from_committed_cache",
                        "terminal_round",
                        "truncated_round",
                        "final_output_source",
                        "guard_prevented_output_divergence",
                        "guard_false_positive",
                        "guard_false_negative",
                        "packed_numerical_decision_divergence",
                        "verifier_relevant_argmax_flip",
                        "packed_q1_decision_disagreement",
                        "packed_fixed_shape_decision_disagreement",
                        "packed_fixed_shape_first_disagreement_depth",
                        "guard_triggered_on_packed_q1_disagreement",
                        "guard_missed_packed_q1_disagreement",
                        "original_dflash_reference_attribution_available",
                        "original_dflash_supports_packed_decision",
                        "original_dflash_supports_q1_decision",
                        "original_dflash_supports_fixed_shape_decision",
                        "replay_decisions",
                        "fallback_decisions",
                    }
                },
            }
        )
    try:
        comparison = _comparison(args, summary, samples)
    except Exception as exc:
        if manifest.get("status") != "error":
            raise
        comparison = {
            "schema_version": SCHEMA_VERSION,
            "status": "unavailable",
            "reason": f"run_failed_before_fixed_budget_comparison:{type(exc).__name__}",
            "same_root_inclusive_node_budget_required": True,
            "rows": [],
        }
    fallback_summary = {
        "schema_version": SCHEMA_VERSION,
        "round_count": len(rounds),
        "fallback_round_count": sum(
            row["phase2d2"].get(
                "fallback_executed",
                row["phase2d2"]["fallback_replay_executed"],
            )
            for row in rounds
        ),
        "fallback_rate": (
            sum(
                row["phase2d2"].get(
                    "fallback_executed",
                    row["phase2d2"]["fallback_replay_executed"],
                )
                for row in rounds
            )
            / len(rounds)
            if rounds
            else None
        ),
        "fallback_replay_token_count": sum(
            row["phase2d2"]["fallback_replay_token_count"] for row in rounds
        ),
        "fallback_executor_distribution": dict(
            sorted(
                Counter(
                    row["phase2d2"].get("fallback_executor", "q1_replay")
                    if row["phase2d2"].get(
                        "fallback_executed",
                        row["phase2d2"]["fallback_replay_executed"],
                    )
                    else "none"
                    for row in rounds
                ).items()
            )
        ),
        "fallback_target_forward_count": sum(
            row["phase2d2"].get(
                "fallback_forward_count",
                row["phase2d2"]["fallback_replay_token_count"],
            )
            for row in rounds
        ),
        "fixed_shape_fallback_round_count": sum(
            row["phase2d2"].get("fixed_shape_fallback_executed", False)
            for row in rounds
        ),
        "packed_fixed_shape_decision_disagreement_count": sum(
            row["phase2d2"].get(
                "packed_fixed_shape_decision_disagreement", False
            )
            for row in rounds
        ),
        "triggered_packed_fixed_shape_decision_disagreement_count": sum(
            row["phase2d2"].get("fixed_shape_fallback_triggered", False)
            and row["phase2d2"].get(
                "packed_fixed_shape_decision_disagreement", False
            )
            for row in rounds
        ),
        "non_triggered_packed_fixed_shape_decision_disagreement_count": sum(
            not row["phase2d2"].get("fixed_shape_fallback_triggered", False)
            and row["phase2d2"].get(
                "packed_fixed_shape_decision_disagreement", False
            )
            for row in rounds
        ),
        "fixed_shape_authoritative_commit_count": sum(
            row["phase2d2"].get("final_output_source")
            == "fixed_shape_recheck"
            for row in rounds
        ),
        "reference_attributed_supports_packed_count": sum(
            row["phase2d2"].get("original_dflash_supports_packed_decision") is True
            for row in rounds
        ),
        "reference_attributed_supports_fixed_shape_count": sum(
            row["phase2d2"].get("original_dflash_supports_fixed_shape_decision")
            is True
            for row in rounds
        ),
        "reference_supported_harmful_fixed_shape_correction_count": sum(
            row["phase2d2"].get("original_dflash_reference_attribution_available")
            and row["phase2d2"].get("original_dflash_supports_packed_decision")
            is True
            and row["phase2d2"].get(
                "original_dflash_supports_fixed_shape_decision"
            )
            is False
            for row in rounds
        ),
        "diagnostic_shadow_replay_round_count": sum(
            row["phase2d2"].get("diagnostic_shadow_replay_executed", False)
            for row in rounds
        ),
        "guard_false_positive_count": sum(
            row["phase2d2"]["guard_false_positive"] for row in rounds
        ),
        "guard_false_negative_count": sum(
            row["phase2d2"]["guard_false_negative"] for row in rounds
        ),
    }
    correctness = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok"
        if samples and all(row["correctness"]["passed"] for row in samples)
        else "failed",
        "official_checker_passed": bool(
            samples and all(row["correctness"]["passed"] for row in samples)
        ),
        "target_only_output_exactness_required": False,
        "target_only_is_auxiliary_numerical_control": True,
        "original_dflash_reference_output_exactness_required": True,
        "all_target_only_output_ids_equal": bool(
            samples
            and all(
                row["correctness"]["target_only_output_ids_equal"] for row in samples
            )
        ),
        "all_target_only_original_dflash_reference_output_ids_equal": bool(
            samples
            and all(
                row["correctness"][
                    "target_only_original_dflash_reference_output_ids_equal"
                ]
                for row in samples
            )
        ),
        "all_original_dflash_reference_output_ids_equal": bool(
            samples
            and all(
                row["correctness"][
                    "original_dflash_reference_output_ids_equal"
                ]
                for row in samples
            )
        ),
        "diagnostic_target_only_variation_observed": any(
            row["correctness"]["target_only_auxiliary_warning_observed"]
            for row in samples
        ),
        "target_only_auxiliary_statuses": [
            row["correctness"]["target_only_auxiliary_status"] for row in samples
        ],
        "actual_execution_correctness_passed": bool(
            samples
            and all(
                row["correctness"]["actual_execution_correctness_passed"]
                for row in samples
            )
        ),
        "structural_correctness_passed": all(
            row["correctness"]["structural_checks_passed"] for row in samples
        ),
        "hard_correctness_failure_reasons": [
            reason
            for row in samples
            for reason in row["correctness"]["hard_correctness_failure_reasons"]
        ],
        "gold_numerical_variation_preserved_as_diagnostics": True,
        "records": [row["correctness"] for row in samples],
    }
    pairwise = [row for item in evidence for row in item["pairwise"]]
    output_divergence = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "pass"
            if correctness["all_original_dflash_reference_output_ids_equal"]
            else "failed"
        ),
        "authoritative_relation": "original_dflash_reference_vs_tree",
        "records": pairwise,
    }
    gold_logits = [row for item in evidence for row in item["gold_logits"]]
    gold_kv = [row for item in evidence for row in item["gold_kv"]]
    gold_layers = [row for item in evidence for row in item["gold_layers"]]
    traversal = [row for item in evidence for row in item["traversal"]]
    verifier_oracle = [
        row for item in evidence for row in item.get("verifier_oracle", [])
    ]
    verifier_oracle_summary = _verifier_oracle_summary(verifier_oracle)
    gold_summary = build_gold_comparison_summary(gold_logits, gold_kv)
    timing_rows = [_sample_timing_row(row, rounds) for row in samples]
    timing_summary = {"schema_version": SCHEMA_VERSION, "rows": timing_rows}
    offline_online = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic",
        "online_acceptance_measured": True,
        "offline_target_labels_not_used_at_runtime": True,
        "rows": [
            {
                "schema_version": SCHEMA_VERSION,
                "task": row["task"],
                "sample_id": row["sample_id"],
                "nominal_prompt_length": row["nominal_prompt_length"],
                "round_index": row["round_index"],
                **row["candidate_quality"],
            }
            for row in rounds
        ],
    }
    stable_coverage = (
        all(row["refined_only_target_recovery_gain"] > 0 for row in comparison["rows"])
        if len(comparison["rows"]) >= 2
        else None
    )
    stable_efficiency = (
        all(
            row["accepted_tokens_per_verified_node_gain"] > 0
            for row in comparison["rows"]
        )
        if len(comparison["rows"]) >= 2
        else None
    )
    decision = build_phase2d2_decision(
        official_checker_passed=correctness["official_checker_passed"],
        structural_correctness_passed=correctness["structural_correctness_passed"],
        guard_false_negative_count=fallback_summary["guard_false_negative_count"],
        independent_task_count=len({row["task"] for row in samples}),
        stable_coverage_gain=stable_coverage,
        stable_online_efficiency_gain=stable_efficiency,
        conditional_cost_acceptable=None,
        full_second_pass_too_heavy=None,
    )
    if manifest.get("status") != "error":
        manifest["status"] = (
            "ok" if correctness["official_checker_passed"] else "failed"
        )
        manifest["error"] = (
            None
            if correctness["official_checker_passed"]
            else {
                "type": "Phase2D2ACorrectnessFailure",
                "message": str(correctness["hard_correctness_failure_reasons"]),
            }
        )
    if provenance.get("status") != "error":
        provenance["status"] = manifest["status"]
    manifest["phase2d2_decision"] = decision["decision"]
    manifest["workload"]["inventory"] = manifest["workload"].get("inventory")
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "environment.json", environment)
    _write_json(output / "source-provenance.json", provenance)
    _write_json(output / "workload-inventory.json", manifest["workload"]["inventory"])
    _write_jsonl(output / "samples.jsonl", samples)
    _write_jsonl(output / "refinement-rounds.jsonl", rounds)
    _write_jsonl(output / "candidate-merge-events.jsonl", merge_events)
    _write_jsonl(output / "refined-tree-rounds.jsonl", tree_rows)
    _write_json(output / "refinement-summary.json", summary)
    _write_csv(output / "refinement-summary.csv", summary["aggregates"])
    _write_csv(output / "candidate-quality-summary.csv", offline_online["rows"])
    _write_json(output / "fixed-budget-comparison.json", comparison)
    _write_csv(output / "fixed-budget-comparison.csv", comparison["rows"])
    _write_jsonl(output / "robust-verification-events.jsonl", robust_events)
    _write_jsonl(output / "verifier-oracle-comparisons.jsonl", verifier_oracle)
    _write_json(
        output / "verifier-oracle-summary.json", verifier_oracle_summary
    )
    _write_jsonl(output / "tree-traversal-checks.jsonl", traversal)
    _write_json(output / "fallback-summary.json", fallback_summary)
    _write_csv(output / "fallback-summary.csv", [fallback_summary])
    _write_json(output / "correctness-summary.json", correctness)
    _write_json(output / "output-divergence.json", output_divergence)
    _write_csv(output / "output-divergence.csv", pairwise)
    _write_json(output / "offline-online-comparison.json", offline_online)
    _write_csv(output / "offline-online-comparison.csv", offline_online["rows"])
    _write_json(output / "timing-summary.json", timing_summary)
    _write_csv(output / "timing-summary.csv", timing_rows)
    _write_json(output / "phase2d2-decision.json", decision)
    _write_jsonl(output / "gold-logit-checks.jsonl", gold_logits)
    _write_jsonl(output / "gold-kv-checks.jsonl", gold_kv)
    _write_jsonl(output / "gold-layer-localization.jsonl", gold_layers)
    _write_json(output / "gold-comparison-summary.json", gold_summary)
    _write_csv(
        output / "gold-comparison-summary.csv",
        [
            {"schema_version": SCHEMA_VERSION, **row}
            for row in gold_summary.get("groups", [])
        ],
    )
    return correctness


def _run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "phase": "2D.2A",
        "baseline_sha": BASELINE_SHA,
        "configuration_id": _configuration_id(args),
        "candidate_merge_mode": args.candidate_merge_mode,
        "base_top_m": args.base_top_m,
        "conditional_top_r": args.conditional_top_r,
        "refine_depth": args.refine_depth,
        "refined_prefix_budget": args.refined_prefix_budget,
        "refinement_weight": args.refinement_weight,
        "node_budget": args.node_budget,
        "verification_mode": args.verification_mode,
        "online_selector_mode": args.online_selector_mode,
        "gold_checks_enabled": args.enable_gold_checks,
        "gold_comparison_modes": list(args.gold_comparison_modes),
        "gold_max_rounds": args.gold_max_rounds,
        "verifier_oracle_diagnostics_enabled": (
            args.enable_verifier_oracle_diagnostics
        ),
        "decision_margin_threshold": args.decision_margin_threshold,
        "frozen_selector_hash": args.frozen_selector["config_sha256"],
        "frozen_refinement_config_hash": (
            args.frozen_refinement.get("config_sha256")
            if args.frozen_refinement is not None
            else None
        ),
        "frozen_refinement_configuration_id": args.frozen_refinement_config_id,
        "evidence_role": args.evidence_role,
        "nr_calibration_hash": (
            args.nr_calibration["calibration_sha256"]
            if args.nr_calibration is not None
            else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Full data, tokenizer, materialization, CUDA, and calibration preflight is
    # intentionally complete before the formal output directory exists.
    preflight = _preflight(args)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    command = shlex.join(
        [sys.executable, *(argv if argv is not None else sys.argv[1:])]
    )
    environment = collect_environment(run_config=_run_config(args), run_command=command)
    manifest = phase1b_manifest(args, environment, command)
    manifest.update(
        {
            "phase": "2D.2A",
            "baseline_sha": BASELINE_SHA,
            "expected_branch": BRANCH,
            "phase2d2": _run_config(args),
            "materialization": {
                "id": preflight["materialization"]["materialization_id"],
                "instrumented_custom_code_changed": True,
                "independent_new_directory_required": True,
            },
            "preflight_model_position_limit": preflight["model_position_limit"],
            "source_audit": {
                "mask_token_id": 151669,
                "draft_block_layout": "root at position 0; draft prefix begins at 1; MASK suffix",
                "conditional_attention": "non-causal DFlash attention over injected context and all 16 block positions",
                "conditional_cache": "deep-copied round-entry draft cache, internal batch expanded, never committed",
                "position_ids": "persistent draft cache offset through target_hidden injection plus 16-token block",
                "conditional_logits": "last 15 hidden positions predict draft positions 1..15",
                "generation_control_paths": {
                    "target_only": (
                        "target.generate with explicit all-ones attention mask, "
                        "greedy decoding, and inactive sampling arguments suppressed"
                    ),
                    "original_dflash_reference": (
                        "unchanged draft.spec_generate at temperature zero without "
                        "online-tree or Phase 2D.2 generation options"
                    ),
                    "base_only_tree": (
                        "frozen-selector Phase 2C packed-tree construction; the "
                        "default selector mode is top1_packed_tree, conditional "
                        "refinement forward count is zero, and no refined tree "
                        "replaces the seed tree"
                    ),
                    "verification_mode_effect": (
                        "packed_native commits packed traversal/cache; always_replay "
                        "commits the existing q_len=1 replay path, whose numerical "
                        "oracle status is not assumed"
                    ),
                    "verifier_oracle_diagnostics": (
                        "packed native, q_len=1 replay, and fixed-shape q_len=16 "
                        "recheck run only on independent diagnostic cache clones"
                    ),
                },
            },
            "measurement_policy": {
                "formal_end_to_end": "unprofiled control repeats",
                "component_timing": "separate profiling run",
                "gold_and_shadow_replay": "separate diagnostic run",
                "verifier_oracle_comparison": "separate unprofiled diagnostic run",
                "target_only_is_authoritative": False,
                "target_only_is_auxiliary_numerical_control": True,
                "original_dflash_is_token_exact_reference": True,
                "fixed_target_budget_includes_root": True,
            },
            "artifacts": [*ROOT_ARTIFACTS, *GOLD_ARTIFACTS],
        }
    )
    manifest["workload"]["inventory"] = preflight["inventory"]
    provenance: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "started",
        "baseline_sha": BASELINE_SHA,
        "materialization_id": preflight["materialization"]["materialization_id"],
        "frozen_selector_hash": args.frozen_selector["config_sha256"],
        "nr_calibration_hash": (
            preflight["calibration"]["calibration_sha256"]
            if preflight["calibration"] is not None
            else None
        ),
    }
    samples: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    exit_code = 1
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with (output / "terminal.log").open("w", encoding="utf-8", buffering=1) as log:
        sys.stdout = TeeStream(original_stdout, log)
        sys.stderr = TeeStream(original_stderr, log)
        try:
            torch_module, tokenizer, target, draft, workloads = _load_inputs(
                args, manifest, provenance
            )
            if (
                manifest["attention_backend"]["target"][
                    "configured_attention_implementation"
                ]
                != "sdpa"
            ):
                raise RuntimeError("Phase 2D.2A requires target SDPA")
            _warmup(
                args,
                torch_module=torch_module,
                tokenizer=tokenizer,
                target=target,
                draft=draft,
                workloads=workloads,
            )
            for index, workload in enumerate(workloads, 1):
                print(
                    f"[{index}/{len(workloads)}] {workload['longbench_task']} "
                    f"{workload['sample_id']} L={workload['nominal_prompt_length']}"
                )
                sample, current_rounds, current_evidence = _run_one(
                    args,
                    torch_module=torch_module,
                    tokenizer=tokenizer,
                    target=target,
                    draft=draft,
                    workload=workload,
                )
                samples.append(sample)
                rounds.extend(current_rounds)
                evidence.append(current_evidence)
                # Atomic partial evidence before any hard-gate exception.
                _write_jsonl(output / "samples.jsonl", samples)
                _write_jsonl(output / "refinement-rounds.jsonl", rounds)
            correctness = _persist(
                output,
                manifest=manifest,
                environment=environment,
                provenance=provenance,
                samples=samples,
                rounds=rounds,
                evidence=evidence,
                args=args,
            )
            if not correctness["official_checker_passed"]:
                raise RuntimeError(
                    "Phase 2D.2A hard correctness failure: "
                    f"{correctness['hard_correctness_failure_reasons']}"
                )
            checker_result = check_output(output)
            manifest["official_checker_result"] = checker_result
            _write_json(output / "manifest.json", manifest)
            exit_code = 0
        except Exception as exc:  # noqa: BLE001 - preserve failure evidence.
            manifest["status"] = "error"
            manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
            provenance["status"] = "error"
            provenance["error"] = manifest["error"]
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
            try:
                _persist(
                    output,
                    manifest=manifest,
                    environment=environment,
                    provenance=provenance,
                    samples=samples,
                    rounds=rounds,
                    evidence=evidence,
                    args=args,
                )
            except Exception:  # noqa: BLE001
                traceback.print_exc()
            exit_code = 1
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
