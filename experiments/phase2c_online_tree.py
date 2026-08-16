#!/usr/bin/env python3
"""Run Phase 2C online bounded marginal-tree verification on one LongBench task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import statistics
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase2c.alignment import align_offline_online
from dflash.phase2c.core import (
    SUPPORTED_BEAM_WIDTHS,
    SUPPORTED_NODE_BUDGETS,
    TIE_BREAKING_RULE,
)
from dflash.phase2c.divergence import (
    build_round_differential,
    classify_layerwise_divergence,
    output_pair_comparison,
)
from dflash.phase2c.gold import (
    DEFAULT_GOLD_THRESHOLDS,
    GOLD_COMPARISON_MODES,
    build_gold_comparison_summary,
    validate_gold_modes,
    validate_gold_thresholds,
)
from dflash.phase2c.metrics import (
    aggregate_tree_results,
    distribution,
    summary_csv_rows,
)
from dflash.phase2c.reference_state import (
    compare_reference_tree_checkpoints,
    verifier_evidence_by_prefix,
)
from dflash.phase2c.stabilization import (
    DECISION_STABILIZATION_MODES,
    DEFAULT_MARGIN_THRESHOLDS,
    decision_margin_frontier,
    stabilization_summary_csv_rows,
    summarize_decision_stabilization,
    validate_decision_stabilization,
)
from experiments.phase0_smoke import TeeStream
from experiments.phase1_observability import _atomic_write_jsonl, _call_generation
from experiments.phase1b_longctx import (
    build_parser as build_phase1b_parser,
)
from experiments.phase1b_longctx import new_manifest as phase1b_new_manifest
from experiments.phase1b_longctx import run_configuration as phase1b_configuration
from experiments.phase1b_longctx import validate_args as validate_phase1b_args
from experiments.phase2a_profile import _load_inputs
from scripts.collect_environment import collect_environment, write_json

SCHEMA_VERSION = "1.0"
TREE_POLICIES = ("top1_chain", "layerwise_beam")
SUMMARY_CSV_FIELDS = (
    "task",
    "nominal_prompt_length",
    "configuration_id",
    "tree_policy",
    "requested_node_budget",
    "beam_width",
    "sample_count",
    "round_count",
    "non_terminal_round_count",
    "returned_token_count",
    "total_committed_token_count",
    "mean_accepted_draft_tokens",
    "mean_non_terminal_committed_tokens",
    "mean_verified_nodes_per_round",
    "mean_additional_verified_nodes_vs_dflash",
    "first_rejection_recovery_fraction",
    "generation_cuda_latency_median_ms",
    "generation_cuda_latency_mean_ms",
    "generation_cuda_latency_p90_ms",
    "generated_tokens_per_second_median",
    "relative_acceptance_improvement",
    "relative_round_count_reduction",
    "relative_end_to_end_speedup",
    "relative_verification_latency_increase",
)
OFFLINE_CSV_FIELDS = (
    "task",
    "sample_id",
    "context_length",
    "generated_root_offset",
    "policy",
    "node_budget",
    "beam_width",
    "status",
    "offline_covered_depth",
    "online_accepted_draft_token_count",
    "difference",
    "mismatch_reason",
)
GOLD_COMPARISON_CSV_FIELDS = (
    "task",
    "sample_id",
    "nominal_prompt_length",
    "configuration_id",
    "comparison_mode",
    "availability",
    "status",
    "backend",
    "dtype",
    "compared_node_count",
    "argmax_mismatch_count",
    "verifier_decision_mismatch_count",
    "top8_ordered_mismatch_count",
    "maximum_absolute_error",
    "mean_absolute_error",
    "p99_absolute_error",
    "first_failing_depth",
    "first_failing_node",
    "first_numerical_threshold_exceeding_depth",
    "first_numerical_threshold_exceeding_node",
    "first_top8_mismatch_depth",
    "first_top8_mismatch_node",
    "first_semantic_mismatch_depth",
    "first_semantic_mismatch_node",
    "first_structural_mismatch_depth",
    "first_structural_mismatch_node",
    "structural_status",
    "semantic_status",
    "numerical_status",
    "warning",
    "reason",
)
OUTPUT_DIVERGENCE_CSV_FIELDS = (
    "task",
    "sample_id",
    "nominal_prompt_length",
    "configuration_id",
    "status",
    "reference_returned_token_count",
    "tree_returned_token_count",
    "longest_common_prefix_length",
    "first_divergent_generated_token_offset",
    "first_divergent_absolute_sequence_position",
    "reference_token_id",
    "reference_token_text",
    "tree_token_id",
    "tree_token_text",
    "divergent_tree_round_index",
    "divergent_token_kind",
    "divergent_round_context_length_before",
    "divergent_round_context_length_after",
    "divergent_round_persistent_kv_length_before",
    "divergent_round_persistent_kv_length_after",
    "failure_class",
    "exception_type",
    "exception_message",
)
STABILIZATION_SUMMARY_CSV_FIELDS = (
    "stabilization_mode",
    "decision_margin_threshold",
    "sample_count",
    "round_count",
    "round_recheck_rate",
    "decision_divergence_rate",
    "output_divergence_rate",
    "threshold_false_negative_rate",
    "recheck_cuda_mean_ms",
    "recheck_cuda_p50_ms",
    "recheck_cuda_p90_ms",
    "end_to_end_cuda_median_ms",
    "mean_accepted_draft_tokens",
    "latency_per_returned_token_ms",
    "speedup_relative_to_original_dflash",
    "speedup_relative_to_target_only",
    "speedup_unavailable_reason",
)
MARGIN_FRONTIER_CSV_FIELDS = (
    "threshold",
    "eligible_round_count",
    "divergent_round_count",
    "triggered_round_count",
    "captured_divergence_count",
    "false_negative_count",
    "trigger_rate",
    "captured_divergence_rate",
    "false_negative_rate",
    "strictly_lossless_supported",
    "estimated_recheck_cuda_mean_ms",
    "estimated_recheck_cuda_p50_ms",
    "estimated_recheck_cuda_p90_ms",
)
REFERENCE_STATE_CSV_FIELDS = (
    "task",
    "sample_id",
    "nominal_prompt_length",
    "actual_prompt_length",
    "configuration_id",
    "generated_prefix_length",
    "comparison_status",
    "semantic_prefix_token_exact",
    "reference_round_index",
    "tree_round_index",
    "target_kv_structural_equal",
    "target_kv_numerically_equal",
    "target_kv_maximum_sample_abs_error",
    "target_hidden_structural_equal",
    "target_hidden_numerically_equal",
    "target_hidden_max_abs_error",
    "draft_top1_candidates_equal",
    "draft_topk_candidates_equal",
    "semantic_probe_argmax_equal",
    "semantic_probe_top8_ordered_equal",
    "local_packed_verifier_decision_flip",
    "commit_transition_semantics_equal",
    "mere_numerical_variation_only",
    "semantic_consequence_observed",
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_phase1b_parser()
    parser.description = __doc__
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Run one stable LongBench sample ID at every requested length.",
    )
    parser.add_argument(
        "--tree-policy",
        action="append",
        choices=TREE_POLICIES,
        dest="tree_policies",
        help="Repeat to run multiple policies after one model load.",
    )
    parser.add_argument(
        "--node-budget",
        action="append",
        type=int,
        dest="node_budgets",
        help="Repeat for multiple root-inclusive budgets.",
    )
    parser.add_argument(
        "--beam-width",
        action="append",
        type=int,
        dest="beam_widths",
        help="Repeat for multiple fixed layerwise widths.",
    )
    parser.add_argument("--draft-top-k", type=int, default=16)
    parser.add_argument(
        "--decision-stabilization",
        choices=DECISION_STABILIZATION_MODES,
        default="off",
    )
    parser.add_argument("--decision-margin-threshold", type=float, default=0.1)
    parser.add_argument("--enable-gold-checks", action="store_true")
    parser.add_argument(
        "--gold-comparison-mode",
        action="append",
        choices=GOLD_COMPARISON_MODES,
        dest="gold_comparison_modes",
        help="Repeat to request multiple Phase 2C.1 logits controls.",
    )
    parser.add_argument("--gold-layer-localization", action="store_true")
    parser.add_argument("--gold-max-rounds", type=int, default=1)
    parser.add_argument("--gold-record-all-nodes", action="store_true")
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
    parser.add_argument("--phase2b-capture-dir", action="append", type=Path, default=[])
    parser.add_argument("--phase2b1-analysis-dir", type=Path, default=None)
    parser.add_argument(
        "--enable-reference-state-diagnostics",
        action="store_true",
        help=(
            "Run diagnostic-only Original-DFlash/tree state checkpoint replays "
            "outside formal timing."
        ),
    )
    parser.set_defaults(output_dir=Path("outputs/phase2c-online-tree"))
    return parser


def _unique(values: list[Any], *, label: str) -> tuple[Any, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} values must be unique")
    return tuple(values)


def tree_configurations(args: argparse.Namespace) -> list[dict[str, Any]]:
    policies = args.tree_policies or ["top1_chain"]
    budgets = args.node_budgets or [16]
    widths = args.beam_widths or [4]
    configurations = []
    if policies == ("top1_chain",) and budgets != (16,):
        raise ValueError("top1_chain supports only --node-budget 16")
    for policy in policies:
        if policy == "top1_chain":
            if 16 not in budgets:
                raise ValueError("top1_chain requires --node-budget 16")
            configurations.append(
                {
                    "tree_policy": policy,
                    "node_budget": 16,
                    "beam_width": 1,
                    "draft_top_k": args.draft_top_k,
                    "configuration_id": "top1_chain-b16",
                    "decision_stabilization": args.decision_stabilization,
                    "decision_margin_threshold": args.decision_margin_threshold,
                }
            )
            continue
        for budget in budgets:
            for width in widths:
                configurations.append(
                    {
                        "tree_policy": policy,
                        "node_budget": budget,
                        "beam_width": width,
                        "draft_top_k": args.draft_top_k,
                        "configuration_id": f"layerwise_beam-b{budget}-w{width}",
                        "decision_stabilization": args.decision_stabilization,
                        "decision_margin_threshold": args.decision_margin_threshold,
                    }
                )
    return configurations


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = validate_phase1b_args(build_parser().parse_args(argv))
    if args.inspect_workload_only:
        raise ValueError("Phase 2C does not support --inspect-workload-only")
    args.tree_policies = _unique(
        args.tree_policies or ["top1_chain"], label="tree policy"
    )
    args.node_budgets = _unique(args.node_budgets or [16], label="node budget")
    args.beam_widths = _unique(args.beam_widths or [4], label="beam width")
    args.gold_comparison_modes = validate_gold_modes(
        args.gold_comparison_modes or ["native"]
    )
    args.gold_thresholds = validate_gold_thresholds(
        {
            "maximum_absolute_error": args.gold_max_abs_threshold,
            "mean_absolute_error": args.gold_mean_abs_threshold,
            "relative_error_epsilon": args.gold_relative_epsilon,
            "layer_localization_p99_error": args.gold_layer_threshold,
        }
    )
    args.decision_stabilization, args.decision_margin_threshold = (
        validate_decision_stabilization(
            args.decision_stabilization, args.decision_margin_threshold
        )
    )
    if any(value not in SUPPORTED_NODE_BUDGETS for value in args.node_budgets):
        raise ValueError("node budgets must be chosen from 16,32,64,128")
    if any(value not in SUPPORTED_BEAM_WIDTHS for value in args.beam_widths):
        raise ValueError("beam widths must be chosen from 2,4,8,16,32")
    if not 2 <= args.draft_top_k <= 16:
        raise ValueError("--draft-top-k must be between 2 and 16")
    if args.sample_id is not None and args.sample_policy != "matched":
        raise ValueError("--sample-id requires --sample-policy matched")
    if args.phase2b_capture_dir and args.phase2b1_analysis_dir is None:
        raise ValueError(
            "--phase2b-capture-dir requires --phase2b1-analysis-dir for comparison"
        )
    if args.enable_gold_checks and any(
        config["node_budget"] > 32 for config in tree_configurations(args)
    ):
        raise ValueError("gold checks support node budgets 16 and 32 only")
    if args.gold_max_rounds < 1:
        raise ValueError("--gold-max-rounds must be positive")
    if args.gold_layer_localization and not args.enable_gold_checks:
        raise ValueError("--gold-layer-localization requires --enable-gold-checks")
    if args.gold_record_all_nodes and not args.enable_gold_checks:
        raise ValueError("--gold-record-all-nodes requires --enable-gold-checks")
    configurations = tree_configurations(args)
    if args.enable_reference_state_diagnostics:
        if args.decision_stabilization != "shadow":
            raise ValueError(
                "Phase 2C.4 reference-state diagnostics require shadow mode"
            )
        if not args.enable_gold_checks:
            raise ValueError(
                "Phase 2C.4 reference-state diagnostics require gold checks"
            )
        required_modes = {"native", "same_mask", "math"}
        missing_modes = required_modes - set(args.gold_comparison_modes)
        if missing_modes:
            raise ValueError(
                "Phase 2C.4 requires gold comparison modes native, same_mask, "
                f"and math; missing {sorted(missing_modes)}"
            )
        if any(config["tree_policy"] != "layerwise_beam" for config in configurations):
            raise ValueError(
                "Phase 2C.4 reference-state diagnostics require layerwise_beam"
            )
    return args


def run_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **phase1b_configuration(args),
        "phase": "2C",
        "phase_detail": (
            "2C.4" if args.enable_reference_state_diagnostics else "2C.3"
        ),
        "mode": "online_bounded_marginal_tree",
        "sample_id": args.sample_id,
        "tree_configurations": tree_configurations(args),
        "draft_top_k": args.draft_top_k,
        "decision_stabilization": args.decision_stabilization,
        "decision_margin_threshold": args.decision_margin_threshold,
        "decision_margin_frontier_thresholds": list(DEFAULT_MARGIN_THRESHOLDS),
        "enable_gold_checks": args.enable_gold_checks,
        "gold_comparison_modes": list(args.gold_comparison_modes),
        "gold_layer_localization": args.gold_layer_localization,
        "gold_max_rounds": args.gold_max_rounds,
        "gold_record_all_nodes": args.gold_record_all_nodes,
        "gold_thresholds": args.gold_thresholds,
        "enable_reference_state_diagnostics": (
            args.enable_reference_state_diagnostics
        ),
        "phase2b_capture_dirs": [str(path) for path in args.phase2b_capture_dir],
        "phase2b1_analysis_dir": (
            str(args.phase2b1_analysis_dir) if args.phase2b1_analysis_dir else None
        ),
    }


def _new_manifest(
    args: argparse.Namespace, environment: dict[str, Any], command: str
) -> dict[str, Any]:
    manifest = phase1b_new_manifest(args, environment, command)
    manifest["configuration"] = run_configuration(args)
    manifest["phase2c"] = {
        "policy_class": "DDTree/OPT-Tree-style marginal-tree baseline",
        "policy_uses_target_information": False,
        "maximum_draft_depth": 15,
        "node_budget_includes_root": True,
        "score_rule": "mean cumulative captured draft log probability",
        "tie_breaking": TIE_BREAKING_RULE,
        "flatten_order": "depth, policy score order, deterministic token-ID ties",
        "mask_semantics": "persistent prefix plus self and tree ancestors only",
        "position_ids": "context_length_before + tree depth",
        "cache_position": "prefix_length + packed node index",
        "kv_compaction": "gather prefix plus root and accepted path at every target layer",
        "correction_bonus_persisted_in_current_round": False,
        "decision_stabilization": {
            "mode": args.decision_stabilization,
            "margin_threshold": args.decision_margin_threshold,
            "always_commit_source": "canonical_recheck",
            "shadow_commit_source": "packed_native",
            "canonical_query_length": 16,
            "canonical_mask": "implicit causal matching original DFlash",
        },
        "gold_validation": {
            "comparison_modes": list(args.gold_comparison_modes),
            "comparison_mode_backends": {
                "native": "configured_sdpa_packed_4d_dense_implicit_causal",
                "equal_shape": "configured_sdpa_equal_shape_explicit_4d",
                "same_mask": "configured_sdpa_explicit_4d_both_sides",
                "math": "forced_math_sdpa_explicit_4d_both_sides",
            },
            "layer_localization": args.gold_layer_localization,
            "maximum_rounds": args.gold_max_rounds,
            "record_all_nodes": args.gold_record_all_nodes,
            "thresholds": args.gold_thresholds,
            "configured_attention_implementation_is_not_a_kernel_name": True,
            "actual_cuda_kernel": None,
            "numerical_variation_requires_control_evidence": True,
        },
    }
    manifest["measurement_policy"] = {
        "reference_control": "original DFlash default spec_generate",
        "target_only_control": "target model generate with greedy temperature-zero decoding",
        "formal_tree_latency": "online tree control path",
        "component_timing": "separate online tree profiling path",
        "control_profile_output_ids_must_match": True,
        "reference_tree_output_ids_must_match": args.decision_stabilization
        in {"always", "margin"},
        "top1_chain_output_ids_must_match_in_every_mode": True,
        "native_layerwise_divergence_is_recorded_not_hidden_in_off_or_shadow": True,
        "gold_checks_excluded_from_performance": True,
        "divergence_diagnostics_excluded_from_performance": True,
        "reference_state_diagnostics_excluded_from_performance": True,
        "shadow_oracle_excluded_from_native_formal_timing": True,
        "repeats": args.timing_repeats,
        "warmup_runs": args.warmup_runs,
    }
    manifest["interpretation"] = {
        "online_acceptance_measured": True,
        "speedup_claim_requires_h100_results": True,
        "offline_coverage_is_not_assumed_equal_to_online_acceptance": True,
        "unmatched_offline_online_rounds_are_not_mismatches": True,
        "round_differentials_use_absolute_generated_token_offsets": True,
        "target_only_is_measured_not_assumed_authoritative": True,
        "reference_state_alignment": "absolute_generated_token_prefix",
        "reference_tree_round_indices_assumed_equal": False,
        "reference_state_full_float32_kv_cpu_copy_created": False,
    }
    return manifest


def _tree_options(
    configuration: dict[str, Any],
    *,
    profiling: bool,
    gold_checks: bool,
    args: argparse.Namespace | None = None,
    divergence_diagnostics: bool = False,
    stabilization_override: str | None = None,
    reference_state_prefixes: list[int] | None = None,
) -> dict[str, Any]:
    options = {
        "enable_online_tree": True,
        "online_tree_policy": configuration["tree_policy"],
        "online_tree_node_budget": configuration["node_budget"],
        "online_tree_beam_width": configuration["beam_width"],
        "online_tree_draft_top_k": configuration["draft_top_k"],
        "enable_online_tree_profiling": profiling,
        "enable_online_tree_gold_checks": gold_checks,
        "enable_online_tree_divergence_diagnostics": divergence_diagnostics,
        "decision_stabilization": (
            stabilization_override
            if stabilization_override is not None
            else configuration.get("decision_stabilization", "off")
        ),
        "decision_margin_threshold": configuration.get(
            "decision_margin_threshold", 0.1
        ),
    }
    if gold_checks:
        if args is None:
            raise ValueError("gold tree options require parsed Phase 2C.1 arguments")
        options.update(
            {
                "online_tree_gold_comparison_modes": list(args.gold_comparison_modes),
                "online_tree_gold_layer_localization": args.gold_layer_localization,
                "online_tree_gold_max_rounds": args.gold_max_rounds,
                "online_tree_gold_record_all_nodes": args.gold_record_all_nodes,
                "online_tree_gold_thresholds": args.gold_thresholds,
            }
        )
    elif divergence_diagnostics and args is not None:
        # Reuse the bounded gold-round knob for the otherwise expensive
        # cache/mask/traversal capture. No gold forward is enabled here.
        options["online_tree_gold_max_rounds"] = args.gold_max_rounds
    if reference_state_prefixes is not None:
        options.update(
            {
                "enable_reference_state_diagnostics": True,
                "reference_state_generated_prefixes": reference_state_prefixes,
            }
        )
    return options


def _require_equal(torch_module: Any, left: Any, right: Any, *, label: str) -> None:
    """Legacy assertion helper; Phase 2C.2 output paths persist before raising."""

    if not bool(torch_module.equal(left, right)):
        raise RuntimeError(f"Phase 2C output token mismatch: {label}")


class Phase2CCorrectnessFailure(RuntimeError):
    """Expected non-zero outcome after complete diagnostics were persisted."""


def _tensor_token_ids(tensor: Any, *, prompt_length: int) -> list[int]:
    return [int(value) for value in tensor[0, prompt_length:].detach().cpu().tolist()]


def _target_only_greedy(
    *, target: Any, input_ids: Any, max_new_tokens: int, eos_token_id: int
) -> Any:
    """Generate a third, target-only greedy reference without draft involvement."""

    return target.generate(
        input_ids=input_ids,
        attention_mask=input_ids.new_ones(input_ids.shape),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        use_cache=True,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
    )


def _decoded_token(tokenizer: Any, token_id: int | None) -> str | None:
    return (
        None
        if token_id is None
        else tokenizer.decode([int(token_id)], skip_special_tokens=False)
    )


def evaluate_output_oracle(
    *,
    token_exact_required: bool,
    reference_tree_output_ids_equal: bool,
    tree_control_repeat_output_ids_equal_reference: bool,
    control_profile_output_ids_equal: bool,
    reference_repeat_output_ids_equal: bool,
    reference_profile_output_ids_equal: bool,
    target_only_reference_output_ids_equal: bool,
    target_only_tree_output_ids_equal: bool,
) -> dict[str, Any]:
    """Separate the speculative hard oracle from target-only diagnostics."""

    if not (reference_repeat_output_ids_equal and reference_profile_output_ids_equal):
        hard_failure_class = "reference_control_profile_output_divergence"
    elif not control_profile_output_ids_equal:
        hard_failure_class = "control_profile_output_divergence"
    elif token_exact_required and not reference_tree_output_ids_equal:
        hard_failure_class = "reference_tree_output_divergence"
    elif token_exact_required and not tree_control_repeat_output_ids_equal_reference:
        hard_failure_class = "tree_control_repeat_output_divergence"
    else:
        hard_failure_class = None

    if not target_only_reference_output_ids_equal:
        target_only_status = "target_only_reference_numerical_divergence"
    elif not target_only_tree_output_ids_equal:
        target_only_status = "target_only_tree_numerical_divergence"
    else:
        target_only_status = "equal"

    return {
        "hard_output_relations_passed": hard_failure_class is None,
        "hard_failure_class": hard_failure_class,
        "target_only_auxiliary_status": target_only_status,
        "target_only_auxiliary_warning": (
            None if target_only_status == "equal" else target_only_status
        ),
        "target_only_reference_output_ids_equal": (
            target_only_reference_output_ids_equal
        ),
        "target_only_tree_output_ids_equal": target_only_tree_output_ids_equal,
    }


def _token_ids_sha256(values: list[int]) -> str:
    payload = json.dumps(values, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _eos_position(generated: list[int], eos_token_id: int) -> int | None:
    try:
        return generated.index(int(eos_token_id))
    except ValueError:
        return None


def _reference_round_map(
    diagnostics: dict[str, Any], *, prompt_length: int
) -> dict[int, dict[str, Any]]:
    return {
        int(row["context_length_before"]) - prompt_length: row
        for row in diagnostics["rounds"]
    }


def _top1_round_exact(
    reference: dict[str, Any], online: dict[str, Any], *, prompt_length: int
) -> tuple[bool, list[str]]:
    reference_rounds = reference["rounds"]
    online_rounds = online["rounds"]
    errors = []
    if len(reference_rounds) != len(online_rounds):
        errors.append("round_count")
    for index, (left, right) in enumerate(zip(reference_rounds, online_rounds)):
        checks = {
            "accepted_draft_token_count": left["accepted_draft_token_count"]
            == right["accepted_draft_token_count"],
            "target_token_kind": left["target_token_kind"]
            == right["target_token_kind"],
            "generated_root_offset": int(left["context_length_before"]) - prompt_length
            == right["generated_root_offset"],
            "committed_kv_length": left["context_length_after"]
            == right["context_length_after"],
        }
        errors.extend(
            f"round_{index}.{name}" for name, passed in checks.items() if not passed
        )
    return not errors, errors


def _run_reference_state_differential(
    args: argparse.Namespace,
    *,
    torch_module: Any,
    draft: Any,
    target: Any,
    input_ids: Any,
    tokenizer: Any,
    workload: dict[str, Any],
    configuration: dict[str, Any],
    reference_ids: Any,
    reference_diagnostics: dict[str, Any],
    tree_ids: Any,
    first_output_divergent_offset: int | None,
    gold_logits: list[dict[str, Any]],
) -> dict[str, Any]:
    if not args.enable_reference_state_diagnostics:
        return {"rows": [], "layer_rows": [], "summary": None}

    prompt_length = int(input_ids.shape[1])
    maximum_prefix = (
        int(first_output_divergent_offset)
        if first_output_divergent_offset is not None
        else min(
            int(reference_ids.shape[1]) - prompt_length,
            int(tree_ids.shape[1]) - prompt_length,
        )
    )
    requested_prefixes = list(range(maximum_prefix + 1))
    tree_state_ids, tree_state_diagnostics, _, _ = _call_generation(
        torch_module=torch_module,
        draft=draft,
        target=target,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
        return_diagnostics=True,
        enable_profiling=False,
        generation_options=_tree_options(
            configuration,
            profiling=False,
            gold_checks=False,
            args=args,
            reference_state_prefixes=requested_prefixes,
        ),
    )
    reference_replay_stable = True
    tree_replay_stable = bool(torch_module.equal(tree_state_ids, tree_ids))
    tree_state_rounds = tree_state_diagnostics["rounds"]
    tree_checkpoints = tree_state_diagnostics["reference_state"]["checkpoints"]
    tree_checkpoint_by_prefix = {
        int(row["generated_prefix_length"]): row for row in tree_checkpoints
    }
    for round_row in tree_state_rounds:
        prefix = int(round_row["generated_root_offset"])
        checkpoint = tree_checkpoint_by_prefix.get(prefix)
        if checkpoint is not None:
            stabilization = round_row.get("decision_stabilization") or {}
            checkpoint["shadow_decision"] = {
                "native_canonical_decision_equal": stabilization.get(
                    "native_canonical_decision_equal"
                ),
                "native_selected_path_token_ids": stabilization.get(
                    "native_selected_path_token_ids"
                ),
                "canonical_accepted_path_token_ids": stabilization.get(
                    "canonical_accepted_path_token_ids"
                ),
                "native_next_root_token_id": stabilization.get(
                    "native_next_root_token_id"
                ),
                "canonical_next_root_token_id": stabilization.get(
                    "canonical_next_root_token_id"
                ),
                "minimum_path_margin": stabilization.get("minimum_path_margin"),
            }
    comparison = compare_reference_tree_checkpoints(
        reference_checkpoints=reference_diagnostics["reference_state"][
            "checkpoints"
        ],
        tree_checkpoints=tree_checkpoints,
        first_output_divergent_offset=first_output_divergent_offset,
        packed_verifier_evidence=verifier_evidence_by_prefix(
            gold_records=gold_logits,
            prompt_length=prompt_length,
        ),
    )
    common = {
        "schema_version": SCHEMA_VERSION,
        "task": workload["longbench_task"],
        "sample_id": workload["sample_id"],
        "nominal_prompt_length": workload["nominal_prompt_length"],
        "actual_prompt_length": workload["actual_prompt_length"],
        "configuration_id": configuration["configuration_id"],
    }
    for row in comparison["rows"]:
        row.update(common)
    for row in comparison["layer_rows"]:
        row.update(common)
    comparison["summary"].update(
        {
            **common,
            "status": "complete"
            if reference_replay_stable and tree_replay_stable
            else "inconclusive_replay_instability",
            "reference_diagnostic_output_ids_equal_formal_reference": (
                reference_replay_stable
            ),
            "tree_diagnostic_output_ids_equal_formal_tree": tree_replay_stable,
            "diagnostic_runs_excluded_from_formal_timing": True,
            "reference_execution_provenance": "actual_reference_checkpoint",
            "tree_execution_provenance": "diagnostic_replay",
            "full_float32_kv_cpu_copy_created": False,
            "tensor_comparison_scope": "bounded_stable_samples",
        }
    )
    if not (reference_replay_stable and tree_replay_stable):
        comparison["summary"]["evidence_completeness"] = "partial"
        comparison["summary"]["failure_classification"] = (
            "unresolved_reference_state_divergence"
        )
        comparison["summary"]["affected_component"] = None
    return comparison


def _warmup(
    args: argparse.Namespace,
    *,
    torch_module: Any,
    tokenizer: Any,
    target: Any,
    draft: Any,
    workloads: list[dict[str, Any]],
) -> None:
    if not args.warmup_runs:
        return
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
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=0.0,
            )
            for configuration in tree_configurations(args):
                draft.spec_generate(
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=0.0,
                    **_tree_options(configuration, profiling=False, gold_checks=False),
                )
        torch_module.cuda.synchronize(input_ids.device)
        print(f"Warmup complete: nominal_prompt_length={nominal_length}")


def _run_sample_configuration(
    args: argparse.Namespace,
    *,
    torch_module: Any,
    tokenizer: Any,
    target: Any,
    draft: Any,
    workload: dict[str, Any],
    configuration: dict[str, Any],
    reference_ids: Any,
    reference_diagnostics: dict[str, Any],
    reference_control_repeats: list[dict[str, Any]],
    reference_profile_repeats: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    input_ids = torch_module.tensor(
        [workload["input_ids"]], dtype=torch_module.long, device=args.device
    )
    control_repeats = []
    profile_repeats = []
    round_rows = []
    profile_reference = None
    formal_control_ids = None
    reference_control_output_equal = True
    control_profile_output_equal = True
    reference_repeat_output_equal = all(
        row.get("output_ids_equal_reference", True) for row in reference_control_repeats
    )
    reference_profile_output_equal = all(
        row.get("output_ids_equal_reference", True) for row in reference_profile_repeats
    )
    reference_by_root = _reference_round_map(
        reference_diagnostics, prompt_length=int(input_ids.shape[1])
    )
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
            generation_options=_tree_options(
                configuration,
                profiling=False,
                gold_checks=False,
                stabilization_override=(
                    "off" if args.decision_stabilization == "shadow" else None
                ),
            ),
        )
        if formal_control_ids is None:
            formal_control_ids = control_ids
        reference_control_output_equal = reference_control_output_equal and bool(
            torch_module.equal(reference_ids, control_ids)
        )
        profile_ids, diagnostics, profile_timing, _ = _call_generation(
            torch_module=torch_module,
            draft=draft,
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            return_diagnostics=True,
            enable_profiling=True,
            generation_options=_tree_options(
                configuration, profiling=True, gold_checks=False
            ),
        )
        current_control_profile_equal = bool(
            torch_module.equal(control_ids, profile_ids)
        )
        control_profile_output_equal = (
            control_profile_output_equal and current_control_profile_equal
        )
        if profile_reference is not None:
            control_profile_output_equal = control_profile_output_equal and bool(
                torch_module.equal(profile_reference, profile_ids)
            )
        profile_reference = profile_ids
        control_repeats.append({"repeat": repeat, **control_timing})
        profile_repeats.append(
            {
                "repeat": repeat,
                **profile_timing,
                "components": diagnostics["profiling"]["components"],
                "round_count": len(diagnostics["rounds"]),
                "committed_token_count": sum(
                    int(row["committed_token_count"]) for row in diagnostics["rounds"]
                ),
            }
        )
        for row in diagnostics["rounds"]:
            current = dict(row)
            current.update(
                {
                    "task": workload["longbench_task"],
                    "sample_id": workload["sample_id"],
                    "source_sample_id": workload["source_sample_id"],
                    "nominal_prompt_length": workload["nominal_prompt_length"],
                    "actual_prompt_length": workload["actual_prompt_length"],
                    "configuration_id": configuration["configuration_id"],
                    "profiling_repeat": repeat,
                    "output_ids_equal_reference": bool(
                        torch_module.equal(reference_ids, profile_ids)
                    ),
                    "gold_check_status": "not_run",
                }
            )
            reference_round = reference_by_root.get(current["generated_root_offset"])
            current["reference_round_comparison"] = (
                {
                    "status": "comparable",
                    "reference_accepted_draft_tokens": reference_round[
                        "accepted_draft_token_count"
                    ],
                    "difference": current["accepted_draft_token_count"]
                    - reference_round["accepted_draft_token_count"],
                }
                if reference_round is not None
                else {
                    "status": "not_comparable",
                    "reference_accepted_draft_tokens": None,
                    "difference": None,
                }
            )
            round_rows.append(current)

    # This third control runs only after the formal tree control/profile timing
    # paths, so it cannot warm or delay the measurements reported above.
    target_only_start = torch_module.cuda.Event(enable_timing=True)
    target_only_end = torch_module.cuda.Event(enable_timing=True)
    target_only_start.record()
    target_only_ids = _target_only_greedy(
        target=target,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    target_only_end.record()
    target_only_end.synchronize()
    target_only_cuda_elapsed_ms = float(
        target_only_start.elapsed_time(target_only_end)
    )

    gold_logits = []
    gold_kv = []
    gold_layers = []
    gold_output_equal = None
    gold_status = "not_run"
    gold_failure_class = None
    gold_completed_checks: list[str] = []
    gold_uncompleted_checks: list[str] = []
    gold_uncompleted_check_details: list[dict[str, str]] = []
    divergence_rounds: list[dict[str, Any]] = []
    traversal_checks: list[dict[str, Any]] = []
    diagnostic_tree_ids = None
    if args.enable_gold_checks:
        gold_ids, gold_diagnostics, _, _ = _call_generation(
            torch_module=torch_module,
            draft=draft,
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            return_diagnostics=True,
            enable_profiling=False,
            generation_options=_tree_options(
                configuration,
                profiling=False,
                gold_checks=True,
                args=args,
                divergence_diagnostics=True,
            ),
        )
        diagnostic_tree_ids = gold_ids
        divergence_rounds = gold_diagnostics["rounds"]
        traversal_checks = gold_diagnostics["phase2c"]["tree_traversal_checks"]
        gold_output_equal = bool(torch_module.equal(reference_ids, gold_ids))
        common = {
            "task": workload["longbench_task"],
            "sample_id": workload["sample_id"],
            "nominal_prompt_length": workload["nominal_prompt_length"],
            "configuration_id": configuration["configuration_id"],
        }
        gold_logits = [
            {"schema_version": SCHEMA_VERSION, **common, **record}
            for record in gold_diagnostics["phase2c"]["packed_logits_gold_checks"]
        ]
        gold_kv = [
            {"schema_version": SCHEMA_VERSION, **common, **record}
            for record in gold_diagnostics["phase2c"]["kv_compaction_gold_checks"]
        ]
        gold_layers = [
            {"schema_version": SCHEMA_VERSION, **common, **record}
            for record in gold_diagnostics["phase2c"]["gold_layer_localization"]
        ]
        gold_status = gold_diagnostics["phase2c"]["gold_overall_status"]
        gold_round_results = gold_diagnostics["phase2c"]["gold_round_results"]
        if not gold_output_equal:
            gold_status = "fail_semantic"
            gold_failure_class = "gold_output_ids_mismatch"
        else:
            gold_failure_class = next(
                (
                    row["failure_class"]
                    for row in gold_round_results
                    if row["failure_class"] is not None
                ),
                None,
            )
        gold_completed_checks = sorted(
            {check for row in gold_round_results for check in row["completed_checks"]}
        )
        gold_uncompleted_checks = sorted(
            {check for row in gold_round_results for check in row["uncompleted_checks"]}
        )
        gold_uncompleted_check_details = [
            {
                "check": check,
                "reason": row["kv"].get("reason") or "gold_check_not_completed",
            }
            for row in gold_round_results
            for check in row["uncompleted_checks"]
        ]
        gold_status_by_round = {
            int(row["round_index"]): row["status"] for row in gold_round_results
        }
        for row in round_rows:
            if row["round_index"] in gold_status_by_round:
                row["gold_check_status"] = gold_status_by_round[row["round_index"]]

    output_diagnostic_required = not (
        reference_control_output_equal
        and bool(torch_module.equal(target_only_ids, reference_ids))
        and bool(torch_module.equal(target_only_ids, formal_control_ids))
    )
    if output_diagnostic_required and diagnostic_tree_ids is None:
        diagnostic_tree_ids, diagnostic_diagnostics, _, _ = _call_generation(
            torch_module=torch_module,
            draft=draft,
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            return_diagnostics=True,
            enable_profiling=False,
            generation_options=_tree_options(
                configuration,
                profiling=False,
                gold_checks=False,
                args=args,
                divergence_diagnostics=True,
            ),
        )
        divergence_rounds = diagnostic_diagnostics["rounds"]
        traversal_checks = diagnostic_diagnostics["phase2c"]["tree_traversal_checks"]

    top1_exact = None
    top1_errors: list[str] = []
    if configuration["tree_policy"] == "top1_chain":
        top1_exact, top1_errors = _top1_round_exact(
            reference_diagnostics,
            {"rounds": round_rows[: len(reference_diagnostics["rounds"])]},
            prompt_length=int(input_ids.shape[1]),
        )
        if not top1_exact and args.enable_gold_checks:
            gold_status = "fail_semantic"
            gold_failure_class = "top1_round_semantics_mismatch"
            gold_completed_checks.append("top1_round_equivalence")

    prompt_length = int(input_ids.shape[1])
    reference_generated = _tensor_token_ids(reference_ids, prompt_length=prompt_length)
    control_generated = _tensor_token_ids(
        formal_control_ids, prompt_length=prompt_length
    )
    target_only_generated = _tensor_token_ids(
        target_only_ids, prompt_length=prompt_length
    )
    pairwise = [
        output_pair_comparison(
            left_name="target_only",
            left_ids=target_only_generated,
            right_name="original_dflash_reference",
            right_ids=reference_generated,
            prompt_length=prompt_length,
        ),
        output_pair_comparison(
            left_name="target_only",
            left_ids=target_only_generated,
            right_name="layerwise_tree_control",
            right_ids=control_generated,
            prompt_length=prompt_length,
        ),
        output_pair_comparison(
            left_name="original_dflash_reference",
            left_ids=reference_generated,
            right_name="layerwise_tree_control",
            right_ids=control_generated,
            prompt_length=prompt_length,
        ),
    ]
    reference_tree = pairwise[-1]
    for comparison in pairwise:
        comparison["left_token_text"] = _decoded_token(
            tokenizer, comparison["left_token_id"]
        )
        comparison["right_token_text"] = _decoded_token(
            tokenizer, comparison["right_token_id"]
        )
    reference_state = _run_reference_state_differential(
        args,
        torch_module=torch_module,
        draft=draft,
        target=target,
        input_ids=input_ids,
        tokenizer=tokenizer,
        workload=workload,
        configuration=configuration,
        reference_ids=reference_ids,
        reference_diagnostics=reference_diagnostics,
        tree_ids=formal_control_ids,
        first_output_divergent_offset=reference_tree[
            "first_divergent_generated_token_offset"
        ],
        gold_logits=gold_logits,
    )
    round_differential = build_round_differential(
        comparison=reference_tree,
        prompt_length=prompt_length,
        reference_rounds=reference_diagnostics["rounds"],
        tree_rounds=divergence_rounds,
    )
    divergent_tree_round_index = (
        round_differential["tree"]["round_index"]
        if round_differential and round_differential["tree"]
        else None
    )
    relevant_traversal = next(
        (
            row
            for row in traversal_checks
            if row["round_index"] == divergent_tree_round_index
        ),
        None,
    )
    native_mismatches = sum(
        1
        for row in gold_logits
        if row["round_index"] == divergent_tree_round_index
        and row["comparison_mode"] == "native"
        and row.get("verifier_decision_relevant")
        and row.get("argmax_equal") is False
    )
    math_rows = [
        row
        for row in gold_logits
        if row["round_index"] == divergent_tree_round_index
        and row["comparison_mode"] == "math"
        and row.get("availability") == "available"
    ]
    math_mismatches = sum(
        1
        for row in math_rows
        if row.get("verifier_decision_relevant") and row.get("argmax_equal") is False
    )
    relevant_kv = next(
        (row for row in gold_kv if row["round_index"] == divergent_tree_round_index),
        None,
    )
    divergence_classification = classify_layerwise_divergence(
        output_ids_equal=reference_tree["output_ids_equal"],
        traversal_checks_passed=(
            relevant_traversal["checks"]["traversal_checks_passed"]
            if relevant_traversal
            else None
        ),
        structural_checks_passed=(
            relevant_traversal["checks"]["structural_checks_passed"]
            if relevant_traversal
            else None
        ),
        native_verifier_mismatch_count=(
            native_mismatches if args.enable_gold_checks else None
        ),
        math_verifier_mismatch_count=(math_mismatches if math_rows else None),
        kv_root_transition_checks_passed=(
            relevant_traversal["checks"]["kv_root_transition_checks_passed"]
            if relevant_traversal
            else None
        ),
        kv_next_token_argmax_equal=(
            relevant_kv.get("argmax_equal") if relevant_kv else None
        ),
        divergence_begins_at_round_root=(
            bool(
                round_differential
                and round_differential["tree"]
                and round_differential["tree"]["token_kind"] == "root"
            )
            if round_differential
            else None
        ),
    )
    token_exact_required = (
        args.decision_stabilization in {"always", "margin"}
        or configuration["tree_policy"] == "top1_chain"
    )
    output_oracle = evaluate_output_oracle(
        token_exact_required=token_exact_required,
        reference_tree_output_ids_equal=reference_tree["output_ids_equal"],
        tree_control_repeat_output_ids_equal_reference=(
            reference_control_output_equal
        ),
        control_profile_output_ids_equal=control_profile_output_equal,
        reference_repeat_output_ids_equal=reference_repeat_output_equal,
        reference_profile_output_ids_equal=reference_profile_output_equal,
        target_only_reference_output_ids_equal=pairwise[0]["output_ids_equal"],
        target_only_tree_output_ids_equal=pairwise[1]["output_ids_equal"],
    )
    if not (reference_repeat_output_equal and reference_profile_output_equal):
        divergence_classification = {
            "status": "failed",
            "failure_class": "reference_control_profile_output_divergence",
            "evidence_complete": True,
        }
    elif not control_profile_output_equal:
        divergence_classification = {
            "status": "failed",
            "failure_class": "control_profile_output_divergence",
            "evidence_complete": True,
        }
    elif reference_tree["output_ids_equal"] and not reference_control_output_equal:
        divergence_classification = {
            "status": "failed",
            "failure_class": "tree_control_repeat_output_divergence",
            "evidence_complete": True,
        }
    reference_tree_relations_stable = (
        reference_tree["output_ids_equal"]
        and reference_control_output_equal
        and control_profile_output_equal
        and reference_repeat_output_equal
        and reference_profile_output_equal
    )
    if reference_tree_relations_stable:
        mismatch_message = None
    elif not reference_tree["output_ids_equal"]:
        mismatch_message = (
            "Phase 2C output token mismatch: "
            f"reference/control {configuration['configuration_id']}"
        )
    elif not (reference_repeat_output_equal and reference_profile_output_equal):
        mismatch_message = (
            "Phase 2C original-reference repeat/profile output mismatch: "
            f"{configuration['configuration_id']}"
        )
    elif not reference_control_output_equal:
        mismatch_message = (
            "Phase 2C tree control repeat output mismatch: "
            f"{configuration['configuration_id']}"
        )
    elif not control_profile_output_equal:
        mismatch_message = (
            "Phase 2C tree control/profile output mismatch: "
            f"{configuration['configuration_id']}"
        )
    else:
        mismatch_message = None
    uncompleted_divergence_checks = []
    if not reference_tree["output_ids_equal"] and not args.enable_gold_checks:
        uncompleted_divergence_checks.append(
            {
                "check": "packed_dense_native_math_and_kv_gold",
                "reason": "gold_checks_not_enabled",
            }
        )
    if not reference_tree["output_ids_equal"] and relevant_traversal is None:
        uncompleted_divergence_checks.append(
            {
                "check": "first_divergent_round_tree_traversal",
                "reason": (
                    "first divergent round is outside --gold-max-rounds capture "
                    f"limit {args.gold_max_rounds}"
                ),
            }
        )
    if not pairwise[0]["output_ids_equal"]:
        uncompleted_divergence_checks.append(
            {
                "check": "target_only_precision_attribution",
                "reason": (
                    "target-only is an observed third control without DFlash round "
                    "state and is not assumed authoritative"
                ),
            }
        )
    divergence_record = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if reference_tree_relations_stable else "failed",
        "task": workload["longbench_task"],
        "sample_id": workload["sample_id"],
        "nominal_prompt_length": workload["nominal_prompt_length"],
        "actual_prompt_length": workload["actual_prompt_length"],
        "configuration_id": configuration["configuration_id"],
        "pairwise_comparisons": pairwise,
        "reference_output_ids": reference_generated,
        "tree_control_output_ids": control_generated,
        "target_only_output_ids": target_only_generated,
        "target_only_auxiliary_status": output_oracle[
            "target_only_auxiliary_status"
        ],
        "target_only_auxiliary_warning": output_oracle[
            "target_only_auxiliary_warning"
        ],
        "target_only_reference_first_divergent_generated_token_offset": pairwise[
            0
        ]["first_divergent_generated_token_offset"],
        "target_only_reference_first_divergent_absolute_sequence_position": pairwise[
            0
        ]["first_divergent_absolute_sequence_position"],
        "target_only_token_id": pairwise[0]["left_token_id"],
        "target_only_token_text": pairwise[0]["left_token_text"],
        "target_only_reference_token_id": pairwise[0]["right_token_id"],
        "target_only_reference_token_text": pairwise[0]["right_token_text"],
        "reference_returned_token_count": len(reference_generated),
        "tree_returned_token_count": len(control_generated),
        "longest_common_prefix_length": reference_tree["longest_common_prefix_length"],
        "first_divergent_generated_token_offset": reference_tree[
            "first_divergent_generated_token_offset"
        ],
        "first_divergent_absolute_sequence_position": reference_tree[
            "first_divergent_absolute_sequence_position"
        ],
        "reference_token_id": reference_tree["left_token_id"],
        "reference_token_text": reference_tree["left_token_text"],
        "tree_token_id": reference_tree["right_token_id"],
        "tree_token_text": reference_tree["right_token_text"],
        "divergent_tree_round_index": divergent_tree_round_index,
        "divergent_token_kind": (
            round_differential["tree"]["token_kind"]
            if round_differential and round_differential["tree"]
            else None
        ),
        "divergent_round_context_length_before": (
            relevant_traversal["context_length_before"] if relevant_traversal else None
        ),
        "divergent_round_context_length_after": (
            relevant_traversal["context_length_after"] if relevant_traversal else None
        ),
        "divergent_round_persistent_kv_length_before": (
            relevant_traversal["persistent_kv_length_before"]
            if relevant_traversal
            else None
        ),
        "divergent_round_persistent_kv_length_after": (
            relevant_traversal["persistent_kv_length_after"]
            if relevant_traversal
            else None
        ),
        "diagnostic_output_ids_equal_formal_control": (
            bool(torch_module.equal(diagnostic_tree_ids, formal_control_ids))
            if diagnostic_tree_ids is not None
            else None
        ),
        "tree_control_repeat_output_ids_equal_reference": (
            reference_control_output_equal
        ),
        "reference_repeat_output_ids_equal": reference_repeat_output_equal,
        "reference_profile_output_ids_equal": reference_profile_output_equal,
        "round_differential_status": (
            "available"
            if round_differential is not None
            else "not_applicable_reference_tree_outputs_equal"
            if reference_tree["output_ids_equal"]
            else "unavailable"
        ),
        "tree_traversal_status": (
            "available"
            if traversal_checks
            else "not_applicable_no_tree_verification_round"
        ),
        "first_divergent_tree_traversal_status": (
            "available"
            if relevant_traversal is not None
            else "outside_diagnostic_capture_limit"
            if not reference_tree["output_ids_equal"]
            else "not_applicable_reference_tree_outputs_equal"
        ),
        "diagnostic_capture_round_limit": args.gold_max_rounds,
        "classification": divergence_classification,
        "exception": (
            None
            if mismatch_message is None
            else {
                "type": "Phase2CCorrectnessFailure",
                "message": mismatch_message,
            }
        ),
        "uncompleted_checks": uncompleted_divergence_checks,
    }
    if round_differential is not None:
        round_differential.update(
            {
                "schema_version": SCHEMA_VERSION,
                "task": workload["longbench_task"],
                "sample_id": workload["sample_id"],
                "nominal_prompt_length": workload["nominal_prompt_length"],
                "configuration_id": configuration["configuration_id"],
            }
        )
    for row in traversal_checks:
        row.update(
            {
                "task": workload["longbench_task"],
                "sample_id": workload["sample_id"],
                "nominal_prompt_length": workload["nominal_prompt_length"],
                "configuration_id": configuration["configuration_id"],
                "is_first_divergent_round": row["round_index"]
                == divergent_tree_round_index,
            }
        )
    gold_completed_checks = sorted(
        set(
            gold_completed_checks
            + ["absolute_token_output_comparison", "target_only_three_way_comparison"]
            + (["tree_traversal"] if traversal_checks else [])
        )
    )
    if not reference_tree["output_ids_equal"] and not args.enable_gold_checks:
        gold_uncompleted_checks = sorted(
            set(
                gold_uncompleted_checks
                + ["packed_dense_native_math_gold", "kv_compaction_gold"]
            )
        )
        gold_uncompleted_check_details = [
            {"check": check, "reason": "gold_checks_not_enabled"}
            for check in gold_uncompleted_checks
        ]

    generated = reference_generated
    cuda_values = [record["cuda_elapsed_ms"] for record in control_repeats]
    profile_memory = [
        record
        for record in (
            repeat.get("profile_breakdown", {}).get("memory")
            for repeat in profile_repeats
        )
        if record
    ]
    component_fields = tuple(profile_repeats[0]["components"])
    component_distributions = {
        field: distribution(
            repeat["components"].get(field) for repeat in profile_repeats
        )
        for field in component_fields
    }
    component_per_returned = {
        field: distribution(
            (
                repeat["components"].get(field) / len(generated)
                if repeat["components"].get(field) is not None
                else None
            )
            for repeat in profile_repeats
        )
        for field in component_fields
    }
    component_per_committed = {
        field: distribution(
            (
                repeat["components"].get(field) / repeat["committed_token_count"]
                if repeat["components"].get(field) is not None
                else None
            )
            for repeat in profile_repeats
        )
        for field in component_fields
    }
    sample = {
        "schema_version": SCHEMA_VERSION,
        "task": workload["longbench_task"],
        "dataset_split": workload["dataset_split"],
        "sample_id": workload["sample_id"],
        "source_sample_id": workload["source_sample_id"],
        "source_text_sha256": workload["source_text_sha256"],
        "context_prefix_sha256": workload["context_prefix_sha256"],
        "nominal_prompt_length": workload["nominal_prompt_length"],
        "actual_prompt_length": workload["actual_prompt_length"],
        **configuration,
        "generated_token_count": len(generated),
        "generated_output_token_ids": generated,
        "generated_output_token_ids_sha256": _token_ids_sha256(generated),
        "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
        "eos_position": _eos_position(generated, tokenizer.eos_token_id),
        "max_new_tokens_truncated": len(generated) == args.max_new_tokens,
        "reference": {
            "output_ids_equal": True,
            "decoded_text_equal": True,
            "eos_position_equal": True,
            "returned_token_count_equal": True,
            "max_new_tokens_truncation_equal": True,
            "control_repeats": reference_control_repeats,
            "profiling_repeats": reference_profile_repeats,
            "round_count": len(reference_diagnostics["rounds"]),
            "accepted_draft_token_lengths": reference_diagnostics[
                "accepted_draft_token_lengths"
            ],
        },
        "control": {
            "formal_end_to_end": True,
            "repeats": control_repeats,
            "total_cuda_elapsed_ms": distribution(cuda_values),
            "latency_per_returned_token_ms": statistics.median(cuda_values)
            / len(generated),
            "generated_tokens_per_second": len(generated)
            / (statistics.median(cuda_values) / 1000.0),
            "all_repeat_output_ids_equal": reference_control_output_equal,
        },
        "target_only": {
            "formal_end_to_end": True,
            "cuda_elapsed_ms": target_only_cuda_elapsed_ms,
            "explicit_attention_mask": True,
            "greedy_sampling_warnings_suppressed_without_sampling_change": True,
            "auxiliary_status": output_oracle["target_only_auxiliary_status"],
            "reference_output_ids_equal": pairwise[0]["output_ids_equal"],
            "tree_output_ids_equal": pairwise[1]["output_ids_equal"],
            "reference_first_divergent_generated_token_offset": pairwise[0][
                "first_divergent_generated_token_offset"
            ],
            "reference_first_divergent_absolute_sequence_position": pairwise[0][
                "first_divergent_absolute_sequence_position"
            ],
            "divergent_token_id": pairwise[0]["left_token_id"],
            "divergent_token_text": pairwise[0]["left_token_text"],
            "reference_token_id": pairwise[0]["right_token_id"],
            "reference_token_text": pairwise[0]["right_token_text"],
        },
        "profiling": {
            "formal_end_to_end": False,
            "repeats": profile_repeats,
            "round_count_mean": statistics.fmean(
                record["round_count"] for record in profile_repeats
            ),
            "peak_allocated_bytes": max(
                (record["peak_allocated_bytes"] for record in profile_memory),
                default=0,
            ),
            "peak_reserved_bytes": max(
                (record["peak_reserved_bytes"] for record in profile_memory),
                default=0,
            ),
            "control_profile_output_ids_equal": control_profile_output_equal,
            "component_cuda_elapsed_ms": component_distributions,
            "component_per_returned_token_ms": component_per_returned,
            "component_per_committed_token_ms": component_per_committed,
        },
        "correctness": {
            "token_exact_required": token_exact_required,
            "reference_output_ids_equal": reference_control_output_equal,
            "original_dflash_reference_tree_output_ids_equal": reference_tree[
                "output_ids_equal"
            ],
            "control_profile_output_ids_equal": control_profile_output_equal,
            "target_only_reference_output_ids_equal": pairwise[0]["output_ids_equal"],
            "target_only_tree_output_ids_equal": pairwise[1]["output_ids_equal"],
            "target_only_auxiliary_status": output_oracle[
                "target_only_auxiliary_status"
            ],
            "target_only_auxiliary_warning": output_oracle[
                "target_only_auxiliary_warning"
            ],
            "divergence_classification": divergence_classification,
            "output_divergence_failure_class": divergence_classification[
                "failure_class"
            ],
            "gold_output_ids_equal": gold_output_equal,
            "gold_status": gold_status,
            "gold_failure_class": gold_failure_class,
            "gold_completed_checks": gold_completed_checks,
            "gold_uncompleted_checks": gold_uncompleted_checks,
            "gold_uncompleted_check_details": gold_uncompleted_check_details,
            "top1_round_exact_equal": top1_exact,
            "top1_round_exact_errors": top1_errors,
        },
    }
    correctness = {
        "schema_version": SCHEMA_VERSION,
        "task": sample["task"],
        "sample_id": sample["sample_id"],
        "nominal_prompt_length": sample["nominal_prompt_length"],
        "configuration_id": sample["configuration_id"],
        **sample["correctness"],
        "native_divergence_allowed_for_analysis": args.decision_stabilization
        in {"off", "shadow"}
        and configuration["tree_policy"] != "top1_chain",
        "passed": (
            (top1_exact is not False)
            and output_oracle["hard_output_relations_passed"]
            and (
                not token_exact_required
                or gold_output_equal is not False
            )
            and (
                not args.enable_gold_checks
                or not token_exact_required
                and args.decision_stabilization in {"off", "shadow"}
                and gold_status != "fail_structural"
                or gold_status
                in {
                    "pass_exact",
                    "pass_semantic_with_numerical_variation",
                }
            )
        ),
    }
    return (
        sample,
        round_rows,
        correctness,
        gold_logits,
        gold_kv,
        gold_layers,
        divergence_record,
        [round_differential] if round_differential is not None else [],
        traversal_checks,
        reference_state,
    )


def _load_reference_and_run(
    args: argparse.Namespace,
    *,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    samples: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
    correctness_records: list[dict[str, Any]],
    gold_logits: list[dict[str, Any]],
    gold_kv: list[dict[str, Any]],
    gold_layers: list[dict[str, Any]],
    divergences: list[dict[str, Any]],
    round_differentials: list[dict[str, Any]],
    traversal_checks: list[dict[str, Any]],
    reference_state_rows: list[dict[str, Any]],
    reference_state_layer_rows: list[dict[str, Any]],
    reference_state_cases: list[dict[str, Any]],
    paths: dict[str, Path],
) -> None:
    torch_module, tokenizer, target, draft, workloads = _load_inputs(
        args, manifest, provenance
    )
    if (
        manifest["attention_backend"]["target"]["configured_attention_implementation"]
        != "sdpa"
    ):
        raise RuntimeError("Phase 2C requires target configured for PyTorch SDPA")
    _warmup(
        args,
        torch_module=torch_module,
        tokenizer=tokenizer,
        target=target,
        draft=draft,
        workloads=workloads,
    )
    configurations = tree_configurations(args)
    for index, workload in enumerate(workloads, 1):
        print(
            f"[{index}/{len(workloads)}] task={workload['longbench_task']} "
            f"sample={workload['sample_id']} nominal={workload['nominal_prompt_length']}"
        )
        input_ids = torch_module.tensor(
            [workload["input_ids"]], dtype=torch_module.long, device=args.device
        )
        reference_ids, reference_diagnostics, _, _ = _call_generation(
            torch_module=torch_module,
            draft=draft,
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            return_diagnostics=True,
            enable_profiling=False,
            generation_options=(
                {
                    "enable_reference_state_diagnostics": True,
                    "reference_state_generated_prefixes": list(
                        range(args.max_new_tokens + 1)
                    ),
                }
                if args.enable_reference_state_diagnostics
                else None
            ),
        )
        reference_repeats = []
        reference_profile_repeats = []
        for repeat in range(args.timing_repeats):
            current_ids, _, timing, _ = _call_generation(
                torch_module=torch_module,
                draft=draft,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                return_diagnostics=False,
                enable_profiling=False,
            )
            reference_repeats.append(
                {
                    "repeat": repeat,
                    "output_ids_equal_reference": bool(
                        torch_module.equal(reference_ids, current_ids)
                    ),
                    **timing,
                }
            )
            profile_ids, profile_diagnostics, profile_timing, _ = _call_generation(
                torch_module=torch_module,
                draft=draft,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                return_diagnostics=True,
                enable_profiling=True,
            )
            reference_profile_repeats.append(
                {
                    "repeat": repeat,
                    "output_ids_equal_reference": bool(
                        torch_module.equal(reference_ids, profile_ids)
                    ),
                    "cuda_elapsed_ms": profile_timing["cuda_elapsed_ms"],
                    "verification_cuda_elapsed_ms": sum(
                        float(row["timing"]["verification_cuda_elapsed_ms"] or 0.0)
                        for row in profile_diagnostics["rounds"]
                    ),
                    "round_cuda_elapsed_ms": sum(
                        float(row["timing"]["round_cuda_elapsed_ms"] or 0.0)
                        for row in profile_diagnostics["rounds"]
                    ),
                    "formal_end_to_end": False,
                }
            )
        for configuration in configurations:
            print(f"  configuration={configuration['configuration_id']}")
            result = _run_sample_configuration(
                args,
                torch_module=torch_module,
                tokenizer=tokenizer,
                target=target,
                draft=draft,
                workload=workload,
                configuration=configuration,
                reference_ids=reference_ids,
                reference_diagnostics=reference_diagnostics,
                reference_control_repeats=reference_repeats,
                reference_profile_repeats=reference_profile_repeats,
            )
            (
                sample,
                current_rounds,
                correctness,
                current_logits,
                current_kv,
                current_layers,
                current_divergence,
                current_round_differentials,
                current_traversal_checks,
                current_reference_state,
            ) = result
            samples.append(sample)
            rounds.extend(current_rounds)
            correctness_records.append(correctness)
            gold_logits.extend(current_logits)
            gold_kv.extend(current_kv)
            gold_layers.extend(current_layers)
            divergences.append(current_divergence)
            round_differentials.extend(current_round_differentials)
            traversal_checks.extend(current_traversal_checks)
            reference_state_rows.extend(current_reference_state["rows"])
            reference_state_layer_rows.extend(
                current_reference_state["layer_rows"]
            )
            if current_reference_state["summary"] is not None:
                reference_state_cases.append(current_reference_state["summary"])
            persist_reference_state_artifacts(
                paths=paths,
                rows=reference_state_rows,
                layer_rows=reference_state_layer_rows,
                cases=reference_state_cases,
                enabled=args.enable_reference_state_diagnostics,
            )
            persisted_divergence = persist_divergence_artifacts(
                paths=paths,
                divergences=divergences,
                round_differentials=round_differentials,
                traversal_checks=traversal_checks,
            )
            _validate_schema(
                persisted_divergence, "phase2c-output-divergence.schema.json"
            )
            for row in current_round_differentials:
                _validate_schema(row, "phase2c-round-differential.schema.json")
            for row in current_traversal_checks:
                _validate_schema(row, "phase2c-tree-traversal-check.schema.json")
            if args.enable_gold_checks:
                persist_gold_artifacts(
                    paths=paths,
                    gold_logits=gold_logits,
                    gold_kv=gold_kv,
                    gold_layers=gold_layers,
                )
                write_json(
                    paths["correctness_summary"],
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "running" if correctness["passed"] else "failed",
                        "gold_status": correctness["gold_status"],
                        "failure_class": (
                            correctness["output_divergence_failure_class"]
                            or correctness["gold_failure_class"]
                        ),
                        "all_checks_passed": all(
                            row["passed"] for row in correctness_records
                        ),
                        "completed_checks": correctness["gold_completed_checks"],
                        "uncompleted_checks": correctness["gold_uncompleted_checks"],
                        "uncompleted_check_details": correctness[
                            "gold_uncompleted_check_details"
                        ],
                        "records": correctness_records,
                        "error": None,
                    },
                )
            if not correctness["passed"]:
                raise Phase2CCorrectnessFailure(
                    "Phase 2C.2 correctness validation failed after diagnostic "
                    "artifacts were persisted: "
                    f"gold_status={correctness['gold_status']} "
                    "failure_class="
                    f"{correctness['output_divergence_failure_class'] or correctness['gold_failure_class']}"
                )
    manifest["status"] = "ok"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _offline_comparison(
    args: argparse.Namespace, rounds: list[dict[str, Any]]
) -> dict[str, Any]:
    if args.phase2b1_analysis_dir is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "disabled",
            "exact_matched_round_count": 0,
            "unmatched_online_round_count": 0,
            "unmatched_offline_round_count": 0,
            "matched_round_agreement_rate": None,
            "unmatched_rounds_are_mismatches": False,
            "rows": [],
        }
    analysis_path = (
        args.phase2b1_analysis_dir.expanduser().resolve() / "path-survival-events.jsonl"
    )
    if not analysis_path.is_file():
        raise ValueError(f"missing Phase 2B.1 event artifact: {analysis_path}")
    online = [row for row in rounds if row["profiling_repeat"] == 0]
    return align_offline_online(online, _read_jsonl(analysis_path))


def _preflight_offline_inputs(args: argparse.Namespace) -> None:
    """Validate optional offline evidence before any target/draft model load."""

    if args.phase2b1_analysis_dir is None:
        return
    analysis_path = (
        args.phase2b1_analysis_dir.expanduser().resolve() / "path-survival-events.jsonl"
    )
    if not analysis_path.is_file():
        raise ValueError(f"missing Phase 2B.1 event artifact: {analysis_path}")
    _read_jsonl(analysis_path)
    if args.phase2b_capture_dir:
        from dflash.phase2b1.validation import validate_capture_directories

        validate_capture_directories(
            args.phase2b_capture_dir, schema_dir=REPO_ROOT / "docs" / "schemas"
        )


def _attach_offline_comparison(
    rounds: list[dict[str, Any]], comparison: dict[str, Any]
) -> None:
    if comparison["status"] == "disabled":
        for row in rounds:
            row["offline_comparison_status"] = "disabled"
            row["offline_comparison"] = None
        return
    by_key = {
        (
            row["task"],
            row["sample_id"],
            row["context_length"],
            row["generated_root_offset"],
            row["policy"],
            row["node_budget"],
            row["beam_width"],
        ): row
        for row in comparison["rows"]
    }
    for row in rounds:
        if row["profiling_repeat"] != 0:
            row["offline_comparison_status"] = "not_evaluated_repeat"
            row["offline_comparison"] = None
            continue
        key = (
            row["task"],
            row["sample_id"],
            row["nominal_prompt_length"],
            row["generated_root_offset"],
            row["tree_policy"],
            row["requested_node_count"],
            row["beam_width"],
        )
        matched = by_key.get(key)
        row["offline_comparison_status"] = (
            matched["status"] if matched else "not_comparable"
        )
        row["offline_comparison"] = matched


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as temporary:
        writer = csv.DictWriter(temporary, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def persist_gold_artifacts(
    *,
    paths: dict[str, Path],
    gold_logits: list[dict[str, Any]],
    gold_kv: list[dict[str, Any]],
    gold_layers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Atomically flush complete gold records before a semantic failure is raised."""

    _atomic_write_jsonl(paths["gold_logit_checks"], gold_logits)
    _atomic_write_jsonl(paths["gold_kv_checks"], gold_kv)
    _atomic_write_jsonl(paths["gold_layer_localization"], gold_layers)
    summary = build_gold_comparison_summary(gold_logits, gold_kv)
    write_json(paths["gold_comparison_summary"], summary)
    _write_csv(
        paths["gold_comparison_summary_csv"],
        summary["groups"],
        GOLD_COMPARISON_CSV_FIELDS,
    )
    return summary


def persist_divergence_artifacts(
    *,
    paths: dict[str, Path],
    divergences: list[dict[str, Any]],
    round_differentials: list[dict[str, Any]],
    traversal_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "failed"
            if any(row["status"] == "failed" for row in divergences)
            else "pass"
            if divergences
            else "not_run"
        ),
        "records": divergences,
    }
    write_json(paths["output_divergence"], payload)
    csv_rows = [
        {
            **row,
            "failure_class": row["classification"]["failure_class"],
            "exception_type": (row.get("exception") or {}).get("type"),
            "exception_message": (row.get("exception") or {}).get("message"),
        }
        for row in divergences
    ]
    _write_csv(
        paths["output_divergence_csv"],
        csv_rows,
        OUTPUT_DIVERGENCE_CSV_FIELDS,
    )
    _atomic_write_jsonl(paths["round_differential"], round_differentials)
    _atomic_write_jsonl(paths["tree_traversal_checks"], traversal_checks)
    return payload


def persist_reference_state_artifacts(
    *,
    paths: dict[str, Path],
    rows: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _atomic_write_jsonl(paths["reference_state_differential"], rows)
    _atomic_write_jsonl(paths["reference_state_layer_differential"], layer_rows)
    csv_rows = [
        {
            **row,
            "target_kv_structural_equal": row["target_kv"].get(
                "structural_equal"
            ),
            "target_kv_numerically_equal": row["target_kv"].get(
                "numerically_equal"
            ),
            "target_kv_maximum_sample_abs_error": row["target_kv"].get(
                "maximum_sample_abs_error"
            ),
            "target_hidden_structural_equal": row["target_hidden"].get(
                "structural_equal"
            ),
            "target_hidden_numerically_equal": row["target_hidden"].get(
                "numerically_equal"
            ),
            "target_hidden_max_abs_error": row["target_hidden"].get(
                "max_abs_error"
            ),
            "draft_top1_candidates_equal": row["draft_state"].get(
                "top1_candidates_equal"
            ),
            "draft_topk_candidates_equal": row["draft_state"].get(
                "topk_candidates_equal"
            ),
            "semantic_probe_argmax_equal": row["semantic_probe"].get(
                "argmax_equal"
            ),
            "semantic_probe_top8_ordered_equal": row["semantic_probe"].get(
                "top8_ordered_equal"
            ),
        }
        for row in rows
    ]
    _write_csv(
        paths["reference_state_differential_csv"],
        csv_rows,
        REFERENCE_STATE_CSV_FIELDS,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if enabled and cases else "not_run",
        "case_count": len(cases),
        "cases": cases,
        "absolute_generated_prefix_alignment_used": True,
        "round_index_alignment_assumed": False,
        "formal_timing_contaminated": False,
        "target_only_used_as_authoritative_oracle": False,
    }
    first = {
        "schema_version": SCHEMA_VERSION,
        "status": summary["status"],
        "cases": [
            {
                key: case.get(key)
                for key in (
                    "task",
                    "sample_id",
                    "nominal_prompt_length",
                    "actual_prompt_length",
                    "configuration_id",
                    "first_output_divergent_generated_token_offset",
                    "first_state_divergent_aligned_prefix",
                    "first_semantically_consequential_state_divergence",
                    "affected_component",
                    "reference_tree_round_mapping",
                    "evidence_completeness",
                    "failure_classification",
                )
            }
            for case in cases
        ],
    }
    write_json(paths["reference_state_summary"], summary)
    write_json(paths["first_reference_state_divergence"], first)
    return summary, first


def build_stabilization_artifacts(
    rounds: list[dict[str, Any]], samples: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Flatten dynamic diagnostics and build run-level stabilization aggregates."""

    rows = []
    for round_row in rounds:
        stabilization = round_row.get("decision_stabilization")
        if stabilization is None:
            continue
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "task": round_row["task"],
                "sample_id": round_row["sample_id"],
                "nominal_prompt_length": round_row["nominal_prompt_length"],
                "configuration_id": round_row["configuration_id"],
                "profiling_repeat": round_row["profiling_repeat"],
                "round_index": round_row["round_index"],
                "accepted_draft_token_count": round_row.get(
                    "accepted_draft_token_count"
                ),
                "output_ids_equal_reference": round_row[
                    "output_ids_equal_reference"
                ],
                **stabilization,
            }
        )
    summary = summarize_decision_stabilization(rows, samples)
    frontier = decision_margin_frontier(
        [row for row in rows if row["profiling_repeat"] == 0]
    )
    return rows, summary, frontier


def margin_frontier_csv_rows(frontier: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in frontier:
        estimated = record["estimated_recheck_cuda_elapsed_ms"]
        rows.append(
            {
                **record,
                "estimated_recheck_cuda_mean_ms": estimated["mean"],
                "estimated_recheck_cuda_p50_ms": estimated["median"],
                "estimated_recheck_cuda_p90_ms": estimated["p90"],
            }
        )
    return rows


def _validate_schema(payload: Any, schema_name: str) -> None:
    from jsonschema import Draft202012Validator

    schema = json.loads(
        (REPO_ROOT / "docs" / "schemas" / schema_name).read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(payload)


def _validate_artifacts(
    manifest: dict[str, Any],
    samples: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
    summary: dict[str, Any],
    correctness: dict[str, Any],
    offline: dict[str, Any],
    gold_logits: list[dict[str, Any]],
    gold_kv: list[dict[str, Any]],
    gold_layers: list[dict[str, Any]],
    gold_comparison_summary: dict[str, Any],
    output_divergence: dict[str, Any],
    round_differentials: list[dict[str, Any]],
    traversal_checks: list[dict[str, Any]],
    stabilization_rounds: list[dict[str, Any]],
    stabilization_summary: dict[str, Any],
    reference_state_rows: list[dict[str, Any]],
    reference_state_layer_rows: list[dict[str, Any]],
    reference_state_summary: dict[str, Any],
    first_reference_state_divergence: dict[str, Any],
) -> None:
    _validate_schema(manifest, "phase2c-manifest.schema.json")
    for sample in samples:
        _validate_schema(sample, "phase2c-sample.schema.json")
    for row in rounds:
        _validate_schema(row, "phase2c-round.schema.json")
    _validate_schema(summary, "phase2c-summary.schema.json")
    _validate_schema(correctness, "phase2c-correctness-summary.schema.json")
    _validate_schema(offline, "phase2c-offline-online.schema.json")
    for row in gold_logits:
        _validate_schema(row, "phase2c-gold-logit-check.schema.json")
    for row in gold_kv:
        _validate_schema(row, "phase2c-gold-kv-check.schema.json")
    for row in gold_layers:
        _validate_schema(row, "phase2c-gold-layer-localization.schema.json")
    _validate_schema(
        gold_comparison_summary, "phase2c-gold-comparison-summary.schema.json"
    )
    _validate_schema(output_divergence, "phase2c-output-divergence.schema.json")
    for row in round_differentials:
        _validate_schema(row, "phase2c-round-differential.schema.json")
    for row in traversal_checks:
        _validate_schema(row, "phase2c-tree-traversal-check.schema.json")
    for row in stabilization_rounds:
        _validate_schema(row, "phase2c-decision-stabilization-round.schema.json")
    _validate_schema(
        stabilization_summary, "phase2c-decision-stabilization-summary.schema.json"
    )
    for row in reference_state_rows:
        _validate_schema(row, "phase2c-reference-state-differential.schema.json")
    for row in reference_state_layer_rows:
        _validate_schema(row, "phase2c-reference-state-layer.schema.json")
    _validate_schema(
        reference_state_summary, "phase2c-reference-state-summary.schema.json"
    )
    _validate_schema(
        first_reference_state_divergence,
        "phase2c-first-reference-state-divergence.schema.json",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": output_dir / "manifest.json",
        "workload_inventory": output_dir / "workload-inventory.json",
        "samples": output_dir / "samples.jsonl",
        "tree_rounds": output_dir / "tree-rounds.jsonl",
        "tree_summary": output_dir / "tree-summary.json",
        "tree_summary_csv": output_dir / "tree-summary.csv",
        "correctness_summary": output_dir / "correctness-summary.json",
        "offline_online_comparison": output_dir / "offline-online-comparison.json",
        "offline_online_comparison_csv": output_dir / "offline-online-comparison.csv",
        "environment": output_dir / "environment.json",
        "source_provenance": output_dir / "source-provenance.json",
        "terminal_log": output_dir / "terminal.log",
        "gold_logit_checks": output_dir / "gold-logit-checks.jsonl",
        "gold_kv_checks": output_dir / "gold-kv-checks.jsonl",
        "gold_layer_localization": output_dir / "gold-layer-localization.jsonl",
        "gold_comparison_summary": output_dir / "gold-comparison-summary.json",
        "gold_comparison_summary_csv": output_dir / "gold-comparison-summary.csv",
        "output_divergence": output_dir / "output-divergence.json",
        "output_divergence_csv": output_dir / "output-divergence.csv",
        "round_differential": output_dir / "round-differential.jsonl",
        "tree_traversal_checks": output_dir / "tree-traversal-checks.jsonl",
        "decision_stabilization_rounds": output_dir
        / "decision-stabilization-rounds.jsonl",
        "decision_stabilization_summary": output_dir
        / "decision-stabilization-summary.json",
        "decision_stabilization_summary_csv": output_dir
        / "decision-stabilization-summary.csv",
        "decision_margin_frontier_csv": output_dir / "decision-margin-frontier.csv",
        "canonical_recheck_events": output_dir / "canonical-recheck-events.jsonl",
        "reference_state_differential": output_dir
        / "reference-state-differential.jsonl",
        "reference_state_differential_csv": output_dir
        / "reference-state-differential.csv",
        "reference_state_summary": output_dir / "reference-state-summary.json",
        "first_reference_state_divergence": output_dir
        / "first-reference-state-divergence.json",
        "reference_state_layer_differential": output_dir
        / "reference-state-layer-differential.jsonl",
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
    correctness_records: list[dict[str, Any]] = []
    gold_logits: list[dict[str, Any]] = []
    gold_kv: list[dict[str, Any]] = []
    gold_layers: list[dict[str, Any]] = []
    divergences: list[dict[str, Any]] = []
    round_differentials: list[dict[str, Any]] = []
    traversal_checks: list[dict[str, Any]] = []
    reference_state_rows: list[dict[str, Any]] = []
    reference_state_layer_rows: list[dict[str, Any]] = []
    reference_state_cases: list[dict[str, Any]] = []
    gold_comparison_summary = build_gold_comparison_summary(gold_logits, gold_kv)
    output_divergence = {
        "schema_version": SCHEMA_VERSION,
        "status": "not_run",
        "records": [],
    }
    summary = aggregate_tree_results(samples, rounds)
    stabilization_rounds: list[dict[str, Any]] = []
    stabilization_summary = summarize_decision_stabilization([], [])
    margin_frontier: list[dict[str, Any]] = []
    correctness = {
        "schema_version": SCHEMA_VERSION,
        "status": "started",
        "all_checks_passed": False,
        "gold_status": "not_run",
        "failure_class": None,
        "completed_checks": [],
        "uncompleted_checks": [],
        "uncompleted_check_details": [],
        "records": correctness_records,
        "error": None,
    }
    offline = {
        "schema_version": SCHEMA_VERSION,
        "status": "disabled",
        "exact_matched_round_count": 0,
        "unmatched_online_round_count": 0,
        "unmatched_offline_round_count": 0,
        "matched_round_agreement_rate": None,
        "unmatched_rounds_are_mismatches": False,
        "rows": [],
    }
    write_json(paths["environment"], environment)
    exit_code = 1
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with paths["terminal_log"].open("w", encoding="utf-8", buffering=1) as log:
        sys.stdout = TeeStream(original_stdout, log)
        sys.stderr = TeeStream(original_stderr, log)
        try:
            print(f"Phase 2C output directory: {output_dir}")
            _preflight_offline_inputs(args)
            _load_reference_and_run(
                args,
                manifest=manifest,
                provenance=provenance,
                samples=samples,
                rounds=rounds,
                correctness_records=correctness_records,
                gold_logits=gold_logits,
                gold_kv=gold_kv,
                gold_layers=gold_layers,
                divergences=divergences,
                round_differentials=round_differentials,
                traversal_checks=traversal_checks,
                reference_state_rows=reference_state_rows,
                reference_state_layer_rows=reference_state_layer_rows,
                reference_state_cases=reference_state_cases,
                paths=paths,
            )
            summary = aggregate_tree_results(samples, rounds)
            correctness.update(
                {
                    "status": "ok",
                    "all_checks_passed": all(
                        record["passed"] for record in correctness_records
                    ),
                    "all_required_token_exact_checks_passed": all(
                        not record.get("token_exact_required", True)
                        or record["reference_output_ids_equal"]
                        and record[
                            "original_dflash_reference_tree_output_ids_equal"
                        ]
                        for record in correctness_records
                    ),
                    "observed_native_output_divergence": any(
                        not record[
                            "original_dflash_reference_tree_output_ids_equal"
                        ]
                        for record in correctness_records
                    ),
                    "gold_status": next(
                        (
                            record["gold_status"]
                            for record in reversed(correctness_records)
                            if record["gold_status"] != "not_run"
                        ),
                        "not_run",
                    ),
                    "failure_class": next(
                        (
                            record["output_divergence_failure_class"]
                            or record["gold_failure_class"]
                            for record in reversed(correctness_records)
                            if record["output_divergence_failure_class"] is not None
                            or record["gold_failure_class"] is not None
                        ),
                        None,
                    ),
                    "completed_checks": sorted(
                        {
                            check
                            for record in correctness_records
                            for check in record["gold_completed_checks"]
                        }
                    ),
                    "uncompleted_checks": sorted(
                        {
                            check
                            for record in correctness_records
                            for check in record["gold_uncompleted_checks"]
                        }
                    ),
                    "uncompleted_check_details": [
                        detail
                        for record in correctness_records
                        for detail in record.get("gold_uncompleted_check_details", [])
                    ],
                    "error": None,
                }
            )
            offline = _offline_comparison(args, rounds)
            _attach_offline_comparison(rounds, offline)
            if not correctness["all_checks_passed"]:
                raise RuntimeError("one or more Phase 2C correctness checks failed")
            exit_code = 0
        except KeyboardInterrupt:
            error = {"type": "KeyboardInterrupt", "message": "Interrupted by user"}
            manifest["status"] = provenance["status"] = correctness["status"] = "error"
            manifest["error"] = provenance["error"] = correctness["error"] = error
            exit_code = 130
            traceback.print_exc()
        except Phase2CCorrectnessFailure as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            manifest["status"] = provenance["status"] = correctness["status"] = "failed"
            manifest["error"] = provenance["error"] = correctness["error"] = error
            if correctness_records:
                last = correctness_records[-1]
                correctness["gold_status"] = last["gold_status"]
                correctness["failure_class"] = (
                    last["output_divergence_failure_class"]
                    or last["gold_failure_class"]
                )
                correctness["completed_checks"] = last["gold_completed_checks"]
                correctness["uncompleted_checks"] = last["gold_uncompleted_checks"]
                correctness["uncompleted_check_details"] = last.get(
                    "gold_uncompleted_check_details", []
                )
            print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
        except Exception as exc:  # noqa: BLE001 - preserve partial evidence.
            error = {"type": type(exc).__name__, "message": str(exc)}
            manifest["status"] = provenance["status"] = correctness["status"] = "error"
            manifest["error"] = provenance["error"] = correctness["error"] = error
            if correctness_records:
                last = correctness_records[-1]
                correctness["gold_status"] = last["gold_status"]
                correctness["failure_class"] = last["gold_failure_class"]
                correctness["completed_checks"] = last["gold_completed_checks"]
                correctness["uncompleted_checks"] = last["gold_uncompleted_checks"]
                correctness["uncompleted_check_details"] = last.get(
                    "gold_uncompleted_check_details", []
                )
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
        finally:
            try:
                summary = aggregate_tree_results(samples, rounds)
                inventory = manifest.get("workload", {}).get("inventory") or {
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "error": manifest.get("error"),
                }
                write_json(paths["manifest"], manifest)
                write_json(paths["workload_inventory"], inventory)
                _atomic_write_jsonl(paths["samples"], samples)
                _atomic_write_jsonl(paths["tree_rounds"], rounds)
                write_json(paths["tree_summary"], summary)
                _write_csv(
                    paths["tree_summary_csv"],
                    summary_csv_rows(summary),
                    SUMMARY_CSV_FIELDS,
                )
                write_json(paths["correctness_summary"], correctness)
                write_json(paths["offline_online_comparison"], offline)
                _write_csv(
                    paths["offline_online_comparison_csv"],
                    offline["rows"],
                    OFFLINE_CSV_FIELDS,
                )
                write_json(paths["source_provenance"], provenance)
                gold_comparison_summary = persist_gold_artifacts(
                    paths=paths,
                    gold_logits=gold_logits,
                    gold_kv=gold_kv,
                    gold_layers=gold_layers,
                )
                output_divergence = persist_divergence_artifacts(
                    paths=paths,
                    divergences=divergences,
                    round_differentials=round_differentials,
                    traversal_checks=traversal_checks,
                )
                (
                    stabilization_rounds,
                    stabilization_summary,
                    margin_frontier,
                ) = build_stabilization_artifacts(rounds, samples)
                _atomic_write_jsonl(
                    paths["decision_stabilization_rounds"], stabilization_rounds
                )
                write_json(
                    paths["decision_stabilization_summary"], stabilization_summary
                )
                _write_csv(
                    paths["decision_stabilization_summary_csv"],
                    stabilization_summary_csv_rows(stabilization_summary),
                    STABILIZATION_SUMMARY_CSV_FIELDS,
                )
                _write_csv(
                    paths["decision_margin_frontier_csv"],
                    margin_frontier_csv_rows(margin_frontier),
                    MARGIN_FRONTIER_CSV_FIELDS,
                )
                _atomic_write_jsonl(
                    paths["canonical_recheck_events"],
                    [
                        row
                        for row in stabilization_rounds
                        if row["recheck_triggered"]
                    ],
                )
                (
                    reference_state_summary,
                    first_reference_state_divergence,
                ) = persist_reference_state_artifacts(
                    paths=paths,
                    rows=reference_state_rows,
                    layer_rows=reference_state_layer_rows,
                    cases=reference_state_cases,
                    enabled=args.enable_reference_state_diagnostics,
                )
                if exit_code == 0:
                    _validate_artifacts(
                        manifest,
                        samples,
                        rounds,
                        summary,
                        correctness,
                        offline,
                        gold_logits,
                        gold_kv,
                        gold_layers,
                        gold_comparison_summary,
                        output_divergence,
                        round_differentials,
                        traversal_checks,
                        stabilization_rounds,
                        stabilization_summary,
                        reference_state_rows,
                        reference_state_layer_rows,
                        reference_state_summary,
                        first_reference_state_divergence,
                    )
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                exit_code = 1
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
