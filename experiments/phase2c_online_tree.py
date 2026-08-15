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
    "structural_status",
    "semantic_status",
    "numerical_status",
    "warning",
    "reason",
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
    tree_configurations(args)
    return args


def run_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **phase1b_configuration(args),
        "phase": "2C",
        "mode": "online_bounded_marginal_tree",
        "sample_id": args.sample_id,
        "tree_configurations": tree_configurations(args),
        "draft_top_k": args.draft_top_k,
        "enable_gold_checks": args.enable_gold_checks,
        "gold_comparison_modes": list(args.gold_comparison_modes),
        "gold_layer_localization": args.gold_layer_localization,
        "gold_max_rounds": args.gold_max_rounds,
        "gold_record_all_nodes": args.gold_record_all_nodes,
        "gold_thresholds": args.gold_thresholds,
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
        "formal_tree_latency": "online tree control path",
        "component_timing": "separate online tree profiling path",
        "control_profile_output_ids_must_match": True,
        "reference_tree_output_ids_must_match": True,
        "gold_checks_excluded_from_performance": True,
        "repeats": args.timing_repeats,
        "warmup_runs": args.warmup_runs,
    }
    manifest["interpretation"] = {
        "online_acceptance_measured": True,
        "speedup_claim_requires_h100_results": True,
        "offline_coverage_is_not_assumed_equal_to_online_acceptance": True,
        "unmatched_offline_online_rounds_are_not_mismatches": True,
    }
    return manifest


def _tree_options(
    configuration: dict[str, Any],
    *,
    profiling: bool,
    gold_checks: bool,
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    options = {
        "enable_online_tree": True,
        "online_tree_policy": configuration["tree_policy"],
        "online_tree_node_budget": configuration["node_budget"],
        "online_tree_beam_width": configuration["beam_width"],
        "online_tree_draft_top_k": configuration["draft_top_k"],
        "enable_online_tree_profiling": profiling,
        "enable_online_tree_gold_checks": gold_checks,
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
    return options


def _require_equal(torch_module: Any, left: Any, right: Any, *, label: str) -> None:
    if not bool(torch_module.equal(left, right)):
        raise RuntimeError(f"Phase 2C output token mismatch: {label}")


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
]:
    input_ids = torch_module.tensor(
        [workload["input_ids"]], dtype=torch_module.long, device=args.device
    )
    control_repeats = []
    profile_repeats = []
    round_rows = []
    profile_reference = None
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
                configuration, profiling=False, gold_checks=False
            ),
        )
        _require_equal(
            torch_module,
            reference_ids,
            control_ids,
            label=f"reference/control {configuration['configuration_id']}",
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
        _require_equal(
            torch_module,
            control_ids,
            profile_ids,
            label=f"control/profile {configuration['configuration_id']}",
        )
        if profile_reference is not None:
            _require_equal(
                torch_module,
                profile_reference,
                profile_ids,
                label=f"profile repeat {configuration['configuration_id']}",
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
                    "output_ids_equal_reference": True,
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

    gold_logits = []
    gold_kv = []
    gold_layers = []
    gold_output_equal = None
    gold_status = "not_run"
    gold_failure_class = None
    gold_completed_checks: list[str] = []
    gold_uncompleted_checks: list[str] = []
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
                configuration, profiling=False, gold_checks=True, args=args
            ),
        )
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
        gold_status_by_round = {
            int(row["round_index"]): row["status"] for row in gold_round_results
        }
        for row in round_rows:
            if row["round_index"] in gold_status_by_round:
                row["gold_check_status"] = gold_status_by_round[row["round_index"]]

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

    generated = [
        int(value)
        for value in reference_ids[0, int(input_ids.shape[1]) :].detach().cpu().tolist()
    ]
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
            (repeat["components"].get(field) or 0.0) / len(generated)
            for repeat in profile_repeats
        )
        for field in component_fields
    }
    component_per_committed = {
        field: distribution(
            (repeat["components"].get(field) or 0.0) / repeat["committed_token_count"]
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
            "all_repeat_output_ids_equal": True,
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
            "control_profile_output_ids_equal": True,
            "component_cuda_elapsed_ms": component_distributions,
            "component_per_returned_token_ms": component_per_returned,
            "component_per_committed_token_ms": component_per_committed,
        },
        "correctness": {
            "reference_output_ids_equal": True,
            "control_profile_output_ids_equal": True,
            "gold_output_ids_equal": gold_output_equal,
            "gold_status": gold_status,
            "gold_failure_class": gold_failure_class,
            "gold_completed_checks": gold_completed_checks,
            "gold_uncompleted_checks": gold_uncompleted_checks,
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
        "passed": (
            (top1_exact is not False)
            and (gold_output_equal is not False)
            and (
                not args.enable_gold_checks
                or gold_status
                in {
                    "pass_exact",
                    "pass_semantic_with_numerical_variation",
                }
            )
        ),
    }
    return sample, round_rows, correctness, gold_logits, gold_kv, gold_layers


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
            _require_equal(
                torch_module, reference_ids, current_ids, label="reference repeat"
            )
            reference_repeats.append({"repeat": repeat, **timing})
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
            _require_equal(
                torch_module,
                reference_ids,
                profile_ids,
                label="reference control/profile",
            )
            reference_profile_repeats.append(
                {
                    "repeat": repeat,
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
            ) = result
            samples.append(sample)
            rounds.extend(current_rounds)
            correctness_records.append(correctness)
            gold_logits.extend(current_logits)
            gold_kv.extend(current_kv)
            gold_layers.extend(current_layers)
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
                        "status": "running" if correctness["passed"] else "error",
                        "gold_status": correctness["gold_status"],
                        "failure_class": correctness["gold_failure_class"],
                        "all_checks_passed": all(
                            row["passed"] for row in correctness_records
                        ),
                        "completed_checks": correctness["gold_completed_checks"],
                        "uncompleted_checks": correctness["gold_uncompleted_checks"],
                        "records": correctness_records,
                        "error": None,
                    },
                )
            if not correctness["passed"]:
                raise RuntimeError(
                    "Phase 2C.1 gold validation failed after diagnostic artifacts "
                    f"were persisted: status={correctness['gold_status']} "
                    f"failure_class={correctness['gold_failure_class']}"
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
    gold_comparison_summary = build_gold_comparison_summary(gold_logits, gold_kv)
    summary = aggregate_tree_results(samples, rounds)
    correctness = {
        "schema_version": SCHEMA_VERSION,
        "status": "started",
        "all_checks_passed": False,
        "gold_status": "not_run",
        "failure_class": None,
        "completed_checks": [],
        "uncompleted_checks": [],
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
                paths=paths,
            )
            summary = aggregate_tree_results(samples, rounds)
            correctness.update(
                {
                    "status": "ok",
                    "all_checks_passed": all(
                        record["passed"] for record in correctness_records
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
                            record["gold_failure_class"]
                            for record in reversed(correctness_records)
                            if record["gold_failure_class"] is not None
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
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
        finally:
            try:
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
                    )
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                exit_code = 1
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
