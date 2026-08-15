"""Pure aggregation helpers for Phase 2C online-tree artifacts."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def distribution(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    index = max(0, math.ceil(0.9 * len(clean)) - 1)
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "p90": clean[index],
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def aggregate_tree_results(
    samples: list[dict[str, Any]], rounds: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate sample/configuration latency and profiling-only round metrics."""

    sample_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    round_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        key = (
            sample["task"],
            int(sample["nominal_prompt_length"]),
            sample["configuration_id"],
        )
        sample_groups[key].append(sample)
    for row in rounds:
        key = (
            row["task"],
            int(row["nominal_prompt_length"]),
            row["configuration_id"],
        )
        round_groups[key].append(row)

    groups = []
    for key in sorted(sample_groups):
        task, nominal_length, configuration_id = key
        group_samples = sample_groups[key]
        group_rounds = round_groups.get(key, [])
        # Acceptance/accounting is deterministic and should represent one
        # generation per sample, rather than being multiplied by timing repeats.
        # Older synthetic fixtures have no repeat field and remain one-run inputs.
        semantic_rounds = [
            row for row in group_rounds if row.get("profiling_repeat", 0) == 0
        ]
        non_terminal = [row for row in semantic_rounds if not row["terminal_round"]]
        comparable = [
            row
            for row in semantic_rounds
            if row.get("reference_round_comparison", {}).get("status") == "comparable"
        ]
        recovery_eligible = [
            row
            for row in comparable
            if row["reference_round_comparison"]["reference_accepted_draft_tokens"] < 15
        ]
        recovered = [
            row
            for row in recovery_eligible
            if row["accepted_draft_token_count"]
            > row["reference_round_comparison"]["reference_accepted_draft_tokens"]
        ]
        component_fields = (
            "draft_hidden_logits_cuda_elapsed_ms",
            "topk_cuda_elapsed_ms",
            "tree_construction_cuda_elapsed_ms",
            "mask_construction_cuda_elapsed_ms",
            "verification_cuda_elapsed_ms",
            "traversal_cuda_elapsed_ms",
            "kv_compaction_cuda_elapsed_ms",
            "total_draft_cuda_elapsed_ms",
            "total_verification_cuda_elapsed_ms",
            "round_cuda_elapsed_ms",
        )
        component_sums = {
            field: sum(float(row["timing"].get(field) or 0.0) for row in group_rounds)
            for field in component_fields
        }
        round_total = component_sums["round_cuda_elapsed_ms"]
        reference_round_counts = [
            float(sample["reference"]["round_count"]) for sample in group_samples
        ]
        online_round_counts = [
            float(sample["profiling"]["round_count_mean"]) for sample in group_samples
        ]
        reference_latencies = [
            repeat["cuda_elapsed_ms"]
            for sample in group_samples
            for repeat in sample["reference"]["control_repeats"]
        ]
        online_latencies = [
            repeat["cuda_elapsed_ms"]
            for sample in group_samples
            for repeat in sample["control"]["repeats"]
        ]
        reference_verification_latencies = [
            repeat["verification_cuda_elapsed_ms"]
            for sample in group_samples
            for repeat in sample["reference"]["profiling_repeats"]
        ]
        online_verification_latencies = [
            repeat["components"]["verification_cuda_elapsed_ms"]
            for sample in group_samples
            for repeat in sample["profiling"]["repeats"]
        ]
        reference_latency_mean = statistics.fmean(reference_latencies)
        online_latency_mean = statistics.fmean(online_latencies)
        baseline_accepted = [
            float(value)
            for sample in group_samples
            for value in sample["reference"]["accepted_draft_token_lengths"]
        ]
        online_accepted = [
            float(row["accepted_draft_token_count"]) for row in semantic_rounds
        ]
        groups.append(
            {
                "task": task,
                "nominal_prompt_length": nominal_length,
                "configuration_id": configuration_id,
                "tree_policy": group_samples[0]["tree_policy"],
                "requested_node_budget": group_samples[0].get(
                    "requested_node_budget", group_samples[0]["node_budget"]
                ),
                "beam_width": group_samples[0]["beam_width"],
                "sample_count": len(group_samples),
                "round_count": len(semantic_rounds),
                "profiling_round_record_count": len(group_rounds),
                "non_terminal_round_count": len(non_terminal),
                "returned_token_count": sum(
                    int(sample["generated_token_count"]) for sample in group_samples
                ),
                "total_committed_token_count": sum(
                    int(row["committed_token_count"]) for row in semantic_rounds
                ),
                "mean_accepted_draft_tokens": (
                    statistics.fmean(online_accepted) if online_accepted else None
                ),
                "mean_non_terminal_committed_tokens": (
                    statistics.fmean(
                        float(row["committed_token_count"]) for row in non_terminal
                    )
                    if non_terminal
                    else None
                ),
                "mean_verified_nodes_per_round": (
                    statistics.fmean(
                        float(row["actual_node_count"]) for row in semantic_rounds
                    )
                    if semantic_rounds
                    else None
                ),
                "mean_additional_verified_nodes_vs_dflash": (
                    statistics.fmean(
                        float(row["actual_node_count"] - 16) for row in semantic_rounds
                    )
                    if semantic_rounds
                    else None
                ),
                "first_rejection_recovery_fraction": _ratio(
                    len(recovered), len(recovery_eligible)
                ),
                "first_rejection_recovery_eligible_round_count": len(recovery_eligible),
                "generation_cuda_latency_ms": distribution(online_latencies),
                "repeat_cuda_latency_coefficient_of_variation": (
                    statistics.pstdev(online_latencies)
                    / statistics.fmean(online_latencies)
                    if len(online_latencies) > 1 and statistics.fmean(online_latencies)
                    else 0.0
                ),
                "latency_per_returned_token_ms": distribution(
                    sample["control"]["latency_per_returned_token_ms"]
                    for sample in group_samples
                ),
                "generated_tokens_per_second": distribution(
                    sample["control"]["generated_tokens_per_second"]
                    for sample in group_samples
                ),
                "profile_round_component_cuda_elapsed_ms": {
                    field: distribution(
                        row["timing"].get(field) for row in group_rounds
                    )
                    for field in component_fields
                },
                "profile_generation_component_cuda_elapsed_ms": {
                    field: distribution(
                        repeat["components"].get(field)
                        for sample in group_samples
                        for repeat in sample["profiling"]["repeats"]
                    )
                    for field in component_fields
                },
                "component_proportions_of_profile_round_time": {
                    "tree_construction": _ratio(
                        component_sums["tree_construction_cuda_elapsed_ms"],
                        round_total,
                    ),
                    "mask_construction": _ratio(
                        component_sums["mask_construction_cuda_elapsed_ms"],
                        round_total,
                    ),
                    "verification": _ratio(
                        component_sums["verification_cuda_elapsed_ms"], round_total
                    ),
                    "kv_compaction": _ratio(
                        component_sums["kv_compaction_cuda_elapsed_ms"], round_total
                    ),
                },
                "peak_allocated_bytes": max(
                    (
                        sample["profiling"]["peak_allocated_bytes"]
                        for sample in group_samples
                    ),
                    default=None,
                ),
                "peak_reserved_bytes": max(
                    (
                        sample["profiling"]["peak_reserved_bytes"]
                        for sample in group_samples
                    ),
                    default=None,
                ),
                "relative_acceptance_improvement": (
                    _ratio(
                        statistics.fmean(online_accepted),
                        statistics.fmean(baseline_accepted),
                    )
                    - 1.0
                    if online_accepted
                    and baseline_accepted
                    and statistics.fmean(baseline_accepted)
                    else None
                ),
                "relative_round_count_reduction": (
                    1.0
                    - _ratio(
                        statistics.fmean(online_round_counts),
                        statistics.fmean(reference_round_counts),
                    )
                    if reference_round_counts
                    and statistics.fmean(reference_round_counts)
                    else None
                ),
                "relative_end_to_end_speedup": _ratio(
                    reference_latency_mean, online_latency_mean
                ),
                "reference_verification_cuda_latency_ms": distribution(
                    reference_verification_latencies
                ),
                "online_verification_cuda_latency_ms": distribution(
                    online_verification_latencies
                ),
                "relative_verification_latency_increase": (
                    _ratio(
                        statistics.fmean(online_verification_latencies),
                        statistics.fmean(reference_verification_latencies),
                    )
                    - 1.0
                    if reference_verification_latencies
                    and online_verification_latencies
                    and statistics.fmean(reference_verification_latencies)
                    else None
                ),
                "formal_latency_source": "control path",
                "component_timing_source": "profiling path",
            }
        )
    return {
        "schema_version": "1.0",
        "sample_record_count": len(samples),
        "round_record_count": len(rounds),
        "groups": groups,
        "interpretation": {
            "formal_latency_uses_control_path": True,
            "component_timing_uses_profiling_path": True,
            "terminal_rounds_included_in_all_round_means": True,
            "non_terminal_committed_metric_excludes_terminal_rounds": True,
            "acceptance_and_accounting_use_profiling_repeat_zero": True,
            "component_timing_distributions_include_all_profiling_repeats": True,
        },
    }


def summary_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for group in summary["groups"]:
        rows.append(
            {
                "task": group["task"],
                "nominal_prompt_length": group["nominal_prompt_length"],
                "configuration_id": group["configuration_id"],
                "tree_policy": group["tree_policy"],
                "requested_node_budget": group["requested_node_budget"],
                "beam_width": group["beam_width"],
                "sample_count": group["sample_count"],
                "round_count": group["round_count"],
                "non_terminal_round_count": group["non_terminal_round_count"],
                "returned_token_count": group["returned_token_count"],
                "total_committed_token_count": group["total_committed_token_count"],
                "mean_accepted_draft_tokens": group["mean_accepted_draft_tokens"],
                "mean_non_terminal_committed_tokens": group[
                    "mean_non_terminal_committed_tokens"
                ],
                "mean_verified_nodes_per_round": group["mean_verified_nodes_per_round"],
                "mean_additional_verified_nodes_vs_dflash": group[
                    "mean_additional_verified_nodes_vs_dflash"
                ],
                "first_rejection_recovery_fraction": group[
                    "first_rejection_recovery_fraction"
                ],
                "generation_cuda_latency_median_ms": group[
                    "generation_cuda_latency_ms"
                ]["median"],
                "generation_cuda_latency_mean_ms": group["generation_cuda_latency_ms"][
                    "mean"
                ],
                "generation_cuda_latency_p90_ms": group["generation_cuda_latency_ms"][
                    "p90"
                ],
                "generated_tokens_per_second_median": group[
                    "generated_tokens_per_second"
                ]["median"],
                "relative_acceptance_improvement": group[
                    "relative_acceptance_improvement"
                ],
                "relative_round_count_reduction": group[
                    "relative_round_count_reduction"
                ],
                "relative_end_to_end_speedup": group["relative_end_to_end_speedup"],
                "relative_verification_latency_increase": group[
                    "relative_verification_latency_increase"
                ],
            }
        )
    return rows
