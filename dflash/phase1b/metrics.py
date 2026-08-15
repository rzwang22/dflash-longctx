"""Pure-Python Phase 1B sample and sweep aggregation."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any


def _finite(values: Iterable[float | int | None]) -> list[float]:
    return [
        float(value) for value in values if value is not None and math.isfinite(value)
    ]


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: Iterable[float | int | None]) -> dict[str, Any]:
    finite = _finite(values)
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "min": None,
            "max": None,
            "standard_deviation": None,
            "coefficient_of_variation": None,
        }
    mean = statistics.fmean(finite)
    deviation = statistics.pstdev(finite) if len(finite) > 1 else 0.0
    return {
        "count": len(finite),
        "mean": mean,
        "median": float(statistics.median(finite)),
        "p90": _percentile(finite, 0.90),
        "min": min(finite),
        "max": max(finite),
        "standard_deviation": deviation,
        "coefficient_of_variation": deviation / mean if mean else None,
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def summarize_profile_rounds(
    rounds: Sequence[dict[str, Any]], *, generated_token_count: int
) -> dict[str, Any]:
    """Summarize one profiling run with explicit terminal-round policy."""

    eligible = [
        row
        for row in rounds
        if row["stop_reason"] == "continue" and row["truncated_token_count"] == 0
    ]
    proposed = sum(row["proposed_draft_token_count"] for row in rounds)
    accepted = sum(row["accepted_draft_token_count"] for row in rounds)
    rejected = sum(row["rejected_draft_token_count"] for row in rounds)
    committed = sum(row["committed_token_count"] for row in rounds)
    corrections = sum(row["correction_token_count"] for row in rounds)
    full_blocks = sum(
        row["accepted_draft_token_count"] == row["proposed_draft_token_count"]
        for row in rounds
    )
    eligible_proposed = sum(row["proposed_draft_token_count"] for row in eligible)
    eligible_accepted = sum(row["accepted_draft_token_count"] for row in eligible)
    eligible_corrections = sum(row["correction_token_count"] for row in eligible)
    eligible_full_blocks = sum(
        row["accepted_draft_token_count"] == row["proposed_draft_token_count"]
        for row in eligible
    )
    attention_calls = [call for row in rounds for call in row["attention_calls"]]
    return {
        "round_count": len(rounds),
        "terminal_round_count": sum(row["stop_reason"] != "continue" for row in rounds),
        "truncated_round_count": sum(
            row["truncated_token_count"] > 0 for row in rounds
        ),
        "acceptance_eligible_round_count": len(eligible),
        "acceptance_eligible_proposed_draft_token_count": eligible_proposed,
        "acceptance_eligible_accepted_draft_token_count": eligible_accepted,
        "acceptance_eligible_correction_token_count": eligible_corrections,
        "acceptance_eligible_full_block_acceptance_count": eligible_full_blocks,
        "proposed_draft_token_count": proposed,
        "accepted_draft_token_count": accepted,
        "rejected_draft_token_count": rejected,
        "committed_token_count": committed,
        "correction_token_count": corrections,
        "full_block_acceptance_count": full_blocks,
        "returned_token_count": generated_token_count,
        "truncated_token_count": sum(row["truncated_token_count"] for row in rounds),
        "raw_all_round_accepted_length": [
            row["accepted_draft_token_count"] for row in rounds
        ],
        "non_terminal_accepted_length": [
            row["accepted_draft_token_count"] for row in eligible
        ],
        "raw_all_round_acceptance_rate": _ratio(accepted, proposed),
        "non_terminal_acceptance_rate": _ratio(eligible_accepted, eligible_proposed),
        "effective_returned_tokens_per_round": _ratio(
            generated_token_count, len(rounds)
        ),
        "correction_rate_all_rounds": _ratio(corrections, len(rounds)),
        "correction_rate_non_terminal": _ratio(eligible_corrections, len(eligible)),
        "full_block_acceptance_rate_all_rounds": _ratio(
            full_blocks,
            len(rounds),
        ),
        "full_block_acceptance_rate_non_terminal": _ratio(
            eligible_full_blocks, len(eligible)
        ),
        "maximum_effective_kv_length": max(
            (call["effective_kv_length"] for call in attention_calls), default=None
        ),
        "maximum_theoretical_qk_score_elements": max(
            (call["theoretical_qk_score_elements"] for call in attention_calls),
            default=None,
        ),
        "acceptance_policy": {
            "primary_scope": "non_terminal_non_truncated_rounds",
            "terminal_rounds_excluded": True,
            "truncated_rounds_excluded": True,
            "raw_all_round_metrics_also_reported": True,
        },
    }


def _sample_metric(sample: dict[str, Any], path: str) -> Any:
    value: Any = sample
    for component in path.split("."):
        value = value.get(component) if isinstance(value, dict) else None
    return value


def _micro(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    accountings = [sample["accounting"] for sample in samples]
    proposed = sum(item["proposed_draft_token_count"] for item in accountings)
    accepted = sum(item["accepted_draft_token_count"] for item in accountings)
    rejected = sum(item["rejected_draft_token_count"] for item in accountings)
    committed = sum(item["committed_token_count"] for item in accountings)
    corrections = sum(item["correction_token_count"] for item in accountings)
    full_blocks = sum(item["full_block_acceptance_count"] for item in accountings)
    eligible_proposed = sum(
        item["acceptance_eligible_proposed_draft_token_count"] for item in accountings
    )
    eligible_accepted = sum(
        item["acceptance_eligible_accepted_draft_token_count"] for item in accountings
    )
    eligible_corrections = sum(
        item["acceptance_eligible_correction_token_count"] for item in accountings
    )
    eligible_full_blocks = sum(
        item["acceptance_eligible_full_block_acceptance_count"] for item in accountings
    )
    returned = sum(item["returned_token_count"] for item in accountings)
    truncated = sum(item["truncated_token_count"] for item in accountings)
    rounds = sum(item["round_count"] for item in accountings)
    eligible_rounds = sum(
        item["acceptance_eligible_round_count"] for item in accountings
    )
    return {
        "raw_all_round_accepted_over_proposed": _ratio(accepted, proposed),
        "non_terminal_accepted_over_proposed": _ratio(
            eligible_accepted, eligible_proposed
        ),
        "effective_returned_tokens_per_round": _ratio(returned, rounds),
        "correction_rate_all_rounds": _ratio(corrections, rounds),
        "full_block_acceptance_rate_all_rounds": _ratio(full_blocks, rounds),
        "correction_rate_non_terminal": _ratio(eligible_corrections, eligible_rounds),
        "full_block_acceptance_rate_non_terminal": _ratio(
            eligible_full_blocks, eligible_rounds
        ),
        "accepted_draft_token_count": accepted,
        "proposed_draft_token_count": proposed,
        "rejected_draft_token_count": rejected,
        "committed_token_count": committed,
        "correction_token_count": corrections,
        "full_block_acceptance_count": full_blocks,
        "returned_token_count": returned,
        "truncated_token_count": truncated,
        "round_count": rounds,
        "acceptance_eligible_round_count": eligible_rounds,
    }


def aggregate_sweep(
    samples: Sequence[dict[str, Any]], rounds: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate one row per sample/length plus profiling round rows."""

    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    round_by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_length[int(sample["nominal_prompt_length"])].append(sample)
    for row in rounds:
        round_by_length[int(row["nominal_prompt_length"])].append(row)

    metric_paths = (
        "actual_prompt_length",
        "generated_token_count",
        "control_timing.total_cuda_elapsed_ms.median",
        "control_timing.generated_tokens_per_second.median",
        "profiling_timing.prefill_cuda_elapsed_ms.median",
        "profiling_timing.decode_cuda_elapsed_ms.median",
        "profiling_timing.total_cuda_elapsed_ms.median",
        "profiling_timing.draft_cuda_elapsed_ms.median",
        "profiling_timing.combined_attention_cuda_elapsed_ms.median",
        "profiling_timing.verification_cuda_elapsed_ms.median",
        "profiling_timing.combined_attention_over_draft_ratio.median",
        "profiling_timing.combined_attention_over_round_ratio.median",
        "profiling_timing.draft_over_round_ratio.median",
        "profiling_timing.verification_over_round_ratio.median",
        "memory.peak_allocated_bytes",
        "memory.peak_reserved_bytes",
        "accounting.non_terminal_acceptance_rate",
        "accounting.effective_returned_tokens_per_round",
        "accounting.correction_rate_non_terminal",
        "accounting.full_block_acceptance_rate_non_terminal",
    )
    result = {}
    for nominal_length in sorted(by_length):
        group = by_length[nominal_length]
        round_group = round_by_length[nominal_length]
        sample_level = {
            path: distribution(_sample_metric(sample, path) for sample in group)
            for path in metric_paths
        }
        round_level = {
            "draft_cuda_elapsed_ms": distribution(
                row["timing"]["draft_cuda_elapsed_ms"] for row in round_group
            ),
            "combined_attention_cuda_elapsed_ms": distribution(
                row["timing"]["combined_attention_cuda_elapsed_ms"]
                for row in round_group
            ),
            "verification_cuda_elapsed_ms": distribution(
                row["timing"]["verification_cuda_elapsed_ms"] for row in round_group
            ),
            "round_cuda_elapsed_ms": distribution(
                row["timing"]["round_cuda_elapsed_ms"] for row in round_group
            ),
            "combined_attention_over_draft_ratio": distribution(
                row["timing"]["combined_attention_over_draft_ratio"]
                for row in round_group
            ),
            "combined_attention_over_round_ratio": distribution(
                row["timing"]["combined_attention_over_round_ratio"]
                for row in round_group
            ),
            "draft_over_round_ratio": distribution(
                row["timing"]["draft_cuda_elapsed_ms"]
                / row["timing"]["round_cuda_elapsed_ms"]
                if row["timing"]["round_cuda_elapsed_ms"]
                else None
                for row in round_group
            ),
            "verification_over_round_ratio": distribution(
                row["timing"]["verification_cuda_elapsed_ms"]
                / row["timing"]["round_cuda_elapsed_ms"]
                if row["timing"]["round_cuda_elapsed_ms"]
                else None
                for row in round_group
            ),
            "raw_all_round_accepted_length": distribution(
                row["accepted_draft_token_count"] for row in round_group
            ),
            "non_terminal_accepted_length": distribution(
                row["accepted_draft_token_count"]
                for row in round_group
                if row["stop_reason"] == "continue"
                and row["truncated_token_count"] == 0
            ),
            "proposed_draft_token_count": distribution(
                row["proposed_draft_token_count"] for row in round_group
            ),
            "rejected_draft_token_count": distribution(
                row["rejected_draft_token_count"] for row in round_group
            ),
            "committed_token_count": distribution(
                row["committed_token_count"] for row in round_group
            ),
            "non_terminal_committed_token_count": distribution(
                row["committed_token_count"]
                for row in round_group
                if row["stop_reason"] == "continue"
                and row["truncated_token_count"] == 0
            ),
            "returned_token_delta": distribution(
                row["returned_token_delta"] for row in round_group
            ),
            "truncated_token_count": distribution(
                row["truncated_token_count"] for row in round_group
            ),
            "effective_kv_length": distribution(
                call["effective_kv_length"]
                for row in round_group
                for call in row["attention_calls"]
            ),
        }
        macro_paths = (
            "raw_all_round_acceptance_rate",
            "non_terminal_acceptance_rate",
            "effective_returned_tokens_per_round",
            "correction_rate_non_terminal",
            "full_block_acceptance_rate_non_terminal",
        )
        result[str(nominal_length)] = {
            "sample_count": len(group),
            "profiling_round_record_count": len(round_group),
            "sample_level": sample_level,
            "round_level": round_level,
            "micro_average": _micro(group),
            "sample_level_macro_average": {
                path: distribution(sample["accounting"].get(path) for sample in group)[
                    "mean"
                ]
                for path in macro_paths
            },
            "repeat_variability": {
                "control_total_cuda_coefficient_of_variation": distribution(
                    sample["control_timing"]["total_cuda_elapsed_ms"].get(
                        "coefficient_of_variation"
                    )
                    for sample in group
                ),
                "profiling_total_cuda_coefficient_of_variation": distribution(
                    sample["profiling_timing"]["total_cuda_elapsed_ms"].get(
                        "coefficient_of_variation"
                    )
                    for sample in group
                ),
            },
        }
    return {
        "by_context_length": result,
        "overall_micro_average": _micro(samples),
        "acceptance_policy": {
            "primary_scope": "non_terminal_non_truncated_rounds",
            "terminal_rounds_excluded": True,
            "truncated_rounds_excluded": True,
            "all_round_raw_counts_retained": True,
        },
        "percentile_method": "linear interpolation at (n - 1) * p",
    }
