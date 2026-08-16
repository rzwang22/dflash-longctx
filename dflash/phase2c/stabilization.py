"""Framework-lazy decision-stabilization helpers for Phase 2C.3.

The materializer copies this module beside the remote custom model.  Keep it free
of Torch imports so CLI and schema tests never import a model framework.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from typing import Any

DECISION_STABILIZATION_MODES = ("off", "always", "margin", "shadow")
DEFAULT_MARGIN_THRESHOLDS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def validate_decision_stabilization(mode: str, threshold: float) -> tuple[str, float]:
    """Validate online decision-stabilization options before cache mutation."""

    if mode not in DECISION_STABILIZATION_MODES:
        raise ValueError(
            "decision stabilization must be one of off, always, margin, shadow"
        )
    value = float(threshold)
    if not math.isfinite(value) or value < 0:
        raise ValueError("decision margin threshold must be finite and non-negative")
    return mode, value


def should_recheck(mode: str, minimum_margin: float | None, threshold: float) -> bool:
    """Return whether the canonical selected-path pass must execute."""

    validate_decision_stabilization(mode, threshold)
    if mode in {"always", "shadow"}:
        return True
    if mode == "off":
        return False
    if minimum_margin is None:
        raise ValueError("margin stabilization requires an observed path margin")
    return float(minimum_margin) <= float(threshold)


def stabilization_commit_source(mode: str, recheck_triggered: bool) -> str:
    """Select the only cache source allowed by each public mode."""

    validate_decision_stabilization(mode, 0.0)
    if recheck_triggered and mode in {"always", "margin"}:
        return "canonical_recheck"
    return "packed_native"


def canonical_selected_path_decision(
    *,
    selected_draft_token_ids: Iterable[int],
    canonical_parent_argmax_token_ids: Iterable[int],
    native_next_root_token_id: int,
    native_next_root_is_bonus: bool,
) -> dict[str, Any]:
    """Executable specification for a canonical recheck of one selected path.

    The argmax sequence contains one decision for every candidate parent plus the
    terminal parent.  A mismatch emits that parent's canonical argmax as the next
    root.  Full acceptance emits the canonical leaf argmax.
    """

    selected = [int(value) for value in selected_draft_token_ids]
    canonical = [int(value) for value in canonical_parent_argmax_token_ids]
    if len(canonical) < len(selected) + 1:
        raise ValueError("canonical recheck lacks a terminal-parent argmax")
    accepted = 0
    for candidate, target_token in zip(selected, canonical):
        if candidate != target_token:
            break
        accepted += 1
    next_root = canonical[accepted]
    full_path_accepted = accepted == len(selected)
    next_root_is_bonus = bool(full_path_accepted and native_next_root_is_bonus)
    changed = (
        accepted != len(selected)
        or next_root != int(native_next_root_token_id)
        or next_root_is_bonus != bool(native_next_root_is_bonus)
    )
    return {
        "accepted_draft_token_count": accepted,
        "accepted_draft_token_ids": selected[:accepted],
        "next_root_token_id": next_root,
        "next_root_is_bonus": next_root_is_bonus,
        "next_root_kind": "bonus" if next_root_is_bonus else "correction",
        "first_corrected_depth": accepted + 1 if changed else None,
        "decision_changed": changed,
    }


def canonical_kv_commit_spec(
    *,
    round_entry_prefix_length: int,
    native_accepted_draft_count: int,
    canonical_accepted_draft_count: int,
    block_size: int = 16,
) -> dict[str, Any]:
    """Pure structural specification for canonical cache replacement tests."""

    if round_entry_prefix_length < 0:
        raise ValueError("round entry prefix length cannot be negative")
    if not 0 <= canonical_accepted_draft_count <= native_accepted_draft_count:
        raise ValueError("canonical selected-path acceptance must be a native prefix")
    if native_accepted_draft_count >= block_size:
        raise ValueError("accepted draft count must be smaller than block size")
    committed = 1 + canonical_accepted_draft_count
    return {
        "committed_cache_source": "canonical_recheck",
        "canonical_query_length": block_size,
        "temporary_cache_length": round_entry_prefix_length + block_size,
        "committed_cache_length": round_entry_prefix_length + committed,
        "committed_root_and_draft_count": committed,
        "native_extra_accepted_node_count_discarded": (
            native_accepted_draft_count - canonical_accepted_draft_count
        ),
        "canonical_tail_discarded_count": (
            block_size - 1 - canonical_accepted_draft_count
        ),
        "correction_or_bonus_in_current_cache": False,
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _distribution(values: Iterable[float | int | None]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean) if clean else None,
        "median": statistics.median(clean) if clean else None,
        "p90": _percentile(clean, 0.9),
    }


def decision_margin_frontier(
    rounds: list[dict[str, Any]],
    *,
    thresholds: Iterable[float] = DEFAULT_MARGIN_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Compute theoretical margin triggers and missed divergences from shadow rows."""

    eligible = [row for row in rounds if row.get("minimum_path_margin") is not None]
    divergent = [
        row for row in eligible if row.get("native_canonical_decision_equal") is False
    ]
    result = []
    for raw_threshold in thresholds:
        threshold = float(raw_threshold)
        triggered = [
            row for row in eligible if float(row["minimum_path_margin"]) <= threshold
        ]
        captured = [row for row in divergent if row in triggered]
        missed = len(divergent) - len(captured)
        recheck_latencies = [
            row.get("canonical_recheck_cuda_elapsed_ms") for row in triggered
        ]
        result.append(
            {
                "threshold": threshold,
                "eligible_round_count": len(eligible),
                "divergent_round_count": len(divergent),
                "triggered_round_count": len(triggered),
                "captured_divergence_count": len(captured),
                "false_negative_count": missed,
                "trigger_rate": len(triggered) / len(eligible) if eligible else None,
                "captured_divergence_rate": (
                    len(captured) / len(divergent) if divergent else None
                ),
                "false_negative_rate": missed / len(divergent) if divergent else None,
                "estimated_recheck_cuda_elapsed_ms": _distribution(recheck_latencies),
                "strictly_lossless_supported": bool(divergent) and missed == 0,
            }
        )
    return result


def summarize_decision_stabilization(
    rounds: list[dict[str, Any]], samples: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate Phase 2C.3 modes without treating unavailable timing as zero."""

    semantic = [row for row in rounds if row.get("profiling_repeat", 0) == 0]
    rechecked = [row for row in semantic if row.get("recheck_triggered")]
    divergent = [
        row for row in semantic if row.get("native_canonical_decision_equal") is False
    ]
    false_negatives = [row for row in semantic if row.get("false_negative") is True]
    output_divergent_samples = [
        sample
        for sample in samples
        if sample.get("correctness", {}).get("reference_output_ids_equal") is False
    ]
    modes = sorted(
        {
            str(row.get("stabilization_mode"))
            for row in semantic
            if row.get("stabilization_mode") is not None
        }
    )
    thresholds = sorted(
        {
            float(row["decision_margin_threshold"])
            for row in semantic
            if row.get("decision_margin_threshold") is not None
        }
    )
    formal_latencies = [
        repeat.get("cuda_elapsed_ms")
        for sample in samples
        for repeat in sample.get("control", {}).get("repeats", [])
    ]
    returned_tokens = sum(
        int(sample.get("generated_token_count", 0)) for sample in samples
    )
    total_formal_ms = sum(
        float(value) for value in formal_latencies if value is not None
    )
    reference_latencies = [
        repeat.get("cuda_elapsed_ms")
        for sample in samples
        for repeat in sample.get("reference", {}).get("control_repeats", [])
    ]
    target_only_latencies = [
        sample.get("target_only", {}).get("cuda_elapsed_ms") for sample in samples
    ]
    formal_clean = [float(value) for value in formal_latencies if value is not None]
    reference_clean = [
        float(value) for value in reference_latencies if value is not None
    ]
    target_only_clean = [
        float(value) for value in target_only_latencies if value is not None
    ]
    formal_median = statistics.median(formal_clean) if formal_clean else None

    def effective_acceptance(row: dict[str, Any]) -> float:
        value = row.get("accepted_draft_token_count")
        if value is None:
            value = (
                row.get("canonical_accepted_depth")
                if row.get("committed_cache_source") == "canonical_recheck"
                else row.get("native_accepted_depth")
            )
        return float(value)

    return {
        "schema_version": "1.0",
        "sample_count": len(samples),
        "stabilization_modes": modes,
        "decision_margin_thresholds": thresholds,
        "round_count": len(semantic),
        "rechecked_round_count": len(rechecked),
        "divergent_decision_round_count": len(divergent),
        "output_divergent_sample_count": len(output_divergent_samples),
        "false_negative_count": len(false_negatives),
        "round_recheck_rate": len(rechecked) / len(semantic) if semantic else None,
        "decision_divergence_rate": len(divergent) / len(semantic)
        if semantic
        else None,
        "output_divergence_rate": (
            len(output_divergent_samples) / len(samples) if samples else None
        ),
        "threshold_false_negative_rate": (
            len(false_negatives) / len(divergent) if divergent else None
        ),
        "canonical_recheck_cuda_elapsed_ms": _distribution(
            row.get("canonical_recheck_cuda_elapsed_ms") for row in rechecked
        ),
        "all_verifier_decision_margins": _distribution(
            margin for row in semantic for margin in (row.get("verifier_margins") or [])
        ),
        "divergent_decision_margins": _distribution(
            row.get("first_corrected_native_margin") for row in divergent
        ),
        "nondivergent_decision_margins": _distribution(
            margin
            for row in semantic
            if row.get("native_canonical_decision_equal") is True
            for margin in (row.get("verifier_margins") or [])
        ),
        "kv_replacement_cuda_elapsed_ms": _distribution(
            row.get("kv_replacement_cuda_elapsed_ms") for row in rechecked
        ),
        "total_recheck_overhead_cuda_elapsed_ms": _distribution(
            row.get("total_recheck_overhead_cuda_elapsed_ms") for row in rechecked
        ),
        "stabilized_round_cuda_elapsed_ms": _distribution(
            row.get("stabilized_round_cuda_elapsed_ms") for row in semantic
        ),
        "formal_end_to_end_cuda_elapsed_ms": _distribution(formal_latencies),
        "mean_accepted_draft_tokens": (
            statistics.fmean(effective_acceptance(row) for row in semantic)
            if semantic
            else None
        ),
        "latency_per_returned_token_ms": (
            total_formal_ms / returned_tokens
            if returned_tokens and total_formal_ms
            else None
        ),
        "speedup_relative_to_original_dflash": (
            statistics.median(reference_clean) / formal_median
            if reference_clean and formal_median
            else None
        ),
        "speedup_relative_to_target_only": (
            statistics.median(target_only_clean) / formal_median
            if target_only_clean and formal_median
            else None
        ),
        "speedup_unavailable_reason": (
            None
            if reference_clean and target_only_clean and formal_median
            else "required CUDA latency control unavailable"
        ),
        "interpretation": {
            "shadow_oracle_excluded_from_formal_native_timing": True,
            "unavailable_values_are_null_not_zero": True,
            "margin_is_empirically_stabilized_unless_false_negative_rate_is_zero_on_the_evaluation_cohort": True,
        },
    }


def stabilization_summary_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one flat summary row for the run's single stabilization mode."""

    recheck = summary["canonical_recheck_cuda_elapsed_ms"]
    formal = summary["formal_end_to_end_cuda_elapsed_ms"]
    return [
        {
            "stabilization_mode": (
                summary["stabilization_modes"][0]
                if len(summary["stabilization_modes"]) == 1
                else None
            ),
            "decision_margin_threshold": (
                summary["decision_margin_thresholds"][0]
                if len(summary["decision_margin_thresholds"]) == 1
                else None
            ),
            "sample_count": summary["sample_count"],
            "round_count": summary["round_count"],
            "round_recheck_rate": summary["round_recheck_rate"],
            "decision_divergence_rate": summary["decision_divergence_rate"],
            "output_divergence_rate": summary["output_divergence_rate"],
            "threshold_false_negative_rate": summary["threshold_false_negative_rate"],
            "recheck_cuda_mean_ms": recheck["mean"],
            "recheck_cuda_p50_ms": recheck["median"],
            "recheck_cuda_p90_ms": recheck["p90"],
            "end_to_end_cuda_median_ms": formal["median"],
            "mean_accepted_draft_tokens": summary["mean_accepted_draft_tokens"],
            "latency_per_returned_token_ms": summary["latency_per_returned_token_ms"],
            "speedup_relative_to_original_dflash": summary[
                "speedup_relative_to_original_dflash"
            ],
            "speedup_relative_to_target_only": summary[
                "speedup_relative_to_target_only"
            ],
            "speedup_unavailable_reason": summary["speedup_unavailable_reason"],
        }
    ]
