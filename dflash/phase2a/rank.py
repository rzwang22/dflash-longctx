"""Pure-Python target-rank aggregation with single-frontier semantics."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

POSITION_CATEGORIES = (
    "accepted_prefix",
    "first_rejection_frontier",
    "post_rejection_counterfactual",
    "full_block_accepted",
)
RANK_BUCKETS = ("rank_1", "rank_2", "rank_3_4", "rank_5_8", "rank_9_16", "rank_over_16")


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _percentile(values: Sequence[int | float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def rank_bucket(rank: int) -> str:
    if rank < 1:
        raise ValueError("rank must be positive")
    if rank == 1:
        return "rank_1"
    if rank == 2:
        return "rank_2"
    if rank <= 4:
        return "rank_3_4"
    if rank <= 8:
        return "rank_5_8"
    if rank <= 16:
        return "rank_9_16"
    return "rank_over_16"


def _round_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event.get("task"),
        event.get("sample_id"),
        event.get("nominal_prompt_length"),
        event.get("diagnostic_run", 0),
        event["round_index"],
    )


def _sample_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event.get("task"),
        event.get("sample_id"),
        event.get("nominal_prompt_length"),
    )


def _recalls(events: Sequence[dict[str, Any]], top_ms: Sequence[int]) -> dict[str, Any]:
    result = {"event_count": len(events)}
    for top_m in top_ms:
        hits = sum(int(event["target_rank_in_draft"]) <= top_m for event in events)
        result[f"top_{top_m}_hit_count"] = hits
        result[f"top_{top_m}_recall"] = _ratio(hits, len(events))
    return result


def summarize_rank_events(
    events: Sequence[dict[str, Any]], *, top_ms: Sequence[int] = (2, 4, 8, 16)
) -> dict[str, Any]:
    """Summarize one group without constructing an independent-position tree."""

    configured_top_ms = tuple(sorted({int(value) for value in top_ms}))
    if not configured_top_ms or configured_top_ms[0] < 2:
        raise ValueError("Top-M recovery metrics begin at M=2")
    invalid_categories = {
        event["position_category"] for event in events
    } - set(POSITION_CATEGORIES)
    if invalid_categories:
        raise ValueError(f"Unknown rank position categories: {invalid_categories}")
    for event in events:
        rank = int(event["target_rank_in_draft"])
        for top_m in (1, 2, 4, 8, 16):
            field = f"target_in_top_{top_m}"
            if bool(event[field]) != (rank <= top_m):
                raise ValueError(f"{field} does not match target_rank_in_draft")
        if (event["target_token_id"] == event["draft_top1_token_id"]) != (rank == 1):
            raise ValueError("Top-1 token equality does not match deterministic rank")

    frontier = [
        event
        for event in events
        if event["position_category"] == "first_rejection_frontier"
    ]
    post_rejection = [
        event
        for event in events
        if event["position_category"] == "post_rejection_counterfactual"
    ]
    round_categories: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for event in events:
        round_categories[_round_key(event)].add(event["position_category"])
    rounds_with_frontier = sum(
        "first_rejection_frontier" in categories
        for categories in round_categories.values()
    )
    full_block_rounds = sum(
        categories == {"full_block_accepted"}
        for categories in round_categories.values()
    )
    if len(frontier) != rounds_with_frontier:
        raise ValueError("Each rejected round must have exactly one frontier event")
    if any(event["target_rank_in_draft"] == 1 for event in frontier):
        raise ValueError("Top-1 recall at the rejection frontier must be zero")

    frontier_by_sample: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in frontier:
        frontier_by_sample[_sample_key(event)].append(event)
    macro = {}
    for top_m in configured_top_ms:
        sample_recalls = [
            sum(int(event["target_rank_in_draft"]) <= top_m for event in group)
            / len(group)
            for group in frontier_by_sample.values()
            if group
        ]
        macro[f"top_{top_m}_recall"] = (
            statistics.fmean(sample_recalls) if sample_recalls else None
        )
    frontier_ranks = [int(event["target_rank_in_draft"]) for event in frontier]
    histogram = {bucket: 0 for bucket in RANK_BUCKETS}
    for rank in frontier_ranks:
        histogram[rank_bucket(rank)] += 1

    return {
        "sample_count": len({_sample_key(event) for event in events}),
        "round_count": len(round_categories),
        "rounds_with_rejection_frontier": rounds_with_frontier,
        "full_block_accepted_round_count": full_block_rounds,
        "position_category_counts": {
            category: sum(event["position_category"] == category for event in events)
            for category in POSITION_CATEGORIES
        },
        "first_rejection_frontier_top_m_recall_micro": _recalls(
            frontier, configured_top_ms
        ),
        "first_rejection_frontier_top_m_recall_macro": {
            **macro,
            "contributing_sample_count": len(frontier_by_sample),
        },
        "all_position_top_m_recall_micro": _recalls(events, configured_top_ms),
        "post_rejection_counterfactual_top_m_recall_micro": _recalls(
            post_rejection, configured_top_ms
        ),
        "frontier_rank": {
            "count": len(frontier_ranks),
            "median": (
                float(statistics.median(frontier_ranks)) if frontier_ranks else None
            ),
            "p90": _percentile(frontier_ranks, 0.90),
            "p95": _percentile(frontier_ranks, 0.95),
            "over_16_fraction": _ratio(
                sum(rank > 16 for rank in frontier_ranks), len(frontier_ranks)
            ),
            "histogram": histogram,
        },
        "interpretation": {
            "primary_metric": "first_rejection_frontier_top_m_recall",
            "frontier_top_1_recall": 0.0 if frontier else None,
            "frontier_top_1_is_recovery_metric": False,
            "single_frontier_recoverability_only": True,
            "post_rejection_is_counterfactual": True,
            "tree_accepted_length_derived": False,
            "target_bonus_token_included": False,
        },
    }


def _group(
    events: Iterable[dict[str, Any]], key_fields: Sequence[str]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[tuple(event.get(field) for field in key_fields)].append(event)
    return grouped


def aggregate_rank_events(
    events: Sequence[dict[str, Any]], *, top_ms: Sequence[int] = (2, 4, 8, 16)
) -> dict[str, Any]:
    by_task_context = {
        f"{task}:{context_length}": {
            "task": task,
            "context_length": context_length,
            **summarize_rank_events(group, top_ms=top_ms),
        }
        for (task, context_length), group in sorted(
            _group(events, ("task", "nominal_prompt_length")).items()
        )
    }
    by_task = {
        str(task): {"task": task, **summarize_rank_events(group, top_ms=top_ms)}
        for (task,), group in sorted(_group(events, ("task",)).items())
    }
    by_sample = {
        f"{task}:{sample_id}:{context_length}": {
            "task": task,
            "sample_id": sample_id,
            "context_length": context_length,
            **summarize_rank_events(group, top_ms=top_ms),
        }
        for (task, sample_id, context_length), group in sorted(
            _group(
                events, ("task", "sample_id", "nominal_prompt_length")
            ).items()
        )
    }
    by_block_position = {
        str(block_position): {
            "block_position": block_position,
            **summarize_rank_events(group, top_ms=top_ms),
        }
        for (block_position,), group in sorted(
            _group(events, ("block_position",)).items()
        )
    }
    return {
        "top_ms": sorted({int(value) for value in top_ms}),
        "overall": summarize_rank_events(events, top_ms=top_ms),
        "by_task_and_context_length": by_task_context,
        "by_task": by_task,
        "by_sample": by_sample,
        "by_block_position": by_block_position,
        "tree_accepted_length_derived": False,
    }
