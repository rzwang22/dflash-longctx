"""Exact, capped same-depth target-prefix rank without Cartesian enumeration."""

from __future__ import annotations

from math import prod
from typing import Any


def _ordered_candidates(
    positions: list[dict[str, Any]], depth: int
) -> list[list[tuple[int, float]]]:
    return [
        [
            (int(candidate["token_id"]), float(candidate["log_probability"]))
            for candidate in position["candidates"]
        ]
        for position in positions[:depth]
    ]


def capped_target_prefix_score_rank(
    positions: list[dict[str, Any]],
    target_prefix: list[int],
    *,
    rank_cap: int = 100_000,
) -> dict[str, Any]:
    """Rank ``target_prefix`` under Phase 2B score/tie semantics.

    At a fixed depth, length-normalized and cumulative scores induce the same
    ordering. Branch-and-bound counts prefixes with a larger cumulative score,
    or the same score and lexicographically smaller token tuple. The search
    stops after ``rank_cap`` prefixes are known to outrank the target; the cap
    is therefore a censoring threshold, never an observed rank.
    """

    depth = len(target_prefix)
    if depth < 1 or depth > len(positions):
        raise ValueError("target prefix depth must be within captured positions")
    if rank_cap < 1:
        raise ValueError("rank_cap must be positive")
    candidates = _ordered_candidates(positions, depth)
    score_maps = [dict(items) for items in candidates]
    target = tuple(int(token) for token in target_prefix)
    if any(token not in score_maps[index] for index, token in enumerate(target)):
        return {
            "target_prefix_score_rank": None,
            "target_prefix_score_rank_lower_bound": None,
            "target_prefix_score_rank_censored": False,
            "target_prefix_score_rank_available": False,
            "unavailable_reason": "target_prefix_not_in_captured_top_k_space",
        }

    target_score = sum(score_maps[index][token] for index, token in enumerate(target))
    suffix_max = [0.0] * (depth + 1)
    suffix_min = [0.0] * (depth + 1)
    suffix_count = [1] * (depth + 1)
    for index in range(depth - 1, -1, -1):
        scores = [score for _, score in candidates[index]]
        suffix_max[index] = suffix_max[index + 1] + max(scores)
        suffix_min[index] = suffix_min[index + 1] + min(scores)
        suffix_count[index] = suffix_count[index + 1] * len(scores)

    outranking = 0

    def count(index: int, score: float, prefix: tuple[int, ...]) -> None:
        nonlocal outranking
        if outranking >= rank_cap:
            return
        minimum = score + suffix_min[index]
        maximum = score + suffix_max[index]
        if minimum > target_score:
            outranking += min(rank_cap - outranking, suffix_count[index])
            return
        if maximum < target_score:
            return
        if index == depth:
            if score > target_score or (score == target_score and prefix < target):
                outranking += 1
            return
        for token_id, token_score in candidates[index]:
            count(index + 1, score + token_score, prefix + (token_id,))
            if outranking >= rank_cap:
                return

    count(0, 0.0, ())
    if outranking >= rank_cap:
        return {
            "target_prefix_score_rank": None,
            "target_prefix_score_rank_lower_bound": rank_cap + 1,
            "target_prefix_score_rank_censored": True,
            "target_prefix_score_rank_available": True,
            "unavailable_reason": None,
        }
    return {
        "target_prefix_score_rank": outranking + 1,
        "target_prefix_score_rank_lower_bound": outranking + 1,
        "target_prefix_score_rank_censored": False,
        "target_prefix_score_rank_available": True,
        "unavailable_reason": None,
    }


def brute_force_target_prefix_score_rank(
    positions: list[dict[str, Any]], target_prefix: list[int]
) -> int | None:
    """Reference implementation restricted to tiny synthetic test spaces."""

    depth = len(target_prefix)
    candidates = _ordered_candidates(positions, depth)
    if prod(len(items) for items in candidates) > 100_000:
        raise ValueError("brute-force rank is limited to 100,000 prefixes")
    score_maps = [dict(items) for items in candidates]
    target = tuple(int(token) for token in target_prefix)
    if any(token not in score_maps[index] for index, token in enumerate(target)):
        return None

    prefixes: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    for items in candidates:
        prefixes = [
            (prefix + (token,), score + token_score)
            for prefix, score in prefixes
            for token, token_score in items
        ]
    prefixes.sort(key=lambda item: (-item[1], item[0]))
    return next(
        index for index, (prefix, _) in enumerate(prefixes, 1) if prefix == target
    )
