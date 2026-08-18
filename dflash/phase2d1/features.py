"""Canonical target-blind runtime features shared by Phases 2D.0/2D.1A/2D.1B.

This module is intentionally framework-free.  The instrumented model receives a
materialized copy and feeds it bounded Top-16 candidate records after the draft
forward.  No target-side value is accepted by this API.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import fmean
from typing import Any

Node = tuple[int, ...]
MAXIMUM_DRAFT_DEPTH = 15
RUNTIME_FEATURE_SCHEMA_VERSION = "phase2d1-runtime-features-v1"
RUNTIME_FEATURE_NAMES = (
    "context_length",
    "node_budget_cap",
    "mean_top1_log_probability",
    "min_top1_log_probability",
    "mean_top1_top2_margin",
    "min_top1_top2_margin",
    "mean_top16_cumulative_probability_mass",
    "mean_top16_renormalized_entropy",
    "mean_adjacent_confidence_change",
    "mean_surviving_prefix_count",
    "mean_static_beam_score",
    "min_static_beam_score",
    "tree_depth",
    "policy_configuration_count",
)
FORBIDDEN_RUNTIME_FIELD_FRAGMENTS = (
    "target",
    "accepted",
    "oracle",
    "winner",
    "coverage",
    "gain",
    "gap_class",
    "continuation",
    "membership",
    "rank",
    "posterior",
    "correction",
)


def feature_definitions() -> dict[str, str]:
    return {
        "context_length": "nominal chat-templated prompt token length",
        "node_budget_cap": "root-inclusive online tree node-budget cap",
        "mean_top1_log_probability": "mean draft Top-1 log probability over 15 positions",
        "min_top1_log_probability": "minimum draft Top-1 log probability over 15 positions",
        "mean_top1_top2_margin": "mean Top-1 minus Top-2 draft log-probability margin",
        "min_top1_top2_margin": "minimum Top-1 minus Top-2 draft log-probability margin",
        "mean_top16_cumulative_probability_mass": "mean captured Top-16 full-softmax probability mass",
        "mean_top16_renormalized_entropy": "mean entropy after renormalizing captured Top-16 mass",
        "mean_adjacent_confidence_change": "mean adjacent change in Top-1/Top-2 margin over positions 2..15",
        "mean_surviving_prefix_count": "mean same-depth node count in the canonical score-beam structural state",
        "mean_static_beam_score": "mean maximum same-depth normalized score in the canonical score-beam structural state",
        "min_static_beam_score": "minimum available maximum same-depth normalized score",
        "tree_depth": "fixed DFlash marginal draft depth (15)",
        "policy_configuration_count": "number of frozen allowlisted policies at this node budget",
    }


def validate_runtime_feature_names(names: Iterable[str]) -> tuple[str, ...]:
    values = tuple(str(value) for value in names)
    if values != RUNTIME_FEATURE_NAMES:
        raise ValueError("runtime feature schema/order does not match the canonical contract")
    leaking = [
        name
        for name in values
        if any(fragment in name for fragment in FORBIDDEN_RUNTIME_FIELD_FRAGMENTS)
    ]
    if leaking:
        raise ValueError(f"target-dependent fields entered runtime schema: {leaking}")
    return values


def _score_maps(positions: list[dict[str, Any]]) -> list[dict[int, float]]:
    return [
        {
            int(candidate["token_id"]): float(candidate["log_probability"])
            for candidate in position["candidates"]
        }
        for position in positions
    ]


def _static_feature_state(
    nodes: set[Node], positions: list[dict[str, Any]], depth: int
) -> dict[str, Any]:
    same_depth = [node for node in nodes if len(node) == depth]
    if not same_depth:
        return {
            "static_beam_score": None,
            "static_beam_score_unavailable_reason": "no_static_node_at_depth",
            "surviving_prefix_count": 0,
        }
    score_maps = _score_maps(positions)
    scores = [
        sum(score_maps[index][token] for index, token in enumerate(node)) / len(node)
        for node in same_depth
    ]
    return {
        "static_beam_score": max(scores),
        "static_beam_score_unavailable_reason": None,
        "surviving_prefix_count": len(same_depth),
    }


def build_deployable_feature_rows(
    *,
    positions: list[dict[str, Any]],
    canonical_nodes: set[Node],
    context_length: int,
    top_m: int,
    node_budget: int,
) -> list[dict[str, Any]]:
    """Build the exact per-position feature records used offline and online."""

    if len(positions) != MAXIMUM_DRAFT_DEPTH:
        raise ValueError("runtime features require exactly 15 draft positions")
    if top_m < 2:
        raise ValueError("deployable confidence features require at least Top-2")
    rows = []
    previous_margin = None
    for depth, position in enumerate(positions, 1):
        candidates = position.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != top_m:
            raise ValueError("each feature position must match the configured Top-M")
        log_probabilities = [float(value["log_probability"]) for value in candidates]
        if any(not math.isfinite(value) for value in log_probabilities):
            raise ValueError("draft candidate log probabilities must be finite")
        probabilities = [math.exp(value) for value in log_probabilities]
        probability_mass = sum(probabilities)
        if not probability_mass > 0.0 or not math.isfinite(probability_mass):
            raise ValueError("captured Top-16 probability mass must be finite and positive")
        normalized = [value / probability_mass for value in probabilities]
        entropy = -sum(value * math.log(value) for value in normalized if value > 0)
        current_margin = log_probabilities[0] - log_probabilities[1]
        rows.append(
            {
                "position_index": depth,
                "top1_log_probability": log_probabilities[0],
                "top1_top2_log_probability_margin": current_margin,
                "top_m_probability_mass": probability_mass,
                "top_m_renormalized_entropy": entropy,
                "adjacent_position_confidence_change": (
                    current_margin - previous_margin
                    if previous_margin is not None
                    else None
                ),
                "adjacent_position_confidence_change_unavailable_reason": (
                    None if previous_margin is not None else "no_previous_position"
                ),
                **_static_feature_state(canonical_nodes, positions, depth),
                "tree_depth": depth,
                "context_length": int(context_length),
                "top_m": int(top_m),
                "node_budget": int(node_budget),
            }
        )
        previous_margin = current_margin
    return rows


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty deployable feature")
    return fmean(values)


def aggregate_runtime_features(
    rows: list[dict[str, Any]],
    *,
    context_length: int,
    node_budget: int,
    policy_configuration_count: int,
) -> dict[str, float]:
    """Aggregate per-position records into the frozen selector input vector."""

    if len(rows) != MAXIMUM_DRAFT_DEPTH:
        raise ValueError("selector runtime features require all 15 draft positions")
    adjacent = [
        float(row["adjacent_position_confidence_change"])
        for row in rows
        if row["adjacent_position_confidence_change"] is not None
    ]
    static_scores = [
        float(row["static_beam_score"])
        for row in rows
        if row["static_beam_score"] is not None
    ]
    features = {
        "context_length": float(context_length),
        "node_budget_cap": float(node_budget),
        "mean_top1_log_probability": _mean(
            [float(row["top1_log_probability"]) for row in rows]
        ),
        "min_top1_log_probability": min(
            float(row["top1_log_probability"]) for row in rows
        ),
        "mean_top1_top2_margin": _mean(
            [float(row["top1_top2_log_probability_margin"]) for row in rows]
        ),
        "min_top1_top2_margin": min(
            float(row["top1_top2_log_probability_margin"]) for row in rows
        ),
        "mean_top16_cumulative_probability_mass": _mean(
            [float(row["top_m_probability_mass"]) for row in rows]
        ),
        "mean_top16_renormalized_entropy": _mean(
            [float(row["top_m_renormalized_entropy"]) for row in rows]
        ),
        "mean_adjacent_confidence_change": _mean(adjacent),
        "mean_surviving_prefix_count": _mean(
            [float(row["surviving_prefix_count"]) for row in rows]
        ),
        "mean_static_beam_score": _mean(static_scores),
        "min_static_beam_score": min(static_scores),
        "tree_depth": float(MAXIMUM_DRAFT_DEPTH),
        "policy_configuration_count": float(policy_configuration_count),
    }
    validate_runtime_feature_names(features)
    return features


def candidate_positions(
    token_ids: list[list[int]], log_probabilities: list[list[float]]
) -> list[dict[str, Any]]:
    """Convert bounded draft Top-16 values into canonical policy input records."""

    if len(token_ids) != len(log_probabilities):
        raise ValueError("candidate token/log-probability position count differs")
    output = []
    for position, (tokens, scores) in enumerate(zip(token_ids, log_probabilities), 1):
        if len(tokens) != len(scores):
            raise ValueError("candidate token/log-probability width differs")
        if len(set(map(int, tokens))) != len(tokens):
            raise ValueError("one draft position contains duplicate candidate tokens")
        output.append(
            {
                "position_index": position,
                "candidates": [
                    {
                        "token_id": int(token),
                        "log_probability": float(score),
                        "rank": rank,
                    }
                    for rank, (token, score) in enumerate(zip(tokens, scores), 1)
                ],
            }
        )
    return output
