"""Pure-Python target-path survival replay and aggregate construction."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from statistics import mean, median
from typing import Any

from dflash.phase2b.tree import (
    confidence_gated_tree,
    covered_target_prefix_depth,
    oracle_prefix_closed_tree,
    score_beam_tree,
    target_candidate_rank,
    top1_chain_tree,
    tree_shape,
)

from .confidence import confidence_diagnostics, draft_confidence_features
from .policies import (
    layerwise_beam_tree,
    minimum_survival_budget,
    per_depth_policy_state,
)
from .ranking import capped_target_prefix_score_rank

DEPTH_CHECKPOINTS = (1, 2, 4, 8, 15)


def _mean_or_none(values: list[float | int | bool]) -> float | None:
    return mean(values) if values else None


def _distribution(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "sum": 0}
    ordered = sorted(values)
    p90_index = max(0, math.ceil(0.9 * len(ordered)) - 1)
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "p90": ordered[p90_index],
        "sum": sum(values),
    }


def _longest_true_span(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _round_identity(capture: dict[str, Any]) -> dict[str, Any]:
    return {
        key: capture.get(key)
        for key in (
            "task",
            "nominal_prompt_length",
            "actual_prompt_length",
            "sample_id",
            "source_sample_id",
            "source_text_sha256",
            "generation_repeat",
            "round_index",
            "terminal_round",
            "stop_reason",
            "reference_right_censored",
            "full_depth_evaluation_eligible",
        )
    }


def _base_depth_events(
    capture: dict[str, Any], *, rank_cap: int
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    reference = [
        int(value) for value in capture["reference_target_continuation_token_ids"]
    ]
    prefix: list[int] = []
    prefix_available = True
    cumulative_score = 0.0
    depth_rows = []
    confidence_rows = []
    first_unavailable = None
    off_top1: list[bool] = []
    observed_ranks: list[int] = []
    for depth, target_token in enumerate(reference, 1):
        position = capture["positions"][depth - 1]
        candidates = position["candidates"]
        rank = target_candidate_rank(position, target_token)
        token_available = rank is not None
        if not token_available and first_unavailable is None:
            first_unavailable = depth
        prefix_available = prefix_available and token_available
        prefix.append(target_token)
        candidate = next(
            (item for item in candidates if int(item["token_id"]) == int(target_token)),
            None,
        )
        if prefix_available and candidate is not None:
            cumulative_score += float(candidate["log_probability"])
        else:
            cumulative_score = float("nan")
        features = draft_confidence_features(position)
        top1_token = int(candidates[0]["token_id"])
        top1_correct = top1_token == target_token
        off_top1.append(not top1_correct)
        if rank is not None:
            observed_ranks.append(rank)
        rank_result = (
            capped_target_prefix_score_rank(
                capture["positions"], prefix, rank_cap=rank_cap
            )
            if prefix_available
            else {
                "target_prefix_score_rank": None,
                "target_prefix_score_rank_lower_bound": None,
                "target_prefix_score_rank_censored": False,
                "target_prefix_score_rank_available": False,
                "unavailable_reason": "target_prefix_not_in_captured_top_k_space",
            }
        )
        target_gap = (
            float(candidates[0]["log_probability"])
            - float(candidate["log_probability"])
            if candidate is not None
            else None
        )
        row = {
            "depth": depth,
            "observable_depth": len(reference),
            "right_censored": bool(capture["reference_right_censored"]),
            "target_token_id": target_token,
            "target_token_in_top_k": token_available,
            "target_token_in_top_16": token_available,
            "target_token_rank": rank,
            "target_token_rank_label": str(rank) if rank is not None else ">16",
            "top1_token_id": top1_token,
            "top1_correct": top1_correct,
            "top1_incorrect": not top1_correct,
            **features,
            "top1_minus_target_log_probability_gap": target_gap,
            "target_prefix_cumulative_draft_log_probability": (
                cumulative_score if prefix_available else None
            ),
            **rank_result,
        }
        depth_rows.append(row)
        confidence_rows.append(
            {
                **_round_identity(capture),
                "depth": depth,
                "top1_incorrect": not top1_correct,
                **features,
                "top1_minus_target_log_probability_gap_post_hoc": target_gap,
            }
        )
    off_depths = [index for index, value in enumerate(off_top1, 1) if value]
    stats = {
        "baseline_accepted_draft_depth": int(capture["accepted_draft_token_count"]),
        "observable_target_horizon": len(reference),
        "first_candidate_unavailable_depth": first_unavailable,
        "off_top1_decision_count": sum(off_top1),
        "off_top1_decision_ratio": mean(off_top1) if off_top1 else None,
        "first_off_top1_decision_depth": off_depths[0] if off_depths else None,
        "longest_consecutive_off_top1_span": _longest_true_span(off_top1),
        "target_rank_mean": mean(observed_ranks) if observed_ranks else None,
        "target_rank_median": median(observed_ranks) if observed_ranks else None,
        "target_rank_maximum": max(observed_ranks) if observed_ranks else None,
        "target_rank_recall_at_2": mean(
            row["target_token_rank"] is not None and row["target_token_rank"] <= 2
            for row in depth_rows
        )
        if depth_rows
        else None,
        "target_rank_recall_at_4": mean(
            row["target_token_rank"] is not None and row["target_token_rank"] <= 4
            for row in depth_rows
        )
        if depth_rows
        else None,
        "target_rank_recall_at_8": mean(
            row["target_token_rank"] is not None and row["target_token_rank"] <= 8
            for row in depth_rows
        )
        if depth_rows
        else None,
        "target_rank_recall_at_16": mean(
            row["target_token_rank"] is not None and row["target_token_rank"] <= 16
            for row in depth_rows
        )
        if depth_rows
        else None,
        "candidate_unavailable_depth_count": sum(
            not row["target_token_in_top_k"] for row in depth_rows
        ),
        "score_rank_censored_depth_count": sum(
            row["target_prefix_score_rank_censored"] for row in depth_rows
        ),
    }
    return depth_rows, stats, confidence_rows


def _policy_specs(
    *, confidence_thresholds: tuple[float, ...], beam_widths: tuple[int, ...]
) -> list[dict[str, Any]]:
    specs = [
        {
            "policy": "top1_chain",
            "variant": "exact_baseline_chain",
            "class": "deployable",
            "uses_target": False,
            "builder": lambda positions, budget: top1_chain_tree(
                positions, budget=max(budget, len(positions) + 1)
            ),
            "parameters": {},
        },
        {
            "policy": "score_beam",
            "variant": "top1_backbone_length_normalized_best_first",
            "class": "deployable",
            "uses_target": False,
            "builder": lambda positions, budget: score_beam_tree(
                positions, budget=budget
            ),
            "parameters": {},
        },
    ]
    for threshold in confidence_thresholds:
        specs.append(
            {
                "policy": "confidence_gated",
                "variant": f"top1_top2_margin_le_{threshold:g}",
                "class": "deployable",
                "uses_target": False,
                "builder": lambda positions, budget, threshold=threshold: (
                    confidence_gated_tree(
                        positions,
                        budget=budget,
                        margin_threshold=threshold,
                        branch_width=4,
                    )
                ),
                "parameters": {"margin_threshold": threshold, "branch_width": 4},
            }
        )
    for width in beam_widths:
        specs.append(
            {
                "policy": "layerwise_beam",
                "variant": f"fixed_width_{width}",
                "class": "deployable_diagnostic",
                "uses_target": False,
                "builder": lambda positions, budget, width=width: layerwise_beam_tree(
                    positions, budget=budget, beam_width=width
                ),
                "parameters": {"beam_width": width},
            }
        )
    specs.append(
        {
            "policy": "oracle_prefix_closed",
            "variant": "target_informed_structural_upper_bound",
            "class": "oracle",
            "uses_target": True,
            "builder": None,
            "parameters": {"not_deployable": True},
        }
    )
    return specs


def _event_from_tree(
    capture: dict[str, Any],
    base_depths: list[dict[str, Any]],
    round_stats: dict[str, Any],
    *,
    spec: dict[str, Any],
    budget: int,
    nodes: set[tuple[int, ...]],
    minimum_by_depth: list[dict[str, Any]],
    top1_coverage: int,
    oracle_coverage: int,
) -> dict[str, Any]:
    reference = [
        int(value) for value in capture["reference_target_continuation_token_ids"]
    ]
    coverage = covered_target_prefix_depth(nodes, reference)
    states = per_depth_policy_state(nodes, capture["positions"], reference)
    depth_events = []
    for base, state, minimum in zip(base_depths, states, minimum_by_depth):
        depth_events.append({**base, **state, **minimum})
    shape = tree_shape(nodes)
    extra = coverage - top1_coverage
    additional_nodes = max(0, int(shape["candidate_node_count"]) - 15)
    baseline_accepted = int(capture["accepted_draft_token_count"])
    first_rejection_eligible = baseline_accepted < len(reference)
    candidate_available_but_pruned = sum(
        row["pruning_reason"] == "candidate_available_but_pruned"
        for row in depth_events
    )
    return {
        "schema_version": "1.0",
        **_round_identity(capture),
        "capture_directory": capture.get("_capture_directory"),
        "policy": spec["policy"],
        "policy_variant": spec["variant"],
        "policy_class": spec["class"],
        "policy_uses_target_information": spec["uses_target"],
        "deployable_policy_is_target_blind": not spec["uses_target"],
        "policy_parameters": spec["parameters"],
        "requested_total_tree_node_budget": budget,
        "root_included_in_budget": True,
        **shape,
        **round_stats,
        "covered_target_prefix_depth": coverage,
        "maximum_survived_target_depth": coverage,
        "top1_covered_target_prefix_depth": top1_coverage,
        "extra_covered_target_tokens_vs_top1": extra,
        "positive_improvement_vs_top1": extra > 0,
        "first_rejection_eligible": first_rejection_eligible,
        "first_rejection_recovered": first_rejection_eligible
        and coverage > baseline_accepted,
        "additional_candidate_nodes_vs_top1": additional_nodes,
        "extra_covered_tokens_per_additional_candidate_node": (
            extra / additional_nodes if additional_nodes else None
        ),
        "candidate_available_but_pruned_depth_count": candidate_available_but_pruned,
        "first_pruned_depth": next(
            (
                row["depth"]
                for row in depth_events
                if row["pruning_reason"] == "candidate_available_but_pruned"
            ),
            None,
        ),
        "oracle_covered_target_prefix_depth": oracle_coverage,
        "oracle_improvable_round": oracle_coverage > top1_coverage,
        "oracle_improvable_but_this_policy_no_improvement": oracle_coverage
        > top1_coverage
        and extra <= 0
        and not spec["uses_target"],
        "score_rank_censored_round": any(
            row["target_prefix_score_rank_censored"] for row in base_depths
        ),
        "depth_events": depth_events,
    }


def analyze_capture_rounds(
    captures: list[dict[str, Any]],
    *,
    node_budgets: tuple[int, ...] = (16, 32, 64, 128, 256),
    beam_widths: tuple[int, ...] = (2, 4, 8, 16, 32),
    confidence_thresholds: tuple[float, ...] = (0.25, 0.5, 1.0),
    rank_cap: int = 100_000,
) -> dict[str, Any]:
    """Replay all policies and build rows without importing model frameworks."""

    if not captures:
        raise ValueError("no capture rounds were provided")
    specs = _policy_specs(
        confidence_thresholds=confidence_thresholds, beam_widths=beam_widths
    )
    events = []
    confidence_rows = []
    for capture in captures:
        reference = [
            int(value) for value in capture["reference_target_continuation_token_ids"]
        ]
        base_depths, round_stats, current_confidence = _base_depth_events(
            capture, rank_cap=rank_cap
        )
        confidence_rows.extend(current_confidence)
        trees: dict[tuple[str, int], set[tuple[int, ...]]] = {}
        for spec in specs:
            for budget in node_budgets:
                if spec["uses_target"]:
                    nodes = oracle_prefix_closed_tree(
                        capture["positions"], reference, budget=budget
                    )
                else:
                    builder: Callable[..., set[tuple[int, ...]]] = spec["builder"]
                    nodes = builder(capture["positions"], budget)
                trees[(spec["variant"], budget)] = nodes

        top1_coverage = covered_target_prefix_depth(
            trees[("exact_baseline_chain", node_budgets[0])], reference
        )
        oracle_coverage = max(
            covered_target_prefix_depth(
                trees[("target_informed_structural_upper_bound", budget)], reference
            )
            for budget in node_budgets
        )
        for spec in specs:
            survival = {
                budget: covered_target_prefix_depth(
                    trees[(spec["variant"], budget)], reference
                )
                for budget in node_budgets
            }
            minimum_by_depth = [
                minimum_survival_budget(survival, depth=depth)
                for depth in range(1, len(reference) + 1)
            ]
            emit_budgets = (
                (node_budgets[0],) if spec["policy"] == "top1_chain" else node_budgets
            )
            for budget in emit_budgets:
                events.append(
                    _event_from_tree(
                        capture,
                        base_depths,
                        round_stats,
                        spec=spec,
                        budget=budget,
                        nodes=trees[(spec["variant"], budget)],
                        minimum_by_depth=minimum_by_depth,
                        top1_coverage=top1_coverage,
                        oracle_coverage=oracle_coverage,
                    )
                )
    return {
        "events": events,
        "confidence": confidence_diagnostics(confidence_rows),
        "confidence_position_rows": confidence_rows,
    }


def aggregate_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[
            (
                event["task"],
                event["nominal_prompt_length"],
                event["policy"],
                event["policy_variant"],
                event["requested_total_tree_node_budget"],
            )
        ].append(event)
    rows = []
    for key, group in sorted(grouped.items()):
        task, context, policy, variant, budget = key
        sample_values: dict[str, list[float]] = defaultdict(list)
        sample_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in group:
            sample_values[event["sample_id"]].append(
                event["covered_target_prefix_depth"]
            )
            sample_events[event["sample_id"]].append(event)
        eligible_first = [event for event in group if event["first_rejection_eligible"]]
        total_observed_depths = sum(
            event["observable_target_horizon"] for event in group
        )
        horizon_minimum_budgets = [
            event["depth_events"][-1][
                "minimum_total_tree_node_budget_to_preserve_target_prefix"
            ]
            for event in group
            if event["depth_events"]
            and event["depth_events"][-1][
                "minimum_total_tree_node_budget_to_preserve_target_prefix"
            ]
            is not None
        ]
        row = {
            "task": task,
            "context_length": context,
            "policy": policy,
            "policy_variant": variant,
            "policy_class": group[0]["policy_class"],
            "requested_total_tree_node_budget": budget,
            "sample_count": len(sample_values),
            "round_count": len(group),
            "full_depth_eligible_round_count": sum(
                event["full_depth_evaluation_eligible"] for event in group
            ),
            "right_censored_round_count": sum(
                event["reference_right_censored"] for event in group
            ),
            "censoring_strata": {
                stratum: {
                    "round_count": len(stratum_events),
                    "covered_target_prefix_depth": _distribution(
                        [
                            event["covered_target_prefix_depth"]
                            for event in stratum_events
                        ]
                    ),
                    "extra_covered_target_tokens_vs_top1": _distribution(
                        [
                            event["extra_covered_target_tokens_vs_top1"]
                            for event in stratum_events
                        ]
                    ),
                }
                for stratum, stratum_events in (
                    (
                        "full_depth_eligible",
                        [
                            event
                            for event in group
                            if event["full_depth_evaluation_eligible"]
                        ],
                    ),
                    (
                        "right_censored",
                        [event for event in group if event["reference_right_censored"]],
                    ),
                )
            },
            "covered_target_prefix_depth": _distribution(
                [event["covered_target_prefix_depth"] for event in group]
            ),
            "baseline_accepted_draft_depth": _distribution(
                [event["baseline_accepted_draft_depth"] for event in group]
            ),
            "observable_target_horizon": _distribution(
                [event["observable_target_horizon"] for event in group]
            ),
            "extra_covered_target_tokens_vs_top1": _distribution(
                [event["extra_covered_target_tokens_vs_top1"] for event in group]
            ),
            "first_pruned_depth": _distribution(
                [
                    event["first_pruned_depth"]
                    for event in group
                    if event["first_pruned_depth"] is not None
                ]
            ),
            "first_candidate_unavailable_depth": _distribution(
                [
                    event["first_candidate_unavailable_depth"]
                    for event in group
                    if event["first_candidate_unavailable_depth"] is not None
                ]
            ),
            "off_top1_decision_count": _distribution(
                [event["off_top1_decision_count"] for event in group]
            ),
            "longest_consecutive_off_top1_span": _distribution(
                [event["longest_consecutive_off_top1_span"] for event in group]
            ),
            "target_rank_mean_per_round": _distribution(
                [
                    event["target_rank_mean"]
                    for event in group
                    if event["target_rank_mean"] is not None
                ]
            ),
            "target_rank_maximum_per_round": _distribution(
                [
                    event["target_rank_maximum"]
                    for event in group
                    if event["target_rank_maximum"] is not None
                ]
            ),
            "target_rank_recall": {
                f"at_{cutoff}": {
                    "depth_micro": (
                        sum(
                            depth["target_token_rank"] is not None
                            and depth["target_token_rank"] <= cutoff
                            for event in group
                            for depth in event["depth_events"]
                        )
                        / total_observed_depths
                        if total_observed_depths
                        else None
                    ),
                    "round_macro": _mean_or_none(
                        [
                            event[f"target_rank_recall_at_{cutoff}"]
                            for event in group
                            if event[f"target_rank_recall_at_{cutoff}"] is not None
                        ]
                    ),
                }
                for cutoff in (2, 4, 8, 16)
            },
            "minimum_survival_budget_at_observable_horizon": {
                **_distribution(horizon_minimum_budgets),
                "censored_above_maximum_count": sum(
                    bool(event["depth_events"])
                    and event["depth_events"][-1]["minimum_budget_censored"]
                    for event in group
                ),
            },
            "candidate_node_count": _distribution(
                [event["candidate_node_count"] for event in group]
            ),
            "round_micro": {
                "positive_improvement_fraction": mean(
                    event["positive_improvement_vs_top1"] for event in group
                ),
                "first_rejection_recovery_fraction": mean(
                    event["first_rejection_recovered"] for event in eligible_first
                )
                if eligible_first
                else None,
                "candidate_unavailable_fraction": sum(
                    event["candidate_unavailable_depth_count"] for event in group
                )
                / total_observed_depths
                if total_observed_depths
                else None,
                "candidate_available_but_pruned_fraction": sum(
                    event["candidate_available_but_pruned_depth_count"]
                    for event in group
                )
                / total_observed_depths
                if total_observed_depths
                else None,
                "score_rank_censored_fraction": mean(
                    event["score_rank_censored_round"] for event in group
                ),
            },
            "sample_macro": {
                "covered_target_prefix_depth_mean": mean(
                    mean(values) for values in sample_values.values()
                ),
                "extra_covered_target_tokens_vs_top1_mean": mean(
                    mean(
                        event["extra_covered_target_tokens_vs_top1"] for event in values
                    )
                    for values in sample_events.values()
                ),
                "positive_improvement_fraction": mean(
                    mean(event["positive_improvement_vs_top1"] for event in values)
                    for values in sample_events.values()
                ),
                "first_rejection_recovery_fraction": (
                    mean(
                        mean(
                            event["first_rejection_recovered"]
                            for event in values
                            if event["first_rejection_eligible"]
                        )
                        for values in sample_events.values()
                        if any(event["first_rejection_eligible"] for event in values)
                    )
                    if any(
                        event["first_rejection_eligible"]
                        for values in sample_events.values()
                        for event in values
                    )
                    else None
                ),
                "candidate_unavailable_fraction": _mean_or_none(
                    [
                        sum(
                            event["candidate_unavailable_depth_count"]
                            for event in values
                        )
                        / sum(event["observable_target_horizon"] for event in values)
                        for values in sample_events.values()
                        if sum(event["observable_target_horizon"] for event in values)
                    ]
                ),
                "candidate_available_but_pruned_fraction": _mean_or_none(
                    [
                        sum(
                            event["candidate_available_but_pruned_depth_count"]
                            for event in values
                        )
                        / sum(event["observable_target_horizon"] for event in values)
                        for values in sample_events.values()
                        if sum(event["observable_target_horizon"] for event in values)
                    ]
                ),
                "sample_is_independence_unit": True,
                "confidence_interval": None,
            },
            "focus_subset": {
                "oracle_improvable_round_count": sum(
                    event["oracle_improvable_round"] for event in group
                ),
                "oracle_improvable_but_policy_no_improvement_count": sum(
                    event["oracle_improvable_but_this_policy_no_improvement"]
                    for event in group
                ),
            },
            "depth_survival": {},
        }
        for depth in DEPTH_CHECKPOINTS:
            eligible = [
                event for event in group if event["observable_target_horizon"] >= depth
            ]
            row["depth_survival"][str(depth)] = {
                "eligible_round_count": len(eligible),
                "survived_round_count": sum(
                    event["covered_target_prefix_depth"] >= depth for event in eligible
                ),
                "round_micro_survival_fraction": mean(
                    event["covered_target_prefix_depth"] >= depth for event in eligible
                )
                if eligible
                else None,
            }
        rows.append(row)
    return {
        "schema_version": "1.0",
        "round_micro_treats_rounds_as_observations_not_independent_samples": True,
        "sample_macro_uses_sample_as_aggregation_unit": True,
        "confidence_intervals": None,
        "full_depth_and_right_censored_counts_reported_separately": True,
        "target_rank_recall_denominator": "all observable target depths; absent captured Top-K targets count as outside every cutoff",
        "groups": rows,
    }


def round_event_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse internal policy-budget rows into one JSONL record per round."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    identity_fields = (
        "task",
        "nominal_prompt_length",
        "actual_prompt_length",
        "sample_id",
        "source_sample_id",
        "source_text_sha256",
        "generation_repeat",
        "round_index",
    )
    for event in events:
        grouped[tuple(event[field] for field in identity_fields)].append(event)
    policy_depth_fields = {
        "target_prefix_survived",
        "first_pruned_depth",
        "pruning_reason",
        "policy_cutoff_normalized_score",
        "target_prefix_normalized_score",
        "target_prefix_score_gap_to_cutoff",
        "minimum_total_tree_node_budget_to_preserve_target_prefix",
        "minimum_budget_label",
        "minimum_budget_censored",
        "maximum_replayed_budget",
        "root_included_in_budget",
    }
    policy_fields = {
        "policy",
        "policy_variant",
        "policy_class",
        "policy_uses_target_information",
        "deployable_policy_is_target_blind",
        "policy_parameters",
        "requested_total_tree_node_budget",
        "root_included_in_budget",
        "total_node_count",
        "candidate_node_count",
        "leaf_count",
        "maximum_depth",
        "prefix_sharing_ratio",
        "covered_target_prefix_depth",
        "maximum_survived_target_depth",
        "top1_covered_target_prefix_depth",
        "extra_covered_target_tokens_vs_top1",
        "positive_improvement_vs_top1",
        "first_rejection_eligible",
        "first_rejection_recovered",
        "additional_candidate_nodes_vs_top1",
        "extra_covered_tokens_per_additional_candidate_node",
        "candidate_available_but_pruned_depth_count",
        "first_pruned_depth",
        "oracle_covered_target_prefix_depth",
        "oracle_improvable_round",
        "oracle_improvable_but_this_policy_no_improvement",
        "score_rank_censored_round",
    }
    rows = []
    for _, group in sorted(grouped.items()):
        first = group[0]
        base_depths = [
            {
                key: value
                for key, value in depth.items()
                if key not in policy_depth_fields
            }
            for depth in first["depth_events"]
        ]
        common = {
            key: value
            for key, value in first.items()
            if key not in policy_fields and key != "depth_events"
        }
        policy_replays = []
        for event in sorted(
            group,
            key=lambda item: (
                item["policy"],
                item["policy_variant"],
                item["requested_total_tree_node_budget"],
            ),
        ):
            policy_replays.append(
                {key: event[key] for key in policy_fields if key in event}
                | {
                    "depth_survival": [
                        {
                            "depth": depth["depth"],
                            **{
                                key: depth[key]
                                for key in policy_depth_fields
                                if key in depth
                            },
                        }
                        for depth in event["depth_events"]
                    ]
                }
            )
        rows.append(
            {
                **common,
                "depth_events": base_depths,
                "policy_replays": policy_replays,
            }
        )
    return rows


def budget_frontier_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    groups_by_variant: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in summary["groups"]:
        groups_by_variant[
            (row["task"], row["context_length"], row["policy"], row["policy_variant"])
        ].append(row)
    for _, groups in sorted(groups_by_variant.items()):
        groups.sort(key=lambda row: row["requested_total_tree_node_budget"])
        previous = None
        for row in groups:
            current_extra = row["extra_covered_target_tokens_vs_top1"]["sum"]
            current_nodes = row["candidate_node_count"]["sum"]
            marginal_extra = current_nodes_delta = efficiency = None
            transition = None
            if previous is not None:
                transition = (
                    f"{previous['requested_total_tree_node_budget']}"
                    f"->{row['requested_total_tree_node_budget']}"
                )
                marginal_extra = (
                    current_extra
                    - previous["extra_covered_target_tokens_vs_top1"]["sum"]
                )
                current_nodes_delta = (
                    current_nodes - previous["candidate_node_count"]["sum"]
                )
                efficiency = (
                    marginal_extra / current_nodes_delta
                    if current_nodes_delta
                    else None
                )
            rows.append(
                {
                    "task": row["task"],
                    "context_length": row["context_length"],
                    "policy": row["policy"],
                    "policy_variant": row["policy_variant"],
                    "node_budget": row["requested_total_tree_node_budget"],
                    "round_count": row["round_count"],
                    "covered_depth_mean": row["covered_target_prefix_depth"]["mean"],
                    "absolute_extra_covered_tokens": current_extra,
                    "positive_improvement_fraction": row["round_micro"][
                        "positive_improvement_fraction"
                    ],
                    "first_rejection_recovery_fraction": row["round_micro"][
                        "first_rejection_recovery_fraction"
                    ],
                    "candidate_nodes_total": current_nodes,
                    "extra_tokens_per_additional_candidate_node": (
                        current_extra / max(1, current_nodes - 15 * row["round_count"])
                    ),
                    "candidate_unavailable_fraction": row["round_micro"][
                        "candidate_unavailable_fraction"
                    ],
                    "candidate_available_but_pruned_fraction": row["round_micro"][
                        "candidate_available_but_pruned_fraction"
                    ],
                    "score_rank_censored_fraction": row["round_micro"][
                        "score_rank_censored_fraction"
                    ],
                    "marginal_transition": transition,
                    "marginal_extra_covered_tokens": marginal_extra,
                    "marginal_additional_candidate_nodes": current_nodes_delta,
                    "marginal_extra_tokens_per_added_node": efficiency,
                    **{
                        f"survival_depth_{depth}_fraction": row["depth_survival"][
                            str(depth)
                        ]["round_micro_survival_fraction"]
                        for depth in DEPTH_CHECKPOINTS
                    },
                }
            )
            previous = row
    return rows


def long_context_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    requested = {("gov_report", 32768), ("qasper", 16384), ("hotpotqa", 16384)}
    candidates = [
        row
        for row in summary["groups"]
        if (row["task"], int(row["context_length"])) in requested
    ]
    output = []
    for task, context in sorted(requested):
        current = [
            row
            for row in candidates
            if row["task"] == task and int(row["context_length"]) == context
        ]
        selected = [
            row
            for row in current
            if row["policy"] == "top1_chain"
            or (
                row["policy"] == "score_beam"
                and row["requested_total_tree_node_budget"] in (32, 64, 128, 256)
            )
            or (
                row["policy"] == "oracle_prefix_closed"
                and row["requested_total_tree_node_budget"] == 256
            )
        ]
        layerwise = [row for row in current if row["policy"] == "layerwise_beam"]
        if layerwise:
            best = max(
                layerwise,
                key=lambda row: (
                    row["extra_covered_target_tokens_vs_top1"]["mean"],
                    -row["requested_total_tree_node_budget"],
                    row["policy_variant"],
                ),
            )
            selected.append(best)
        for row in selected:
            previous = next(
                (
                    candidate
                    for candidate in current
                    if candidate["policy_variant"] == row["policy_variant"]
                    and candidate["requested_total_tree_node_budget"]
                    < row["requested_total_tree_node_budget"]
                    and not any(
                        other["policy_variant"] == row["policy_variant"]
                        and candidate["requested_total_tree_node_budget"]
                        < other["requested_total_tree_node_budget"]
                        < row["requested_total_tree_node_budget"]
                        for other in current
                    )
                ),
                None,
            )
            marginal_extra = marginal_nodes = None
            if previous is not None:
                marginal_extra = (
                    row["extra_covered_target_tokens_vs_top1"]["sum"]
                    - previous["extra_covered_target_tokens_vs_top1"]["sum"]
                )
                marginal_nodes = (
                    row["candidate_node_count"]["sum"]
                    - previous["candidate_node_count"]["sum"]
                )
            output.append(
                {
                    "task": task,
                    "context_length": context,
                    "policy": row["policy"],
                    "policy_variant": row["policy_variant"],
                    "node_budget": row["requested_total_tree_node_budget"],
                    "layerwise_configuration_selected_descriptively_on_same_data": row[
                        "policy"
                    ]
                    == "layerwise_beam",
                    "covered_depth_mean": row["covered_target_prefix_depth"]["mean"],
                    "absolute_extra_covered_tokens": row[
                        "extra_covered_target_tokens_vs_top1"
                    ]["sum"],
                    "first_pruned_depth_median": row["first_pruned_depth"]["median"],
                    "candidate_unavailable_fraction": row["round_micro"][
                        "candidate_unavailable_fraction"
                    ],
                    "available_but_pruned_fraction": row["round_micro"][
                        "candidate_available_but_pruned_fraction"
                    ],
                    "minimum_survival_budget_at_observable_horizon_median": row[
                        "minimum_survival_budget_at_observable_horizon"
                    ]["median"],
                    "minimum_survival_budget_censored_count": row[
                        "minimum_survival_budget_at_observable_horizon"
                    ]["censored_above_maximum_count"],
                    "extra_tokens_per_additional_candidate_node": (
                        row["extra_covered_target_tokens_vs_top1"]["sum"]
                        / max(
                            1,
                            row["candidate_node_count"]["sum"]
                            - 15 * row["round_count"],
                        )
                    ),
                    "marginal_from_previous_budget": (
                        f"{previous['requested_total_tree_node_budget']}->{row['requested_total_tree_node_budget']}"
                        if previous is not None
                        else None
                    ),
                    "marginal_extra_covered_tokens": marginal_extra,
                    "marginal_extra_tokens_per_added_node": (
                        marginal_extra / marginal_nodes if marginal_nodes else None
                    ),
                    "oracle_improvable_but_no_improvement_count": row["focus_subset"][
                        "oracle_improvable_but_policy_no_improvement_count"
                    ],
                }
            )
    return output
