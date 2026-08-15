"""Aggregation for offline bounded-tree coverage events."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from dflash.phase1b.metrics import distribution

DEPTHS = (1, 2, 4, 8, 15)


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _sample_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event.get("task"),
        event.get("sample_id"),
        event.get("nominal_prompt_length"),
    )


def summarize_tree_events(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        raise ValueError("cannot summarize an empty tree-event group")
    policy_classes = {event["policy_class"] for event in events}
    if len(policy_classes) != 1:
        raise ValueError("oracle and deployable events must not be mixed in one group")

    depth_recall = {}
    for depth in DEPTHS:
        eligible = [
            event for event in events if event["depth_recall"][str(depth)]["eligible"]
        ]
        covered = sum(
            bool(event["depth_recall"][str(depth)]["covered"])
            for event in eligible
        )
        depth_recall[str(depth)] = {
            "eligible_round_count": len(eligible),
            "covered_round_count": covered,
            "round_micro_recall": _ratio(covered, len(eligible)),
        }

    by_sample: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_sample[_sample_key(event)].append(event)
    sample_values = []
    for key, sample_events in by_sample.items():
        sample_values.append(
            {
                "sample_key": list(key),
                "round_count": len(sample_events),
                "mean_covered_target_prefix_depth": statistics.fmean(
                    event["covered_target_prefix_depth"] for event in sample_events
                ),
                "mean_extra_covered_depth_vs_top1": statistics.fmean(
                    event["extra_covered_depth_vs_top1"] for event in sample_events
                ),
                "positive_improvement_fraction": _ratio(
                    sum(event["extra_covered_depth_vs_top1"] > 0 for event in sample_events),
                    len(sample_events),
                ),
            }
        )

    full_depth_eligible = [
        event for event in events if event["full_depth_evaluation_eligible"]
    ]
    frontier_eligible = [
        event for event in events if event["first_rejection_recovery_eligible"]
    ]
    oracle_gap_values = [
        event["deployable_oracle_coverage_gap"]
        for event in events
        if event["deployable_oracle_coverage_gap"] is not None
    ]
    return {
        "policy_class": next(iter(policy_classes)),
        "round_count": len(events),
        "sample_count": len(by_sample),
        "terminal_round_count": sum(event["terminal_round"] for event in events),
        "right_censored_round_count": sum(
            event["reference_right_censored"] for event in events
        ),
        "full_depth_eligible_round_count": len(full_depth_eligible),
        "full_depth_excluded_censored_round_count": len(events)
        - len(full_depth_eligible),
        "covered_target_prefix_depth": distribution(
            event["covered_target_prefix_depth"] for event in events
        ),
        "extra_covered_depth_vs_top1": distribution(
            event["extra_covered_depth_vs_top1"] for event in events
        ),
        "committed_tokens_upper_bound": distribution(
            event["committed_tokens_upper_bound"] for event in events
        ),
        "candidate_node_count": distribution(
            event["candidate_node_count"] for event in events
        ),
        "leaf_count": distribution(event["leaf_count"] for event in events),
        "maximum_depth": distribution(event["maximum_depth"] for event in events),
        "prefix_sharing_ratio": distribution(
            event["prefix_sharing_ratio"] for event in events
        ),
        "target_path_candidate_recall": distribution(
            event["target_path_candidate_recall"] for event in events
        ),
        "covered_target_tokens_per_candidate_node": distribution(
            event["covered_target_tokens_per_candidate_node"] for event in events
        ),
        "deployable_oracle_coverage_gap": distribution(oracle_gap_values),
        "round_micro": {
            "mean_covered_target_prefix_depth": statistics.fmean(
                event["covered_target_prefix_depth"] for event in events
            ),
            "mean_extra_covered_depth_vs_top1": statistics.fmean(
                event["extra_covered_depth_vs_top1"] for event in events
            ),
            "positive_improvement_fraction": _ratio(
                sum(event["extra_covered_depth_vs_top1"] > 0 for event in events),
                len(events),
            ),
            "first_rejection_recovery_fraction": _ratio(
                sum(bool(event["first_rejection_recovered"]) for event in frontier_eligible),
                len(frontier_eligible),
            ),
            "full_depth_coverage_fraction": _ratio(
                sum(bool(event["full_depth_covered"]) for event in full_depth_eligible),
                len(full_depth_eligible),
            ),
            "oracle_improvable_but_deployable_not_fraction": _ratio(
                sum(
                    bool(event["oracle_improvable_but_deployable_not"])
                    for event in events
                    if event["oracle_improvable_but_deployable_not"] is not None
                ),
                sum(
                    event["oracle_improvable_but_deployable_not"] is not None
                    for event in events
                ),
            ),
        },
        "sample_macro": {
            "mean_covered_target_prefix_depth": (
                statistics.fmean(
                    item["mean_covered_target_prefix_depth"] for item in sample_values
                )
                if sample_values
                else None
            ),
            "mean_extra_covered_depth_vs_top1": (
                statistics.fmean(
                    item["mean_extra_covered_depth_vs_top1"] for item in sample_values
                )
                if sample_values
                else None
            ),
            "mean_positive_improvement_fraction": (
                statistics.fmean(
                    item["positive_improvement_fraction"] for item in sample_values
                )
                if sample_values
                else None
            ),
            "covered_depth_per_sample_mean_distribution": distribution(
                item["mean_covered_target_prefix_depth"] for item in sample_values
            ),
            "extra_depth_per_sample_mean_distribution": distribution(
                item["mean_extra_covered_depth_vs_top1"] for item in sample_values
            ),
            "positive_improvement_per_sample_fraction_distribution": distribution(
                item["positive_improvement_fraction"] for item in sample_values
            ),
            "per_sample": sample_values,
        },
        "depth_recall": depth_recall,
        "terminal_policy": {
            "prefix_metrics_include_observable_censored_prefixes": True,
            "full_depth_metrics_exclude_right_censored_rounds": True,
            "no_unobserved_tokens_imputed": True,
        },
        "confidence_intervals": None,
        "confidence_interval_reason": (
            "No round-independent confidence interval; rounds share sample state"
        ),
    }


def _group_events(
    events: Sequence[dict[str, Any]], fields: tuple[str, ...]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[tuple(event.get(field) for field in fields)].append(event)
    return grouped


def aggregate_tree_events(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate by task/context/budget/policy with explicit micro/macro scopes."""

    detailed_fields = (
        "task",
        "nominal_prompt_length",
        "requested_total_tree_node_budget",
        "policy",
        "policy_variant",
    )
    overall_fields = (
        "requested_total_tree_node_budget",
        "policy",
        "policy_variant",
    )

    def render(fields: tuple[str, ...]) -> dict[str, Any]:
        output = {}
        for key, group in sorted(
            _group_events(events, fields).items(), key=lambda item: str(item[0])
        ):
            label = ":".join(str(value) for value in key)
            output[label] = {
                **dict(zip(fields, key)),
                **summarize_tree_events(group),
            }
        return output

    return {
        "schema_version": "1.0",
        "event_count": len(events),
        "by_task_context_budget_policy": render(detailed_fields),
        "overall_by_budget_policy": render(overall_fields),
        "interpretation": {
            "offline_coverage_feasibility_only": True,
            "actual_tree_acceptance_measured": False,
            "latency_measured": False,
            "speedup_claim_supported": False,
            "committed_tokens_are_coverage_upper_bounds": True,
            "oracle_uses_target_information": True,
            "deployable_policies_use_target_information": False,
        },
    }
