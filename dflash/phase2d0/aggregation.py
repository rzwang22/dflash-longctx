"""Gap decomposition and deterministic research-question summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from .analysis import GAP_CLASS_PRIORITY


def _fractions(
    events: list[dict[str, Any]], gap_class: str
) -> tuple[int, int, int, float | None, float | None]:
    total = len(events)
    eligible = sum(event["classification_eligible"] for event in events)
    count = sum(event["gap_class"] == gap_class for event in events)
    micro = count / total if total else None
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_sample[str(event["sample_id"])].append(event)
    macro = (
        mean(
            sum(event["gap_class"] == gap_class for event in rows) / len(rows)
            for rows in by_sample.values()
        )
        if by_sample
        else None
    )
    return count, total, eligible, micro, macro


def _group_rows(
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]],
    *,
    scope: str,
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = []
    for key, events in sorted(grouped.items()):
        identity = dict(zip(key_fields, key))
        for gap_class in GAP_CLASS_PRIORITY:
            count, total, eligible, micro, macro = _fractions(events, gap_class)
            rows.append(
                {
                    "schema_version": "1.0",
                    "scope": scope,
                    **identity,
                    "gap_class": gap_class,
                    "gap_class_round_count": count,
                    "round_count": total,
                    "eligible_round_count": eligible,
                    "terminal_or_censored_excluded_count": total - eligible,
                    "fraction": micro,
                    "micro_fraction": micro,
                    "macro_fraction": macro,
                    "fraction_denominator": "all mutually-exclusive configuration-round events",
                    "eligible_fraction_denominator": (
                        "non-terminal and non-right-censored configuration-round events"
                    ),
                }
            )
    return rows


def aggregate_gap_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        raise ValueError("cannot aggregate empty opportunity events")
    invalid = {event["gap_class"] for event in events} - set(GAP_CLASS_PRIORITY)
    if invalid:
        raise ValueError(f"unknown gap classes: {sorted(invalid)}")
    task_context: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    task: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        task_context[
            (
                event["task"],
                int(event["nominal_prompt_length"]),
                int(event["top_m"]),
                int(event["node_budget"]),
            )
        ].append(event)
        task[(event["task"], int(event["top_m"]), int(event["node_budget"]))].append(
            event
        )
    context_rows = _group_rows(
        task_context,
        scope="task_context",
        key_fields=("task", "context_length", "top_m", "node_budget"),
    )
    task_rows = _group_rows(
        task,
        scope="task_micro_and_sample_macro",
        key_fields=("task", "top_m", "node_budget"),
    )

    cross_task_rows = []
    grouped_task_rows: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in task_rows:
        grouped_task_rows[
            (int(row["top_m"]), int(row["node_budget"]), row["gap_class"])
        ].append(row)
    for (top_m, budget, gap_class), rows in sorted(grouped_task_rows.items()):
        cross_task_rows.append(
            {
                "schema_version": "1.0",
                "scope": "cross_task_macro",
                "top_m": top_m,
                "node_budget": budget,
                "gap_class": gap_class,
                "task_count": len(rows),
                "gap_class_round_count": sum(
                    row["gap_class_round_count"] for row in rows
                ),
                "round_count": sum(row["round_count"] for row in rows),
                "eligible_round_count": sum(
                    row["eligible_round_count"] for row in rows
                ),
                "terminal_or_censored_excluded_count": sum(
                    row["terminal_or_censored_excluded_count"] for row in rows
                ),
                "fraction": mean(row["micro_fraction"] for row in rows),
                "micro_fraction": (
                    sum(row["gap_class_round_count"] for row in rows)
                    / sum(row["round_count"] for row in rows)
                ),
                "macro_fraction": mean(row["macro_fraction"] for row in rows),
                "fraction_denominator": "tasks",
                "eligible_fraction_denominator": "tasks",
            }
        )
    return {
        "schema_version": "1.0",
        "phase": "2D.0",
        "gap_class_priority": list(GAP_CLASS_PRIORITY),
        "event_count": len(events),
        "raw_round_count": len(
            {
                (
                    event["task"],
                    event["sample_id"],
                    event["nominal_prompt_length"],
                    event["generation_repeat"],
                    event["round_index"],
                )
                for event in events
            }
        ),
        "task_context": context_rows,
        "task_level": task_rows,
        "cross_task_macro": cross_task_rows,
        "round_micro_treats_rounds_as_observations": True,
        "sample_macro_uses_sample_as_independence_unit": True,
        "no_cross_task_extrapolation": True,
    }


def availability_by_depth(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated = {}
    for event in events:
        key = (
            event["task"],
            event["sample_id"],
            event["generation_repeat"],
            event["round_index"],
            event["top_m"],
        )
        deduplicated.setdefault(key, event)
    grouped: dict[tuple[Any, ...], list[tuple[dict[str, Any], int]]] = defaultdict(list)
    for event in deduplicated.values():
        for index, available in enumerate(event["target_membership_top_m"], 1):
            grouped[
                (
                    event["task"],
                    event["nominal_prompt_length"],
                    event["top_m"],
                    index,
                )
            ].append((event, int(available)))
    rows = []
    for (task, context, top_m, depth), values in sorted(grouped.items()):
        observable = len(values)
        available = sum(value for _, value in values)
        prefix_available = sum(
            event["candidate_availability_prefix"] >= depth for event, _ in values
        )
        rows.append(
            {
                "schema_version": "1.0",
                "task": task,
                "context_length": context,
                "top_m": top_m,
                "depth": depth,
                "observable_round_count": observable,
                "target_token_available_count": available,
                "target_token_available_fraction": available / observable,
                "target_prefix_available_count": prefix_available,
                "target_prefix_available_fraction": prefix_available / observable,
                "right_censored_round_count": sum(
                    event["reference_right_censored"] for event, _ in values
                ),
            }
        )
    return rows


def aggregate_frontier_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = {}
    for row in rows:
        key = (
            row["task"],
            row["sample_id"],
            row["generation_repeat"],
            row["round_index"],
            row["top_m"],
            row["policy"],
            row["policy_variant"],
            row["node_budget"],
        )
        unique.setdefault(key, row)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in unique.values():
        grouped[
            (
                row["task"],
                row["nominal_prompt_length"],
                row["top_m"],
                row["policy"],
                row["policy_variant"],
                row["node_budget"],
            )
        ].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        task, context, top_m, policy, variant, budget = key
        gains = [
            row["coverage_gain_per_added_node"]
            for row in values
            if row["coverage_gain_per_added_node"] is not None
        ]
        output.append(
            {
                "schema_version": "1.0",
                "task": task,
                "context_length": context,
                "top_m": top_m,
                "policy": policy,
                "policy_variant": variant,
                "node_budget": budget,
                "round_count": len(values),
                "mean_verified_nodes": mean(row["verified_nodes"] for row in values),
                "mean_coverage_depth": mean(row["coverage_depth"] for row in values),
                "coverage_depth_per_verified_node": mean(
                    row["coverage_depth_per_verified_node"] for row in values
                ),
                "mean_coverage_gain_per_added_node": mean(gains) if gains else None,
                "mean_coverage_gain_per_added_node_unavailable_reason": (
                    None if gains else "no_evaluable_frontier_transition"
                ),
                "interpretation": "offline_verification_efficiency_proxy",
            }
        )
    return output


def dependency_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = {
        "dependency_realization_opportunity",
        "possible_dependency_opportunity",
    }
    fields = (
        "schema_version",
        "task",
        "sample_id",
        "source_sample_id",
        "nominal_prompt_length",
        "generation_repeat",
        "round_index",
        "top_m",
        "node_budget",
        "gap_class",
        "critical_depth",
        "candidate_availability_prefix",
        "best_static_coverage_depth",
        "oracle_coverage_depth",
        "oracle_minus_static_coverage",
        "oracle_required_nodes",
        "oracle_required_nodes_unavailable_reason",
        "best_static_required_nodes",
        "best_static_required_nodes_unavailable_reason",
        "static_oracle_budget_ratio",
        "static_oracle_budget_ratio_unavailable_reason",
        "target_prefix_score_rank",
        "target_prefix_score_rank_lower_bound",
        "target_prefix_score_rank_censored",
        "first_pruning_depth",
        "first_pruning_depth_unavailable_reason",
        "shallow_target_prefix_entered_tree",
        "correct_path_pruned_at_depth",
        "correct_path_pruned_at_depth_unavailable_reason",
        "significant_rank_inflation",
        "significant_budget_inflation",
    )
    return [
        {field: event.get(field) for field in fields}
        for event in events
        if event["gap_class"] in selected
    ]


def _share(events: list[dict[str, Any]], classes: set[str]) -> float | None:
    eligible = [event for event in events if event["classification_eligible"]]
    return (
        sum(event["gap_class"] in classes for event in eligible) / len(eligible)
        if eligible
        else None
    )


def _difference(maximum: float | None, minimum: float | None) -> float | None:
    if maximum is None or minimum is None:
        return None
    return maximum - minimum


def _candidate_available_failure_events(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event["classification_eligible"]
        and event["critical_depth"] is not None
        and event["candidate_availability_prefix"] >= event["critical_depth"]
    ]


def build_research_summary(
    events: list[dict[str, Any]], signal_results: list[dict[str, Any]]
) -> dict[str, Any]:
    core = [event for event in events if event["top_m"] == 16]
    task_trends = []
    task_budget_keys = sorted(
        {(event["task"], int(event["node_budget"])) for event in core}
    )
    for task, node_budget in task_budget_keys:
        task_events = [
            event
            for event in core
            if event["task"] == task and event["node_budget"] == node_budget
        ]
        contexts = sorted({event["nominal_prompt_length"] for event in task_events})
        if len(contexts) < 2:
            task_trends.append(
                {
                    "task": task,
                    "node_budget": node_budget,
                    "minimum_context_length": contexts[0] if contexts else None,
                    "maximum_context_length": contexts[-1] if contexts else None,
                    "availability_failure_fraction_change": None,
                    "static_path_failure_fraction_change": None,
                    "unavailable_reason": "fewer_than_two_context_lengths",
                }
            )
            continue
        minimum, maximum = contexts[0], contexts[-1]
        minimum_events = [
            event for event in task_events if event["nominal_prompt_length"] == minimum
        ]
        maximum_events = [
            event for event in task_events if event["nominal_prompt_length"] == maximum
        ]
        availability_classes = {"candidate_availability_failure"}
        static_failure_classes = {
            "budget_failure",
            "dependency_realization_opportunity",
            "possible_dependency_opportunity",
            "unresolved",
        }
        minimum_candidate_available = _candidate_available_failure_events(
            minimum_events
        )
        maximum_candidate_available = _candidate_available_failure_events(
            maximum_events
        )
        task_trends.append(
            {
                "task": task,
                "node_budget": node_budget,
                "minimum_context_length": minimum,
                "maximum_context_length": maximum,
                "availability_failure_fraction_change": _difference(
                    _share(maximum_events, availability_classes),
                    _share(minimum_events, availability_classes),
                ),
                "static_path_failure_fraction_change": _difference(
                    _share(maximum_candidate_available, static_failure_classes),
                    _share(minimum_candidate_available, static_failure_classes),
                ),
                "minimum_context_candidate_available_failure_round_count": len(
                    minimum_candidate_available
                ),
                "maximum_context_candidate_available_failure_round_count": len(
                    maximum_candidate_available
                ),
                "unavailable_reason": None,
            }
        )

    micro_signal = [
        row
        for row in signal_results
        if row["scope"] == "micro"
        and row["label"] == "dependency_realization_opportunity"
        and row["auroc"] is not None
    ]
    best_signal = max(
        micro_signal,
        key=lambda row: (row["auroc"], row["auprc"], row["feature"]),
        default=None,
    )
    eligible = [event for event in core if event["classification_eligible"]]
    candidate_available_failures = _candidate_available_failure_events(core)
    shares = {
        "candidate_availability": _share(core, {"candidate_availability_failure"}),
        "selection": _share(core, {"selection_failure"}),
        "budget": _share(core, {"budget_failure"}),
        "dependency": _share(core, {"dependency_realization_opportunity"}),
        "possible_dependency": _share(core, {"possible_dependency_opportunity"}),
    }
    decision_rules = {
        "minimum_eligible_configuration_rounds": 100,
        "minimum_tasks": 2,
        "dominant_gap_share": 0.5,
        "dependency_share_to_continue": 0.15,
        "mixed_component_share": 0.15,
        "minimum_predictive_auroc": 0.65,
        "minimum_auprc_lift_over_prevalence": 0.05,
    }
    task_count = len({event["task"] for event in eligible})
    predictor_ready = bool(
        best_signal
        and not best_signal["result_unstable"]
        and best_signal["auroc"] >= decision_rules["minimum_predictive_auroc"]
        and best_signal["auprc"]
        >= best_signal["positive_rate"]
        + decision_rules["minimum_auprc_lift_over_prevalence"]
    )
    if (
        len(eligible) < decision_rules["minimum_eligible_configuration_rounds"]
        or task_count < decision_rules["minimum_tasks"]
    ):
        decision = "insufficient_evidence"
    elif shares["candidate_availability"] >= decision_rules["dominant_gap_share"]:
        decision = "prioritize_candidate_generation"
    elif shares["selection"] + shares["budget"] >= decision_rules["dominant_gap_share"]:
        decision = "prioritize_selector"
    elif (
        shares["dependency"] >= decision_rules["dependency_share_to_continue"]
        and predictor_ready
    ):
        decision = "continue_dependency_refinement"
    elif (
        sum(
            share >= decision_rules["mixed_component_share"]
            for share in shares.values()
        )
        >= 2
    ):
        decision = "mixed_problem"
    else:
        decision = "insufficient_evidence"

    class_counts = Counter(event["gap_class"] for event in eligible)
    return {
        "schema_version": "1.0",
        "phase": "2D.0",
        "rq1": {
            "question": "Does target-token Top-M availability decline with context length?",
            "task_specific_context_trends": task_trends,
            "cross_task_extrapolation_performed": False,
        },
        "rq2": {
            "question": "Does fixed-budget static target-prefix survival worsen when candidates remain available?",
            "task_specific_context_trends": task_trends,
        },
        "rq3": {
            "question": "Can selector changes or modest budget increases explain static failures?",
            "selection_failure_share": shares["selection"],
            "budget_failure_share": shares["budget"],
            "candidate_available_baseline_failure_count": len(
                candidate_available_failures
            ),
            "selector_or_modest_budget_share_among_candidate_available_failures": (
                _share(
                    candidate_available_failures,
                    {"selection_failure", "budget_failure"},
                )
            ),
        },
        "rq4": {
            "question": "What share meets the conservative dependency opportunity definition?",
            "dependency_realization_opportunity_share": shares["dependency"],
            "possible_dependency_opportunity_share": shares["possible_dependency"],
            "dependency_share_among_candidate_available_failures": _share(
                candidate_available_failures,
                {"dependency_realization_opportunity"},
            ),
        },
        "rq5": {
            "question": "Can captured cheap draft-side signals predict dependency-critical cases?",
            "best_single_feature_result": best_signal,
            "predictive_evidence_ready": predictor_ready,
        },
        "rq6": {
            "question": "Should Phase 2D.1 continue?",
            "decision": decision,
            "decision_rules": decision_rules,
        },
        "eligible_configuration_round_count": len(eligible),
        "task_count": task_count,
        "gap_class_counts": dict(sorted(class_counts.items())),
        "core_configuration": {"top_m": 16, "node_budgets": [16, 32]},
        "decision": decision,
        "no_dependency_dominance_claim_without_capture_results": True,
    }
