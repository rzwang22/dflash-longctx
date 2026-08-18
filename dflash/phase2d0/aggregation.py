"""Gap decomposition and deterministic research-question summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from .analysis import GAP_CLASS_PRIORITY


def _fraction_metrics(events: list[dict[str, Any]], gap_class: str) -> dict[str, Any]:
    raw = len(events)
    eligible_events = [event for event in events if event["classification_eligible"]]
    eligible = len(eligible_events)
    censored = raw - eligible
    count = sum(event["gap_class"] == gap_class for event in events)
    is_censored_class = gap_class == "right_censored_or_terminal"
    eligible_count = sum(event["gap_class"] == gap_class for event in eligible_events)
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        source_id = event.get("source_sample_id", event["sample_id"])
        by_sample[f"{event['task']}::{source_id}"].append(event)
    macro_all = mean(
        sum(event["gap_class"] == gap_class for event in rows) / len(rows)
        for rows in by_sample.values()
    )
    eligible_sample_fractions = []
    if not is_censored_class:
        for rows in by_sample.values():
            sample_eligible = [row for row in rows if row["classification_eligible"]]
            if sample_eligible:
                eligible_sample_fractions.append(
                    sum(row["gap_class"] == gap_class for row in sample_eligible)
                    / len(sample_eligible)
                )
    eligible_fraction = (
        None if is_censored_class or not eligible else eligible_count / eligible
    )
    eligible_reason = (
        "class_excluded_from_eligible_denominator"
        if is_censored_class
        else "no_eligible_configuration_rounds"
        if not eligible
        else None
    )
    return {
        "gap_class_round_count": count,
        "raw_round_count": raw,
        "eligible_round_count": eligible,
        "right_censored_or_terminal_count": censored,
        "fraction_over_all_rounds": count / raw,
        "fraction_among_eligible_rounds": eligible_fraction,
        "fraction_among_eligible_rounds_unavailable_reason": eligible_reason,
        "censoring_fraction": censored / raw,
        "micro_fraction_over_all_rounds": count / raw,
        "micro_fraction_among_eligible_rounds": eligible_fraction,
        "macro_fraction_over_all_rounds": macro_all,
        "macro_fraction_among_eligible_rounds": (
            mean(eligible_sample_fractions) if eligible_sample_fractions else None
        ),
        "macro_fraction_among_eligible_rounds_unavailable_reason": (
            None
            if eligible_sample_fractions
            else eligible_reason or "no_sample_with_eligible_configuration_rounds"
        ),
        "fraction_denominator_name": (
            "all_configuration_rounds"
            if is_censored_class
            else "eligible_configuration_rounds"
        ),
        "fraction_denominator_count": raw if is_censored_class else eligible,
        "count_unit": "configuration_round_at_fixed_node_budget",
    }


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
            rows.append(
                {
                    "schema_version": "2.0",
                    "scope": scope,
                    **identity,
                    "gap_class": gap_class,
                    **_fraction_metrics(events, gap_class),
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
    cross_task_events: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        cross_task_events[(int(event["top_m"]), int(event["node_budget"]))].append(
            event
        )
    for (top_m, budget), grouped_events in sorted(cross_task_events.items()):
        for gap_class in GAP_CLASS_PRIORITY:
            metrics = _fraction_metrics(grouped_events, gap_class)
            matching_task_rows = [
                row
                for row in task_rows
                if row["top_m"] == top_m
                and row["node_budget"] == budget
                and row["gap_class"] == gap_class
            ]
            eligible_task_values = [
                row["fraction_among_eligible_rounds"]
                for row in matching_task_rows
                if row["fraction_among_eligible_rounds"] is not None
            ]
            metrics["macro_fraction_over_all_rounds"] = mean(
                row["fraction_over_all_rounds"] for row in matching_task_rows
            )
            metrics["macro_fraction_among_eligible_rounds"] = (
                mean(eligible_task_values) if eligible_task_values else None
            )
            cross_task_rows.append(
                {
                    "schema_version": "2.0",
                    "scope": "cross_task_macro",
                    "top_m": top_m,
                    "node_budget": budget,
                    "gap_class": gap_class,
                    "task_count": len(matching_task_rows),
                    **metrics,
                }
            )
    return {
        "schema_version": "2.0",
        "phase": "2D.0",
        "gap_class_priority": list(GAP_CLASS_PRIORITY),
        "event_count": len(events),
        "configuration_round_count": len(events),
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
        "raw_physical_round_count": len(
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
        "node_budgets_are_separate_configuration_strata": True,
        "cross_node_budget_counts_are_configuration_round_counts": True,
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
    core_physical_round_count = len(
        {
            (
                event["task"],
                event["sample_id"],
                event["nominal_prompt_length"],
                event["generation_repeat"],
                event["round_index"],
            )
            for event in core
        }
    )
    core_shares_by_budget = []
    for node_budget in sorted({event["node_budget"] for event in core}):
        budget_events = [event for event in core if event["node_budget"] == node_budget]
        budget_eligible = [
            event for event in budget_events if event["classification_eligible"]
        ]
        core_shares_by_budget.append(
            {
                "node_budget": node_budget,
                "raw_physical_round_count": len(budget_events),
                "eligible_physical_round_count": len(budget_eligible),
                "right_censored_or_terminal_count": (
                    len(budget_events) - len(budget_eligible)
                ),
                "shares_among_eligible_physical_rounds": {
                    name: _share(budget_events, classes)
                    for name, classes in (
                        (
                            "candidate_availability_failure",
                            {"candidate_availability_failure"},
                        ),
                        ("selection_failure", {"selection_failure"}),
                        ("budget_failure", {"budget_failure"}),
                        (
                            "dependency_realization_opportunity",
                            {"dependency_realization_opportunity"},
                        ),
                        (
                            "possible_dependency_opportunity",
                            {"possible_dependency_opportunity"},
                        ),
                    )
                },
            }
        )
    censoring_by_task_and_budget = []
    for task, node_budget in sorted(
        {(event["task"], event["node_budget"]) for event in core}
    ):
        task_budget_events = [
            event
            for event in core
            if event["task"] == task and event["node_budget"] == node_budget
        ]
        censored_count = sum(
            not event["classification_eligible"] for event in task_budget_events
        )
        censoring_by_task_and_budget.append(
            {
                "task": task,
                "node_budget": node_budget,
                "raw_physical_round_count": len(task_budget_events),
                "right_censored_or_terminal_count": censored_count,
                "censoring_fraction": censored_count / len(task_budget_events),
                "mechanism_class_denominator_excludes_censored_rounds": True,
            }
        )
    return {
        "schema_version": "2.0",
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
            "selection_failure_share_among_eligible_configuration_rounds": shares[
                "selection"
            ],
            "budget_failure_share_among_eligible_configuration_rounds": shares[
                "budget"
            ],
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
            "dependency_realization_opportunity_share_among_eligible_configuration_rounds": shares[
                "dependency"
            ],
            "possible_dependency_opportunity_share_among_eligible_configuration_rounds": shares[
                "possible_dependency"
            ],
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
        "raw_physical_round_count": core_physical_round_count,
        "combined_node_budget_count_unit": "configuration_round",
        "core_shares_by_node_budget": core_shares_by_budget,
        "censoring_by_task_and_node_budget": censoring_by_task_and_budget,
        "core_share_denominator_name": "eligible_configuration_rounds",
        "task_count": task_count,
        "gap_class_counts": dict(sorted(class_counts.items())),
        "core_configuration": {"top_m": 16, "node_budgets": [16, 32]},
        "decision": decision,
        "no_dependency_dominance_claim_without_capture_results": True,
    }
