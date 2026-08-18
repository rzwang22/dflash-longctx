"""Leakage-safe fixed-policy and decision-stump evaluation."""

from __future__ import annotations

import itertools
from collections import defaultdict
from statistics import fmean
from typing import Any

from .events import RUNTIME_FEATURE_NAMES, policy_map

EVALUATION_SCHEMES = ("leave-one-source-out", "leave-one-task-out")
SELECTOR_FAMILIES = ("global_fixed", "context_rule", "decision_stump")


def build_cross_validation_folds(
    events: list[dict[str, Any]],
    evaluations: tuple[str, ...] = EVALUATION_SCHEMES,
) -> dict[str, Any]:
    unknown = set(evaluations) - set(EVALUATION_SCHEMES)
    if unknown:
        raise ValueError(f"unsupported evaluation schemes: {sorted(unknown)}")
    sources = sorted({event["source_cluster_id"] for event in events})
    tasks = sorted({event["task"] for event in events})
    source_task = {}
    for event in events:
        previous = source_task.setdefault(event["source_cluster_id"], event["task"])
        if previous != event["task"]:
            raise ValueError("one source cluster cannot span multiple tasks")
    folds = []
    if "leave-one-source-out" in evaluations:
        for held_out in sources:
            training = [source for source in sources if source != held_out]
            folds.append(
                {
                    "fold_id": f"loso::{held_out}",
                    "evaluation_scheme": "leave-one-source-out",
                    "training_source_ids": training,
                    "held_out_source_ids": [held_out],
                    "training_task_ids": sorted(
                        {source_task[value] for value in training}
                    ),
                    "held_out_task_ids": [source_task[held_out]],
                }
            )
    if "leave-one-task-out" in evaluations:
        for task in tasks:
            held_out = [source for source in sources if source_task[source] == task]
            training = [source for source in sources if source_task[source] != task]
            folds.append(
                {
                    "fold_id": f"loto::{task}",
                    "evaluation_scheme": "leave-one-task-out",
                    "training_source_ids": training,
                    "held_out_source_ids": held_out,
                    "training_task_ids": [value for value in tasks if value != task],
                    "held_out_task_ids": [task],
                }
            )
    if not folds:
        raise ValueError("at least one cross-validation scheme is required")
    for fold in folds:
        if set(fold["training_source_ids"]) & set(fold["held_out_source_ids"]):
            raise RuntimeError("source leakage detected in cross-validation folds")
    return {
        "schema_version": "1.0",
        "phase": "2D.1A",
        "primary_evaluation": "leave-one-source-out",
        "supplemental_evaluation": "leave-one-task-out",
        "source_cluster_count": len(sources),
        "task_count": len(tasks),
        "source_cluster_ids": sources,
        "folds": folds,
        "round_level_random_split_used": False,
        "same_source_all_contexts_and_rounds_in_one_fold": True,
        "source_leakage_detected": False,
    }


def _events_for_fold(
    events: list[dict[str, Any]], fold: dict[str, Any], node_budget: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    training_ids = set(fold["training_source_ids"])
    held_out_ids = set(fold["held_out_source_ids"])
    training = [
        event
        for event in events
        if event["node_budget_cap"] == node_budget
        and event["source_cluster_id"] in training_ids
    ]
    held_out = [
        event
        for event in events
        if event["node_budget_cap"] == node_budget
        and event["source_cluster_id"] in held_out_ids
    ]
    if not training or not held_out:
        raise ValueError(
            f"fold {fold['fold_id']} has empty train/test data at N={node_budget}"
        )
    if {event["source_cluster_id"] for event in training} & {
        event["source_cluster_id"] for event in held_out
    }:
        raise RuntimeError("held-out source entered selector training data")
    return training, held_out


def _config_ids(events: list[dict[str, Any]]) -> tuple[str, ...]:
    sets = [set(policy_map(event)) for event in events]
    common = set.intersection(*sets)
    if any(value != common for value in sets):
        raise ValueError("policy configuration family differs across training rounds")
    return tuple(sorted(common))


def choose_best_fixed_config(events: list[dict[str, Any]]) -> str:
    """Choose only from training events with a deterministic target-labelled tie-break."""

    candidates = _config_ids(events)
    scores = []
    for config_id in candidates:
        coverage = sum(
            policy_map(event)[config_id]["coverage_depth"] for event in events
        )
        actual_nodes = sum(
            policy_map(event)[config_id]["actual_constructed_node_count"]
            for event in events
        )
        scores.append((-coverage, actual_nodes, config_id))
    return min(scores)[2]


def _prediction(
    event: dict[str, Any],
    *,
    fold: dict[str, Any],
    selector_family: str,
    config_id: str,
    selector_parameters: dict[str, Any],
) -> dict[str, Any]:
    policy = policy_map(event)[config_id]
    headroom = int(event["policy_oracle_headroom"])
    gain = int(policy["coverage_depth"] - event["top1_coverage_depth"])
    return {
        "schema_version": "1.0",
        "evaluation_scheme": fold["evaluation_scheme"],
        "fold_id": fold["fold_id"],
        "training_source_ids": list(fold["training_source_ids"]),
        "held_out_source_ids": list(fold["held_out_source_ids"]),
        "held_out_source_id": event["source_cluster_id"],
        "held_out_task": event["task"],
        "physical_round_id": event["physical_round_id"],
        "sample_id": event["sample_id"],
        "nominal_prompt_length": event["nominal_prompt_length"],
        "generation_repeat": event["generation_repeat"],
        "round_index": event["round_index"],
        "node_budget_cap": event["node_budget_cap"],
        "selector_family": selector_family,
        "selected_config_id": config_id,
        "selected_policy": policy["policy"],
        "selected_policy_variant": policy["policy_variant"],
        "selector_parameters": selector_parameters,
        "selected_policy_coverage_depth": policy["coverage_depth"],
        "selected_actual_constructed_node_count": policy[
            "actual_constructed_node_count"
        ],
        "same_budget_cap_comparison": True,
        "same_actual_node_count_as_top1": policy["same_actual_node_count_comparison"],
        "top1_coverage_depth": event["top1_coverage_depth"],
        "top1_actual_constructed_node_count": event[
            "top1_actual_constructed_node_count"
        ],
        "policy_oracle_coverage_depth": event["policy_oracle_coverage_depth"],
        "prefix_closed_target_oracle_coverage_depth": event[
            "prefix_closed_target_oracle_coverage_depth"
        ],
        "candidate_availability_prefix": event["candidate_availability_prefix"],
        "candidate_available_for_extension": event["candidate_available_for_extension"],
        "policy_oracle_headroom": headroom,
        "realized_gain": gain,
        "per_round_headroom_capture_ratio": gain / headroom if headroom else None,
        "per_round_headroom_capture_ratio_unavailable_reason": (
            None if headroom else "zero_policy_oracle_headroom"
        ),
        "regret_to_policy_oracle": (
            event["policy_oracle_coverage_depth"] - policy["coverage_depth"]
        ),
        "regret_to_prefix_closed_target_oracle": (
            event["prefix_closed_target_oracle_coverage_depth"]
            - policy["coverage_depth"]
        ),
        "selector_parameters_selected_from_training_only": True,
        "runtime_features_use_target_information": False,
        "offline_target_prefix_coverage_only": True,
        "actual_online_acceptance_derived": False,
        "latency_or_speedup_derived": False,
    }


def _global_predictions(
    training: list[dict[str, Any]],
    held_out: list[dict[str, Any]],
    fold: dict[str, Any],
) -> list[dict[str, Any]]:
    selected = choose_best_fixed_config(training)
    return [
        _prediction(
            event,
            fold=fold,
            selector_family="global_fixed",
            config_id=selected,
            selector_parameters={
                "config_id": selected,
                "selection_objective": "maximum_training_sum_coverage_then_minimum_actual_nodes_then_config_id",
            },
        )
        for event in held_out
    ]


def _context_predictions(
    training: list[dict[str, Any]],
    held_out: list[dict[str, Any]],
    fold: dict[str, Any],
) -> list[dict[str, Any]]:
    global_fallback = choose_best_fixed_config(training)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in training:
        grouped[int(event["nominal_prompt_length"])].append(event)
    rules = {
        context: choose_best_fixed_config(rows) for context, rows in grouped.items()
    }
    output = []
    for event in held_out:
        context = int(event["nominal_prompt_length"])
        fallback = context not in rules
        selected = rules.get(context, global_fallback)
        output.append(
            _prediction(
                event,
                fold=fold,
                selector_family="context_rule",
                config_id=selected,
                selector_parameters={
                    "context_length_to_config_id": {
                        str(key): value for key, value in sorted(rules.items())
                    },
                    "global_fallback_config_id": global_fallback,
                    "held_out_context_used_fallback": fallback,
                    "uses_task_id": False,
                    "bin_boundaries_selected_from_training_only": True,
                },
            )
        )
    return output


def _split_indexes(values: list[float], maximum: int) -> list[int]:
    candidates = [
        index for index in range(1, len(values)) if values[index - 1] < values[index]
    ]
    if len(candidates) <= maximum:
        return candidates
    if maximum == 1:
        return [candidates[(len(candidates) - 1) // 2]]
    return sorted(
        {
            candidates[index * (len(candidates) - 1) // (maximum - 1)]
            for index in range(maximum)
        }
    )


def train_decision_stump(
    training: list[dict[str, Any]],
    *,
    minimum_leaf_support: int,
    minimum_leaf_source_support: int,
    maximum_thresholds_per_feature: int,
) -> dict[str, Any]:
    if minimum_leaf_support < 1 or minimum_leaf_source_support < 1:
        raise ValueError("decision-stump leaf support must be positive")
    global_config = choose_best_fixed_config(training)
    configs = _config_ids(training)
    best = None
    for feature in RUNTIME_FEATURE_NAMES:
        ordered = sorted(
            training,
            key=lambda event: (
                float(event["runtime_features"][feature]),
                event["physical_round_id"],
            ),
        )
        values = [float(event["runtime_features"][feature]) for event in ordered]
        cumulative_coverage = {config: [0] for config in configs}
        cumulative_nodes = {config: [0] for config in configs}
        for event in ordered:
            policies = policy_map(event)
            for config in configs:
                cumulative_coverage[config].append(
                    cumulative_coverage[config][-1]
                    + int(policies[config]["coverage_depth"])
                )
                cumulative_nodes[config].append(
                    cumulative_nodes[config][-1]
                    + int(policies[config]["actual_constructed_node_count"])
                )
        for split in _split_indexes(values, maximum_thresholds_per_feature):
            left = ordered[:split]
            right = ordered[split:]
            if min(len(left), len(right)) < minimum_leaf_support:
                continue
            left_sources = {event["source_cluster_id"] for event in left}
            right_sources = {event["source_cluster_id"] for event in right}
            if min(len(left_sources), len(right_sources)) < minimum_leaf_source_support:
                continue
            left_config = min(
                (
                    -cumulative_coverage[config][split],
                    cumulative_nodes[config][split],
                    config,
                )
                for config in configs
            )[2]
            right_config = min(
                (
                    -(
                        cumulative_coverage[config][-1]
                        - cumulative_coverage[config][split]
                    ),
                    cumulative_nodes[config][-1] - cumulative_nodes[config][split],
                    config,
                )
                for config in configs
            )[2]
            objective = cumulative_coverage[left_config][split] + (
                cumulative_coverage[right_config][-1]
                - cumulative_coverage[right_config][split]
            )
            threshold = (values[split - 1] + values[split]) / 2.0
            selected_actual_nodes = (
                cumulative_nodes[left_config][split]
                + cumulative_nodes[right_config][-1]
                - cumulative_nodes[right_config][split]
            )
            candidate = (
                -objective,
                selected_actual_nodes,
                feature,
                threshold,
                left_config,
                right_config,
                {
                    "stump_available": True,
                    "feature": feature,
                    "threshold": threshold,
                    "comparison": "feature <= threshold",
                    "left_config_id": left_config,
                    "right_config_id": right_config,
                    "left_training_round_count": len(left),
                    "right_training_round_count": len(right),
                    "left_training_source_count": len(left_sources),
                    "right_training_source_count": len(right_sources),
                    "minimum_leaf_support": minimum_leaf_support,
                    "minimum_leaf_source_support": minimum_leaf_source_support,
                    "threshold_selected_from_training_only": True,
                    "policies_selected_from_training_only": True,
                    "candidate_config_ids": list(configs),
                },
            )
            if best is None or candidate[:6] < best[:6]:
                best = candidate
    if best is None:
        return {
            "stump_available": False,
            "unavailable_reason": "no_training_split_satisfies_leaf_support",
            "global_fallback_config_id": global_config,
            "minimum_leaf_support": minimum_leaf_support,
            "minimum_leaf_source_support": minimum_leaf_source_support,
            "threshold_selected_from_training_only": True,
            "policies_selected_from_training_only": True,
        }
    return best[6]


def _stump_predictions(
    training: list[dict[str, Any]],
    held_out: list[dict[str, Any]],
    fold: dict[str, Any],
    *,
    minimum_leaf_support: int,
    minimum_leaf_source_support: int,
    maximum_thresholds_per_feature: int,
) -> list[dict[str, Any]]:
    rule = train_decision_stump(
        training,
        minimum_leaf_support=minimum_leaf_support,
        minimum_leaf_source_support=minimum_leaf_source_support,
        maximum_thresholds_per_feature=maximum_thresholds_per_feature,
    )
    output = []
    for event in held_out:
        if rule["stump_available"]:
            selected = (
                rule["left_config_id"]
                if event["runtime_features"][rule["feature"]] <= rule["threshold"]
                else rule["right_config_id"]
            )
        else:
            selected = rule["global_fallback_config_id"]
        output.append(
            _prediction(
                event,
                fold=fold,
                selector_family="decision_stump",
                config_id=selected,
                selector_parameters=rule,
            )
        )
    return output


def evaluate_selectors(
    events: list[dict[str, Any]],
    folds_document: dict[str, Any],
    *,
    selector_families: tuple[str, ...] = SELECTOR_FAMILIES,
    minimum_leaf_support: int = 50,
    minimum_leaf_source_support: int = 2,
    maximum_thresholds_per_feature: int = 64,
) -> dict[str, list[dict[str, Any]]]:
    unknown = set(selector_families) - set(SELECTOR_FAMILIES)
    if unknown:
        raise ValueError(f"unsupported selector families: {sorted(unknown)}")
    budgets = sorted({int(event["node_budget_cap"]) for event in events})
    output = {family: [] for family in selector_families}
    for fold in folds_document["folds"]:
        for node_budget in budgets:
            training, held_out = _events_for_fold(events, fold, node_budget)
            if "global_fixed" in output:
                output["global_fixed"].extend(
                    _global_predictions(training, held_out, fold)
                )
            if "context_rule" in output:
                output["context_rule"].extend(
                    _context_predictions(training, held_out, fold)
                )
            if "decision_stump" in output:
                output["decision_stump"].extend(
                    _stump_predictions(
                        training,
                        held_out,
                        fold,
                        minimum_leaf_support=minimum_leaf_support,
                        minimum_leaf_source_support=minimum_leaf_source_support,
                        maximum_thresholds_per_feature=maximum_thresholds_per_feature,
                    )
                )
    return output


def evaluate_policy_portfolios(
    events: list[dict[str, Any]],
    folds_document: dict[str, Any],
    *,
    portfolio_sizes: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows = []
    primary_folds = [
        fold
        for fold in folds_document["folds"]
        if fold["evaluation_scheme"] == "leave-one-source-out"
    ]
    budgets = sorted({event["node_budget_cap"] for event in events})
    for fold in primary_folds:
        for node_budget in budgets:
            training, held_out = _events_for_fold(events, fold, node_budget)
            configs = _config_ids(training)
            for size in portfolio_sizes:
                if not 1 <= size <= len(configs):
                    raise ValueError(
                        "portfolio size exceeds policy configuration count"
                    )
                candidates = []
                for portfolio in itertools.combinations(configs, size):
                    training_coverage = sum(
                        max(
                            policy_map(event)[config]["coverage_depth"]
                            for config in portfolio
                        )
                        for event in training
                    )
                    candidates.append((-training_coverage, portfolio))
                _, selected = min(candidates)
                held_out_coverage = [
                    max(
                        policy_map(event)[config]["coverage_depth"]
                        for config in selected
                    )
                    for event in held_out
                ]
                baseline = [event["top1_coverage_depth"] for event in held_out]
                headroom = [event["policy_oracle_headroom"] for event in held_out]
                gain_sum = sum(
                    value - base for value, base in zip(held_out_coverage, baseline)
                )
                headroom_sum = sum(headroom)
                rows.append(
                    {
                        "schema_version": "1.0",
                        "evaluation_scheme": "leave-one-source-out",
                        "fold_id": fold["fold_id"],
                        "training_source_ids": fold["training_source_ids"],
                        "held_out_source_ids": fold["held_out_source_ids"],
                        "node_budget_cap": node_budget,
                        "portfolio_size": size,
                        "selected_config_ids": list(selected),
                        "training_sum_coverage": -min(candidates)[0],
                        "held_out_round_count": len(held_out),
                        "held_out_mean_coverage_depth": fmean(held_out_coverage),
                        "held_out_mean_gain_over_top1": fmean(
                            value - base
                            for value, base in zip(held_out_coverage, baseline)
                        ),
                        "held_out_sum_gain_over_top1": gain_sum,
                        "held_out_policy_oracle_headroom_sum": headroom_sum,
                        "held_out_headroom_capture_ratio_of_sums": (
                            gain_sum / headroom_sum if headroom_sum else None
                        ),
                        "held_out_headroom_capture_ratio_unavailable_reason": (
                            None if headroom_sum else "zero_policy_oracle_headroom"
                        ),
                        "portfolio_is_oracle": True,
                        "portfolio_is_deployable": False,
                        "non_deployable_reason": (
                            "best portfolio member is selected per held-out round "
                            "using target-labelled coverage"
                        ),
                        "offline_target_prefix_coverage_only": True,
                    }
                )
    return rows
