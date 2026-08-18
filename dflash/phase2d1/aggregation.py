"""Selector realization summaries, winner heterogeneity, and decision gates."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from statistics import fmean, median
from typing import Any

from .events import policy_map

DECISIONS = (
    "integrate_fixed_policy_online",
    "integrate_context_rule_online",
    "integrate_lightweight_selector_online",
    "selector_headroom_not_realizable",
    "deprioritize_selector_prioritize_candidate_generation",
    "insufficient_evidence",
)


def _distribution(values: list[int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items())}


def _source_macro(records: list[dict[str, Any]], field: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        grouped[row["source_cluster_id"]].append(float(row[field]))
    return fmean(fmean(values) for values in grouped.values())


def _task_macro(records: list[dict[str, Any]], field: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        grouped[row["task"]].append(float(row[field]))
    return fmean(fmean(values) for values in grouped.values())


def _metric_row(
    records: list[dict[str, Any]],
    *,
    identity: dict[str, Any],
    candidate_subset: str,
) -> dict[str, Any]:
    if candidate_subset == "candidate_available_for_extension":
        records = [row for row in records if row["candidate_available_for_extension"]]
    elif candidate_subset != "all_eligible_rounds":
        raise ValueError(f"unsupported candidate subset: {candidate_subset}")
    if not records:
        return {
            "schema_version": "1.0",
            **identity,
            "candidate_subset": candidate_subset,
            "round_count": 0,
            "source_count": 0,
            "task_count": 0,
            "metrics_available": False,
            "metrics_unavailable_reason": "no_rounds_in_subset",
        }
    coverage = [int(row["selected_coverage_depth"]) for row in records]
    nodes = [int(row["selected_actual_node_count"]) for row in records]
    top1 = [int(row["top1_coverage_depth"]) for row in records]
    gains = [int(row["realized_gain"]) for row in records]
    headroom = [int(row["policy_oracle_headroom"]) for row in records]
    positive_ratios = [gain / room for gain, room in zip(gains, headroom) if room > 0]
    gain_sum = sum(gains)
    headroom_sum = sum(headroom)
    top1_sum = sum(top1)
    return {
        "schema_version": "1.0",
        **identity,
        "candidate_subset": candidate_subset,
        "round_count": len(records),
        "source_count": len({row["source_cluster_id"] for row in records}),
        "task_count": len({row["task"] for row in records}),
        "count_unit": "eligible_physical_round_at_fixed_node_budget",
        "metrics_available": True,
        "metrics_unavailable_reason": None,
        "mean_coverage_depth": fmean(coverage),
        "median_coverage_depth": median(coverage),
        "coverage_depth_distribution": _distribution(coverage),
        "sum_coverage_depth": sum(coverage),
        "mean_actual_constructed_node_count": fmean(nodes),
        "coverage_depth_per_actual_node_ratio_of_sums": sum(coverage) / sum(nodes),
        "mean_per_round_coverage_depth_per_actual_node": fmean(
            value / node for value, node in zip(coverage, nodes)
        ),
        "mean_gain_over_top1": fmean(gains),
        "sum_gain_over_top1": gain_sum,
        "relative_gain_over_top1_ratio_of_sums": (
            gain_sum / top1_sum if top1_sum else None
        ),
        "relative_gain_over_top1_unavailable_reason": (
            None if top1_sum else "zero_top1_coverage_sum"
        ),
        "policy_oracle_headroom_sum": headroom_sum,
        "headroom_capture_ratio_of_sums": (
            gain_sum / headroom_sum if headroom_sum else None
        ),
        "headroom_capture_ratio_of_sums_unavailable_reason": (
            None if headroom_sum else "zero_policy_oracle_headroom_sum"
        ),
        "mean_per_round_headroom_capture_ratio": (
            fmean(positive_ratios) if positive_ratios else None
        ),
        "mean_per_round_headroom_capture_ratio_unavailable_reason": (
            None if positive_ratios else "no_round_with_positive_policy_oracle_headroom"
        ),
        "headroom_capture_primary_metric": "ratio_of_sums",
        "mean_regret_to_policy_oracle": fmean(
            row["regret_to_policy_oracle"] for row in records
        ),
        "mean_regret_to_prefix_closed_target_oracle": fmean(
            row["regret_to_prefix_closed_target_oracle"] for row in records
        ),
        "mean_candidate_availability_ceiling": fmean(
            row["candidate_availability_prefix"] for row in records
        ),
        "source_macro_mean_coverage_depth": _source_macro(
            records, "selected_coverage_depth"
        ),
        "source_macro_mean_gain_over_top1": _source_macro(records, "realized_gain"),
        "task_macro_mean_coverage_depth": _task_macro(
            records, "selected_coverage_depth"
        ),
        "task_macro_mean_gain_over_top1": _task_macro(records, "realized_gain"),
        "same_budget_cap_comparison": True,
        "same_actual_node_count_fraction": fmean(
            bool(row["same_actual_node_count_as_top1"]) for row in records
        ),
        "offline_metric_semantics": "offline target-prefix coverage and verification-efficiency proxy",
        "actual_online_acceptance_derived": False,
        "latency_or_speedup_derived": False,
    }


def _group_metric_rows(
    records: list[dict[str, Any]],
    *,
    selector_level: str,
    selector_id: str,
    evaluation_scheme: str,
) -> list[dict[str, Any]]:
    grouped: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    budgets = sorted({row["node_budget_cap"] for row in records})
    for budget in budgets:
        budget_rows = [row for row in records if row["node_budget_cap"] == budget]
        grouped.append(("overall", {}, budget_rows))
        for task in sorted({row["task"] for row in budget_rows}):
            grouped.append(
                (
                    "task",
                    {"task": task},
                    [row for row in budget_rows if row["task"] == task],
                )
            )
        for context in sorted({row["nominal_prompt_length"] for row in budget_rows}):
            grouped.append(
                (
                    "context",
                    {"context_length": context},
                    [
                        row
                        for row in budget_rows
                        if row["nominal_prompt_length"] == context
                    ],
                )
            )
        for task, context in sorted(
            {(row["task"], row["nominal_prompt_length"]) for row in budget_rows}
        ):
            grouped.append(
                (
                    "task_context",
                    {"task": task, "context_length": context},
                    [
                        row
                        for row in budget_rows
                        if row["task"] == task
                        and row["nominal_prompt_length"] == context
                    ],
                )
            )
    output = []
    for scope, extra, rows in grouped:
        identity = {
            "scope": scope,
            "evaluation_scheme": evaluation_scheme,
            "selector_level": selector_level,
            "selector_id": selector_id,
            "node_budget_cap": rows[0]["node_budget_cap"],
            **extra,
        }
        for subset in (
            "all_eligible_rounds",
            "candidate_available_for_extension",
        ):
            output.append(_metric_row(rows, identity=identity, candidate_subset=subset))
    return output


def _event_record(
    event: dict[str, Any],
    *,
    selector_level: str,
    selector_id: str,
    coverage: int,
    actual_nodes: int,
) -> dict[str, Any]:
    return {
        "task": event["task"],
        "source_cluster_id": event["source_cluster_id"],
        "nominal_prompt_length": event["nominal_prompt_length"],
        "node_budget_cap": event["node_budget_cap"],
        "selector_level": selector_level,
        "selector_id": selector_id,
        "selected_coverage_depth": coverage,
        "selected_actual_node_count": actual_nodes,
        "same_actual_node_count_as_top1": (
            actual_nodes == event["top1_actual_constructed_node_count"]
        ),
        "top1_coverage_depth": event["top1_coverage_depth"],
        "realized_gain": coverage - event["top1_coverage_depth"],
        "policy_oracle_headroom": event["policy_oracle_headroom"],
        "regret_to_policy_oracle": event["policy_oracle_coverage_depth"] - coverage,
        "regret_to_prefix_closed_target_oracle": (
            event["prefix_closed_target_oracle_coverage_depth"] - coverage
        ),
        "candidate_availability_prefix": event["candidate_availability_prefix"],
        "candidate_available_for_extension": event["candidate_available_for_extension"],
    }


def fixed_policy_summaries(events: list[dict[str, Any]]) -> dict[str, Any]:
    inventory = {}
    records_by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        for policy in event["policy_configurations"]:
            inventory.setdefault(
                policy["config_id"],
                {
                    key: policy[key]
                    for key in (
                        "config_id",
                        "policy",
                        "policy_variant",
                        "policy_parameters",
                        "top_m",
                        "node_budget_cap",
                    )
                },
            )
            records_by_config[policy["config_id"]].append(
                _event_record(
                    event,
                    selector_level="fixed_policy_configuration",
                    selector_id=policy["config_id"],
                    coverage=policy["coverage_depth"],
                    actual_nodes=policy["actual_constructed_node_count"],
                )
            )
    rows = []
    for config_id, records in sorted(records_by_config.items()):
        rows.extend(
            _group_metric_rows(
                records,
                selector_level="fixed_policy_configuration",
                selector_id=config_id,
                evaluation_scheme="full_offline_fixed_configuration",
            )
        )
    return {
        "schema_version": "1.0",
        "phase": "2D.1A",
        "configuration_inventory": [inventory[key] for key in sorted(inventory)],
        "rows": rows,
        "fixed_configurations_are_reported_without_per_round_target_selection": True,
    }


def _prediction_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "task": row["held_out_task"],
            "source_cluster_id": row["held_out_source_id"],
            "nominal_prompt_length": row["nominal_prompt_length"],
            "node_budget_cap": row["node_budget_cap"],
            "selected_coverage_depth": row["selected_policy_coverage_depth"],
            "selected_actual_node_count": row["selected_actual_constructed_node_count"],
            "same_actual_node_count_as_top1": row["same_actual_node_count_as_top1"],
            "top1_coverage_depth": row["top1_coverage_depth"],
            "realized_gain": row["realized_gain"],
            "policy_oracle_headroom": row["policy_oracle_headroom"],
            "regret_to_policy_oracle": row["regret_to_policy_oracle"],
            "regret_to_prefix_closed_target_oracle": row[
                "regret_to_prefix_closed_target_oracle"
            ],
            "candidate_availability_prefix": row["candidate_availability_prefix"],
            "candidate_available_for_extension": row[
                "candidate_available_for_extension"
            ],
        }
        for row in rows
    ]


def selector_summary_rows(
    events: list[dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for family, rows in predictions.items():
        for evaluation_scheme in (
            "leave-one-source-out",
            "leave-one-task-out",
        ):
            evaluation_rows = [
                row for row in rows if row["evaluation_scheme"] == evaluation_scheme
            ]
            output.extend(
                _group_metric_rows(
                    _prediction_records(evaluation_rows),
                    selector_level=family,
                    selector_id=family,
                    evaluation_scheme=evaluation_scheme,
                )
            )
    special_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        top1_nodes = event["top1_actual_constructed_node_count"]
        special_records["top1_chain"].append(
            _event_record(
                event,
                selector_level="top1_baseline",
                selector_id="top1_chain",
                coverage=event["top1_coverage_depth"],
                actual_nodes=top1_nodes,
            )
        )
        winner_nodes = min(
            policy_map(event)[config]["actual_constructed_node_count"]
            for config in event["policy_oracle_winner_config_ids"]
        )
        special_records["policy_oracle"].append(
            _event_record(
                event,
                selector_level="per_round_policy_oracle",
                selector_id="policy_oracle",
                coverage=event["policy_oracle_coverage_depth"],
                actual_nodes=winner_nodes,
            )
        )
        special_records["prefix_closed_target_oracle"].append(
            _event_record(
                event,
                selector_level="prefix_closed_target_oracle",
                selector_id="prefix_closed_target_oracle",
                coverage=event["prefix_closed_target_oracle_coverage_depth"],
                actual_nodes=event["prefix_closed_target_oracle_coverage_depth"] + 1,
            )
        )
    for selector_id, records in special_records.items():
        output.extend(
            _group_metric_rows(
                records,
                selector_level=records[0]["selector_level"],
                selector_id=selector_id,
                evaluation_scheme="offline_ceiling_or_baseline",
            )
        )
    return output


def winner_composition(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    for budget in sorted({event["node_budget_cap"] for event in events}):
        budget_rows = [event for event in events if event["node_budget_cap"] == budget]
        groups.append(("overall", {}, budget_rows))
        for task in sorted({event["task"] for event in budget_rows}):
            groups.append(
                (
                    "task",
                    {"task": task},
                    [row for row in budget_rows if row["task"] == task],
                )
            )
        for context in sorted(
            {event["nominal_prompt_length"] for event in budget_rows}
        ):
            groups.append(
                (
                    "context",
                    {"context_length": context},
                    [
                        row
                        for row in budget_rows
                        if row["nominal_prompt_length"] == context
                    ],
                )
            )
        for task, context in sorted(
            {(event["task"], event["nominal_prompt_length"]) for event in budget_rows}
        ):
            groups.append(
                (
                    "task_context",
                    {"task": task, "context_length": context},
                    [
                        row
                        for row in budget_rows
                        if row["task"] == task
                        and row["nominal_prompt_length"] == context
                    ],
                )
            )
    output = []
    for scope, identity, rows in groups:
        configs = sorted(policy_map(rows[0]))
        fractional = Counter()
        for event in rows:
            for config in event["policy_oracle_winner_config_ids"]:
                fractional[config] += 1 / len(event["policy_oracle_winner_config_ids"])
        probabilities = [fractional[config] / len(rows) for config in configs]
        entropy = -sum(value * math.log(value) for value in probabilities if value)
        for config in configs:
            wins = [
                event
                for event in rows
                if config in event["policy_oracle_winner_config_ids"]
            ]
            unique = [
                event
                for event in wins
                if len(event["policy_oracle_winner_config_ids"]) == 1
            ]
            policy_values = [policy_map(event)[config] for event in rows]
            output.append(
                {
                    "schema_version": "1.0",
                    "scope": scope,
                    **identity,
                    "node_budget_cap": rows[0]["node_budget_cap"],
                    "config_id": config,
                    "round_count": len(rows),
                    "policy_win_count": len(wins),
                    "unique_win_count": len(unique),
                    "tie_win_count": len(wins) - len(unique),
                    "any_tie_round_count": sum(
                        len(event["policy_oracle_winner_config_ids"]) > 1
                        for event in rows
                    ),
                    "win_fraction": len(wins) / len(rows),
                    "fractional_tie_adjusted_win_fraction": fractional[config]
                    / len(rows),
                    "mean_coverage_depth": fmean(
                        value["coverage_depth"] for value in policy_values
                    ),
                    "mean_gain_over_top1": fmean(
                        value["gain_over_top1"] for value in policy_values
                    ),
                    "winner_entropy_nats": entropy,
                }
            )
    return output


def policy_overlap(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for budget in sorted({event["node_budget_cap"] for event in events}):
        rows = [event for event in events if event["node_budget_cap"] == budget]
        configs = sorted(policy_map(rows[0]))
        win_sets = {
            config: {
                event["physical_round_id"]
                for event in rows
                if config in event["policy_oracle_winner_config_ids"]
            }
            for config in configs
        }
        for left, right in itertools_combinations(configs):
            intersection = win_sets[left] & win_sets[right]
            union = win_sets[left] | win_sets[right]
            exclusive = win_sets[left] ^ win_sets[right]
            output.append(
                {
                    "schema_version": "1.0",
                    "node_budget_cap": budget,
                    "left_config_id": left,
                    "right_config_id": right,
                    "round_count": len(rows),
                    "co_win_count": len(intersection),
                    "either_win_count": len(union),
                    "pairwise_win_overlap_jaccard": (
                        len(intersection) / len(union) if union else None
                    ),
                    "pairwise_win_overlap_unavailable_reason": (
                        None if union else "neither_policy_wins_any_round"
                    ),
                    "pairwise_complementarity_count": len(exclusive),
                    "pairwise_complementarity_fraction": len(exclusive) / len(rows),
                }
            )
    return output


def itertools_combinations(values: list[str]):
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            yield left, right


def source_cluster_bootstrap(
    prediction_rows: list[dict[str, Any]],
    *,
    repeats: int,
    seed: int,
    minimum_stable_clusters: int,
) -> list[dict[str, Any]]:
    primary = [
        row
        for row in prediction_rows
        if row["evaluation_scheme"] == "leave-one-source-out"
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in primary:
        groups[
            (row["selector_family"], row["node_budget_cap"], "overall", "all")
        ].append(row)
        groups[
            (
                row["selector_family"],
                row["node_budget_cap"],
                "task",
                row["held_out_task"],
            )
        ].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        selector, budget, scope, group = key
        clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            clusters[row["held_out_source_id"]].append(row)
        cluster_ids = sorted(clusters)
        stable_seed = int.from_bytes(
            hashlib.sha256(repr(key).encode("utf-8")).digest()[:8], "big"
        )
        generator = random.Random(seed + stable_seed)
        gain_samples = []
        ratio_samples = []
        for _ in range(repeats):
            sampled = [generator.choice(cluster_ids) for _ in cluster_ids]
            sampled_rows = [row for cluster in sampled for row in clusters[cluster]]
            gains = sum(row["realized_gain"] for row in sampled_rows)
            headroom = sum(row["policy_oracle_headroom"] for row in sampled_rows)
            gain_samples.append(gains / len(sampled_rows))
            if headroom:
                ratio_samples.append(gains / headroom)
        gain_samples.sort()
        ratio_samples.sort()

        def interval(values: list[float]) -> tuple[float | None, float | None]:
            if not values:
                return None, None
            return (
                values[int(0.025 * (len(values) - 1))],
                values[int(0.975 * (len(values) - 1))],
            )

        gain_low, gain_high = interval(gain_samples)
        ratio_low, ratio_high = interval(ratio_samples)
        output.append(
            {
                "schema_version": "1.0",
                "evaluation_scheme": "leave-one-source-out",
                "selector_family": selector,
                "node_budget_cap": budget,
                "scope": scope,
                "group": group,
                "source_cluster_count": len(cluster_ids),
                "round_count": len(rows),
                "bootstrap_repeats": repeats,
                "bootstrap_seed": seed,
                "sampling_unit": "source_document_cluster",
                "round_level_bootstrap_used": False,
                "mean_gain_ci_lower": gain_low,
                "mean_gain_ci_upper": gain_high,
                "headroom_capture_ratio_of_sums_ci_lower": ratio_low,
                "headroom_capture_ratio_of_sums_ci_upper": ratio_high,
                "headroom_capture_ci_unavailable_reason": (
                    None if ratio_samples else "zero_headroom_in_every_bootstrap_sample"
                ),
                "ci_unstable": len(cluster_ids) < minimum_stable_clusters,
                "ci_instability_reason": (
                    f"fewer_than_{minimum_stable_clusters}_source_clusters"
                    if len(cluster_ids) < minimum_stable_clusters
                    else None
                ),
            }
        )
    return output


def context_censoring_lookup(
    gap: dict[str, Any],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    output = {}
    for row in gap["task_context"]:
        if row["top_m"] != 16 or row["gap_class"] != "right_censored_or_terminal":
            continue
        output[(row["task"], row["context_length"], row["node_budget"])] = {
            "raw_round_count_including_censored": row["raw_round_count"],
            "eligible_round_count": row["eligible_round_count"],
            "right_censored_or_terminal_count": row["right_censored_or_terminal_count"],
            "censoring_fraction": row["censoring_fraction"],
        }
    return output


def add_context_censoring(
    summary_rows: list[dict[str, Any]], gap: dict[str, Any]
) -> list[dict[str, Any]]:
    lookup = context_censoring_lookup(gap)
    output = []
    for row in summary_rows:
        if row["scope"] != "task_context":
            continue
        key = (row["task"], row["context_length"], row["node_budget_cap"])
        output.append({**row, **lookup[key]})
    return output


def build_decision(
    selector_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    winner_rows: list[dict[str, Any]],
    portfolio_rows: list[dict[str, Any]],
    *,
    thresholds: dict[str, Any],
    source_count: int,
    task_count: int,
    eligible_physical_round_count: int,
    candidate_availability_degradation_present: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    overall = {
        (row["selector_level"], row["node_budget_cap"]): row
        for row in selector_rows
        if row["scope"] == "overall"
        and row["evaluation_scheme"] == "leave-one-source-out"
        and row["candidate_subset"] == "all_eligible_rounds"
        and row["selector_level"] in {"global_fixed", "context_rule", "decision_stump"}
    }
    budgets = sorted({budget for _, budget in overall})
    ci_available = all(
        not row["ci_unstable"] for row in bootstrap_rows if row["scope"] == "overall"
    )
    evidence_sufficient = (
        source_count >= thresholds["minimum_source_document_count"]
        and task_count >= thresholds["minimum_task_count"]
        and eligible_physical_round_count
        >= thresholds["minimum_eligible_physical_rounds"]
        and (ci_available or not thresholds["require_stable_clustered_ci"])
    )

    def minimum_metric(selector: str, field: str) -> float:
        values = [overall[(selector, budget)][field] for budget in budgets]
        available = [float(value) for value in values if value is not None]
        return min(available) if available else -math.inf

    task_rows = [
        row
        for row in selector_rows
        if row["scope"] == "task"
        and row["evaluation_scheme"] == "leave-one-source-out"
        and row["candidate_subset"] == "all_eligible_rounds"
    ]
    supported_stratum_rows = [
        row
        for row in selector_rows
        if row["scope"] in {"context", "task_context"}
        and row["evaluation_scheme"] == "leave-one-source-out"
        and row["candidate_subset"] == "all_eligible_rounds"
        and row["round_count"] >= thresholds["minimum_supported_stratum_rounds"]
    ]

    def no_material_regression(selector: str) -> bool:
        return all(
            row["mean_gain_over_top1"]
            >= -thresholds["maximum_supported_stratum_regression"]
            for row in supported_stratum_rows
            if row["selector_level"] == selector
        )

    primary_tasks = {"gov_report", "hotpotqa"}
    fixed_task_macro = {
        (row["task"], row["node_budget_cap"]): row["source_macro_mean_gain_over_top1"]
        for row in task_rows
        if row["selector_level"] == "global_fixed" and row["task"] in primary_tasks
    }
    required_primary_task_budget_pairs = {
        (task, budget) for task in primary_tasks for budget in budgets
    }
    fixed_task_ok = set(fixed_task_macro) == required_primary_task_budget_pairs and all(
        value >= 0 for value in fixed_task_macro.values()
    )
    global_fixed_fold_choices: dict[int, dict[str, str]] = defaultdict(dict)
    for row in prediction_rows:
        if (
            row["selector_family"] == "global_fixed"
            and row["evaluation_scheme"] == "leave-one-source-out"
        ):
            global_fixed_fold_choices[row["node_budget_cap"]][row["fold_id"]] = row[
                "selected_config_id"
            ]
    global_fixed_selection_stability = {}
    for budget in budgets:
        choices = list(global_fixed_fold_choices[budget].values())
        modal_count = max(Counter(choices).values(), default=0)
        global_fixed_selection_stability[budget] = {
            "fold_count": len(choices),
            "distinct_selected_config_count": len(set(choices)),
            "modal_selected_config_id": (
                min(
                    config
                    for config, count in Counter(choices).items()
                    if count == modal_count
                )
                if choices
                else None
            ),
            "modal_selection_fraction": modal_count / len(choices) if choices else 0.0,
        }
    fixed_selection_stable = all(
        row["modal_selection_fraction"]
        >= thresholds["minimum_global_fixed_selection_stability"]
        for row in global_fixed_selection_stability.values()
    )
    fixed_ready = bool(
        evidence_sufficient
        and minimum_metric("global_fixed", "mean_gain_over_top1")
        >= thresholds["minimum_held_out_gain_over_top1"]
        and minimum_metric("global_fixed", "headroom_capture_ratio_of_sums")
        >= thresholds["minimum_headroom_capture_ratio"]
        and fixed_task_ok
        and fixed_selection_stable
        and no_material_regression("global_fixed")
    )
    context_additional = min(
        overall[("context_rule", budget)]["mean_gain_over_top1"]
        - overall[("global_fixed", budget)]["mean_gain_over_top1"]
        for budget in budgets
    )
    context_task_improvements = [
        (
            context["task"],
            context["source_macro_mean_gain_over_top1"]
            - fixed["source_macro_mean_gain_over_top1"],
        )
        for context in task_rows
        for fixed in task_rows
        if context["selector_level"] == "context_rule"
        and fixed["selector_level"] == "global_fixed"
        and context["task"] == fixed["task"]
        and context["node_budget_cap"] == fixed["node_budget_cap"]
    ]
    context_improvements_grouped: dict[str, list[float]] = defaultdict(list)
    for task, value in context_task_improvements:
        context_improvements_grouped[task].append(value)
    context_improvement_by_task = {
        task: fmean(values)
        for task, values in sorted(context_improvements_grouped.items())
    }
    context_ready = bool(
        evidence_sufficient
        and minimum_metric("context_rule", "mean_gain_over_top1")
        >= thresholds["minimum_held_out_gain_over_top1"]
        and minimum_metric("context_rule", "headroom_capture_ratio_of_sums")
        >= thresholds["minimum_headroom_capture_ratio"]
        and context_additional >= thresholds["minimum_additional_gain_over_best_fixed"]
        and sum(value > 0 for value in context_improvement_by_task.values()) >= 2
        and no_material_regression("context_rule")
    )
    stump_additional = min(
        overall[("decision_stump", budget)]["mean_gain_over_top1"]
        - max(
            overall[("global_fixed", budget)]["mean_gain_over_top1"],
            overall[("context_rule", budget)]["mean_gain_over_top1"],
        )
        for budget in budgets
    )
    loto_stump = [
        row
        for row in prediction_rows
        if row["selector_family"] == "decision_stump"
        and row["evaluation_scheme"] == "leave-one-task-out"
    ]
    loto_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in loto_stump:
        loto_by_task[row["held_out_task"]].append(row)
    loto_no_reverse = bool(loto_by_task) and all(
        fmean(row["realized_gain"] for row in rows)
        >= -thresholds["maximum_supported_stratum_regression"]
        for rows in loto_by_task.values()
    )
    stump_rules_supported = all(
        row["selector_parameters"].get("stump_available", False)
        for row in prediction_rows
        if row["selector_family"] == "decision_stump"
        and row["evaluation_scheme"] == "leave-one-source-out"
    )
    stump_ready = bool(
        evidence_sufficient
        and minimum_metric("decision_stump", "mean_gain_over_top1")
        >= thresholds["minimum_held_out_gain_over_top1"]
        and minimum_metric("decision_stump", "headroom_capture_ratio_of_sums")
        >= thresholds["minimum_headroom_capture_ratio"]
        and stump_additional >= thresholds["minimum_additional_gain_over_best_fixed"]
        and no_material_regression("decision_stump")
        and loto_no_reverse
        and stump_rules_supported
    )

    winner_heterogeneity = []
    for budget in budgets:
        rows = [
            row
            for row in winner_rows
            if row["scope"] == "overall" and row["node_budget_cap"] == budget
        ]
        dominant_fraction = max(
            row["fractional_tie_adjusted_win_fraction"] for row in rows
        )
        dominant_configs = sorted(
            row["config_id"]
            for row in rows
            if row["fractional_tie_adjusted_win_fraction"] == dominant_fraction
        )
        winner_heterogeneity.append(
            {
                "node_budget_cap": budget,
                "winner_entropy_nats": rows[0]["winner_entropy_nats"],
                "dominant_fractional_tie_adjusted_win_fraction": dominant_fraction,
                "dominant_config_ids": dominant_configs,
                "majority_concentration_threshold": 0.5,
                "winner_interpretation": (
                    "concentrated_at_majority_level"
                    if dominant_fraction >= 0.5
                    else "heterogeneous_below_majority_level"
                ),
            }
        )
    portfolio_summary = []
    for budget in budgets:
        for size in sorted({row["portfolio_size"] for row in portfolio_rows}):
            rows = [
                row
                for row in portfolio_rows
                if row["node_budget_cap"] == budget and row["portfolio_size"] == size
            ]
            gain_sum = sum(row["held_out_sum_gain_over_top1"] for row in rows)
            headroom_sum = sum(
                row["held_out_policy_oracle_headroom_sum"] for row in rows
            )
            portfolio_summary.append(
                {
                    "node_budget_cap": budget,
                    "portfolio_size": size,
                    "fold_count": len(rows),
                    "held_out_mean_gain_over_top1": fmean(
                        row["held_out_mean_gain_over_top1"] for row in rows
                    ),
                    "held_out_headroom_capture_ratio_of_sums": (
                        gain_sum / headroom_sum if headroom_sum else None
                    ),
                    "held_out_headroom_capture_ratio_unavailable_reason": (
                        None if headroom_sum else "zero_policy_oracle_headroom_sum"
                    ),
                    "portfolio_is_oracle": True,
                    "portfolio_is_deployable": False,
                }
            )
    candidate_subset_comparison = []
    for selector in ("global_fixed", "context_rule", "decision_stump"):
        for budget in budgets:
            rows = [
                row
                for row in selector_rows
                if row["scope"] == "overall"
                and row["evaluation_scheme"] == "leave-one-source-out"
                and row["selector_level"] == selector
                and row["node_budget_cap"] == budget
            ]
            by_subset = {row["candidate_subset"]: row for row in rows}
            candidate_subset_comparison.append(
                {
                    "selector_level": selector,
                    "node_budget_cap": budget,
                    "all_eligible_round_count": by_subset["all_eligible_rounds"][
                        "round_count"
                    ],
                    "all_eligible_mean_gain_over_top1": by_subset[
                        "all_eligible_rounds"
                    ]["mean_gain_over_top1"],
                    "candidate_available_round_count": by_subset[
                        "candidate_available_for_extension"
                    ]["round_count"],
                    "candidate_available_mean_gain_over_top1": by_subset[
                        "candidate_available_for_extension"
                    ].get("mean_gain_over_top1"),
                    "candidate_available_metric_unavailable_reason": by_subset[
                        "candidate_available_for_extension"
                    ]["metrics_unavailable_reason"],
                }
            )

    if not evidence_sufficient:
        decision = "insufficient_evidence"
    elif stump_ready:
        decision = "integrate_lightweight_selector_online"
    elif context_ready:
        decision = "integrate_context_rule_online"
    elif fixed_ready:
        decision = "integrate_fixed_policy_online"
    else:
        oracle_headroom = fmean(
            row["policy_oracle_headroom_sum"] / row["round_count"]
            for row in overall.values()
        )
        best_realized = max(row["mean_gain_over_top1"] for row in overall.values())
        if (
            best_realized < thresholds["minimum_held_out_gain_over_top1"]
            and candidate_availability_degradation_present
        ):
            decision = "deprioritize_selector_prioritize_candidate_generation"
        elif oracle_headroom >= thresholds["minimum_held_out_gain_over_top1"]:
            decision = "selector_headroom_not_realizable"
        else:
            decision = "insufficient_evidence"

    decision_document = {
        "schema_version": "1.0",
        "phase": "2D.1A",
        "decision": decision,
        "decision_rules": thresholds,
        "evidence_sufficient": evidence_sufficient,
        "fixed_policy_gate_passed": fixed_ready,
        "global_fixed_selection_stability_by_node_budget": (
            global_fixed_selection_stability
        ),
        "global_fixed_selection_stability_gate_passed": fixed_selection_stable,
        "primary_task_source_macro_nonnegative_gate_passed": fixed_task_ok,
        "context_rule_source_macro_improvement_by_task": context_improvement_by_task,
        "context_rule_gate_passed": context_ready,
        "decision_stump_gate_passed": stump_ready,
        "context_rule_minimum_additional_gain": context_additional,
        "decision_stump_minimum_additional_gain": stump_additional,
        "leave_one_task_out_stump_no_systematic_reverse": loto_no_reverse,
        "clustered_ci_available_and_stable": ci_available,
        "candidate_availability_degradation_present": (
            candidate_availability_degradation_present
        ),
        "engineering_gate_not_statistical_significance_theorem": True,
        "online_integration_not_performed": True,
    }
    research = {
        "schema_version": "1.0",
        "phase": "2D.1A",
        "rq1": {
            "question": "Does one fixed policy robustly outperform Top-1?",
            "global_fixed_overall_by_node_budget": [
                overall[("global_fixed", budget)] for budget in budgets
            ],
            "engineering_gate_passed": fixed_ready,
            "global_fixed_selection_stability_by_node_budget": (
                global_fixed_selection_stability
            ),
        },
        "rq2": {
            "question": "How much policy-oracle headroom does the held-out fixed policy realize?",
            "primary_metric": "headroom_capture_ratio_of_sums",
            "results": [overall[("global_fixed", budget)] for budget in budgets],
        },
        "rq3": {
            "question": "Are policy winners heterogeneous or concentrated?",
            "artifact": "policy-winner-composition.csv",
            "winner_heterogeneity_by_node_budget": winner_heterogeneity,
            "portfolio_frontier": portfolio_summary,
            "portfolio_is_non_deployable_oracle": True,
        },
        "rq4": {
            "question": "Does context conditioning improve over global fixed policy?",
            "minimum_additional_gain": context_additional,
            "engineering_gate_passed": context_ready,
        },
        "rq5": {
            "question": "Does a source-held-out draft-feature stump add value?",
            "minimum_additional_gain": stump_additional,
            "leave_one_task_out_no_systematic_reverse": loto_no_reverse,
            "engineering_gate_passed": stump_ready,
        },
        "rq6": {
            "question": "Are gains candidate-available and context-stable?",
            "candidate_subset_results_present": True,
            "candidate_subset_comparison": candidate_subset_comparison,
            "supported_stratum_regression_gate_by_selector": {
                selector: no_material_regression(selector)
                for selector in (
                    "global_fixed",
                    "context_rule",
                    "decision_stump",
                )
            },
            "context_artifact": "context-selector-summary.csv",
            "qasper_censoring_and_small_source_cohort_reported_separately": True,
        },
        "rq7": {
            "question": "Which selector route should follow?",
            "decision": decision,
            "decision_artifact": "selector-decision.json",
        },
        "decision": decision,
        "offline_target_prefix_coverage_only": True,
        "no_acceptance_latency_or_speedup_claim": True,
    }
    return decision_document, research
