"""Build selector-realization events from validated Phase 2D.0 records."""

from __future__ import annotations

import json
from typing import Any

from .features import (
    FORBIDDEN_RUNTIME_FIELD_FRAGMENTS,
    RUNTIME_FEATURE_NAMES,
    aggregate_runtime_features,
)

__all__ = [
    "FORBIDDEN_RUNTIME_FIELD_FRAGMENTS",
    "RUNTIME_FEATURE_NAMES",
    "build_selector_events",
    "physical_round_id",
    "policy_config_id",
    "policy_map",
    "source_cluster_id",
]


def source_cluster_id(row: dict[str, Any]) -> str:
    return f"{row['task']}::{row['source_sample_id']}"


def physical_round_id(row: dict[str, Any]) -> str:
    return "::".join(
        str(value)
        for value in (
            row["task"],
            row["sample_id"],
            row["nominal_prompt_length"],
            row["generation_repeat"],
            row["round_index"],
        )
    )


def policy_config_id(
    policy: str,
    variant: str,
    parameters: dict[str, Any],
    *,
    top_m: int,
    node_budget: int,
) -> str:
    parameter_text = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return (
        f"policy={policy}|variant={variant}|parameters={parameter_text}|"
        f"top_m={top_m}|node_budget={node_budget}"
    )


def _runtime_features(event: dict[str, Any], policy_count: int) -> dict[str, float]:
    return aggregate_runtime_features(
        event["deployable_features_by_position"],
        context_length=int(event["nominal_prompt_length"]),
        node_budget=int(event["node_budget"]),
        policy_configuration_count=policy_count,
    )


def _policy_rows(event: dict[str, Any]) -> list[dict[str, Any]]:
    top_m = int(event["top_m"])
    node_budget = int(event["node_budget"])
    rows = []
    for value in event["static_policy_coverage"]:
        actual_nodes = int(value["verified_nodes"])
        if not 1 <= actual_nodes <= node_budget:
            raise ValueError("actual constructed node count exceeds its budget cap")
        config_id = policy_config_id(
            str(value["policy"]),
            str(value["policy_variant"]),
            dict(value["policy_parameters"]),
            top_m=top_m,
            node_budget=node_budget,
        )
        rows.append(
            {
                "config_id": config_id,
                "policy": value["policy"],
                "policy_variant": value["policy_variant"],
                "policy_parameters": value["policy_parameters"],
                "top_m": top_m,
                "coverage_depth": int(value["coverage_depth"]),
                "actual_constructed_node_count": actual_nodes,
                "node_budget_cap": node_budget,
                "budget_utilization": actual_nodes / node_budget,
                "coverage_depth_per_actual_node": (
                    int(value["coverage_depth"]) / actual_nodes
                ),
                "same_budget_cap_comparison": True,
                "root_included_in_node_count": True,
                "policy_construction_uses_target_information": False,
            }
        )
    rows.sort(key=lambda row: row["config_id"])
    if len({row["config_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate concrete policy configuration")
    return rows


def build_selector_events(
    phase2d0_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one eligible selector event per physical round and fixed N."""

    output = []
    expected_configs: dict[int, tuple[str, ...]] = {}
    for source in phase2d0_events:
        if not source["classification_eligible"]:
            continue
        policies = _policy_rows(source)
        node_budget = int(source["node_budget"])
        config_ids = tuple(row["config_id"] for row in policies)
        if node_budget not in expected_configs:
            expected_configs[node_budget] = config_ids
        elif expected_configs[node_budget] != config_ids:
            raise ValueError(f"policy configuration set changed within N={node_budget}")
        baselines = [row for row in policies if row["policy"] == "top1_chain"]
        if len(baselines) != 1:
            raise ValueError("each selector event requires exactly one Top-1 baseline")
        baseline = baselines[0]
        top1_coverage = int(baseline["coverage_depth"])
        policy_oracle_coverage = max(row["coverage_depth"] for row in policies)
        winners = sorted(
            row["config_id"]
            for row in policies
            if row["coverage_depth"] == policy_oracle_coverage
        )
        prefix_oracle = int(source["oracle_coverage_depth"])
        availability = int(source["candidate_availability_prefix"])
        if not (
            top1_coverage <= policy_oracle_coverage <= prefix_oracle <= availability
        ):
            raise ValueError(
                "coverage invariant failed for "
                f"round={physical_round_id(source)} N={node_budget}: "
                f"top1={top1_coverage} policy_oracle={policy_oracle_coverage} "
                f"prefix_oracle={prefix_oracle} availability={availability}"
            )
        for policy in policies:
            policy["gain_over_top1"] = policy["coverage_depth"] - top1_coverage
            policy["regret_to_policy_oracle"] = (
                policy_oracle_coverage - policy["coverage_depth"]
            )
            policy["regret_to_prefix_closed_target_oracle"] = (
                prefix_oracle - policy["coverage_depth"]
            )
            policy["candidate_availability_prefix"] = availability
            policy["same_actual_node_count_comparison"] = (
                policy["actual_constructed_node_count"]
                == baseline["actual_constructed_node_count"]
            )
        runtime_features = _runtime_features(source, len(policies))
        output.append(
            {
                "schema_version": "1.0",
                "task": source["task"],
                "sample_id": source["sample_id"],
                "source_sample_id": source["source_sample_id"],
                "source_cluster_id": source_cluster_id(source),
                "source_text_sha256": source["source_text_sha256"],
                "nominal_prompt_length": int(source["nominal_prompt_length"]),
                "actual_prompt_length": int(source["actual_prompt_length"]),
                "generation_repeat": int(source["generation_repeat"]),
                "round_index": int(source["round_index"]),
                "physical_round_id": physical_round_id(source),
                "top_m": int(source["top_m"]),
                "node_budget_cap": node_budget,
                "count_unit": "eligible_physical_round_at_fixed_node_budget",
                "root_included_in_node_budget": True,
                "top1_config_id": baseline["config_id"],
                "top1_coverage_depth": top1_coverage,
                "top1_actual_constructed_node_count": baseline[
                    "actual_constructed_node_count"
                ],
                "policy_configurations": policies,
                "policy_oracle_coverage_depth": policy_oracle_coverage,
                "policy_oracle_winner_config_ids": winners,
                "policy_oracle_tie_count": len(winners),
                "policy_oracle_is_deployable": False,
                "policy_oracle_non_deployable_reason": (
                    "winner is selected per round using target-labelled coverage"
                ),
                "prefix_closed_target_oracle_coverage_depth": prefix_oracle,
                "prefix_closed_target_oracle_is_deployable": False,
                "prefix_closed_target_oracle_non_deployable_reason": (
                    "tree construction directly uses the target continuation path"
                ),
                "candidate_availability_prefix": availability,
                "candidate_available_for_extension": availability > top1_coverage,
                "policy_oracle_headroom": policy_oracle_coverage - top1_coverage,
                "runtime_features": runtime_features,
                "runtime_feature_names": list(RUNTIME_FEATURE_NAMES),
                "runtime_features_use_target_information": False,
                "top16_entropy_semantics": (
                    "renormalized entropy of captured Top-16 probability mass; "
                    "not full-vocabulary entropy"
                ),
                "offline_target_prefix_coverage_only": True,
                "actual_online_acceptance_derived": False,
                "latency_or_speedup_derived": False,
            }
        )
    if not output:
        raise ValueError("no eligible Phase 2D.1A selector events")
    output.sort(
        key=lambda row: (
            row["task"],
            row["source_cluster_id"],
            row["nominal_prompt_length"],
            row["generation_repeat"],
            row["round_index"],
            row["node_budget_cap"],
        )
    )
    return output


def policy_map(event: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["config_id"]: row for row in event["policy_configurations"]}
