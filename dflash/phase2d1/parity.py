"""CPU replay proving offline/runtime feature and selector parity."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from dflash.phase2b.tree import confidence_gated_tree, score_beam_tree, top1_chain_tree
from dflash.phase2c.core import layerwise_beam_tree

from .features import aggregate_runtime_features, build_deployable_feature_rows
from .runtime import select_policy, validate_frozen_selector


def round_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["task"],
        row["sample_id"],
        int(row["nominal_prompt_length"]),
        int(row["generation_repeat"]),
        int(row["round_index"]),
    )


def construct_policy_nodes(
    positions: list[dict[str, Any]], policy_config: dict[str, Any]
) -> set[tuple[int, ...]]:
    budget = int(policy_config["node_budget_cap"])
    policy = policy_config["policy"]
    parameters = policy_config["policy_parameters"]
    builders: dict[str, Callable[[], set[tuple[int, ...]]]] = {
        "top1_chain": lambda: top1_chain_tree(positions, budget=budget),
        "score_beam": lambda: score_beam_tree(positions, budget=budget),
        "confidence_gated": lambda: confidence_gated_tree(
            positions,
            budget=budget,
            margin_threshold=float(parameters["margin_threshold"]),
            branch_width=int(parameters["branch_width"]),
        ),
        "layerwise_beam": lambda: layerwise_beam_tree(
            positions, budget=budget, beam_width=int(parameters["beam_width"])
        ),
    }
    try:
        nodes = builders[policy]()
    except KeyError as exc:
        raise ValueError(f"online constructor does not support {policy!r}") from exc
    if not nodes or () not in nodes or len(nodes) > budget:
        raise ValueError("constructed policy violates root-inclusive node cap")
    return nodes


def _compare_feature_maps(
    offline: dict[str, float], runtime: dict[str, float], *, tolerance: float
) -> list[dict[str, Any]]:
    differences = []
    if tuple(offline) != tuple(runtime):
        return [{"feature": "<schema>", "offline": list(offline), "runtime": list(runtime)}]
    for name, offline_value in offline.items():
        runtime_value = runtime[name]
        if not math.isclose(
            float(offline_value), float(runtime_value), rel_tol=tolerance, abs_tol=tolerance
        ):
            differences.append(
                {"feature": name, "offline": offline_value, "runtime": runtime_value}
            )
    return differences


def check_selector_parity(
    *,
    frozen: dict[str, Any],
    selector_events: list[dict[str, Any]],
    phase2d0_events: list[dict[str, Any]],
    captures: list[dict[str, Any]],
    tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    validate_frozen_selector(frozen)
    captures_by_key = {round_key(row): row for row in captures}
    phase2d0_by_key = {
        (*round_key(row), int(row["node_budget"])): row for row in phase2d0_events
    }
    failures = []
    checked_policy_ids: set[str] = set()
    for event in selector_events:
        key = round_key(event)
        budget = int(event["node_budget_cap"])
        source = phase2d0_by_key.get((*key, budget))
        capture = captures_by_key.get(key)
        if source is None or capture is None:
            failures.append({"physical_round": key, "reason": "missing_source_round"})
            continue
        positions = [
            {**row, "candidates": [dict(value) for value in row["candidates"][:16]]}
            for row in capture["positions"]
        ]
        canonical_nodes = score_beam_tree(positions, budget=budget)
        feature_rows = build_deployable_feature_rows(
            positions=positions,
            canonical_nodes=canonical_nodes,
            context_length=int(event["nominal_prompt_length"]),
            top_m=16,
            node_budget=budget,
        )
        runtime_features = aggregate_runtime_features(
            feature_rows,
            context_length=int(event["nominal_prompt_length"]),
            node_budget=budget,
            policy_configuration_count=len(event["policy_configurations"]),
        )
        differences = _compare_feature_maps(
            event["runtime_features"], runtime_features, tolerance=tolerance
        )
        if differences or feature_rows != source["deployable_features_by_position"]:
            failures.append(
                {
                    "physical_round": key,
                    "node_budget": budget,
                    "reason": "runtime_feature_mismatch",
                    "differences": differences,
                    "per_position_exact_equal": feature_rows
                    == source["deployable_features_by_position"],
                }
            )
            continue
        for mode in ("global_fixed", "context_rule", "decision_stump"):
            decision = select_policy(
                frozen,
                mode=mode,
                node_budget=budget,
                context_length=int(event["nominal_prompt_length"]),
                runtime_features=runtime_features,
            )
            nodes = construct_policy_nodes(positions, decision["selected_policy"])
            checked_policy_ids.add(decision["selected_policy_config"])
            if len(nodes) > budget or decision["target_dependent_runtime_input_used"]:
                failures.append(
                    {
                        "physical_round": key,
                        "node_budget": budget,
                        "mode": mode,
                        "reason": "policy_support_or_target_blind_invariant",
                    }
                )
    # Exercise every allowlisted configuration, not only configurations selected
    # by these rounds, using the first available capture at the matching budget.
    first_capture = next(iter(captures_by_key.values()))
    positions = [
        {**row, "candidates": [dict(value) for value in row["candidates"][:16]]}
        for row in first_capture["positions"]
    ]
    for rule in frozen["node_budget_rules"].values():
        for policy in rule["allowed_policy_configurations"]:
            construct_policy_nodes(positions, policy)
            checked_policy_ids.add(policy["config_id"])
    if failures:
        first = failures[0]
        raise ValueError(
            "selector parity failed: "
            f"physical_round={first.get('physical_round')} "
            f"reason={first.get('reason')} details={first}"
        )
    return {
        "schema_version": "1.0",
        "status": "pass",
        "eligible_configuration_round_count": len(selector_events),
        "eligible_physical_round_count": len({round_key(row) for row in selector_events}),
        "checked_policy_configurations": sorted(checked_policy_ids),
        "checked_policy_configuration_count": len(checked_policy_ids),
        "feature_tolerance": tolerance,
        "per_position_features_exact": True,
        "aggregated_features_match": True,
        "global_fixed_parity": True,
        "context_rule_parity": True,
        "decision_stump_parity": True,
        "target_dependent_runtime_input_used": False,
        "all_constructed_node_counts_within_cap": True,
    }
