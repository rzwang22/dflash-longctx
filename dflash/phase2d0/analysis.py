"""Pure-Python bounded-tree gap characterization from Phase 2B captures."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from dflash.phase2b.tree import (
    Node,
    confidence_gated_tree,
    covered_target_prefix_depth,
    oracle_prefix_closed_tree,
    score_beam_tree,
    target_candidate_rank,
    top1_chain_tree,
)
from dflash.phase2b1.policies import layerwise_beam_tree
from dflash.phase2b1.ranking import capped_target_prefix_score_rank
from dflash.phase2d1.features import build_deployable_feature_rows

GAP_CLASS_PRIORITY = (
    "right_censored_or_terminal",
    "no_failure_within_observed_horizon",
    "candidate_availability_failure",
    "selection_failure",
    "budget_failure",
    "dependency_realization_opportunity",
    "possible_dependency_opportunity",
    "unresolved",
)
SUPPORTED_POLICIES = (
    "top1_chain",
    "score_beam",
    "confidence_gated",
    "layerwise_beam",
)
DEFAULT_CONFIDENCE_THRESHOLDS = (0.25, 0.5, 1.0)
DEFAULT_BEAM_WIDTHS = (2, 4, 8, 16)


def _truncate_positions(
    positions: list[dict[str, Any]], top_m: int
) -> list[dict[str, Any]]:
    truncated = []
    for position in positions:
        candidates = [dict(value) for value in position["candidates"][:top_m]]
        truncated.append({**position, "candidates": candidates})
    return truncated


def _policy_specs(
    policies: tuple[str, ...],
    *,
    confidence_thresholds: tuple[float, ...],
    beam_widths: tuple[int, ...],
) -> list[dict[str, Any]]:
    unknown = set(policies) - set(SUPPORTED_POLICIES)
    if unknown:
        raise ValueError(f"unsupported static policies: {sorted(unknown)}")
    specs: list[dict[str, Any]] = []
    if "top1_chain" in policies:
        specs.append(
            {
                "policy": "top1_chain",
                "variant": "exact_baseline_chain",
                "parameters": {},
                "builder": lambda positions, budget: top1_chain_tree(
                    positions, budget=budget
                ),
            }
        )
    if "score_beam" in policies:
        specs.append(
            {
                "policy": "score_beam",
                "variant": "top1_backbone_length_normalized_best_first",
                "parameters": {},
                "builder": lambda positions, budget: score_beam_tree(
                    positions, budget=budget
                ),
            }
        )
    if "confidence_gated" in policies:
        for threshold in confidence_thresholds:
            specs.append(
                {
                    "policy": "confidence_gated",
                    "variant": f"top1_top2_margin_le_{threshold:g}",
                    "parameters": {
                        "margin_threshold": threshold,
                        "branch_width": 4,
                    },
                    "builder": (
                        lambda positions, budget, threshold=threshold: (
                            confidence_gated_tree(
                                positions,
                                budget=budget,
                                margin_threshold=threshold,
                                branch_width=4,
                            )
                        )
                    ),
                }
            )
    if "layerwise_beam" in policies:
        for width in beam_widths:
            specs.append(
                {
                    "policy": "layerwise_beam",
                    "variant": f"fixed_width_{width}",
                    "parameters": {"beam_width": width},
                    "builder": (
                        lambda positions, budget, width=width: layerwise_beam_tree(
                            positions, budget=budget, beam_width=width
                        )
                    ),
                }
            )
    if not specs:
        raise ValueError("at least one target-blind static policy is required")
    return specs


def _availability(capture: dict[str, Any], top_m: int) -> dict[str, Any]:
    reference = [
        int(token) for token in capture["reference_target_continuation_token_ids"]
    ]
    ranks = [
        target_candidate_rank(capture["positions"][index], token)
        for index, token in enumerate(reference)
    ]
    membership = [rank is not None and rank <= top_m for rank in ranks]
    prefix = 0
    for available in membership:
        if not available:
            break
        prefix += 1
    first_unavailable = prefix + 1 if prefix < len(reference) else None
    first_rank = ranks[first_unavailable - 1] if first_unavailable else None
    return {
        "target_candidate_ranks_top16": ranks,
        "target_membership_top_m": membership,
        "availability_prefix": prefix,
        "first_unavailable_depth": first_unavailable,
        "first_unavailable_target_outside_top16": (
            first_rank is None if first_unavailable else None
        ),
        "first_unavailable_target_rank_lower_bound": (
            17 if first_unavailable and first_rank is None else first_rank
        ),
    }


class _Replay:
    def __init__(
        self,
        positions: list[dict[str, Any]],
        reference: list[int],
        specs: list[dict[str, Any]],
    ) -> None:
        self.positions = positions
        self.reference = reference
        self.specs = specs
        self._trees: dict[tuple[str, int], set[Node]] = {}
        self._minimum_budgets: dict[tuple[str, int, int], int | None] = {}
        self._target_prefix_ranks: dict[tuple[int, int], dict[str, Any]] = {}

    def tree(self, spec: dict[str, Any], budget: int) -> set[Node]:
        key = (str(spec["variant"]), int(budget))
        if key not in self._trees:
            builder: Callable[..., set[Node]] = spec["builder"]
            self._trees[key] = builder(self.positions, int(budget))
        return self._trees[key]

    def coverage(self, spec: dict[str, Any], budget: int) -> int:
        return covered_target_prefix_depth(self.tree(spec, budget), self.reference)

    def minimum_budget(
        self, spec: dict[str, Any], *, depth: int, maximum: int
    ) -> int | None:
        if depth < 1:
            return 1
        key = (str(spec["variant"]), int(depth), int(maximum))
        if key in self._minimum_budgets:
            return self._minimum_budgets[key]
        minimum = len(self.positions) + 1
        if self.coverage(spec, maximum) < depth:
            self._minimum_budgets[key] = None
            return None
        low, high = minimum, maximum
        while low < high:
            middle = (low + high) // 2
            if self.coverage(spec, middle) >= depth:
                high = middle
            else:
                low = middle + 1
        self._minimum_budgets[key] = low
        return low

    def target_prefix_rank(self, *, depth: int, rank_cap: int) -> dict[str, Any]:
        key = (int(depth), int(rank_cap))
        if key not in self._target_prefix_ranks:
            self._target_prefix_ranks[key] = capped_target_prefix_score_rank(
                self.positions,
                self.reference[:depth],
                rank_cap=rank_cap,
            )
        return self._target_prefix_ranks[key]


def classify_gap(
    *,
    terminal_or_censored: bool,
    no_failure_within_horizon: bool,
    critical_token_available: bool,
    best_static_extends: bool,
    modest_budget_resolves: bool,
    oracle_extends: bool,
    significant_rank_or_budget_inflation: bool,
) -> str:
    """Apply the documented mutually exclusive Phase 2D.0 priority order."""

    if terminal_or_censored:
        return "right_censored_or_terminal"
    if no_failure_within_horizon:
        return "no_failure_within_observed_horizon"
    if not critical_token_available:
        return "candidate_availability_failure"
    if best_static_extends:
        return "selection_failure"
    if modest_budget_resolves:
        return "budget_failure"
    if oracle_extends and significant_rank_or_budget_inflation:
        return "dependency_realization_opportunity"
    if oracle_extends:
        return "possible_dependency_opportunity"
    return "unresolved"


def _round_identity(capture: dict[str, Any]) -> dict[str, Any]:
    return {
        key: capture.get(key)
        for key in (
            "task",
            "sample_id",
            "source_sample_id",
            "source_text_sha256",
            "nominal_prompt_length",
            "actual_prompt_length",
            "generation_repeat",
            "round_index",
            "terminal_round",
            "reference_right_censored",
            "stop_reason",
        )
    }


def _event_for_configuration(
    capture: dict[str, Any],
    *,
    top_m: int,
    node_budget: int,
    replay: _Replay,
    availability: dict[str, Any],
    frontier_budgets: tuple[int, ...],
    survival_budget_cap: int,
    rank_cap: int,
    budget_failure_multiplier: float,
    rank_inflation_multiplier: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    reference = replay.reference
    observed_horizon = len(reference)
    baseline = int(capture["accepted_draft_token_count"])
    baseline_observable = min(baseline, observed_horizon)
    critical_depth = baseline + 1 if baseline < observed_horizon else None

    policy_at_budget = []
    frontier_rows = []
    minimum_rows = []
    for spec in replay.specs:
        nodes = replay.tree(spec, node_budget)
        coverage = covered_target_prefix_depth(nodes, reference)
        policy_at_budget.append(
            {
                "policy": spec["policy"],
                "policy_variant": spec["variant"],
                "policy_parameters": spec["parameters"],
                "coverage_depth": coverage,
                "verified_nodes": len(nodes),
                "coverage_depth_per_verified_node": coverage / len(nodes),
            }
        )
        previous_budget = None
        previous_coverage = None
        previous_nodes = None
        for budget in frontier_budgets:
            frontier_nodes = replay.tree(spec, budget)
            frontier_coverage = covered_target_prefix_depth(frontier_nodes, reference)
            added_nodes = (
                len(frontier_nodes) - previous_nodes
                if previous_nodes is not None
                else None
            )
            coverage_gain = (
                frontier_coverage - previous_coverage
                if previous_coverage is not None
                else None
            )
            frontier_rows.append(
                {
                    "schema_version": "1.0",
                    **_round_identity(capture),
                    "top_m": top_m,
                    "policy": spec["policy"],
                    "policy_variant": spec["variant"],
                    "node_budget": budget,
                    "root_included_in_budget": True,
                    "verified_nodes": len(frontier_nodes),
                    "coverage_depth": frontier_coverage,
                    "coverage_depth_per_verified_node": (
                        frontier_coverage / len(frontier_nodes)
                    ),
                    "previous_node_budget": previous_budget,
                    "added_verified_nodes": added_nodes,
                    "coverage_gain": coverage_gain,
                    "coverage_gain_per_added_node": (
                        coverage_gain / added_nodes
                        if added_nodes not in (None, 0)
                        else None
                    ),
                    "coverage_gain_unavailable_reason": (
                        "first_frontier_point"
                        if previous_budget is None
                        else "no_added_verified_nodes"
                        if added_nodes == 0
                        else None
                    ),
                    "interpretation": "offline_verification_efficiency_proxy",
                }
            )
            previous_budget = budget
            previous_coverage = frontier_coverage
            previous_nodes = len(frontier_nodes)

        minimum = (
            replay.minimum_budget(
                spec, depth=critical_depth, maximum=survival_budget_cap
            )
            if critical_depth is not None
            and availability["availability_prefix"] >= critical_depth
            else None
        )
        minimum_rows.append(
            {
                "schema_version": "1.0",
                **_round_identity(capture),
                "top_m": top_m,
                "core_node_budget": node_budget,
                "policy": spec["policy"],
                "policy_variant": spec["variant"],
                "target_depth": critical_depth,
                "minimum_survival_budget": minimum,
                "minimum_survival_budget_unavailable_reason": (
                    "no_failure_within_observed_horizon"
                    if critical_depth is None
                    else "critical_target_token_unavailable"
                    if availability["availability_prefix"] < critical_depth
                    else f"censored_above_{survival_budget_cap}"
                    if minimum is None
                    else None
                ),
                "minimum_budget_censored": (
                    critical_depth is not None
                    and availability["availability_prefix"] >= critical_depth
                    and minimum is None
                ),
                "survival_budget_cap": survival_budget_cap,
                "root_included_in_budget": True,
            }
        )

    top1 = next(row for row in policy_at_budget if row["policy"] == "top1_chain")
    if top1["coverage_depth"] != baseline_observable:
        raise ValueError(
            "captured accepted depth does not match replayed Top-1 coverage"
        )
    best = max(
        policy_at_budget,
        key=lambda row: (
            row["coverage_depth"],
            -row["verified_nodes"],
            row["policy"],
            row["policy_variant"],
        ),
    )
    oracle_nodes = oracle_prefix_closed_tree(
        replay.positions, reference, budget=node_budget
    )
    oracle_coverage = covered_target_prefix_depth(oracle_nodes, reference)
    oracle_ceiling = availability["availability_prefix"]

    valid_minimums = [
        row["minimum_survival_budget"]
        for row in minimum_rows
        if row["minimum_survival_budget"] is not None
    ]
    best_required = min(valid_minimums) if valid_minimums else None
    oracle_required = (
        critical_depth + 1
        if critical_depth is not None
        and availability["availability_prefix"] >= critical_depth
        else None
    )
    modest_limit = min(
        survival_budget_cap, math.floor(node_budget * budget_failure_multiplier)
    )
    modest_resolves = (
        best_required is not None
        and best_required > node_budget
        and best_required <= modest_limit
    )

    rank_result = None
    if (
        critical_depth is not None
        and availability["availability_prefix"] >= critical_depth
    ):
        rank_result = replay.target_prefix_rank(depth=critical_depth, rank_cap=rank_cap)
    rank_lower_bound = (
        rank_result["target_prefix_score_rank_lower_bound"]
        if rank_result is not None
        else None
    )
    rank_threshold = max(1, math.ceil((node_budget - 1) * rank_inflation_multiplier))
    rank_inflated = (
        rank_result is not None
        and rank_lower_bound is not None
        and rank_lower_bound > rank_threshold
    )
    budget_inflated = best_required is not None and best_required > modest_limit
    significant_inflation = bool(rank_inflated or budget_inflated)

    terminal_or_censored = bool(
        capture["terminal_round"] or capture["reference_right_censored"]
    )
    no_failure = critical_depth is None
    critical_available = bool(
        critical_depth is not None
        and availability["availability_prefix"] >= critical_depth
    )
    best_extends = bool(
        critical_depth is not None and best["coverage_depth"] >= critical_depth
    )
    oracle_extends = bool(
        critical_depth is not None and oracle_coverage >= critical_depth
    )
    gap_class = classify_gap(
        terminal_or_censored=terminal_or_censored,
        no_failure_within_horizon=no_failure,
        critical_token_available=critical_available,
        best_static_extends=best_extends,
        modest_budget_resolves=modest_resolves,
        oracle_extends=oracle_extends,
        significant_rank_or_budget_inflation=significant_inflation,
    )

    canonical_spec = next(
        (spec for spec in replay.specs if spec["policy"] == "score_beam"),
        replay.specs[0],
    )
    canonical_nodes = replay.tree(canonical_spec, node_budget)
    deployable_features = build_deployable_feature_rows(
        positions=replay.positions,
        canonical_nodes=canonical_nodes,
        context_length=int(capture["nominal_prompt_length"]),
        top_m=top_m,
        node_budget=node_budget,
    )

    event = {
        "schema_version": "1.0",
        **_round_identity(capture),
        "capture_directory": capture.get("_capture_directory"),
        "top_m": top_m,
        "node_budget": node_budget,
        "root_included_in_node_budget": True,
        "observed_target_horizon": observed_horizon,
        "baseline_accepted_draft_depth_a0": baseline,
        "critical_depth": critical_depth,
        "gap_class": gap_class,
        "classification_priority": GAP_CLASS_PRIORITY.index(gap_class) + 1,
        "classification_eligible": not terminal_or_censored,
        "candidate_availability_prefix": availability["availability_prefix"],
        "availability_ceiling": oracle_ceiling,
        "first_unavailable_depth": availability["first_unavailable_depth"],
        "first_unavailable_target_outside_top16": availability[
            "first_unavailable_target_outside_top16"
        ],
        "first_unavailable_target_rank_lower_bound": availability[
            "first_unavailable_target_rank_lower_bound"
        ],
        "target_candidate_ranks_top16": availability["target_candidate_ranks_top16"],
        "target_membership_top_m": availability["target_membership_top_m"],
        "availability_extension_vs_a0": max(0, oracle_ceiling - baseline_observable),
        "best_static_policy": best["policy"],
        "best_static_policy_variant": best["policy_variant"],
        "static_policy_coverage": policy_at_budget,
        "best_static_coverage_depth": best["coverage_depth"],
        "static_coverage_ceiling_at_cap": max(
            replay.coverage(spec, survival_budget_cap) for spec in replay.specs
        ),
        "oracle_coverage_depth": oracle_coverage,
        "oracle_coverage_ceiling": oracle_ceiling,
        "oracle_minus_static_coverage": oracle_coverage - best["coverage_depth"],
        "oracle_required_nodes": oracle_required,
        "oracle_required_nodes_unavailable_reason": (
            None
            if oracle_required is not None
            else "no_failure_within_observed_horizon"
            if critical_depth is None
            else "critical_target_token_unavailable"
        ),
        "best_static_required_nodes": best_required,
        "best_static_required_nodes_unavailable_reason": (
            None
            if best_required is not None
            else "no_failure_within_observed_horizon"
            if critical_depth is None
            else "critical_target_token_unavailable"
            if not critical_available
            else f"censored_above_{survival_budget_cap}"
        ),
        "static_oracle_budget_ratio": (
            best_required / oracle_required
            if best_required is not None and oracle_required is not None
            else None
        ),
        "static_oracle_budget_ratio_unavailable_reason": (
            None
            if best_required is not None and oracle_required is not None
            else "best_static_or_oracle_required_nodes_unavailable"
        ),
        "minimum_nodes_per_covered_depth": (
            best_required / critical_depth
            if best_required is not None and critical_depth
            else None
        ),
        "minimum_nodes_per_covered_depth_unavailable_reason": (
            None
            if best_required is not None and critical_depth
            else "best_static_required_nodes_or_critical_depth_unavailable"
        ),
        "target_prefix_score_rank": (
            rank_result["target_prefix_score_rank"] if rank_result else None
        ),
        "target_prefix_score_rank_lower_bound": rank_lower_bound,
        "target_prefix_score_rank_censored": (
            rank_result["target_prefix_score_rank_censored"] if rank_result else False
        ),
        "target_prefix_score_rank_unavailable_reason": (
            rank_result["unavailable_reason"]
            if rank_result
            else "no_failure_within_observed_horizon"
            if critical_depth is None
            else "critical_target_token_unavailable"
        ),
        "first_pruning_depth": (
            best["coverage_depth"] + 1
            if best["coverage_depth"] < availability["availability_prefix"]
            else None
        ),
        "first_pruning_depth_unavailable_reason": (
            None
            if best["coverage_depth"] < availability["availability_prefix"]
            else "no_observed_candidate_available_static_pruning"
        ),
        "shallow_target_prefix_entered_tree": best["coverage_depth"] > 0,
        "correct_path_pruned_at_depth": (
            best["coverage_depth"] + 1
            if critical_available and best["coverage_depth"] < critical_depth
            else None
        ),
        "correct_path_pruned_at_depth_unavailable_reason": (
            None
            if critical_available and best["coverage_depth"] < critical_depth
            else "critical_path_not_pruned_by_all_tested_static_policies"
        ),
        "budget_failure_multiplier": budget_failure_multiplier,
        "modest_budget_limit": modest_limit,
        "rank_inflation_multiplier": rank_inflation_multiplier,
        "rank_inflation_threshold": rank_threshold,
        "significant_rank_inflation": rank_inflated,
        "significant_budget_inflation": budget_inflated,
        "survival_budget_cap": survival_budget_cap,
        "coverage_depth_per_verified_node": (
            best["coverage_depth"] / best["verified_nodes"]
        ),
        "oracle_coverage_depth_per_verified_node": (
            oracle_coverage / len(oracle_nodes)
        ),
        "interpretation": "offline_verification_efficiency_proxy_only",
        "actual_online_acceptance_derived": False,
        "latency_or_speedup_derived": False,
        "deployable_features_by_position": deployable_features,
    }
    return event, frontier_rows, minimum_rows


def _frontier_budgets(
    node_budgets: tuple[int, ...], survival_budget_cap: int
) -> tuple[int, ...]:
    values = set(node_budgets)
    value = 16
    while value < survival_budget_cap:
        values.add(value)
        value *= 2
    values.add(survival_budget_cap)
    return tuple(sorted(value for value in values if value <= survival_budget_cap))


def analyze_captures(
    captures: list[dict[str, Any]],
    *,
    top_ms: tuple[int, ...] = (2, 4, 8, 16),
    node_budgets: tuple[int, ...] = (16, 32),
    survival_budget_cap: int = 256,
    policies: tuple[str, ...] = SUPPORTED_POLICIES,
    confidence_thresholds: tuple[float, ...] = DEFAULT_CONFIDENCE_THRESHOLDS,
    beam_widths: tuple[int, ...] = DEFAULT_BEAM_WIDTHS,
    rank_cap: int = 100_000,
    budget_failure_multiplier: float = 2.0,
    rank_inflation_multiplier: float = 2.0,
) -> dict[str, Any]:
    """Replay all requested marginal policies without importing ML frameworks."""

    if not captures:
        raise ValueError("no validated capture rounds were provided")
    if tuple(sorted(set(top_ms))) != top_ms or not set(top_ms) <= {2, 4, 8, 16}:
        raise ValueError("top_ms must be an increasing subset of {2, 4, 8, 16}")
    if tuple(sorted(set(node_budgets))) != node_budgets or node_budgets[0] < 16:
        raise ValueError("node_budgets must be unique, increasing, and at least 16")
    if survival_budget_cap < node_budgets[-1]:
        raise ValueError("survival budget cap must cover every core node budget")
    specs = _policy_specs(
        policies,
        confidence_thresholds=confidence_thresholds,
        beam_widths=beam_widths,
    )
    if not any(spec["policy"] == "top1_chain" for spec in specs):
        raise ValueError("policies must include top1_chain as the baseline")
    frontier = _frontier_budgets(node_budgets, survival_budget_cap)
    events = []
    frontier_rows = []
    minimum_rows = []
    for capture in captures:
        reference = [
            int(token) for token in capture["reference_target_continuation_token_ids"]
        ]
        for top_m in top_ms:
            positions = _truncate_positions(capture["positions"], top_m)
            availability = _availability(capture, top_m)
            replay = _Replay(positions, reference, specs)
            for budget in node_budgets:
                event, current_frontier, current_minimum = _event_for_configuration(
                    capture,
                    top_m=top_m,
                    node_budget=budget,
                    replay=replay,
                    availability=availability,
                    frontier_budgets=frontier,
                    survival_budget_cap=survival_budget_cap,
                    rank_cap=rank_cap,
                    budget_failure_multiplier=budget_failure_multiplier,
                    rank_inflation_multiplier=rank_inflation_multiplier,
                )
                events.append(event)
                frontier_rows.extend(current_frontier)
                minimum_rows.extend(current_minimum)
    return {
        "events": events,
        "frontier_rows": frontier_rows,
        "minimum_budget_rows": minimum_rows,
        "configuration": {
            "top_ms": list(top_ms),
            "node_budgets": list(node_budgets),
            "survival_frontier_budgets": list(frontier),
            "survival_budget_cap": survival_budget_cap,
            "policies": list(policies),
            "confidence_thresholds": list(confidence_thresholds),
            "beam_widths": list(beam_widths),
            "rank_cap": rank_cap,
            "budget_failure_multiplier": budget_failure_multiplier,
            "rank_inflation_multiplier": rank_inflation_multiplier,
            "root_included_in_node_budget": True,
        },
    }
