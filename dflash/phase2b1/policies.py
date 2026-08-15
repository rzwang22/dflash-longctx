"""Additional target-blind diagnostic policies and replay helpers."""

from __future__ import annotations

from typing import Any

from dflash.phase2b.tree import Node, validate_prefix_closed


def _score_maps(positions: list[dict[str, Any]]) -> list[dict[int, float]]:
    return [
        {
            int(candidate["token_id"]): float(candidate["log_probability"])
            for candidate in position["candidates"]
        }
        for position in positions
    ]


def _priority(prefix: Node, score_maps: list[dict[int, float]]) -> tuple[Any, ...]:
    cumulative = sum(score_maps[index][token] for index, token in enumerate(prefix))
    return (-cumulative / len(prefix), -cumulative, len(prefix), prefix)


def layerwise_beam_tree(
    positions: list[dict[str, Any]], *, budget: int, beam_width: int
) -> set[Node]:
    """Retain best same-depth prefixes while respecting a total-node budget.

    The fixed-width beam is independent of the requested total budget. Layers
    are inserted in depth/score order until the budget is exhausted, making
    every smaller-budget result a subset of every larger-budget result.
    Selection uses only captured draft scores.
    """

    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if budget < 1:
        raise ValueError("total-node budget must include at least the root")
    score_maps = _score_maps(positions)
    nodes: set[Node] = {()}
    active: list[Node] = [()]
    for index, position in enumerate(positions):
        token_ids = [int(item["token_id"]) for item in position["candidates"]]
        expanded = [prefix + (token,) for prefix in active for token in token_ids]
        expanded.sort(key=lambda prefix: _priority(prefix, score_maps))
        active = expanded[:beam_width]
        remaining = budget - len(nodes)
        if remaining <= 0:
            break
        nodes.update(active[:remaining])
        if remaining < len(active):
            break
    return validate_prefix_closed(nodes, budget=budget)


def per_depth_policy_state(
    nodes: set[Node],
    positions: list[dict[str, Any]],
    reference: list[int],
) -> list[dict[str, Any]]:
    """Evaluate target-labelled survival after a target-blind tree is built."""

    score_maps = _score_maps(positions)
    target_prefix: Node = ()
    prefix_available = True
    first_pruned: int | None = None
    rows = []
    for depth, target_token in enumerate(reference, 1):
        target_token = int(target_token)
        token_available = target_token in score_maps[depth - 1]
        prefix_available = prefix_available and token_available
        target_prefix += (target_token,)
        survived = prefix_available and target_prefix in nodes
        if prefix_available and not survived and first_pruned is None:
            first_pruned = depth
        same_depth = [node for node in nodes if len(node) == depth]
        cutoff = None
        target_score = None
        gap = None
        if same_depth:
            cutoff = min(
                sum(score_maps[index][token] for index, token in enumerate(node))
                / depth
                for node in same_depth
            )
        if prefix_available:
            target_score = (
                sum(
                    score_maps[index][token]
                    for index, token in enumerate(target_prefix)
                )
                / depth
            )
            gap = target_score - cutoff if cutoff is not None else None
        reason = (
            "target_path_survived"
            if survived
            else "candidate_unavailable"
            if not prefix_available
            else "candidate_available_but_pruned"
        )
        rows.append(
            {
                "target_prefix_survived": survived,
                "first_pruned_depth": first_pruned,
                "pruning_reason": reason,
                "policy_cutoff_normalized_score": cutoff,
                "target_prefix_normalized_score": target_score,
                "target_prefix_score_gap_to_cutoff": gap,
            }
        )
    return rows


def minimum_survival_budget(
    survival_by_budget: dict[int, int], *, depth: int
) -> dict[str, Any]:
    """Return the first replayed root-inclusive budget preserving a depth."""

    ordered = sorted(survival_by_budget.items())
    coverages = [coverage for _, coverage in ordered]
    if coverages != sorted(coverages):
        raise RuntimeError("policy target survival is not monotonic with node budget")
    budget = next((budget for budget, coverage in ordered if coverage >= depth), None)
    maximum = ordered[-1][0]
    return {
        "minimum_total_tree_node_budget_to_preserve_target_prefix": budget,
        "minimum_budget_label": str(budget) if budget is not None else f">{maximum}",
        "minimum_budget_censored": budget is None,
        "maximum_replayed_budget": maximum,
        "root_included_in_budget": True,
    }
