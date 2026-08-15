"""Deterministic bounded candidate-tree policies for offline Phase 2B replay."""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from typing import Any

Node = tuple[int, ...]


def _candidate_maps(
    positions: list[dict[str, Any]],
) -> tuple[list[list[tuple[int, float, int]]], list[dict[int, float]]]:
    ordered = []
    scores = []
    for position in positions:
        candidates = [
            (
                int(candidate["token_id"]),
                float(candidate["log_probability"]),
                int(candidate["rank"]),
            )
            for candidate in position["candidates"]
        ]
        ordered.append(candidates)
        scores.append({token_id: score for token_id, score, _ in candidates})
    return ordered, scores


def validate_prefix_closed(nodes: Iterable[Node], *, budget: int) -> set[Node]:
    tree = set(nodes)
    if () not in tree:
        raise ValueError("candidate tree must contain the root")
    if len(tree) > budget:
        raise ValueError("candidate tree exceeds its total-node budget")
    for node in tree:
        if node and node[:-1] not in tree:
            raise ValueError("candidate tree is not prefix closed")
    return tree


def top1_chain_tree(positions: list[dict[str, Any]], *, budget: int = 16) -> set[Node]:
    """Build the root plus exact Top-1 chain; total size is always depth+1."""

    if budget < len(positions) + 1:
        raise ValueError("Top-1 chain does not fit the requested total-node budget")
    prefix: Node = ()
    nodes = {prefix}
    for position in positions:
        prefix += (int(position["candidates"][0]["token_id"]),)
        nodes.add(prefix)
    return validate_prefix_closed(nodes, budget=budget)


def _prefix_score(prefix: Node, score_maps: list[dict[int, float]]) -> float:
    return sum(score_maps[index][token] for index, token in enumerate(prefix))


def _priority(prefix: Node, score_maps: list[dict[int, float]]) -> tuple[Any, ...]:
    cumulative = _prefix_score(prefix, score_maps)
    normalized = cumulative / len(prefix)
    # Higher normalized and cumulative score first; stable structural ties use
    # shorter depth and lexicographically ascending token IDs.
    return (-normalized, -cumulative, len(prefix), prefix)


def _bounded_expand(
    *,
    positions: list[dict[str, Any]],
    budget: int,
    child_selector: Any,
    initial_selector: Any,
) -> set[Node]:
    ordered, score_maps = _candidate_maps(positions)
    nodes = top1_chain_tree(positions, budget=budget)
    if len(nodes) == budget:
        return nodes
    top1 = tuple(candidates[0][0] for candidates in ordered)
    frontier: list[tuple[Any, ...]] = []
    queued: set[Node] = set()

    def queue(prefix: Node) -> None:
        if prefix not in nodes and prefix not in queued:
            heapq.heappush(frontier, (*_priority(prefix, score_maps), prefix))
            queued.add(prefix)

    for depth, candidates in enumerate(ordered, 1):
        parent = top1[: depth - 1]
        for token_id, _, rank in initial_selector(depth, candidates):
            if rank != 1:
                queue(parent + (token_id,))

    while frontier and len(nodes) < budget:
        *_, prefix = heapq.heappop(frontier)
        queued.discard(prefix)
        if prefix in nodes or prefix[:-1] not in nodes:
            continue
        nodes.add(prefix)
        depth = len(prefix)
        if depth < len(ordered):
            for token_id, _, _ in child_selector(depth + 1, ordered[depth]):
                queue(prefix + (token_id,))
    return validate_prefix_closed(nodes, budget=budget)


def score_beam_tree(positions: list[dict[str, Any]], *, budget: int) -> set[Node]:
    """Top-1 backbone plus deterministic best-first length-normalized branches."""

    return _bounded_expand(
        positions=positions,
        budget=budget,
        initial_selector=lambda _depth, candidates: candidates,
        child_selector=lambda _depth, candidates: candidates,
    )


def confidence_gated_tree(
    positions: list[dict[str, Any]],
    *,
    budget: int,
    margin_threshold: float,
    branch_width: int = 4,
) -> set[Node]:
    """Branch only where draft Top-1/Top-2 log-probability margin is small."""

    if margin_threshold < 0:
        raise ValueError("confidence margin threshold must be non-negative")
    if branch_width < 2:
        raise ValueError("confidence branch width must be at least two")

    def gated(depth: int, candidates: list[tuple[int, float, int]]):
        margin = candidates[0][1] - candidates[1][1]
        return candidates[:branch_width] if margin <= margin_threshold else candidates[:1]

    return _bounded_expand(
        positions=positions,
        budget=budget,
        initial_selector=gated,
        child_selector=gated,
    )


def oracle_prefix_closed_tree(
    positions: list[dict[str, Any]],
    reference: list[int],
    *,
    budget: int,
) -> set[Node]:
    """Target-informed structural upper bound under the same total-node budget."""

    nodes: set[Node] = {()}
    prefix: Node = ()
    for depth, target_token in enumerate(reference[: len(positions)], 1):
        if len(nodes) >= budget:
            break
        candidate_ids = {
            int(candidate["token_id"])
            for candidate in positions[depth - 1]["candidates"]
        }
        if int(target_token) not in candidate_ids:
            break
        prefix += (int(target_token),)
        nodes.add(prefix)
    return validate_prefix_closed(nodes, budget=budget)


def brute_force_oracle_coverage(
    positions: list[dict[str, Any]], reference: list[int], *, budget: int
) -> int:
    """Exhaustively validate tiny synthetic oracle cases used by unit tests."""

    all_nodes: set[Node] = {()}
    parents = {()}
    for position in positions:
        token_ids = [int(item["token_id"]) for item in position["candidates"]]
        parents = {parent + (token_id,) for parent in parents for token_id in token_ids}
        all_nodes.update(parents)
    candidate_nodes = sorted(all_nodes - {()}, key=lambda node: (len(node), node))
    if len(candidate_nodes) > 20:
        raise ValueError("brute-force oracle is limited to 20 synthetic nodes")
    best = 0
    for mask in range(1 << len(candidate_nodes)):
        selected = {()}
        selected.update(
            node for index, node in enumerate(candidate_nodes) if mask & (1 << index)
        )
        if len(selected) > budget or any(
            node and node[:-1] not in selected for node in selected
        ):
            continue
        best = max(best, covered_target_prefix_depth(selected, reference))
    return best


def covered_target_prefix_depth(nodes: Iterable[Node], reference: list[int]) -> int:
    tree = set(nodes)
    depth = 0
    for next_depth in range(1, len(reference) + 1):
        if tuple(int(value) for value in reference[:next_depth]) not in tree:
            break
        depth = next_depth
    return depth


def tree_shape(nodes: Iterable[Node]) -> dict[str, int | float]:
    tree = set(nodes)
    leaves = [
        node
        for node in tree
        if node and not any(
            len(other) == len(node) + 1 and other[:-1] == node for other in tree
        )
    ]
    candidate_nodes = len(tree) - 1
    unshared_path_slots = sum(len(leaf) for leaf in leaves)
    sharing = (
        1.0 - candidate_nodes / unshared_path_slots if unshared_path_slots else 0.0
    )
    return {
        "total_node_count": len(tree),
        "candidate_node_count": candidate_nodes,
        "leaf_count": len(leaves),
        "maximum_depth": max((len(node) for node in tree), default=0),
        "prefix_sharing_ratio": sharing,
    }


def target_candidate_rank(position: dict[str, Any], target_token: int) -> int | None:
    for candidate in position["candidates"]:
        if int(candidate["token_id"]) == int(target_token):
            return int(candidate["rank"])
    return None


def policy_event(
    *,
    capture: dict[str, Any],
    nodes: set[Node],
    policy: str,
    policy_class: str,
    policy_variant: str,
    requested_budget: int,
    policy_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Compute coverage-only event metrics; no online acceptance is inferred."""

    reference = [int(value) for value in capture["reference_target_continuation_token_ids"]]
    coverage = covered_target_prefix_depth(nodes, reference)
    baseline_accepted = int(capture["accepted_draft_token_count"])
    baseline_observable = min(baseline_accepted, len(reference))
    first_rejection_eligible = baseline_accepted < len(reference)
    shape = tree_shape(nodes)
    depths = {}
    for depth in (1, 2, 4, 8, 15):
        eligible = len(reference) >= depth
        depths[str(depth)] = {
            "eligible": eligible,
            "covered": coverage >= depth if eligible else None,
        }
    target_candidate_ranks = [
        target_candidate_rank(capture["positions"][index], target_token)
        for index, target_token in enumerate(reference)
    ]
    return {
        "schema_version": "1.0",
        **{
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
                "round_output_start_offset",
            )
        },
        "policy": policy,
        "policy_class": policy_class,
        "oracle": policy_class == "oracle",
        "deployable": policy_class == "deployable",
        "policy_variant": policy_variant,
        "policy_uses_target_information": policy_class == "oracle",
        "policy_parameters": policy_parameters,
        "requested_total_tree_node_budget": int(requested_budget),
        **shape,
        "total_tree_nodes": shape["total_node_count"],
        "draft_candidate_nodes": shape["candidate_node_count"],
        "baseline_accepted_draft_token_count": baseline_accepted,
        "baseline_observable_top1_covered_depth": baseline_observable,
        "covered_target_prefix_depth": coverage,
        "committed_tokens_upper_bound": 1 + coverage,
        "committed_tokens_upper_bound_semantics": (
            "coverage-derived root plus covered continuation; not an online result"
        ),
        "extra_covered_depth_vs_top1": coverage - baseline_observable,
        "extra_covered_tokens": coverage - baseline_observable,
        "first_rejection_recovery_eligible": first_rejection_eligible,
        "first_rejection_recovered": (
            coverage > baseline_accepted if first_rejection_eligible else None
        ),
        "depth_recall": depths,
        "full_depth_evaluation_eligible": len(reference) == 15,
        "full_depth_covered": coverage == 15 if len(reference) == 15 else None,
        "target_path_candidate_ranks": target_candidate_ranks,
        "target_path_covered_candidate_count": sum(
            rank is not None for rank in target_candidate_ranks
        ),
        "target_path_observed_token_count": len(reference),
        "target_path_candidate_recall": (
            sum(rank is not None for rank in target_candidate_ranks) / len(reference)
            if reference
            else None
        ),
        "covered_target_tokens_per_candidate_node": (
            coverage / shape["candidate_node_count"]
            if shape["candidate_node_count"]
            else None
        ),
        "terminal_round": bool(capture["terminal_round"]),
        "reference_right_censored": bool(capture["reference_right_censored"]),
        "reference_observed_depth": len(reference),
        "truncated_token_count": int(capture["truncated_token_count"]),
        "deployable_oracle_coverage_gap": None,
        "oracle_improvable_but_deployable_not": None,
        "interpretation": "offline coverage feasibility only",
        "actual_accepted_length_derived": False,
        "latency_or_speedup_derived": False,
    }
