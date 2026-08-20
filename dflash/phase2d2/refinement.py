"""Target-blind conditional-remask construction and fixed-budget tree selection.

This module deliberately has no Torch import.  The materialized custom model uses
the same pure functions, while tensor creation remains in ``dflash.py`` so importing
the Phase 2D.2A CLI cannot initialize a model framework.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

Node = tuple[int, ...]
MAXIMUM_DRAFT_DEPTH = 15
BLOCK_SIZE = 16
BASE_TOP_M = 16
CANDIDATE_MERGE_MODES = ("base_only", "conditional_rerank", "union")
VERIFICATION_MODES = ("packed_native", "always_replay", "margin_guarded_replay")
REFINE_DEPTHS = (1, 2, 4)
REFINED_PREFIX_BUDGETS = (1, 2, 4)
CONDITIONAL_TOP_RS = (2, 4)
NODE_BUDGETS = (16, 32)
MISSING_LOGIT_POLICY = (
    "Full-vocabulary base and conditional log probabilities are gathered for every "
    "merged token. Missing logits are invalid and are never replaced by zero."
)
MERGED_SCORE_RULE = (
    "prefix cumulative merged score + (1-lambda)*base_logprob + "
    "lambda*conditional_logprob for refined descendants; base logprob otherwise"
)
TIE_BREAKING_RULE = (
    "higher mean cumulative merged score, higher cumulative merged score, "
    "shorter depth, canonical token-ID-prefix lexicographic order"
)
FORBIDDEN_PREFIX_FIELDS = (
    "target",
    "accepted",
    "oracle",
    "posterior",
    "gold",
    "reference",
    "correction",
    "final_output",
)


def validate_refinement_options(
    *,
    enabled: bool,
    merge_mode: str,
    base_top_m: int,
    conditional_top_r: int,
    refine_depth: int,
    refined_prefix_budget: int,
    refinement_weight: float,
    node_budget: int,
    block_size: int,
    external_batch_size: int,
    verification_mode: str,
    calibration_validated: bool,
) -> dict[str, Any]:
    """Validate the opt-in online contract before either cache can be mutated."""

    if not enabled:
        return {
            "enabled": False,
            "merge_mode": "base_only",
            "verification_mode": "packed_native",
        }
    if merge_mode not in CANDIDATE_MERGE_MODES:
        raise ValueError(f"unsupported candidate merge mode: {merge_mode}")
    if verification_mode not in VERIFICATION_MODES:
        raise ValueError(f"unsupported robust verification mode: {verification_mode}")
    if int(base_top_m) != BASE_TOP_M:
        raise ValueError("Phase 2D.2A requires base Top-16")
    if int(conditional_top_r) not in CONDITIONAL_TOP_RS:
        raise ValueError("conditional Top-R must be 2 or 4")
    if int(refine_depth) not in REFINE_DEPTHS:
        raise ValueError("refine depth must be 1, 2, or 4")
    if int(refined_prefix_budget) not in REFINED_PREFIX_BUDGETS:
        raise ValueError("refined prefix budget must be 1, 2, or 4")
    if int(node_budget) not in NODE_BUDGETS:
        raise ValueError("Phase 2D.2A node budget must be 16 or 32")
    if int(block_size) != BLOCK_SIZE:
        raise ValueError("Phase 2D.2A requires the pinned 16-token DFlash block")
    if int(external_batch_size) != 1:
        raise ValueError("Phase 2D.2A supports external request batch size one")
    weight = float(refinement_weight)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("refinement weight must be finite and in [0, 1]")
    if verification_mode == "margin_guarded_replay" and not calibration_validated:
        raise ValueError(
            "margin_guarded_replay requires a provenance-matched NR calibration"
        )
    return {
        "enabled": True,
        "merge_mode": merge_mode,
        "base_top_m": int(base_top_m),
        "conditional_top_r": int(conditional_top_r),
        "refine_depth": int(refine_depth),
        "refined_prefix_budget": int(refined_prefix_budget),
        "refinement_weight": weight,
        "node_budget": int(node_budget),
        "block_size": int(block_size),
        "external_batch_size": int(external_batch_size),
        "verification_mode": verification_mode,
        "calibration_validated": bool(calibration_validated),
    }


def build_conditional_remask_inputs(
    *,
    root_token_id: int,
    prefixes: Iterable[Iterable[int]],
    mask_token_id: int,
    block_size: int = BLOCK_SIZE,
) -> list[list[int]]:
    """Return internal branch rows ``[root, prefix..., MASK...]``.

    Prefix elements map to draft positions 1..d.  The root is represented once at
    position zero and is never copied into the prefix.
    """

    if int(block_size) != BLOCK_SIZE:
        raise ValueError("conditional-remask expects a 16-token block")
    rows = []
    seen: set[Node] = set()
    for raw_prefix in prefixes:
        prefix = tuple(int(token) for token in raw_prefix)
        if not prefix or len(prefix) >= block_size:
            raise ValueError("conditional prefix depth must be in [1, 15]")
        if prefix in seen:
            raise ValueError("conditional prefixes must be unique")
        seen.add(prefix)
        row = [int(root_token_id), *prefix]
        row.extend([int(mask_token_id)] * (block_size - len(row)))
        if len(row) != block_size:
            raise RuntimeError("conditional-remask row length invariant failed")
        rows.append(row)
    return rows


def _candidate_maps(positions: list[dict[str, Any]]) -> list[dict[int, dict[str, Any]]]:
    if len(positions) != MAXIMUM_DRAFT_DEPTH:
        raise ValueError("Phase 2D.2A requires 15 base candidate positions")
    result = []
    for expected_position, position in enumerate(positions, 1):
        if int(position.get("position_index", expected_position)) != expected_position:
            raise ValueError("candidate positions are not contiguous")
        candidates = position.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("every base position requires candidates")
        mapping = {int(row["token_id"]): row for row in candidates}
        if len(mapping) != len(candidates):
            raise ValueError("one candidate position contains duplicate token IDs")
        result.append(mapping)
    return result


def _base_prefix_score(prefix: Node, maps: list[dict[int, dict[str, Any]]]) -> float:
    try:
        return sum(
            float(maps[index][token]["log_probability"])
            for index, token in enumerate(prefix)
        )
    except KeyError as exc:
        raise ValueError("seed tree contains a token outside base Top-M") from exc


def _prefix_priority(prefix: Node, cumulative: float) -> tuple[Any, ...]:
    return (-cumulative / len(prefix), -cumulative, len(prefix), prefix)


def _validate_target_blind_record(record: Mapping[str, Any]) -> None:
    leaking = sorted(
        key
        for key in record
        if any(fragment in key.lower() for fragment in FORBIDDEN_PREFIX_FIELDS)
    )
    if leaking:
        raise ValueError(f"target-dependent fields entered prefix selection: {leaking}")


def select_refinement_prefixes(
    *,
    seed_nodes: Iterable[Node],
    base_positions: list[dict[str, Any]],
    refine_depth: int,
    prefix_budget: int,
) -> list[dict[str, Any]]:
    """Rank eligible seed-tree prefixes using draft-only scores and stable ties."""

    if int(refine_depth) not in REFINE_DEPTHS:
        raise ValueError("refine depth must be 1, 2, or 4")
    if int(prefix_budget) not in REFINED_PREFIX_BUDGETS:
        raise ValueError("prefix budget must be 1, 2, or 4")
    nodes = {tuple(int(token) for token in node) for node in seed_nodes}
    if () not in nodes or any(node and node[:-1] not in nodes for node in nodes):
        raise ValueError("seed tree must be prefix closed")
    maps = _candidate_maps(base_positions)
    eligible = []
    for prefix in nodes:
        if len(prefix) != int(refine_depth):
            continue
        cumulative = _base_prefix_score(prefix, maps)
        terminal = maps[len(prefix) - 1][prefix[-1]]
        record = {
            "prefix_token_ids": list(prefix),
            "refine_depth": len(prefix),
            "base_cumulative_log_probability": cumulative,
            "base_mean_log_probability": cumulative / len(prefix),
            "terminal_base_rank": int(terminal["rank"]),
            "selection_uses_target_information": False,
            "selection_features": (
                "draft_log_probability, frozen seed-tree membership, depth, "
                "canonical token-ID-prefix tie"
            ),
        }
        eligible.append((prefix, cumulative, record))
    eligible.sort(key=lambda row: _prefix_priority(row[0], row[1]))
    selected = []
    for rank, (_, _, record) in enumerate(eligible[: int(prefix_budget)], 1):
        selected.append({**record, "refined_prefix_selection_rank": rank})
    return selected


def _closure(nodes: Iterable[Node]) -> set[Node]:
    result = {()}
    for node in nodes:
        for depth in range(1, len(node) + 1):
            result.add(node[:depth])
    return result


def _node_priority(record: dict[str, Any]) -> tuple[Any, ...]:
    path = tuple(record["path"])
    cumulative = float(record["cumulative_merged_score"])
    return _prefix_priority(path, cumulative)


def _base_node_records(
    nodes: set[Node], base_maps: list[dict[int, dict[str, Any]]]
) -> dict[Node, dict[str, Any]]:
    records: dict[Node, dict[str, Any]] = {
        (): {
            "path": [],
            "parent_path": None,
            "depth": 0,
            "token_id": None,
            "candidate_provenance": "root",
            "positional_merged_score": 0.0,
            "cumulative_merged_score": 0.0,
        }
    }
    for node in sorted(nodes - {()}, key=lambda value: (len(value), value)):
        candidate = base_maps[len(node) - 1][node[-1]]
        positional = float(candidate["log_probability"])
        records[node] = {
            "path": list(node),
            "parent_path": list(node[:-1]),
            "depth": len(node),
            "token_id": int(node[-1]),
            "candidate_provenance": "base_only",
            "base_rank": int(candidate["rank"]),
            "conditional_rank": None,
            "base_log_probability": positional,
            "conditional_log_probability": None,
            "positional_merged_score": positional,
            "cumulative_merged_score": (
                records[node[:-1]]["cumulative_merged_score"] + positional
            ),
            "parent_refined_prefix": None,
            "refine_depth": None,
            "refined_prefix_selection_rank": None,
        }
    return records


def _validate_merged_candidate(candidate: Mapping[str, Any]) -> None:
    for field in ("token_id", "base_log_probability", "conditional_log_probability"):
        if candidate.get(field) is None:
            raise ValueError(f"merged candidate lacks full-vocabulary {field}")
    if not all(
        math.isfinite(float(candidate[field]))
        for field in ("base_log_probability", "conditional_log_probability")
    ):
        raise ValueError("merged candidate log probabilities must be finite")


def build_refined_tree(
    *,
    seed_nodes: Iterable[Node],
    base_positions: list[dict[str, Any]],
    selected_prefixes: list[dict[str, Any]],
    conditional_candidates: Mapping[tuple[int, ...], list[list[dict[str, Any]]]],
    merge_mode: str,
    refinement_weight: float,
    node_budget: int,
) -> dict[str, Any]:
    """Rebuild a deterministic prefix-closed tree under the same root-inclusive N.

    ``conditional_candidates[prefix][position-1]`` contains the exact bounded
    merged token pool for that conditional branch.  Each row carries both gathered
    full-vocabulary log probabilities, so no absent score is treated as zero.
    """

    if merge_mode not in CANDIDATE_MERGE_MODES:
        raise ValueError(f"unsupported candidate merge mode: {merge_mode}")
    if int(node_budget) not in NODE_BUDGETS:
        raise ValueError("node budget must be 16 or 32")
    weight = float(refinement_weight)
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("refinement weight must be in [0, 1]")
    seed = {tuple(int(token) for token in node) for node in seed_nodes}
    if () not in seed or len(seed) > node_budget:
        raise ValueError("invalid seed-tree root or budget")
    if any(node and node[:-1] not in seed for node in seed):
        raise ValueError("seed tree must be prefix closed")
    base_maps = _candidate_maps(base_positions)
    base_records = _base_node_records(seed, base_maps)
    if merge_mode == "base_only" or not selected_prefixes:
        ordered = [()] + sorted(
            seed - {()},
            key=lambda node: (len(node), _node_priority(base_records[node])),
        )
        return _finalize_tree(
            ordered=ordered,
            records=base_records,
            requested_node_budget=node_budget,
            merge_mode=merge_mode,
            available_candidate_count=len(seed) - 1,
            merged_candidate_count=0,
            selected_prefixes=selected_prefixes,
        )

    prefix_records = {
        tuple(int(token) for token in row["prefix_token_ids"]): row
        for row in selected_prefixes
    }
    if len(prefix_records) != len(selected_prefixes):
        raise ValueError("selected refinement prefixes must be unique")
    if any(prefix not in seed for prefix in prefix_records):
        raise ValueError("every refinement prefix must belong to the seed tree")
    if set(prefix_records) != set(conditional_candidates):
        raise ValueError("conditional candidate batches do not match selected prefixes")

    pool = set(seed)
    records = dict(base_records)
    merged_event_count = 0
    for prefix, selection in sorted(
        prefix_records.items(),
        key=lambda item: item[1]["refined_prefix_selection_rank"],
    ):
        per_position = conditional_candidates[prefix]
        if len(per_position) != MAXIMUM_DRAFT_DEPTH:
            raise ValueError("conditional candidate record must cover 15 positions")
        active = [prefix]
        for depth in range(len(prefix) + 1, MAXIMUM_DRAFT_DEPTH + 1):
            expansions: list[tuple[Node, dict[str, Any]]] = []
            for parent in active:
                parent_score = float(records[parent]["cumulative_merged_score"])
                for candidate in per_position[depth - 1]:
                    _validate_merged_candidate(candidate)
                    in_base = candidate.get("base_rank") is not None
                    in_conditional = candidate.get("conditional_rank") is not None
                    if merge_mode == "conditional_rerank" and not in_base:
                        continue
                    provenance = (
                        "base_and_refined"
                        if in_base and in_conditional
                        else "base_only"
                        if in_base
                        else "refined_only"
                    )
                    positional = (1.0 - weight) * float(
                        candidate["base_log_probability"]
                    ) + weight * float(candidate["conditional_log_probability"])
                    node = parent + (int(candidate["token_id"]),)
                    event = {
                        "path": list(node),
                        "parent_path": list(parent),
                        "depth": depth,
                        "token_id": int(candidate["token_id"]),
                        "candidate_provenance": provenance,
                        "base_rank": candidate.get("base_rank"),
                        "conditional_rank": candidate.get("conditional_rank"),
                        "base_rank_unavailable_reason": (
                            None if in_base else "outside_captured_base_top_m"
                        ),
                        "conditional_rank_unavailable_reason": (
                            None if in_conditional else "outside_conditional_top_r"
                        ),
                        "base_log_probability": float(
                            candidate["base_log_probability"]
                        ),
                        "conditional_log_probability": float(
                            candidate["conditional_log_probability"]
                        ),
                        "positional_merged_score": positional,
                        "cumulative_merged_score": parent_score + positional,
                        "parent_refined_prefix": list(prefix),
                        "refine_depth": len(prefix),
                        "refined_prefix_selection_rank": int(
                            selection["refined_prefix_selection_rank"]
                        ),
                    }
                    expansions.append((node, event))
            expansions.sort(key=lambda item: _node_priority(item[1]))
            # Bound speculative host work without changing the target budget.  At
            # most N frontier nodes per refined prefix can ever enter an N-node tree.
            active = []
            for node, event in expansions[:node_budget]:
                if node not in records or _node_priority(event) < _node_priority(
                    records[node]
                ):
                    records[node] = event
                pool.add(node)
                active.append(node)
                merged_event_count += 1

    mandatory = _closure(prefix_records)
    if len(mandatory) > node_budget:
        raise ValueError("selected prefix closure exceeds target node budget")
    selected = set(mandatory)
    while len(selected) < node_budget:
        available = [node for node in pool - selected if node and node[:-1] in selected]
        if not available:
            break
        available.sort(key=lambda node: _node_priority(records[node]))
        selected.add(available[0])
    if any(node and node[:-1] not in selected for node in selected):
        raise RuntimeError("refined tree selection lost prefix closure")
    ordered = [()] + sorted(
        selected - {()}, key=lambda node: (len(node), _node_priority(records[node]))
    )
    return _finalize_tree(
        ordered=ordered,
        records=records,
        requested_node_budget=node_budget,
        merge_mode=merge_mode,
        available_candidate_count=len(pool) - 1,
        merged_candidate_count=merged_event_count,
        selected_prefixes=selected_prefixes,
    )


def _finalize_tree(
    *,
    ordered: list[Node],
    records: dict[Node, dict[str, Any]],
    requested_node_budget: int,
    merge_mode: str,
    available_candidate_count: int,
    merged_candidate_count: int,
    selected_prefixes: list[dict[str, Any]],
) -> dict[str, Any]:
    if not ordered or ordered[0] != () or len(ordered) > requested_node_budget:
        raise RuntimeError("refined tree violates root-inclusive node budget")
    indices = {path: index for index, path in enumerate(ordered)}
    rows = []
    for node in ordered:
        source = records[node]
        rows.append(
            {
                **source,
                "node_index": indices[node],
                "parent_index": indices[node[:-1]] if node else None,
                "normalized_merged_score": (
                    float(source["cumulative_merged_score"]) / len(node)
                    if node
                    else None
                ),
            }
        )
    counts = Counter(row["depth"] for row in rows)
    maximum_depth = max(counts, default=0)
    refined_only_selected = sum(
        row.get("candidate_provenance") == "refined_only" for row in rows
    )
    return {
        "schema_version": "1.0",
        "merge_mode": merge_mode,
        "requested_node_budget": int(requested_node_budget),
        "actual_total_node_count": len(rows),
        "root_count": 1,
        "unused_node_budget": int(requested_node_budget) - len(rows),
        "unused_node_budget_reason": (
            None
            if len(rows) == requested_node_budget
            else "candidate_pool_exhausted_under_prefix_closure"
        ),
        "available_candidate_count": int(available_candidate_count),
        "merged_candidate_count": int(merged_candidate_count),
        "refined_only_nodes_selected": refined_only_selected,
        "maximum_depth": maximum_depth,
        "per_depth_node_counts": [
            counts.get(depth, 0) for depth in range(maximum_depth + 1)
        ],
        "selected_prefixes": selected_prefixes,
        "score_rule": MERGED_SCORE_RULE,
        "missing_logit_policy": MISSING_LOGIT_POLICY,
        "tie_breaking": TIE_BREAKING_RULE,
        "nodes": rows,
    }


def audit_prefix_selection_inputs(runtime_inputs: Mapping[str, Any]) -> None:
    """Reject target-side fields at the public refinement-selection boundary."""

    _validate_target_blind_record(runtime_inputs)
