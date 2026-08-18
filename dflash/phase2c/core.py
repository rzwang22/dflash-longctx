"""Framework-lazy Phase 2C tree construction, masking, traversal, and cache helpers.

This module intentionally does not import Torch.  The repository imports the pure
helpers in CPU-only tests, while materialization copies this exact file beside the
remote custom model as ``phase2c_online.py``.  Tensor helpers receive the active
Torch module explicitly, preserving CLI lazy-import behavior.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Iterable
from typing import Any

Node = tuple[int, ...]
MAXIMUM_DRAFT_DEPTH = 15
SUPPORTED_BEAM_WIDTHS = (2, 4, 8, 16, 32)
SUPPORTED_NODE_BUDGETS = (16, 32, 64, 128)
SCORE_RULE = "mean cumulative captured draft log probability"
TIE_BREAKING_RULE = (
    "higher normalized score, higher cumulative score, shorter depth, "
    "lexicographically ascending token IDs"
)


def _score_maps(positions: list[dict[str, Any]]) -> list[dict[int, float]]:
    maps = []
    for position in positions:
        candidates = position["candidates"]
        token_ids = [int(candidate["token_id"]) for candidate in candidates]
        if len(token_ids) != len(set(token_ids)):
            raise ValueError("one parent cannot contain duplicate token children")
        maps.append(
            {
                int(candidate["token_id"]): float(candidate["log_probability"])
                for candidate in candidates
            }
        )
    return maps


def _priority(prefix: Node, score_maps: list[dict[int, float]]) -> tuple[Any, ...]:
    cumulative = sum(score_maps[index][token] for index, token in enumerate(prefix))
    return (-cumulative / len(prefix), -cumulative, len(prefix), prefix)


def validate_tree(
    nodes: Iterable[Node], *, budget: int, maximum_depth: int = MAXIMUM_DRAFT_DEPTH
) -> set[Node]:
    """Validate root inclusion, budget, depth, prefix closure, and sibling uniqueness."""

    tree = set(nodes)
    if () not in tree:
        raise ValueError("candidate tree must contain the root")
    if len(tree) > budget:
        raise ValueError("candidate tree exceeds its root-inclusive node budget")
    if any(len(node) > maximum_depth for node in tree):
        raise ValueError("candidate tree exceeds maximum draft depth")
    for node in tree:
        if node and node[:-1] not in tree:
            raise ValueError("candidate tree is not prefix closed")
    siblings: dict[Node, list[int]] = {}
    for node in tree:
        if node:
            siblings.setdefault(node[:-1], []).append(node[-1])
    if any(len(tokens) != len(set(tokens)) for tokens in siblings.values()):
        raise ValueError("one parent cannot contain duplicate token children")
    return tree


def top1_chain_tree(positions: list[dict[str, Any]], *, budget: int = 16) -> set[Node]:
    """Return the exact root plus marginal Top-1 chain."""

    if len(positions) != MAXIMUM_DRAFT_DEPTH:
        raise ValueError("Phase 2C requires exactly 15 draft positions")
    if budget != MAXIMUM_DRAFT_DEPTH + 1:
        raise ValueError("Phase 2C top1_chain requires root-inclusive budget 16")
    prefix: Node = ()
    nodes: set[Node] = {prefix}
    for position in positions:
        if not position.get("candidates"):
            raise ValueError("every draft position requires candidates")
        prefix += (int(position["candidates"][0]["token_id"]),)
        nodes.add(prefix)
    return validate_tree(nodes, budget=budget)


def layerwise_beam_tree(
    positions: list[dict[str, Any]], *, budget: int, beam_width: int
) -> set[Node]:
    """Canonical target-blind Phase 2B.1/2C layerwise marginal policy."""

    if len(positions) > MAXIMUM_DRAFT_DEPTH:
        raise ValueError("candidate positions exceed maximum draft depth")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if budget < 1:
        raise ValueError("node budget must include at least the root")
    score_maps = _score_maps(positions)
    nodes: set[Node] = {()}
    active: list[Node] = [()]
    for position in positions:
        token_ids = [int(candidate["token_id"]) for candidate in position["candidates"]]
        expanded = [prefix + (token,) for prefix in active for token in token_ids]
        expanded.sort(key=lambda prefix: _priority(prefix, score_maps))
        active = expanded[:beam_width]
        remaining = budget - len(nodes)
        if remaining <= 0:
            break
        nodes.update(active[:remaining])
        if remaining < len(active):
            break
    return validate_tree(nodes, budget=budget)


def flatten_tree(
    nodes: Iterable[Node],
    positions: list[dict[str, Any]],
    *,
    root_token_id: int,
    policy: str,
    requested_node_budget: int,
    beam_width: int,
) -> dict[str, Any]:
    """Flatten a tree into deterministic depth/score order with parents first."""

    tree = validate_tree(nodes, budget=requested_node_budget)
    score_maps = _score_maps(positions)
    ordered = [()] + sorted(
        tree - {()}, key=lambda node: (len(node), _priority(node, score_maps))
    )
    index_by_path = {path: index for index, path in enumerate(ordered)}
    records = []
    for index, path in enumerate(ordered):
        cumulative = (
            sum(score_maps[offset][token] for offset, token in enumerate(path))
            if path
            else 0.0
        )
        records.append(
            {
                "node_index": index,
                "parent_index": index_by_path[path[:-1]] if path else None,
                "depth": len(path),
                "token_id": int(path[-1]) if path else int(root_token_id),
                "positional_draft_log_probability": (
                    score_maps[len(path) - 1][path[-1]] if path else None
                ),
                "cumulative_score": cumulative,
                "normalized_score": cumulative / len(path) if path else None,
                "draft_token_path": list(path),
                "root_to_node_token_path": [int(root_token_id), *path],
            }
        )
    depth_counts = Counter(record["depth"] for record in records)
    return {
        "policy": policy,
        "requested_node_budget": int(requested_node_budget),
        "actual_total_node_count": len(records),
        "beam_width": int(beam_width),
        "maximum_depth": max(depth_counts, default=0),
        "per_depth_node_counts": {
            str(depth): int(depth_counts.get(depth, 0))
            for depth in range(max(depth_counts, default=0) + 1)
        },
        "score_rule": SCORE_RULE,
        "tie_breaking": TIE_BREAKING_RULE,
        "nodes": records,
    }


def build_tree_attention_visibility(
    parent_indices: list[int | None], *, prefix_length: int
) -> list[list[bool]]:
    """Build a pure-Python prefix-plus-ancestor visibility matrix for tests."""

    if prefix_length < 0:
        raise ValueError("prefix_length cannot be negative")
    if not parent_indices or parent_indices[0] is not None:
        raise ValueError("flattened tree index zero must be the root")
    visibility = []
    for query, parent in enumerate(parent_indices):
        ancestors = {query}
        cursor = parent
        while cursor is not None:
            if not 0 <= cursor < query:
                raise ValueError("every parent must precede its child")
            ancestors.add(cursor)
            cursor = parent_indices[cursor]
        visibility.append(
            [True] * prefix_length
            + [key in ancestors for key in range(len(parent_indices))]
        )
    return visibility


def semantic_position_ids(
    context_length_before: int, depths: Iterable[int]
) -> list[int]:
    """Return RoPE positions; packed cache storage indices are intentionally separate."""

    if context_length_before < 0:
        raise ValueError("context_length_before cannot be negative")
    result = [context_length_before + int(depth) for depth in depths]
    if any(position < context_length_before for position in result):
        raise ValueError("tree depth cannot be negative")
    return result


def greedy_tree_traversal(
    nodes: list[dict[str, Any]], target_next_token_by_node: dict[int, int]
) -> dict[str, Any]:
    """Pure greedy equality traversal used for synthetic correctness tests."""

    children: dict[int, dict[int, int]] = {}
    for node in nodes[1:]:
        parent = int(node["parent_index"])
        token = int(node["token_id"])
        if token in children.setdefault(parent, {}):
            raise ValueError("duplicate token child under one parent")
        children[parent][token] = int(node["node_index"])
    current = 0
    accepted = []
    for _ in range(MAXIMUM_DRAFT_DEPTH):
        target_token = int(target_next_token_by_node[current])
        child = children.get(current, {}).get(target_token)
        if child is None:
            return {
                "accepted_node_indices": accepted,
                "accepted_draft_token_count": len(accepted),
                "next_root_token_id": target_token,
                "next_root_kind": ("correction" if children.get(current) else "bonus"),
            }
        accepted.append(child)
        current = child
    target_token = int(target_next_token_by_node[current])
    return {
        "accepted_node_indices": accepted,
        "accepted_draft_token_count": len(accepted),
        "next_root_token_id": target_token,
        "next_root_kind": "bonus",
    }


def path_compaction_storage_indices(
    *,
    prefix_length: int,
    flattened_tree_node_count: int,
    accepted_node_indices: Iterable[int],
) -> list[int]:
    """Return storage gather indices for prefix + root + accepted path."""

    if prefix_length < 0 or flattened_tree_node_count < 1:
        raise ValueError("invalid prefix or flattened tree length")
    accepted = [int(index) for index in accepted_node_indices]
    committed = [0, *accepted]
    if any(not 1 <= index < flattened_tree_node_count for index in accepted):
        raise ValueError("accepted path indices must identify non-root tree nodes")
    if len(committed) != len(set(committed)):
        raise ValueError("accepted path cannot contain duplicate nodes")
    return [
        *range(prefix_length),
        *(prefix_length + index for index in committed),
    ]


def compact_synthetic_cache_layers(
    layers: list[list[Any]],
    *,
    prefix_length: int,
    flattened_tree_node_count: int,
    accepted_node_indices: Iterable[int],
) -> list[list[Any]]:
    """List-backed executable specification used by cache-compaction tests."""

    indices = path_compaction_storage_indices(
        prefix_length=prefix_length,
        flattened_tree_node_count=flattened_tree_node_count,
        accepted_node_indices=accepted_node_indices,
    )
    expected = prefix_length + flattened_tree_node_count
    if not layers or any(len(layer) != expected for layer in layers):
        raise ValueError(
            "every synthetic cache layer must contain prefix + packed tree"
        )
    return [[layer[index] for index in indices] for layer in layers]


def validate_online_tree_options(
    *,
    enabled: bool,
    policy: str,
    node_budget: int,
    beam_width: int,
    draft_top_k: int,
    temperature: float,
    block_size: int,
    batch_size: int,
    return_diagnostics: bool,
    enable_profiling: bool,
    enable_gold_checks: bool,
) -> None:
    """Reject unsupported Phase 2C modes before generation mutates any cache."""

    if not enabled:
        return
    if batch_size != 1:
        raise ValueError("Phase 2C online tree requires batch_size=1")
    if temperature != 0.0:
        raise ValueError("Phase 2C online tree requires temperature=0")
    if block_size != 16:
        raise ValueError("Phase 2C online tree requires block_size=16")
    if policy not in {"top1_chain", "layerwise_beam"}:
        raise ValueError("unsupported Phase 2C tree policy")
    if not 2 <= draft_top_k <= 16:
        raise ValueError("Phase 2C draft_top_k must be between 2 and 16")
    if policy == "top1_chain" and node_budget != 16:
        raise ValueError("top1_chain requires root-inclusive node_budget=16")
    if policy == "layerwise_beam" and node_budget not in SUPPORTED_NODE_BUDGETS:
        raise ValueError("layerwise_beam node budget must be one of 16,32,64,128")
    if policy == "layerwise_beam" and beam_width not in SUPPORTED_BEAM_WIDTHS:
        raise ValueError("layerwise_beam width must be one of 2,4,8,16,32")
    if enable_profiling and not return_diagnostics:
        raise ValueError("Phase 2C profiling requires return_diagnostics=True")
    if enable_gold_checks and not return_diagnostics:
        raise ValueError("Phase 2C gold checks require return_diagnostics=True")
    if enable_gold_checks and enable_profiling:
        raise ValueError("Phase 2C gold checks are excluded from profiling runs")
    if enable_gold_checks and node_budget > 32:
        raise ValueError(
            "Phase 2C gold checks are limited to trees of at most 32 nodes"
        )


def extract_topk_candidates(
    torch_module: Any, draft_logits: Any, *, top_k: int
) -> dict[str, Any]:
    """Extract deterministic Top-K IDs and normalized log probabilities on device."""

    if len(draft_logits.shape) != 3 or int(draft_logits.shape[0]) != 1:
        raise ValueError("draft logits must have shape [1, positions, vocabulary]")
    if int(draft_logits.shape[1]) != MAXIMUM_DRAFT_DEPTH:
        raise ValueError("Phase 2C requires exactly 15 marginal draft positions")
    vocabulary_size = int(draft_logits.shape[2])
    if not 2 <= top_k <= min(16, vocabulary_size):
        raise ValueError("invalid draft Top-K")
    logits = draft_logits[0]
    token_ids = torch_module.argsort(logits, dim=-1, descending=True, stable=True)[
        :, :top_k
    ]
    log_probabilities = torch_module.gather(
        torch_module.log_softmax(logits.float(), dim=-1), 1, token_ids
    )
    return {
        "token_ids": token_ids,
        "log_probabilities": log_probabilities,
        "top_k": int(top_k),
    }


def _stable_tensor_priority_order(
    torch_module: Any, paths: Any, cumulative_scores: Any
) -> Any:
    """Stable score ordering with lexicographic token IDs as the final tie break."""

    order = torch_module.arange(paths.shape[0], device=paths.device)
    for column in range(int(paths.shape[1]) - 1, -1, -1):
        local = torch_module.argsort(paths[order, column], stable=True)
        order = order[local]
    local = torch_module.argsort(cumulative_scores[order], descending=True, stable=True)
    return order[local]


def build_online_tree_tensors(
    torch_module: Any,
    *,
    candidates: dict[str, Any],
    root_token_id: Any,
    policy: str,
    node_budget: int,
    beam_width: int,
) -> dict[str, Any]:
    """Build the canonical marginal tree entirely on the active device."""

    candidate_ids = candidates["token_ids"]
    # Phase 2B persists float32 log probabilities and Phase 2B.1 accumulates
    # their Python-float values. Accumulate the same float32 inputs in float64
    # online so close policy scores cannot reorder solely from float32 summation.
    candidate_scores = candidates["log_probabilities"].to(dtype=torch_module.float64)
    device = candidate_ids.device
    depth_limit, top_k = map(int, candidate_ids.shape)
    if depth_limit != MAXIMUM_DRAFT_DEPTH:
        raise ValueError("Phase 2C requires 15 candidate positions")

    root_path = torch_module.full(
        (1, depth_limit), -1, dtype=torch_module.long, device=device
    )
    paths = [root_path]
    token_ids = [root_token_id.reshape(1).to(dtype=torch_module.long)]
    parent_indices = [
        torch_module.full((1,), -1, dtype=torch_module.long, device=device)
    ]
    depths = [torch_module.zeros(1, dtype=torch_module.long, device=device)]
    positional_scores = [
        torch_module.zeros(1, dtype=candidate_scores.dtype, device=device)
    ]
    cumulative_scores = [
        torch_module.zeros(1, dtype=candidate_scores.dtype, device=device)
    ]

    if policy == "top1_chain":
        prefix = root_path.clone()
        cumulative = torch_module.zeros(1, dtype=candidate_scores.dtype, device=device)
        for depth in range(1, depth_limit + 1):
            prefix = prefix.clone()
            prefix[:, depth - 1] = candidate_ids[depth - 1, 0]
            score = candidate_scores[depth - 1, 0].reshape(1)
            cumulative = cumulative + score
            paths.append(prefix)
            token_ids.append(candidate_ids[depth - 1, 0].reshape(1))
            parent_indices.append(
                torch_module.tensor([depth - 1], dtype=torch_module.long, device=device)
            )
            depths.append(
                torch_module.tensor([depth], dtype=torch_module.long, device=device)
            )
            positional_scores.append(score)
            cumulative_scores.append(cumulative.clone())
        per_depth_counts = [1] * (depth_limit + 1)
    elif policy == "layerwise_beam":
        active_paths = root_path
        active_scores = torch_module.zeros(
            1, dtype=candidate_scores.dtype, device=device
        )
        active_global_indices = torch_module.zeros(
            1, dtype=torch_module.long, device=device
        )
        total_nodes = 1
        per_depth_counts = [1]
        for depth in range(1, depth_limit + 1):
            active_count = int(active_paths.shape[0])
            expanded_paths = (
                active_paths[:, None, :]
                .expand(active_count, top_k, depth_limit)
                .clone()
            )
            expanded_paths[:, :, depth - 1] = candidate_ids[depth - 1][None, :]
            expanded_paths = expanded_paths.reshape(-1, depth_limit)
            expanded_scores = (
                active_scores[:, None] + candidate_scores[depth - 1][None, :]
            ).reshape(-1)
            expanded_parent_indices = (
                active_global_indices[:, None].expand(active_count, top_k).reshape(-1)
            )
            expanded_positional = (
                candidate_scores[depth - 1][None, :]
                .expand(active_count, top_k)
                .reshape(-1)
            )
            order = _stable_tensor_priority_order(
                torch_module,
                expanded_paths[:, :depth],
                expanded_scores,
            )
            selected = order[: min(beam_width, int(order.shape[0]))]
            active_paths = expanded_paths[selected]
            active_scores = expanded_scores[selected]
            active_parents = expanded_parent_indices[selected]
            active_positional = expanded_positional[selected]
            remaining = node_budget - total_nodes
            if remaining <= 0:
                break
            take = min(remaining, int(active_paths.shape[0]))
            paths.append(active_paths[:take])
            token_ids.append(active_paths[:take, depth - 1])
            parent_indices.append(active_parents[:take])
            depths.append(
                torch_module.full(
                    (take,), depth, dtype=torch_module.long, device=device
                )
            )
            positional_scores.append(active_positional[:take])
            cumulative_scores.append(active_scores[:take])
            per_depth_counts.append(take)
            first_global_index = total_nodes
            total_nodes += take
            if take < int(active_paths.shape[0]):
                break
            active_global_indices = torch_module.arange(
                first_global_index,
                first_global_index + take,
                dtype=torch_module.long,
                device=device,
            )
    else:
        raise ValueError(f"unsupported online policy: {policy}")

    flat_paths = torch_module.cat(paths, dim=0)
    flat_tokens = torch_module.cat(token_ids, dim=0)
    flat_parents = torch_module.cat(parent_indices, dim=0)
    flat_depths = torch_module.cat(depths, dim=0)
    flat_positional = torch_module.cat(positional_scores, dim=0)
    flat_cumulative = torch_module.cat(cumulative_scores, dim=0)
    if int(flat_tokens.shape[0]) > node_budget:
        raise RuntimeError("online tree exceeded its requested node budget")
    normalized = torch_module.where(
        flat_depths > 0,
        flat_cumulative / flat_depths.clamp_min(1),
        torch_module.zeros_like(flat_cumulative),
    )
    return {
        "policy": policy,
        "requested_node_budget": int(node_budget),
        "actual_total_node_count": int(flat_tokens.shape[0]),
        "beam_width": int(beam_width),
        "maximum_depth": int(len(per_depth_counts) - 1),
        "per_depth_node_counts": per_depth_counts,
        "token_ids": flat_tokens,
        "parent_indices": flat_parents,
        "depths": flat_depths,
        "path_tokens": flat_paths,
        "positional_log_probabilities": flat_positional,
        "cumulative_scores": flat_cumulative,
        "normalized_scores": normalized,
        "score_rule": SCORE_RULE,
        "tie_breaking": TIE_BREAKING_RULE,
    }


def build_online_tree_tensors_from_nodes(
    torch_module: Any,
    *,
    positions: list[dict[str, Any]],
    nodes: set[Node],
    root_token_id: Any,
    policy: str,
    node_budget: int,
    beam_width: int,
    device: Any,
) -> dict[str, Any]:
    """Pack nodes emitted by a canonical pure-Python policy onto the device.

    Phase 2D.1B uses this adapter for frozen policies that are already defined
    by Phase 2B/2D.  It does not reconstruct or approximate policy decisions.
    """

    root_id = int(root_token_id.item()) if hasattr(root_token_id, "item") else int(root_token_id)
    flat = flatten_tree(
        nodes,
        positions,
        root_token_id=root_id,
        policy=policy,
        requested_node_budget=node_budget,
        beam_width=beam_width,
    )
    records = flat["nodes"]
    path_rows = [
        [*record["draft_token_path"], *([-1] * (MAXIMUM_DRAFT_DEPTH - record["depth"]))]
        for record in records
    ]
    token_ids = torch_module.tensor(
        [record["token_id"] for record in records], dtype=torch_module.long, device=device
    )
    parents = torch_module.tensor(
        [-1 if record["parent_index"] is None else record["parent_index"] for record in records],
        dtype=torch_module.long,
        device=device,
    )
    depths = torch_module.tensor(
        [record["depth"] for record in records], dtype=torch_module.long, device=device
    )
    positional = torch_module.tensor(
        [0.0 if record["positional_draft_log_probability"] is None else record["positional_draft_log_probability"] for record in records],
        dtype=torch_module.float64,
        device=device,
    )
    cumulative = torch_module.tensor(
        [record["cumulative_score"] for record in records],
        dtype=torch_module.float64,
        device=device,
    )
    normalized = torch_module.tensor(
        [0.0 if record["normalized_score"] is None else record["normalized_score"] for record in records],
        dtype=torch_module.float64,
        device=device,
    )
    return {
        "policy": policy,
        "requested_node_budget": int(node_budget),
        "actual_total_node_count": len(records),
        "beam_width": int(beam_width),
        "maximum_depth": int(flat["maximum_depth"]),
        "per_depth_node_counts": [
            int(flat["per_depth_node_counts"].get(str(depth), 0))
            for depth in range(int(flat["maximum_depth"]) + 1)
        ],
        "token_ids": token_ids,
        "parent_indices": parents,
        "depths": depths,
        "path_tokens": torch_module.tensor(
            path_rows, dtype=torch_module.long, device=device
        ),
        "positional_log_probabilities": positional,
        "cumulative_scores": cumulative,
        "normalized_scores": normalized,
        "score_rule": SCORE_RULE,
        "tie_breaking": TIE_BREAKING_RULE,
        "construction_provenance": "canonical_pure_python_policy_then_tensor_pack",
    }


def build_tree_attention_mask(
    torch_module: Any,
    *,
    tree: dict[str, Any],
    prefix_length: int,
    dtype: Any,
) -> Any:
    """Return additive 4D prefix-plus-ancestor mask for packed verification."""

    paths = tree["path_tokens"]
    depths = tree["depths"]
    node_count, maximum_depth = map(int, paths.shape)
    positions = torch_module.arange(maximum_depth, device=paths.device)
    prefix_equal = (paths[:, None, :] == paths[None, :, :]) | (
        positions[None, None, :] >= depths[None, :, None]
    )
    ancestry = prefix_equal.all(dim=-1) & (depths[None, :] <= depths[:, None])
    visible = torch_module.cat(
        [
            torch_module.ones(
                (node_count, prefix_length),
                dtype=torch_module.bool,
                device=paths.device,
            ),
            ancestry,
        ],
        dim=1,
    )
    mask = torch_module.full(
        (1, 1, node_count, prefix_length + node_count),
        torch_module.finfo(dtype).min,
        dtype=dtype,
        device=paths.device,
    )
    return mask.masked_fill(visible[None, None, :, :], 0.0)


def tree_position_tensors(
    torch_module: Any, *, tree: dict[str, Any], context_length_before: int
) -> tuple[Any, Any]:
    """Return distinct semantic RoPE positions and packed cache storage indices."""

    semantic = (tree["depths"] + int(context_length_before)).unsqueeze(0)
    storage = torch_module.arange(
        context_length_before,
        context_length_before + tree["actual_total_node_count"],
        dtype=torch_module.long,
        device=tree["depths"].device,
    )
    return semantic, storage


def greedy_tree_traversal_tensors(
    torch_module: Any, *, tree: dict[str, Any], target_logits: Any
) -> dict[str, Any]:
    """Traverse a packed tree with greedy target argmax and one scalar sync."""

    node_count = tree["actual_total_node_count"]
    if list(target_logits.shape[:2]) != [1, node_count]:
        raise ValueError("packed target logits do not align with flattened tree")
    target_tokens = torch_module.argmax(target_logits[0], dim=-1)
    parents = tree["parent_indices"]
    node_tokens = tree["token_ids"]
    parent_grid = torch_module.arange(node_count, device=parents.device)[:, None]
    matches = (parents[None, :] == parent_grid) & (
        node_tokens[None, :] == target_tokens[:, None]
    )
    child_counts = (parents[None, :] == parent_grid).sum(dim=1)
    has_match = matches.any(dim=1)
    next_child = matches.to(dtype=torch_module.long).argmax(dim=1)
    current = torch_module.zeros((), dtype=torch_module.long, device=parents.device)
    active = torch_module.ones((), dtype=torch_module.bool, device=parents.device)
    accepted = []
    for _ in range(MAXIMUM_DRAFT_DEPTH):
        valid = active & has_match[current]
        child = next_child[current]
        accepted.append(
            torch_module.where(valid, child, -torch_module.ones_like(child))
        )
        current = torch_module.where(valid, child, current)
        active = valid
    accepted_tensor = torch_module.stack(accepted)
    accepted_count = int((accepted_tensor >= 0).sum().item())
    accepted_indices = accepted_tensor[:accepted_count]
    next_root_token = target_tokens[current]
    is_leaf = child_counts[current] == 0
    return {
        "accepted_node_indices": accepted_indices,
        "accepted_draft_token_count": accepted_count,
        "next_root_token_id": next_root_token,
        "next_root_is_bonus": is_leaf,
        "terminal_parent_index": current,
        "target_greedy_token_ids": target_tokens,
    }


def dynamic_cache_lengths(cache: Any) -> list[int]:
    """Return every initialized DynamicCache layer length without tensor transfer."""

    if not hasattr(cache, "layers"):
        raise TypeError("Phase 2C requires Transformers 4.57 DynamicCache.layers")
    return [int(layer.get_seq_length()) for layer in cache.layers]


def compact_dynamic_cache(
    torch_module: Any,
    *,
    cache: Any,
    prefix_length: int,
    actual_tree_node_count: int,
    accepted_node_indices: Any,
) -> dict[str, Any]:
    """Gather prefix + root + accepted path from temporary packed-tree KV."""

    if not hasattr(cache, "layers"):
        raise TypeError("Phase 2C requires Transformers 4.57 DynamicCache.layers")
    root = torch_module.zeros(
        (1,), dtype=torch_module.long, device=accepted_node_indices.device
    )
    committed_nodes = torch_module.cat(
        [root, accepted_node_indices.to(torch_module.long)]
    )
    prefix = torch_module.arange(
        prefix_length, dtype=torch_module.long, device=committed_nodes.device
    )
    packed_storage = prefix_length + committed_nodes
    gather_indices = torch_module.cat([prefix, packed_storage])
    before_lengths = []
    after_lengths = []
    expected_temporary = prefix_length + actual_tree_node_count
    for layer in cache.layers:
        length = int(layer.get_seq_length())
        before_lengths.append(length)
        if length != expected_temporary:
            raise RuntimeError(
                "temporary packed cache length does not equal prefix + tree nodes"
            )
        layer.keys = layer.keys.index_select(-2, gather_indices)
        layer.values = layer.values.index_select(-2, gather_indices)
        after_lengths.append(int(layer.get_seq_length()))
    expected_after = prefix_length + int(committed_nodes.shape[0])
    if not after_lengths or any(length != expected_after for length in after_lengths):
        raise RuntimeError("compacted cache length invariant failed")
    return {
        "temporary_layer_lengths": before_lengths,
        "compacted_layer_lengths": after_lengths,
        "expected_compacted_length": expected_after,
        "committed_node_indices": committed_nodes,
        "packed_storage_indices": packed_storage,
        "gather_indices": gather_indices,
        "discarded_tree_node_count": actual_tree_node_count
        - int(committed_nodes.shape[0]),
        "correction_or_bonus_written": False,
    }


def clone_dynamic_cache(cache: Any) -> Any:
    """Clone a DynamicCache for debug-only dense/path gold checks."""

    return copy.deepcopy(cache)


def tensor_tree_metadata(tree: dict[str, Any]) -> dict[str, Any]:
    """Resolve compact tree tensors to JSON-compatible metadata after timing."""

    def values(name: str) -> list[Any]:
        return tree[name].detach().cpu().tolist()

    tokens = values("token_ids")
    parents = values("parent_indices")
    depths = values("depths")
    paths = values("path_tokens")
    positional = values("positional_log_probabilities")
    cumulative = values("cumulative_scores")
    normalized = values("normalized_scores")
    root = int(tokens[0])
    records = []
    for index, depth in enumerate(depths):
        draft_path = [int(token) for token in paths[index][:depth]]
        records.append(
            {
                "node_index": index,
                "parent_index": int(parents[index]) if index else None,
                "depth": int(depth),
                "token_id": int(tokens[index]),
                "positional_draft_log_probability": (
                    float(positional[index]) if index else None
                ),
                "cumulative_score": float(cumulative[index]),
                "normalized_score": float(normalized[index]) if index else None,
                "draft_token_path": draft_path,
                "root_to_node_token_path": [root, *draft_path],
            }
        )
    return {
        "policy": tree["policy"],
        "requested_node_budget": tree["requested_node_budget"],
        "actual_total_node_count": tree["actual_total_node_count"],
        "beam_width": tree["beam_width"],
        "maximum_depth": tree["maximum_depth"],
        "per_depth_node_counts": {
            str(depth): count
            for depth, count in enumerate(tree["per_depth_node_counts"])
        },
        "score_rule": tree["score_rule"],
        "tie_breaking": tree["tie_breaking"],
        "nodes": records,
    }
