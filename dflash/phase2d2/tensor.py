"""Torch-parameterized tensor adapters for materialized Phase 2D.2A code.

The module imports no model framework at import time.  The active custom model
passes its already imported ``torch`` module explicitly.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _refinement_module() -> Any:
    """Resolve the local-package or materialized remote-code module name.

    Transformers statically scans every ``from .name import ...`` occurrence,
    including imports inside ``try`` blocks.  Using ``import_module`` here avoids
    advertising the repository-only ``refinement.py`` name to the Hugging Face
    dynamic-module copier while retaining normal ``dflash.phase2d2`` imports.
    """

    materialized_name = f"{__package__}.phase2d2_refinement"
    try:
        return import_module(materialized_name)
    except ModuleNotFoundError as exc:
        if exc.name != materialized_name:
            raise
        return import_module(f"{__package__}.refinement")


MAXIMUM_DRAFT_DEPTH = _refinement_module().MAXIMUM_DRAFT_DEPTH


def repeat_dynamic_cache_batch(cache: Any, repeats: int) -> Any:
    """Expand a cloned Transformers 4.57 DynamicCache along batch dimension."""

    if int(repeats) < 1:
        raise ValueError("cache batch repeats must be positive")
    if not hasattr(cache, "layers"):
        raise TypeError(
            "conditional-remask requires Transformers 4.57 DynamicCache.layers"
        )
    for layer in cache.layers:
        if layer.keys is None or layer.values is None:
            continue
        if int(layer.keys.shape[0]) != 1 or int(layer.values.shape[0]) != 1:
            raise ValueError(
                "round-entry draft cache must have external batch size one"
            )
        layer.keys = layer.keys.repeat_interleave(int(repeats), dim=0)
        layer.values = layer.values.repeat_interleave(int(repeats), dim=0)
    return cache


def conditional_candidate_records(
    torch_module: Any,
    *,
    base_logits: Any,
    conditional_logits: Any,
    prefixes: list[tuple[int, ...]],
    base_top_m: int,
    conditional_top_r: int,
    merge_mode: str,
) -> dict[tuple[int, ...], list[list[dict[str, Any]]]]:
    """Gather bounded union IDs plus exact full-vocabulary log probabilities."""

    if list(base_logits.shape[:2]) != [1, MAXIMUM_DRAFT_DEPTH]:
        raise ValueError("base logits must have shape [1, 15, vocabulary]")
    if (
        int(conditional_logits.shape[0]) != len(prefixes)
        or int(conditional_logits.shape[1]) != MAXIMUM_DRAFT_DEPTH
    ):
        raise ValueError(
            "conditional logits do not align with the internal branch batch"
        )
    if int(base_logits.shape[-1]) != int(conditional_logits.shape[-1]):
        raise ValueError("base and conditional vocabularies differ")
    base_logp = torch_module.log_softmax(base_logits[0].float(), dim=-1)
    conditional_logp = torch_module.log_softmax(conditional_logits.float(), dim=-1)
    base_ids = torch_module.argsort(
        base_logits[0], dim=-1, descending=True, stable=True
    )[:, : int(base_top_m)]
    conditional_ids = torch_module.argsort(
        conditional_logits, dim=-1, descending=True, stable=True
    )[:, :, : int(conditional_top_r)]
    result: dict[tuple[int, ...], list[list[dict[str, Any]]]] = {}
    for branch_index, prefix in enumerate(prefixes):
        positions = []
        for position in range(MAXIMUM_DRAFT_DEPTH):
            base_row = [
                int(value) for value in base_ids[position].detach().cpu().tolist()
            ]
            conditional_row = [
                int(value)
                for value in conditional_ids[branch_index, position]
                .detach()
                .cpu()
                .tolist()
            ]
            candidate_ids = (
                base_row
                if merge_mode == "conditional_rerank"
                else sorted(set(base_row) | set(conditional_row))
            )
            ids = torch_module.tensor(
                candidate_ids,
                dtype=torch_module.long,
                device=base_logits.device,
            )
            base_values = (
                base_logp[position].index_select(0, ids).detach().cpu().tolist()
            )
            conditional_values = (
                conditional_logp[branch_index, position]
                .index_select(0, ids)
                .detach()
                .cpu()
                .tolist()
            )
            base_rank = {token: rank for rank, token in enumerate(base_row, 1)}
            conditional_rank = {
                token: rank for rank, token in enumerate(conditional_row, 1)
            }
            positions.append(
                [
                    {
                        "token_id": token,
                        "base_rank": base_rank.get(token),
                        "conditional_rank": conditional_rank.get(token),
                        "base_log_probability": float(base_value),
                        "conditional_log_probability": float(conditional_value),
                    }
                    for token, base_value, conditional_value in zip(
                        candidate_ids, base_values, conditional_values
                    )
                ]
            )
        result[prefix] = positions
    return result


def pack_refined_tree_tensors(
    torch_module: Any,
    *,
    tree_spec: dict[str, Any],
    root_token_id: Any,
    device: Any,
) -> dict[str, Any]:
    """Pack a parent-conditioned pure-Python tree for the Phase 2C verifier."""

    records = tree_spec["nodes"]
    if not records or records[0]["depth"] != 0:
        raise ValueError("refined tree record zero must be the root")
    root = (
        int(root_token_id.item())
        if hasattr(root_token_id, "item")
        else int(root_token_id)
    )
    path_rows = [
        [*row["path"], *([-1] * (MAXIMUM_DRAFT_DEPTH - int(row["depth"])))]
        for row in records
    ]
    token_ids = [root if row["depth"] == 0 else int(row["token_id"]) for row in records]
    positional = [float(row["positional_merged_score"]) for row in records]
    cumulative = [float(row["cumulative_merged_score"]) for row in records]
    normalized = [
        0.0
        if row["normalized_merged_score"] is None
        else float(row["normalized_merged_score"])
        for row in records
    ]
    result = {
        "policy": f"phase2d2_{tree_spec['merge_mode']}",
        "requested_node_budget": int(tree_spec["requested_node_budget"]),
        "actual_total_node_count": int(tree_spec["actual_total_node_count"]),
        "beam_width": 0,
        "maximum_depth": int(tree_spec["maximum_depth"]),
        "per_depth_node_counts": list(tree_spec["per_depth_node_counts"]),
        "token_ids": torch_module.tensor(
            token_ids, dtype=torch_module.long, device=device
        ),
        "parent_indices": torch_module.tensor(
            [
                -1 if row["parent_index"] is None else int(row["parent_index"])
                for row in records
            ],
            dtype=torch_module.long,
            device=device,
        ),
        "depths": torch_module.tensor(
            [int(row["depth"]) for row in records],
            dtype=torch_module.long,
            device=device,
        ),
        "path_tokens": torch_module.tensor(
            path_rows, dtype=torch_module.long, device=device
        ),
        "positional_log_probabilities": torch_module.tensor(
            positional, dtype=torch_module.float64, device=device
        ),
        "cumulative_scores": torch_module.tensor(
            cumulative, dtype=torch_module.float64, device=device
        ),
        "normalized_scores": torch_module.tensor(
            normalized, dtype=torch_module.float64, device=device
        ),
        "score_rule": tree_spec["score_rule"],
        "tie_breaking": tree_spec["tie_breaking"],
        "construction_provenance": "phase2d2_parent_conditioned_pure_python_then_tensor_pack",
        "phase2d2_tree_spec": tree_spec,
    }
    if result["actual_total_node_count"] > result["requested_node_budget"]:
        raise RuntimeError("packed refined tree exceeded the fixed target budget")
    return result
