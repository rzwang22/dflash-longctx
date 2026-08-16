"""Framework-lazy Phase 2C.2 absolute-token divergence diagnostics."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def token_ids_sha256(values: list[int]) -> str:
    payload = json.dumps([int(value) for value in values], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def first_divergent_offset(left: list[int], right: list[int]) -> int | None:
    """Return the first unequal offset, including a length-only divergence."""

    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if int(left_token) != int(right_token):
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def output_pair_comparison(
    *,
    left_name: str,
    left_ids: list[int],
    right_name: str,
    right_ids: list[int],
    prompt_length: int,
) -> dict[str, Any]:
    offset = first_divergent_offset(left_ids, right_ids)
    lcp = min(len(left_ids), len(right_ids)) if offset is None else offset
    return {
        "left_name": left_name,
        "right_name": right_name,
        "output_ids_equal": offset is None,
        "left_returned_token_count": len(left_ids),
        "right_returned_token_count": len(right_ids),
        "longest_common_prefix_length": lcp,
        "first_divergent_generated_token_offset": offset,
        "first_divergent_absolute_sequence_position": (
            prompt_length + offset if offset is not None else None
        ),
        "left_token_id": left_ids[offset]
        if offset is not None and offset < len(left_ids)
        else None,
        "right_token_id": right_ids[offset]
        if offset is not None and offset < len(right_ids)
        else None,
        "left_output_ids": [int(value) for value in left_ids],
        "right_output_ids": [int(value) for value in right_ids],
        "left_output_sha256": token_ids_sha256(left_ids),
        "right_output_sha256": token_ids_sha256(right_ids),
    }


def _root_offset(row: dict[str, Any], *, prompt_length: int) -> int:
    if "generated_root_offset" in row:
        return int(row["generated_root_offset"])
    return int(row["context_length_before"]) - int(prompt_length)


def locate_token_producer_round(
    rounds: list[dict[str, Any]], *, generated_offset: int, prompt_length: int
) -> dict[str, Any] | None:
    """Locate a token by absolute generated offset, independent of round index."""

    if generated_offset < 0:
        raise ValueError("generated offset cannot be negative")
    for index, row in enumerate(rounds):
        root_offset = _root_offset(row, prompt_length=prompt_length)
        accepted = int(row["accepted_draft_token_count"])
        next_root_offset = root_offset + 1 + accepted
        if generated_offset == 0 and root_offset == 0:
            kind = "root"
        elif root_offset < generated_offset < next_root_offset:
            kind = "accepted_draft"
        elif generated_offset == next_root_offset:
            kind = str(row.get("target_token_kind") or "correction_or_bonus")
        else:
            continue
        return {
            "round_index": int(row.get("round_index", index)),
            "root_generated_offset": root_offset,
            "token_kind": kind,
            "token_offset_within_round": generated_offset - root_offset,
            "round": row,
        }
    return None


def build_round_differential(
    *,
    comparison: dict[str, Any],
    prompt_length: int,
    reference_rounds: list[dict[str, Any]],
    tree_rounds: list[dict[str, Any]],
) -> dict[str, Any] | None:
    offset = comparison["first_divergent_generated_token_offset"]
    if offset is None:
        return None
    reference = locate_token_producer_round(
        reference_rounds, generated_offset=offset, prompt_length=prompt_length
    )
    tree = locate_token_producer_round(
        tree_rounds, generated_offset=offset, prompt_length=prompt_length
    )
    lcp = comparison["longest_common_prefix_length"]
    left_prefix = comparison["left_output_ids"][:lcp]
    right_prefix = comparison["right_output_ids"][:lcp]
    reference_entry_offset = reference["root_generated_offset"] if reference else None
    tree_entry_offset = tree["root_generated_offset"] if tree else None
    round_entry_prefix_equal = bool(
        reference_entry_offset is not None
        and tree_entry_offset is not None
        and reference_entry_offset == tree_entry_offset
        and comparison["left_output_ids"][:reference_entry_offset]
        == comparison["right_output_ids"][:tree_entry_offset]
    )
    return {
        "generated_token_offset": offset,
        "absolute_sequence_position": prompt_length + offset,
        "predivergence_committed_prefix_token_exact": left_prefix == right_prefix,
        "predivergence_prefix_sha256": token_ids_sha256(left_prefix),
        "reference_round_entry_persistent_kv_length": (
            prompt_length + reference_entry_offset
            if reference_entry_offset is not None
            else None
        ),
        "tree_round_entry_persistent_kv_length": (
            prompt_length + tree_entry_offset if tree_entry_offset is not None else None
        ),
        "round_entry_persistent_kv_same_semantic_prefix": round_entry_prefix_equal,
        "absolute_position_semantic_prefix_token_exact": left_prefix == right_prefix,
        "reference": reference,
        "tree": tree,
        "round_indices_equal": bool(
            reference and tree and reference["round_index"] == tree["round_index"]
        ),
        "absolute_token_lockstep_used": True,
    }


def classify_layerwise_divergence(
    *,
    output_ids_equal: bool,
    traversal_checks_passed: bool | None,
    structural_checks_passed: bool | None,
    native_verifier_mismatch_count: int | None,
    math_verifier_mismatch_count: int | None,
    kv_root_transition_checks_passed: bool | None,
    kv_next_token_argmax_equal: bool | None,
    divergence_begins_at_round_root: bool | None,
) -> dict[str, Any]:
    """Conservative evidence-based Phase 2C.2 divergence classification."""

    if output_ids_equal:
        return {"status": "pass", "failure_class": None, "evidence_complete": True}
    if structural_checks_passed is False:
        failure = "branch_mask_or_position_mismatch"
    elif traversal_checks_passed is False:
        failure = "tree_traversal_mismatch"
    elif (
        native_verifier_mismatch_count is not None
        and native_verifier_mismatch_count > 0
        and math_verifier_mismatch_count == 0
    ):
        failure = "numerical_backend_decision_divergence"
    elif math_verifier_mismatch_count is not None and math_verifier_mismatch_count > 0:
        failure = "branch_mask_or_position_mismatch"
    elif (
        kv_root_transition_checks_passed is False
        or kv_next_token_argmax_equal is False
        or divergence_begins_at_round_root is True
    ):
        failure = "kv_compaction_or_root_transition_mismatch"
    else:
        return {
            "status": "inconclusive",
            "failure_class": "unresolved_layerwise_output_divergence",
            "evidence_complete": False,
        }
    return {"status": "failed", "failure_class": failure, "evidence_complete": True}
