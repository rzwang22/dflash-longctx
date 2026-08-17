"""Framework-lazy Phase 2C.4 reference/tree state alignment and classification."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from typing import Any

REFERENCE_STATE_FAILURE_CLASSES = (
    "local_packed_verifier_numerical_divergence",
    "target_kv_state_drift",
    "target_hidden_state_drift",
    "draft_state_or_candidate_drift",
    "commit_or_root_transition_mismatch",
    "unresolved_reference_state_divergence",
)


def _sha256(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _p99(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.99 * len(ordered)) - 1)]


def compare_tensor_samples(
    reference: dict[str, Any] | None,
    tree: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare bounded stable samples without claiming a full-tensor reduction."""

    if reference is None or tree is None:
        return {
            "status": "not_directly_comparable",
            "structural_equal": None,
            "numerically_equal": None,
            "max_abs_error": None,
            "mean_abs_error": None,
            "p99_abs_error": None,
            "sample_count": 0,
            "reduction_scope": "bounded_stable_samples",
            "reference_sample_sha256": None,
            "tree_sample_sha256": None,
        }
    structural_equal = all(
        reference.get(field) == tree.get(field)
        for field in ("shape", "numel", "sample_indices")
    )
    reference_values = [float(value) for value in reference.get("sample_values", [])]
    tree_values = [float(value) for value in tree.get("sample_values", [])]
    comparable = structural_equal and len(reference_values) == len(tree_values)
    errors = (
        [abs(left - right) for left, right in zip(reference_values, tree_values)]
        if comparable
        else []
    )
    maximum = max(errors, default=0.0) if comparable else None
    return {
        "status": "available" if comparable else "structural_mismatch",
        "structural_equal": structural_equal,
        "numerically_equal": maximum == 0.0 if comparable else None,
        "max_abs_error": maximum,
        "mean_abs_error": statistics.fmean(errors)
        if errors
        else 0.0
        if comparable
        else None,
        "p99_abs_error": _p99(errors) if comparable else None,
        "sample_count": len(errors),
        "reduction_scope": "bounded_stable_samples",
        "reference_sample_sha256": _sha256(
            {
                "indices": reference.get("sample_indices", []),
                "values": reference_values,
            }
        ),
        "tree_sample_sha256": _sha256(
            {
                "indices": tree.get("sample_indices", []),
                "values": tree_values,
            }
        ),
    }


def compare_cache_samples(
    reference: list[dict[str, Any]] | None,
    tree: list[dict[str, Any]] | None,
    *,
    component: str,
    generated_prefix_length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if reference is None or tree is None:
        return (
            {
                "status": "not_directly_comparable",
                "structural_equal": None,
                "numerically_equal": None,
                "maximum_sample_abs_error": None,
                "mean_layer_sample_abs_error": None,
                "layer_count": 0,
                "reduction_scope": "bounded_stable_samples",
            },
            [],
        )
    reference_by_layer = {int(row["layer_index"]): row for row in reference}
    tree_by_layer = {int(row["layer_index"]): row for row in tree}
    layer_indices = sorted(set(reference_by_layer) | set(tree_by_layer))
    layers = []
    for layer_index in layer_indices:
        left = reference_by_layer.get(layer_index)
        right = tree_by_layer.get(layer_index)
        k = compare_tensor_samples(
            left.get("key_comparable_sample") if left else None,
            right.get("key_comparable_sample") if right else None,
        )
        v = compare_tensor_samples(
            left.get("value_comparable_sample") if left else None,
            right.get("value_comparable_sample") if right else None,
        )
        shape_equal = bool(
            left
            and right
            and left.get("sequence_length") == right.get("sequence_length")
            and left.get("key_shape") == right.get("key_shape")
            and left.get("value_shape") == right.get("value_shape")
        )
        layers.append(
            {
                "generated_prefix_length": int(generated_prefix_length),
                "component": component,
                "layer_index": layer_index,
                "status": (
                    "available"
                    if left is not None and right is not None
                    else "not_directly_comparable"
                ),
                "shape_equal": shape_equal,
                "reference_sequence_length": left.get("sequence_length")
                if left
                else None,
                "tree_sequence_length": right.get("sequence_length") if right else None,
                "k": k,
                "v": v,
            }
        )
    available = [row for row in layers if row["status"] == "available"]
    structural_equal = (
        bool(available)
        and len(available) == len(layers)
        and all(
            row["shape_equal"]
            and row["k"]["structural_equal"]
            and row["v"]["structural_equal"]
            for row in available
        )
    )
    maxima = [
        float(row[kind]["max_abs_error"])
        for row in available
        for kind in ("k", "v")
        if row[kind]["max_abs_error"] is not None
    ]
    return (
        {
            "status": "available" if available else "not_directly_comparable",
            "structural_equal": structural_equal if available else None,
            "numerically_equal": all(value == 0.0 for value in maxima)
            if maxima
            else None,
            "maximum_sample_abs_error": max(maxima, default=0.0) if available else None,
            "mean_layer_sample_abs_error": statistics.fmean(maxima)
            if maxima
            else 0.0
            if available
            else None,
            "layer_count": len(available),
            "reduction_scope": "bounded_stable_samples",
        },
        layers,
    )


def compare_logits_summary(
    reference: dict[str, Any] | None, tree: dict[str, Any] | None
) -> dict[str, Any]:
    if reference is None or tree is None:
        return {
            "status": "not_directly_comparable",
            "argmax_equal": None,
            "top8_ordered_equal": None,
            "top8_set_overlap_count": None,
            "margin_abs_error": None,
        }
    reference_top = [int(value) for value in reference.get("top_token_ids", [])]
    tree_top = [int(value) for value in tree.get("top_token_ids", [])]
    reference_margin = reference.get("top1_top2_margin")
    tree_margin = tree.get("top1_top2_margin")
    return {
        "status": "available",
        "argmax_equal": reference.get("top1_token_id") == tree.get("top1_token_id"),
        "reference_argmax_token_id": reference.get("top1_token_id"),
        "tree_argmax_token_id": tree.get("top1_token_id"),
        "top8_ordered_equal": reference_top == tree_top,
        "top8_set_overlap_count": len(set(reference_top) & set(tree_top)),
        "reference_top8_token_ids": reference_top,
        "tree_top8_token_ids": tree_top,
        "reference_margin": reference_margin,
        "tree_margin": tree_margin,
        "margin_abs_error": abs(float(reference_margin) - float(tree_margin))
        if reference_margin is not None and tree_margin is not None
        else None,
    }


def _round_index(checkpoint: dict[str, Any] | None, side: str) -> int | None:
    if checkpoint is None:
        return None
    direct = checkpoint.get(f"{side}_round_index")
    if direct is not None:
        return int(direct)
    entry = checkpoint.get("round_entry")
    return int(entry["round_index"]) if entry else None


def _draft_comparison(
    reference: dict[str, Any] | None,
    tree: dict[str, Any] | None,
    *,
    generated_prefix_length: int,
    reference_checkpoint: dict[str, Any] | None,
    tree_checkpoint: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if reference is None or tree is None:
        return (
            {
                "status": "not_directly_comparable",
                "top1_candidates_equal": None,
                "topk_candidates_equal": None,
                "cache": None,
                "candidate_alignment_status": "not_directly_comparable",
                "candidate_alignment_reason": "missing_round_entry_draft_state",
                "reference_actual_proposal_token_ids": None,
                "reference_candidate_top1_token_ids": None,
                "tree_candidate_top1_token_ids": None,
                "proposal_top1_semantically_aligned": None,
                "actual_proposal_top1_drift": False,
            },
            [],
        )
    cache, layers = compare_cache_samples(
        reference.get("cache"),
        tree.get("cache"),
        component="draft_kv",
        generated_prefix_length=generated_prefix_length,
    )
    reference_logits = (reference.get("logits") or {}).get("positions", [])
    tree_logits = (tree.get("logits") or {}).get("positions", [])
    logit_positions = [
        {
            "draft_position": index + 1,
            **compare_logits_summary(left, right),
        }
        for index, (left, right) in enumerate(zip(reference_logits, tree_logits))
    ]
    reference_tree_input = reference.get("candidate_tree_input") or {}
    tree_tree_input = tree.get("candidate_tree_input") or {}
    reference_proposal = _optional_int_list(
        reference_tree_input.get("actual_proposed_token_ids")
    )
    reference_top1 = _optional_int_list(reference.get("candidate_top1_token_ids"))
    tree_top1 = _optional_int_list(tree.get("candidate_top1_token_ids"))
    reference_root = _round_root(reference_checkpoint)
    tree_root = _round_root(tree_checkpoint)
    expected_count = 15
    counts = {
        "reference_actual_proposal": len(reference_proposal)
        if reference_proposal is not None
        else None,
        "reference_candidate_top1": len(reference_top1)
        if reference_top1 is not None
        else None,
        "tree_candidate_top1": len(tree_top1) if tree_top1 is not None else None,
    }
    if reference_proposal is None:
        alignment_status = "missing_actual_proposal"
        alignment_reason = "Original DFlash actual proposed chain was not captured"
    elif reference_root is None or tree_root is None or reference_root != tree_root:
        alignment_status = "semantic_context_mismatch"
        alignment_reason = "round-entry roots are missing or unequal"
    elif any(count != expected_count for count in counts.values()):
        alignment_status = "invalid_draft_token_count"
        alignment_reason = "candidate chains must each contain 15 root-excluding tokens"
    elif reference_proposal != reference_top1:
        alignment_status = "candidate_capture_alignment_inconsistent"
        alignment_reason = (
            "captured reference Top-1 chain disagrees with the actual Original "
            "DFlash proposed chain"
        )
    else:
        alignment_status = "aligned"
        alignment_reason = None
    semantically_aligned = alignment_status == "aligned"
    actual_proposal_drift = bool(
        semantically_aligned
        and reference_proposal is not None
        and tree_top1 is not None
        and reference_proposal != tree_top1
    )
    return (
        {
            "status": "available",
            "top1_candidates_equal": reference_top1 == tree_top1,
            "topk_candidates_equal": reference.get("candidate_topk_token_ids")
            == tree.get("candidate_topk_token_ids"),
            "reference_top1_token_ids": reference_top1,
            "tree_top1_token_ids": tree_top1,
            "logit_positions": logit_positions,
            "any_logit_argmax_difference": any(
                row.get("argmax_equal") is False for row in logit_positions
            ),
            "any_logit_margin_difference": any(
                row.get("margin_abs_error") not in {None, 0.0}
                for row in logit_positions
            ),
            "original_proposal_equals_tree_top1": (
                reference_proposal == tree_top1
                if reference_proposal is not None and tree_top1 is not None
                else None
            ),
            "candidate_alignment_status": alignment_status,
            "candidate_alignment_reason": alignment_reason,
            "reference_actual_proposal_token_ids": reference_proposal,
            "reference_candidate_top1_token_ids": reference_top1,
            "tree_candidate_top1_token_ids": tree_top1,
            "candidate_root_included": False,
            "candidate_position_offset": 1,
            "candidate_expected_draft_token_count": expected_count,
            "candidate_chain_lengths": counts,
            "reference_round_entry_root_token_id": reference_root,
            "tree_round_entry_root_token_id": tree_root,
            "proposal_top1_semantically_aligned": semantically_aligned,
            "actual_proposal_top1_drift": actual_proposal_drift,
            "cache": cache,
            "reference_candidate_tree_input": reference_tree_input,
            "tree_candidate_tree_input": tree_tree_input,
        },
        layers,
    )


def _optional_int_list(values: Any) -> list[int] | None:
    if values is None:
        return None
    return [int(value) for value in values]


def _round_root(checkpoint: dict[str, Any] | None) -> int | None:
    entry = (checkpoint or {}).get("round_entry")
    if not entry or entry.get("root_token_id") is None:
        return None
    return int(entry["root_token_id"])


def _transitions_by_entry(
    checkpoints: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    result = {}
    for checkpoint in checkpoints:
        transition = checkpoint.get("commit_transition")
        if transition is None or transition.get("generated_prefix_before") is None:
            continue
        entry_prefix = int(transition["generated_prefix_before"])
        if entry_prefix in result:
            raise ValueError(f"duplicate commit transition at prefix {entry_prefix}")
        result[entry_prefix] = {
            "transition": transition,
            "checkpoint_provenance": checkpoint.get("checkpoint_provenance"),
            "execution_path": checkpoint.get("execution_path"),
        }
    return result


def _transition_comparison(
    reference: dict[str, Any] | None,
    tree: dict[str, Any] | None,
    *,
    entry_prefix: int,
) -> dict[str, Any]:
    if reference is None or tree is None:
        return {
            "comparable": False,
            "same_entry_prefix": False,
            "actual_execution_evidence": False,
            "alignment_status": "not_directly_comparable",
            "semantics_equal": None,
            "selected_path_equal": None,
            "next_root_equal": None,
            "actual_mismatch": False,
            "reference": reference.get("transition") if reference else None,
            "tree": tree.get("transition") if tree else None,
        }
    left = reference["transition"]
    right = tree["transition"]
    same_entry = (
        int(left["generated_prefix_before"])
        == int(right["generated_prefix_before"])
        == int(entry_prefix)
    )
    actual_execution = bool(
        reference.get("checkpoint_provenance") == "actual_reference_checkpoint"
        and tree.get("checkpoint_provenance")
        in {"actual_tree_checkpoint", "diagnostic_replay"}
    )
    selected_path_equal = bool(
        left.get("accepted_draft_token_count")
        == right.get("accepted_draft_token_count")
        and left.get("committed_token_ids") == right.get("committed_token_ids")
    )
    next_root_equal = bool(
        left.get("next_root_token_id") == right.get("next_root_token_id")
        and left.get("next_root_kind") == right.get("next_root_kind")
    )
    exit_equal = left.get("generated_prefix_after") == right.get(
        "generated_prefix_after"
    )
    semantics_equal = bool(selected_path_equal and next_root_equal and exit_equal)
    return {
        "comparable": True,
        "same_entry_prefix": same_entry,
        "actual_execution_evidence": actual_execution,
        "alignment_status": (
            "same_absolute_entry_prefix"
            if same_entry
            else "different_absolute_entry_prefix"
        ),
        "semantics_equal": semantics_equal if same_entry else None,
        "selected_path_equal": selected_path_equal if same_entry else None,
        "next_root_equal": next_root_equal if same_entry else None,
        "actual_mismatch": bool(
            same_entry and actual_execution and not semantics_equal
        ),
        "reference_execution_provenance": ("actual_original_dflash_commit_transition"),
        "tree_execution_provenance": "actual_tree_diagnostic_execution",
        "reference": left,
        "tree": right,
    }


def _probe_fidelity(
    checkpoint: dict[str, Any] | None,
    actual_generated_token_ids: list[int] | None,
    *,
    generated_prefix_length: int,
    side: str,
) -> dict[str, Any]:
    probe = (checkpoint or {}).get("semantic_probe")
    probe_argmax = probe.get("top1_token_id") if probe else None
    actual_token = (
        int(actual_generated_token_ids[generated_prefix_length])
        if actual_generated_token_ids is not None
        and generated_prefix_length < len(actual_generated_token_ids)
        else None
    )
    if probe_argmax is None:
        faithful = None
        status = "unavailable_no_probe"
    elif actual_token is None:
        faithful = None
        status = "unavailable_no_actual_token"
    else:
        faithful = int(probe_argmax) == actual_token
        status = "faithful" if faithful else "unfaithful"
    return {
        f"{side}_actual_token_id": actual_token,
        f"{side}_probe_argmax_token_id": int(probe_argmax)
        if probe_argmax is not None
        else None,
        f"{side}_probe_faithful": faithful,
        f"{side}_probe_fidelity_status": status,
        f"{side}_probe_execution_provenance": (checkpoint or {}).get(
            "semantic_probe_provenance", "not_directly_comparable"
        ),
        f"{side}_actual_token_provenance": (
            f"actual_{side}_diagnostic_output_token_ids"
            if actual_token is not None
            else "not_directly_comparable"
        ),
    }


def compare_reference_tree_checkpoints(
    *,
    reference_checkpoints: list[dict[str, Any]],
    tree_checkpoints: list[dict[str, Any]],
    first_output_divergent_offset: int | None,
    packed_verifier_evidence: dict[int, list[dict[str, Any]]] | None = None,
    reference_actual_generated_token_ids: list[int] | None = None,
    tree_actual_generated_token_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Align checkpoints solely by absolute generated-token prefix."""

    reference_by_prefix = {
        int(row["generated_prefix_length"]): row for row in reference_checkpoints
    }
    tree_by_prefix = {
        int(row["generated_prefix_length"]): row for row in tree_checkpoints
    }
    prefixes = sorted(set(reference_by_prefix) | set(tree_by_prefix))
    if first_output_divergent_offset is not None:
        prefixes = [
            prefix for prefix in prefixes if prefix <= first_output_divergent_offset
        ]
    packed_verifier_evidence = packed_verifier_evidence or {}
    reference_transitions = _transitions_by_entry(reference_checkpoints)
    tree_transitions = _transitions_by_entry(tree_checkpoints)
    rows = []
    layer_rows = []
    for prefix in prefixes:
        reference = reference_by_prefix.get(prefix)
        tree = tree_by_prefix.get(prefix)
        directly_comparable = reference is not None and tree is not None
        semantic_prefix_equal = bool(
            directly_comparable
            and reference["semantic_prefix_token_ids_sha256"]
            == tree["semantic_prefix_token_ids_sha256"]
        )
        target_kv, target_layers = compare_cache_samples(
            reference.get("persistent_target_kv") if reference else None,
            tree.get("persistent_target_kv") if tree else None,
            component="persistent_target_kv",
            generated_prefix_length=prefix,
        )
        target_hidden = compare_tensor_samples(
            reference.get("target_hidden") if reference else None,
            tree.get("target_hidden") if tree else None,
        )
        draft_input_target_hidden = compare_tensor_samples(
            reference.get("target_hidden_used_by_draft") if reference else None,
            tree.get("target_hidden_used_by_draft") if tree else None,
        )
        semantic_probe = compare_logits_summary(
            reference.get("semantic_probe") if reference else None,
            tree.get("semantic_probe") if tree else None,
        )
        draft, draft_layers = _draft_comparison(
            reference.get("draft_state") if reference else None,
            tree.get("draft_state") if tree else None,
            generated_prefix_length=prefix,
            reference_checkpoint=reference,
            tree_checkpoint=tree,
        )
        verifier = packed_verifier_evidence.get(prefix, [])
        native_online_verifier_flip = any(
            row.get("verifier_decision_relevant")
            and row.get("argmax_equal") is False
            and row.get("comparison_mode") == "native"
            for row in verifier
        )
        same_mask_control_flip = any(
            row.get("verifier_decision_relevant")
            and row.get("argmax_equal") is False
            and row.get("comparison_mode") == "same_mask"
            for row in verifier
        )
        math_control_flip = any(
            row.get("verifier_decision_relevant")
            and row.get("argmax_equal") is False
            and row.get("comparison_mode") == "math"
            for row in verifier
        )
        transition = _transition_comparison(
            reference_transitions.get(prefix),
            tree_transitions.get(prefix),
            entry_prefix=prefix,
        )
        probe_fidelity = {
            **_probe_fidelity(
                reference,
                reference_actual_generated_token_ids,
                generated_prefix_length=prefix,
                side="reference",
            ),
            **_probe_fidelity(
                tree,
                tree_actual_generated_token_ids,
                generated_prefix_length=prefix,
                side="tree",
            ),
        }
        faithful_probe_decision_divergence = bool(
            semantic_probe.get("argmax_equal") is False
            and probe_fidelity["reference_probe_faithful"] is True
            and probe_fidelity["tree_probe_faithful"] is True
        )
        numerical_state_difference = any(
            value is False
            for value in (
                target_kv.get("numerically_equal"),
                target_hidden.get("numerically_equal"),
                draft_input_target_hidden.get("numerically_equal"),
                (draft.get("cache") or {}).get("numerically_equal"),
            )
        )
        structural_state_difference = any(
            value is False
            for value in (
                target_kv.get("structural_equal"),
                target_hidden.get("structural_equal"),
                draft_input_target_hidden.get("structural_equal"),
                (draft.get("cache") or {}).get("structural_equal"),
            )
        )
        candidate_variation = any(
            value is False
            for value in (
                draft.get("top1_candidates_equal"),
                draft.get("topk_candidates_equal"),
                draft.get("original_proposal_equals_tree_top1"),
            )
        )
        proposal_selected_path_drift = bool(
            draft.get("actual_proposal_top1_drift")
            and transition["actual_mismatch"]
            and (
                transition["selected_path_equal"] is False
                or transition["next_root_equal"] is False
            )
        )
        semantic_consequence = bool(
            native_online_verifier_flip
            or transition["actual_mismatch"]
            or faithful_probe_decision_divergence
            or proposal_selected_path_drift
        )
        rows.append(
            {
                "generated_prefix_length": prefix,
                "absolute_prefix_alignment_used": True,
                "comparison_status": (
                    f"{reference.get('checkpoint_provenance')}"
                    f"_vs_{tree.get('checkpoint_provenance')}"
                    if directly_comparable
                    else "not_directly_comparable"
                ),
                "reference_checkpoint_provenance": reference.get(
                    "checkpoint_provenance"
                )
                if reference
                else "not_directly_comparable",
                "tree_checkpoint_provenance": tree.get("checkpoint_provenance")
                if tree
                else "not_directly_comparable",
                "semantic_prefix_token_exact": semantic_prefix_equal
                if directly_comparable
                else None,
                "reference_round_index": _round_index(reference, "reference"),
                "tree_round_index": _round_index(tree, "tree"),
                "round_segmentation_equal_assumed": False,
                "target_kv": target_kv,
                "target_hidden": target_hidden,
                "target_hidden_used_by_draft": draft_input_target_hidden,
                "draft_state": draft,
                "semantic_probe": semantic_probe,
                **probe_fidelity,
                "faithful_probe_decision_divergence": (
                    faithful_probe_decision_divergence
                ),
                "packed_verifier_evidence": verifier,
                "shadow_native_canonical_decision": tree.get("shadow_decision")
                if tree
                else None,
                "native_online_verifier_flip": native_online_verifier_flip,
                "same_mask_control_flip": same_mask_control_flip,
                "math_control_flip": math_control_flip,
                "local_packed_verifier_decision_flip": (native_online_verifier_flip),
                "commit_transition_comparable": transition["comparable"],
                "commit_transition_same_entry_prefix": transition["same_entry_prefix"],
                "commit_transition_actual_execution_evidence": transition[
                    "actual_execution_evidence"
                ],
                "commit_transition_alignment_status": transition["alignment_status"],
                "commit_transition_semantics_equal": transition["semantics_equal"],
                "commit_selected_path_equal": transition["selected_path_equal"],
                "commit_next_root_equal": transition["next_root_equal"],
                "reference_commit_transition": transition["reference"],
                "tree_commit_transition": transition["tree"],
                "numerical_state_difference_observed": numerical_state_difference,
                "structural_state_difference_observed": structural_state_difference,
                "draft_candidate_difference_observed": candidate_variation,
                "candidate_alignment_status": draft.get("candidate_alignment_status"),
                "candidate_alignment_reason": draft.get("candidate_alignment_reason"),
                "reference_actual_proposal_token_ids": draft.get(
                    "reference_actual_proposal_token_ids"
                ),
                "reference_candidate_top1_token_ids": draft.get(
                    "reference_candidate_top1_token_ids"
                ),
                "tree_candidate_top1_token_ids": draft.get(
                    "tree_candidate_top1_token_ids"
                ),
                "proposal_top1_semantically_aligned": draft.get(
                    "proposal_top1_semantically_aligned"
                ),
                "actual_proposal_top1_drift": draft.get(
                    "actual_proposal_top1_drift", False
                ),
                "proposal_selected_path_drift": proposal_selected_path_drift,
                "semantic_consequence_observed": semantic_consequence,
                "mere_numerical_variation_only": bool(
                    (numerical_state_difference or candidate_variation)
                    and not structural_state_difference
                    and not semantic_consequence
                ),
                "downstream_consequence_only": bool(
                    transition["actual_mismatch"]
                    and (
                        proposal_selected_path_drift
                        or native_online_verifier_flip
                        or faithful_probe_decision_divergence
                    )
                ),
            }
        )
        layer_rows.extend(target_layers)
        layer_rows.extend(draft_layers)

    first_state = next(
        (
            row
            for row in rows
            if row["structural_state_difference_observed"]
            or row["numerical_state_difference_observed"]
            or row["draft_candidate_difference_observed"]
            or row["native_online_verifier_flip"]
        ),
        None,
    )
    first_consequential = next(
        (row for row in rows if row["semantic_consequence_observed"]), None
    )
    classification = classify_reference_state_divergence(
        rows=rows,
        first_output_divergent_offset=first_output_divergent_offset,
    )
    for row in rows:
        row.update(
            {
                "classification_evidence_prefix": classification[
                    "classification_evidence_prefix"
                ],
                "classification_evidence_source": classification[
                    "classification_evidence_source"
                ],
                "classification_priority": classification["classification_priority"],
                "unresolved_reason": classification["unresolved_reason"],
            }
        )
        if (
            row["semantic_consequence_observed"]
            and row["generated_prefix_length"]
            != classification["classification_evidence_prefix"]
        ):
            row["downstream_consequence_only"] = True
    all_expected = (
        set(range(first_output_divergent_offset + 1))
        if first_output_divergent_offset is not None
        else set(prefixes)
    )
    directly_aligned = {
        row["generated_prefix_length"]
        for row in rows
        if row["comparison_status"] != "not_directly_comparable"
    }
    evidence_issues = []
    if not all_expected <= directly_aligned:
        evidence_issues.append("missing_aligned_checkpoint")
    if any(row["semantic_prefix_token_exact"] is False for row in rows):
        evidence_issues.append("semantic_prefix_token_mismatch")
    if any(
        row["candidate_alignment_status"] == "candidate_capture_alignment_inconsistent"
        for row in rows
    ):
        evidence_issues.append("candidate_capture_alignment_inconsistent")
    if any(row["reference_probe_faithful"] is False for row in rows):
        evidence_issues.append("unfaithful_reference_decision_probe")
    evidence_complete = not evidence_issues
    summary = {
        "first_output_divergent_generated_token_offset": (
            first_output_divergent_offset
        ),
        "first_state_divergent_aligned_prefix": first_state["generated_prefix_length"]
        if first_state
        else None,
        "first_semantically_consequential_state_divergence": first_consequential[
            "generated_prefix_length"
        ]
        if first_consequential
        else None,
        "affected_component": classification["affected_component"],
        "reference_tree_round_mapping": (
            {
                "reference_round_index": first_consequential["reference_round_index"],
                "tree_round_index": first_consequential["tree_round_index"],
                "generated_prefix_length": first_consequential[
                    "generated_prefix_length"
                ],
            }
            if first_consequential
            else None
        ),
        "evidence_completeness": "complete" if evidence_complete else "partial",
        "evidence_completeness_issues": evidence_issues,
        "failure_classification": classification["failure_classification"],
        "classification_evidence_prefix": classification[
            "classification_evidence_prefix"
        ],
        "classification_evidence_source": classification[
            "classification_evidence_source"
        ],
        "classification_priority": classification["classification_priority"],
        "unresolved_reason": classification["unresolved_reason"],
        "downstream_consequence_count": sum(
            bool(row["downstream_consequence_only"]) for row in rows
        ),
        "numerical_variation_only_prefix_count": sum(
            bool(row["mere_numerical_variation_only"]) for row in rows
        ),
        "aligned_checkpoint_count": len(directly_aligned),
        "requested_checkpoint_count": len(all_expected),
        "absolute_generated_prefix_alignment_used": True,
        "target_only_used_as_authoritative_oracle": False,
    }
    return {"rows": rows, "layer_rows": layer_rows, "summary": summary}


def classify_reference_state_divergence(
    *, rows: list[dict[str, Any]], first_output_divergent_offset: int | None
) -> dict[str, Any]:
    """Classify only semantically consequential evidence, never raw drift alone."""

    if first_output_divergent_offset is None:
        return _classification_result(None, None, None, None, None)

    native = next((row for row in rows if row["native_online_verifier_flip"]), None)
    if native is not None:
        return _classification_result(
            "local_packed_verifier_numerical_divergence",
            "target_verifier",
            native["generated_prefix_length"],
            "actual_native_online_verifier_argmax_flip",
            1,
        )

    actual_semantic_prefixes = [
        row["generated_prefix_length"]
        for row in rows
        if row["faithful_probe_decision_divergence"]
        or row["proposal_selected_path_drift"]
    ]
    for row in rows:
        transition_is_first = not any(
            prefix <= row["generated_prefix_length"]
            for prefix in actual_semantic_prefixes
        )
        transition_explained_by_proposal = row["proposal_selected_path_drift"]
        if (
            row["commit_transition_semantics_equal"] is False
            and row["commit_transition_same_entry_prefix"]
            and row["commit_transition_actual_execution_evidence"]
            and transition_is_first
            and not transition_explained_by_proposal
        ):
            return _classification_result(
                "commit_or_root_transition_mismatch",
                "root_or_commit_transition",
                row["generated_prefix_length"],
                "actual_same_entry_commit_transition",
                2,
            )

    for row in rows:
        if not row["faithful_probe_decision_divergence"]:
            continue
        prior = [
            candidate
            for candidate in rows
            if candidate["generated_prefix_length"] <= row["generated_prefix_length"]
        ]
        if any(
            candidate["target_kv"].get("numerically_equal") is False
            or candidate["target_kv"].get("structural_equal") is False
            for candidate in prior
        ):
            return _classification_result(
                "target_kv_state_drift",
                "persistent_target_kv",
                row["generated_prefix_length"],
                "faithful_actual_target_decision_after_target_kv_drift",
                3,
            )
        if any(
            candidate["target_hidden"].get("numerically_equal") is False
            or candidate["target_hidden"].get("structural_equal") is False
            or candidate["target_hidden_used_by_draft"].get("numerically_equal")
            is False
            or candidate["target_hidden_used_by_draft"].get("structural_equal") is False
            for candidate in prior
        ):
            return _classification_result(
                "target_hidden_state_drift",
                "target_hidden",
                row["generated_prefix_length"],
                "faithful_actual_target_decision_after_target_hidden_drift",
                3,
            )

    proposal = next((row for row in rows if row["proposal_selected_path_drift"]), None)
    if proposal is not None:
        return _classification_result(
            "draft_state_or_candidate_drift",
            "draft_state_or_candidates",
            proposal["generated_prefix_length"],
            "aligned_actual_proposal_drift_changed_selected_path",
            4,
        )

    if any(row["reference_probe_faithful"] is False for row in rows):
        reason = "unfaithful_reference_decision_probe"
    elif any(
        row["candidate_alignment_status"] == "candidate_capture_alignment_inconsistent"
        for row in rows
    ):
        reason = "candidate_capture_alignment_inconsistent"
    elif any(row["same_mask_control_flip"] for row in rows) and not any(
        row["native_online_verifier_flip"] for row in rows
    ):
        reason = "counterfactual_same_mask_control_only"
    elif any(row["math_control_flip"] for row in rows):
        reason = "counterfactual_math_control_only"
    else:
        reason = "no_faithful_actual_semantic_cause_observed"
    return _classification_result(
        "unresolved_reference_state_divergence",
        None,
        None,
        "insufficient_actual_causal_evidence",
        5,
        unresolved_reason=reason,
    )


def _classification_result(
    failure_classification: str | None,
    affected_component: str | None,
    evidence_prefix: int | None,
    evidence_source: str | None,
    priority: int | None,
    *,
    unresolved_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "failure_classification": failure_classification,
        "affected_component": affected_component,
        "classification_evidence_prefix": evidence_prefix,
        "classification_evidence_source": evidence_source,
        "classification_priority": priority,
        "unresolved_reason": unresolved_reason,
    }


def verifier_evidence_by_prefix(
    *,
    gold_records: list[dict[str, Any]],
    prompt_length: int,
) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for record in gold_records:
        round_index = int(record["round_index"])
        depth = record.get("node_depth")
        predicted_position = record.get("predicted_token_semantic_position")
        if predicted_position is not None:
            prefix = int(predicted_position) - int(prompt_length)
        elif record.get("persistent_prefix_length") is not None and depth is not None:
            prefix = (
                int(record["persistent_prefix_length"])
                - int(prompt_length)
                + int(depth)
                + 1
            )
        else:
            continue
        if prefix < 0:
            continue
        result.setdefault(prefix, []).append(
            {
                "round_index": round_index,
                "generated_prefix_length": prefix,
                "comparison_mode": record.get("comparison_mode"),
                "availability": record.get("availability"),
                "structural_status": record.get("structural_status"),
                "semantic_status": record.get("semantic_status"),
                "argmax_equal": record.get("argmax_equal"),
                "verifier_decision_relevant": record.get("verifier_decision_relevant"),
                "max_abs_error": record.get("max_abs_error"),
                "mean_abs_error": record.get("mean_abs_error"),
                "p99_abs_error": record.get("p99_abs_error"),
                "packed_top1_top2_margin": record.get("packed_top1_top2_margin"),
                "dense_top1_top2_margin": record.get("dense_top1_top2_margin"),
            }
        )
    return result
