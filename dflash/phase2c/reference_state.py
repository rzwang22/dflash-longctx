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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if reference is None or tree is None:
        return (
            {
                "status": "not_directly_comparable",
                "top1_candidates_equal": None,
                "topk_candidates_equal": None,
                "cache": None,
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
    reference_proposal = reference_tree_input.get("actual_proposed_token_ids")
    tree_top1 = tree.get("candidate_top1_token_ids")
    return (
        {
            "status": "available",
            "top1_candidates_equal": reference.get("candidate_top1_token_ids")
            == tree.get("candidate_top1_token_ids"),
            "topk_candidates_equal": reference.get("candidate_topk_token_ids")
            == tree.get("candidate_topk_token_ids"),
            "reference_top1_token_ids": reference.get("candidate_top1_token_ids"),
            "tree_top1_token_ids": tree.get("candidate_top1_token_ids"),
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
            "cache": cache,
            "reference_candidate_tree_input": reference_tree_input,
            "tree_candidate_tree_input": tree_tree_input,
        },
        layers,
    )


def compare_reference_tree_checkpoints(
    *,
    reference_checkpoints: list[dict[str, Any]],
    tree_checkpoints: list[dict[str, Any]],
    first_output_divergent_offset: int | None,
    packed_verifier_evidence: dict[int, list[dict[str, Any]]] | None = None,
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
        )
        verifier = packed_verifier_evidence.get(prefix, [])
        verifier_flip = any(
            row.get("verifier_decision_relevant")
            and row.get("argmax_equal") is False
            and row.get("comparison_mode") in {"native", "same_mask"}
            for row in verifier
        )
        transition_reference = reference.get("commit_transition") if reference else None
        transition_tree = tree.get("commit_transition") if tree else None
        transition_comparable = (
            transition_reference is not None and transition_tree is not None
        )
        transition_same_entry_prefix = bool(
            transition_comparable
            and transition_reference.get("generated_prefix_before")
            == transition_tree.get("generated_prefix_before")
        )
        transition_semantics_equal = (
            transition_reference.get("committed_token_ids")
            == transition_tree.get("committed_token_ids")
            and transition_reference.get("next_root_token_id")
            == transition_tree.get("next_root_token_id")
            and transition_reference.get("next_root_kind")
            == transition_tree.get("next_root_kind")
            if transition_same_entry_prefix
            else None
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
        candidate_difference = any(
            value is False
            for value in (
                draft.get("top1_candidates_equal"),
                draft.get("topk_candidates_equal"),
                draft.get("original_proposal_equals_tree_top1"),
            )
        )
        semantic_consequence = (
            any(
                value is False
                for value in (
                    semantic_probe.get("argmax_equal"),
                    draft.get("top1_candidates_equal"),
                    transition_semantics_equal,
                )
            )
            or verifier_flip
            or candidate_difference
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
                "packed_verifier_evidence": verifier,
                "shadow_native_canonical_decision": tree.get("shadow_decision")
                if tree
                else None,
                "local_packed_verifier_decision_flip": verifier_flip,
                "commit_transition_comparable": transition_comparable,
                "commit_transition_same_entry_prefix": (transition_same_entry_prefix),
                "commit_transition_alignment_status": (
                    "same_absolute_entry_and_exit_prefix"
                    if transition_same_entry_prefix
                    else "different_round_segmentation"
                    if transition_comparable
                    else "not_directly_comparable"
                ),
                "commit_transition_semantics_equal": transition_semantics_equal,
                "reference_commit_transition": transition_reference,
                "tree_commit_transition": transition_tree,
                "numerical_state_difference_observed": numerical_state_difference,
                "structural_state_difference_observed": structural_state_difference,
                "draft_candidate_difference_observed": candidate_difference,
                "semantic_consequence_observed": semantic_consequence,
                "mere_numerical_variation_only": bool(
                    numerical_state_difference
                    and not structural_state_difference
                    and not candidate_difference
                    and not semantic_consequence
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
            or row["local_packed_verifier_decision_flip"]
        ),
        None,
    )
    first_consequential = next(
        (row for row in rows if row["semantic_consequence_observed"]), None
    )
    classification, component = classify_reference_state_divergence(
        rows=rows,
        first_output_divergent_offset=first_output_divergent_offset,
    )
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
    evidence_complete = all_expected <= directly_aligned and all(
        row["semantic_prefix_token_exact"] is not False for row in rows
    )
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
        "affected_component": component,
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
        "failure_classification": classification,
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
) -> tuple[str | None, str | None]:
    """Classify only semantically consequential evidence, never raw drift alone."""

    if first_output_divergent_offset is None:
        return None, None
    for row in rows:
        if not row["semantic_consequence_observed"]:
            continue
        if row["local_packed_verifier_decision_flip"]:
            return "local_packed_verifier_numerical_divergence", "target_verifier"
        if row["commit_transition_semantics_equal"] is False:
            return "commit_or_root_transition_mismatch", "root_or_commit_transition"
        if row["semantic_probe"].get("argmax_equal") is False:
            prior = [
                candidate
                for candidate in rows
                if candidate["generated_prefix_length"]
                <= row["generated_prefix_length"]
            ]
            if any(
                candidate["target_kv"].get("numerically_equal") is False
                or candidate["target_kv"].get("structural_equal") is False
                for candidate in prior
            ):
                return "target_kv_state_drift", "persistent_target_kv"
            if any(
                candidate["target_hidden"].get("numerically_equal") is False
                or candidate["target_hidden"].get("structural_equal") is False
                or candidate["target_hidden_used_by_draft"].get("numerically_equal")
                is False
                or candidate["target_hidden_used_by_draft"].get("structural_equal")
                is False
                for candidate in prior
            ):
                return "target_hidden_state_drift", "target_hidden"
            return "unresolved_reference_state_divergence", "target_verifier"
        if any(
            row["draft_state"].get(field) is False
            for field in (
                "top1_candidates_equal",
                "topk_candidates_equal",
                "original_proposal_equals_tree_top1",
            )
        ):
            prior = [
                candidate
                for candidate in rows
                if candidate["generated_prefix_length"]
                <= row["generated_prefix_length"]
            ]
            if any(
                candidate["target_hidden"].get("numerically_equal") is False
                or candidate["target_hidden"].get("structural_equal") is False
                or candidate["target_hidden_used_by_draft"].get("numerically_equal")
                is False
                or candidate["target_hidden_used_by_draft"].get("structural_equal")
                is False
                for candidate in prior
            ):
                return "target_hidden_state_drift", "target_hidden"
            return "draft_state_or_candidate_drift", "draft_state_or_candidates"
    return "unresolved_reference_state_divergence", None


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
