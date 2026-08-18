"""Phase 2D.1B actual-execution correctness gate.

Phase 2C Gold replays remain diagnostic attribution controls.  Phase 2D.1B
requires token-exact actual execution plus structural, KV, and traversal
correctness, while retaining consequence-free numerical variation as evidence.
"""

from __future__ import annotations

from typing import Any

DIAGNOSTIC_NUMERICAL_FAILURE_CLASSES = {
    "numerical_backend_variation",
    "unresolved_numerical_variation",
    "strict_numerical_threshold_exceedance",
    "topk_ordering_variation",
}


def _failed_status(rows: list[dict[str, Any]], field: str) -> list[int]:
    return [
        int(row.get("round_index", -1))
        for row in rows
        if row.get(field) == "fail"
    ]


def classify_actual_execution_correctness(
    correctness: dict[str, Any],
    *,
    gold_logits: list[dict[str, Any]],
    gold_kv: list[dict[str, Any]],
    traversal_checks: list[dict[str, Any]],
    gold_checks_requested: bool,
) -> dict[str, Any]:
    """Separate hard actual-execution failures from numerical diagnostics."""

    hard_failures: list[str] = []
    diagnostic_reasons: list[str] = []
    actual_output_relations = {
        "reference_repeat_output_exact": correctness.get(
            "reference_output_ids_equal"
        )
        is True,
        "original_dflash_reference_tree_output_exact": correctness.get(
            "original_dflash_reference_tree_output_ids_equal"
        )
        is True,
        "control_profile_output_exact": correctness.get(
            "control_profile_output_ids_equal"
        )
        is True,
    }
    if gold_checks_requested:
        actual_output_relations["gold_output_exact"] = (
            correctness.get("gold_output_ids_equal") is True
        )
    failed_output_relations = [
        name for name, passed in actual_output_relations.items() if not passed
    ]
    hard_failures.extend(f"actual_output:{name}" for name in failed_output_relations)

    if correctness.get("top1_round_exact_equal") is False:
        hard_failures.append("top1_round_semantics_mismatch")

    logit_structural = _failed_status(gold_logits, "structural_status")
    logit_structural_issues = [
        int(row.get("round_index", -1))
        for row in gold_logits
        if row.get("structural_issues")
    ]
    logit_semantic = _failed_status(gold_logits, "semantic_status")
    kv_structural = _failed_status(gold_kv, "structural_status")
    kv_semantic = _failed_status(gold_kv, "semantic_status")
    kv_status_failures = [
        int(row.get("round_index", -1))
        for row in gold_kv
        if row.get("status") in {"fail_structural", "fail_semantic"}
    ]
    traversal_failed = [
        int(row.get("round_index", -1))
        for row in traversal_checks
        if row.get("status") == "fail"
        or row.get("checks", {}).get("all_checks_passed") is False
        or bool(row.get("structural_mask_issues"))
    ]
    if logit_structural:
        hard_failures.append(f"gold_logit_structural_failure:{logit_structural}")
    if logit_structural_issues:
        hard_failures.append(
            f"gold_logit_structural_issues:{logit_structural_issues}"
        )
    if logit_semantic:
        hard_failures.append(f"gold_logit_semantic_failure:{logit_semantic}")
    if kv_structural:
        hard_failures.append(f"gold_kv_structural_failure:{kv_structural}")
    if kv_semantic:
        hard_failures.append(f"gold_kv_semantic_failure:{kv_semantic}")
    if kv_status_failures:
        hard_failures.append(f"gold_kv_failure:{kv_status_failures}")
    if traversal_failed:
        hard_failures.append(f"tree_traversal_failure:{traversal_failed}")

    if correctness.get("gold_status") == "fail_structural":
        hard_failures.append("gold_overall_structural_failure")
    if gold_checks_requested and correctness.get("gold_status") == "not_run":
        hard_failures.append("gold_checks_not_run")
    if gold_checks_requested and correctness.get("gold_uncompleted_checks"):
        hard_failures.append("gold_checks_incomplete")

    numerical_threshold_variation = any(
        row.get("numerical_status") == "outside_strict_thresholds"
        for row in [*gold_logits, *gold_kv]
    )
    topk_only_variation = any(
        row.get("top8_ordered_equal") is False
        and row.get("argmax_equal") is not False
        for row in gold_logits
    )
    failure_class = correctness.get("gold_failure_class")
    classified_numerical_variation = (
        failure_class in DIAGNOSTIC_NUMERICAL_FAILURE_CLASSES
    )
    if classified_numerical_variation:
        diagnostic_reasons.append(str(failure_class))
    if numerical_threshold_variation:
        diagnostic_reasons.append("strict_numerical_threshold_exceedance")
    if topk_only_variation:
        diagnostic_reasons.append("topk_ordering_variation")
    diagnostic_variation_evidence = bool(
        classified_numerical_variation
        or numerical_threshold_variation
        or topk_only_variation
    )

    raw_semantics_clean = not any(
        (
            logit_structural,
            logit_structural_issues,
            logit_semantic,
            kv_structural,
            kv_semantic,
            kv_status_failures,
            traversal_failed,
        )
    )
    actual_outputs_exact = not failed_output_relations
    gold_fail_semantic_is_diagnostic = bool(
        correctness.get("gold_status") == "fail_semantic"
        and diagnostic_variation_evidence
        and raw_semantics_clean
        and actual_outputs_exact
    )
    if (
        correctness.get("gold_status") == "fail_semantic"
        and not gold_fail_semantic_is_diagnostic
    ):
        hard_failures.append(
            f"gold_semantic_failure:{failure_class or 'unclassified'}"
        )

    output_failure_class = correctness.get("output_divergence_failure_class")
    if output_failure_class is not None:
        hard_failures.append(f"output_relation_failure:{output_failure_class}")

    unique_hard_failures = list(dict.fromkeys(hard_failures))
    unique_diagnostics = list(dict.fromkeys(diagnostic_reasons))
    passed = not unique_hard_failures
    return {
        "actual_execution_correctness_passed": passed,
        "actual_output_exactness_required": True,
        "actual_output_relations": actual_output_relations,
        "hard_correctness_failure_reasons": unique_hard_failures,
        "diagnostic_numerical_variation_observed": bool(unique_diagnostics),
        "diagnostic_numerical_variation_reasons": unique_diagnostics,
        "gold_fail_semantic_treated_as_diagnostic": (
            gold_fail_semantic_is_diagnostic
        ),
        "raw_structural_and_semantic_checks_passed": raw_semantics_clean,
        "actual_execution_is_semantic_oracle": True,
        "gold_replays_are_diagnostic_attribution_controls": True,
    }
