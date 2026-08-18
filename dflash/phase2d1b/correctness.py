"""Phase 2D.1B actual-execution correctness gate.

Phase 2C Gold replays remain diagnostic attribution controls. Phase 2D.1B
requires token-exact actual execution plus structural, KV-transition, and
traversal correctness. A replay-side semantic mismatch is hard only when it
has an actual-execution consequence (or when structural/evidence requirements
independently make the run invalid).
"""

from __future__ import annotations

from typing import Any

DIAGNOSTIC_NUMERICAL_FAILURE_CLASSES = {
    "numerical_backend_variation",
    "unresolved_numerical_variation",
    "strict_numerical_threshold_exceedance",
    "topk_ordering_variation",
}

_REPLAY_EVIDENCE_FIELDS = (
    "round_index",
    "comparison_mode",
    "status",
    "structural_status",
    "semantic_status",
    "numerical_status",
    "verifier_decision_changed",
    "argmax_equal",
    "selected_token_id",
    "selected_token_packed_rank",
    "selected_token_dense_rank",
    "top8_ordered_equal",
    "thresholds",
    "warning",
    "reason",
)


def _failed_status(rows: list[dict[str, Any]], field: str) -> list[int]:
    return [
        int(row.get("round_index", -1))
        for row in rows
        if row.get(field) == "fail"
    ]


def _replay_semantic_mismatch(row: dict[str, Any]) -> bool:
    return bool(
        row.get("semantic_status") == "fail"
        or row.get("status") == "fail_semantic"
        or row.get("verifier_decision_changed") is True
        or row.get("argmax_equal") is False
    )


def _replay_evidence(
    source: str, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "source": source,
            **{
                field: row[field]
                for field in _REPLAY_EVIDENCE_FIELDS
                if field in row
            },
        }
        for row in rows
        if _replay_semantic_mismatch(row)
    ]


def classify_actual_execution_correctness(
    correctness: dict[str, Any],
    *,
    gold_logits: list[dict[str, Any]],
    gold_kv: list[dict[str, Any]],
    traversal_checks: list[dict[str, Any]],
    gold_checks_requested: bool,
) -> dict[str, Any]:
    """Separate hard online failures from replay-only semantic variation."""

    hard_failures: list[str] = []
    actual_semantic_failures: list[str] = []
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
    actual_semantic_failures.extend(
        f"actual_output:{name}" for name in failed_output_relations
    )

    if correctness.get("top1_round_exact_equal") is False:
        actual_semantic_failures.append("top1_round_semantics_mismatch")
    for field in (
        "actual_round_semantics_equal",
        "actual_committed_path_equal",
        "actual_kv_commit_equal",
    ):
        if correctness.get(field) is False:
            actual_semantic_failures.append(f"{field}:false")

    logit_structural = _failed_status(gold_logits, "structural_status")
    logit_structural_issues = [
        int(row.get("round_index", -1))
        for row in gold_logits
        if row.get("structural_issues")
    ]
    kv_structural = _failed_status(gold_kv, "structural_status")
    kv_status_structural = [
        int(row.get("round_index", -1))
        for row in gold_kv
        if row.get("status") == "fail_structural"
    ]

    traversal_structural = []
    traversal_semantic = []
    for row in traversal_checks:
        round_index = int(row.get("round_index", -1))
        checks = row.get("checks", {})
        structural_failed = bool(
            row.get("structural_mask_issues")
            or checks.get("structural_checks_passed") is False
        )
        semantic_failed = bool(
            checks.get("traversal_checks_passed") is False
            or checks.get("kv_root_transition_checks_passed") is False
        )
        if (
            row.get("status") == "fail"
            or checks.get("all_checks_passed") is False
        ):
            # Legacy rows may only expose the aggregate status. Treat that as
            # an actual traversal failure rather than silently downgrading it.
            semantic_failed = semantic_failed or not structural_failed
        if structural_failed:
            traversal_structural.append(round_index)
        if semantic_failed:
            traversal_semantic.append(round_index)

    if logit_structural:
        hard_failures.append(f"gold_logit_structural_failure:{logit_structural}")
    if logit_structural_issues:
        hard_failures.append(
            f"gold_logit_structural_issues:{logit_structural_issues}"
        )
    if kv_structural:
        hard_failures.append(f"gold_kv_structural_failure:{kv_structural}")
    if kv_status_structural:
        hard_failures.append(f"gold_kv_failure:{kv_status_structural}")
    if traversal_structural:
        hard_failures.append(
            f"tree_traversal_structural_failure:{traversal_structural}"
        )
    if traversal_semantic:
        actual_semantic_failures.append(
            f"tree_traversal_or_kv_transition_failure:{traversal_semantic}"
        )

    if correctness.get("gold_status") == "fail_structural":
        hard_failures.append("gold_overall_structural_failure")
    if gold_checks_requested and correctness.get("gold_status") == "not_run":
        hard_failures.append("gold_checks_not_run")
    if gold_checks_requested and correctness.get("gold_uncompleted_checks"):
        hard_failures.append("gold_checks_incomplete")

    output_failure_class = correctness.get("output_divergence_failure_class")
    if output_failure_class is not None:
        actual_semantic_failures.append(
            f"output_relation_failure:{output_failure_class}"
        )

    replay_evidence = [
        *_replay_evidence("gold_logit", gold_logits),
        *_replay_evidence("gold_kv", gold_kv),
    ]
    replay_rounds = sorted(
        {int(row.get("round_index", -1)) for row in replay_evidence}
    )
    actual_execution_semantic_failure = bool(actual_semantic_failures)
    replay_with_actual_consequence = bool(
        replay_evidence and actual_execution_semantic_failure
    )
    if replay_with_actual_consequence:
        hard_failures.append(
            "replay_semantic_mismatch_with_actual_consequence:"
            f"{replay_rounds}"
        )

    numerical_threshold_variation = any(
        row.get("numerical_status") == "outside_strict_thresholds"
        for row in [*gold_logits, *gold_kv]
    )
    topk_only_variation = any(
        row.get("top8_ordered_equal") is False
        and row.get("argmax_equal") is not False
        for row in [*gold_logits, *gold_kv]
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

    structural_checks_passed = not any(
        (
            logit_structural,
            logit_structural_issues,
            kv_structural,
            kv_status_structural,
            traversal_structural,
        )
    )
    evidence_complete = not any(
        reason in {"gold_checks_not_run", "gold_checks_incomplete"}
        for reason in hard_failures
    )
    replay_without_actual_consequence = bool(
        replay_evidence
        and not actual_execution_semantic_failure
        and structural_checks_passed
        and evidence_complete
    )

    gold_fail_semantic_is_diagnostic = bool(
        correctness.get("gold_status") == "fail_semantic"
        and not actual_execution_semantic_failure
        and structural_checks_passed
        and evidence_complete
        and (
            replay_without_actual_consequence
            or classified_numerical_variation
            or numerical_threshold_variation
            or topk_only_variation
        )
    )
    if (
        correctness.get("gold_status") == "fail_semantic"
        and not gold_fail_semantic_is_diagnostic
        and not replay_with_actual_consequence
    ):
        hard_failures.append(
            f"gold_semantic_failure:{failure_class or 'unclassified'}"
        )

    hard_failures.extend(actual_semantic_failures)
    unique_hard_failures = list(dict.fromkeys(hard_failures))
    unique_actual_failures = list(dict.fromkeys(actual_semantic_failures))
    unique_diagnostics = list(dict.fromkeys(diagnostic_reasons))
    passed = not unique_hard_failures
    return {
        "actual_execution_correctness_passed": passed,
        "actual_output_exactness_required": True,
        "actual_output_relations": actual_output_relations,
        "actual_execution_semantic_failure": actual_execution_semantic_failure,
        "actual_execution_semantic_failure_reasons": unique_actual_failures,
        "hard_correctness_failure_reasons": unique_hard_failures,
        "diagnostic_numerical_variation_observed": bool(
            classified_numerical_variation
            or numerical_threshold_variation
            or topk_only_variation
        ),
        "diagnostic_numerical_variation_reasons": unique_diagnostics,
        "replay_semantic_mismatch": bool(replay_evidence),
        "replay_semantic_mismatch_with_actual_consequence": (
            replay_with_actual_consequence
        ),
        "replay_semantic_variation_without_actual_consequence": (
            replay_without_actual_consequence
        ),
        "replay_semantic_variation_observed": bool(replay_evidence),
        "replay_semantic_variation_rounds": replay_rounds,
        "replay_semantic_variation_with_actual_consequence": (
            replay_with_actual_consequence
        ),
        "replay_semantic_classification": (
            "replay_semantic_mismatch_with_actual_consequence"
            if replay_with_actual_consequence
            else "replay_semantic_variation_without_actual_consequence"
            if replay_without_actual_consequence
            else "none"
        ),
        "replay_semantic_variation_evidence": replay_evidence,
        "gold_fail_semantic_treated_as_diagnostic": (
            gold_fail_semantic_is_diagnostic
        ),
        "structural_checks_passed": structural_checks_passed,
        "raw_structural_and_semantic_checks_passed": bool(
            structural_checks_passed
            and not replay_evidence
            and not traversal_semantic
        ),
        "actual_execution_is_semantic_oracle": True,
        "gold_replays_are_diagnostic_attribution_controls": True,
    }
