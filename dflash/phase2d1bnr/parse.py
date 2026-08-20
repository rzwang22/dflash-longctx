"""Parse Phase 2C subprocess artifacts into NR diagnostic observations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXPECTED_ARTIFACTS = (
    "manifest.json",
    "correctness-summary.json",
    "output-divergence.json",
    "samples.jsonl",
    "tree-rounds.jsonl",
    "decision-stabilization-rounds.jsonl",
    "tree-traversal-checks.jsonl",
    "gold-logit-checks.jsonl",
    "gold-kv-checks.jsonl",
    "source-provenance.json",
)

OUTCOME_CATEGORIES = (
    "pass_actual_exact",
    "actual_numerical_decision_divergence",
    "diagnostic_only_numerical_variation",
    "actual_structural_failure",
    "reference_instability",
    "incomplete_correctness_evidence",
    "infrastructure_failure",
)

REQUIRED_DIVERGENCE_GOLD_LOGIT_MODES = frozenset({"native", "same_mask", "math"})
CAUSALLY_SKIPPABLE_DOWNSTREAM_GOLD_CHECKS = frozenset(
    {"kv_compaction", "kv_next_step_logits"}
)
CONFIRMED_DIVERGENCE_SKIP_REASONS = frozenset({"prior_logits_gold_failure"})


class ArtifactError(ValueError):
    """A subprocess output cannot support a valid NR observation."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"line {line_number} is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ArtifactError(f"cannot read valid JSONL: {path}") from exc
    return rows


def _pairwise(record: dict[str, Any], left: str, right: str) -> dict[str, Any]:
    for row in record.get("pairwise_comparisons", []):
        if row.get("left_name") == left and row.get("right_name") == right:
            return row
    raise ArtifactError(f"missing pairwise comparison {left!r} vs {right!r}")


def _single(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if len(rows) != 1:
        raise ArtifactError(f"expected exactly one {label}, found {len(rows)}")
    return rows[0]


def _structural_failure(
    gold_logits: list[dict[str, Any]],
    gold_kv: list[dict[str, Any]],
    traversal: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    reasons = []
    for source, rows in (("gold_logit", gold_logits), ("gold_kv", gold_kv)):
        failed = sorted(
            {
                int(row.get("round_index", -1))
                for row in rows
                if row.get("structural_status") == "fail"
                or row.get("status") == "fail_structural"
                or bool(row.get("structural_issues"))
            }
        )
        if failed:
            reasons.append(f"{source}:{failed}")
    failed_traversal = []
    for row in traversal:
        checks = row.get("checks", {})
        if (
            row.get("status") == "fail"
            or checks.get("all_checks_passed") is False
            or checks.get("structural_checks_passed") is False
            or checks.get("traversal_checks_passed") is False
            or checks.get("kv_root_transition_checks_passed") is False
            or bool(row.get("structural_mask_issues"))
        ):
            failed_traversal.append(int(row.get("round_index", -1)))
    if failed_traversal:
        reasons.append(f"tree_traversal:{sorted(set(failed_traversal))}")
    return bool(reasons), reasons


def _gold_row_has_semantic_mismatch(row: dict[str, Any]) -> bool:
    return bool(
        row.get("semantic_status") == "fail"
        or row.get("status") == "fail_semantic"
        or row.get("verifier_decision_changed") is True
        or row.get("argmax_equal") is False
    )


def _gold_row_has_numerical_variation(row: dict[str, Any]) -> bool:
    return bool(
        row.get("numerical_status") == "outside_strict_thresholds"
        or row.get("top8_ordered_equal") is False
    )


def replay_semantic_rows(
    identity: dict[str, Any],
    gold_logits: list[dict[str, Any]],
    gold_kv: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retain every replay semantic/numerical variation without rewriting it."""

    result = []
    fields = (
        "comparison_mode",
        "availability",
        "status",
        "structural_status",
        "semantic_status",
        "numerical_status",
        "packed_argmax_token_id",
        "reference_argmax_token_id",
        "dense_argmax_token_id",
        "selected_token_id",
        "selected_token_packed_rank",
        "selected_token_dense_rank",
        "packed_top1_top2_margin",
        "dense_top1_top2_margin",
        "verifier_decision_changed",
        "verifier_decision_relevant",
        "argmax_equal",
        "top8_ordered_equal",
        "max_abs_error",
        "mean_abs_error",
        "p99_abs_error",
        "maximum_absolute_error",
        "mean_absolute_error",
        "p99_absolute_error",
        "thresholds",
        "backend",
        "dtype",
        "provenance",
        "warning",
        "reason",
    )
    for source, rows in (("gold_logit", gold_logits), ("gold_kv", gold_kv)):
        for row in rows:
            semantic_mismatch = _gold_row_has_semantic_mismatch(row)
            numerical_variation = _gold_row_has_numerical_variation(row)
            if not semantic_mismatch and not numerical_variation:
                continue
            result.append(
                {
                    "schema_version": "1.0",
                    **identity,
                    "source": source,
                    "round_index": int(row.get("round_index", -1)),
                    "semantic_mismatch": semantic_mismatch,
                    "numerical_variation": numerical_variation,
                    "variation_kind": (
                        "semantic_mismatch"
                        if semantic_mismatch
                        else "numerical_variation_without_semantic_mismatch"
                    ),
                    **{field: row.get(field) for field in fields},
                }
            )
    return result


def _numerical_attribution(failure_class: str | None) -> bool:
    value = (failure_class or "").lower()
    return any(
        marker in value
        for marker in (
            "numerical",
            "packed_verifier",
            "verifier_decision_divergence",
        )
    )


def _traversal_complete_through_round(
    rounds: list[dict[str, Any]],
    traversal: list[dict[str, Any]],
    *,
    boundary_round: int | None,
) -> bool:
    if boundary_round is None:
        return False
    required_rounds = {
        int(row["round_index"])
        for row in rounds
        if int(row.get("profiling_repeat", 0)) == 0
        and int(row["round_index"]) <= boundary_round
    }
    required_rounds.add(boundary_round)
    by_round = {int(row["round_index"]): row for row in traversal}
    if not required_rounds <= set(by_round):
        return False
    for round_index in required_rounds:
        row = by_round[round_index]
        checks = row.get("checks", {})
        if (
            row.get("status") != "pass"
            or checks.get("all_checks_passed") is not True
            or checks.get("structural_checks_passed") is not True
            or checks.get("traversal_checks_passed") is not True
            or checks.get("kv_root_transition_checks_passed") is not True
            or bool(row.get("structural_mask_issues"))
        ):
            return False
    return True


def _gold_evidence_assessment(
    *,
    correctness: dict[str, Any],
    divergence: dict[str, Any],
    rounds: list[dict[str, Any]],
    stabilization: list[dict[str, Any]],
    traversal: list[dict[str, Any]],
    gold_logits: list[dict[str, Any]],
    gold_kv: list[dict[str, Any]],
    actual_equal: bool,
    structural_failure: bool,
    failure_class: str | None,
) -> dict[str, Any]:
    """Separate causally truncated downstream Gold from missing evidence."""

    raw_uncompleted = correctness.get("uncompleted_checks") or []
    raw_details = correctness.get("uncompleted_check_details") or []
    if not isinstance(raw_uncompleted, list) or not all(
        isinstance(value, str) for value in raw_uncompleted
    ):
        raise ArtifactError("correctness uncompleted_checks must be strings")
    if not isinstance(raw_details, list) or not all(
        isinstance(value, dict)
        and isinstance(value.get("check"), str)
        and isinstance(value.get("reason"), str)
        for value in raw_details
    ):
        raise ArtifactError("correctness uncompleted_check_details are malformed")
    uncompleted_checks = sorted(set(raw_uncompleted))
    detail_checks = {str(row["check"]) for row in raw_details}
    skip_reasons = sorted({str(row["reason"]) for row in raw_details})
    divergence_uncompleted = [
        row
        for row in divergence.get("uncompleted_checks", [])
        if row.get("check") != "target_only_precision_attribution"
    ]
    all_modes = {
        str(row["comparison_mode"])
        for row in gold_logits
        if row.get("availability") == "available" and row.get("comparison_mode")
    }
    first_round_raw = divergence.get("divergent_tree_round_index")
    first_round = int(first_round_raw) if first_round_raw is not None else None
    first_round_logits = [
        row
        for row in gold_logits
        if first_round is not None
        and int(row.get("round_index", -1)) == first_round
        and row.get("availability") == "available"
    ]
    first_round_modes = {
        str(row["comparison_mode"])
        for row in first_round_logits
        if row.get("comparison_mode")
    }
    first_round_decision_mismatch = any(
        _gold_row_has_semantic_mismatch(row) for row in first_round_logits
    )
    traversal_complete = _traversal_complete_through_round(
        rounds,
        traversal,
        boundary_round=first_round,
    )
    variation_logits = [
        row
        for row in gold_logits
        if row.get("availability") == "available"
        and (
            _gold_row_has_semantic_mismatch(row)
            or _gold_row_has_numerical_variation(row)
        )
    ]
    variation_rounds = sorted(
        {int(row.get("round_index", -1)) for row in variation_logits}
    )
    first_variation_round = variation_rounds[0] if variation_rounds else None
    first_variation_logits = [
        row
        for row in gold_logits
        if first_variation_round is not None
        and int(row.get("round_index", -1)) == first_variation_round
        and row.get("availability") == "available"
    ]
    first_variation_modes = {
        str(row["comparison_mode"])
        for row in first_variation_logits
        if row.get("comparison_mode")
    }
    first_variation_evidence = any(
        _gold_row_has_semantic_mismatch(row) or _gold_row_has_numerical_variation(row)
        for row in first_variation_logits
    )
    variation_traversal_complete = _traversal_complete_through_round(
        rounds,
        traversal,
        boundary_round=first_variation_round,
    )
    causal_skip_anchor = first_round if not actual_equal else first_variation_round
    causal_skip_shape = bool(
        uncompleted_checks
        and set(uncompleted_checks) <= CAUSALLY_SKIPPABLE_DOWNSTREAM_GOLD_CHECKS
        and detail_checks == set(uncompleted_checks)
        and skip_reasons
        and set(skip_reasons) <= CONFIRMED_DIVERGENCE_SKIP_REASONS
        and any(
            causal_skip_anchor is not None
            and int(row.get("round_index", -1)) == causal_skip_anchor
            and row.get("status") == "not_run"
            and row.get("reason") in CONFIRMED_DIVERGENCE_SKIP_REASONS
            for row in gold_kv
        )
    )
    downstream_after_confirmed_divergence = bool(
        causal_skip_shape
        and not actual_equal
        and divergence.get("classification", {}).get("evidence_complete") is True
        and _numerical_attribution(failure_class)
        and not structural_failure
        and not divergence_uncompleted
        and REQUIRED_DIVERGENCE_GOLD_LOGIT_MODES <= first_round_modes
        and first_round_decision_mismatch
        and traversal_complete
        and rounds
        and stabilization
    )
    downstream_after_confirmed_replay_variation = bool(
        causal_skip_shape
        and actual_equal
        and _numerical_attribution(failure_class)
        and correctness.get("actual_execution_semantic_failure") is not True
        and not structural_failure
        and not divergence_uncompleted
        and REQUIRED_DIVERGENCE_GOLD_LOGIT_MODES <= first_variation_modes
        and first_variation_evidence
        and variation_traversal_complete
        and rounds
        and stabilization
    )
    conventional_complete = bool(
        correctness.get("gold_status") != "not_run"
        and not uncompleted_checks
        and not divergence_uncompleted
        and REQUIRED_DIVERGENCE_GOLD_LOGIT_MODES <= all_modes
        and any(row.get("status") != "not_run" for row in gold_kv)
        and rounds
        and stabilization
        and traversal
    )
    complete = bool(
        conventional_complete
        or downstream_after_confirmed_divergence
        or downstream_after_confirmed_replay_variation
    )
    incomplete_reasons = []
    if not complete:
        if uncompleted_checks:
            incomplete_reasons.append("uncompleted_gold_checks_not_causally_exempt")
        if divergence_uncompleted:
            incomplete_reasons.append("uncompleted_output_divergence_checks")
        if REQUIRED_DIVERGENCE_GOLD_LOGIT_MODES - all_modes:
            incomplete_reasons.append("required_gold_logit_modes_missing")
        if not rounds or not stabilization or not traversal:
            incomplete_reasons.append("round_or_traversal_diagnostics_missing")
        if not any(row.get("status") != "not_run" for row in gold_kv):
            incomplete_reasons.append("completed_gold_kv_evidence_missing")
        if causal_skip_shape and not (
            downstream_after_confirmed_divergence
            or downstream_after_confirmed_replay_variation
        ):
            incomplete_reasons.append("causal_downstream_skip_preconditions_failed")
        if not incomplete_reasons:
            incomplete_reasons.append("gold_evidence_incomplete")
    return {
        "complete": complete,
        "comparison_modes": sorted(all_modes),
        "first_divergence_comparison_modes": sorted(first_round_modes),
        "first_divergence_replay_decision_mismatch_captured": (
            first_round_decision_mismatch
        ),
        "structural_traversal_complete_through_first_divergence": (traversal_complete),
        "first_replay_variation_round": first_variation_round,
        "first_replay_variation_comparison_modes": sorted(first_variation_modes),
        "first_replay_variation_evidence_captured": first_variation_evidence,
        "structural_traversal_complete_through_first_replay_variation": (
            variation_traversal_complete
        ),
        "downstream_gold_checks_skipped_after_confirmed_divergence": (
            downstream_after_confirmed_divergence
        ),
        "downstream_gold_checks_skipped_after_confirmed_replay_variation": (
            downstream_after_confirmed_replay_variation
        ),
        "downstream_gold_skipped_checks": uncompleted_checks,
        "downstream_gold_skip_reasons": skip_reasons,
        "genuinely_incomplete_correctness_evidence": not complete,
        "gold_incompleteness_reasons": incomplete_reasons,
    }


def _source_provenance_validation(
    provenance: dict[str, Any], *, expected_correctness_exit: bool
) -> tuple[bool, list[str]]:
    """Validate loaded-source evidence despite Phase 2C's shared exit status.

    Phase 2C deliberately changes the manifest, correctness summary, and source
    provenance wrapper statuses together when a correctness gate raises.  The
    loaded-source fields remain valid in that case, so accept only that exact
    wrapper transition while continuing to validate the materialized source
    hashes and dependencies themselves.
    """

    failures = []
    status = provenance.get("status")
    error = provenance.get("error")
    error_type = error.get("type") if isinstance(error, dict) else None
    status_valid = bool(
        status == "ok"
        and error is None
        or expected_correctness_exit
        and status == "failed"
        and error_type == "Phase2CCorrectnessFailure"
    )
    if not status_valid:
        failures.append("source_provenance_wrapper_status")
    if provenance.get("loaded_source_matches_repository") is not True:
        failures.append("loaded_source_matches_repository")
    module_hash = provenance.get("actual_loaded_module_sha256")
    custom_hashes = provenance.get("custom_code_sha256")
    if not isinstance(module_hash, str) or len(module_hash) != 64:
        failures.append("actual_loaded_module_sha256")
    if not isinstance(provenance.get("actual_loaded_module_path"), str):
        failures.append("actual_loaded_module_path")
    if (
        not isinstance(custom_hashes, dict)
        or custom_hashes.get("dflash.py") != module_hash
    ):
        failures.append("materialized_dflash_hash_alignment")
    for field in (
        "repository_custom_code_commit",
        "base_model_id",
        "base_model_revision",
        "materialization_id",
    ):
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            failures.append(field)
    if not isinstance(provenance.get("spec_generate"), dict):
        failures.append("spec_generate_provenance")
    dependencies = provenance.get("actual_loaded_dependencies")
    if not isinstance(dependencies, dict) or not dependencies:
        failures.append("actual_loaded_dependencies")
    else:
        for filename, record in dependencies.items():
            if not isinstance(record, dict):
                failures.append(f"dependency_record:{filename}")
                continue
            if record.get("matches_repository") is not True or record.get(
                "actual_loaded_sha256"
            ) != record.get("repository_sha256"):
                failures.append(f"dependency_hash_alignment:{filename}")
    return not failures, failures


def classify_subrun_artifacts(
    output_dir: Path,
    *,
    identity: dict[str, Any],
    return_code: int,
) -> dict[str, Any]:
    """Classify one independent Phase 2C process from persisted evidence."""

    root = output_dir.expanduser().resolve()
    missing = [name for name in EXPECTED_ARTIFACTS if not (root / name).is_file()]
    if missing:
        raise ArtifactError(f"subrun lacks required artifacts: {missing}")
    manifest = read_json(root / "manifest.json")
    correctness = read_json(root / "correctness-summary.json")
    output_divergence = read_json(root / "output-divergence.json")
    provenance = read_json(root / "source-provenance.json")
    sample = _single(read_jsonl(root / "samples.jsonl"), "sample record")
    divergence = _single(
        output_divergence.get("records", []), "output-divergence record"
    )
    rounds = read_jsonl(root / "tree-rounds.jsonl")
    stabilization = read_jsonl(root / "decision-stabilization-rounds.jsonl")
    traversal = read_jsonl(root / "tree-traversal-checks.jsonl")
    gold_logits = read_jsonl(root / "gold-logit-checks.jsonl")
    gold_kv = read_jsonl(root / "gold-kv-checks.jsonl")

    expected = (
        identity["task"],
        identity["sample_id"],
        int(identity["nominal_prompt_length"]),
    )
    actual = (
        sample.get("task"),
        sample.get("sample_id"),
        int(sample.get("nominal_prompt_length", -1)),
    )
    if actual != expected:
        raise ArtifactError(
            f"subrun identity mismatch: expected={expected}, actual={actual}"
        )
    if int(sample.get("node_budget", -1)) != int(identity["node_budget"]):
        raise ArtifactError("sample node budget does not match requested identity")

    reference_tree = _pairwise(
        divergence, "original_dflash_reference", "layerwise_tree_control"
    )
    target_reference = _pairwise(divergence, "target_only", "original_dflash_reference")
    target_tree = _pairwise(divergence, "target_only", "layerwise_tree_control")
    reference_stable = bool(
        divergence.get("reference_repeat_output_ids_equal") is True
        and divergence.get("reference_profile_output_ids_equal") is True
    )
    structural_failure, structural_reasons = _structural_failure(
        gold_logits, gold_kv, traversal
    )
    classification = divergence.get("classification", {})
    gold_failure_class = correctness.get("failure_class") or sample.get(
        "correctness", {}
    ).get("gold_failure_class")
    failure_class = classification.get("failure_class") or gold_failure_class
    actual_output_relation = reference_tree.get("output_ids_equal")
    reference_output_ids = reference_tree.get("left_output_ids")
    tree_output_ids = reference_tree.get("right_output_ids")
    actual_output_relation_reliable = bool(
        isinstance(actual_output_relation, bool)
        and isinstance(reference_output_ids, list)
        and isinstance(tree_output_ids, list)
        and actual_output_relation == (reference_output_ids == tree_output_ids)
    )
    actual_equal = actual_output_relation is True
    gold_evidence = _gold_evidence_assessment(
        correctness=correctness,
        divergence=divergence,
        rounds=rounds,
        stabilization=stabilization,
        traversal=traversal,
        gold_logits=gold_logits,
        gold_kv=gold_kv,
        actual_equal=actual_equal,
        structural_failure=structural_failure,
        failure_class=str(failure_class) if failure_class is not None else None,
    )
    gold_complete = bool(gold_evidence["complete"])
    replay_rows = replay_semantic_rows(identity, gold_logits, gold_kv)
    replay_semantic_rounds = sorted(
        {int(row["round_index"]) for row in replay_rows if row["semantic_mismatch"]}
    )
    diagnostic_numerical_variation_observed = bool(
        correctness.get("diagnostic_numerical_variation_observed", False)
        or any(row["numerical_variation"] for row in replay_rows)
        or (
            replay_rows
            and _numerical_attribution(
                str(failure_class) if failure_class is not None else None
            )
        )
    )
    actual_execution_semantic_failure = bool(
        correctness.get("actual_execution_semantic_failure", not actual_equal)
    )
    replay_semantic_variation_with_actual_consequence = bool(
        correctness.get("replay_semantic_variation_with_actual_consequence", False)
    )
    replay_only_variation_observed = bool(
        actual_equal and replay_rows and diagnostic_numerical_variation_observed
    )
    manifest_error = manifest.get("error")
    manifest_error_type = (
        manifest_error.get("type") if isinstance(manifest_error, dict) else None
    )
    process_is_expected_correctness_exit = bool(
        int(return_code) != 0
        and manifest.get("status") == "failed"
        and manifest_error_type == "Phase2CCorrectnessFailure"
    )
    process_status_contract_passed = bool(
        int(return_code) == 0
        and manifest.get("status") == "ok"
        and manifest_error is None
        or process_is_expected_correctness_exit
    )
    source_provenance_valid, source_provenance_failures = _source_provenance_validation(
        provenance,
        expected_correctness_exit=process_is_expected_correctness_exit,
    )

    if manifest.get("status") == "error":
        outcome = "infrastructure_failure"
    elif not reference_stable:
        outcome = "reference_instability"
    elif structural_failure:
        outcome = "actual_structural_failure"
    elif not actual_output_relation_reliable:
        outcome = "incomplete_correctness_evidence"
    elif not actual_equal:
        outcome = (
            "actual_numerical_decision_divergence"
            if gold_complete
            and classification.get("evidence_complete") is True
            and _numerical_attribution(str(failure_class))
            else "incomplete_correctness_evidence"
        )
    elif replay_only_variation_observed:
        outcome = (
            "diagnostic_only_numerical_variation"
            if gold_complete
            and not actual_execution_semantic_failure
            and not replay_semantic_variation_with_actual_consequence
            and _numerical_attribution(
                str(failure_class) if failure_class is not None else None
            )
            else "incomplete_correctness_evidence"
        )
    elif not gold_complete:
        outcome = "incomplete_correctness_evidence"
    else:
        outcome = "pass_actual_exact"
    if not process_status_contract_passed or (
        int(return_code) != 0
        and outcome
        not in {
            "actual_numerical_decision_divergence",
            "diagnostic_only_numerical_variation",
            "actual_structural_failure",
            "reference_instability",
            "incomplete_correctness_evidence",
        }
    ):
        outcome = "infrastructure_failure"

    selector_records = [
        row["online_selector"] for row in rounds if row.get("online_selector")
    ]
    selected_policies = sorted(
        {
            str(row["selected_policy_config"])
            for row in selector_records
            if row.get("selected_policy_config")
        }
    )
    if not selected_policies:
        selected_policies = [str(sample.get("configuration_id"))]
    actual_nodes = sorted(
        {int(row["actual_node_count"]) for row in rounds if "actual_node_count" in row}
    )
    target_forward_count_added = sum(
        int(row.get("target_forward_count_added", 0)) for row in selector_records
    )
    target_blind = (
        all(
            row.get("decision_made_before_target_verification") is True
            and row.get("target_dependent_runtime_input_used") is False
            for row in selector_records
        )
        if selector_records
        else True
    )
    attention = manifest.get("attention_backend", {})
    experiment_precondition_failures = []
    if not source_provenance_valid:
        experiment_precondition_failures.extend(source_provenance_failures)
    if attention.get("target", {}).get("calls_pytorch_sdpa") is not True:
        experiment_precondition_failures.append("target_attention_not_sdpa")
    if attention.get("draft", {}).get("calls_pytorch_sdpa") is not True:
        experiment_precondition_failures.append("draft_attention_not_sdpa")
    if not target_blind or target_forward_count_added != 0:
        experiment_precondition_failures.append("selector_target_blind_invariant")
    if experiment_precondition_failures:
        outcome = "infrastructure_failure"

    return {
        "schema_version": "1.0",
        **identity,
        "raw_subrun_dir": str(root),
        "subprocess_return_code": int(return_code),
        "subprocess_nonzero_retained_as_diagnostic": bool(
            int(return_code) != 0
            and outcome
            in {
                "actual_numerical_decision_divergence",
                "diagnostic_only_numerical_variation",
                "actual_structural_failure",
                "reference_instability",
            }
        ),
        "outcome_category": outcome,
        "actual_reference_output_sha256": reference_tree.get("left_output_sha256"),
        "actual_tree_output_sha256": reference_tree.get("right_output_sha256"),
        "target_only_output_sha256": target_reference.get("left_output_sha256"),
        "actual_reference_output_ids": reference_tree.get("left_output_ids"),
        "actual_tree_output_ids": reference_tree.get("right_output_ids"),
        "target_only_output_ids": target_reference.get("left_output_ids"),
        "actual_reference_tree_output_ids_equal": actual_equal,
        "actual_output_relation_reliable": actual_output_relation_reliable,
        "target_only_reference_output_ids_equal": target_reference.get(
            "output_ids_equal"
        ),
        "target_only_tree_output_ids_equal": target_tree.get("output_ids_equal"),
        "reference_repeat_stable": divergence.get("reference_repeat_output_ids_equal"),
        "reference_profile_stable": divergence.get(
            "reference_profile_output_ids_equal"
        ),
        "first_divergent_generated_offset": divergence.get(
            "first_divergent_generated_token_offset"
        ),
        "first_divergent_absolute_position": divergence.get(
            "first_divergent_absolute_sequence_position"
        ),
        "first_divergent_tree_round": divergence.get("divergent_tree_round_index"),
        "divergent_token_kind": divergence.get("divergent_token_kind"),
        "reference_token_id": divergence.get("reference_token_id"),
        "reference_token_text": divergence.get("reference_token_text"),
        "tree_token_id": divergence.get("tree_token_id"),
        "tree_token_text": divergence.get("tree_token_text"),
        "divergence_failure_class": failure_class,
        "divergence_evidence_complete": classification.get("evidence_complete"),
        "actual_execution_correctness_passed": correctness.get(
            "actual_execution_correctness_passed"
        ),
        "actual_execution_semantic_failure": actual_execution_semantic_failure,
        "structural_correctness_passed": not structural_failure,
        "structural_failure_reasons": structural_reasons,
        "diagnostic_numerical_variation_observed": (
            diagnostic_numerical_variation_observed
        ),
        "replay_only_variation_observed": replay_only_variation_observed,
        "gold_status": correctness.get("gold_status"),
        "gold_failure_class": gold_failure_class,
        "hard_correctness_failure_reasons": list(
            correctness.get("hard_correctness_failure_reasons") or []
        ),
        "diagnostic_numerical_variation_reasons": list(
            correctness.get("diagnostic_numerical_variation_reasons") or []
        ),
        "replay_semantic_variation_with_actual_consequence": (
            replay_semantic_variation_with_actual_consequence
        ),
        "replay_semantic_mismatch_observed": bool(replay_semantic_rounds),
        "replay_semantic_variation_rounds": replay_semantic_rounds,
        "gold_evidence_complete": gold_complete,
        "gold_comparison_modes_observed": gold_evidence["comparison_modes"],
        "first_divergence_gold_comparison_modes_observed": gold_evidence[
            "first_divergence_comparison_modes"
        ],
        "first_divergence_replay_decision_mismatch_captured": gold_evidence[
            "first_divergence_replay_decision_mismatch_captured"
        ],
        "structural_traversal_complete_through_first_divergence": gold_evidence[
            "structural_traversal_complete_through_first_divergence"
        ],
        "first_replay_variation_round": gold_evidence["first_replay_variation_round"],
        "first_replay_variation_gold_comparison_modes_observed": gold_evidence[
            "first_replay_variation_comparison_modes"
        ],
        "first_replay_variation_evidence_captured": gold_evidence[
            "first_replay_variation_evidence_captured"
        ],
        "structural_traversal_complete_through_first_replay_variation": (
            gold_evidence[
                "structural_traversal_complete_through_first_replay_variation"
            ]
        ),
        "downstream_gold_checks_skipped_after_confirmed_divergence": gold_evidence[
            "downstream_gold_checks_skipped_after_confirmed_divergence"
        ],
        "downstream_gold_checks_skipped_after_confirmed_replay_variation": (
            gold_evidence[
                "downstream_gold_checks_skipped_after_confirmed_replay_variation"
            ]
        ),
        "downstream_gold_skipped_checks": gold_evidence[
            "downstream_gold_skipped_checks"
        ],
        "downstream_gold_skip_reasons": gold_evidence["downstream_gold_skip_reasons"],
        "genuinely_incomplete_correctness_evidence": gold_evidence[
            "genuinely_incomplete_correctness_evidence"
        ],
        "gold_incompleteness_reasons": gold_evidence["gold_incompleteness_reasons"],
        "selected_frozen_policy_configs": selected_policies,
        "selected_frozen_policy_config": (
            selected_policies[0] if len(selected_policies) == 1 else None
        ),
        "actual_node_counts": actual_nodes,
        "actual_node_count": actual_nodes[0] if len(actual_nodes) == 1 else None,
        "selector_target_blind": target_blind,
        "target_forward_count_added": target_forward_count_added,
        "tree_traversal_status": divergence.get("tree_traversal_status"),
        "target_attention_calls_pytorch_sdpa": attention.get("target", {}).get(
            "calls_pytorch_sdpa"
        ),
        "draft_attention_calls_pytorch_sdpa": attention.get("draft", {}).get(
            "calls_pytorch_sdpa"
        ),
        "source_provenance_status": provenance.get("status"),
        "source_provenance_validation_passed": source_provenance_valid,
        "source_provenance_validation_failures": source_provenance_failures,
        "source_provenance_wrapper_failure_is_expected_correctness": bool(
            provenance.get("status") == "failed"
            and process_is_expected_correctness_exit
        ),
        "phase2c_manifest_status": manifest.get("status"),
        "phase2c_manifest_error_type": manifest_error_type,
        "process_is_expected_correctness_exit": (process_is_expected_correctness_exit),
        "process_status_contract_passed": process_status_contract_passed,
        "experiment_precondition_failures": experiment_precondition_failures,
        "_rounds": rounds,
        "_stabilization_rounds": stabilization,
        "_traversal_checks": traversal,
        "_replay_rows": replay_rows,
        "_replay_semantic_rounds": replay_semantic_rounds,
    }


def infrastructure_failure_observation(
    identity: dict[str, Any],
    *,
    output_dir: Path,
    return_code: int,
    error: Exception,
) -> dict[str, Any]:
    """Create a persisted failure record when artifacts cannot be parsed."""

    return {
        "schema_version": "1.0",
        **identity,
        "raw_subrun_dir": str(output_dir.expanduser().resolve()),
        "subprocess_return_code": int(return_code),
        "subprocess_nonzero_retained_as_diagnostic": False,
        "outcome_category": "infrastructure_failure",
        "infrastructure_error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def public_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Remove parser-only joined raw rows before writing the subrun index."""

    return {key: value for key, value in observation.items() if not key.startswith("_")}


def build_divergence_event(
    observation: dict[str, Any],
) -> dict[str, Any] | None:
    """Build one canonical event for the first actual output divergence."""

    if observation.get("actual_reference_tree_output_ids_equal") is not False:
        return None
    divergent_round = observation.get("first_divergent_tree_round")
    relevant_replay = [
        public_observation(row)
        for row in observation.get("_replay_rows", [])
        if row.get("round_index") == divergent_round
    ]
    relevant_margin = next(
        (
            row
            for row in observation.get("_stabilization_rounds", [])
            if row.get("round_index") == divergent_round
            and int(row.get("profiling_repeat", 0)) == 0
        ),
        None,
    )
    return {
        "schema_version": "1.0",
        **{
            field: observation.get(field)
            for field in (
                "experiment",
                "task",
                "sample_id",
                "source_sample_id",
                "nominal_prompt_length",
                "selector_mode",
                "node_budget",
                "repeat_index",
                "physical_case_id",
                "configuration_case_id",
                "independent_repeat_observation_id",
                "selected_frozen_policy_configs",
                "selected_frozen_policy_config",
                "actual_node_counts",
                "actual_node_count",
                "first_divergent_generated_offset",
                "first_divergent_absolute_position",
                "first_divergent_tree_round",
                "divergent_token_kind",
                "reference_token_id",
                "reference_token_text",
                "tree_token_id",
                "tree_token_text",
                "actual_reference_output_sha256",
                "actual_tree_output_sha256",
                "target_only_output_sha256",
                "actual_reference_output_ids",
                "actual_tree_output_ids",
                "target_only_output_ids",
                "divergence_failure_class",
                "divergence_evidence_complete",
                "reference_repeat_stable",
                "reference_profile_stable",
                "target_only_reference_output_ids_equal",
                "target_only_tree_output_ids_equal",
                "tree_traversal_status",
                "structural_correctness_passed",
            )
        },
        "replay_evidence": relevant_replay,
        "minimum_verifier_margin": (
            relevant_margin.get("minimum_path_margin")
            if relevant_margin is not None
            else None
        ),
        "first_corrected_native_margin": (
            relevant_margin.get("first_corrected_native_margin")
            if relevant_margin is not None
            else None
        ),
        "native_accepted_depth": (
            relevant_margin.get("native_accepted_depth")
            if relevant_margin is not None
            else None
        ),
        "canonical_accepted_depth": (
            relevant_margin.get("canonical_accepted_depth")
            if relevant_margin is not None
            else None
        ),
        "native_canonical_decision_equal": (
            relevant_margin.get("native_canonical_decision_equal")
            if relevant_margin is not None
            else None
        ),
    }
