#!/usr/bin/env python3
"""Validate a target-blind Phase 2D.2A-H2.0 cohort freeze."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

if __package__:
    from scripts.freeze_phase2d2a_h2_cohort import (
        ARTIFACTS,
        BASE_FAMILY,
        CANDIDATE_LABEL_AUTHORITY,
        CANDIDATE_LABEL_SOURCE,
        FAMILIES,
        FROZEN_REFINEMENT_FAMILIES,
        INDEPENDENT_TASKS,
        INVENTORY_LENGTHS,
        KNOWN_TASK,
        LONG_CONTEXT_ORDER,
        PRIMARY_NODE_BUDGET,
        SCHEMAS,
        SECONDARY_NODE_BUDGET,
        SHORT_CONTEXT,
        TASK_SAMPLE_COUNTS,
        TASKS,
        VERIFICATION_MODE,
        _joint_eligible,
        _sha256_json,
        _task,
        choose_common_long_context,
        choose_govreport_long_context,
        deterministic_select,
        exclusion_index,
        exclusion_matches,
        validate_exclusion_preflight,
    )
else:
    from freeze_phase2d2a_h2_cohort import (
        ARTIFACTS,
        BASE_FAMILY,
        CANDIDATE_LABEL_AUTHORITY,
        CANDIDATE_LABEL_SOURCE,
        FAMILIES,
        FROZEN_REFINEMENT_FAMILIES,
        INDEPENDENT_TASKS,
        INVENTORY_LENGTHS,
        KNOWN_TASK,
        LONG_CONTEXT_ORDER,
        PRIMARY_NODE_BUDGET,
        SCHEMAS,
        SECONDARY_NODE_BUDGET,
        SHORT_CONTEXT,
        TASK_SAMPLE_COUNTS,
        TASKS,
        VERIFICATION_MODE,
        _joint_eligible,
        _sha256_json,
        _task,
        choose_common_long_context,
        choose_govreport_long_context,
        deterministic_select,
        exclusion_index,
        exclusion_matches,
        validate_exclusion_preflight,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = REPO_ROOT / "docs" / "schemas"


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc


def _validate(payload: Any, schema_name: str) -> None:
    schema = _json(SCHEMA_ROOT / schema_name)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)


def _unique(rows: list[tuple[Any, ...]], expected_count: int, label: str) -> None:
    if len(rows) != expected_count or any(
        count != 1 for count in Counter(rows).values()
    ):
        raise ValueError(f"H2.0 {label} is not an exact unique inventory")


def _check_inventory(inventory: dict[str, Any]) -> None:
    if {row["task"] for row in inventory["tasks"]} != set(TASKS):
        raise ValueError("H2.0 inventory does not contain exactly three tasks")
    for task in inventory["tasks"]:
        if task["requested_physical_sample_count"] != TASK_SAMPLE_COUNTS[task["task"]]:
            raise ValueError("H2.0 requested task sample count changed")
        if task["source_record_count"] != (
            task["usable_record_count"] + task["unusable_record_count"]
        ):
            raise ValueError("H2.0 source/usable record accounting is inconsistent")
        contexts = task["contexts"]
        if [row["nominal_prompt_length"] for row in contexts] != list(
            INVENTORY_LENGTHS
        ):
            raise ValueError("H2.0 inventory context order changed")
        for context in contexts:
            eligible = context["eligible_samples"]
            if context["eligible_sample_count"] != len(eligible):
                raise ValueError("H2.0 eligible sample count is inconsistent")
            if context["eligible_before_exclusion_count"] != (
                context["eligible_sample_count"]
                + context["excluded_previous_sample_count"]
            ):
                raise ValueError("H2.0 pre/post exclusion accounting is inconsistent")
            if context["eligible_sample_ids"] != [row["sample_id"] for row in eligible]:
                raise ValueError("H2.0 eligible sample IDs are inconsistent")
            if any(
                row["actual_prompt_length"] > context["nominal_prompt_length"]
                for row in eligible
            ):
                raise ValueError("H2.0 inventory prefix exceeds nominal length")


def _check_exclusions(exclusions: dict[str, Any]) -> None:
    validate_exclusion_preflight(exclusions)
    if exclusions["scanned_artifact_count"] != len(exclusions["scanned_artifacts"]):
        raise ValueError("H2.0 scanned artifact count is inconsistent")
    if exclusions["excluded_identity_count"] != len(exclusions["excluded_identities"]):
        raise ValueError("H2.0 exclusion identity count is inconsistent")
    if (
        exclusions["selection_or_exclusion_uses_model_outputs"]
        or exclusions["forbidden_outcome_fields_read"]
    ):
        raise ValueError("H2.0 exclusion is not target blind")


def _context_for_checker(task_inventory: dict[str, Any], length: int) -> dict[str, Any]:
    return next(
        row
        for row in task_inventory["contexts"]
        if row["nominal_prompt_length"] == length
    )


def _check_cohort(
    inventory: dict[str, Any], exclusions: dict[str, Any], cohort: dict[str, Any]
) -> None:
    common_long, common_audit = choose_common_long_context(inventory)
    gov_long, gov_audit = choose_govreport_long_context(inventory)
    if common_long != cohort["common_independent_task_long_context_length"]:
        raise ValueError("H2.0 common long context is not the first eligible fallback")
    if gov_long != cohort["govreport_replication_long_context_length"]:
        raise ValueError(
            "H2.0 GovReport long context is not the first eligible fallback"
        )
    if common_audit != cohort["common_long_context_selection_audit"]:
        raise ValueError("H2.0 common long-context audit is inconsistent")
    if gov_audit != cohort["govreport_long_context_selection_audit"]:
        raise ValueError("H2.0 GovReport long-context audit is inconsistent")
    if cohort["common_long_context_selection_order"] != list(LONG_CONTEXT_ORDER):
        raise ValueError("H2.0 common long-context order changed")
    if (
        cohort["independent_task_count"] != 2
        or tuple(cohort["independent_tasks"]) != INDEPENDENT_TASKS
    ):
        raise ValueError("H2.0 independent task count changed")

    excluded = exclusion_index(exclusions)
    if {row["task"] for row in cohort["tasks"]} != set(TASKS):
        raise ValueError("H2.0 frozen cohort does not contain exactly three tasks")
    for task_row in cohort["tasks"]:
        task = task_row["task"]
        expected_long = gov_long if task == KNOWN_TASK else common_long
        if task_row["paired_context_lengths"] != [SHORT_CONTEXT, expected_long]:
            raise ValueError("H2.0 task does not use its frozen paired contexts")
        if task_row["requested_physical_sample_count"] != TASK_SAMPLE_COUNTS[task]:
            raise ValueError("H2.0 requested task cohort count changed")
        if task_row["frozen_physical_sample_count"] != TASK_SAMPLE_COUNTS[task]:
            raise ValueError("H2.0 silently reduced a task cohort")
        candidates = _joint_eligible(
            _task(inventory, task), SHORT_CONTEXT, expected_long
        )
        expected = deterministic_select(candidates, count=TASK_SAMPLE_COUNTS[task])
        selected = task_row["selected_physical_samples"]
        if [row["sample_id"] for row in selected] != [
            row["sample_id"] for row in expected
        ]:
            raise ValueError("H2.0 selected cohort does not follow frozen hash order")
        for rank, sample in enumerate(selected, 1):
            if sample["selection_rank"] != rank:
                raise ValueError("H2.0 sample selection rank is inconsistent")
            if set(map(int, sample["contexts"])) != {SHORT_CONTEXT, expected_long}:
                raise ValueError(
                    "H2.0 selected sample differs across short/long contexts"
                )
            if not sample["short_context_is_nested_prefix_of_long_context"]:
                raise ValueError("H2.0 selected short/long context is not nested")
            if (
                sample["contexts"][str(SHORT_CONTEXT)]["context_prefix_characters"]
                > sample["contexts"][str(expected_long)]["context_prefix_characters"]
            ):
                raise ValueError(
                    "H2.0 selected prefix character lengths are not nested"
                )
            for length in (SHORT_CONTEXT, expected_long):
                expected_context = next(
                    row
                    for row in _context_for_checker(_task(inventory, task), length)[
                        "eligible_samples"
                    ]
                    if row["sample_id"] == sample["sample_id"]
                )
                if sample["contexts"][str(length)] != expected_context:
                    raise ValueError(
                        "H2.0 selected context metadata differs from inventory"
                    )
            probe = {
                "longbench_task": task,
                "sample_id": sample["sample_id"],
                "source_sample_id": sample["source_sample_id"],
                "source_text_sha256": sample["source_text_sha256"],
            }
            if exclusion_matches(probe, excluded):
                raise ValueError("H2.0 selected a previously used physical sample")


def _expected_matrix_keys(cohort: dict[str, Any]) -> set[tuple[Any, ...]]:
    return {
        (task["task"], sample["sample_id"], context, family)
        for task in cohort["tasks"]
        for sample in task["selected_physical_samples"]
        for context in task["paired_context_lengths"]
        for family in FAMILIES
    }


def _check_matrix(
    matrix: dict[str, Any], *, cohort: dict[str, Any], budget: int, role: str
) -> None:
    if matrix["node_budget"] != budget or matrix["matrix_role"] != role:
        raise ValueError("H2.0 matrix role/budget changed")
    expected_regime = (
        "same_actual_verification_work"
        if budget == PRIMARY_NODE_BUDGET
        else "budget_cap_expansion"
    )
    if matrix["actual_work_regime"] != expected_regime:
        raise ValueError("H2.0 matrix work regime changed")
    if tuple(matrix["frozen_configuration_families"]) != FAMILIES:
        raise ValueError("H2.0 matrix selected or removed a family")
    if tuple(matrix["frozen_refinement_families"]) != FROZEN_REFINEMENT_FAMILIES:
        raise ValueError("H2.0 matrix selected or removed a refinement family")
    expected = _expected_matrix_keys(cohort)
    observed = [
        (
            row["task"],
            row["sample_id"],
            row["nominal_prompt_length"],
            row["configuration_family"],
        )
        for row in matrix["rows"]
    ]
    _unique(observed, len(expected), "matrix")
    if set(observed) != expected or matrix["row_count"] != len(expected):
        raise ValueError("H2.0 matrix is not the exact frozen cohort product")
    for row in matrix["rows"]:
        if (
            row["node_budget"] != budget
            or row["verification_mode"] != VERIFICATION_MODE
            or row["candidate_label_authority"] != CANDIDATE_LABEL_AUTHORITY
            or row["candidate_label_source"] != CANDIDATE_LABEL_SOURCE
            or row["q1_replay_used_as_target_label_source"]
            or row["execution_authorized"]
        ):
            raise ValueError("H2.0 matrix violates a frozen execution setting")
        cohort_task = next(
            item for item in cohort["tasks"] if item["task"] == row["task"]
        )
        cohort_sample = next(
            item
            for item in cohort_task["selected_physical_samples"]
            if item["sample_id"] == row["sample_id"]
        )
        cohort_context = cohort_sample["contexts"][str(row["nominal_prompt_length"])]
        if (
            row["source_sample_id"] != cohort_sample["source_sample_id"]
            or row["source_text_sha256"] != cohort_sample["source_text_sha256"]
            or row["selection_hash"] != cohort_sample["selection_hash"]
            or row["actual_prompt_length"] != cohort_context["actual_prompt_length"]
            or row["context_prefix_sha256"] != cohort_context["context_prefix_sha256"]
        ):
            raise ValueError("H2.0 matrix identity/prefix differs from frozen cohort")
        family = row["configuration_family"]
        expected_mode = BASE_FAMILY if family == BASE_FAMILY else "refined_tree"
        expected_refinement = None if family == BASE_FAMILY else family
        if (
            row["candidate_merge_mode"] != expected_mode
            or row["frozen_refinement_configuration_id"] != expected_refinement
        ):
            raise ValueError("H2.0 family configuration mapping is inconsistent")


def _check_rules(rules: dict[str, Any]) -> None:
    if tuple(rules["independent_tasks"]) != INDEPENDENT_TASKS:
        raise ValueError("H2.0 aggregation changed independent tasks")
    if rules["combined_sample_weighted_global_pool_can_replace_task_level_rule"]:
        raise ValueError("H2.0 permits a global pool to erase task-level evidence")
    stable = rules["stable_cross_task_same_work_signal_rule"]
    if (
        stable["required_task_context_stratum_count_per_family"] != 4
        or not stable["coverage_gain_strictly_positive_in_every_stratum"]
        or not stable[
            "accepted_tokens_per_verified_node_gain_strictly_positive_in_every_stratum"
        ]
        or not stable["all_contributing_cells_correctness_valid"]
    ):
        raise ValueError("H2.0 cross-task stability rule was weakened")
    if rules["speedup_claim_authorized"]:
        raise ValueError("H2.0 incorrectly authorizes a speedup claim")


def _check_hashes(
    manifest: dict[str, Any],
    *,
    inventory: dict[str, Any],
    exclusions: dict[str, Any],
    cohort: dict[str, Any],
    primary: dict[str, Any],
    secondary: dict[str, Any],
    rules: dict[str, Any],
) -> None:
    payload = {
        "frozen_cohort": cohort,
        "primary_matrix": primary,
        "secondary_matrix": secondary,
        "aggregation_rules": rules,
    }
    components = {name: _sha256_json(value) for name, value in payload.items()}
    if manifest["frozen_content_component_sha256"] != components:
        raise ValueError("H2.0 component content hash mismatch")
    if manifest["freeze_content_sha256"] != _sha256_json(payload):
        raise ValueError("H2.0 freeze content hash mismatch")
    if not manifest["h2_execution_must_require_exact_freeze_content_sha256"]:
        raise ValueError("H2.0 does not require the exact freeze hash for execution")
    if manifest["inventory_sha256"] != _sha256_json(inventory):
        raise ValueError("H2.0 inventory hash mismatch")
    if manifest["exclusion_provenance_sha256"] != _sha256_json(exclusions):
        raise ValueError("H2.0 exclusion hash mismatch")


def check_h2_freeze(output_dir: Path) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    missing = [name for name in ARTIFACTS if not (output / name).is_file()]
    if missing:
        raise ValueError(f"missing Phase 2D.2A-H2.0 artifacts: {missing}")
    payloads = {name: _json(output / name) for name in SCHEMAS}
    for name, schema in SCHEMAS.items():
        _validate(payloads[name], schema)
    inventory = payloads["phase2d2a-h2-heldout-inventory.json"]
    exclusions = payloads["phase2d2a-h2-exclusion-provenance.json"]
    cohort = payloads["phase2d2a-h2-frozen-cohort.json"]
    primary = payloads["phase2d2a-h2-primary-matrix.json"]
    secondary = payloads["phase2d2a-h2-secondary-matrix.json"]
    rules = payloads["phase2d2a-h2-aggregation-rules.json"]
    manifest = payloads["phase2d2a-h2-freeze-manifest.json"]

    with (output / "phase2d2a-h2-heldout-inventory.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != len(TASKS) * len(INVENTORY_LENGTHS):
        raise ValueError("H2.0 inventory CSV does not contain every task/context row")

    if manifest["status"] != "frozen_not_executed" or manifest["error"] is not None:
        raise ValueError(f"H2.0 cohort was not frozen: status={manifest['status']!r}")

    _check_inventory(inventory)
    _check_exclusions(exclusions)
    _check_cohort(inventory, exclusions, cohort)
    _check_matrix(
        primary,
        cohort=cohort,
        budget=PRIMARY_NODE_BUDGET,
        role="primary_same_actual_verification_work",
    )
    _check_matrix(
        secondary,
        cohort=cohort,
        budget=SECONDARY_NODE_BUDGET,
        role="secondary_budget_cap_expansion_ablation",
    )
    primary_identity = {
        (
            row["task"],
            row["sample_id"],
            row["nominal_prompt_length"],
            row["configuration_family"],
        )
        for row in primary["rows"]
    }
    secondary_identity = {
        (
            row["task"],
            row["sample_id"],
            row["nominal_prompt_length"],
            row["configuration_family"],
        )
        for row in secondary["rows"]
    }
    if primary_identity != secondary_identity:
        raise ValueError("H2.0 primary and secondary matrices use different cohorts")
    _check_rules(rules)
    _check_hashes(
        manifest,
        inventory=inventory,
        exclusions=exclusions,
        cohort=cohort,
        primary=primary,
        secondary=secondary,
        rules=rules,
    )
    if any(
        manifest[field]
        for field in (
            "model_forward_executed",
            "target_model_loaded",
            "draft_model_loaded",
            "cuda_required",
            "h2_primary_execution_authorized",
            "h2_secondary_execution_authorized",
        )
    ):
        raise ValueError("H2.0 freeze executed or authorized H2")
    if manifest["h11_prerequisite_provenance"]["checker_status"] != "accepted":
        raise ValueError("H2.0 H1.1 prerequisite was not accepted")
    return {
        "status": "accepted",
        "phase": "2D.2A-H2.0",
        "freeze_status": manifest["status"],
        "common_long_context": cohort["common_independent_task_long_context_length"],
        "govreport_replication_long_context": cohort[
            "govreport_replication_long_context_length"
        ],
        "frozen_physical_sample_count": sum(
            row["frozen_physical_sample_count"] for row in cohort["tasks"]
        ),
        "primary_matrix_row_count": primary["row_count"],
        "secondary_matrix_row_count": secondary["row_count"],
        "freeze_content_sha256": manifest["freeze_content_sha256"],
        "model_forward_executed": False,
        "h2_execution_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = check_h2_freeze(args.output_dir)
    except Exception as exc:  # noqa: BLE001 - clean acceptance failure.
        print(f"REJECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
