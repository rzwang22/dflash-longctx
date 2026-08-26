#!/usr/bin/env python3
"""Reconcile accepted H2.1 evidence into the frozen Phase 2D.2A route."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_phase2d2a_h21_acceptance import check_h21_output
from scripts.phase2d2a_h21_common import (
    AGGREGATE_JSON_ARTIFACTS,
    FROZEN_REFINEMENT_FAMILIES,
    H21_ARTIFACTS,
    INDEPENDENT_TASKS,
    REQUIRED_FREEZE_SHA256,
    read_json,
    validate_h21_freeze,
    write_csv,
    write_json,
)

PHASE = "2D.2A-H2.1.1"
SCHEMA_VERSION = "1.0"
AUDIT_NAME = "phase2d2a-h211-final-route-audit.json"
AUDIT_CSV_NAME = "phase2d2a-h211-final-route-audit.csv"
MANIFEST_NAME = "phase2d2a-h211-route-manifest.json"
SCHEMA_NAME = "phase2d2a-h211-final-route.schema.json"
FROZEN_JOINT_POSITIVE_COUNTS = dict(
    zip(FROZEN_REFINEMENT_FAMILIES, (3, 3, 2), strict=True)
)
FROZEN_ALL_STRATA_EFFICIENCY_POSITIVE_FAMILIES = tuple(FROZEN_REFINEMENT_FAMILIES[:2])


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _validate_schema(payload: dict[str, Any]) -> None:
    schema = json.loads(
        (REPO_ROOT / "docs" / "schemas" / SCHEMA_NAME).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)


def _h21_artifact_hashes(root: Path) -> dict[str, str]:
    missing = [name for name in H21_ARTIFACTS if not (root / name).is_file()]
    if missing:
        raise ValueError(f"accepted H2.1 root lacks artifacts: {missing}")
    return {name: _sha256_file(root / name) for name in H21_ARTIFACTS}


def _sign(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _sum_sign_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row["sample_heterogeneity"][field])
    return dict(sorted(counts.items()))


def derive_final_route(
    *,
    freeze_content_sha256: str,
    h21_acceptance: dict[str, Any],
    primary: dict[str, Any],
    family_summary: dict[str, Any],
    task_context_summary: dict[str, Any],
    mechanism_summary: dict[str, Any],
    cost_summary: dict[str, Any],
) -> dict[str, Any]:
    if h21_acceptance.get("status") != "accepted":
        raise ValueError("H2.1 acceptance anchor is not accepted")
    expected_counts = {
        "cell_count": 112,
        "valid_cell_count": 112,
        "correctness_invalid_cell_count": 0,
        "infrastructure_failed_cell_count": 0,
        "pair_count": 84,
    }
    for field, expected in expected_counts.items():
        if h21_acceptance.get(field) != expected:
            raise ValueError(f"accepted H2.1 {field} differs from {expected}")
    if not primary.get("execution_complete"):
        raise ValueError("H2.1 primary execution is incomplete")
    if primary.get("stable_cross_task_same_work_signal_exists") is not False:
        raise ValueError("H2.1 primary stable-signal conclusion changed")
    if primary.get("winner_selected") or primary.get("final_phase2d2a_branch_entered"):
        raise ValueError("H2.1 unexpectedly selected a winner or branch")

    family_by_id = {row["refined_family"]: row for row in family_summary["rows"]}
    if set(family_by_id) != set(FROZEN_REFINEMENT_FAMILIES):
        raise ValueError("H2.1 frozen refinement family inventory changed")
    task_rows = task_context_summary["rows"]
    mechanism_rows = mechanism_summary["rows"]
    cost_rows = cost_summary["rows"]
    family_rows = []
    for family in FROZEN_REFINEMENT_FAMILIES:
        formal = family_by_id[family]
        if formal["cross_task_classification"] != ("mixed_cross_task_same_work_signal"):
            raise ValueError(f"{family}: H2.1 primary family classification changed")
        independent = [
            row
            for row in task_rows
            if row["refined_family"] == family and row["task"] in INDEPENDENT_TASKS
        ]
        if len(independent) != 4:
            raise ValueError(f"{family}: expected four independent task/context strata")
        accepted_signs = Counter(
            _sign(row["accepted_per_verified"]["absolute_gain"]) for row in independent
        )
        coverage_signs = Counter(
            _sign(row["candidate_coverage"]["coverage_absolute_gain"])
            for row in independent
        )
        joint_count = sum(
            row["stratum_status"] == "joint_positive" for row in independent
        )
        if joint_count != formal["joint_positive_independent_stratum_count"]:
            raise ValueError(f"{family}: joint-positive stratum accounting differs")
        if joint_count != FROZEN_JOINT_POSITIVE_COUNTS[family]:
            raise ValueError(
                f"{family}: derived joint-positive count no longer reproduces "
                "the accepted frozen H2.1 result"
            )
        base_accepted = sum(
            int(row["accepted_per_verified"]["base_accepted_draft_token_count"])
            for row in independent
        )
        refined_accepted = sum(
            int(row["accepted_per_verified"]["refined_accepted_draft_token_count"])
            for row in independent
        )
        base_verified = sum(
            int(row["accepted_per_verified"]["base_actual_verified_node_count"])
            for row in independent
        )
        refined_verified = sum(
            int(row["accepted_per_verified"]["refined_actual_verified_node_count"])
            for row in independent
        )
        base_pooled_rate = base_accepted / base_verified if base_verified else None
        refined_pooled_rate = (
            refined_accepted / refined_verified if refined_verified else None
        )
        mechanisms = [
            row
            for row in mechanism_rows
            if row["refined_family"] == family and row["task"] in INDEPENDENT_TASKS
        ]
        all_mechanisms = [
            row for row in mechanism_rows if row["refined_family"] == family
        ]
        recovery_count = sum(
            int(row["refined_only_target_recovery_count"]) for row in all_mechanisms
        )
        labelled_count = sum(
            int(row["labelled_reference_position_count"]) for row in all_mechanisms
        )
        costs = [row for row in cost_rows if row["refined_family"] == family]
        hierarchy = bool(
            costs
            and all(
                row["h1_cost_hierarchy_generalizes_for_all_comparable_samples"]
                for row in costs
            )
        )
        family_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "refined_family": family,
                "formal_primary_evidence": {
                    "cross_task_classification": formal["cross_task_classification"],
                    "joint_positive_independent_stratum_count": joint_count,
                    "required_independent_stratum_count": 4,
                    "required_strata": formal["required_strata"],
                },
                "secondary_efficiency_observation": {
                    "accepted_per_verified_positive_independent_stratum_count": (
                        accepted_signs["positive"]
                    ),
                    "required_independent_stratum_count": 4,
                    "accepted_per_verified_sign_counts": dict(
                        sorted(accepted_signs.items())
                    ),
                    "coverage_sign_counts": dict(sorted(coverage_signs.items())),
                    "sample_accepted_per_verified_gain_sign_counts": (
                        _sum_sign_counts(
                            independent, "accepted_per_verified_gain_sign_counts"
                        )
                    ),
                    "sample_coverage_gain_sign_counts": _sum_sign_counts(
                        independent, "coverage_gain_sign_counts"
                    ),
                    "sample_joint_signal_counts": _sum_sign_counts(
                        independent, "joint_signal_counts"
                    ),
                    "positive_efficiency_in_all_independent_strata": (
                        accepted_signs["positive"] == 4
                    ),
                    "pooled_across_independent_strata": {
                        "aggregation": (
                            "sum_accepted_draft_tokens_over_sum_actual_verified_nodes"
                        ),
                        "base_accepted_draft_token_count": base_accepted,
                        "refined_accepted_draft_token_count": refined_accepted,
                        "base_actual_verified_node_count": base_verified,
                        "refined_actual_verified_node_count": refined_verified,
                        "base_rate": base_pooled_rate,
                        "refined_rate": refined_pooled_rate,
                        "absolute_gain": (
                            refined_pooled_rate - base_pooled_rate
                            if base_pooled_rate is not None
                            and refined_pooled_rate is not None
                            else None
                        ),
                    },
                },
                "mechanism_and_cost_interpretation": {
                    "independent_task_mechanism_signal_distribution": dict(
                        sorted(
                            Counter(
                                row["mechanism_signal"] for row in mechanisms
                            ).items()
                        )
                    ),
                    "all_task_mechanism_signal_distribution": dict(
                        sorted(
                            Counter(
                                row["mechanism_signal"] for row in all_mechanisms
                            ).items()
                        )
                    ),
                    "refined_only_target_recovery_count": recovery_count,
                    "refined_only_target_recovery_rate": (
                        recovery_count / labelled_count if labelled_count else None
                    ),
                    "govreport_replication_classification": formal[
                        "same_task_replication_classification"
                    ],
                    "verifier_cost_greater_than_dependency_cost_for_all_comparable_samples": hierarchy,
                    "comparable_cost_sample_count": sum(
                        int(row["comparable_sample_count"]) for row in costs
                    ),
                },
            }
        )
    observed_counts = {
        row["refined_family"]: row["formal_primary_evidence"][
            "joint_positive_independent_stratum_count"
        ]
        for row in family_rows
    }
    all_efficiency_positive = [
        row["refined_family"]
        for row in family_rows
        if row["secondary_efficiency_observation"][
            "positive_efficiency_in_all_independent_strata"
        ]
    ]
    if tuple(all_efficiency_positive) != (
        FROZEN_ALL_STRATA_EFFICIENCY_POSITIVE_FAMILIES
    ):
        raise ValueError(
            "derived all-strata accepted/verified positives no longer reproduce "
            "the accepted frozen H2.1 secondary observation"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "artifact_type": "final_route_audit",
        "status": "complete",
        "freeze_content_sha256": freeze_content_sha256,
        "model_forward_executed": False,
        "h21_primary_accepted": True,
        "evidence_levels_kept_separate": True,
        "formal_primary_evidence": {
            "stable_cross_task_same_work_signal_exists": False,
            "family_cross_task_classifications": primary[
                "family_cross_task_classifications"
            ],
            "joint_positive_independent_stratum_counts": observed_counts,
            "primary_dependency_refinement_status": (
                "no_stable_cross_task_joint_signal"
            ),
        },
        "secondary_efficiency_observation": {
            "positive_efficiency_observation_preserved": bool(all_efficiency_positive),
            "families_positive_in_all_four_independent_strata": (
                all_efficiency_positive
            ),
            "primary_criterion_rewritten": False,
        },
        "mechanism_and_cost_interpretation": {
            "dependency_information_declared_useless": False,
            "new_candidate_availability_declared_dominant": False,
            "verifier_cost_greater_than_dependency_cost_for_every_family": all(
                row["mechanism_and_cost_interpretation"][
                    "verifier_cost_greater_than_dependency_cost_for_all_comparable_samples"
                ]
                for row in family_rows
            ),
            "speedup_claimed": False,
        },
        "route": {
            "proceed_to_phase2d2b_lightweight_adapter": False,
            "select_refinement_winner": False,
            "increase_dependency_model_complexity": False,
            "h22_secondary_characterization_allowed": True,
            "h22_can_override_h21_primary": False,
            "recommended_next_primary_design_focus": (
                "budget_constrained_candidate_allocation"
            ),
            "recommended_next_primary_design_focus_evidence_role": (
                "project_route_recommendation_not_experimentally_proven_conclusion"
            ),
            "later_system_optimization_direction": (
                "sparse_kv_and_long_context_traffic_reduction"
            ),
            "later_system_optimization_direction_evidence_role": (
                "project_route_recommendation_not_h21_scientific_conclusion"
            ),
            "enter_phase2d3": False,
        },
        "families": family_rows,
    }


def build_h211_artifacts(
    *,
    freeze_dir: Path,
    freeze_content_sha256: str,
    h21_output_root: Path,
) -> dict[str, dict[str, Any]]:
    freeze_dir = freeze_dir.expanduser().resolve()
    h21_output_root = h21_output_root.expanduser().resolve()
    bundle = validate_h21_freeze(
        freeze_dir,
        freeze_content_sha256,
        required_freeze_sha256=REQUIRED_FREEZE_SHA256,
    )
    accepted = check_h21_output(
        h21_output_root,
        freeze_dir,
        freeze_content_sha256,
        required_freeze_sha256=REQUIRED_FREEZE_SHA256,
    )
    evidence = {
        name: read_json(h21_output_root / name) for name in AGGREGATE_JSON_ARTIFACTS
    }
    audit = derive_final_route(
        freeze_content_sha256=freeze_content_sha256,
        h21_acceptance=accepted,
        primary=evidence["phase2d2a-h21-primary-conclusion.json"],
        family_summary=evidence["phase2d2a-h21-family-summary.json"],
        task_context_summary=evidence["phase2d2a-h21-task-context-summary.json"],
        mechanism_summary=evidence["phase2d2a-h21-mechanism-summary.json"],
        cost_summary=evidence["phase2d2a-h21-cost-summary.json"],
    )
    h21_hashes = _h21_artifact_hashes(h21_output_root)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "artifact_type": "route_manifest",
        "status": "complete",
        "freeze_content_sha256": freeze_content_sha256,
        "freeze_directory": str(freeze_dir),
        "freeze_artifact_file_sha256": bundle["freeze_artifact_file_sha256"],
        "h21_output_root": str(h21_output_root),
        "h21_artifact_file_sha256": h21_hashes,
        "h21_anchor_binding_sha256": _sha256_json(h21_hashes),
        "h21_acceptance": accepted,
        "repository_commit": _git_value("rev-parse", "HEAD"),
        "branch": _git_value("branch", "--show-current"),
        "model_forward_executed": False,
        "h20_artifacts_mutated": False,
        "h21_artifacts_mutated": False,
        "artifacts": [AUDIT_NAME, AUDIT_CSV_NAME, MANIFEST_NAME],
    }
    audit["h21_anchor_binding_sha256"] = manifest["h21_anchor_binding_sha256"]
    _validate_schema(audit)
    _validate_schema(manifest)
    return {"audit": audit, "manifest": manifest}


def audit_csv_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in audit["families"]:
        primary = row["formal_primary_evidence"]
        secondary = row["secondary_efficiency_observation"]
        interpretation = row["mechanism_and_cost_interpretation"]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "freeze_content_sha256": audit["freeze_content_sha256"],
                "h21_anchor_binding_sha256": audit["h21_anchor_binding_sha256"],
                "refined_family": row["refined_family"],
                "cross_task_classification": primary["cross_task_classification"],
                "joint_positive_independent_stratum_count": primary[
                    "joint_positive_independent_stratum_count"
                ],
                "accepted_per_verified_positive_independent_stratum_count": secondary[
                    "accepted_per_verified_positive_independent_stratum_count"
                ],
                "pooled_accepted_per_verified_absolute_gain": secondary[
                    "pooled_across_independent_strata"
                ]["absolute_gain"],
                "coverage_positive_stratum_count": secondary[
                    "coverage_sign_counts"
                ].get("positive", 0),
                "coverage_zero_stratum_count": secondary["coverage_sign_counts"].get(
                    "zero", 0
                ),
                "coverage_negative_stratum_count": secondary[
                    "coverage_sign_counts"
                ].get("negative", 0),
                "refined_only_target_recovery_count": interpretation[
                    "refined_only_target_recovery_count"
                ],
                "refined_only_target_recovery_rate": interpretation[
                    "refined_only_target_recovery_rate"
                ],
                "govreport_replication_classification": interpretation[
                    "govreport_replication_classification"
                ],
                "verifier_cost_greater_than_dependency_cost": interpretation[
                    "verifier_cost_greater_than_dependency_cost_for_all_comparable_samples"
                ],
            }
        )
    return rows


def persist_h211(output_dir: Path, artifacts: dict[str, dict[str, Any]]) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / AUDIT_NAME, artifacts["audit"])
    write_json(output_dir / MANIFEST_NAME, artifacts["manifest"])
    write_csv(output_dir / AUDIT_CSV_NAME, audit_csv_rows(artifacts["audit"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", required=True, type=Path)
    parser.add_argument("--freeze-content-sha256", required=True)
    parser.add_argument("--h21-output-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = build_h211_artifacts(
        freeze_dir=args.freeze_dir,
        freeze_content_sha256=args.freeze_content_sha256,
        h21_output_root=args.h21_output_root,
    )
    persist_h211(args.output_dir, artifacts)
    print(json.dumps(artifacts["audit"]["route"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
