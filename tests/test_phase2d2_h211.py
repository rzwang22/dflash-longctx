from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from jsonschema import ValidationError

from scripts.audit_phase2d2a_h211_final_route import (
    AUDIT_CSV_NAME,
    AUDIT_NAME,
    MANIFEST_NAME,
    derive_final_route,
    persist_h211,
)
from scripts.check_phase2d2a_h211_acceptance import check_h211_output
from scripts.phase2d2a_h21_common import (
    FROZEN_REFINEMENT_FAMILIES,
    write_json,
)


def _fixtures() -> dict:
    joint_counts = dict(zip(FROZEN_REFINEMENT_FAMILIES, (3, 3, 2), strict=True))
    acceptance = {
        "status": "accepted",
        "cell_count": 112,
        "valid_cell_count": 112,
        "correctness_invalid_cell_count": 0,
        "infrastructure_failed_cell_count": 0,
        "pair_count": 84,
    }
    primary = {
        "execution_complete": True,
        "stable_cross_task_same_work_signal_exists": False,
        "family_cross_task_classifications": {
            family: "mixed_cross_task_same_work_signal"
            for family in FROZEN_REFINEMENT_FAMILIES
        },
        "winner_selected": False,
        "final_phase2d2a_branch_entered": False,
    }
    family_rows = []
    task_rows = []
    mechanism_rows = []
    cost_rows = []
    strata = (
        ("qasper", 1024),
        ("qasper", 8192),
        ("hotpotqa", 1024),
        ("hotpotqa", 8192),
    )
    for family in FROZEN_REFINEMENT_FAMILIES:
        joint = joint_counts[family]
        family_rows.append(
            {
                "refined_family": family,
                "cross_task_classification": "mixed_cross_task_same_work_signal",
                "joint_positive_independent_stratum_count": joint,
                "required_strata": [
                    {
                        "task": task,
                        "nominal_prompt_length": context,
                        "status": "joint_positive"
                        if index < joint
                        else "not_joint_positive",
                    }
                    for index, (task, context) in enumerate(strata)
                ],
                "same_task_replication_classification": "mixed_same_task_replication",
            }
        )
        efficiency_positive = 4 if family != FROZEN_REFINEMENT_FAMILIES[2] else 2
        for index, (task, context) in enumerate(strata):
            coverage = 0.1 if index < joint else 0.0
            accepted = 0.02 if index < efficiency_positive else 0.0
            task_rows.append(
                {
                    "refined_family": family,
                    "task": task,
                    "nominal_prompt_length": context,
                    "stratum_status": "joint_positive"
                    if coverage > 0 and accepted > 0
                    else "not_joint_positive",
                    "accepted_per_verified": {"absolute_gain": accepted},
                    "candidate_coverage": {"coverage_absolute_gain": coverage},
                    "sample_heterogeneity": {
                        "accepted_per_verified_gain_sign_counts": {
                            "positive": 3 if accepted > 0 else 0,
                            "zero": 0 if accepted > 0 else 3,
                        },
                        "coverage_gain_sign_counts": {
                            "positive": 3 if coverage > 0 else 0,
                            "zero": 0 if coverage > 0 else 3,
                        },
                        "joint_signal_counts": {
                            "positive_positive": 3
                            if coverage > 0 and accepted > 0
                            else 0
                        },
                    },
                }
            )
            task_rows[-1]["accepted_per_verified"].update(
                {
                    "base_accepted_draft_token_count": 10,
                    "refined_accepted_draft_token_count": 12 if accepted > 0 else 10,
                    "base_actual_verified_node_count": 100,
                    "refined_actual_verified_node_count": 100,
                }
            )
            mechanism_rows.append(
                {
                    "refined_family": family,
                    "task": task,
                    "nominal_prompt_length": context,
                    "mechanism_signal": "reranking_or_tree_reallocation_signal"
                    if coverage > 0
                    else "no_observed_candidate_mechanism_gain",
                    "refined_only_target_recovery_count": 0
                    if family == FROZEN_REFINEMENT_FAMILIES[0]
                    else 1,
                    "labelled_reference_position_count": 100,
                }
            )
        cost_rows.append(
            {
                "refined_family": family,
                "h1_cost_hierarchy_generalizes_for_all_comparable_samples": True,
                "comparable_sample_count": 12,
            }
        )
    return {
        "acceptance": acceptance,
        "primary": primary,
        "family": {"rows": family_rows},
        "task": {"rows": task_rows},
        "mechanism": {"rows": mechanism_rows},
        "cost": {"rows": cost_rows},
    }


def _derive() -> dict:
    fixture = _fixtures()
    return derive_final_route(
        freeze_content_sha256="a" * 64,
        h21_acceptance=fixture["acceptance"],
        primary=fixture["primary"],
        family_summary=fixture["family"],
        task_context_summary=fixture["task"],
        mechanism_summary=fixture["mechanism"],
        cost_summary=fixture["cost"],
    )


def _checker_artifacts(
    *, repository_commit: str = "a" * 40, branch: str = "artifact-branch"
) -> dict:
    audit = _derive()
    anchor = "b" * 64
    audit["h21_anchor_binding_sha256"] = anchor
    return {
        "audit": audit,
        "manifest": {
            "schema_version": "1.0",
            "phase": "2D.2A-H2.1.1",
            "artifact_type": "route_manifest",
            "status": "complete",
            "freeze_content_sha256": "a" * 64,
            "freeze_directory": "/immutable/freeze",
            "freeze_artifact_file_sha256": {"freeze.json": "c" * 64},
            "h21_output_root": "/immutable/h21",
            "h21_artifact_file_sha256": {},
            "h21_anchor_binding_sha256": anchor,
            "h21_acceptance": _fixtures()["acceptance"],
            "repository_commit": repository_commit,
            "branch": branch,
            "model_forward_executed": False,
            "h20_artifacts_mutated": False,
            "h21_artifacts_mutated": False,
            "artifacts": [AUDIT_NAME, AUDIT_CSV_NAME, MANIFEST_NAME],
        },
    }


def _artifact_hashes(output_dir: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        for name in (AUDIT_NAME, AUDIT_CSV_NAME, MANIFEST_NAME)
    }


def _route_binding(output_dir: Path) -> str:
    return hashlib.sha256(
        json.dumps(
            _artifact_hashes(output_dir),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


class Phase2D2H211Tests(unittest.TestCase):
    def _check_fixture(
        self,
        root: Path,
        observed: dict,
        expected: dict,
    ) -> dict:
        output_dir = root / "h211"
        h21_root = root / "h21"
        h21_root.mkdir()
        persist_h211(output_dir, observed)
        with patch(
            "scripts.check_phase2d2a_h211_acceptance.build_h211_artifacts",
            return_value=expected,
        ):
            return check_h211_output(
                output_dir,
                root / "freeze",
                "a" * 64,
                h21_root,
            )

    def test_primary_stable_signal_false_is_reproduced(self) -> None:
        audit = _derive()
        self.assertFalse(
            audit["formal_primary_evidence"][
                "stable_cross_task_same_work_signal_exists"
            ]
        )

    def test_joint_positive_counts_are_derived(self) -> None:
        audit = _derive()
        self.assertEqual(
            audit["formal_primary_evidence"][
                "joint_positive_independent_stratum_counts"
            ],
            dict(zip(FROZEN_REFINEMENT_FAMILIES, (3, 3, 2), strict=True)),
        )

    def test_secondary_efficiency_does_not_rewrite_primary(self) -> None:
        audit = _derive()
        secondary = audit["secondary_efficiency_observation"]
        self.assertEqual(
            secondary["families_positive_in_all_four_independent_strata"],
            list(FROZEN_REFINEMENT_FAMILIES[:2]),
        )
        self.assertFalse(secondary["primary_criterion_rewritten"])
        self.assertTrue(secondary["positive_efficiency_observation_preserved"])
        first = audit["families"][0]["secondary_efficiency_observation"]
        self.assertGreater(
            first["pooled_across_independent_strata"]["absolute_gain"], 0
        )

    def test_no_adapter_or_dependency_complexity_authorized(self) -> None:
        route = _derive()["route"]
        self.assertFalse(route["proceed_to_phase2d2b_lightweight_adapter"])
        self.assertFalse(route["increase_dependency_model_complexity"])

    def test_no_winner_or_later_branch(self) -> None:
        route = _derive()["route"]
        self.assertFalse(route["select_refinement_winner"])
        self.assertFalse(route["enter_phase2d3"])

    def test_h22_is_secondary_and_cannot_override_h21(self) -> None:
        route = _derive()["route"]
        self.assertTrue(route["h22_secondary_characterization_allowed"])
        self.assertFalse(route["h22_can_override_h21_primary"])

    def test_route_recommendation_is_not_scientific_conclusion(self) -> None:
        route = _derive()["route"]
        self.assertEqual(
            route["recommended_next_primary_design_focus_evidence_role"],
            "project_route_recommendation_not_experimentally_proven_conclusion",
        )

    def test_stable_family_input_is_rejected_closed(self) -> None:
        fixture = _fixtures()
        fixture["family"]["rows"][0]["cross_task_classification"] = (
            "stable_cross_task_same_work_signal"
        )
        with self.assertRaisesRegex(ValueError, "classification changed"):
            derive_final_route(
                freeze_content_sha256="a" * 64,
                h21_acceptance=fixture["acceptance"],
                primary=fixture["primary"],
                family_summary=fixture["family"],
                task_context_summary=fixture["task"],
                mechanism_summary=fixture["mechanism"],
                cost_summary=fixture["cost"],
            )

    def test_historical_creation_provenance_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            observed = _checker_artifacts()
            expected = deepcopy(observed)
            expected["manifest"]["repository_commit"] = "d" * 40
            expected["manifest"]["branch"] = "later-checker-branch"
            result = self._check_fixture(Path(temporary), observed, expected)
        self.assertEqual(result["status"], "accepted")

    def test_repository_commit_difference_alone_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            observed = _checker_artifacts()
            expected = deepcopy(observed)
            expected["manifest"]["repository_commit"] = "d" * 40
            result = self._check_fixture(Path(temporary), observed, expected)
        self.assertEqual(result["status"], "accepted")

    def test_branch_difference_alone_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            observed = _checker_artifacts()
            expected = deepcopy(observed)
            expected["manifest"]["branch"] = "later-checker-branch"
            result = self._check_fixture(Path(temporary), observed, expected)
        self.assertEqual(result["status"], "accepted")

    def test_immutable_h20_or_h21_hash_change_is_rejected(self) -> None:
        for field, changed in (
            ("freeze_artifact_file_sha256", {"freeze.json": "0" * 64}),
            ("h21_artifact_file_sha256", {"h21.json": "0" * 64}),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                observed = _checker_artifacts()
                expected = deepcopy(observed)
                observed["manifest"][field] = changed
                with self.assertRaisesRegex(ValueError, "immutable inputs"):
                    self._check_fixture(Path(temporary), observed, expected)

    def test_h21_anchor_binding_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            observed = _checker_artifacts()
            expected = deepcopy(observed)
            observed["manifest"]["h21_anchor_binding_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "immutable inputs"):
                self._check_fixture(Path(temporary), observed, expected)

    def test_model_forward_claim_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            observed = _checker_artifacts()
            expected = deepcopy(observed)
            observed["manifest"]["model_forward_executed"] = True
            with self.assertRaises(ValidationError):
                self._check_fixture(Path(temporary), observed, expected)

    def test_route_scientific_content_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observed = _checker_artifacts()
            expected = deepcopy(observed)
            output_dir = root / "h211"
            h21_root = root / "h21"
            h21_root.mkdir()
            persist_h211(output_dir, observed)
            changed = deepcopy(observed["audit"])
            changed["route"]["select_refinement_winner"] = True
            write_json(output_dir / AUDIT_NAME, changed)
            with (
                patch(
                    "scripts.check_phase2d2a_h211_acceptance.build_h211_artifacts",
                    return_value=expected,
                ),
                self.assertRaisesRegex(ValueError, "route audit differs"),
            ):
                check_h211_output(
                    output_dir,
                    root / "freeze",
                    "a" * 64,
                    h21_root,
                )

    def test_invalid_creation_provenance_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            observed = _checker_artifacts()
            expected = deepcopy(observed)
            observed["manifest"]["repository_commit"] = None
            with self.assertRaisesRegex(ValueError, "creation provenance"):
                self._check_fixture(Path(temporary), observed, expected)

    def test_checker_preserves_h211_files_and_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observed = _checker_artifacts()
            expected = deepcopy(observed)
            expected["manifest"]["repository_commit"] = "d" * 40
            output_dir = root / "h211"
            h21_root = root / "h21"
            h21_root.mkdir()
            persist_h211(output_dir, observed)
            hashes_before = _artifact_hashes(output_dir)
            binding_before = _route_binding(output_dir)
            with patch(
                "scripts.check_phase2d2a_h211_acceptance.build_h211_artifacts",
                return_value=expected,
            ):
                check_h211_output(
                    output_dir,
                    root / "freeze",
                    "a" * 64,
                    h21_root,
                )
            self.assertEqual(_artifact_hashes(output_dir), hashes_before)
            self.assertEqual(_route_binding(output_dir), binding_before)

    def test_creation_provenance_mutation_changes_h22_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "h211"
            persist_h211(output_dir, _checker_artifacts())
            binding_before = _route_binding(output_dir)
            manifest = json.loads((output_dir / MANIFEST_NAME).read_text())
            manifest["repository_commit"] = "f" * 40
            write_json(output_dir / MANIFEST_NAME, manifest)
            self.assertNotEqual(_route_binding(output_dir), binding_before)


if __name__ == "__main__":
    unittest.main()
