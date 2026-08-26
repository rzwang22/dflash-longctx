from __future__ import annotations

import unittest

from scripts.audit_phase2d2a_h211_final_route import derive_final_route
from scripts.phase2d2a_h21_common import FROZEN_REFINEMENT_FAMILIES


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


class Phase2D2H211Tests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
