from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_phase2d2a_h_known import (
    BASE_FAMILY,
    EXPECTED_CONTEXT_LENGTHS,
    EXPECTED_FAMILIES,
    EXPECTED_NODE_BUDGETS,
    FROZEN_REFINEMENT_FAMILIES,
    _candidate_metrics,
)
from scripts.audit_phase2d2a_h1_budget_mechanism import (
    _h2_preregistered_design,
    build_h11_artifacts,
    classify_actual_work_regime,
    marginal_extra_node_yield,
    mechanism_signal,
    node_utilization_from_rounds,
    persist_h11,
)
from scripts.check_phase2d2a_h11_acceptance import check_h11_output

REPO_ROOT = Path(__file__).resolve().parents[1]


def _audit(budget: int, values: list[int]) -> dict:
    return node_utilization_from_rounds(
        configured_node_budget=budget,
        actual_verified_node_counts=values,
        selected_tree_node_counts=list(values),
    )


def _raw_audits() -> dict[tuple[str, int, int], dict]:
    rows = {}
    for family in EXPECTED_FAMILIES:
        for context in EXPECTED_CONTEXT_LENGTHS:
            for budget in EXPECTED_NODE_BUDGETS:
                values = [16, 16] if family == BASE_FAMILY or budget == 16 else [32, 32]
                rows[(family, context, budget)] = {
                    "schema_version": "1.0",
                    "task": "gov_report",
                    "sample_id": "gov_report-fd287bbfd3bdbdc9",
                    "source_sample_id": "gov_report-fd287bbfd3bdbdc9",
                    "nominal_prompt_length": context,
                    "actual_prompt_length": context,
                    "configuration_family": family,
                    "configuration_id": f"{family}-n{budget}",
                    "input_directory": f"/fixture/{family}-{context}-{budget}",
                    **_audit(budget, values),
                }
    return rows


def _synthetic_h1() -> tuple[dict, dict, dict, dict, dict]:
    raw = _raw_audits()
    cells = []
    for family, context, budget in sorted(raw):
        refined = family != BASE_FAMILY
        work = raw[(family, context, budget)]
        accepted = 6 if refined else 4
        committed = accepted + 2
        cells.append(
            {
                "configuration_family": family,
                "nominal_prompt_length": context,
                "node_budget": budget,
                "online_efficiency": {
                    "round_count": 2,
                    "verified_target_node_count": work[
                        "total_actual_verified_node_count"
                    ],
                    "accepted_draft_tokens_per_round": accepted / 2,
                    "committed_tokens_per_round": committed / 2,
                },
                "candidate_quality": {
                    "candidate_quality_available_round_count": 1,
                    "candidate_quality_unavailable_round_count": 1,
                    "labelled_reference_position_count": 4,
                    "unavailable_reason_distribution": {
                        "terminal_reference_too_short": 1
                    },
                    "base_candidate_coverage_rate": 0.5,
                    "conditional_candidate_coverage_rate": 0.25,
                    "merged_candidate_coverage_rate": 0.75,
                    "selected_tree_coverage_rate_at_fixed_n": 0.75,
                    "refined_only_target_recovery_count": 0,
                    "refined_only_target_recovery_rate": 0.0,
                    "target_token_rank_improvement_distribution": {
                        "count": 4,
                        "mean": 1.0,
                    },
                    "target_token_rank_improvement_histogram": {"1": 4},
                    "correct_prefix_survival_depth_distribution": {
                        "count": 1,
                        "mean": 3.0,
                    },
                    "correct_prefix_survival_depth_histogram": {"3": 1},
                },
                "component_cost": {
                    "conditional_remask_share_of_profiled_round_time": (
                        0.08 if refined else 0.0
                    ),
                    "fixed_shape_recheck_share_of_profiled_round_time": 0.35,
                    "combined_additional_work_share_of_profiled_round_time": (
                        0.43 if refined else 0.35
                    ),
                },
                "formal_timing": {
                    "repeat_count": 3,
                    "cuda_elapsed_ms_repeats": [99.0, 100.0, 101.0],
                    "generation_cuda_median_ms": 100.0,
                    "generation_cuda_ms_per_returned_token": 100.0 / 128,
                },
            }
        )
    pairs = []
    for family in FROZEN_REFINEMENT_FAMILIES:
        for context in EXPECTED_CONTEXT_LENGTHS:
            for budget in EXPECTED_NODE_BUDGETS:
                base_work = raw[(BASE_FAMILY, context, budget)][
                    "total_actual_verified_node_count"
                ]
                refined_work = raw[(family, context, budget)][
                    "total_actual_verified_node_count"
                ]
                base_efficiency = 4 / base_work
                refined_efficiency = 6 / refined_work
                pairs.append(
                    {
                        "refined_family": family,
                        "nominal_prompt_length": context,
                        "node_budget": budget,
                        "correctness_valid": True,
                        "candidate_quality": {"coverage_absolute_gain": 0.25},
                        "online_efficiency": {
                            "accepted_draft_tokens_per_verified_node": {
                                "absolute_gain": refined_efficiency - base_efficiency,
                                "relative_gain": (refined_efficiency - base_efficiency)
                                / base_efficiency,
                            }
                        },
                        "formal_timing": {
                            "cuda_ms_per_returned_token_relative_delta": -0.01
                        },
                    }
                )
    manifest = {"cells": cells}
    paired = {"rows": pairs}
    config = {
        "rows": [
            {
                "refined_family": family,
                "known_sample_stability_classification": "mixed_known_sample",
            }
            for family in FROZEN_REFINEMENT_FAMILIES
        ]
    }
    decision = {"final_phase2d2a_decision": "insufficient_evidence"}
    return manifest, paired, config, decision, raw


def _available_quality(available: bool) -> dict:
    return {
        "availability": "available" if available else "unavailable",
        "unavailable_reason": None if available else "terminal_reference_too_short",
        "target_label_authority": "original_dflash_reference",
        "target_label_source": "original_dflash_absolute_continuation",
        "q1_replay_used_as_target_label_source": False,
        "target_label_used_for_runtime_selection": False,
        "positions": (
            [
                {
                    "base_top_m_membership": True,
                    "conditional_top_r_membership": False,
                    "merged_membership": True,
                    "selected_tree_membership": True,
                    "refined_only_target_recovery": False,
                    "target_token_rank_improvement": 0,
                }
            ]
            if available
            else []
        ),
        "correct_prefix_survival_depth": 1 if available else 0,
        "first_pruning_depth": None,
    }


class Phase2D2H11Tests(unittest.TestCase):
    def test_utilization_uses_actual_verified_node_count(self) -> None:
        audit = _audit(32, [14, 16, 15])
        self.assertEqual(audit["total_actual_verified_node_count"], 45)
        self.assertEqual(audit["actual_verified_nodes_per_round"]["mean"], 15)
        self.assertAlmostEqual(audit["node_budget_utilization_rate"], 45 / 96)

    def test_configured_budget_is_not_treated_as_actual_work(self) -> None:
        audit = _audit(32, [16, 16])
        self.assertEqual(audit["actual_verified_nodes_per_round"]["mean"], 16)
        self.assertNotEqual(audit["actual_verified_nodes_per_round"]["mean"], 32)
        self.assertTrue(audit["configured_budget_not_substituted_for_actual_work"])

    def test_base_n32_with_16_nodes_is_underutilized(self) -> None:
        audit = _audit(32, [16, 16])
        self.assertEqual(audit["node_budget_utilization_rate"], 0.5)
        self.assertEqual(audit["utilization_classification"], "underutilized")

    def test_same_work_requires_direct_actual_work_equality(self) -> None:
        same = classify_actual_work_regime(_audit(16, [16, 16]), _audit(16, [16, 16]))
        expanded = classify_actual_work_regime(
            _audit(32, [16, 16]), _audit(32, [32, 32])
        )
        self.assertEqual(same["regime"], "same_actual_verification_work")
        self.assertEqual(expanded["regime"], "budget_cap_expansion")

    def test_budget_labels_alone_do_not_determine_regime(self) -> None:
        relation = classify_actual_work_regime(
            _audit(16, [16, 16]), _audit(32, [16, 16])
        )
        self.assertEqual(relation["regime"], "same_actual_verification_work")
        self.assertFalse(relation["configured_node_budget_used_to_infer_regime"])

    def test_marginal_extra_node_yield(self) -> None:
        result = marginal_extra_node_yield(
            base_verified_per_round=16,
            refined_verified_per_round=32,
            base_accepted_per_round=2,
            refined_accepted_per_round=3,
            base_committed_per_round=3,
            refined_committed_per_round=4.5,
        )
        self.assertEqual(result["delta_actual_verified_nodes_per_round"], 16)
        self.assertEqual(
            result["marginal_accepted_tokens_per_additional_verified_node"],
            1 / 16,
        )
        self.assertEqual(
            result["marginal_committed_tokens_per_additional_verified_node"],
            1.5 / 16,
        )

    def test_zero_work_delta_has_no_marginal_yield(self) -> None:
        result = marginal_extra_node_yield(
            base_verified_per_round=16,
            refined_verified_per_round=16,
            base_accepted_per_round=2,
            refined_accepted_per_round=3,
            base_committed_per_round=3,
            refined_committed_per_round=4,
        )
        self.assertFalse(result["marginal_yield_available"])
        self.assertIsNone(
            result["marginal_accepted_tokens_per_additional_verified_node"]
        )

    def test_unavailable_candidate_rows_remain_excluded(self) -> None:
        metrics = _candidate_metrics(
            [
                {"candidate_quality": _available_quality(True)},
                {"candidate_quality": _available_quality(False)},
            ],
            Path("fixture"),
        )
        self.assertEqual(metrics["labelled_reference_position_count"], 1)
        self.assertEqual(metrics["candidate_quality_unavailable_round_count"], 1)
        self.assertEqual(metrics["selected_tree_coverage_rate_at_fixed_n"], 1.0)

    def test_zero_refined_only_recovery_is_descriptive_not_failure(self) -> None:
        signal = mechanism_signal(
            selected_tree_coverage_gain=0.1, refined_only_recovery_count=0
        )
        self.assertEqual(
            signal["mechanism_signal"], "reranking_or_tree_reallocation_signal"
        )
        self.assertFalse(signal["refined_only_recovery_zero_treated_as_failure"])
        self.assertTrue(signal["descriptive_post_hoc_interpretation_only"])
        self.assertFalse(signal["causal_exclusivity_claimed"])

    def test_build_preserves_h1_and_adds_same_work_view(self) -> None:
        manifest, paired, config, decision, raw = _synthetic_h1()
        artifacts = build_h11_artifacts(
            h1_manifest=manifest,
            h1_paired=paired,
            h1_config=config,
            h1_decision=decision,
            raw_audits=raw,
            source={},
        )
        rows = artifacts["reinterpretation"]["configuration_rows"]
        self.assertTrue(
            all(
                row["original_cap_based_h1_classification"] == "mixed_known_sample"
                for row in rows
            )
        )
        self.assertTrue(
            all(
                row["primary_same_work_classification"]
                == "consistent_positive_same_work"
                for row in rows
            )
        )
        self.assertEqual(
            artifacts["reinterpretation"]["final_phase2d2a_decision"],
            "insufficient_evidence",
        )

    def test_h2_preregistration_keeps_all_frozen_families(self) -> None:
        h2 = _h2_preregistered_design(direct_regime_confirmed=True)
        self.assertIsNotNone(h2)
        self.assertEqual(
            tuple(h2["frozen_refinement_families"]),
            FROZEN_REFINEMENT_FAMILIES,
        )
        self.assertFalse(h2["winner_selected_from_h1"])
        self.assertFalse(h2["frozen_family_selected_or_removed"])
        self.assertFalse(h2["h2_execution_authorized"])

    def test_h2_is_not_emitted_without_direct_regime_confirmation(self) -> None:
        self.assertIsNone(_h2_preregistered_design(direct_regime_confirmed=False))

    def test_persisted_synthetic_audit_passes_checker(self) -> None:
        manifest, paired, config, decision, raw = _synthetic_h1()
        artifacts = build_h11_artifacts(
            h1_manifest=manifest,
            h1_paired=paired,
            h1_config=config,
            h1_decision=decision,
            raw_audits=raw,
            source={},
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "h11"
            persist_h11(output, artifacts)
            result = check_h11_output(output)
            self.assertEqual(result["status"], "accepted")
            self.assertEqual(result["matrix_cell_count"], 16)
            self.assertEqual(result["pair_count"], 12)
            self.assertTrue(result["h2_preregistered_design_present"])

    def test_online_and_third_party_sources_are_unchanged(self) -> None:
        expected = {
            "third_party/qwen3_8b_dflash_b16/instrumented/dflash.py": (
                "6b20d1476af5746ffa511df3441230ee25cdc5f25f25d98b7eae8f72dbf8e533"
            ),
            "dflash/phase2d2/refinement.py": (
                "9c4e5acd20d4076332841d15277a54b821c2f3b88007cea53ff90410f517b9bc"
            ),
            "dflash/phase2d2/robustness.py": (
                "063b2689f63d89504ac319e3c0b46b717ff0e4d5717c7514d965043ff130cf01"
            ),
            "dflash/phase2d2/tensor.py": (
                "dec781f37c0b35fe5b546c457ae2f420fa6578281d8dce696366bb4d83d863ab"
            ),
        }
        for relative, digest in expected.items():
            observed = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(observed, digest, relative)


if __name__ == "__main__":
    unittest.main()
