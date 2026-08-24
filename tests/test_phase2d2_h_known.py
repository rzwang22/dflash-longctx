from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_phase2d2a_h_known import (
    BASE_FAMILY,
    COMMON_PROVENANCE_FIELDS,
    EXPECTED_CONTEXT_LENGTHS,
    EXPECTED_FAMILIES,
    EXPECTED_NODE_BUDGETS,
    FROZEN_REFINEMENT_FAMILIES,
    _aggregate_cells,
    _candidate_metrics,
    _component_cost,
    _online_metrics,
    _pair_cells,
    _validate_cross_run_provenance,
    _validate_matrix,
    persist_h1,
)
from scripts.check_phase2d2a_h_known_acceptance import check_h1_output

REPO_ROOT = Path(__file__).resolve().parents[1]


def _provenance(*, refined: bool) -> dict:
    return {
        "repository_commit": "a" * 40,
        "target_model_id": "Qwen/Qwen3-8B",
        "target_model_revision": "b" * 40,
        "draft_model_id": "z-lab/Qwen3-8B-DFlash-b16",
        "draft_model_revision": "c" * 40,
        "materialization_id": "phase2d2a-materialized",
        "selector_hash": "d" * 64,
        "frozen_refinement_hash": "e" * 64 if refined else None,
        "online_selector_mode": "top1_packed_tree",
        "verification_mode": "always_fixed_shape_recheck",
        "dtype": "bfloat16",
        "device": "cuda:0",
        "block_size": 16,
        "max_new_tokens": 128,
        "timing_repeats": 3,
        "warmup_runs": 1,
        "batch_size": 1,
        "temperature": 0.0,
    }


def _component(*, refined: bool) -> dict:
    conditional = 2.0 if refined else 0.0
    return {
        "component_timing_source": "separate_profile_and_diagnostics_paths",
        "conditional_remask_cuda_median_ms_per_round": conditional / 2,
        "conditional_remask_profile_total_cuda_ms": conditional,
        "conditional_remask_share_of_profiled_round_time": conditional / 20.0,
        "fixed_shape_recheck_cuda_median_ms_per_round": 1.5,
        "fixed_shape_recheck_profile_total_cuda_ms": 3.0,
        "fixed_shape_recheck_share_of_profiled_round_time": 0.15,
        "fixed_shape_round_entry_snapshot_profile_total_cuda_ms": 0.5,
        "fixed_shape_kv_replacement_profile_total_cuda_ms": 0.5,
        "extended_fixed_shape_path_profile_total_cuda_ms": 4.0,
        "extended_fixed_shape_path_share_of_profiled_round_time": 0.2,
        "fallback_timer_excludes_snapshot_and_kv_replacement": True,
        "combined_conditional_and_fixed_profile_total_cuda_ms": conditional + 3.0,
        "combined_additional_work_share_of_profiled_round_time": (
            conditional + 3.0
        )
        / 20.0,
        "profile_initial_parallel_draft_cuda_ms": 2.0,
        "profile_total_draft_cuda_ms": 8.0,
        "profile_packed_verification_cuda_ms": 5.0,
        "profile_total_verification_cuda_ms": 10.0,
        "profile_round_cuda_ms": 20.0,
        "formal_target_forward_count": 20,
        "formal_target_forward_count_per_round": 2.0,
        "conditional_over_initial_draft_ratio": None,
        "conditional_over_initial_draft_ratio_unavailable_reason": "scope mismatch",
        "nested_timer_double_counting_avoided": True,
        "combined_sum_contains_only_non_overlapping_conditional_and_fixed_recheck": True,
    }


def _cell(
    family: str,
    context: int,
    budget: int,
    *,
    valid: bool = True,
) -> dict:
    refined = family != BASE_FAMILY
    accepted = 12 if refined else 10
    coverage = 0.6 if refined else 0.5
    latency = 110.0 if refined else 100.0
    return {
        "schema_version": "1.0",
        "input_directory": f"/results/{family}-{context}-{budget}",
        "task": "gov_report",
        "sample_id": "gov_report-fd287bbfd3bdbdc9",
        "source_sample_id": "gov_report-fd287bbfd3bdbdc9",
        "source_text_sha256": "f" * 64,
        "context_prefix_sha256": hashlib.sha256(str(context).encode()).hexdigest(),
        "nominal_prompt_length": context,
        "actual_prompt_length": context,
        "node_budget": budget,
        "configuration_family": family,
        "configuration_id": f"{family}-n{budget}-always-fixed",
        "provenance": _provenance(refined=refined),
        "correctness": {
            "runner_status": "ok" if valid else "error",
            "checker_status": "accepted" if valid else "rejected",
            "checker_result": {"status": "accepted"} if valid else None,
            "checker_error": None if valid else "output mismatch",
            "original_dflash_reference_tree_output_exact": valid,
            "formal_repeat_stable": True,
            "profile_formal_output_exact": True,
            "diagnostic_formal_output_exact": True,
            "structural_correctness_passed": True,
            "actual_execution_correctness_passed": valid,
            "correctness_valid": valid,
            "hard_correctness_failure_reasons": [] if valid else ["output mismatch"],
            "target_only_is_auxiliary": True,
            "target_only_output_ids_equal": False,
        },
        "candidate_quality": {
            "candidate_quality_total_round_count": 10,
            "candidate_quality_available_round_count": 8,
            "candidate_quality_unavailable_round_count": 2,
            "candidate_label_available_round_rate": 0.8,
            "labelled_reference_position_count": 80,
            "unavailable_reason_distribution": {
                "original_dflash_target_labels_unavailable": 2
            },
            "selected_tree_coverage_rate_at_fixed_n": coverage,
            "base_candidate_coverage_rate": 0.5,
            "conditional_candidate_coverage_rate": 0.2 if refined else 0.0,
            "merged_candidate_coverage_rate": 0.6 if refined else 0.5,
            "refined_only_target_recovery_count": 8 if refined else 0,
            "refined_only_target_recovery_rate": 0.1 if refined else 0.0,
            "correct_prefix_survival_depth_distribution": {"count": 8},
            "first_pruning_depth_distribution": {"count": 7},
            "first_pruning_not_observed_count": 1,
            "target_token_rank_improvement_distribution": {"count": 4},
            "target_label_authority": "original_dflash_reference",
            "target_label_source": "original_dflash_absolute_continuation",
            "q1_replay_used_as_target_label_source": False,
            "target_labels_used_for_runtime_selection": False,
            "zero_placeholder_rows_excluded_from_denominators": True,
        },
        "online_efficiency": {
            "generated_returned_token_count": 128,
            "round_count": 10,
            "accepted_draft_token_count": accepted,
            "committed_token_count": accepted + 10,
            "verified_target_node_count": 100,
            "accepted_draft_tokens_per_verified_node": accepted / 100,
            "accepted_draft_tokens_per_round": accepted / 10,
            "committed_tokens_per_round": (accepted + 10) / 10,
            "verified_target_nodes_per_returned_token": 100 / 128,
            "rounds_per_returned_token": 10 / 128,
        },
        "formal_timing": {
            "formal_timing_source": "unprofiled_control_repeats",
            "repeat_count": 3,
            "cuda_elapsed_ms_repeats": [latency - 1, latency, latency + 1],
            "generation_cuda_median_ms": latency,
            "generation_cuda_mean_ms": latency,
            "generation_cuda_p90_ms": latency + 1,
            "generation_cuda_min_ms": latency - 1,
            "generation_cuda_max_ms": latency + 1,
            "generation_cuda_range_ms": 2.0,
            "generation_cuda_ms_per_returned_token": latency / 128,
            "generation_cuda_ms_per_committed_token": latency / (accepted + 10),
            "returned_tokens_per_second": 128000 / latency,
            "diagnostic_or_gold_latency_used": False,
        },
        "component_cost": _component(refined=refined),
    }


def _matrix() -> list[dict]:
    return [
        _cell(family, context, budget)
        for family in EXPECTED_FAMILIES
        for context in EXPECTED_CONTEXT_LENGTHS
        for budget in EXPECTED_NODE_BUDGETS
    ]


def _quality(*, available: bool, q1: bool = False) -> dict:
    return {
        "availability": "available" if available else "unavailable",
        "unavailable_reason": None if available else "terminal_reference_too_short",
        "target_label_authority": "original_dflash_reference",
        "target_label_source": "original_dflash_absolute_continuation",
        "q1_replay_used_as_target_label_source": q1,
        "target_label_used_for_runtime_selection": False,
        "positions": (
            [
                {
                    "base_top_m_membership": True,
                    "conditional_top_r_membership": True,
                    "merged_membership": True,
                    "selected_tree_membership": True,
                    "refined_only_target_recovery": False,
                    "target_token_rank_improvement": 2,
                }
            ]
            if available
            else []
        ),
        "correct_prefix_survival_depth": 1 if available else 0,
        "first_pruning_depth": None,
    }


class Phase2D2HKnownTests(unittest.TestCase):
    def test_exact_16_cell_inventory(self) -> None:
        cells = _matrix()
        _validate_matrix(cells)
        with self.assertRaisesRegex(ValueError, "matrix inventory"):
            _validate_matrix(cells[:-1])

    def test_pairing_is_exact_by_sample_context_and_n(self) -> None:
        pairs = _pair_cells(_matrix())
        self.assertEqual(len(pairs), 12)
        self.assertTrue(
            all(
                row["pairing_key"]["context_length"]
                == row["nominal_prompt_length"]
                and row["pairing_key"]["node_budget"] == row["node_budget"]
                for row in pairs
            )
        )

    def test_n16_cannot_pair_with_n32(self) -> None:
        cells = _matrix()
        row = next(
            row
            for row in cells
            if row["configuration_family"] == FROZEN_REFINEMENT_FAMILIES[0]
            and row["nominal_prompt_length"] == 1024
            and row["node_budget"] == 16
        )
        row["node_budget"] = 32
        with self.assertRaisesRegex(ValueError, "missing exact same-N/context pair"):
            _pair_cells(cells)

    def test_1k_cannot_pair_with_16k(self) -> None:
        cells = _matrix()
        row = next(
            row
            for row in cells
            if row["configuration_family"] == FROZEN_REFINEMENT_FAMILIES[0]
            and row["nominal_prompt_length"] == 1024
            and row["node_budget"] == 16
        )
        row["nominal_prompt_length"] = 16384
        with self.assertRaisesRegex(ValueError, "missing exact same-N/context pair"):
            _pair_cells(cells)

    def test_incompatible_provenance_is_rejected(self) -> None:
        cells = _matrix()
        cells[-1]["provenance"]["repository_commit"] = "changed"
        with self.assertRaisesRegex(ValueError, "repository_commit"):
            _validate_cross_run_provenance(cells)
        self.assertIn("materialization_id", COMMON_PROVENANCE_FIELDS)

    def test_correctness_failed_pair_is_excluded_from_benefit_and_cost(self) -> None:
        cells = _matrix()
        cells[-1] = _cell(
            FROZEN_REFINEMENT_FAMILIES[-1], 16384, 32, valid=False
        )
        aggregate = _aggregate_cells(
            cells, [Path(f"/input/{index}") for index in range(16)]
        )
        pair = next(
            row
            for row in aggregate["paired"]["rows"]
            if row["refined_family"] == FROZEN_REFINEMENT_FAMILIES[-1]
            and row["nominal_prompt_length"] == 16384
            and row["node_budget"] == 32
        )
        self.assertFalse(pair["correctness_valid"])
        self.assertIsNone(pair["candidate_quality"]["coverage_absolute_gain"])
        self.assertIsNone(
            pair["online_efficiency"][
                "accepted_draft_tokens_per_verified_node"
            ]["absolute_gain"]
        )
        cost = next(
            row
            for row in aggregate["cost_summary"]["rows"]
            if row["refined_family"] == FROZEN_REFINEMENT_FAMILIES[-1]
            and row["nominal_prompt_length"] == 16384
            and row["node_budget"] == 32
        )
        self.assertFalse(cost["eligible_for_cost_conclusions"])
        self.assertTrue(
            all(value is None for value in cost["dependency_modeling_cost"].values())
        )

    def test_unavailable_candidate_rows_do_not_enter_denominators(self) -> None:
        metrics = _candidate_metrics(
            [
                {"candidate_quality": _quality(available=True)},
                {"candidate_quality": _quality(available=False)},
            ],
            Path("fixture"),
        )
        self.assertEqual(metrics["labelled_reference_position_count"], 1)
        self.assertEqual(metrics["selected_tree_coverage_rate_at_fixed_n"], 1.0)
        self.assertEqual(metrics["candidate_quality_unavailable_round_count"], 1)

    def test_candidate_authority_is_original_dflash_and_q1_is_rejected(self) -> None:
        metrics = _candidate_metrics(
            [{"candidate_quality": _quality(available=True)}], Path("fixture")
        )
        self.assertEqual(metrics["target_label_authority"], "original_dflash_reference")
        self.assertFalse(metrics["q1_replay_used_as_target_label_source"])
        with self.assertRaisesRegex(ValueError, "q1"):
            _candidate_metrics(
                [{"candidate_quality": _quality(available=True, q1=True)}],
                Path("fixture"),
            )

    def test_absolute_and_relative_gains(self) -> None:
        pair = _pair_cells(_matrix())[0]
        metric = pair["online_efficiency"][
            "accepted_draft_tokens_per_verified_node"
        ]
        self.assertAlmostEqual(metric["absolute_gain"], 0.02)
        self.assertAlmostEqual(metric["relative_gain"], 0.2)
        self.assertAlmostEqual(pair["candidate_quality"]["coverage_absolute_gain"], 0.1)

    def test_online_efficiency_denominators(self) -> None:
        sample = {"generated_token_count": 8}
        rounds = [
            {
                "accepted_draft_token_count": 2,
                "committed_token_count": 3,
                "phase2d2": {"actual_verified_node_count": 16},
            },
            {
                "accepted_draft_token_count": 4,
                "committed_token_count": 5,
                "phase2d2": {"actual_verified_node_count": 32},
            },
        ]
        metrics = _online_metrics(sample, rounds)
        self.assertEqual(metrics["accepted_draft_tokens_per_verified_node"], 6 / 48)
        self.assertEqual(metrics["verified_target_nodes_per_returned_token"], 6.0)
        self.assertEqual(metrics["rounds_per_returned_token"], 0.25)

    def test_cost_decomposition_avoids_nested_timer_double_counting(self) -> None:
        sample = {
            "profiling": {
                "components": {
                    "conditional_remask_cuda_elapsed_ms": 2.0,
                    "canonical_recheck_cuda_elapsed_ms": 3.0,
                    "round_cuda_elapsed_ms": 20.0,
                    "draft_hidden_logits_cuda_elapsed_ms": 2.0,
                    "total_draft_cuda_elapsed_ms": 8.0,
                    "verification_cuda_elapsed_ms": 5.0,
                    "total_verification_cuda_elapsed_ms": 10.0,
                    "stabilization_cache_snapshot_cuda_elapsed_ms": 0.5,
                    "kv_replacement_cuda_elapsed_ms": 0.5,
                }
            }
        }
        rounds = [
            {
                "phase2d2": {
                    "conditional_remask_cuda_elapsed_ms": 2.0,
                    "fallback_cuda_elapsed_ms": 3.0,
                    "formal_target_forward_count": 2,
                }
            }
        ]
        cost = _component_cost(sample, rounds)
        self.assertEqual(
            cost["combined_conditional_and_fixed_profile_total_cuda_ms"], 5.0
        )
        self.assertEqual(
            cost["combined_additional_work_share_of_profiled_round_time"], 0.25
        )
        self.assertIsNone(cost["conditional_over_initial_draft_ratio"])
        self.assertTrue(cost["nested_timer_double_counting_avoided"])
        self.assertEqual(
            cost["extended_fixed_shape_path_profile_total_cuda_ms"], 4.0
        )

    def test_h1_decision_remains_insufficient_and_frozen(self) -> None:
        aggregate = _aggregate_cells(
            _matrix(), [Path(f"/input/{index}") for index in range(16)]
        )
        decision = aggregate["decision"]
        self.assertEqual(decision["final_phase2d2a_decision"], "insufficient_evidence")
        self.assertEqual(decision["independent_task_count"], 1)
        self.assertFalse(decision["winner_selected"])
        self.assertFalse(
            aggregate["config_summary"][
                "frozen_refinement_configuration_set_reselected"
            ]
        )
        self.assertEqual(
            {row["refined_family"] for row in aggregate["config_summary"]["rows"]},
            set(FROZEN_REFINEMENT_FAMILIES),
        )

    def test_persisted_artifacts_pass_h1_checker(self) -> None:
        aggregate = _aggregate_cells(
            _matrix(), [Path(f"/input/{index}") for index in range(16)]
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "aggregate"
            persist_h1(output, aggregate)
            result = check_h1_output(output)
            self.assertEqual(result["status"], "accepted")
            self.assertEqual(result["cell_count"], 16)
            self.assertEqual(result["pair_count"], 12)

    def test_online_execution_sources_are_unchanged(self) -> None:
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
