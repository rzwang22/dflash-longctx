from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import venv
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from jsonschema import ValidationError

from dflash.phase2d2.freeze import config_sha256, validate_frozen_refinement
from scripts.aggregate_phase2d2a_h_known import _candidate_metrics
from scripts.freeze_phase2d2a_h2_cohort import (
    BASE_FAMILY,
    FROZEN_REFINEMENT_FAMILIES,
    build_freeze_artifacts,
    persist_freeze,
)
from scripts.phase2d2a_h21_common import (
    AGGREGATE_JSON_ARTIFACTS,
    REQUIRED_FREEZE_SHA256,
    build_aggregate,
    build_execution_manifest,
    build_execution_plan,
    cost_summaries,
    ensure_immutable_freeze,
    family_summaries,
    matrix_identity,
    mechanism_summaries,
    pair_cells,
    persist_aggregate,
    result_relative_dir,
    task_context_summaries,
    validate_cell_provenance,
    validate_h21_freeze,
    validate_h21_schema,
    validate_no_extra_outputs,
    validate_plan,
    write_json,
    _validate_runtime_refinement_identity,
)
from scripts.run_phase2d2a_h21_primary import (
    _absolute_without_symlink_dereference,
    _child_environment_preflight,
    _execute_row,
    _index_rows,
    _probe_child_environment,
    _refresh_index_and_manifest,
    _run_rows_with_circuit_breaker,
    _runner_command,
    _runtime_paths,
    _sentinel_cell_gate,
    _sentinel_gate_return_code,
    _sentinel_rows,
    _status_for_row,
    main as run_h21_main,
)
from tests.test_phase2d2_h2_freeze import _exclusions, _inventory

REPO_ROOT = Path(__file__).resolve().parents[1]


def _freeze_artifacts() -> dict:
    return build_freeze_artifacts(
        inventory=_inventory(qasper_16k=2, hotpot_16k=7),
        exclusions=_exclusions(),
        tokenizer_provenance={
            "model_id": "Qwen/Qwen3-8B",
            "revision": "b" * 40,
            "model_loaded": False,
        },
        h11_provenance={
            "checker_status": "accepted",
            "h2_execution_authorized_by_h11": False,
        },
    )


def _frozen_refinement() -> dict:
    payload = {
        "schema_version": "phase2d2a-frozen-refinement-v1",
        "source_artifacts": [],
        "selection_rule": "fixture canonical refinement set",
        "configurations": [
            {
                "configuration_id": "conditional_rerank-d1-p1-r2-w0.5",
                "candidate_merge_mode": "conditional_rerank",
                "base_top_m": 16,
                "conditional_top_r": 2,
                "refine_depth": 1,
                "refined_prefix_budget": 1,
                "refinement_weight": 0.5,
            },
            {
                "configuration_id": "union-d2-p2-r2-w0.5",
                "candidate_merge_mode": "union",
                "base_top_m": 16,
                "conditional_top_r": 2,
                "refine_depth": 2,
                "refined_prefix_budget": 2,
                "refinement_weight": 0.5,
            },
            {
                "configuration_id": "union-d4-p2-r4-w0.5",
                "candidate_merge_mode": "union",
                "base_top_m": 16,
                "conditional_top_r": 4,
                "refine_depth": 4,
                "refined_prefix_budget": 2,
                "refinement_weight": 0.5,
            },
        ],
        "maximum_selected_configuration_count": 3,
        "target_outputs_used_for_final_online_selection": False,
        "held_out_evaluation_results_used": False,
        "manual_override_used": False,
    }
    payload["config_sha256"] = config_sha256(payload)
    return validate_frozen_refinement(payload)


def _runtime_phase(frozen: dict, family: str) -> dict:
    selected = next(
        row for row in frozen["configurations"] if row["configuration_id"] == family
    )
    return {
        **selected,
        "frozen_refinement_configuration_id": family,
        "frozen_refinement_config_hash": frozen["config_sha256"],
    }


def _external_status(row: dict, *, checker_return_code: int = 0) -> dict:
    return {
        "schema_version": "1.0",
        "phase": "2D.2A-H2.1",
        "freeze_content_sha256": row["freeze_content_sha256"],
        "execution_id": row["execution_id"],
        "frozen_matrix_row_identity": matrix_identity(row),
        "repository_commit": "a" * 40,
        "execution_start_utc": "start",
        "execution_end_utc": "end",
        "execution_start_status": "started",
        "execution_end_status": "completed",
        "runner_return_code": 0,
        "external_phase2d2_checker_return_code": checker_return_code,
        "result_identity_valid": True,
        "result_identity_error": None,
        "command": ["python"],
    }


def _distribution(value: float) -> dict:
    return {
        "count": 1,
        "mean": value,
        "median": value,
        "p90": value,
        "min": value,
        "max": value,
    }


def _component(refined: bool) -> dict:
    conditional = 2.0 if refined else 0.0
    return {
        "component_timing_source": "separate_profile_and_diagnostics_paths",
        "conditional_remask_cuda_median_ms_per_round": conditional / 2,
        "conditional_remask_profile_total_cuda_ms": conditional,
        "conditional_remask_share_of_profiled_round_time": conditional / 20,
        "fixed_shape_recheck_cuda_median_ms_per_round": 1.5,
        "fixed_shape_recheck_profile_total_cuda_ms": 3.0,
        "fixed_shape_recheck_share_of_profiled_round_time": 0.15,
        "fixed_shape_round_entry_snapshot_profile_total_cuda_ms": 0.5,
        "fixed_shape_kv_replacement_profile_total_cuda_ms": 0.5,
        "extended_fixed_shape_path_profile_total_cuda_ms": 4.0,
        "extended_fixed_shape_path_share_of_profiled_round_time": 0.2,
        "fallback_timer_excludes_snapshot_and_kv_replacement": True,
        "combined_conditional_and_fixed_profile_total_cuda_ms": conditional + 3.0,
        "combined_additional_work_share_of_profiled_round_time": (conditional + 3.0)
        / 20,
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


def _provenance(refined: bool) -> dict:
    return {
        "repository_commit": "a" * 40,
        "target_model_id": "Qwen/Qwen3-8B",
        "target_model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "draft_model_id": "z-lab/Qwen3-8B-DFlash-b16",
        "draft_model_revision": "9b41424b7109f9c5413454f481b09a82b85333f4",
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


def _cell(
    row: dict,
    *,
    valid: bool = True,
    infrastructure: bool = False,
    positive: bool = True,
    candidate_available: bool = True,
) -> dict:
    refined = row["configuration_family"] != BASE_FAMILY
    if infrastructure:
        return {
            "schema_version": "1.0",
            **matrix_identity(row),
            "result_directory": row["result_relative_dir"],
            "cell_status": "infrastructure_failed",
            "infrastructure_failure_reason": "fixture infrastructure failure",
            "correctness": None,
            "candidate_quality": None,
            "online_efficiency": None,
            "formal_timing": None,
            "component_cost": None,
            "provenance": None,
        }
    labelled = 10 if candidate_available else 0
    selected = (6 if positive else 4) if refined else 5
    accepted = (12 if positive else 8) if refined else 10
    latency = 110.0 if refined else 100.0
    candidate = {
        "candidate_quality_total_round_count": 2,
        "candidate_quality_available_round_count": 2 if candidate_available else 0,
        "candidate_quality_unavailable_round_count": 0 if candidate_available else 2,
        "candidate_label_available_round_rate": 1.0 if candidate_available else 0.0,
        "labelled_reference_position_count": labelled,
        "unavailable_reason_distribution": {}
        if candidate_available
        else {"terminal_reference_too_short": 2},
        "selected_tree_coverage_rate_at_fixed_n": selected / labelled
        if labelled
        else None,
        "base_candidate_coverage_rate": 0.5 if labelled else None,
        "conditional_candidate_coverage_rate": 0.2
        if refined and labelled
        else 0.0
        if labelled
        else None,
        "merged_candidate_coverage_rate": selected / labelled if labelled else None,
        "refined_only_target_recovery_count": 0,
        "refined_only_target_recovery_rate": 0.0 if labelled else None,
        "correct_prefix_survival_depth_distribution": _distribution(2.0),
        "first_pruning_depth_distribution": _distribution(3.0),
        "first_pruning_not_observed_count": 0,
        "target_token_rank_improvement_distribution": _distribution(1.0),
        "target_label_authority": "original_dflash_reference",
        "target_label_source": "original_dflash_absolute_continuation",
        "q1_replay_used_as_target_label_source": False,
        "target_labels_used_for_runtime_selection": False,
        "zero_placeholder_rows_excluded_from_denominators": True,
        "base_candidate_membership_count": 5 if labelled else 0,
        "conditional_candidate_membership_count": 2 if refined and labelled else 0,
        "merged_candidate_membership_count": selected if labelled else 0,
        "selected_tree_membership_count": selected if labelled else 0,
    }
    online = {
        "generated_returned_token_count": 20,
        "round_count": 2,
        "accepted_draft_token_count": accepted,
        "committed_token_count": accepted + 2,
        "verified_target_node_count": 100,
        "accepted_draft_tokens_per_verified_node": accepted / 100,
        "accepted_draft_tokens_per_round": accepted / 2,
        "committed_tokens_per_round": (accepted + 2) / 2,
        "verified_target_nodes_per_returned_token": 5.0,
        "rounds_per_returned_token": 0.1,
    }
    return {
        "schema_version": "1.0",
        **matrix_identity(row),
        "result_directory": row["result_relative_dir"],
        "cell_status": "valid" if valid else "correctness_invalid",
        "infrastructure_failure_reason": None,
        "correctness": {
            "correctness_valid": valid,
            "original_dflash_reference_tree_output_exact": valid,
        },
        "candidate_quality": candidate,
        "online_efficiency": online,
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
            "generation_cuda_ms_per_returned_token": latency / 20,
            "generation_cuda_ms_per_committed_token": latency / (accepted + 2),
            "returned_tokens_per_second": 20000 / latency,
            "diagnostic_or_gold_latency_used": False,
        },
        "component_cost": _component(refined),
        "provenance": _provenance(refined),
    }


def _quality(
    *,
    available: bool = True,
    q1: bool = False,
    authority: str = "original_dflash_reference",
) -> dict:
    return {
        "availability": "available" if available else "unavailable",
        "unavailable_reason": None if available else "terminal_reference_too_short",
        "target_label_authority": authority,
        "target_label_source": "original_dflash_absolute_continuation",
        "q1_replay_used_as_target_label_source": q1,
        "target_label_used_for_runtime_selection": False,
        "positions": [
            {
                "base_top_m_membership": True,
                "conditional_top_r_membership": True,
                "merged_membership": True,
                "selected_tree_membership": True,
                "refined_only_target_recovery": False,
                "target_token_rank_improvement": 1,
            }
        ]
        if available
        else [],
        "correct_prefix_survival_depth": 1,
        "first_pruning_depth": None,
    }


class Phase2D2H21Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.freeze_dir = cls.root / "freeze"
        cls.artifacts = _freeze_artifacts()
        persist_freeze(cls.freeze_dir, cls.artifacts)
        cls.freeze_sha = cls.artifacts["manifest"]["freeze_content_sha256"]
        cls.bundle = validate_h21_freeze(
            cls.freeze_dir,
            cls.freeze_sha,
            required_freeze_sha256=cls.freeze_sha,
        )
        cls.plan = build_execution_plan(cls.bundle)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def cells(self) -> list[dict]:
        return [_cell(row) for row in self.plan["rows"]]

    def test_01_exact_freeze_sha_precedes_checker_or_model_path(self) -> None:
        with patch("scripts.phase2d2a_h21_common.check_h2_freeze") as checker:
            with self.assertRaisesRegex(ValueError, "exact frozen"):
                validate_h21_freeze(self.freeze_dir, "0" * 64)
            checker.assert_not_called()

    def test_02_literal_server_freeze_sha_is_pinned(self) -> None:
        self.assertEqual(
            REQUIRED_FREEZE_SHA256,
            "d5fdc816b183c19bd5683fb89a3c5e1ace641a15d2ac68da7b6dc0e56a678856",
        )

    def test_03_plan_has_exact_112_rows(self) -> None:
        self.assertEqual(self.plan["row_count"], 112)
        self.assertEqual(len(self.plan["rows"]), 112)
        self.assertEqual(
            self.plan["task_row_counts"],
            {"gov_report": 24, "hotpotqa": 64, "qasper": 24},
        )
        self.assertEqual(set(self.plan["family_row_counts"].values()), {28})

    def test_04_plan_row_identity_equals_frozen_matrix(self) -> None:
        validate_plan(self.plan, self.bundle)

    def test_05_no_n32_row_enters_h21(self) -> None:
        self.assertEqual({row["node_budget"] for row in self.plan["rows"]}, {16})

    def test_06_frozen_sample_replacement_is_rejected(self) -> None:
        changed = deepcopy(self.plan)
        changed["rows"][0]["sample_id"] = "replacement"
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_plan(changed, self.bundle)

    def test_07_family_removal_is_rejected(self) -> None:
        changed = deepcopy(self.plan)
        changed["rows"] = changed["rows"][:-1]
        changed["row_count"] -= 1
        with self.assertRaises(ValidationError):
            validate_plan(changed, self.bundle)

    def test_08_exact_base_refined_pairing(self) -> None:
        pairs = pair_cells(self.plan, self.cells())
        self.assertEqual(len(pairs), 84)
        self.assertTrue(all(row["base_family"] == BASE_FAMILY for row in pairs))

    def test_09_context_cross_pair_is_rejected(self) -> None:
        cells = self.cells()
        target = next(
            row for row in cells if row["configuration_family"] != BASE_FAMILY
        )
        target["context_prefix_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "pairing mismatch"):
            pair_cells(self.plan, cells)

    def test_10_wrong_context_prefix_is_rejected(self) -> None:
        changed = deepcopy(self.plan)
        changed["rows"][0]["context_prefix_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_plan(changed, self.bundle)

    def test_11_wrong_source_identity_is_rejected(self) -> None:
        changed = deepcopy(self.plan)
        changed["rows"][0]["source_text_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_plan(changed, self.bundle)

    def test_12_duplicate_execution_row_is_rejected(self) -> None:
        changed = deepcopy(self.plan)
        changed["rows"][1] = deepcopy(changed["rows"][0])
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_plan(changed, self.bundle)

    def test_13_missing_frozen_cell_is_incomplete(self) -> None:
        with self.assertRaisesRegex(ValueError, "112"):
            build_aggregate(
                output_root=self.root,
                bundle=self.bundle,
                plan=self.plan,
                cells=self.cells()[:-1],
            )

    def test_14_extra_non_frozen_output_is_rejected(self) -> None:
        output = self.root / "extra-output"
        extra = output / "runs" / "qasper" / "wrong" / "1024" / BASE_FAMILY
        extra.mkdir(parents=True)
        write_json(extra / "manifest.json", {})
        with self.assertRaisesRegex(ValueError, "extra non-frozen"):
            validate_no_extra_outputs(output, self.plan)

    def test_15_completed_matching_status_is_detected_for_resume(self) -> None:
        output = self.root / "resume-complete"
        row = self.plan["rows"][0]
        result = output / row["result_relative_dir"]
        status = {
            "schema_version": "1.0",
            "phase": "2D.2A-H2.1",
            "freeze_content_sha256": self.freeze_sha,
            "execution_id": row["execution_id"],
            "frozen_matrix_row_identity": matrix_identity(row),
            "repository_commit": "a" * 40,
            "execution_start_utc": "start",
            "execution_end_utc": "end",
            "execution_start_status": "started",
            "execution_end_status": "completed",
            "runner_return_code": 0,
            "external_phase2d2_checker_return_code": 0,
            "result_identity_valid": True,
            "result_identity_error": None,
            "command": ["python"],
        }
        write_json(result / "h21-external-status.json", status)
        self.assertEqual(
            _status_for_row(output, row)["execution_end_status"], "completed"
        )

    def test_16_incomplete_directory_is_not_overwritten(self) -> None:
        output = self.root / "resume-incomplete"
        row = self.plan["rows"][0]
        result = output / row["result_relative_dir"]
        result.mkdir(parents=True)
        marker = result / "partial.txt"
        marker.write_text("keep", encoding="utf-8")
        with patch("scripts.run_phase2d2a_h21_primary.subprocess.run") as run:
            outcome = _execute_row(output, row, {})
        self.assertEqual(outcome, "left_incomplete")
        self.assertEqual(marker.read_text(), "keep")
        run.assert_not_called()

    def test_17_q1_candidate_labels_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "q1"):
            _candidate_metrics(
                [{"candidate_quality": _quality(q1=True)}], Path("fixture")
            )

    def test_18_original_dflash_label_authority_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "authority"):
            _candidate_metrics(
                [{"candidate_quality": _quality(authority="target_only")}],
                Path("fixture"),
            )

    def test_19_correctness_invalid_is_excluded_from_benefit(self) -> None:
        cells = self.cells()
        target_index = next(
            index
            for index, row in enumerate(cells)
            if row["configuration_family"] != BASE_FAMILY
        )
        cells[target_index] = _cell(self.plan["rows"][target_index], valid=False)
        pair = next(
            row
            for row in pair_cells(self.plan, cells)
            if row["refined_family"] == cells[target_index]["configuration_family"]
            and row["sample_id"] == cells[target_index]["sample_id"]
            and row["nominal_prompt_length"]
            == cells[target_index]["nominal_prompt_length"]
        )
        self.assertFalse(pair["correctness_valid"])
        self.assertIsNone(
            pair["candidate_quality"]["selected_tree_coverage_rate_at_fixed_n"][
                "absolute_gain"
            ]
        )

    def test_20_unavailable_positions_do_not_enter_denominator(self) -> None:
        metrics = _candidate_metrics(
            [
                {"candidate_quality": _quality()},
                {"candidate_quality": _quality(available=False)},
            ],
            Path("fixture"),
        )
        self.assertEqual(metrics["labelled_reference_position_count"], 1)
        self.assertEqual(metrics["selected_tree_coverage_rate_at_fixed_n"], 1.0)

    def test_21_task_metric_uses_sum_accepted_over_sum_verified(self) -> None:
        pairs = pair_cells(self.plan, self.cells())
        row = next(
            row
            for row in task_context_summaries(pairs)
            if row["task"] == "hotpotqa" and row["nominal_prompt_length"] == 1024
        )
        metric = row["accepted_per_verified"]
        self.assertEqual(
            metric["refined_rate"],
            metric["refined_accepted_draft_token_count"]
            / metric["refined_actual_verified_node_count"],
        )

    def test_22_mean_of_ratios_cannot_replace_pooled_metric(self) -> None:
        cells = self.cells()
        candidates = [
            row
            for row in cells
            if row["task"] == "hotpotqa"
            and row["nominal_prompt_length"] == 1024
            and row["configuration_family"] == FROZEN_REFINEMENT_FAMILIES[0]
        ]
        candidates[0]["online_efficiency"]["accepted_draft_token_count"] = 1
        candidates[0]["online_efficiency"]["verified_target_node_count"] = 10
        candidates[0]["online_efficiency"][
            "accepted_draft_tokens_per_verified_node"
        ] = 0.1
        candidates[1]["online_efficiency"]["accepted_draft_token_count"] = 90
        candidates[1]["online_efficiency"]["verified_target_node_count"] = 1000
        candidates[1]["online_efficiency"][
            "accepted_draft_tokens_per_verified_node"
        ] = 0.09
        row = next(
            row
            for row in task_context_summaries(pair_cells(self.plan, cells))
            if row["task"] == "hotpotqa"
            and row["nominal_prompt_length"] == 1024
            and row["refined_family"] == FROZEN_REFINEMENT_FAMILIES[0]
        )
        self.assertFalse(
            row["accepted_per_verified"]["mean_of_sample_ratios_used_as_primary"]
        )
        self.assertNotEqual(row["accepted_per_verified"]["refined_rate"], 0.095)

    def test_23_qasper_hotpot_remain_separate_strata(self) -> None:
        rows = task_context_summaries(pair_cells(self.plan, self.cells()))
        independent = {
            (row["task"], row["nominal_prompt_length"])
            for row in rows
            if row["task"] != "gov_report"
        }
        self.assertEqual(
            independent,
            {
                ("qasper", 1024),
                ("qasper", 8192),
                ("hotpotqa", 1024),
                ("hotpotqa", 8192),
            },
        )

    def test_24_govreport_does_not_affect_cross_task_stability(self) -> None:
        cells = self.cells()
        for index, row in enumerate(cells):
            if (
                row["task"] == "gov_report"
                and row["configuration_family"] != BASE_FAMILY
            ):
                cells[index] = _cell(self.plan["rows"][index], positive=False)
        families = family_summaries(
            task_context_summaries(pair_cells(self.plan, cells))
        )
        self.assertTrue(
            all(
                row["cross_task_classification"] == "stable_cross_task_same_work_signal"
                for row in families
            )
        )

    def test_25_stable_requires_all_four_independent_strata_positive(self) -> None:
        families = family_summaries(
            task_context_summaries(pair_cells(self.plan, self.cells()))
        )
        self.assertTrue(
            all(
                row["cross_task_classification"] == "stable_cross_task_same_work_signal"
                for row in families
            )
        )

    def test_26_one_negative_stratum_prevents_stable(self) -> None:
        cells = self.cells()
        family = FROZEN_REFINEMENT_FAMILIES[0]
        for index, row in enumerate(cells):
            if (
                row["task"] == "qasper"
                and row["nominal_prompt_length"] == 8192
                and row["configuration_family"] == family
            ):
                cells[index] = _cell(self.plan["rows"][index], positive=False)
        summary = next(
            row
            for row in family_summaries(
                task_context_summaries(pair_cells(self.plan, cells))
            )
            if row["refined_family"] == family
        )
        self.assertEqual(
            summary["cross_task_classification"], "mixed_cross_task_same_work_signal"
        )

    def test_27_per_sample_heterogeneity_is_retained(self) -> None:
        row = task_context_summaries(pair_cells(self.plan, self.cells()))[0]
        self.assertIn("sample_heterogeneity", row)
        self.assertIn("joint_signal_counts", row["sample_heterogeneity"])

    def test_28_h1_refined_only_zero_is_not_rewritten(self) -> None:
        rows = mechanism_summaries(pair_cells(self.plan, self.cells()))
        self.assertTrue(
            all(row["h1_refined_only_zero_rewritten"] is False for row in rows)
        )

    def test_29_cost_timers_are_not_double_counted(self) -> None:
        rows = cost_summaries(pair_cells(self.plan, self.cells()))
        self.assertTrue(
            all(row["nested_timer_double_counting_avoided"] for row in rows)
        )
        self.assertTrue(
            all(
                not row["always_fixed_cost_attributed_to_conditional_remask"]
                for row in rows
            )
        )

    def test_30_no_final_branch_decision_is_issued(self) -> None:
        primary = build_aggregate(
            output_root=self.root,
            bundle=self.bundle,
            plan=self.plan,
            cells=self.cells(),
        )["primary"]
        self.assertFalse(primary["winner_selected"])
        self.assertFalse(primary["final_phase2d2a_branch_entered"])

    def test_31_online_and_instrumented_sources_remain_unchanged(self) -> None:
        expected = {
            "third_party/qwen3_8b_dflash_b16/instrumented/dflash.py": "6b20d1476af5746ffa511df3441230ee25cdc5f25f25d98b7eae8f72dbf8e533",
            "dflash/phase2d2/refinement.py": "9c4e5acd20d4076332841d15277a54b821c2f3b88007cea53ff90410f517b9bc",
        }
        for relative, digest in expected.items():
            self.assertEqual(
                hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest(), digest
            )

    def test_32_h20_freeze_remains_byte_identical(self) -> None:
        before = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.freeze_dir.iterdir()
            if path.is_file()
        }
        build_execution_plan(self.bundle)
        after = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.freeze_dir.iterdir()
            if path.is_file()
        }
        self.assertEqual(before, after)

    def test_33_all_json_artifacts_pass_schema(self) -> None:
        aggregate = build_aggregate(
            output_root=self.root,
            bundle=self.bundle,
            plan=self.plan,
            cells=self.cells(),
        )
        output = self.root / "schema-output"
        persist_aggregate(output, aggregate)
        for name in AGGREGATE_JSON_ARTIFACTS:
            validate_h21_schema(__import__("json").loads((output / name).read_text()))

    def test_34_runner_command_pairs_exact_base_context(self) -> None:
        refined = next(
            row
            for row in self.plan["rows"]
            if row["configuration_family"] != BASE_FAMILY
        )
        paths = {
            "python_executable": "/python",
            "target_model_path": "/target",
            "materialized_draft_path": "/draft",
            "longbench_root": "/data",
            "frozen_selector_config": "/selector.json",
            "frozen_refinement_config": "/refinement.json",
        }
        command = _runner_command(refined, output_root=Path("/output"), paths=paths)
        base = command[command.index("--base-result-dir") + 1]
        self.assertTrue(base.endswith(f"/{refined['nominal_prompt_length']}/base_only"))
        self.assertIn(refined["sample_id"], base)

    def test_35_sentinel_is_one_frozen_base_refined_pair(self) -> None:
        rows = _sentinel_rows(self.plan)
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {row["configuration_family"] for row in rows},
            {BASE_FAMILY, FROZEN_REFINEMENT_FAMILIES[0]},
        )
        self.assertEqual(
            len(
                {
                    (row["task"], row["sample_id"], row["nominal_prompt_length"])
                    for row in rows
                }
            ),
            1,
        )

    def test_36_manifest_detects_h20_byte_mutation(self) -> None:
        manifest = build_execution_manifest(
            self.bundle, plan=self.plan, runtime_paths={}, execution_authorized=False
        )
        changed = deepcopy(self.bundle)
        changed["freeze_artifact_file_sha256"] = dict(
            changed["freeze_artifact_file_sha256"]
        )
        changed["freeze_artifact_file_sha256"][
            next(iter(changed["freeze_artifact_file_sha256"]))
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "changed"):
            ensure_immutable_freeze(manifest, changed)

    def test_37_result_paths_are_collision_free(self) -> None:
        paths = [result_relative_dir(row) for row in self.plan["rows"]]
        self.assertEqual(len(paths), len(set(paths)))

    def test_38_mixed_repository_provenance_is_rejected(self) -> None:
        cells = self.cells()
        cells[-1]["provenance"]["repository_commit"] = "changed"
        with self.assertRaisesRegex(ValueError, "repository_commit"):
            validate_cell_provenance(cells)

    def test_39_venv_python_symlink_is_not_dereferenced(self) -> None:
        root = self.root / "python-symlink"
        root.mkdir()
        entry = root / "python"
        entry.symlink_to(Path(sys.executable).resolve())
        normalized = _absolute_without_symlink_dereference(entry)
        self.assertEqual(normalized, os.path.abspath(entry))
        self.assertTrue(Path(normalized).is_symlink())
        self.assertNotEqual(normalized, os.path.realpath(entry))

    def test_40_runtime_python_uses_abspath_not_realpath(self) -> None:
        entry = self.root / "python-symlink" / "python"
        args = Namespace(
            target_model_path=None,
            materialized_draft_path=None,
            longbench_root=None,
            frozen_selector_config=None,
            frozen_refinement_config=None,
            python_executable=str(entry.parent / ".." / entry.parent.name / entry.name),
        )
        observed = _runtime_paths(args)["python_executable"]
        self.assertEqual(observed, os.path.abspath(args.python_executable))
        self.assertNotEqual(observed, os.path.realpath(args.python_executable))

    def test_41_child_sees_package_installed_only_in_virtualenv(self) -> None:
        environment = self.root / "child-only-venv"
        venv.EnvBuilder(with_pip=False).create(environment)
        child = environment / "bin" / "python"
        site_packages = Path(
            subprocess.check_output(
                [
                    str(child),
                    "-c",
                    "import site; print(site.getsitepackages()[0])",
                ],
                text=True,
            ).strip()
        )
        module_name = "h21_child_only_fixture"
        (site_packages / f"{module_name}.py").write_text(
            "__version__ = 'fixture-only'\n", encoding="utf-8"
        )
        probe = _probe_child_environment(str(child), (module_name,))
        self.assertEqual(probe["status"], "passed")
        self.assertTrue(probe["required_imports"][module_name]["available"])
        self.assertEqual(
            probe["required_imports"][module_name]["version"], "fixture-only"
        )

    def test_42_child_environment_preflight_passes_valid_interpreter(self) -> None:
        paths = {"python_executable": sys.executable}
        with patch(
            "scripts.run_phase2d2a_h21_primary.REQUIRED_CHILD_IMPORTS", ("json",)
        ):
            probe = _child_environment_preflight(paths)
        self.assertEqual(probe["status"], "passed")
        self.assertTrue(probe["required_imports"]["json"]["available"])
        self.assertEqual(probe["configured_child_executable"], sys.executable)

    def test_43_missing_child_import_fails_preflight(self) -> None:
        missing = "h21_package_that_must_not_exist"
        with patch(
            "scripts.run_phase2d2a_h21_primary.REQUIRED_CHILD_IMPORTS", (missing,)
        ):
            probe = _child_environment_preflight({"python_executable": sys.executable})
        self.assertEqual(probe["status"], "failed")
        self.assertFalse(probe["required_imports"][missing]["available"])

    def test_44_preflight_failure_creates_zero_scientific_directories(self) -> None:
        runtime = self.root / "preflight-runtime"
        for name in ("target", "draft", "longbench"):
            (runtime / name).mkdir(parents=True)
        for name in ("selector.json", "refinement.json"):
            (runtime / name).write_text("{}\n", encoding="utf-8")
        output = self.root / "preflight-failure-output"
        failed_probe = {
            "status": "failed",
            "parent_sys_executable": sys.executable,
            "configured_child_executable": sys.executable,
            "configured_child_is_symlink": Path(sys.executable).is_symlink(),
            "configured_child_realpath": os.path.realpath(sys.executable),
            "configured_child_abspath": os.path.abspath(sys.executable),
            "symlink_preserved": True,
            "child_sys_executable": sys.executable,
            "child_python_version": None,
            "required_imports": {
                "torch": {
                    "available": False,
                    "error": "ModuleNotFoundError",
                    "version": None,
                }
            },
            "required_import_status": False,
            "return_code": 1,
            "probe_error": None,
            "command": [sys.executable, "-c", "probe"],
            "stdout": "",
            "stderr": "",
        }
        argv = [
            "--freeze-dir",
            str(self.freeze_dir),
            "--freeze-content-sha256",
            self.freeze_sha,
            "--output-root",
            str(output),
            "--target-model-path",
            str(runtime / "target"),
            "--materialized-draft-path",
            str(runtime / "draft"),
            "--longbench-root",
            str(runtime / "longbench"),
            "--frozen-selector-config",
            str(runtime / "selector.json"),
            "--frozen-refinement-config",
            str(runtime / "refinement.json"),
            "--execute-frozen-primary",
        ]
        with (
            patch(
                "scripts.run_phase2d2a_h21_primary.validate_h21_freeze",
                return_value=self.bundle,
            ),
            patch(
                "scripts.run_phase2d2a_h21_primary._child_environment_preflight",
                return_value=failed_probe,
            ),
            patch("scripts.run_phase2d2a_h21_primary._execute_row") as execute,
        ):
            return_code = run_h21_main(argv)
        self.assertEqual(return_code, 1)
        execute.assert_not_called()
        self.assertFalse((output / "runs").exists())
        index = [
            json.loads(line)
            for line in (output / "phase2d2a-h21-subrun-index.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(
            {row["execution_end_status"] for row in index}, {"not_started"}
        )
        manifest = json.loads(
            (output / "phase2d2a-h21-execution-manifest.json").read_text()
        )
        self.assertEqual(manifest["status"], "infrastructure_failure")
        self.assertFalse(manifest["model_forward_executed"])

    def test_45_sentinel_infrastructure_failure_returns_nonzero(self) -> None:
        row = self.plan["rows"][0]
        status = {
            "execution_end_status": "infrastructure_failed",
            "runner_return_code": 1,
            "external_phase2d2_checker_return_code": None,
            "result_identity_valid": False,
        }
        gate = _sentinel_cell_gate(
            row,
            status,
            complete_required_artifacts=False,
            cell_status="infrastructure_failed",
            infrastructure_failure_reason="fixture identity failure",
        )
        self.assertEqual(_sentinel_gate_return_code({"passed": gate["passed"]}), 1)
        self.assertIn(
            "cell_status_infrastructure_failed:fixture identity failure",
            gate["failure_reasons"],
        )
        self.assertNotIn("hard_correctness_not_accepted", gate["failure_reasons"])

    def test_46_sentinel_scientific_failure_returns_nonzero(self) -> None:
        row = self.plan["rows"][0]
        status = {
            "execution_end_status": "completed",
            "runner_return_code": 0,
            "external_phase2d2_checker_return_code": 1,
            "result_identity_valid": True,
        }
        gate = _sentinel_cell_gate(
            row,
            status,
            complete_required_artifacts=True,
            cell_status="correctness_invalid",
        )
        self.assertEqual(_sentinel_gate_return_code({"passed": gate["passed"]}), 1)
        self.assertIn("hard_correctness_not_accepted", gate["failure_reasons"])

    def test_47_sentinel_valid_pair_returns_zero(self) -> None:
        status = {
            "execution_end_status": "completed",
            "runner_return_code": 0,
            "external_phase2d2_checker_return_code": 0,
            "result_identity_valid": True,
        }
        gates = [
            _sentinel_cell_gate(
                row,
                status,
                complete_required_artifacts=True,
                cell_status="valid",
            )
            for row in _sentinel_rows(self.plan)
        ]
        self.assertTrue(all(gate["passed"] for gate in gates))
        self.assertEqual(_sentinel_gate_return_code({"passed": True}), 0)

    def test_48_full_execution_circuits_on_first_infrastructure_failure(self) -> None:
        rows = self.plan["rows"][:4]
        scheduled = []

        def execute(row: dict) -> str:
            scheduled.append(row["execution_id"])
            return "infrastructure_failed"

        circuit = _run_rows_with_circuit_breaker(rows, execute)
        self.assertTrue(circuit["triggered"])
        self.assertEqual(circuit["scheduled_row_count"], 1)
        self.assertEqual(scheduled, [rows[0]["execution_id"]])

    def test_49_untouched_rows_remain_not_started_after_circuit_break(self) -> None:
        output = self.root / "circuit-index"
        row = self.plan["rows"][0]
        status = {
            "schema_version": "1.0",
            "phase": "2D.2A-H2.1",
            "freeze_content_sha256": self.freeze_sha,
            "execution_id": row["execution_id"],
            "frozen_matrix_row_identity": matrix_identity(row),
            "repository_commit": "a" * 40,
            "execution_start_utc": "start",
            "execution_end_utc": "end",
            "execution_start_status": "started",
            "execution_end_status": "infrastructure_failed",
            "runner_return_code": 1,
            "external_phase2d2_checker_return_code": None,
            "result_identity_valid": False,
            "result_identity_error": "missing dependency",
            "command": ["python"],
        }
        write_json(
            output / row["result_relative_dir"] / "h21-external-status.json",
            status,
        )
        index = _index_rows(output, self.plan)
        self.assertEqual(index[0]["execution_end_status"], "infrastructure_failed")
        self.assertEqual(
            sum(item["execution_end_status"] == "not_started" for item in index),
            111,
        )

    def test_50_scientific_invalid_does_not_trigger_infrastructure_circuit(
        self,
    ) -> None:
        rows = self.plan["rows"][:4]
        circuit = _run_rows_with_circuit_breaker(rows, lambda _row: "completed")
        self.assertFalse(circuit["triggered"])
        self.assertEqual(circuit["scheduled_row_count"], len(rows))

    def test_51_sentinel_schema_records_hard_gate_evidence(self) -> None:
        rows = _sentinel_rows(self.plan)
        payload = {
            "schema_version": "1.0",
            "phase": "2D.2A-H2.1",
            "artifact_type": "sentinel_gate",
            "status": "complete",
            "freeze_content_sha256": self.freeze_sha,
            "passed": False,
            "requested_execution_ids": [row["execution_id"] for row in rows],
            "cell_gates": [{}, {}],
            "failure_reasons": ["fixture"],
        }
        validate_h21_schema(payload)

    def test_52_refined_tree_category_is_not_runtime_algorithmic_mode(self) -> None:
        frozen = _frozen_refinement()
        family = "conditional_rerank-d1-p1-r2-w0.5"
        row = next(
            item for item in self.plan["rows"] if item["configuration_family"] == family
        )
        phase = _runtime_phase(frozen, family)
        self.assertEqual(row["candidate_merge_mode"], "refined_tree")
        self.assertEqual(phase["candidate_merge_mode"], "conditional_rerank")
        _validate_runtime_refinement_identity(
            Path("fixture"),
            row,
            phase,
            {"candidate_merge_mode": "conditional_rerank"},
            frozen,
        )

    def test_53_base_only_runtime_identity_validates(self) -> None:
        row = next(
            item
            for item in self.plan["rows"]
            if item["configuration_family"] == BASE_FAMILY
        )
        phase = {
            "candidate_merge_mode": BASE_FAMILY,
            "frozen_refinement_configuration_id": None,
            "frozen_refinement_config_hash": None,
        }
        _validate_runtime_refinement_identity(
            Path("fixture"),
            row,
            phase,
            {"candidate_merge_mode": BASE_FAMILY},
            None,
        )

    def test_54_conditional_rerank_runtime_identity_validates(self) -> None:
        self._assert_refined_runtime_identity_validates(
            "conditional_rerank-d1-p1-r2-w0.5", "conditional_rerank"
        )

    def test_55_union_d2_runtime_identity_validates(self) -> None:
        self._assert_refined_runtime_identity_validates("union-d2-p2-r2-w0.5", "union")

    def test_56_union_d4_runtime_identity_validates(self) -> None:
        self._assert_refined_runtime_identity_validates("union-d4-p2-r4-w0.5", "union")

    def _assert_refined_runtime_identity_validates(
        self, family: str, expected_mode: str
    ) -> None:
        frozen = _frozen_refinement()
        row = next(
            item for item in self.plan["rows"] if item["configuration_family"] == family
        )
        phase = _runtime_phase(frozen, family)
        self.assertEqual(phase["candidate_merge_mode"], expected_mode)
        _validate_runtime_refinement_identity(
            Path("fixture"),
            row,
            phase,
            {"candidate_merge_mode": expected_mode},
            frozen,
        )

    def test_57_wrong_runtime_merge_mode_is_rejected(self) -> None:
        frozen = _frozen_refinement()
        family = "union-d2-p2-r2-w0.5"
        row = next(
            item for item in self.plan["rows"] if item["configuration_family"] == family
        )
        phase = _runtime_phase(frozen, family)
        phase["candidate_merge_mode"] = "conditional_rerank"
        with self.assertRaisesRegex(ValueError, "runtime candidate merge mode"):
            _validate_runtime_refinement_identity(
                Path("fixture"),
                row,
                phase,
                {"candidate_merge_mode": "conditional_rerank"},
                frozen,
            )

    def test_58_wrong_frozen_refinement_family_id_is_rejected(self) -> None:
        frozen = _frozen_refinement()
        family = "union-d2-p2-r2-w0.5"
        row = next(
            item for item in self.plan["rows"] if item["configuration_family"] == family
        )
        phase = _runtime_phase(frozen, family)
        phase["frozen_refinement_configuration_id"] = "union-d4-p2-r4-w0.5"
        with self.assertRaisesRegex(ValueError, "configuration ID differs"):
            _validate_runtime_refinement_identity(
                Path("fixture"),
                row,
                phase,
                {"candidate_merge_mode": "union"},
                frozen,
            )

    def test_59_wrong_frozen_refinement_hash_is_rejected(self) -> None:
        frozen = _frozen_refinement()
        family = "union-d4-p2-r4-w0.5"
        row = next(
            item for item in self.plan["rows"] if item["configuration_family"] == family
        )
        phase = _runtime_phase(frozen, family)
        phase["frozen_refinement_config_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "config hash differs"):
            _validate_runtime_refinement_identity(
                Path("fixture"),
                row,
                phase,
                {"candidate_merge_mode": "union"},
                frozen,
            )

    def test_60_base_refined_runtime_confusion_is_rejected(self) -> None:
        frozen = _frozen_refinement()
        base = next(
            item
            for item in self.plan["rows"]
            if item["configuration_family"] == BASE_FAMILY
        )
        family = "conditional_rerank-d1-p1-r2-w0.5"
        refined = next(
            item for item in self.plan["rows"] if item["configuration_family"] == family
        )
        with self.subTest("base pretending to be refined"):
            phase = _runtime_phase(frozen, family)
            with self.assertRaisesRegex(ValueError, "base result"):
                _validate_runtime_refinement_identity(
                    Path("fixture"),
                    base,
                    phase,
                    {"candidate_merge_mode": "conditional_rerank"},
                    frozen,
                )
        with self.subTest("refined pretending to be base"):
            phase = {
                "candidate_merge_mode": BASE_FAMILY,
                "frozen_refinement_configuration_id": None,
                "frozen_refinement_config_hash": None,
            }
            with self.assertRaisesRegex(ValueError, "configuration ID differs"):
                _validate_runtime_refinement_identity(
                    Path("fixture"),
                    refined,
                    phase,
                    {"candidate_merge_mode": BASE_FAMILY},
                    frozen,
                )

    def test_61_known_h100_scientific_pass_and_runtime_identity_passes_gate(
        self,
    ) -> None:
        frozen = _frozen_refinement()
        family = "conditional_rerank-d1-p1-r2-w0.5"
        row = next(
            item for item in self.plan["rows"] if item["configuration_family"] == family
        )
        phase = _runtime_phase(frozen, family)
        _validate_runtime_refinement_identity(
            Path("fixture"),
            row,
            phase,
            {"candidate_merge_mode": "conditional_rerank"},
            frozen,
        )
        gate = _sentinel_cell_gate(
            row,
            _external_status(row),
            complete_required_artifacts=True,
            cell_status="valid",
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["canonical_cell_status"], "valid")

    def test_62_canonical_infrastructure_counter_matches_loader(self) -> None:
        output = self.root / "canonical-infrastructure-counter"
        rows = _sentinel_rows(self.plan)
        for row in rows:
            write_json(
                output / row["result_relative_dir"] / "h21-external-status.json",
                _external_status(row),
            )
        manifest = build_execution_manifest(
            self.bundle,
            plan=self.plan,
            runtime_paths={},
            execution_authorized=True,
        )
        with patch(
            "scripts.run_phase2d2a_h21_primary.load_cell",
            side_effect=(
                {"cell_status": "valid", "infrastructure_failure_reason": None},
                {
                    "cell_status": "infrastructure_failed",
                    "infrastructure_failure_reason": "fixture provenance mismatch",
                },
            ),
        ):
            _refresh_index_and_manifest(
                output,
                self.plan,
                manifest,
                attempted=True,
                canonical=True,
            )
        self.assertEqual(manifest["completed_cell_count"], 1)
        self.assertEqual(manifest["infrastructure_failed_cell_count"], 1)
        self.assertEqual(manifest["scientific_correctness_invalid_cell_count"], 0)
        self.assertEqual(manifest["not_started_cell_count"], 110)

    def test_63_canonical_scientific_invalid_counter_remains_distinct(self) -> None:
        output = self.root / "canonical-scientific-counter"
        rows = _sentinel_rows(self.plan)
        for row in rows:
            write_json(
                output / row["result_relative_dir"] / "h21-external-status.json",
                _external_status(row),
            )
        manifest = build_execution_manifest(
            self.bundle,
            plan=self.plan,
            runtime_paths={},
            execution_authorized=True,
        )
        with patch(
            "scripts.run_phase2d2a_h21_primary.load_cell",
            side_effect=(
                {"cell_status": "valid", "infrastructure_failure_reason": None},
                {
                    "cell_status": "correctness_invalid",
                    "infrastructure_failure_reason": None,
                },
            ),
        ):
            _refresh_index_and_manifest(
                output,
                self.plan,
                manifest,
                attempted=True,
                canonical=True,
            )
        self.assertEqual(manifest["completed_cell_count"], 2)
        self.assertEqual(manifest["infrastructure_failed_cell_count"], 0)
        self.assertEqual(manifest["scientific_correctness_invalid_cell_count"], 1)
        self.assertEqual(manifest["not_started_cell_count"], 110)


if __name__ == "__main__":
    unittest.main()
