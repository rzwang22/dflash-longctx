from __future__ import annotations

import hashlib
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from jsonschema import ValidationError

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
)
from scripts.run_phase2d2a_h21_primary import (
    _execute_row,
    _runner_command,
    _sentinel_rows,
    _status_for_row,
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


if __name__ == "__main__":
    unittest.main()
