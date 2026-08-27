from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from scripts.freeze_phase2d2a_h2_cohort import (
    BASE_FAMILY,
    FAMILIES,
    FROZEN_REFINEMENT_FAMILIES,
    persist_freeze,
)
from scripts.phase2d2a_h21_common import (
    build_execution_plan as build_h21_execution_plan,
)
from scripts.phase2d2a_h21_common import write_json
from scripts.phase2d2a_h22_common import (
    ACTUAL_WORK_REGIME,
    EXPECTED_PAIR_COUNT,
    EXPECTED_ROW_COUNT,
    build_aggregate,
    build_analysis_contract,
    build_execution_manifest,
    build_execution_plan,
    cross_budget_pairs,
    ensure_immutable_sources,
    incremental_budget_decomposition,
    validate_h22_freeze,
    validate_h22_schema,
    verification_work,
    within_n32_pairs,
)
from scripts.run_phase2d2a_h21_primary import (
    _child_environment_preflight as h21_child_environment_preflight,
)
from scripts.run_phase2d2a_h21_primary import _run_rows_with_circuit_breaker
from scripts.run_phase2d2a_h22_secondary import (
    _child_environment_preflight as h22_child_environment_preflight,
)
from scripts.run_phase2d2a_h22_secondary import (
    _execute_row,
    _sentinel_rows,
    build_parser,
)
from tests.test_phase2d2_h21 import _cell, _freeze_artifacts

REPO_ROOT = Path(__file__).resolve().parents[1]


def _bindings() -> dict:
    h21_hashes = {"anchor.json": "1" * 64}
    h211_hashes = {"route.json": "2" * 64}
    return {
        "h21_output_root": "/h21",
        "h211_output_dir": "/h211",
        "h21_acceptance": {"status": "accepted"},
        "h211_acceptance": {"status": "accepted"},
        "h21_artifact_file_sha256": h21_hashes,
        "h211_artifact_file_sha256": h211_hashes,
        "h21_anchor_binding_sha256": "3" * 64,
        "h211_route_binding_sha256": "4" * 64,
        "source_binding_sha256": "5" * 64,
    }


def _measured_cell(
    row: dict,
    *,
    actual_nodes_per_round: int,
    n32_progress_gain: bool = False,
    valid: bool = True,
    infrastructure: bool = False,
) -> dict:
    cell = _cell(row, valid=valid, infrastructure=infrastructure)
    if infrastructure:
        cell["verification_work"] = None
        cell["generated_output_token_ids"] = None
        return cell
    rounds = 2
    accepted = int(cell["online_efficiency"]["accepted_draft_token_count"])
    committed = int(cell["online_efficiency"]["committed_token_count"])
    if n32_progress_gain and row["configuration_family"] != BASE_FAMILY:
        accepted += 4
        committed += 4
    verified = actual_nodes_per_round * rounds
    cell["online_efficiency"].update(
        {
            "accepted_draft_token_count": accepted,
            "committed_token_count": committed,
            "verified_target_node_count": verified,
            "accepted_draft_tokens_per_verified_node": accepted / verified,
            "accepted_draft_tokens_per_round": accepted / rounds,
            "committed_tokens_per_round": committed / rounds,
            "verified_target_nodes_per_returned_token": verified / 20,
        }
    )
    cell["verification_work"] = verification_work(
        [
            {
                "phase2d2": {
                    "actual_verified_node_count": actual_nodes_per_round,
                    "selected_tree_node_count": actual_nodes_per_round,
                }
            }
            for _ in range(rounds)
        ]
    )
    cell["generated_output_token_ids"] = [1, 2, 3]
    return cell


class Phase2D2H22Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.freeze_dir = cls.root / "freeze"
        cls.freeze_artifacts = _freeze_artifacts()
        persist_freeze(cls.freeze_dir, cls.freeze_artifacts)
        cls.freeze_sha = cls.freeze_artifacts["manifest"]["freeze_content_sha256"]
        cls.bundle = validate_h22_freeze(
            cls.freeze_dir,
            cls.freeze_sha,
            required_freeze_sha256=cls.freeze_sha,
        )
        cls.bindings = _bindings()
        cls.plan = build_execution_plan(cls.bundle, cls.bindings)
        cls.h21_plan = build_h21_execution_plan(cls.bundle)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def cells(self) -> tuple[list[dict], list[dict]]:
        h21 = [
            _measured_cell(row, actual_nodes_per_round=16)
            for row in self.h21_plan["rows"]
        ]
        h22 = [
            _measured_cell(
                row,
                actual_nodes_per_round=(
                    16 if row["configuration_family"] == BASE_FAMILY else 20
                ),
                n32_progress_gain=True,
            )
            for row in self.plan["rows"]
        ]
        return h21, h22

    def test_01_secondary_plan_is_exact_112_n32_rows(self) -> None:
        self.assertEqual(self.plan["row_count"], EXPECTED_ROW_COUNT)
        self.assertEqual(len(self.plan["rows"]), EXPECTED_ROW_COUNT)
        self.assertTrue(all(row["node_budget"] == 32 for row in self.plan["rows"]))
        self.assertTrue(
            all(row["actual_work_regime"] == ACTUAL_WORK_REGIME for row in [self.plan])
        )

    def test_02_plan_uses_exact_frozen_secondary_cohort(self) -> None:
        frozen = {
            (
                row["task"],
                row["sample_id"],
                row["nominal_prompt_length"],
                row["configuration_family"],
            )
            for row in self.bundle["secondary"]["rows"]
        }
        planned = {
            (
                row["task"],
                row["sample_id"],
                row["nominal_prompt_length"],
                row["configuration_family"],
            )
            for row in self.plan["rows"]
        }
        self.assertEqual(planned, frozen)
        self.assertEqual(
            {row["configuration_family"] for row in self.plan["rows"]},
            set(FAMILIES),
        )

    def test_03_plan_only_contract_forbids_model_forward_and_rescue(self) -> None:
        contract = build_analysis_contract(self.bundle, self.bindings)
        self.assertFalse(contract["model_forward_executed"])
        self.assertFalse(contract["h22_can_override_h21_primary"])
        self.assertFalse(contract["outcome_dependent_thresholds_allowed"])
        self.assertFalse(contract["arbitrary_low_marginal_yield_threshold_defined"])
        self.assertIsNone(
            build_parser()
            .parse_args(
                [
                    "--freeze-dir",
                    "/freeze",
                    "--freeze-content-sha256",
                    "f" * 64,
                    "--h21-output-root",
                    "/h21",
                    "--h211-output-dir",
                    "/h211",
                    "--output-root",
                    "/output",
                ]
            )
            .target_model_path
        )

    def test_04_child_preflight_is_the_qualified_h21_implementation(self) -> None:
        self.assertIs(h22_child_environment_preflight, h21_child_environment_preflight)
        with patch(
            "scripts.run_phase2d2a_h21_primary.REQUIRED_CHILD_IMPORTS", ("json",)
        ):
            probe = h22_child_environment_preflight(
                {"python_executable": sys.executable}
            )
        self.assertEqual(probe["status"], "passed")

    def test_05_sentinel_is_exact_first_qasper_1k_pair(self) -> None:
        rows = _sentinel_rows(self.plan)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["task"] for row in rows}, {"qasper"})
        self.assertEqual({row["nominal_prompt_length"] for row in rows}, {1024})
        self.assertEqual(
            {row["configuration_family"] for row in rows},
            {BASE_FAMILY, FROZEN_REFINEMENT_FAMILIES[0]},
        )
        self.assertEqual(len({row["sample_id"] for row in rows}), 1)

    def test_06_safe_resume_skips_matching_completed_row(self) -> None:
        row = self.plan["rows"][0]
        output = self.root / "resume"
        result = output / row["result_relative_dir"]
        result.mkdir(parents=True)
        write_json(
            result / "h22-external-status.json",
            {
                "schema_version": "1.0",
                "phase": "2D.2A-H2.2",
                "freeze_content_sha256": row["freeze_content_sha256"],
                "h21_anchor_binding_sha256": self.bindings["h21_anchor_binding_sha256"],
                "h211_route_binding_sha256": self.bindings["h211_route_binding_sha256"],
                "execution_id": row["execution_id"],
                "frozen_matrix_row_identity": {
                    key: row[key]
                    for key in (
                        "matrix_row_index",
                        "execution_id",
                        "task",
                        "task_role",
                        "sample_id",
                        "source_sample_id",
                        "source_text_sha256",
                        "selection_rank",
                        "selection_hash",
                        "nominal_prompt_length",
                        "actual_prompt_length",
                        "context_prefix_sha256",
                        "configuration_family",
                        "frozen_refinement_configuration_id",
                        "node_budget",
                        "verification_mode",
                        "online_selector_mode",
                        "max_new_tokens",
                        "warmup_runs",
                        "timing_repeats",
                        "dtype",
                        "freeze_content_sha256",
                    )
                },
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
            },
        )
        with patch("scripts.run_phase2d2a_h22_secondary.subprocess.run") as run:
            self.assertEqual(
                _execute_row(output, row, {}, self.bindings), "skipped_completed"
            )
        run.assert_not_called()

    def test_07_first_infrastructure_failure_trips_circuit(self) -> None:
        rows = self.plan["rows"][:4]
        circuit = _run_rows_with_circuit_breaker(
            rows, lambda _row: "infrastructure_failed"
        )
        self.assertTrue(circuit["triggered"])
        self.assertEqual(circuit["scheduled_row_count"], 1)

    def test_08_scientific_invalid_is_distinct_and_excluded(self) -> None:
        h21, h22 = self.cells()
        h22[1] = _measured_cell(
            self.plan["rows"][1],
            actual_nodes_per_round=20,
            valid=False,
        )
        aggregate = build_aggregate(
            bundle=self.bundle,
            bindings=self.bindings,
            plan=self.plan,
            h21_cells=h21,
            h22_cells=h22,
        )
        self.assertEqual(aggregate["cell_summary"]["correctness_invalid_cell_count"], 1)
        self.assertEqual(
            aggregate["cell_summary"]["infrastructure_failed_cell_count"], 0
        )
        affected = next(
            row
            for row in aggregate["cross_budget"]["rows"]
            if row["configuration_family"]
            == self.plan["rows"][1]["configuration_family"]
            and row["sample_id"] == self.plan["rows"][1]["sample_id"]
            and row["nominal_prompt_length"]
            == self.plan["rows"][1]["nominal_prompt_length"]
        )
        self.assertFalse(affected["comparison_valid"])

    def test_09_cross_budget_pairing_is_exact_same_identity(self) -> None:
        h21, h22 = self.cells()
        pairs = cross_budget_pairs(h21, h22)
        self.assertEqual(len(pairs), EXPECTED_ROW_COUNT)
        self.assertTrue(all(row["comparison_valid"] for row in pairs))

    def test_10_wrong_family_pairing_is_rejected(self) -> None:
        h21, h22 = self.cells()
        h22[0]["configuration_family"] = FROZEN_REFINEMENT_FAMILIES[0]
        with self.assertRaisesRegex(ValueError, "pairing inventory"):
            cross_budget_pairs(h21, h22)

    def test_11_wrong_sample_pairing_is_rejected(self) -> None:
        h21, h22 = self.cells()
        h22[0]["sample_id"] = "wrong-sample"
        with self.assertRaisesRegex(ValueError, "pairing inventory"):
            cross_budget_pairs(h21, h22)

    def test_12_wrong_context_pairing_is_rejected(self) -> None:
        h21, h22 = self.cells()
        h22[0]["nominal_prompt_length"] = 999
        with self.assertRaisesRegex(ValueError, "pairing inventory"):
            cross_budget_pairs(h21, h22)

    def test_13_within_n32_pair_rejects_source_identity_mismatch(self) -> None:
        _, h22 = self.cells()
        refined_index = next(
            index
            for index, row in enumerate(h22)
            if row["configuration_family"] != BASE_FAMILY
        )
        h22[refined_index]["source_text_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "source_text_sha256"):
            within_n32_pairs(h22)

    def test_14_actual_round_work_drives_marginal_yield(self) -> None:
        h21, h22 = self.cells()
        refined = next(
            row
            for row in cross_budget_pairs(h21, h22)
            if row["configuration_family"] != BASE_FAMILY
        )
        self.assertEqual(refined["delta_actual_verified_nodes_per_round"], 4)
        self.assertEqual(refined["delta_accepted_draft_tokens_per_round"], 2)
        self.assertEqual(refined["accepted_tokens_per_additional_verified_node"], 0.5)
        self.assertFalse(refined["configured_budget_used_as_denominator"])

    def test_15_zero_actual_node_delta_makes_yield_unavailable(self) -> None:
        h21, h22 = self.cells()
        base = next(
            row
            for row in cross_budget_pairs(h21, h22)
            if row["configuration_family"] == BASE_FAMILY
        )
        self.assertEqual(base["delta_actual_verified_nodes_per_round"], 0)
        self.assertIsNone(base["accepted_tokens_per_additional_verified_node"])
        self.assertEqual(
            base["marginal_yield_unavailable_reason"],
            "zero_actual_verified_node_delta",
        )

    def test_16_incremental_budget_decomposition_uses_frozen_formula(self) -> None:
        h21, h22 = self.cells()
        n16_within = within_n32_pairs(h21)
        n32_within = within_n32_pairs(h22)
        row = incremental_budget_decomposition(n16_within, n32_within)[0]
        self.assertEqual(
            row["incremental_budget_contribution_accepted_tokens_per_round"],
            row["combined_gain_n32_accepted_tokens_per_round"]
            - row["same_work_refinement_gain_n16_accepted_tokens_per_round"],
        )
        self.assertEqual(
            row["incremental_actual_work"],
            n32_within[0]["delta_actual_verified_nodes_per_round"]
            - n16_within[0]["delta_actual_verified_nodes_per_round"],
        )
        self.assertFalse(row["causal_estimator_claimed"])

    def test_17_base_n16_n32_sanity_is_descriptive(self) -> None:
        h21, h22 = self.cells()
        conclusion = build_aggregate(
            bundle=self.bundle,
            bindings=self.bindings,
            plan=self.plan,
            h21_cells=h21,
            h22_cells=h22,
        )["conclusion"]
        sanity = conclusion["base_n16_n32_sanity"]
        self.assertTrue(sanity["base_budget_cap_not_consumed"])
        self.assertEqual(len(sanity["rows"]), 28)
        self.assertTrue(
            all("generated_output_ids_equal" in row for row in sanity["rows"])
        )

    def test_18_within_n32_different_work_is_not_called_same_work(self) -> None:
        _, h22 = self.cells()
        pairs = within_n32_pairs(h22)
        self.assertEqual(len(pairs), EXPECTED_PAIR_COUNT)
        self.assertTrue(
            all(
                row["comparison_interpretation"]
                == "combined_refinement_plus_actual_work_difference"
                and not row["called_same_work"]
                for row in pairs
            )
        )

    def test_19_no_primary_rewrite_winner_or_later_branch(self) -> None:
        h21, h22 = self.cells()
        conclusion = build_aggregate(
            bundle=self.bundle,
            bindings=self.bindings,
            plan=self.plan,
            h21_cells=h21,
            h22_cells=h22,
        )["conclusion"]
        for field in (
            "stable_cross_task_same_work_signal_produced",
            "h21_primary_rewritten",
            "h22_can_override_h21_primary",
            "winner_selected",
            "phase2d2b_entered",
            "phase2d3_entered",
            "speedup_claimed",
        ):
            self.assertFalse(conclusion[field])

    def test_20_govreport_remains_replication_only(self) -> None:
        h21, h22 = self.cells()
        task_rows = build_aggregate(
            bundle=self.bundle,
            bindings=self.bindings,
            plan=self.plan,
            h21_cells=h21,
            h22_cells=h22,
        )["task_context"]["rows"]
        self.assertTrue(
            all(
                row["task_role"] == "same_task_replication_only"
                for row in task_rows
                if row["task"] == "gov_report"
            )
        )

    def test_21_manifest_detects_anchor_hash_mutation(self) -> None:
        manifest = build_execution_manifest(
            self.bundle,
            self.bindings,
            self.plan,
            {},
            execution_authorized=False,
            repository_commit="a" * 40,
            branch="branch",
        )
        changed = deepcopy(self.bindings)
        changed["h21_anchor_binding_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "immutable source binding"):
            ensure_immutable_sources(manifest, self.bundle, changed)

    def test_22_h20_and_h21_inputs_remain_byte_identical(self) -> None:
        freeze_before = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.freeze_dir.iterdir()
            if path.is_file()
        }
        h21, h22 = self.cells()
        h21_before = hashlib.sha256(
            json.dumps(h21, sort_keys=True).encode()
        ).hexdigest()
        build_aggregate(
            bundle=self.bundle,
            bindings=self.bindings,
            plan=self.plan,
            h21_cells=h21,
            h22_cells=h22,
        )
        freeze_after = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.freeze_dir.iterdir()
            if path.is_file()
        }
        self.assertEqual(freeze_before, freeze_after)
        self.assertEqual(
            h21_before,
            hashlib.sha256(json.dumps(h21, sort_keys=True).encode()).hexdigest(),
        )

    def test_23_all_aggregate_json_artifacts_validate(self) -> None:
        h21, h22 = self.cells()
        artifacts = build_aggregate(
            bundle=self.bundle,
            bindings=self.bindings,
            plan=self.plan,
            h21_cells=h21,
            h22_cells=h22,
        )
        for payload in artifacts.values():
            validate_h22_schema(payload)

    def test_24_online_instrumented_and_third_party_are_unchanged(self) -> None:
        changed = subprocess.check_output(
            (
                "git",
                "diff",
                "--name-only",
                "HEAD",
                "--",
                "third_party",
                "dflash/phase2d2",
            ),
            cwd=REPO_ROOT,
            text=True,
        ).splitlines()
        self.assertEqual(changed, [])

    def test_25_acceptance_checker_imports_in_fresh_interpreter(self) -> None:
        checker_import = (
            "from scripts.check_phase2d2a_h22_acceptance "
            "import check_h22_output; assert callable(check_h22_output)"
        )
        result = subprocess.run(
            (
                sys.executable,
                "-c",
                checker_import,
            ),
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
