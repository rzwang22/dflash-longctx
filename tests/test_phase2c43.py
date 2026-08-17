from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_phase2c43_evidence import (
    LOCAL_NUMERICAL,
    UNRESOLVED,
    build_aggregate,
    validate_aggregate,
    write_aggregate,
)
from scripts.check_phase2c43_acceptance import check_aggregate

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_run(
    root: Path,
    name: str,
    *,
    sample_id: str,
    nominal_length: int,
    divergence_offset: int,
    reference_token: int,
    tree_token: int,
    packed_token: int,
    native_token: int | None,
    same_mask_token: int | None,
    math_token: int | None,
    classification: str,
    classification_complete: bool,
    numerical_support: bool,
) -> Path:
    output_dir = root / name
    output_dir.mkdir()
    case = {
        "sample_id": sample_id,
        "nominal_prompt_length": nominal_length,
        "first_actual_semantic_divergence": divergence_offset,
        "failure_classification": classification,
        "classification_evidence_complete": classification_complete,
        "unresolved_reason": (
            None
            if classification == LOCAL_NUMERICAL
            else "no_faithful_actual_semantic_cause_observed"
        ),
    }
    summary = {
        "schema_version": "1.0",
        "status": "complete",
        "cases": [case],
        "actual_execution_is_semantic_oracle": True,
        "dense_replay_controls_are_attribution_only": True,
        "formal_timing_contaminated": False,
    }
    row = {
        "schema_version": "1.0",
        "sample_id": sample_id,
        "nominal_prompt_length": nominal_length,
        "generated_prefix_length": divergence_offset,
        "actual_token_divergence": True,
        "actual_decision_oracle_source": "actual_original_dflash_output",
        "dense_replay_control_only": True,
        "reference_actual_token_id": reference_token,
        "tree_actual_token_id": tree_token,
        "packed_tree_argmax_token_id": packed_token,
        "packed_tree_argmax_matches_actual_reference": (
            packed_token == reference_token
        ),
        "packed_tree_argmax_matches_actual_tree": packed_token == tree_token,
        "native_dense_replay_token_id": native_token,
        "native_dense_replay_matches_actual_reference": (
            None if native_token is None else native_token == reference_token
        ),
        "same_mask_replay_token_id": same_mask_token,
        "same_mask_replay_matches_actual_reference": (
            None if same_mask_token is None else same_mask_token == reference_token
        ),
        "math_replay_token_id": math_token,
        "math_replay_matches_actual_reference": (
            None if math_token is None else math_token == reference_token
        ),
        "local_packed_verifier_numerical_divergence_supported": numerical_support,
        "classification_evidence_complete": classification_complete,
    }
    (output_dir / "reference-state-summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (output_dir / "reference-state-differential.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    return output_dir


def _case4_runs(root: Path) -> list[Path]:
    runs = []
    for index in range(5):
        supports_reference = index < 3
        runs.append(
            _write_run(
                root,
                f"case4-run-{index + 1}",
                sample_id="gov_report-4a773bcb56f65ce3",
                nominal_length=32768,
                divergence_offset=28,
                reference_token=8706,
                tree_token=41044,
                packed_token=41044,
                native_token=41044,
                same_mask_token=8706 if supports_reference else 41044,
                math_token=41044,
                classification=(LOCAL_NUMERICAL if supports_reference else UNRESOLVED),
                classification_complete=supports_reference,
                numerical_support=supports_reference,
            )
        )
    return runs


class Phase2C43AggregationTests(unittest.TestCase):
    def test_case4_five_run_nonreproducible_numerical_support(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = build_aggregate(_case4_runs(root))
            validate_aggregate(payload)
            self.assertEqual(payload["input_run_count"], 5)
            self.assertEqual(payload["group_count"], 1)
            group = payload["groups"][0]
            self.assertEqual(group["actual_reference_token_distribution"], {"8706": 5})
            self.assertEqual(group["actual_tree_token_distribution"], {"41044": 5})
            self.assertEqual(group["packed_tree_token_distribution"], {"41044": 5})
            self.assertEqual(group["native_replay_token_distribution"], {"41044": 5})
            self.assertEqual(
                group["same_mask_replay_token_distribution"],
                {"8706": 3, "41044": 2},
            )
            self.assertEqual(group["math_replay_token_distribution"], {"41044": 5})
            self.assertTrue(group["actual_reference_stability"])
            self.assertTrue(group["actual_tree_stability"])
            self.assertTrue(group["packed_tree_stability"])
            self.assertTrue(group["stable_actual_reference_tree_divergence"])
            self.assertEqual(group["native_reference_support_rate"], 0.0)
            self.assertEqual(group["same_mask_reference_support_count"], 3)
            self.assertEqual(group["same_mask_reference_support_rate"], 0.6)
            self.assertEqual(group["math_reference_support_rate"], 0.0)
            self.assertEqual(
                group["per_run_classification_distribution"],
                {LOCAL_NUMERICAL: 3, UNRESOLVED: 2},
            )
            self.assertEqual(group["aggregate_failure_classification"], UNRESOLVED)
            self.assertEqual(
                group["aggregate_unresolved_reason"],
                "nonreproducible_numerical_control_support",
            )
            self.assertFalse(group["aggregate_evidence_complete"])

    def test_case3_reproducible_support_remains_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = [
                _write_run(
                    root,
                    f"case3-run-{index}",
                    sample_id="gov_report-fd287bbfd3bdbdc9",
                    nominal_length=16384,
                    divergence_offset=18,
                    reference_token=323,
                    tree_token=320,
                    packed_token=320,
                    native_token=320,
                    same_mask_token=323,
                    math_token=323,
                    classification=LOCAL_NUMERICAL,
                    classification_complete=True,
                    numerical_support=True,
                )
                for index in range(3)
            ]
            artifact, payload = write_aggregate(runs, root / "case3-aggregate")
            group = payload["groups"][0]
            self.assertEqual(group["aggregate_failure_classification"], LOCAL_NUMERICAL)
            self.assertIsNone(group["aggregate_unresolved_reason"])
            self.assertTrue(group["aggregate_evidence_complete"])
            self.assertTrue(check_aggregate("case3", artifact)["passed"])

    def test_case2_without_control_support_remains_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = [
                _write_run(
                    root,
                    f"case2-run-{index}",
                    sample_id="gov_report-4a773bcb56f65ce3",
                    nominal_length=1024,
                    divergence_offset=33,
                    reference_token=785,
                    tree_token=1115,
                    packed_token=1115,
                    native_token=1115,
                    same_mask_token=1115,
                    math_token=1115,
                    classification=UNRESOLVED,
                    classification_complete=False,
                    numerical_support=False,
                )
                for index in range(2)
            ]
            artifact, payload = write_aggregate(runs, root / "case2-aggregate")
            group = payload["groups"][0]
            self.assertEqual(group["aggregate_failure_classification"], UNRESOLVED)
            self.assertEqual(
                group["aggregate_unresolved_reason"],
                "no_numerical_control_support",
            )
            self.assertTrue(check_aggregate("case2", artifact)["passed"])

    def test_runs_are_grouped_by_all_three_semantic_key_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = []
            for length, offset in ((1024, 33), (32768, 28)):
                runs.append(
                    _write_run(
                        root,
                        f"case-{length}",
                        sample_id="gov_report-4a773bcb56f65ce3",
                        nominal_length=length,
                        divergence_offset=offset,
                        reference_token=1,
                        tree_token=2,
                        packed_token=2,
                        native_token=2,
                        same_mask_token=2,
                        math_token=2,
                        classification=UNRESOLVED,
                        classification_complete=False,
                        numerical_support=False,
                    )
                )
            payload = build_aggregate(runs)
            self.assertEqual(payload["group_count"], 2)
            self.assertEqual(
                {
                    (
                        group["nominal_prompt_length"],
                        group["first_actual_semantic_divergence"],
                    )
                    for group in payload["groups"]
                },
                {(1024, 33), (32768, 28)},
            )

    def test_match_flag_contradiction_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _case4_runs(root)[0]
            row_path = run / "reference-state-differential.jsonl"
            row = json.loads(row_path.read_text())
            row["same_mask_replay_matches_actual_reference"] = False
            row_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contradicts"):
                build_aggregate([run])

    def test_duplicate_directory_is_not_an_independent_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _case4_runs(Path(temporary))[0]
            with self.assertRaisesRegex(ValueError, "duplicate input"):
                build_aggregate([run, run])

    def test_output_directory_cannot_mutate_a_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _case4_runs(Path(temporary))[0]
            with self.assertRaisesRegex(ValueError, "outside every capture"):
                write_aggregate([run], run / "aggregate")

    def test_artifact_and_case4_acceptance_checker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = _case4_runs(root)
            artifact, _ = write_aggregate(runs, root / "aggregate")
            result = check_aggregate("case4", artifact)
            self.assertTrue(result["passed"], result["checks"])

    def test_aggregation_import_does_not_load_model_frameworks(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import scripts.aggregate_phase2c43_evidence; "
                    "assert 'torch' not in sys.modules; "
                    "assert 'transformers' not in sys.modules"
                ),
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
