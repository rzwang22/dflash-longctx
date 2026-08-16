from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dflash.phase2c.divergence import (
    build_round_differential,
    classify_layerwise_divergence,
    first_divergent_offset,
    locate_token_producer_round,
    output_pair_comparison,
)
from experiments.phase2c_online_tree import (
    Phase2CCorrectnessFailure,
    _validate_schema,
    persist_divergence_artifacts,
)
from experiments.phase2c_online_tree import (
    main as phase2c_main,
)


def _round(index: int, root: int, accepted: int, kind: str = "correction") -> dict:
    return {
        "round_index": index,
        "generated_root_offset": root,
        "context_length_before": 100 + root,
        "context_length_after": 101 + root + accepted,
        "accepted_draft_token_count": accepted,
        "committed_token_count": accepted + 1,
        "target_token_kind": kind,
    }


class Phase2C2AbsoluteAlignmentTests(unittest.TestCase):
    def test_first_divergent_absolute_token_offset(self) -> None:
        self.assertEqual(first_divergent_offset([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(first_divergent_offset([1, 2], [1, 2, 3]), 2)
        comparison = output_pair_comparison(
            left_name="reference",
            left_ids=[1, 2, 3],
            right_name="tree",
            right_ids=[1, 9, 3],
            prompt_length=1024,
        )
        self.assertEqual(comparison["first_divergent_generated_token_offset"], 1)
        self.assertEqual(comparison["first_divergent_absolute_sequence_position"], 1025)

    def test_different_round_counts_align_by_absolute_token(self) -> None:
        reference = [_round(0, 0, 0), _round(1, 1, 3)]
        tree = [_round(0, 0, 2), _round(1, 3, 1)]
        located_reference = locate_token_producer_round(
            reference, generated_offset=3, prompt_length=100
        )
        located_tree = locate_token_producer_round(
            tree, generated_offset=3, prompt_length=100
        )
        self.assertEqual(located_reference["round_index"], 1)
        self.assertEqual(located_tree["round_index"], 0)
        comparison = output_pair_comparison(
            left_name="reference",
            left_ids=[1, 2, 3, 4],
            right_name="tree",
            right_ids=[1, 2, 3, 9],
            prompt_length=100,
        )
        differential = build_round_differential(
            comparison=comparison,
            prompt_length=100,
            reference_rounds=reference,
            tree_rounds=tree,
        )
        self.assertFalse(differential["round_indices_equal"])
        self.assertTrue(differential["predivergence_committed_prefix_token_exact"])

    def test_divergence_classifications_are_evidence_driven(self) -> None:
        common = {
            "output_ids_equal": False,
            "traversal_checks_passed": True,
            "structural_checks_passed": True,
            "kv_root_transition_checks_passed": True,
            "kv_next_token_argmax_equal": True,
            "divergence_begins_at_round_root": False,
        }
        numerical = classify_layerwise_divergence(
            **common,
            native_verifier_mismatch_count=1,
            math_verifier_mismatch_count=0,
        )
        self.assertEqual(
            numerical["failure_class"], "numerical_backend_decision_divergence"
        )
        traversal = classify_layerwise_divergence(
            **{**common, "traversal_checks_passed": False},
            native_verifier_mismatch_count=0,
            math_verifier_mismatch_count=0,
        )
        self.assertEqual(traversal["failure_class"], "tree_traversal_mismatch")
        kv = classify_layerwise_divergence(
            **{**common, "kv_root_transition_checks_passed": False},
            native_verifier_mismatch_count=0,
            math_verifier_mismatch_count=0,
        )
        self.assertEqual(
            kv["failure_class"], "kv_compaction_or_root_transition_mismatch"
        )


class Phase2C2PersistenceTests(unittest.TestCase):
    def _paths(self, root: Path) -> dict[str, Path]:
        return {
            "output_divergence": root / "output-divergence.json",
            "output_divergence_csv": root / "output-divergence.csv",
            "round_differential": root / "round-differential.jsonl",
            "tree_traversal_checks": root / "tree-traversal-checks.jsonl",
        }

    def _divergence(self) -> dict:
        return {
            "schema_version": "1.0",
            "status": "failed",
            "task": "gov_report",
            "sample_id": "sample",
            "nominal_prompt_length": 1024,
            "actual_prompt_length": 1024,
            "configuration_id": "layerwise_beam-b16-w4",
            "classification": {
                "failure_class": "unresolved_layerwise_output_divergence"
            },
        }

    def test_mismatch_artifacts_are_flushed_before_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._paths(Path(temporary))
            persist_divergence_artifacts(
                paths=paths,
                divergences=[self._divergence()],
                round_differentials=[{"generated_token_offset": 3}],
                traversal_checks=[{"round_index": 0}],
            )
            with self.assertRaises(Phase2CCorrectnessFailure):
                raise Phase2CCorrectnessFailure("synthetic mismatch")
            self.assertEqual(
                json.loads(paths["output_divergence"].read_text())["status"],
                "failed",
            )
            self.assertTrue(paths["round_differential"].read_text().strip())
            self.assertTrue(paths["tree_traversal_checks"].read_text().strip())

    def test_main_marks_expected_correctness_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text("{}\n")
            output = root / "out"

            def fail_after_persist(_args, **kwargs):
                divergence = self._divergence()
                kwargs["divergences"].append(divergence)
                kwargs["round_differentials"].append({"generated_token_offset": 3})
                kwargs["traversal_checks"].append({"round_index": 0})
                kwargs["correctness_records"].append(
                    {
                        "gold_status": "not_run",
                        "gold_failure_class": None,
                        "output_divergence_failure_class": (
                            "unresolved_layerwise_output_divergence"
                        ),
                        "gold_completed_checks": ["absolute_token_output_comparison"],
                        "gold_uncompleted_checks": ["packed_dense_native_math_gold"],
                        "passed": False,
                    }
                )
                persist_divergence_artifacts(
                    paths=kwargs["paths"],
                    divergences=kwargs["divergences"],
                    round_differentials=kwargs["round_differentials"],
                    traversal_checks=kwargs["traversal_checks"],
                )
                raise Phase2CCorrectnessFailure("synthetic mismatch")

            environment = {
                "schema_version": "1.0",
                "repository": {"commit": "test", "branch": "test"},
            }
            argv = [
                "--dataset-jsonl",
                str(dataset),
                "--materialized-draft-path",
                "draft",
                "--output-dir",
                str(output),
            ]
            with (
                mock.patch(
                    "experiments.phase2c_online_tree.collect_environment",
                    return_value=environment,
                ),
                mock.patch("experiments.phase2c_online_tree._preflight_offline_inputs"),
                mock.patch(
                    "experiments.phase2c_online_tree._load_reference_and_run",
                    side_effect=fail_after_persist,
                ),
            ):
                self.assertEqual(phase2c_main(argv), 1)
            correctness = json.loads((output / "correctness-summary.json").read_text())
            self.assertEqual(correctness["status"], "failed")
            self.assertEqual(
                correctness["failure_class"],
                "unresolved_layerwise_output_divergence",
            )
            self.assertTrue((output / "round-differential.jsonl").read_text().strip())


class Phase2C2SchemaTests(unittest.TestCase):
    def test_new_diagnostic_schemas_accept_complete_records(self) -> None:
        pairwise = [
            output_pair_comparison(
                left_name=f"left-{index}",
                left_ids=[1, 2],
                right_name=f"right-{index}",
                right_ids=[1, 3],
                prompt_length=1024,
            )
            for index in range(3)
        ]
        divergence = {
            "schema_version": "1.0",
            "status": "failed",
            "task": "gov_report",
            "sample_id": "sample",
            "nominal_prompt_length": 1024,
            "actual_prompt_length": 1024,
            "configuration_id": "layerwise_beam-b16-w4",
            "pairwise_comparisons": pairwise,
            "reference_output_ids": [1, 2],
            "tree_control_output_ids": [1, 3],
            "target_only_output_ids": [1, 2],
            "reference_returned_token_count": 2,
            "tree_returned_token_count": 2,
            "longest_common_prefix_length": 1,
            "first_divergent_generated_token_offset": 1,
            "first_divergent_absolute_sequence_position": 1025,
            "reference_token_id": 2,
            "reference_token_text": "two",
            "tree_token_id": 3,
            "tree_token_text": "three",
            "divergent_tree_round_index": 0,
            "divergent_token_kind": "correction",
            "divergent_round_context_length_before": 1024,
            "divergent_round_context_length_after": 1025,
            "divergent_round_persistent_kv_length_before": 1024,
            "divergent_round_persistent_kv_length_after": 1025,
            "diagnostic_output_ids_equal_formal_control": True,
            "tree_control_repeat_output_ids_equal_reference": True,
            "reference_repeat_output_ids_equal": True,
            "reference_profile_output_ids_equal": True,
            "round_differential_status": "available",
            "tree_traversal_status": "available",
            "first_divergent_tree_traversal_status": "available",
            "diagnostic_capture_round_limit": 1,
            "classification": {
                "status": "inconclusive",
                "failure_class": "unresolved_layerwise_output_divergence",
                "evidence_complete": False,
            },
            "exception": {
                "type": "Phase2CCorrectnessFailure",
                "message": "synthetic",
            },
            "uncompleted_checks": [],
        }
        _validate_schema(
            {
                "schema_version": "1.0",
                "status": "failed",
                "records": [divergence],
            },
            "phase2c-output-divergence.schema.json",
        )
        _validate_schema(
            {
                "schema_version": "1.0",
                "task": "gov_report",
                "sample_id": "sample",
                "nominal_prompt_length": 1024,
                "configuration_id": "layerwise_beam-b16-w4",
                "generated_token_offset": 1,
                "absolute_sequence_position": 1025,
                "predivergence_committed_prefix_token_exact": True,
                "predivergence_prefix_sha256": "0" * 64,
                "reference_round_entry_persistent_kv_length": None,
                "tree_round_entry_persistent_kv_length": None,
                "round_entry_persistent_kv_same_semantic_prefix": False,
                "absolute_position_semantic_prefix_token_exact": True,
                "reference": None,
                "tree": None,
                "round_indices_equal": False,
                "absolute_token_lockstep_used": True,
            },
            "phase2c-round-differential.schema.json",
        )
        _validate_schema(
            {
                "schema_version": "1.0",
                "task": "gov_report",
                "sample_id": "sample",
                "nominal_prompt_length": 1024,
                "configuration_id": "layerwise_beam-b16-w4",
                "round_index": 0,
                "status": "pass",
                "root_token_id": 1,
                "nodes": [{}],
                "accepted_path_node_indices": [],
                "accepted_depth": 0,
                "terminal_parent_index": 0,
                "correction_or_bonus_token_id": 2,
                "correction_or_bonus_kind": "correction",
                "committed_node_indices": [0],
                "kv_gather_indices": [0],
                "next_round_root_token_id": 2,
                "next_round_root_absolute_position": 1025,
                "context_length_before": 1024,
                "context_length_after": 1025,
                "persistent_kv_length_before": 1024,
                "persistent_kv_length_after": 1025,
                "transition_checks": [],
                "structural_mask_issues": [],
                "allowed_attention_positions_by_node": [[]],
                "checks": {
                    "structural_checks_passed": True,
                    "traversal_checks_passed": True,
                    "kv_root_transition_checks_passed": True,
                    "all_checks_passed": True,
                },
                "cache_before_comparable_summary": [],
                "cache_after_comparable_summary": [],
                "is_first_divergent_round": True,
            },
            "phase2c-tree-traversal-check.schema.json",
        )


if __name__ == "__main__":
    unittest.main()
