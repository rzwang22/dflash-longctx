from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jsonschema import Draft202012Validator

from dflash.phase2c.gold import (
    DEFAULT_GOLD_THRESHOLDS,
    classify_comparison_matrix,
    expected_node_alignment,
    structural_alignment_issues,
    summarize_comparison_mode,
    validate_gold_thresholds,
    validate_visibility_matrix,
)
from experiments.phase2c_online_tree import (
    main as phase2c_main,
)
from experiments.phase2c_online_tree import (
    parse_args,
    persist_gold_artifacts,
)


def _node_record(
    *,
    argmax_equal: bool = True,
    top8_equal: bool = True,
    max_error: float = 0.1,
    mean_error: float = 0.001,
    structural_issues: list[str] | None = None,
    verifier_relevant: bool = True,
) -> dict:
    return {
        "availability": "available",
        "node_depth": 0,
        "flattened_node_index": 0,
        "argmax_equal": argmax_equal,
        "top8_ordered_equal": top8_equal,
        "max_abs_error": max_error,
        "mean_abs_error": mean_error,
        "p99_abs_error": max_error,
        "logit_element_count": 10,
        "structural_issues": structural_issues or [],
        "verifier_decision_relevant": verifier_relevant,
    }


def _mode_summary(mode: str, node: dict, issues: list[str] | None = None) -> dict:
    return summarize_comparison_mode(
        mode=mode,
        backend="test_backend",
        dtype="bfloat16",
        nodes=[node],
        structural_issues=issues or [],
        thresholds=DEFAULT_GOLD_THRESHOLDS,
    )


class Phase2C1ClassificationTests(unittest.TestCase):
    def test_off_by_one_logits_mapping_is_detected(self) -> None:
        alignment = expected_node_alignment(
            node={
                "node_index": 3,
                "parent_index": 1,
                "depth": 2,
                "token_id": 9,
                "root_to_node_token_path": [7, 8, 9],
            },
            persistent_prefix_length=100,
            packed_query_length=16,
        )
        self.assertEqual(alignment["packed_logits_query_index"], 3)
        self.assertEqual(alignment["dense_logits_query_index"], 2)
        self.assertEqual(alignment["predicted_token_semantic_position"], 103)
        alignment["dense_logits_query_index"] = 1
        self.assertIn(
            "dense_logits_query_index", structural_alignment_issues(alignment)
        )

    def test_semantic_and_storage_position_confusion_is_detected(self) -> None:
        alignment = expected_node_alignment(
            node={
                "node_index": 7,
                "parent_index": 2,
                "depth": 2,
                "token_id": 9,
                "root_to_node_token_path": [7, 8, 9],
            },
            persistent_prefix_length=100,
            packed_query_length=16,
        )
        self.assertNotEqual(
            alignment["packed_position_id"], alignment["packed_cache_position"]
        )
        alignment["packed_cache_position"] = alignment["packed_position_id"]
        self.assertIn("packed_storage_position", structural_alignment_issues(alignment))

    def test_sibling_leakage_is_structural_failure(self) -> None:
        expected = [[True, True, False], [True, False, True]]
        actual = [[True, True, True], [True, False, True]]
        issues = validate_visibility_matrix(actual, expected)
        self.assertTrue(any("leakage" in issue for issue in issues))
        summary = _mode_summary("native", _node_record(), issues)
        result = classify_comparison_matrix(
            [summary], output_ids_equal=True, round_semantics_equal=True
        )
        self.assertEqual(result["status"], "fail_structural")

    def test_numerical_only_mismatch_requires_exact_control_evidence(self) -> None:
        native = _mode_summary(
            "native",
            _node_record(top8_equal=False, max_error=0.3125, mean_error=0.0039),
        )
        same_mask = _mode_summary("same_mask", _node_record())
        result = classify_comparison_matrix(
            [native, same_mask], output_ids_equal=True, round_semantics_equal=True
        )
        self.assertEqual(result["status"], "pass_semantic_with_numerical_variation")
        self.assertTrue(result["numerical_backend_variation_proven"])

    def test_numerical_mismatch_without_control_evidence_does_not_pass(self) -> None:
        native = _mode_summary(
            "native", _node_record(top8_equal=False, max_error=0.3125)
        )
        result = classify_comparison_matrix(
            [native], output_ids_equal=True, round_semantics_equal=True
        )
        self.assertEqual(result["status"], "fail_semantic")
        self.assertEqual(result["failure_class"], "unresolved_gold_mismatch")

    def test_verifier_argmax_mismatch_is_hard_semantic_failure(self) -> None:
        native = _mode_summary("native", _node_record(argmax_equal=False))
        control = _mode_summary("math", _node_record())
        result = classify_comparison_matrix(
            [native, control], output_ids_equal=True, round_semantics_equal=True
        )
        self.assertEqual(result["status"], "fail_semantic")
        self.assertEqual(result["failure_class"], "semantic_mismatch")

    def test_top8_mismatch_is_reported_separately_from_argmax(self) -> None:
        summary = _mode_summary("native", _node_record(top8_equal=False))
        self.assertEqual(summary["argmax_mismatch_count"], 0)
        self.assertEqual(summary["verifier_decision_mismatch_count"], 0)
        self.assertEqual(summary["top8_ordered_mismatch_count"], 1)
        self.assertIsNotNone(summary["warning"])

    def test_thresholds_are_named_and_never_silently_changed(self) -> None:
        configured = dict(DEFAULT_GOLD_THRESHOLDS)
        configured["maximum_absolute_error"] = 0.25
        self.assertEqual(
            validate_gold_thresholds(configured)["maximum_absolute_error"], 0.25
        )
        with self.assertRaises(ValueError):
            validate_gold_thresholds({"maximum_absolute_error": 1.0})

    def test_unavailable_math_backend_is_not_success(self) -> None:
        unavailable = summarize_comparison_mode(
            mode="math",
            backend="forced_math_sdpa",
            dtype="bfloat16",
            nodes=[],
            structural_issues=[],
            thresholds=DEFAULT_GOLD_THRESHOLDS,
            unavailable_reason="sdpa_kernel unavailable",
        )
        self.assertEqual(unavailable["availability"], "unavailable")
        self.assertEqual(unavailable["status"], "not_run")
        result = classify_comparison_matrix(
            [_mode_summary("native", _node_record(top8_equal=False)), unavailable],
            output_ids_equal=True,
            round_semantics_equal=True,
        )
        self.assertFalse(result["all_checks_passed"])

    def test_equal_shape_top1_mapping_is_index_aligned(self) -> None:
        for index in range(16):
            alignment = expected_node_alignment(
                node={
                    "node_index": index,
                    "parent_index": index - 1 if index else None,
                    "depth": index,
                    "token_id": 100 + index,
                    "root_to_node_token_path": list(range(100, 101 + index)),
                },
                persistent_prefix_length=1024,
                packed_query_length=16,
            )
            self.assertEqual(alignment["packed_logits_query_index"], index)
            self.assertEqual(alignment["input_token_semantic_position"], 1024 + index)


class Phase2C1PersistenceTests(unittest.TestCase):
    def _paths(self, root: Path) -> dict[str, Path]:
        return {
            "gold_logit_checks": root / "gold-logit-checks.jsonl",
            "gold_kv_checks": root / "gold-kv-checks.jsonl",
            "gold_layer_localization": root / "gold-layer-localization.jsonl",
            "gold_comparison_summary": root / "gold-comparison-summary.json",
            "gold_comparison_summary_csv": root / "gold-comparison-summary.csv",
        }

    def _persistable_record(self) -> dict:
        return {
            "schema_version": "1.0",
            "task": "gov_report",
            "sample_id": "sample",
            "nominal_prompt_length": 1024,
            "configuration_id": "top1_chain-b16",
            "round_index": 0,
            "availability": "available",
            "status": "diagnostic",
            "comparison_mode": "native",
            "backend": "test",
            "dtype": "bfloat16",
            "structural_status": "pass",
            "semantic_status": "pass",
            "numerical_status": "within_strict_thresholds",
            "thresholds": DEFAULT_GOLD_THRESHOLDS,
            "warning": "ordered Top-8 differs",
            "reason": None,
            "provenance": {"actual_cuda_kernel": None},
            "argmax_equal": True,
            "top8_ordered_equal": False,
            "max_abs_error": 0.3125,
            "mean_abs_error": 0.0039,
            "p99_abs_error": 0.02,
        }

    def test_failure_record_and_kv_not_run_are_flushed_before_raise(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._paths(Path(temporary))
            logits = [self._persistable_record()]
            kv = [
                {
                    "schema_version": "1.0",
                    "task": "gov_report",
                    "sample_id": "sample",
                    "nominal_prompt_length": 1024,
                    "configuration_id": "top1_chain-b16",
                    "round_index": 0,
                    "availability": "unavailable",
                    "status": "not_run",
                    "execution_status": "not_run_due_to_prior_failure",
                    "comparison_mode": "dense_committed_path",
                    "backend": "configured_sdpa",
                    "dtype": "bfloat16",
                    "reason": "prior_logits_gold_failure",
                }
            ]
            persist_gold_artifacts(
                paths=paths, gold_logits=logits, gold_kv=kv, gold_layers=[]
            )
            with self.assertRaisesRegex(RuntimeError, "synthetic gold failure"):
                raise RuntimeError("synthetic gold failure")
            self.assertTrue(paths["gold_logit_checks"].read_text().strip())
            persisted_kv = json.loads(paths["gold_kv_checks"].read_text())
            self.assertEqual(persisted_kv["status"], "not_run")
            self.assertEqual(
                persisted_kv["execution_status"],
                "not_run_due_to_prior_failure",
            )
            self.assertEqual(persisted_kv["reason"], "prior_logits_gold_failure")

    def test_cli_records_modes_and_unchanged_threshold_defaults(self) -> None:
        args = parse_args(
            [
                "--dataset-jsonl",
                "local.jsonl",
                "--materialized-draft-path",
                "draft",
                "--enable-gold-checks",
                "--gold-comparison-mode",
                "native",
                "--gold-comparison-mode",
                "math",
                "--gold-record-all-nodes",
            ]
        )
        self.assertEqual(args.gold_comparison_modes, ("native", "math"))
        self.assertEqual(args.gold_thresholds, DEFAULT_GOLD_THRESHOLDS)

    def test_phase2c1_records_validate_new_schemas(self) -> None:
        root = Path(__file__).resolve().parents[1] / "docs" / "schemas"
        common = self._persistable_record()
        logit_schema = json.loads(
            (root / "phase2c-gold-logit-check.schema.json").read_text()
        )
        Draft202012Validator(logit_schema).validate(common)
        kv = {
            **common,
            "status": "not_run",
            "availability": "unavailable",
            "comparison_mode": "dense_committed_path",
            "structural_status": "not_run",
            "semantic_status": "not_run",
            "numerical_status": "not_run",
            "reason": "prior_logits_gold_failure",
        }
        kv_schema = json.loads((root / "phase2c-gold-kv-check.schema.json").read_text())
        Draft202012Validator(kv_schema).validate(kv)
        layer = {
            **common,
            "comparison_mode": "same_mask",
            "flattened_node_index": 0,
            "node_depth": 0,
            "stage": "embedding",
            "diagnostic_p99_threshold": 0.1,
            "exceeds_diagnostic_threshold": False,
            "is_first_stage_exceeding_threshold": False,
            "first_stage_exceeding_threshold": None,
            "discrete_metric_applicable": False,
        }
        layer_schema = json.loads(
            (root / "phase2c-gold-layer-localization.schema.json").read_text()
        )
        Draft202012Validator(layer_schema).validate(layer)
        summary = {
            "schema_version": "1.0",
            "status": "error",
            "logit_record_count": 1,
            "kv_record_count": 1,
            "groups": [],
            "interpretation": {},
        }
        summary_schema = json.loads(
            (root / "phase2c-gold-comparison-summary.schema.json").read_text()
        )
        Draft202012Validator(summary_schema).validate(summary)

    def test_failure_main_marks_manifest_and_correctness_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text("{}\n")
            output = root / "out"

            def fail_after_persist(_args, **kwargs):
                record = self._persistable_record()
                kv = {
                    **record,
                    "availability": "unavailable",
                    "status": "not_run",
                    "comparison_mode": "dense_committed_path",
                    "reason": "prior_logits_gold_failure",
                }
                kwargs["gold_logits"].append(record)
                kwargs["gold_kv"].append(kv)
                kwargs["correctness_records"].append(
                    {
                        "gold_status": "fail_semantic",
                        "gold_failure_class": "synthetic",
                        "gold_completed_checks": ["logits"],
                        "gold_uncompleted_checks": ["kv"],
                        "passed": False,
                    }
                )
                persist_gold_artifacts(
                    paths=kwargs["paths"],
                    gold_logits=kwargs["gold_logits"],
                    gold_kv=kwargs["gold_kv"],
                    gold_layers=kwargs["gold_layers"],
                )
                raise RuntimeError("synthetic gold failure")

            environment = {
                "schema_version": "1.0",
                "repository": {"commit": "test", "branch": "test"},
            }
            argv = [
                "--dataset-jsonl",
                str(dataset),
                "--materialized-draft-path",
                "draft",
                "--enable-gold-checks",
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
            manifest = json.loads((output / "manifest.json").read_text())
            correctness = json.loads((output / "correctness-summary.json").read_text())
            self.assertEqual(manifest["status"], "error")
            self.assertEqual(correctness["status"], "error")
            self.assertFalse(correctness["all_checks_passed"])
            self.assertEqual(correctness["gold_status"], "fail_semantic")
            self.assertTrue((output / "gold-logit-checks.jsonl").read_text().strip())


if __name__ == "__main__":
    unittest.main()
