from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jsonschema import Draft202012Validator

from dflash.phase1.materialize import _effective_sources
from dflash.phase2c.stabilization import (
    canonical_kv_commit_spec,
    canonical_selected_path_decision,
    decision_margin_frontier,
    evenly_spaced_sample_indices,
    should_recheck,
    stabilization_commit_source,
    summarize_decision_stabilization,
    validate_decision_stabilization,
)
from experiments.phase2c_online_tree import (
    _target_only_greedy,
    build_stabilization_artifacts,
    evaluate_output_oracle,
    margin_frontier_csv_rows,
    parse_args,
)
from experiments.phase2c_online_tree import main as phase2c_main

REPO_ROOT = Path(__file__).resolve().parents[1]


def _args(*extra: str):
    return parse_args(
        [
            "--dataset-jsonl",
            "/tmp/longbench.jsonl",
            "--materialized-draft-path",
            "/tmp/draft",
            *extra,
        ]
    )


def _decision_row(**updates):
    row = {
        "stabilization_mode": "shadow",
        "decision_margin_threshold": 0.1,
        "recheck_triggered": True,
        "trigger_reason": "shadow_oracle",
        "minimum_path_margin": 0.05,
        "verifier_margins": [0.2, 0.05],
        "native_canonical_decision_equal": False,
        "native_selected_path_token_ids": [1, 26722],
        "canonical_accepted_path_token_ids": [1],
        "native_accepted_depth": 1,
        "canonical_accepted_depth": 0,
        "first_corrected_depth": 1,
        "first_corrected_native_margin": 0.05,
        "native_next_root_token_id": 9,
        "native_next_root_kind": "correction",
        "canonical_next_root_token_id": 43282,
        "canonical_next_root_kind": "correction",
        "corrected_decision_count": 1,
        "false_negative": False,
        "recheck_input_shape": [1, 16],
        "recheck_query_length": 16,
        "recheck_round_entry_kv_length": 1024,
        "recheck_effective_kv_length": 1040,
        "canonical_cache_length": 1025,
        "canonical_cache_committed": False,
        "committed_cache_source": "packed_native",
        "unavailable_reason": None,
        "packed_verification_cuda_elapsed_ms": 2.0,
        "canonical_recheck_cuda_elapsed_ms": 4.0,
        "kv_replacement_cuda_elapsed_ms": None,
        "total_recheck_overhead_cuda_elapsed_ms": 4.0,
        "stabilized_round_cuda_elapsed_ms": 9.0,
        "recheck_over_round_ratio": 4.0 / 9.0,
        "generated_token_count": 2,
        "committed_token_count": 2,
    }
    row.update(updates)
    return row


def _round(**decision_updates):
    return {
        "task": "gov_report",
        "sample_id": "gov_report-test",
        "nominal_prompt_length": 1024,
        "configuration_id": "layerwise_beam-b16-w4",
        "profiling_repeat": 0,
        "round_index": 0,
        "output_ids_equal_reference": True,
        "decision_stabilization": _decision_row(**decision_updates),
    }


def _sample(*, formal_ms: float = 10.0, reference_ms: float = 20.0):
    return {
        "generated_token_count": 2,
        "control": {"repeats": [{"cuda_elapsed_ms": formal_ms}]},
        "reference": {"control_repeats": [{"cuda_elapsed_ms": reference_ms}]},
        "target_only": {"cuda_elapsed_ms": 30.0},
        "correctness": {"reference_output_ids_equal": True},
    }


class Phase2C3DecisionTests(unittest.TestCase):
    def test_integer_diagnostic_indices_cover_large_kv_boundaries(self) -> None:
        for numel in (
            1_000,
            2**24,
            2**24 + 1_024,
            32_768 * 1_024,
        ):
            with self.subTest(numel=numel):
                indices = evenly_spaced_sample_indices(numel)
                self.assertEqual(len(indices), 16)
                self.assertEqual(indices[0], 0)
                self.assertEqual(indices[-1], numel - 1)
                self.assertEqual(
                    indices,
                    [(index * (numel - 1)) // 15 for index in range(16)],
                )
                self.assertTrue(all(0 <= index < numel for index in indices))
                self.assertEqual(indices, sorted(set(indices)))

    def test_integer_diagnostic_indices_handle_empty_and_singleton(self) -> None:
        self.assertEqual(evenly_spaced_sample_indices(0), [])
        self.assertEqual(evenly_spaced_sample_indices(1), [0])

    def test_off_keeps_native_decision_without_recheck(self) -> None:
        self.assertFalse(should_recheck("off", 0.001, 0.1))

    def test_always_rechecks_without_margin(self) -> None:
        self.assertTrue(should_recheck("always", None, 0.1))

    def test_shadow_records_but_commits_native(self) -> None:
        self.assertTrue(should_recheck("shadow", None, 0.1))
        self.assertEqual(stabilization_commit_source("shadow", True), "packed_native")

    def test_synthetic_actions_flip_restores_activities(self) -> None:
        result = canonical_selected_path_decision(
            selected_draft_token_ids=[26722],
            canonical_parent_argmax_token_ids=[43282, 7],
            native_next_root_token_id=9,
            native_next_root_is_bonus=False,
        )
        self.assertEqual(result["accepted_draft_token_count"], 0)
        self.assertEqual(result["next_root_token_id"], 43282)
        self.assertTrue(result["decision_changed"])

    def test_canonical_rejection_emits_parent_correction(self) -> None:
        result = canonical_selected_path_decision(
            selected_draft_token_ids=[10, 20, 30],
            canonical_parent_argmax_token_ids=[10, 99, 30, 40],
            native_next_root_token_id=40,
            native_next_root_is_bonus=True,
        )
        self.assertEqual(result["accepted_draft_token_ids"], [10])
        self.assertEqual(result["next_root_kind"], "correction")
        self.assertEqual(result["next_root_token_id"], 99)

    def test_canonical_full_acceptance_emits_leaf_bonus(self) -> None:
        result = canonical_selected_path_decision(
            selected_draft_token_ids=[10, 20],
            canonical_parent_argmax_token_ids=[10, 20, 30],
            native_next_root_token_id=30,
            native_next_root_is_bonus=True,
        )
        self.assertEqual(result["accepted_draft_token_count"], 2)
        self.assertEqual(result["next_root_kind"], "bonus")
        self.assertFalse(result["decision_changed"])

    def test_canonical_kv_length_invariant(self) -> None:
        spec = canonical_kv_commit_spec(
            round_entry_prefix_length=1024,
            native_accepted_draft_count=3,
            canonical_accepted_draft_count=1,
        )
        self.assertEqual(spec["temporary_cache_length"], 1040)
        self.assertEqual(spec["committed_cache_length"], 1026)
        self.assertEqual(spec["committed_cache_source"], "canonical_recheck")

    def test_native_extra_nodes_and_canonical_tail_are_discarded(self) -> None:
        spec = canonical_kv_commit_spec(
            round_entry_prefix_length=5,
            native_accepted_draft_count=4,
            canonical_accepted_draft_count=2,
        )
        self.assertEqual(spec["native_extra_accepted_node_count_discarded"], 2)
        self.assertEqual(spec["canonical_tail_discarded_count"], 13)

    def test_correction_or_bonus_is_not_in_current_kv(self) -> None:
        spec = canonical_kv_commit_spec(
            round_entry_prefix_length=5,
            native_accepted_draft_count=0,
            canonical_accepted_draft_count=0,
        )
        self.assertEqual(spec["committed_cache_length"], 6)
        self.assertFalse(spec["correction_or_bonus_in_current_cache"])

    def test_margin_threshold_triggers(self) -> None:
        self.assertTrue(should_recheck("margin", 0.05, 0.05))

    def test_margin_threshold_does_not_trigger(self) -> None:
        self.assertFalse(should_recheck("margin", 0.05001, 0.05))

    def test_shadow_false_negative_frontier(self) -> None:
        frontier = decision_margin_frontier(
            [
                _decision_row(minimum_path_margin=0.2),
                _decision_row(
                    minimum_path_margin=0.01,
                    native_canonical_decision_equal=True,
                ),
            ],
            thresholds=[0.1, 0.2],
        )
        self.assertEqual(frontier[0]["false_negative_count"], 1)
        self.assertEqual(frontier[1]["false_negative_count"], 0)
        self.assertEqual(len(margin_frontier_csv_rows(frontier)), 2)

    def test_shadow_oracle_latency_is_excluded_from_formal_native_timing(self) -> None:
        summary = summarize_decision_stabilization(
            [{"profiling_repeat": 0, **_decision_row()}], [_sample(formal_ms=10)]
        )
        self.assertEqual(summary["formal_end_to_end_cuda_elapsed_ms"]["median"], 10)
        self.assertEqual(summary["canonical_recheck_cuda_elapsed_ms"]["median"], 4)

    def test_always_formal_latency_is_reported_as_actual_online_cost(self) -> None:
        row = _decision_row(
            stabilization_mode="always", committed_cache_source="canonical_recheck"
        )
        summary = summarize_decision_stabilization(
            [{"profiling_repeat": 0, **row}], [_sample(formal_ms=14)]
        )
        self.assertEqual(summary["formal_end_to_end_cuda_elapsed_ms"]["median"], 14)
        self.assertEqual(summary["speedup_relative_to_original_dflash"], 20 / 14)


class Phase2C3RunnerTests(unittest.TestCase):
    def _output_oracle(self, **updates):
        inputs = {
            "token_exact_required": True,
            "reference_tree_output_ids_equal": True,
            "tree_control_repeat_output_ids_equal_reference": True,
            "control_profile_output_ids_equal": True,
            "reference_repeat_output_ids_equal": True,
            "reference_profile_output_ids_equal": True,
            "target_only_reference_output_ids_equal": True,
            "target_only_tree_output_ids_equal": True,
        }
        inputs.update(updates)
        return evaluate_output_oracle(**inputs)

    def test_target_only_divergence_is_auxiliary_when_reference_matches_tree(
        self,
    ) -> None:
        result = self._output_oracle(
            target_only_reference_output_ids_equal=False,
            target_only_tree_output_ids_equal=False,
        )
        self.assertTrue(result["hard_output_relations_passed"])
        self.assertIsNone(result["hard_failure_class"])
        self.assertEqual(
            result["target_only_auxiliary_status"],
            "target_only_reference_numerical_divergence",
        )

    def test_reference_tree_divergence_fails_token_exact_oracle(self) -> None:
        result = self._output_oracle(reference_tree_output_ids_equal=False)
        self.assertFalse(result["hard_output_relations_passed"])
        self.assertEqual(
            result["hard_failure_class"], "reference_tree_output_divergence"
        )

    def test_reference_repeat_or_profile_instability_fails(self) -> None:
        for field in (
            "reference_repeat_output_ids_equal",
            "reference_profile_output_ids_equal",
        ):
            with self.subTest(field=field):
                result = self._output_oracle(**{field: False})
                self.assertFalse(result["hard_output_relations_passed"])
                self.assertEqual(
                    result["hard_failure_class"],
                    "reference_control_profile_output_divergence",
                )

    def test_target_only_equality_remains_recorded(self) -> None:
        result = self._output_oracle()
        self.assertTrue(result["target_only_reference_output_ids_equal"])
        self.assertTrue(result["target_only_tree_output_ids_equal"])
        self.assertEqual(result["target_only_auxiliary_status"], "equal")
        self.assertIsNone(result["target_only_auxiliary_warning"])

    def test_cli_defaults_and_overrides(self) -> None:
        default = _args()
        self.assertEqual(default.decision_stabilization, "off")
        self.assertEqual(default.decision_margin_threshold, 0.1)
        custom = _args(
            "--decision-stabilization",
            "margin",
            "--decision-margin-threshold",
            "0.2",
        )
        self.assertEqual(custom.decision_stabilization, "margin")
        self.assertEqual(custom.decision_margin_threshold, 0.2)

    def test_invalid_margin_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_decision_stabilization("margin", -0.1)

    def test_target_only_passes_attention_mask_and_suppresses_sampling_knobs(
        self,
    ) -> None:
        class Input:
            shape = (1, 3)

            def new_ones(self, shape):
                return ("ones", shape)

        target = mock.Mock()
        target.generate.return_value = "ids"
        result = _target_only_greedy(
            target=target, input_ids=Input(), max_new_tokens=4, eos_token_id=9
        )
        self.assertEqual(result, "ids")
        kwargs = target.generate.call_args.kwargs
        self.assertEqual(kwargs["attention_mask"], ("ones", (1, 3)))
        self.assertFalse(kwargs["do_sample"])
        self.assertIsNone(kwargs["temperature"])
        self.assertIsNone(kwargs["top_p"])
        self.assertIsNone(kwargs["top_k"])

    def test_stabilization_artifacts_and_schemas(self) -> None:
        rows, summary, frontier = build_stabilization_artifacts([_round()], [_sample()])
        root = REPO_ROOT / "docs" / "schemas"
        round_schema = json.loads(
            (root / "phase2c-decision-stabilization-round.schema.json").read_text()
        )
        summary_schema = json.loads(
            (root / "phase2c-decision-stabilization-summary.schema.json").read_text()
        )
        Draft202012Validator.check_schema(round_schema)
        Draft202012Validator.check_schema(summary_schema)
        Draft202012Validator(round_schema).validate(rows[0])
        Draft202012Validator(summary_schema).validate(summary)
        self.assertEqual(len(frontier), 7)

    def test_all_new_artifact_files_are_created_on_early_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            argv = [
                "--dataset-jsonl",
                str(Path(directory) / "missing.jsonl"),
                "--materialized-draft-path",
                str(Path(directory) / "draft"),
                "--output-dir",
                str(output),
            ]
            with (
                mock.patch(
                    "experiments.phase2c_online_tree.collect_environment",
                    return_value={"repository": {"commit": "a", "branch": "b"}},
                ),
                mock.patch("experiments.phase2c_online_tree._preflight_offline_inputs"),
                mock.patch(
                    "experiments.phase2c_online_tree._load_reference_and_run",
                    side_effect=RuntimeError("synthetic"),
                ),
            ):
                self.assertEqual(phase2c_main(argv), 1)
            for name in (
                "decision-stabilization-rounds.jsonl",
                "decision-stabilization-summary.json",
                "decision-stabilization-summary.csv",
                "decision-margin-frontier.csv",
                "canonical-recheck-events.jsonl",
                "reference-state-differential.jsonl",
                "reference-state-differential.csv",
                "reference-state-summary.json",
                "first-reference-state-divergence.json",
                "reference-state-layer-differential.jsonl",
            ):
                self.assertTrue((output / name).is_file(), name)

    def test_materialization_includes_stabilization_dependency(self) -> None:
        sources = _effective_sources("instrumented")
        self.assertIn("phase2c_stabilization.py", sources)
        self.assertEqual(sources["phase2c_stabilization.py"].name, "stabilization.py")

    def test_dynamic_source_commits_canonical_cache_before_output_progress(
        self,
    ) -> None:
        source = (
            REPO_ROOT / "third_party/qwen3_8b_dflash_b16/instrumented/dflash.py"
        ).read_text()
        self.assertIn('"committed_cache_source": "canonical_recheck"', source)
        self.assertIn("past_key_values_target = canonical", source)
        self.assertIn("phase2c.canonical_recheck", source)

    def test_dynamic_source_uses_integer_diagnostic_indices(self) -> None:
        source = (
            REPO_ROOT / "third_party/qwen3_8b_dflash_b16/instrumented/dflash.py"
        ).read_text()
        self.assertIn("evenly_spaced_sample_indices", source)
        self.assertNotIn("torch.linspace(", source)

    def test_lazy_import_does_not_load_torch_or_transformers(self) -> None:
        code = (
            "import sys; import dflash.phase2c.stabilization; "
            "import experiments.phase2c_online_tree; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        )
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
