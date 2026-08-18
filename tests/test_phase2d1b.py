from __future__ import annotations

import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jsonschema import Draft202012Validator

from dflash.phase2d1.features import (
    FORBIDDEN_RUNTIME_FIELD_FRAGMENTS,
    RUNTIME_FEATURE_NAMES,
    aggregate_runtime_features,
    build_deployable_feature_rows,
    candidate_positions,
    validate_runtime_feature_names,
)
from dflash.phase2d1.freeze import (
    PHASE2D1_BASELINE_SHA,
    freeze_selector_config,
    validate_phase2d1_freeze_inputs,
)
from dflash.phase2d1.parity import construct_policy_nodes
from dflash.phase2d1.runtime import (
    audit_runtime_inputs,
    select_policy,
    validate_frozen_selector,
)
from dflash.phase2d1b.alignment import build_offline_online_alignment
from dflash.phase2d1b.cohort import build_selector_inventory
from dflash.phase2d1b.correctness import classify_actual_execution_correctness
from dflash.phase2d1b.metrics import (
    aggregate_selector_results,
    build_decision,
    selector_round_record,
)
from experiments.phase2d1b_online_selector import (
    PHASE2C_DIAGNOSTIC_ARTIFACTS,
    ROOT_ARTIFACTS,
    _cross_run_comparison,
    _initialize_semantic_artifacts,
    _phase2d1b_subrun_correctness,
    _preflight,
    _tree_sample,
    parse_args,
)
from scripts.check_phase2d1b_acceptance import (
    _validate_correctness,
)
from scripts.check_phase2d1b_acceptance import (
    check_output as check_phase2d1b_output,
)
from scripts.check_phase2d1b_acceptance import main as phase2d1b_checker_main
from tests.test_phase2d1 import _selector_events

REPO_ROOT = Path(__file__).resolve().parents[1]


def _frozen() -> dict:
    events = _selector_events()
    return freeze_selector_config(
        events,
        phase2d1={
            "artifact_hashes": {"selector-events.jsonl": "a" * 64},
            "manifest": {
                "configuration": {
                    "minimum_leaf_support": 1,
                    "minimum_leaf_source_support": 1,
                    "stump_max_thresholds_per_feature": 64,
                }
            },
        },
        phase2d0={
            "directory_sha256": "b" * 64,
            "artifact_sha256": {"opportunity-events.jsonl": "c" * 64},
        },
        validated_capture={
            "inputs": [
                {
                    "task": task,
                    "capture_directory_sha256": character * 64,
                }
                for task, character in (
                    ("gov_report", "d"),
                    ("qasper", "e"),
                    ("hotpotqa", "f"),
                )
            ]
        },
    )


def _positions() -> list[dict]:
    return candidate_positions(
        [[depth * 100 + rank for rank in range(16)] for depth in range(15)],
        [[-0.1 * (rank + 1) for rank in range(16)] for _ in range(15)],
    )


class FrozenSelectorTests(unittest.TestCase):
    def test_freeze_is_deterministic_and_pinned(self) -> None:
        left = _frozen()
        right = _frozen()
        self.assertEqual(left, right)
        self.assertEqual(left["phase2d1_commit_sha"], PHASE2D1_BASELINE_SHA)
        self.assertEqual(left["top_m"], 16)

    def test_config_hash_detects_mutation(self) -> None:
        config = _frozen()
        config["training_source_ids"].append("leak")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_frozen_selector(config)

    def test_freeze_rejects_phase2d0_provenance_mismatch(self) -> None:
        events = _selector_events()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "selector-events.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in events)
            )
            common = {"decision": "integrate_lightweight_selector_online"}
            (root / "selector-decision.json").write_text(json.dumps(common))
            (root / "research-question-summary.json").write_text(
                json.dumps(common)
            )
            (root / "analysis-manifest.json").write_text(
                json.dumps(
                    {
                        **common,
                        "status": "ok",
                        "repository": {"commit_sha": PHASE2D1_BASELINE_SHA},
                        "phase2d0_input": {"directory_sha256": "expected"},
                        "capture_inputs": [
                            {"capture_directory_sha256": "capture"}
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "Phase 2D.0"):
                validate_phase2d1_freeze_inputs(
                    root,
                    rebuilt_events=events,
                    phase2d0={"directory_sha256": "different"},
                    validated_capture={
                        "inputs": [
                            {"capture_directory_sha256": "capture"}
                        ]
                    },
                )

    def test_rules_are_target_blind_and_task_agnostic(self) -> None:
        config = _frozen()
        self.assertTrue(config["target_blind"])
        self.assertFalse(config["task_id_used_as_feature"])
        self.assertFalse(config["target_dependent_fields_present"])
        self.assertEqual(sorted(map(int, config["node_budget_rules"])), [16, 32])

    def test_global_fixed_matches_realized_policy_family(self) -> None:
        config = _frozen()
        for budget in (16, 32):
            decision = select_policy(
                config,
                mode="global_fixed",
                node_budget=budget,
                context_length=1024,
                runtime_features=None,
            )
            self.assertEqual(decision["selected_policy"]["policy"], "layerwise_beam")
            self.assertEqual(
                decision["selected_policy"]["policy_parameters"]["beam_width"],
                4 if budget == 16 else 8,
            )

    def test_schema_validates(self) -> None:
        schema = json.loads(
            (REPO_ROOT / "docs/schemas/phase2d1b-frozen-selector.schema.json").read_text()
        )
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(_frozen())


class CanonicalFeatureAndRuntimeTests(unittest.TestCase):
    def test_runtime_schema_has_no_forbidden_field(self) -> None:
        validate_runtime_feature_names(RUNTIME_FEATURE_NAMES)
        self.assertFalse(
            any(
                fragment in name
                for name in RUNTIME_FEATURE_NAMES
                for fragment in FORBIDDEN_RUNTIME_FIELD_FRAGMENTS
            )
        )

    def test_position_and_aggregate_features_are_exact(self) -> None:
        positions = _positions()
        from dflash.phase2b.tree import score_beam_tree

        nodes = score_beam_tree(positions, budget=16)
        rows = build_deployable_feature_rows(
            positions=positions,
            canonical_nodes=nodes,
            context_length=1024,
            top_m=16,
            node_budget=16,
        )
        features = aggregate_runtime_features(
            rows,
            context_length=1024,
            node_budget=16,
            policy_configuration_count=9,
        )
        self.assertEqual(tuple(features), RUNTIME_FEATURE_NAMES)
        self.assertEqual(features["context_length"], 1024.0)
        self.assertAlmostEqual(features["mean_top1_top2_margin"], 0.1)

    def test_target_runtime_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "target-dependent"):
            audit_runtime_inputs({"target_logits": [1], "context_length": 1024})
        features = {name: 0.0 for name in RUNTIME_FEATURE_NAMES}
        features["target_logits"] = 1.0
        with self.assertRaisesRegex(ValueError, "target-dependent"):
            select_policy(
                _frozen(),
                mode="decision_stump",
                node_budget=16,
                context_length=1024,
                runtime_features=features,
            )

    def test_stump_equal_threshold_goes_left(self) -> None:
        config = _frozen()
        rule = config["node_budget_rules"]["16"]["decision_stump_rule"]
        self.assertTrue(rule["stump_available"])
        features = {name: 0.0 for name in RUNTIME_FEATURE_NAMES}
        features[rule["feature_name"]] = rule["threshold"]
        decision = select_policy(
            config,
            mode="decision_stump",
            node_budget=16,
            context_length=1024,
            runtime_features=features,
        )
        self.assertTrue(decision["comparison_result"])
        self.assertEqual(decision["selected_policy_config"], rule["left_policy_config"])

    def test_nonfinite_stump_feature_uses_deterministic_fallback(self) -> None:
        config = _frozen()
        rule = config["node_budget_rules"]["16"]["decision_stump_rule"]
        for value in (math.nan, math.inf, -math.inf):
            features = {name: 0.0 for name in RUNTIME_FEATURE_NAMES}
            features[rule["feature_name"]] = value
            decision = select_policy(
                config,
                mode="decision_stump",
                node_budget=16,
                context_length=1024,
                runtime_features=features,
            )
            self.assertTrue(decision["fallback_used"])
            self.assertEqual(decision["fallback_reason"], "non_finite_feature")

    def test_context_unknown_uses_global_fallback_without_task(self) -> None:
        config = _frozen()
        decision = select_policy(
            config,
            mode="context_rule",
            node_budget=16,
            context_length=12345,
            runtime_features=None,
        )
        self.assertTrue(decision["fallback_used"])
        self.assertEqual(decision["fallback_reason"], "unseen_context_length")
        self.assertEqual(decision["runtime_input_keys"], [])

    def test_all_allowlisted_policies_construct_within_cap(self) -> None:
        config = _frozen()
        for budget, rule in config["node_budget_rules"].items():
            for policy in rule["allowed_policy_configurations"]:
                nodes = construct_policy_nodes(_positions(), policy)
                self.assertLessEqual(len(nodes), int(budget))


class CohortTests(unittest.TestCase):
    def _prepared(self) -> list[dict]:
        return [
            {
                "source_sample_id": source,
                "sample_id": f"gov_report-{source}",
                "document_token_length": length,
                "fixed_prompt_length": 100,
                "source_full_prompt_length": length,
            }
            for source, length in (("source-1", 33000), ("new-1", 33000), ("new-2", 9000))
        ]

    def test_training_sources_are_excluded_before_inventory(self) -> None:
        config = _frozen()
        filtered, inventory = build_selector_inventory(
            self._prepared(),
            frozen=config,
            task="gov_report",
            nominal_lengths=(1024, 32768),
            requested_sample_count=8,
            exclude_training_sources=True,
        )
        self.assertNotIn("source-1", {row["source_sample_id"] for row in filtered})
        self.assertTrue(
            inventory["training_source_exclusion_completed_before_model_loading"]
        )
        self.assertFalse(inventory["training_sources_reintroduced"])

    def test_mechanics_and_independent_cohorts_are_distinct(self) -> None:
        config = _frozen()
        _, mechanics = build_selector_inventory(
            self._prepared(), frozen=config, task="gov_report", nominal_lengths=(1024,), requested_sample_count=1, exclude_training_sources=False
        )
        _, independent = build_selector_inventory(
            self._prepared(), frozen=config, task="gov_report", nominal_lengths=(1024,), requested_sample_count=1, exclude_training_sources=True
        )
        self.assertEqual(mechanics["evaluation_cohort"], "in_sample_mechanics")
        self.assertEqual(independent["evaluation_cohort"], "independent_evaluation")


class MetricsAndAlignmentTests(unittest.TestCase):
    def _source_round(self, *, nodes: int = 16) -> dict:
        return {
            "task": "gov_report",
            "sample_id": "sample",
            "source_sample_id": "source-1",
            "source_text_sha256": "1" * 64,
            "nominal_prompt_length": 1024,
            "actual_prompt_length": 1020,
            "profiling_repeat": 0,
            "round_index": 0,
            "generated_root_offset": 0,
            "accepted_draft_token_count": 2,
            "committed_token_count": 3,
            "terminal_round": False,
            "truncated_token_count": 0,
            "actual_node_count": nodes,
            "tree_metadata": {"nodes": [{"token_id": 42}]},
            "timing": {
                "round_cuda_elapsed_ms": 4.0,
                "total_draft_cuda_elapsed_ms": 1.0,
                "total_verification_cuda_elapsed_ms": 2.0,
                "kv_compaction_cuda_elapsed_ms": 0.5,
            },
            "online_selector": None,
        }

    def test_accepted_and_committed_semantics(self) -> None:
        row = selector_round_record(
            self._source_round(), mode="top1_packed_tree", node_budget=16
        )
        self.assertEqual(row["committed_token_count"], 1 + row["accepted_draft_token_count"])
        malformed = self._source_round()
        malformed["committed_token_count"] = 2
        with self.assertRaisesRegex(ValueError, "root plus"):
            selector_round_record(malformed, mode="top1_packed_tree", node_budget=16)

    def test_n16_and_n32_cost_flags(self) -> None:
        n16 = selector_round_record(self._source_round(), mode="top1_packed_tree", node_budget=16)
        n32 = selector_round_record(self._source_round(nodes=32), mode="global_fixed", node_budget=32)
        self.assertTrue(n16["same_actual_node_count_as_top1"])
        self.assertFalse(n32["same_actual_node_count_as_top1"])

    def test_selector_cpu_sync_is_retained(self) -> None:
        source = self._source_round()
        source["online_selector"] = {
            "selected_policy_config": "x",
            "feature_value": 1.0,
            "fallback_used": False,
            "decision_boundary_nearby": False,
            "cpu_timing": {
                "draft_topk_device_to_host_sync_cpu_ms": 2.0,
                "feature_extraction_cpu_ms": 1.0,
                "decision_cpu_ms": 0.5,
                "canonical_policy_construction_cpu_ms": 0.25,
                "tree_tensor_pack_host_to_device_cpu_ms": 0.25,
            },
        }
        row = selector_round_record(source, mode="global_fixed", node_budget=16)
        self.assertEqual(row["selector_latency"], 4.0)
        self.assertEqual(row["draft_topk_device_to_host_sync_cpu_ms"], 2.0)

    def test_decision_priorities(self) -> None:
        summary = {"aggregates": []}
        self.assertEqual(
            build_decision(summary, correctness_passed=False, independent_evidence=True, minimum_accepted_gain=0, minimum_latency_improvement_fraction=0, stump_minimum_additional_improvement_fraction=0.01)["decision"],
            "correctness_failure",
        )
        self.assertEqual(
            build_decision(summary, correctness_passed=True, independent_evidence=False, minimum_accepted_gain=0, minimum_latency_improvement_fraction=0, stump_minimum_additional_improvement_fraction=0.01)["decision"],
            "insufficient_independent_evidence",
        )

    def test_aggregate_metrics_exclude_terminal_round(self) -> None:
        primary = selector_round_record(
            self._source_round(), mode="top1_packed_tree", node_budget=16
        )
        terminal = {
            **primary,
            "round_index": 1,
            "terminal": True,
            "right_censored": True,
            "truncated_token_count": 2,
        }
        samples = [
            {
                "task": "gov_report",
                "sample_id": "sample",
                "source_sample_id": "source-1",
                "nominal_prompt_length": 1024,
                "selector_mode": "top1_packed_tree",
                "node_budget_cap": 16,
                "generated_token_count": 8,
                "control_repeats": [{"cpu_wall_elapsed_ms": 8.0}],
            }
        ]
        summary = aggregate_selector_results(samples, [primary, terminal])
        aggregate = summary["aggregates"][0]
        self.assertEqual(aggregate["acceptance_eligible_round_count"], 1)
        self.assertEqual(aggregate["terminal_round_count"], 1)
        self.assertEqual(aggregate["truncated_token_count"], 2)
        self.assertEqual(aggregate["latency_per_returned_token_ms"], 1.0)

    def test_alignment_never_forces_different_root(self) -> None:
        config = _frozen()
        event = next(row for row in _selector_events() if row["node_budget_cap"] == 16)
        decision = select_policy(config, mode="global_fixed", node_budget=16, context_length=event["nominal_prompt_length"], runtime_features=event["runtime_features"])
        online = {
            "task": event["task"], "sample_id": event["sample_id"], "source_sample_id": event["source_sample_id"], "source_text_sha256": event["source_text_sha256"],
            "nominal_prompt_length": event["nominal_prompt_length"], "profiling_repeat": 0, "round_index": event["round_index"], "generated_root_offset": 1,
            "root_token_id": 42, "node_budget_cap": 16, "selector_mode": "global_fixed", "selected_policy_config": decision["selected_policy_config"], "selector_feature_value": None, "accepted_draft_token_count": 1,
        }
        capture = {
            "task": event["task"], "source_sample_id": event["source_sample_id"], "source_text_sha256": event["source_text_sha256"], "nominal_prompt_length": event["nominal_prompt_length"], "generation_repeat": 0, "round_index": event["round_index"], "round_output_start_offset": 0, "root_token_id": 42,
        }
        rows, summary = build_offline_online_alignment(online_rounds=[online], offline_events=[event], captures=[capture], frozen=config)
        self.assertFalse(rows[0]["alignment_valid"])
        self.assertEqual(summary["different_round_segmentation_forced_matches"], 0)


class ActualExecutionCorrectnessTests(unittest.TestCase):
    def _correctness(self, **updates: object) -> dict:
        value = {
            "reference_output_ids_equal": True,
            "original_dflash_reference_tree_output_ids_equal": True,
            "control_profile_output_ids_equal": True,
            "gold_output_ids_equal": True,
            "top1_round_exact_equal": True,
            "gold_status": "fail_semantic",
            "gold_failure_class": "numerical_backend_variation",
            "gold_uncompleted_checks": [],
            "output_divergence_failure_class": None,
        }
        value.update(updates)
        return value

    def test_numerical_backend_variation_without_actual_consequence_passes(
        self,
    ) -> None:
        result = classify_actual_execution_correctness(
            self._correctness(),
            gold_logits=[
                {
                    "round_index": 0,
                    "structural_status": "pass",
                    "semantic_status": "pass",
                    "numerical_status": "outside_strict_thresholds",
                    "argmax_equal": True,
                    "top8_ordered_equal": False,
                }
            ],
            gold_kv=[
                {
                    "round_index": 0,
                    "structural_status": "pass",
                    "semantic_status": "pass",
                    "numerical_status": "within_strict_thresholds",
                }
            ],
            traversal_checks=[
                {
                    "round_index": 0,
                    "status": "pass",
                    "checks": {"all_checks_passed": True},
                }
            ],
            gold_checks_requested=True,
        )
        self.assertTrue(result["actual_execution_correctness_passed"])
        self.assertTrue(result["diagnostic_numerical_variation_observed"])
        self.assertTrue(result["gold_fail_semantic_treated_as_diagnostic"])
        self.assertEqual(result["hard_correctness_failure_reasons"], [])
        self.assertTrue(
            _phase2d1b_subrun_correctness(
                {
                    "all_checks_passed": True,
                    "actual_execution_correctness_passed": result[
                        "actual_execution_correctness_passed"
                    ],
                    "all_required_token_exact_checks_passed": True,
                    "observed_native_output_divergence": False,
                }
            )
        )

    def test_actual_output_or_structural_failure_remains_hard(self) -> None:
        output = classify_actual_execution_correctness(
            self._correctness(
                original_dflash_reference_tree_output_ids_equal=False,
                gold_output_ids_equal=False,
            ),
            gold_logits=[],
            gold_kv=[],
            traversal_checks=[],
            gold_checks_requested=True,
        )
        self.assertFalse(output["actual_execution_correctness_passed"])
        structural = classify_actual_execution_correctness(
            self._correctness(),
            gold_logits=[
                {
                    "round_index": 2,
                    "structural_status": "fail",
                    "semantic_status": "pass",
                }
            ],
            gold_kv=[],
            traversal_checks=[],
            gold_checks_requested=True,
        )
        self.assertFalse(structural["actual_execution_correctness_passed"])
        semantic = classify_actual_execution_correctness(
            self._correctness(),
            gold_logits=[
                {
                    "round_index": 3,
                    "structural_status": "pass",
                    "semantic_status": "fail",
                }
            ],
            gold_kv=[],
            traversal_checks=[],
            gold_checks_requested=True,
        )
        self.assertFalse(semantic["actual_execution_correctness_passed"])

    def test_acceptance_checker_rejects_reference_tree_output_mismatch(self) -> None:
        correctness = {
            "all_checks_passed": True,
            "target_blind_invariants_passed": True,
            "actual_execution_correctness_passed": True,
            "actual_output_exactness_required": True,
            "records": [{"phase2d1b_online_correctness_passed": True}],
        }
        cross_run = {
            "all_control_profile_output_ids_equal": True,
            "all_original_dflash_reference_tree_output_ids_equal": False,
            "reference_output_stable_across_subruns": True,
            "tree_output_stable_across_subruns": True,
        }
        with self.assertRaisesRegex(ValueError, "output stability"):
            _validate_correctness(correctness, cross_run)


class RunnerContractTests(unittest.TestCase):
    def test_cli_defaults_and_overrides_are_lazy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "gov_report.jsonl"
            dataset.write_text("{}\n")
            config = root / "frozen.json"
            config.write_text(json.dumps(_frozen()))
            before = set(sys.modules)
            args = parse_args(
                [
                    "--dataset-jsonl", str(dataset),
                    "--inspect-workload-only",
                    "--frozen-selector-config", str(config),
                    "--online-selector-mode", "global_fixed",
                    "--selector-node-budget", "16",
                ]
            )
            self.assertEqual(args.selector_node_budgets, (16,))
            self.assertEqual(args.online_selector_modes, ("global_fixed",))
            self.assertNotIn("torch", set(sys.modules) - before)
            self.assertNotIn("transformers", set(sys.modules) - before)

    def test_inventory_only_stops_before_strict_cohort_or_model_path(self) -> None:
        prepared = [
            {
                "source_sample_id": "independent",
                "sample_id": "gov_report-independent",
                "document_token_length": 2000,
                "fixed_prompt_length": 10,
                "source_full_prompt_length": 2000,
            }
        ]
        args = SimpleNamespace(
            longbench_task="gov_report",
            context_lengths=(1024, 32768),
            pilot_samples=8,
            exclude_selector_training_sources=True,
            inspect_workload_only=True,
        )
        with (
            patch(
                "experiments.phase2d1b_online_selector._load_tokenizer_only",
                return_value=(object(), None, None),
            ),
            patch(
                "experiments.phase2d1b_online_selector._prepare_workload",
                return_value=(prepared, None, None, None),
            ),
            patch(
                "experiments.phase2d1b_online_selector.build_longbench_workloads"
            ) as cohort_builder,
        ):
            inventory, events, captures = _preflight(args, _frozen())
        cohort_builder.assert_not_called()
        self.assertTrue(inventory["independent_32k_evidence_unavailable"])
        self.assertEqual(events, [])
        self.assertEqual(captures, [])

    def test_instrumented_selector_precedes_target_verification(self) -> None:
        source = (
            REPO_ROOT / "third_party/qwen3_8b_dflash_b16/instrumented/dflash.py"
        ).read_text()
        decision = source.index("selector_record = select_policy(")
        verification = source.index("packed_output = target(", decision)
        self.assertLess(decision, verification)
        self.assertIn('"target_forward_count_added"] = 0', source)

    def test_materialization_includes_shared_runtime_modules(self) -> None:
        from dflash.phase1.materialize import _effective_sources

        sources = _effective_sources("instrumented")
        self.assertIn("phase2d1_features.py", sources)
        self.assertIn("phase2d1_selector.py", sources)
        self.assertIn("phase2d1_tree_policies.py", sources)

    def test_all_phase2d1b_schemas_are_valid(self) -> None:
        schemas = sorted((REPO_ROOT / "docs/schemas").glob("phase2d1b-*.schema.json"))
        self.assertGreaterEqual(len(schemas), 5)
        for path in schemas:
            Draft202012Validator.check_schema(json.loads(path.read_text()))

    def test_cross_run_comparison_preserves_correctness_signals(self) -> None:
        value = _cross_run_comparison(
            [{"mode": "global_fixed", "budget": 16}],
            [
                {
                    "task": "gov_report",
                    "sample_id": "sample",
                    "nominal_prompt_length": 1024,
                    "selector_mode": "global_fixed",
                    "node_budget_cap": 16,
                    "reference_output_token_ids_sha256": "a" * 64,
                    "generated_output_token_ids_sha256": "a" * 64,
                    "control_profile_output_ids_equal": True,
                    "reference_output_ids_equal": True,
                }
            ],
        )
        self.assertTrue(value["all_control_profile_output_ids_equal"])
        self.assertTrue(
            value["all_original_dflash_reference_tree_output_ids_equal"]
        )
        self.assertTrue(value["reference_output_stable_across_subruns"])
        self.assertTrue(value["tree_output_stable_across_subruns"])

    def test_tree_sample_uses_original_dflash_tree_oracle(self) -> None:
        row = {
            "task": "gov_report",
            "sample_id": "sample",
            "source_sample_id": "source",
            "nominal_prompt_length": 1024,
            "actual_prompt_length": 1020,
            "generated_token_count": 2,
            "generated_output_token_ids_sha256": "a" * 64,
            "control": {
                "repeats": [{"cpu_wall_elapsed_ms": 2.0}],
            },
            "profiling": {
                "round_count_mean": 1,
                "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0,
                "control_profile_output_ids_equal": True,
            },
            "reference": {
                "output_ids_equal": True,
                "generated_output_token_ids_sha256": "b" * 64,
            },
            "correctness": {
                "original_dflash_reference_tree_output_ids_equal": False
            },
        }
        sample = _tree_sample(row, mode="global_fixed", budget=16)
        self.assertFalse(sample["reference_output_ids_equal"])

    def test_shadow_output_divergence_is_a_phase2d1b_failure(self) -> None:
        self.assertFalse(
            _phase2d1b_subrun_correctness(
                {
                    "all_checks_passed": True,
                    "actual_execution_correctness_passed": False,
                    "all_required_token_exact_checks_passed": True,
                    "observed_native_output_divergence": True,
                }
            )
        )
        self.assertTrue(
            _phase2d1b_subrun_correctness(
                {
                    "all_checks_passed": True,
                    "actual_execution_correctness_passed": True,
                    "all_required_token_exact_checks_passed": True,
                    "observed_native_output_divergence": False,
                }
            )
        )

    def test_failure_atomic_artifact_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            _initialize_semantic_artifacts(output)
            required = set(ROOT_ARTIFACTS + PHASE2C_DIAGNOSTIC_ARTIFACTS)
            initialized_elsewhere = {
                "manifest.json",
                "workload-inventory.json",
                "frozen-selector.json",
                "environment.json",
                "source-provenance.json",
            }
            self.assertTrue(
                all(
                    (output / name).exists()
                    for name in required - initialized_elsewhere
                )
            )

    def test_checker_reports_started_run_before_schema_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "manifest.json").write_text(
                json.dumps({"schema_version": "1.0", "status": "started"})
            )
            with self.assertRaisesRegex(ValueError, "incomplete.*status='started'"):
                check_phase2d1b_output(output)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                return_code = phase2d1b_checker_main(
                    ["--output-dir", str(output)]
                )
            self.assertEqual(return_code, 1)
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertIn('"status": "fail"', stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
