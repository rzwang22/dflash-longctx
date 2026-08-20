from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from jsonschema import Draft202012Validator

from dflash.phase2c.gold import kv_cache_shape_metadata
from dflash.phase2d2.freeze import (
    freeze_refinement_config,
    validate_frozen_refinement,
)
from dflash.phase2d2.metrics import (
    build_phase2d2_decision,
    candidate_quality_for_round,
)
from dflash.phase2d2.refinement import (
    audit_prefix_selection_inputs,
    build_conditional_remask_inputs,
    build_refined_tree,
    select_refinement_prefixes,
    validate_refinement_options,
)
from dflash.phase2d2.robustness import (
    CALIBRATION_SCHEMA_VERSION,
    calibration_sha256,
    classify_guard_event,
    validate_nr_calibration,
)
from dflash.phase2d2.tensor import repeat_dynamic_cache_batch
from experiments.phase2d2_robust_conditional_refinement import (
    BASE_CONTROL_SELECTOR_MODE,
    GOLD_ARTIFACTS,
    ROOT_ARTIFACTS,
    _comparison,
    _generation_options,
    _output_relations,
    _persist,
    _sample_timing_row,
    build_parser,
)
from scripts.aggregate_phase2d2_decision import _stable_positive_by_task
from scripts.check_phase2d2_acceptance import check_output

REPO_ROOT = Path(__file__).resolve().parents[1]


def _positions() -> list[dict]:
    return [
        {
            "position_index": position,
            "candidates": [
                {
                    "token_id": position * 100 + rank,
                    "log_probability": -0.1 * rank,
                    "rank": rank,
                }
                for rank in range(1, 17)
            ],
        }
        for position in range(1, 16)
    ]


def _seed() -> set[tuple[int, ...]]:
    prefix: tuple[int, ...] = ()
    nodes = {prefix}
    for position in range(1, 16):
        prefix += (position * 100 + 1,)
        nodes.add(prefix)
    return nodes


def _selected(prefix=(101, 201)) -> list[dict]:
    return [
        {
            "prefix_token_ids": list(prefix),
            "refine_depth": len(prefix),
            "base_cumulative_log_probability": -0.2,
            "base_mean_log_probability": -0.1,
            "terminal_base_rank": 1,
            "selection_uses_target_information": False,
            "selection_features": "draft-only",
            "refined_prefix_selection_rank": 1,
        }
    ]


def _conditional(prefix=(101, 201)) -> dict:
    positions = []
    for position in range(1, 16):
        rows = [
            {
                "token_id": position * 100 + rank,
                "base_rank": rank,
                "conditional_rank": rank if rank <= 2 else None,
                "base_log_probability": -0.1 * rank,
                "conditional_log_probability": -0.2 * rank,
            }
            for rank in range(1, 17)
        ]
        rows.append(
            {
                "token_id": position * 100 + 99,
                "base_rank": None,
                "conditional_rank": 1,
                "base_log_probability": -0.01,
                "conditional_log_probability": -0.01,
            }
        )
        positions.append(rows)
    return {tuple(prefix): positions}


class ConditionalConstructionTests(unittest.TestCase):
    def test_exact_prefix_suffix_mask_construction(self) -> None:
        rows = build_conditional_remask_inputs(
            root_token_id=7,
            prefixes=[(11, 12), (21, 22)],
            mask_token_id=151669,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][:3], [7, 11, 12])
        self.assertEqual(rows[0][3:], [151669] * 13)
        self.assertEqual(len(rows[0]), 16)

    def test_root_occupies_only_structural_position_zero(self) -> None:
        row = build_conditional_remask_inputs(
            root_token_id=7, prefixes=[(11,)], mask_token_id=151669
        )[0]
        self.assertEqual(row[0], 7)
        self.assertEqual(row[1], 11)

    def test_external_and_internal_batches_are_separate(self) -> None:
        config = validate_refinement_options(
            enabled=True,
            merge_mode="union",
            base_top_m=16,
            conditional_top_r=2,
            refine_depth=2,
            refined_prefix_budget=4,
            refinement_weight=0.5,
            node_budget=16,
            block_size=16,
            external_batch_size=1,
            verification_mode="always_replay",
            calibration_validated=False,
        )
        self.assertEqual(config["external_batch_size"], 1)
        self.assertEqual(config["refined_prefix_budget"], 4)

    def test_invalid_external_batch_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "external request batch"):
            validate_refinement_options(
                enabled=True,
                merge_mode="union",
                base_top_m=16,
                conditional_top_r=2,
                refine_depth=2,
                refined_prefix_budget=2,
                refinement_weight=0.5,
                node_budget=16,
                block_size=16,
                external_batch_size=2,
                verification_mode="always_replay",
                calibration_validated=False,
            )


class PrefixAndMergeTests(unittest.TestCase):
    def test_prefix_selection_is_target_blind_and_deterministic(self) -> None:
        left = select_refinement_prefixes(
            seed_nodes=_seed(),
            base_positions=_positions(),
            refine_depth=2,
            prefix_budget=2,
        )
        right = select_refinement_prefixes(
            seed_nodes=sorted(_seed(), reverse=True),
            base_positions=deepcopy(_positions()),
            refine_depth=2,
            prefix_budget=2,
        )
        self.assertEqual(left, right)
        self.assertFalse(left[0]["selection_uses_target_information"])

    def test_target_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "target-dependent"):
            audit_prefix_selection_inputs(
                {"draft_log_probability": -1.0, "target_token_ids": [1]}
            )

    def test_base_only_preserves_exact_seed_tree(self) -> None:
        tree = build_refined_tree(
            seed_nodes=_seed(),
            base_positions=_positions(),
            selected_prefixes=_selected(),
            conditional_candidates={},
            merge_mode="base_only",
            refinement_weight=0.5,
            node_budget=16,
        )
        self.assertEqual({tuple(row["path"]) for row in tree["nodes"]}, _seed())
        self.assertEqual(tree["actual_total_node_count"], 16)

    def test_conditional_rerank_does_not_add_availability(self) -> None:
        tree = build_refined_tree(
            seed_nodes=_seed(),
            base_positions=_positions(),
            selected_prefixes=_selected(),
            conditional_candidates=_conditional(),
            merge_mode="conditional_rerank",
            refinement_weight=0.5,
            node_budget=16,
        )
        self.assertFalse(
            any(
                row.get("candidate_provenance") == "refined_only"
                for row in tree["nodes"]
            )
        )

    def test_union_retains_refined_only_provenance(self) -> None:
        tree = build_refined_tree(
            seed_nodes=_seed(),
            base_positions=_positions(),
            selected_prefixes=_selected(),
            conditional_candidates=_conditional(),
            merge_mode="union",
            refinement_weight=0.5,
            node_budget=16,
        )
        self.assertGreater(tree["refined_only_nodes_selected"], 0)
        self.assertTrue(
            any(
                row.get("candidate_provenance") == "refined_only"
                for row in tree["nodes"]
            )
        )

    def test_fixed_budget_and_prefix_closure(self) -> None:
        for budget in (16, 32):
            tree = build_refined_tree(
                seed_nodes=_seed(),
                base_positions=_positions(),
                selected_prefixes=_selected(),
                conditional_candidates=_conditional(),
                merge_mode="union",
                refinement_weight=0.5,
                node_budget=budget,
            )
            paths = {tuple(row["path"]) for row in tree["nodes"]}
            self.assertEqual(len(paths), budget)
            self.assertIn((), paths)
            self.assertTrue(all(not path or path[:-1] in paths for path in paths))
            self.assertEqual(tree["root_count"], 1)

    def test_merged_tree_ties_are_token_prefix_deterministic(self) -> None:
        conditional = _conditional()
        for position in conditional[(101, 201)]:
            for row in position:
                row["base_log_probability"] = -1.0
                row["conditional_log_probability"] = -1.0
        left = build_refined_tree(
            seed_nodes=_seed(),
            base_positions=_positions(),
            selected_prefixes=_selected(),
            conditional_candidates=conditional,
            merge_mode="union",
            refinement_weight=0.5,
            node_budget=32,
        )
        reversed_rows = {
            key: [list(reversed(position)) for position in positions]
            for key, positions in conditional.items()
        }
        right = build_refined_tree(
            seed_nodes=sorted(_seed(), reverse=True),
            base_positions=deepcopy(_positions()),
            selected_prefixes=_selected(),
            conditional_candidates=reversed_rows,
            merge_mode="union",
            refinement_weight=0.5,
            node_budget=32,
        )
        self.assertEqual(
            [row["path"] for row in left["nodes"]],
            [row["path"] for row in right["nodes"]],
        )

    def test_missing_logit_is_never_zero_filled(self) -> None:
        conditional = _conditional()
        conditional[(101, 201)][2][0]["conditional_log_probability"] = None
        with self.assertRaisesRegex(ValueError, "full-vocabulary"):
            build_refined_tree(
                seed_nodes=_seed(),
                base_positions=_positions(),
                selected_prefixes=_selected(),
                conditional_candidates=conditional,
                merge_mode="union",
                refinement_weight=0.5,
                node_budget=16,
            )


class CacheAndGuardTests(unittest.TestCase):
    class _Tensor:
        def __init__(self, batch: int):
            self.shape = (batch, 2, 3, 4)

        def repeat_interleave(self, repeats: int, dim: int):
            if dim != 0:
                raise AssertionError
            return CacheAndGuardTests._Tensor(self.shape[0] * repeats)

    class _Layer:
        def __init__(self):
            self.keys = CacheAndGuardTests._Tensor(1)
            self.values = CacheAndGuardTests._Tensor(1)

    class _Cache:
        def __init__(self):
            self.layers = [CacheAndGuardTests._Layer(), CacheAndGuardTests._Layer()]

    def test_internal_cache_batch_expansion_isolated_helper(self) -> None:
        cache = self._Cache()
        repeat_dynamic_cache_batch(cache, 4)
        self.assertTrue(all(layer.keys.shape[0] == 4 for layer in cache.layers))

    def test_internal_cache_expansion_does_not_mutate_round_entry_source(self) -> None:
        source = self._Cache()
        temporary = deepcopy(source)
        repeat_dynamic_cache_batch(temporary, 4)
        self.assertTrue(all(layer.keys.shape[0] == 1 for layer in source.layers))
        self.assertTrue(all(layer.keys.shape[0] == 4 for layer in temporary.layers))

    def test_guard_true_false_positive_negative(self) -> None:
        self.assertTrue(
            classify_guard_event(
                guard_triggered=True,
                replay_executed=True,
                packed_replay_decision_equal=False,
                packed_replay_output_equal=False,
            )["guard_true_positive"]
        )
        self.assertTrue(
            classify_guard_event(
                guard_triggered=False,
                replay_executed=True,
                packed_replay_decision_equal=False,
                packed_replay_output_equal=False,
            )["guard_false_negative"]
        )
        self.assertTrue(
            classify_guard_event(
                guard_triggered=True,
                replay_executed=True,
                packed_replay_decision_equal=True,
                packed_replay_output_equal=True,
            )["guard_false_positive"]
        )

    def test_margin_guard_requires_calibration(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a provenance-matched"):
            validate_refinement_options(
                enabled=True,
                merge_mode="union",
                base_top_m=16,
                conditional_top_r=2,
                refine_depth=2,
                refined_prefix_budget=2,
                refinement_weight=0.5,
                node_budget=16,
                block_size=16,
                external_batch_size=1,
                verification_mode="margin_guarded_replay",
                calibration_validated=False,
            )

    def test_always_replay_kv_gold_shape_metadata_is_explicit(self) -> None:
        candidate = [
            {"k": [1, 8, 1024, 128], "v": [1, 8, 1024, 128]},
            {"k": [1, 8, 1024, 128], "v": [1, 8, 1024, 128]},
        ]
        metadata = kv_cache_shape_metadata(
            candidate_shape=candidate,
            reference_shape=deepcopy(candidate),
        )
        self.assertTrue(metadata["shape_metadata_available"])
        self.assertTrue(metadata["shape_layer_count_matches"])
        self.assertTrue(metadata["all_layer_shapes_match"])
        self.assertEqual(metadata["candidate_shape"], candidate)
        self.assertEqual(metadata["reference_shape"], candidate)
        self.assertTrue(
            all(row["k"] and row["v"] for row in metadata["layer_shape_matches"])
        )

    def test_kv_shape_mismatch_remains_structural_evidence(self) -> None:
        metadata = kv_cache_shape_metadata(
            candidate_shape=[{"k": [1, 8, 1024, 128], "v": [1, 8, 1024, 128]}],
            reference_shape=[{"k": [1, 8, 1025, 128], "v": [1, 8, 1025, 128]}],
        )
        self.assertFalse(metadata["all_layer_shapes_match"])
        self.assertFalse(metadata["layer_shape_matches"][0]["k"])

    def test_calibration_provenance_and_hash_are_strict(self) -> None:
        provenance = {
            "gpu_name": "NVIDIA H100 PCIe",
            "compute_capability": "9.0",
            "dtype": "bfloat16",
            "target_model_revision": "a" * 40,
            "draft_model_revision": "b" * 40,
            "torch_version": "2.8.0",
            "transformers_version": "4.57.1",
            "target_attention_implementation": "sdpa",
        }
        calibration = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "source_phase": "2D.1B-NR",
            "independent_calibration_cohort": True,
            "decision_margin_threshold": 0.1,
            "source_artifacts": {},
            "provenance": provenance,
            "evaluation_outputs_used_to_choose_threshold": False,
            "formal_error_bound_claimed": False,
            "empirically_validated_guard_only": True,
        }
        calibration["calibration_sha256"] = calibration_sha256(calibration)
        self.assertEqual(validate_nr_calibration(calibration, provenance), calibration)
        calibration_schema = json.loads(
            (REPO_ROOT / "docs/schemas/phase2d2-nr-calibration.schema.json").read_text()
        )
        Draft202012Validator(calibration_schema).validate(calibration)
        changed = {**provenance, "dtype": "float16"}
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            validate_nr_calibration(calibration, changed)


class MetricsFreezeAndSchemaTests(unittest.TestCase):
    def test_cli_defaults_and_overrides(self) -> None:
        defaults = build_parser().parse_args(
            [
                "--longbench-root",
                "/datasets/LongBench",
                "--materialized-draft-path",
                "/models/instrumented",
                "--frozen-selector-config",
                "/outputs/frozen-selector.json",
            ]
        )
        self.assertEqual(defaults.candidate_merge_mode, "base_only")
        self.assertEqual(defaults.base_top_m, 16)
        self.assertEqual(defaults.conditional_top_r, 2)
        self.assertEqual(defaults.refine_depth, 2)
        self.assertEqual(defaults.refined_prefix_budget, 2)
        self.assertEqual(defaults.refinement_weight, 0.5)
        self.assertEqual(defaults.node_budget, 16)
        self.assertEqual(defaults.verification_mode, "always_replay")
        self.assertEqual(defaults.online_selector_mode, BASE_CONTROL_SELECTOR_MODE)

        overridden = build_parser().parse_args(
            [
                "--longbench-root",
                "/datasets/LongBench",
                "--materialized-draft-path",
                "/models/instrumented",
                "--frozen-selector-config",
                "/outputs/frozen-selector.json",
                "--candidate-merge-mode",
                "union",
                "--conditional-top-r",
                "4",
                "--refine-depth",
                "4",
                "--refined-prefix-budget",
                "4",
                "--refinement-weight",
                "0.25",
                "--node-budget",
                "32",
                "--verification-mode",
                "packed_native",
            ]
        )
        self.assertEqual(overridden.candidate_merge_mode, "union")
        self.assertEqual(overridden.conditional_top_r, 4)
        self.assertEqual(overridden.refine_depth, 4)
        self.assertEqual(overridden.refined_prefix_budget, 4)
        self.assertEqual(overridden.refinement_weight, 0.25)
        self.assertEqual(overridden.node_budget, 32)
        self.assertEqual(overridden.verification_mode, "packed_native")

    def test_base_only_uses_phase2c_top1_control_configuration(self) -> None:
        args = build_parser().parse_args(
            [
                "--longbench-root",
                "/datasets/LongBench",
                "--materialized-draft-path",
                "/models/instrumented",
                "--frozen-selector-config",
                "/outputs/frozen-selector.json",
                "--candidate-merge-mode",
                "base_only",
                "--verification-mode",
                "packed_native",
            ]
        )
        args.nr_calibration = None
        args.frozen_selector = {"config_sha256": "selector"}
        args.gold_comparison_modes = ("native",)
        args.gold_thresholds = {
            "maximum_absolute_error": 0.5,
            "mean_absolute_error": 0.02,
            "relative_error_epsilon": 1.0e-6,
            "layer_localization_p99_error": 0.1,
        }
        options = _generation_options(
            args,
            profiling=False,
            gold=False,
            shadow_replay=False,
            context_length=1024,
        )
        self.assertEqual(options["online_selector_mode"], "top1_packed_tree")
        self.assertEqual(options["phase2d2_candidate_merge_mode"], "base_only")
        self.assertEqual(options["phase2d2_verification_mode"], "packed_native")

    def test_base_only_explicit_three_way_output_relation(self) -> None:
        relations = _output_relations(
            target_generated=[1, 2, 3],
            reference_generated=[1, 2, 3],
            formal_generated=[1, 2, 3],
            profile_generated=[1, 2, 3],
            diagnostic_generated=[1, 2, 3],
            formal_repeat_output_ids_equal=True,
        )
        self.assertTrue(relations["target_only_output_ids_equal"])
        self.assertTrue(
            relations["target_only_original_dflash_reference_output_ids_equal"]
        )
        self.assertTrue(relations["original_dflash_reference_output_ids_equal"])

    def test_target_reference_relation_is_not_inferred_from_tree(self) -> None:
        relations = _output_relations(
            target_generated=[1, 2, 3],
            reference_generated=[1, 2, 4],
            formal_generated=[1, 2, 3],
            profile_generated=[1, 2, 3],
            diagnostic_generated=[1, 2, 3],
            formal_repeat_output_ids_equal=True,
        )
        self.assertTrue(relations["target_only_output_ids_equal"])
        self.assertFalse(
            relations["target_only_original_dflash_reference_output_ids_equal"]
        )
        self.assertFalse(relations["original_dflash_reference_output_ids_equal"])

    def test_candidate_quality_is_evaluation_only(self) -> None:
        row = {
            "phase2d2": {
                "replay_decisions": [{"decision_depth": 0, "top1_token_id": 101}],
                "candidate_merge_events": [
                    {
                        "position_index": 1,
                        "token_id": 101,
                        "base_rank": None,
                        "conditional_rank": 1,
                    }
                ],
            }
        }
        quality = candidate_quality_for_round(row)
        self.assertEqual(quality["refined_only_target_recovery_count"], 1)
        self.assertFalse(
            quality["positions"][0]["target_label_used_for_runtime_selection"]
        )
        self.assertIsNone(quality["offline_oracle_coverage"])
        self.assertTrue(quality["offline_oracle_coverage_unavailable_reason"])

    def test_same_budget_comparison_requires_identical_nested_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "phase2d2": {
                    "node_budget": 16,
                    "candidate_merge_mode": "base_only",
                    "frozen_selector_hash": "selector",
                    "online_selector_mode": "top1_packed_tree",
                }
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            sample = {
                "task": "gov_report",
                "sample_id": "sample",
                "nominal_prompt_length": 1024,
                "context_prefix_sha256": "prefix",
            }
            (root / "samples.jsonl").write_text(json.dumps(sample) + "\n")
            aggregate = {
                "task": "gov_report",
                "nominal_prompt_length": 1024,
                "configuration_id": "base",
                "accepted_tokens_per_verified_node": 0.2,
                "candidate_refined_only_recovery_count": 0,
                "candidate_selected_tree_coverage_rate_at_fixed_n": 0.5,
            }
            (root / "refinement-summary.json").write_text(
                json.dumps({"aggregates": [aggregate]})
            )
            args = SimpleNamespace(
                base_result_dir=[root],
                node_budget=16,
                frozen_selector={"config_sha256": "selector"},
                online_selector_mode="top1_packed_tree",
            )
            current = {
                "aggregates": [
                    {
                        **aggregate,
                        "configuration_id": "refined",
                        "accepted_tokens_per_verified_node": 0.3,
                        "candidate_refined_only_recovery_count": 1,
                        "candidate_selected_tree_coverage_rate_at_fixed_n": 0.75,
                    }
                ]
            }
            result = _comparison(args, current, [sample])
            self.assertEqual(result["status"], "ok")
            self.assertAlmostEqual(
                result["rows"][0]["selected_tree_target_coverage_rate_at_fixed_n_gain"],
                0.25,
            )
            with self.assertRaisesRegex(ValueError, "cohorts or prefixes"):
                _comparison(
                    args,
                    current,
                    [{**sample, "context_prefix_sha256": "different"}],
                )

    def test_timing_row_separates_formal_profile_and_memory(self) -> None:
        sample = {
            "task": "gov_report",
            "sample_id": "sample",
            "nominal_prompt_length": 1024,
            "configuration_id": "config",
            "generated_token_count": 4,
            "formal_timing_repeats": [
                {"cuda_elapsed_ms": 8.0},
                {"cuda_elapsed_ms": 10.0},
                {"cuda_elapsed_ms": 12.0},
            ],
            "profiling": {
                "measurement": {
                    "profile_breakdown": {
                        "prefill": {"cuda_elapsed_ms": 2.0},
                        "total": {"cuda_elapsed_ms": 11.0},
                    }
                },
                "components": {
                    "draft_hidden_logits_cuda_elapsed_ms": 1.0,
                    "conditional_remask_cuda_elapsed_ms": 2.0,
                    "total_draft_cuda_elapsed_ms": 4.0,
                    "verification_cuda_elapsed_ms": 3.0,
                    "total_verification_cuda_elapsed_ms": 6.0,
                    "canonical_recheck_cuda_elapsed_ms": 2.0,
                },
                "memory": {
                    "peak_allocated_bytes": 100,
                    "peak_reserved_bytes": 200,
                },
            },
            "target_only": {"cuda_elapsed_ms": 7.0},
        }
        row = _sample_timing_row(
            sample,
            [
                {
                    "task": "gov_report",
                    "sample_id": "sample",
                    "nominal_prompt_length": 1024,
                    "configuration_id": "config",
                    "committed_token_count": 4,
                }
            ],
        )
        self.assertEqual(row["generation_cuda_latency_median_ms"], 10.0)
        self.assertEqual(row["profile_decode_cuda_elapsed_ms"], 9.0)
        self.assertEqual(row["peak_reserved_bytes"], 200)

    def test_decision_priority_correctness_before_benefit(self) -> None:
        decision = build_phase2d2_decision(
            official_checker_passed=False,
            structural_correctness_passed=True,
            guard_false_negative_count=0,
            independent_task_count=3,
            stable_coverage_gain=True,
            stable_online_efficiency_gain=True,
            conditional_cost_acceptable=True,
            full_second_pass_too_heavy=False,
        )
        self.assertEqual(decision["decision"], "numerical_safety_unresolved")

    def test_cross_task_stability_requires_two_directionally_positive_tasks(
        self,
    ) -> None:
        self.assertIsNone(_stable_positive_by_task({"gov_report": [0.1]}))
        self.assertTrue(
            _stable_positive_by_task({"gov_report": [0.1, 0.2], "qasper": [0.05]})
        )
        self.assertFalse(
            _stable_positive_by_task({"gov_report": [0.1, 0.2], "qasper": [-0.01]})
        )

    def test_insufficient_evidence_is_not_manual_success(self) -> None:
        decision = build_phase2d2_decision(
            official_checker_passed=True,
            structural_correctness_passed=True,
            guard_false_negative_count=0,
            independent_task_count=1,
            stable_coverage_gain=None,
            stable_online_efficiency_gain=None,
            conditional_cost_acceptable=None,
            full_second_pass_too_heavy=None,
        )
        self.assertEqual(decision["decision"], "insufficient_evidence")

    def test_freeze_limits_families_and_is_deterministic(self) -> None:
        candidates = []
        for index, mode in enumerate(
            ("conditional_rerank", "conditional_rerank", "union", "union", "union")
        ):
            candidates.append(
                {
                    "configuration_id": f"c{index}",
                    "candidate_merge_mode": mode,
                    "base_top_m": 16,
                    "conditional_top_r": 2,
                    "refine_depth": 2,
                    "refined_prefix_budget": 2,
                    "refinement_weight": 0.5,
                    "node_budget": 16,
                    "mechanism_correctness_passed": True,
                    "refined_only_target_recovery_count": index,
                    "accepted_tokens_per_verified_node": index / 10,
                    "conditional_remask_cuda_median_ms": 1.0,
                }
            )
        left = freeze_refinement_config(candidates, [{"sha": "a"}])
        right = freeze_refinement_config(list(reversed(candidates)), [{"sha": "a"}])
        self.assertEqual(left, right)
        self.assertLessEqual(len(left["configurations"]), 3)
        validate_frozen_refinement(left)
        frozen_schema = json.loads(
            (
                REPO_ROOT / "docs/schemas/phase2d2-frozen-refinement.schema.json"
            ).read_text()
        )
        Draft202012Validator(frozen_schema).validate(left)

    def test_all_phase2d2_schemas_are_valid_draft_202012(self) -> None:
        schemas = sorted(
            (REPO_ROOT / "docs" / "schemas").glob("phase2d2-*.schema.json")
        )
        self.assertGreaterEqual(len(schemas), 10)
        for path in schemas:
            Draft202012Validator.check_schema(json.loads(path.read_text()))

    def test_lazy_import_does_not_load_model_frameworks(self) -> None:
        code = (
            "import sys; "
            "import experiments.phase2d2_robust_conditional_refinement; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules; "
            "assert 'datasets' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)

    def test_checker_rejects_incomplete_run_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(
                json.dumps({"schema_version": "1.0", "status": "started"})
            )
            with self.assertRaisesRegex(ValueError, "missing Phase 2D.2A artifacts"):
                check_output(root)

    def test_failure_persistence_writes_complete_diagnostic_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "terminal.log").write_text("failure\n")
            manifest = {
                "status": "error",
                "error": {"type": "SyntheticFailure", "message": "boom"},
                "workload": {"inventory": {}},
            }
            provenance = {
                "status": "error",
                "error": manifest["error"],
            }
            _persist(
                root,
                manifest=manifest,
                environment={},
                provenance=provenance,
                samples=[],
                rounds=[],
                evidence=[],
                args=SimpleNamespace(
                    base_result_dir=[],
                    node_budget=16,
                    frozen_selector={"config_sha256": "selector"},
                ),
            )
            self.assertEqual(
                json.loads((root / "manifest.json").read_text())["status"], "error"
            )
            self.assertTrue(
                all(
                    (root / name).is_file()
                    for name in (*ROOT_ARTIFACTS, *GOLD_ARTIFACTS)
                )
            )

    def test_instrumented_source_uses_q_len_one_and_one_conditional_call(self) -> None:
        source = (
            REPO_ROOT / "third_party/qwen3_8b_dflash_b16/instrumented/dflash.py"
        ).read_text()
        self.assertIn("_phase2d2_q1_authoritative_replay", source)
        self.assertIn('"query_length": 1', source)
        self.assertIn("conditional_forward_count = 1", source)
        self.assertIn("conditional_refinement_forward_count", source)
        self.assertIn('"base_only_conditional_refinement_bypassed"', source)
        self.assertIn('"phase2c_frozen_selector_seed_tree"', source)
        self.assertIn('"phase2d2_q1_replay",', source)
        self.assertIn('"packed_temporary_kv_discarded"', source)
        self.assertIn('"candidate_shape": {', source)
        self.assertIn('"reference_shape": {', source)

    def test_original_vendored_code_is_not_modified_by_phase2d2(self) -> None:
        changed = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                "1ba29e8ce4a165c03af997c2f3b3b595ab79c1d2",
                "--",
                "third_party/qwen3_8b_dflash_b16/original",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(changed, "")


if __name__ == "__main__":
    unittest.main()
