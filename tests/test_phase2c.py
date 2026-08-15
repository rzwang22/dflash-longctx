from __future__ import annotations

import inspect
import json
import subprocess
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from dflash.phase2b1 import policies as phase2b1_policies
from dflash.phase2c.alignment import align_offline_online
from dflash.phase2c.core import (
    MAXIMUM_DRAFT_DEPTH,
    build_tree_attention_visibility,
    compact_synthetic_cache_layers,
    flatten_tree,
    greedy_tree_traversal,
    layerwise_beam_tree,
    path_compaction_storage_indices,
    semantic_position_ids,
    top1_chain_tree,
    validate_online_tree_options,
    validate_tree,
)
from dflash.phase2c.metrics import aggregate_tree_results, summary_csv_rows
from experiments.phase2c_online_tree import (
    _require_equal,
    parse_args,
    tree_configurations,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = (
    "phase2c-manifest.schema.json",
    "phase2c-sample.schema.json",
    "phase2c-round.schema.json",
    "phase2c-summary.schema.json",
    "phase2c-correctness-summary.schema.json",
    "phase2c-offline-online.schema.json",
    "phase2c-gold-logit-check.schema.json",
    "phase2c-gold-kv-check.schema.json",
    "phase2c-gold-layer-localization.schema.json",
    "phase2c-gold-comparison-summary.schema.json",
)


def _positions(depth: int = 15, width: int = 3, *, ties: bool = False):
    rows = []
    for position in range(depth):
        candidates = []
        for rank in range(1, width + 1):
            candidates.append(
                {
                    "rank": rank,
                    "token_id": (position + 1) * 100 + rank,
                    "log_probability": -1.0 if ties else -float(rank),
                }
            )
        rows.append({"block_position": position + 1, "candidates": candidates})
    return rows


def _branch_tree():
    positions = _positions(depth=2, width=2)
    nodes = {(), (101,), (102,), (101, 201), (102, 202)}
    return flatten_tree(
        nodes,
        positions,
        root_token_id=42,
        policy="layerwise_beam",
        requested_node_budget=8,
        beam_width=2,
    )


def _runner_args(*extra: str):
    return parse_args(
        [
            "--dataset-jsonl",
            "/tmp/longbench.jsonl",
            "--materialized-draft-path",
            "/tmp/draft",
            *extra,
        ]
    )


class Phase2CTreeTests(unittest.TestCase):
    def test_node_budget_includes_root(self) -> None:
        nodes = layerwise_beam_tree(_positions(), budget=16, beam_width=4)
        self.assertIn((), nodes)
        self.assertEqual(len(nodes), 16)
        self.assertEqual(len(nodes) - 1, 15)

    def test_prefix_closure(self) -> None:
        nodes = layerwise_beam_tree(_positions(), budget=64, beam_width=8)
        self.assertTrue(all(not node or node[:-1] in nodes for node in nodes))

    def test_flatten_parent_precedes_child(self) -> None:
        tree = _branch_tree()
        self.assertTrue(
            all(
                node["parent_index"] is None
                or node["parent_index"] < node["node_index"]
                for node in tree["nodes"]
            )
        )

    def test_duplicate_sibling_token_is_rejected(self) -> None:
        positions = _positions()
        positions[0]["candidates"][1]["token_id"] = positions[0]["candidates"][0][
            "token_id"
        ]
        with self.assertRaisesRegex(ValueError, "duplicate token"):
            layerwise_beam_tree(positions, budget=16, beam_width=2)

    def test_maximum_depth(self) -> None:
        too_deep = {()}
        prefix = ()
        for token in range(MAXIMUM_DRAFT_DEPTH + 1):
            prefix += (token,)
            too_deep.add(prefix)
        with self.assertRaisesRegex(ValueError, "maximum draft depth"):
            validate_tree(too_deep, budget=32)

    def test_tie_breaking_is_deterministic(self) -> None:
        positions = _positions(ties=True)
        first = layerwise_beam_tree(positions, budget=32, beam_width=4)
        second = layerwise_beam_tree(positions, budget=32, beam_width=4)
        self.assertEqual(first, second)
        depth_one = sorted(node for node in first if len(node) == 1)
        self.assertEqual(depth_one, [(101,), (102,), (103,)])

    def test_repeated_construction_is_identical(self) -> None:
        positions = _positions()
        first = flatten_tree(
            layerwise_beam_tree(positions, budget=32, beam_width=4),
            positions,
            root_token_id=42,
            policy="layerwise_beam",
            requested_node_budget=32,
            beam_width=4,
        )
        second = flatten_tree(
            layerwise_beam_tree(positions, budget=32, beam_width=4),
            positions,
            root_token_id=42,
            policy="layerwise_beam",
            requested_node_budget=32,
            beam_width=4,
        )
        self.assertEqual(first, second)

    def test_top1_chain_is_exact_15_token_chain(self) -> None:
        positions = _positions()
        nodes = top1_chain_tree(positions)
        expected = tuple(
            position["candidates"][0]["token_id"] for position in positions
        )
        self.assertIn(expected, nodes)
        self.assertEqual(len(nodes), 16)

    def test_phase2b1_reexports_canonical_layerwise_policy(self) -> None:
        self.assertIs(phase2b1_policies.layerwise_beam_tree, layerwise_beam_tree)
        positions = _positions()
        self.assertEqual(
            phase2b1_policies.layerwise_beam_tree(positions, budget=32, beam_width=4),
            layerwise_beam_tree(positions, budget=32, beam_width=4),
        )

    def test_layerwise_policy_has_no_target_argument(self) -> None:
        parameters = inspect.signature(layerwise_beam_tree).parameters
        self.assertNotIn("target", parameters)
        self.assertNotIn("reference", parameters)


class Phase2CMaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = _branch_tree()
        self.index = {
            tuple(node["draft_token_path"]): node["node_index"]
            for node in self.tree["nodes"]
        }
        parents = [node["parent_index"] for node in self.tree["nodes"]]
        self.visibility = build_tree_attention_visibility(parents, prefix_length=3)

    def test_ancestor_and_self_visible(self) -> None:
        query = self.index[(101, 201)]
        self.assertTrue(self.visibility[query][3 + self.index[()]])
        self.assertTrue(self.visibility[query][3 + self.index[(101,)]])
        self.assertTrue(self.visibility[query][3 + query])

    def test_sibling_and_other_branch_invisible(self) -> None:
        query = self.index[(101, 201)]
        self.assertFalse(self.visibility[query][3 + self.index[(102,)]])
        self.assertFalse(self.visibility[query][3 + self.index[(102, 202)]])

    def test_prefix_fully_visible(self) -> None:
        self.assertTrue(all(row[:3] == [True, True, True] for row in self.visibility))

    def test_future_node_invisible(self) -> None:
        root = self.index[()]
        self.assertFalse(self.visibility[root][3 + self.index[(101,)]])

    def test_semantic_positions_follow_depth(self) -> None:
        depths = [node["depth"] for node in self.tree["nodes"]]
        positions = semantic_position_ids(100, depths)
        self.assertEqual(positions[self.index[()]], 100)
        self.assertEqual(positions[self.index[(101,)]], 101)
        self.assertEqual(positions[self.index[(101, 201)]], 102)

    def test_same_depth_branches_share_position(self) -> None:
        depths = [node["depth"] for node in self.tree["nodes"]]
        positions = semantic_position_ids(100, depths)
        self.assertEqual(positions[self.index[(101,)]], positions[self.index[(102,)]])

    def test_mask_shape_covers_prefix_and_all_nodes(self) -> None:
        self.assertEqual(len(self.visibility), len(self.tree["nodes"]))
        self.assertTrue(
            all(len(row) == 3 + len(self.tree["nodes"]) for row in self.visibility)
        )


class Phase2CTraversalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = _branch_tree()
        self.index = {
            tuple(node["draft_token_path"]): node["node_index"]
            for node in self.tree["nodes"]
        }

    def test_matching_child_advances(self) -> None:
        result = greedy_tree_traversal(
            self.tree["nodes"],
            {
                self.index[()]: 101,
                self.index[(101,)]: 201,
                self.index[(101, 201)]: 999,
            },
        )
        self.assertEqual(result["accepted_draft_token_count"], 2)

    def test_missing_child_produces_correction(self) -> None:
        result = greedy_tree_traversal(self.tree["nodes"], {self.index[()]: 999})
        self.assertEqual(result["accepted_draft_token_count"], 0)
        self.assertEqual(result["next_root_kind"], "correction")
        self.assertEqual(result["next_root_token_id"], 999)

    def test_leaf_match_produces_bonus(self) -> None:
        result = greedy_tree_traversal(
            self.tree["nodes"],
            {
                self.index[()]: 102,
                self.index[(102,)]: 202,
                self.index[(102, 202)]: 777,
            },
        )
        self.assertEqual(result["next_root_kind"], "bonus")
        self.assertEqual(result["next_root_token_id"], 777)

    def test_accepted_zero_commits_only_root(self) -> None:
        result = greedy_tree_traversal(self.tree["nodes"], {self.index[()]: 999})
        self.assertEqual(1 + result["accepted_draft_token_count"], 1)

    def test_eos_is_preserved_as_target_next_token(self) -> None:
        eos = 151645
        result = greedy_tree_traversal(self.tree["nodes"], {self.index[()]: eos})
        self.assertEqual(result["next_root_token_id"], eos)

    def test_eos_and_max_new_tokens_accounting(self) -> None:
        from tests.test_phase1 import _load_observability_helper

        helper = _load_observability_helper()
        eos = helper.build_round_accounting(
            round_index=0,
            context_length_before=100,
            prompt_length=100,
            max_new_tokens=32,
            block_size=16,
            accepted_draft_tokens=2,
            returned_tokens_before=1,
            eos_generated_count=3,
        )
        truncated = helper.build_round_accounting(
            round_index=1,
            context_length_before=128,
            prompt_length=100,
            max_new_tokens=32,
            block_size=16,
            accepted_draft_tokens=15,
            returned_tokens_before=29,
            eos_generated_count=None,
        )
        self.assertEqual(eos["stop_reason"], "eos")
        self.assertEqual(truncated["stop_reason"], "max_new_tokens")
        self.assertGreater(truncated["truncated_token_count"], 0)

    def test_duplicate_child_guard(self) -> None:
        nodes = [dict(node) for node in self.tree["nodes"]]
        duplicate = dict(nodes[1])
        duplicate["node_index"] = len(nodes)
        nodes.append(duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate token child"):
            greedy_tree_traversal(nodes, {0: 101})


class Phase2CCacheTests(unittest.TestCase):
    def test_gather_accepted_path_not_flattened_prefix(self) -> None:
        indices = path_compaction_storage_indices(
            prefix_length=3,
            flattened_tree_node_count=6,
            accepted_node_indices=[2, 5],
        )
        self.assertEqual(indices, [0, 1, 2, 3, 5, 8])
        self.assertNotEqual(indices, list(range(6)))

    def test_sibling_values_are_discarded(self) -> None:
        layer = ["p0", "p1", "root", "sib", "path1", "sib2", "path2"]
        compacted = compact_synthetic_cache_layers(
            [layer],
            prefix_length=2,
            flattened_tree_node_count=5,
            accepted_node_indices=[2, 4],
        )[0]
        self.assertEqual(compacted, ["p0", "p1", "root", "path1", "path2"])

    def test_multilayer_indices_and_lengths_match(self) -> None:
        layers = [list(range(7)), list(range(10, 17)), list(range(20, 27))]
        compacted = compact_synthetic_cache_layers(
            layers,
            prefix_length=2,
            flattened_tree_node_count=5,
            accepted_node_indices=[2, 4],
        )
        self.assertEqual({len(layer) for layer in compacted}, {5})
        self.assertEqual([layer[-1] for layer in compacted], [6, 16, 26])

    def test_correction_bonus_is_not_in_compaction_indices(self) -> None:
        indices = path_compaction_storage_indices(
            prefix_length=2,
            flattened_tree_node_count=5,
            accepted_node_indices=[2],
        )
        self.assertEqual(indices, [0, 1, 2, 4])
        self.assertEqual(len(indices), 2 + 1 + 1)

    def test_next_round_cache_length_invariant(self) -> None:
        indices = path_compaction_storage_indices(
            prefix_length=100,
            flattened_tree_node_count=32,
            accepted_node_indices=[3, 11, 20],
        )
        self.assertEqual(len(indices), 100 + 1 + 3)

    def test_synthetic_compaction_equals_dense_committed_path(self) -> None:
        prefix = ["p0", "p1"]
        packed = ["root", "sibling", "a", "other", "b"]
        compacted = compact_synthetic_cache_layers(
            [prefix + packed],
            prefix_length=2,
            flattened_tree_node_count=5,
            accepted_node_indices=[2, 4],
        )[0]
        self.assertEqual(compacted, prefix + ["root", "a", "b"])


class Phase2CRunnerTests(unittest.TestCase):
    def test_cli_defaults(self) -> None:
        args = _runner_args()
        self.assertEqual(args.tree_policies, ("top1_chain",))
        self.assertEqual(args.node_budgets, (16,))
        self.assertEqual(args.beam_widths, (4,))
        self.assertEqual(
            tree_configurations(args)[0]["configuration_id"], "top1_chain-b16"
        )

    def test_cli_multiple_configurations(self) -> None:
        args = _runner_args(
            "--tree-policy",
            "top1_chain",
            "--tree-policy",
            "layerwise_beam",
            "--node-budget",
            "16",
            "--node-budget",
            "32",
            "--beam-width",
            "4",
            "--beam-width",
            "8",
        )
        self.assertEqual(len(tree_configurations(args)), 5)

    def test_temperature_fails_during_parse(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature=0"):
            _runner_args("--temperature", "0.5")

    def test_batch_size_fails_during_parse(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            _runner_args("--batch-size", "2")

    def test_block_size_fails_during_parse(self) -> None:
        with self.assertRaisesRegex(ValueError, "block_size=16"):
            _runner_args("--block-size", "8")

    def test_gold_budget_limit_fails_during_parse(self) -> None:
        with self.assertRaisesRegex(ValueError, "gold checks"):
            _runner_args(
                "--tree-policy",
                "layerwise_beam",
                "--node-budget",
                "64",
                "--enable-gold-checks",
            )

    def test_online_option_validation(self) -> None:
        validate_online_tree_options(
            enabled=True,
            policy="layerwise_beam",
            node_budget=32,
            beam_width=4,
            draft_top_k=16,
            temperature=0.0,
            block_size=16,
            batch_size=1,
            return_diagnostics=True,
            enable_profiling=True,
            enable_gold_checks=False,
        )

    def test_control_profile_equality_guard(self) -> None:
        class FakeTorch:
            @staticmethod
            def equal(left, right):
                return left == right

        _require_equal(FakeTorch, [1, 2], [1, 2], label="equal")
        with self.assertRaisesRegex(RuntimeError, "output token mismatch"):
            _require_equal(FakeTorch, [1], [2], label="different")

    def test_cli_help_lazy_import(self) -> None:
        code = (
            "import runpy,sys;"
            "sys.argv=['phase2c_online_tree.py','--help'];"
            "\ntry: runpy.run_path('experiments/phase2c_online_tree.py',run_name='__main__')"
            "\nexcept SystemExit as exc: assert exc.code == 0;"
            "\nfor name in ('torch','transformers','datasets'): assert name not in sys.modules,name"
        )
        subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT, check=True, capture_output=True
        )

    def test_new_schemas_are_valid_draft_2020_12(self) -> None:
        for name in SCHEMAS:
            schema = json.loads((REPO_ROOT / "docs" / "schemas" / name).read_text())
            Draft202012Validator.check_schema(schema)

    def test_new_schemas_accept_minimal_artifacts(self) -> None:
        timing = {
            name: 1.0
            for name in (
                "draft_hidden_logits_cuda_elapsed_ms",
                "topk_cuda_elapsed_ms",
                "tree_construction_cuda_elapsed_ms",
                "mask_construction_cuda_elapsed_ms",
                "verification_cuda_elapsed_ms",
                "traversal_cuda_elapsed_ms",
                "kv_compaction_cuda_elapsed_ms",
                "total_draft_cuda_elapsed_ms",
                "total_verification_cuda_elapsed_ms",
                "round_cuda_elapsed_ms",
            )
        }
        payloads = {
            "phase2c-manifest.schema.json": {
                "schema_version": "1.0",
                "status": "ok",
                "configuration": {
                    "phase": "2C",
                    "mode": "online_bounded_marginal_tree",
                    "batch_size": 1,
                    "temperature": 0.0,
                    "block_size": 16,
                    "tree_configurations": [{}],
                },
                "workload": {},
                "phase2c": {
                    "policy_uses_target_information": False,
                    "maximum_draft_depth": 15,
                    "node_budget_includes_root": True,
                    "score_rule": "mean",
                    "tie_breaking": "stable",
                    "mask_semantics": "tree",
                    "position_ids": "depth",
                    "cache_position": "packed",
                    "kv_compaction": "gather",
                },
                "measurement_policy": {},
                "interpretation": {},
                "artifacts": {},
                "error": None,
            },
            "phase2c-sample.schema.json": {
                "schema_version": "1.0",
                "task": "gov_report",
                "sample_id": "a",
                "nominal_prompt_length": 1024,
                "actual_prompt_length": 1024,
                "configuration_id": "top1_chain-b16",
                "tree_policy": "top1_chain",
                "node_budget": 16,
                "beam_width": 1,
                "generated_token_count": 1,
                "generated_output_token_ids": [1],
                "reference": {},
                "control": {},
                "profiling": {},
                "correctness": {},
            },
            "phase2c-round.schema.json": {
                "schema_version": "1.0",
                "task": "gov_report",
                "sample_id": "a",
                "nominal_prompt_length": 1024,
                "configuration_id": "top1_chain-b16",
                "profiling_repeat": 0,
                "round_index": 0,
                "generated_root_offset": 0,
                "prefix_cache_length_before_round": 1024,
                "requested_node_count": 16,
                "actual_node_count": 16,
                "beam_width": 1,
                "maximum_tree_depth": 15,
                "per_depth_node_counts": [1] * 16,
                "accepted_draft_token_count": 1,
                "committed_token_count": 2,
                "target_token_kind": "correction",
                "terminal_round": False,
                "raw_cumulative_generated_token_count": 3,
                "cumulative_generated_token_count": 3,
                "truncated_token_count": 0,
                "tree_metadata": {},
                "tree_attention": {},
                "cache_compaction": {},
                "timing": timing,
                "output_ids_equal_reference": True,
                "gold_check_status": "not_run",
                "offline_comparison_status": "disabled",
            },
            "phase2c-summary.schema.json": {
                "schema_version": "1.0",
                "sample_record_count": 1,
                "round_record_count": 1,
                "groups": [{}],
                "interpretation": {},
            },
            "phase2c-correctness-summary.schema.json": {
                "schema_version": "1.0",
                "status": "ok",
                "all_checks_passed": True,
                "records": [
                    {
                        "task": "gov_report",
                        "sample_id": "a",
                        "nominal_prompt_length": 1024,
                        "configuration_id": "top1_chain-b16",
                        "reference_output_ids_equal": True,
                        "control_profile_output_ids_equal": True,
                        "passed": True,
                    }
                ],
                "error": None,
            },
            "phase2c-offline-online.schema.json": {
                "schema_version": "1.0",
                "status": "disabled",
                "exact_matched_round_count": 0,
                "unmatched_online_round_count": 0,
                "unmatched_offline_round_count": 0,
                "matched_round_agreement_rate": None,
                "unmatched_rounds_are_mismatches": False,
                "rows": [],
            },
            "phase2c-gold-logit-check.schema.json": {
                "schema_version": "1.0",
                "task": "gov_report",
                "sample_id": "a",
                "nominal_prompt_length": 1024,
                "configuration_id": "top1_chain-b16",
                "round_index": 0,
                "status": "passed",
                "checked_node_count": 1,
                "all_tree_nodes_checked": True,
                "greedy_argmax_all_equal": True,
                "top_k_token_ids_all_equal": True,
                "maximum_absolute_error": 0.0,
                "mean_absolute_error": 0.0,
                "maximum_absolute_tolerance": 0.5,
                "mean_absolute_tolerance": 0.02,
                "nodes": [{}],
            },
            "phase2c-gold-kv-check.schema.json": {
                "schema_version": "1.0",
                "task": "gov_report",
                "sample_id": "a",
                "nominal_prompt_length": 1024,
                "configuration_id": "top1_chain-b16",
                "round_index": 0,
                "status": "passed",
                "greedy_argmax_equal": True,
                "top_k_token_ids_equal": True,
                "within_bfloat16_tolerance": True,
                "compacted_layer_lengths": [1025],
                "next_step_layer_lengths": [1026],
                "dense_reference_layer_lengths": [1026],
                "length_checks": {},
                "sibling_nodes_discarded": True,
                "correction_or_bonus_absent_from_persistent_cache": True,
            },
        }
        for name, payload in payloads.items():
            schema = json.loads((REPO_ROOT / "docs" / "schemas" / name).read_text())
            Draft202012Validator(schema).validate(payload)


class Phase2CAggregationTests(unittest.TestCase):
    def _sample(self):
        return {
            "task": "gov_report",
            "nominal_prompt_length": 1024,
            "configuration_id": "top1_chain-b16",
            "tree_policy": "top1_chain",
            "node_budget": 16,
            "beam_width": 1,
            "generated_token_count": 32,
            "reference": {
                "round_count": 3,
                "accepted_draft_token_lengths": [1, 2, 3],
                "control_repeats": [{"cuda_elapsed_ms": 12.0}],
                "profiling_repeats": [{"verification_cuda_elapsed_ms": 4.0}],
            },
            "control": {
                "repeats": [{"cuda_elapsed_ms": 10.0}],
                "latency_per_returned_token_ms": 0.3125,
                "generated_tokens_per_second": 3200.0,
            },
            "profiling": {
                "round_count_mean": 2.0,
                "peak_allocated_bytes": 100,
                "peak_reserved_bytes": 200,
                "repeats": [{"components": {"verification_cuda_elapsed_ms": 5.0}}],
            },
        }

    def _round(self, index: int, accepted: int, terminal: bool):
        return {
            "task": "gov_report",
            "nominal_prompt_length": 1024,
            "configuration_id": "top1_chain-b16",
            "terminal_round": terminal,
            "accepted_draft_token_count": accepted,
            "committed_token_count": accepted + 1,
            "actual_node_count": 16,
            "reference_round_comparison": {
                "status": "comparable",
                "reference_accepted_draft_tokens": index,
            },
            "timing": {
                "tree_construction_cuda_elapsed_ms": 1.0,
                "mask_construction_cuda_elapsed_ms": 1.0,
                "verification_cuda_elapsed_ms": 4.0,
                "kv_compaction_cuda_elapsed_ms": 1.0,
                "round_cuda_elapsed_ms": 10.0,
            },
        }

    def test_aggregate_metrics_and_csv(self) -> None:
        repeat_zero = [self._round(0, 1, False), self._round(1, 2, True)]
        repeat_one = [self._round(0, 1, False), self._round(1, 2, True)]
        for row in repeat_zero:
            row["profiling_repeat"] = 0
        for row in repeat_one:
            row["profiling_repeat"] = 1
        summary = aggregate_tree_results([self._sample()], repeat_zero + repeat_one)
        group = summary["groups"][0]
        self.assertEqual(group["round_count"], 2)
        self.assertEqual(group["profiling_round_record_count"], 4)
        self.assertEqual(group["non_terminal_round_count"], 1)
        self.assertEqual(group["total_committed_token_count"], 5)
        self.assertEqual(group["mean_verified_nodes_per_round"], 16)
        self.assertEqual(
            group["profile_round_component_cuda_elapsed_ms"][
                "verification_cuda_elapsed_ms"
            ]["count"],
            4,
        )
        self.assertEqual(len(summary_csv_rows(summary)), 1)

    def test_offline_alignment_does_not_count_unmatched_as_mismatch(self) -> None:
        online = [
            {
                "task": "gov_report",
                "sample_id": "a",
                "nominal_prompt_length": 1024,
                "tree_policy": "layerwise_beam",
                "requested_node_count": 32,
                "beam_width": 4,
                "generated_root_offset": 0,
                "accepted_draft_token_count": 3,
            },
            {
                "task": "gov_report",
                "sample_id": "a",
                "nominal_prompt_length": 1024,
                "tree_policy": "layerwise_beam",
                "requested_node_count": 32,
                "beam_width": 4,
                "generated_root_offset": 4,
                "accepted_draft_token_count": 2,
            },
        ]
        offline = [
            {
                "task": "gov_report",
                "sample_id": "a",
                "nominal_prompt_length": 1024,
                "policy": "layerwise_beam",
                "requested_total_tree_node_budget": 32,
                "policy_parameters": {"beam_width": 4},
                "round_output_start_offset": 0,
                "covered_target_prefix_depth": 3,
            }
        ]
        result = align_offline_online(online, offline)
        self.assertEqual(result["exact_matched_round_count"], 1)
        self.assertEqual(result["unmatched_online_round_count"], 1)
        self.assertEqual(result["matched_round_agreement_rate"], 1.0)
        self.assertFalse(result["unmatched_rounds_are_mismatches"])


if __name__ == "__main__":
    unittest.main()
