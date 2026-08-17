from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from dflash.phase2c.reference_state import (
    compare_reference_tree_checkpoints,
    verifier_evidence_by_prefix,
)
from experiments.phase2c_online_tree import (
    REFERENCE_STATE_CSV_FIELDS,
    parse_args,
    run_configuration,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tensor(values=(0.0, 1.0), *, shape=(1, 1, 2, 1)):
    return {
        "dtype": "bfloat16",
        "shape": list(shape),
        "numel": len(values),
        "sample_indices": list(range(len(values))),
        "sample_values": list(values),
        "sample_sum": float(sum(values)),
    }


def _cache(values=(0.0, 1.0)):
    return [
        {
            "layer_index": 0,
            "sequence_length": 2,
            "actual_cache_sequence_length": 2,
            "key_shape": [1, 1, 2, 1],
            "value_shape": [1, 1, 2, 1],
            "key_comparable_sample": _tensor(values),
            "value_comparable_sample": _tensor(values),
        }
    ]


def _probe(top=(10, 20), margin=1.0):
    return {
        "dtype": "bfloat16",
        "shape": [100],
        "top_k": len(top),
        "top_token_ids": list(top),
        "top_values": [2.0, 1.0],
        "top1_token_id": top[0],
        "top1_top2_margin": margin,
        "full_vocabulary_logits_saved": False,
    }


DEFAULT_CHAIN = tuple(range(1, 16))


def _draft(
    top=DEFAULT_CHAIN,
    *,
    proposal=None,
    lower_rank_offset=10_000,
):
    top = tuple(top)
    return {
        "cache": _cache(),
        "cache_layer_lengths": [2],
        "logits": {"shape": [1, 15, 100], "positions": []},
        "candidate_top1_token_ids": list(top),
        "candidate_topk_token_ids": [
            [value, value + lower_rank_offset] for value in top
        ],
        "candidate_tree_input": (
            {"actual_proposed_token_ids": list(proposal)}
            if proposal is not None
            else {}
        ),
    }


def _checkpoint(
    prefix: int,
    *,
    side: str,
    kv=(0.0, 1.0),
    hidden=(0.0, 1.0),
    probe=(10, 20),
    draft_top=DEFAULT_CHAIN,
    proposal=DEFAULT_CHAIN,
    lower_rank_offset=10_000,
    round_index=0,
    root_token_id=99,
    transition=None,
):
    draft = _draft(
        draft_top,
        proposal=proposal if side == "reference" else None,
        lower_rank_offset=lower_rank_offset,
    )
    return {
        "generated_prefix_length": prefix,
        "absolute_sequence_prefix_length": 100 + prefix,
        "execution_path": side,
        "checkpoint_provenance": (
            "actual_reference_checkpoint"
            if side == "reference"
            else "actual_tree_checkpoint"
        ),
        "semantic_prefix_token_ids_sha256": f"prefix-{prefix}",
        "semantic_prefix_tail_token_ids": [prefix],
        "persistent_target_kv": _cache(kv),
        "target_hidden": _tensor(hidden),
        "target_hidden_provenance": "actual",
        "semantic_probe": _probe(probe),
        "semantic_probe_provenance": "actual",
        "draft_state": draft,
        "draft_state_provenance": "actual",
        "round_entry": {
            "round_index": round_index,
            "root_token_id": root_token_id,
        },
        "commit_transition": transition,
        f"{side}_round_index": round_index,
    }


def _transition(before, committed, next_root, *, kind="correction"):
    return {
        "round_index": 0,
        "generated_prefix_before": before,
        "generated_prefix_after": before + len(committed),
        "accepted_draft_token_count": len(committed) - 1,
        "committed_token_ids": list(committed),
        "next_root_token_id": next_root,
        "next_root_kind": kind,
        "committed_cache_source": "actual",
        "persistent_target_kv_length_after": 100 + before + len(committed),
    }


def _compare(
    reference,
    tree,
    *,
    output_offset=None,
    evidence=None,
    reference_actual=None,
    tree_actual=None,
):
    return compare_reference_tree_checkpoints(
        reference_checkpoints=reference,
        tree_checkpoints=tree,
        first_output_divergent_offset=output_offset,
        packed_verifier_evidence=evidence,
        reference_actual_generated_token_ids=reference_actual,
        tree_actual_generated_token_ids=tree_actual,
    )


class Phase2C4DifferentialTests(unittest.TestCase):
    def test_same_tokens_and_same_state(self) -> None:
        result = _compare(
            [_checkpoint(0, side="reference")],
            [_checkpoint(0, side="tree")],
            reference_actual=[10],
            tree_actual=[10],
        )
        row = result["rows"][0]
        self.assertTrue(row["semantic_prefix_token_exact"])
        self.assertTrue(row["target_kv"]["numerically_equal"])
        self.assertTrue(row["semantic_probe"]["argmax_equal"])
        self.assertIsNone(result["summary"]["failure_classification"])

    def test_kv_numerical_drift_without_decision_change_is_not_root_cause(
        self,
    ) -> None:
        result = _compare(
            [_checkpoint(0, side="reference")],
            [_checkpoint(0, side="tree", kv=(0.0, 1.01))],
            output_offset=0,
            reference_actual=[10],
            tree_actual=[10],
        )
        row = result["rows"][0]
        self.assertTrue(row["mere_numerical_variation_only"])
        self.assertNotEqual(
            result["summary"]["failure_classification"],
            "target_kv_state_drift",
        )

    def test_faithful_probe_divergence_after_kv_drift_is_classified(self) -> None:
        reference = [
            _checkpoint(0, side="reference"),
            _checkpoint(1, side="reference"),
        ]
        tree = [
            _checkpoint(0, side="tree", kv=(0.0, 1.01)),
            _checkpoint(1, side="tree", kv=(0.0, 1.01), probe=(30, 20)),
        ]
        result = _compare(
            reference,
            tree,
            output_offset=1,
            reference_actual=[10, 10],
            tree_actual=[10, 30],
        )
        self.assertEqual(result["summary"]["first_state_divergent_aligned_prefix"], 0)
        self.assertEqual(
            result["summary"]["first_semantically_consequential_state_divergence"],
            1,
        )
        self.assertEqual(
            result["summary"]["failure_classification"], "target_kv_state_drift"
        )

    def test_direct_packed_linear_flip_has_priority(self) -> None:
        evidence = {
            0: [
                {
                    "comparison_mode": "native",
                    "verifier_decision_relevant": True,
                    "argmax_equal": False,
                }
            ]
        }
        result = _compare(
            [_checkpoint(0, side="reference")],
            [_checkpoint(0, side="tree")],
            output_offset=0,
            evidence=evidence,
            reference_actual=[10],
            tree_actual=[10],
        )
        self.assertEqual(
            result["summary"]["failure_classification"],
            "local_packed_verifier_numerical_divergence",
        )

    def test_lower_rank_topk_difference_is_diagnostic_only(self) -> None:
        reference = _checkpoint(0, side="reference", lower_rank_offset=10_000)
        tree = _checkpoint(0, side="tree", lower_rank_offset=20_000)
        result = _compare(
            [reference],
            [tree],
            output_offset=0,
            reference_actual=[10],
            tree_actual=[10],
        )
        row = result["rows"][0]
        self.assertTrue(row["draft_state"]["top1_candidates_equal"])
        self.assertFalse(row["draft_state"]["topk_candidates_equal"])
        self.assertFalse(row["semantic_consequence_observed"])
        self.assertNotEqual(
            result["summary"]["failure_classification"],
            "draft_state_or_candidate_drift",
        )

    def test_contradictory_candidate_capture_is_inconsistent_and_partial(self) -> None:
        captured = (777, *DEFAULT_CHAIN[1:])
        reference = _checkpoint(
            0,
            side="reference",
            draft_top=captured,
            proposal=DEFAULT_CHAIN,
        )
        tree = _checkpoint(0, side="tree", draft_top=DEFAULT_CHAIN)
        result = _compare(
            [reference],
            [tree],
            output_offset=0,
            reference_actual=[10],
            tree_actual=[10],
        )
        row = result["rows"][0]
        self.assertTrue(row["draft_state"]["original_proposal_equals_tree_top1"])
        self.assertFalse(row["draft_state"]["top1_candidates_equal"])
        self.assertEqual(
            row["candidate_alignment_status"],
            "candidate_capture_alignment_inconsistent",
        )
        self.assertFalse(row["semantic_consequence_observed"])
        self.assertEqual(result["summary"]["evidence_completeness"], "partial")

    def test_native_flip_globally_preempts_earlier_topk_variation(self) -> None:
        reference = [
            _checkpoint(0, side="reference", lower_rank_offset=10_000),
            _checkpoint(1, side="reference"),
        ]
        tree = [
            _checkpoint(0, side="tree", lower_rank_offset=20_000),
            _checkpoint(1, side="tree"),
        ]
        evidence = {
            1: [
                {
                    "comparison_mode": "native",
                    "verifier_decision_relevant": True,
                    "argmax_equal": False,
                }
            ]
        }
        result = _compare(
            reference,
            tree,
            output_offset=1,
            evidence=evidence,
            reference_actual=[10, 10],
            tree_actual=[10, 10],
        )
        self.assertEqual(
            result["summary"]["failure_classification"],
            "local_packed_verifier_numerical_divergence",
        )
        self.assertEqual(result["summary"]["classification_evidence_prefix"], 1)
        self.assertEqual(result["summary"]["classification_priority"], 1)

    def test_same_mask_only_flip_is_counterfactual_not_online_cause(self) -> None:
        evidence = {
            0: [
                {
                    "comparison_mode": "native",
                    "verifier_decision_relevant": True,
                    "argmax_equal": True,
                },
                {
                    "comparison_mode": "same_mask",
                    "verifier_decision_relevant": True,
                    "argmax_equal": False,
                },
                {
                    "comparison_mode": "math",
                    "verifier_decision_relevant": True,
                    "argmax_equal": True,
                },
            ]
        }
        result = _compare(
            [_checkpoint(0, side="reference")],
            [_checkpoint(0, side="tree")],
            output_offset=0,
            evidence=evidence,
            reference_actual=[10],
            tree_actual=[10],
        )
        row = result["rows"][0]
        self.assertFalse(row["native_online_verifier_flip"])
        self.assertTrue(row["same_mask_control_flip"])
        self.assertFalse(row["math_control_flip"])
        self.assertEqual(
            result["summary"]["failure_classification"],
            "unresolved_reference_state_divergence",
        )

    def test_unfaithful_reference_probe_cannot_attribute_state_drift(self) -> None:
        result = _compare(
            [_checkpoint(0, side="reference", kv=(0.0, 1.0), probe=(20, 10))],
            [_checkpoint(0, side="tree", kv=(0.0, 1.01), probe=(20, 10))],
            output_offset=0,
            reference_actual=[10],
            tree_actual=[20],
        )
        row = result["rows"][0]
        self.assertFalse(row["reference_probe_faithful"])
        self.assertEqual(row["reference_actual_token_id"], 10)
        self.assertEqual(row["reference_probe_argmax_token_id"], 20)
        self.assertEqual(row["reference_probe_fidelity_status"], "unfaithful")
        self.assertEqual(row["reference_probe_execution_provenance"], "actual")
        self.assertFalse(row["faithful_probe_decision_divergence"])
        self.assertEqual(result["summary"]["evidence_completeness"], "partial")
        self.assertEqual(
            result["summary"]["failure_classification"],
            "unresolved_reference_state_divergence",
        )

    def test_aligned_actual_proposal_drift_that_changes_path_is_draft_cause(
        self,
    ) -> None:
        reference_chain = DEFAULT_CHAIN
        tree_chain = (500, *DEFAULT_CHAIN[1:])
        reference = _checkpoint(
            0,
            side="reference",
            draft_top=reference_chain,
            proposal=reference_chain,
            transition=_transition(0, [99, reference_chain[0]], 800),
        )
        tree = _checkpoint(
            0,
            side="tree",
            draft_top=tree_chain,
            transition=_transition(0, [99, tree_chain[0]], 801),
        )
        result = _compare(
            [reference],
            [tree],
            output_offset=0,
            reference_actual=[10],
            tree_actual=[10],
        )
        self.assertTrue(result["rows"][0]["proposal_selected_path_drift"])
        self.assertEqual(
            result["summary"]["failure_classification"],
            "draft_state_or_candidate_drift",
        )

    def test_same_entry_actual_transition_mismatch_is_commit_cause(self) -> None:
        reference = _checkpoint(
            0,
            side="reference",
            transition=_transition(0, [99, 1], 800),
        )
        tree = _checkpoint(
            0,
            side="tree",
            transition=_transition(0, [99], 700),
        )
        result = _compare(
            [reference],
            [tree],
            output_offset=0,
            reference_actual=[10],
            tree_actual=[10],
        )
        self.assertEqual(
            result["summary"]["failure_classification"],
            "commit_or_root_transition_mismatch",
        )
        self.assertEqual(result["summary"]["classification_priority"], 2)

    def test_case2_pattern_is_unresolved_not_prefix_zero_draft_drift(self) -> None:
        inconsistent = (777, *DEFAULT_CHAIN[1:])
        reference = [
            _checkpoint(
                0,
                side="reference",
                draft_top=inconsistent,
                proposal=DEFAULT_CHAIN,
            ),
            _checkpoint(33, side="reference", probe=(2, 1)),
        ]
        tree = [
            _checkpoint(0, side="tree", draft_top=DEFAULT_CHAIN),
            _checkpoint(33, side="tree", probe=(2, 1)),
        ]
        result = _compare(
            reference,
            tree,
            output_offset=33,
            reference_actual=[10] * 33 + [1],
            tree_actual=[10] * 33 + [2],
        )
        self.assertEqual(
            result["summary"]["failure_classification"],
            "unresolved_reference_state_divergence",
        )
        self.assertNotEqual(
            result["summary"]["failure_classification"],
            "draft_state_or_candidate_drift",
        )
        self.assertFalse(result["rows"][-1]["reference_probe_faithful"])

    def test_case3_pattern_prefers_actual_native_verifier_flip(self) -> None:
        inconsistent = (777, *DEFAULT_CHAIN[1:])
        reference = [
            _checkpoint(
                0,
                side="reference",
                draft_top=inconsistent,
                proposal=DEFAULT_CHAIN,
            ),
            _checkpoint(18, side="reference", probe=(1, 2)),
        ]
        tree = [
            _checkpoint(0, side="tree", draft_top=DEFAULT_CHAIN),
            _checkpoint(18, side="tree", probe=(2, 1)),
        ]
        evidence = {
            18: [
                {
                    "comparison_mode": "native",
                    "verifier_decision_relevant": True,
                    "argmax_equal": False,
                },
                {
                    "comparison_mode": "same_mask",
                    "verifier_decision_relevant": True,
                    "argmax_equal": False,
                },
                {
                    "comparison_mode": "math",
                    "verifier_decision_relevant": True,
                    "argmax_equal": True,
                },
            ]
        }
        result = _compare(
            reference,
            tree,
            output_offset=18,
            evidence=evidence,
            reference_actual=[10] * 18 + [1],
            tree_actual=[10] * 18 + [2],
        )
        self.assertEqual(
            result["summary"]["failure_classification"],
            "local_packed_verifier_numerical_divergence",
        )
        self.assertEqual(result["summary"]["classification_evidence_prefix"], 18)

    def test_case4_pattern_same_mask_only_and_unfaithful_probe_is_unresolved(
        self,
    ) -> None:
        inconsistent = (777, *DEFAULT_CHAIN[1:])
        reference = [
            _checkpoint(
                0,
                side="reference",
                draft_top=inconsistent,
                proposal=DEFAULT_CHAIN,
            ),
            _checkpoint(28, side="reference", probe=(2, 1)),
        ]
        tree = [
            _checkpoint(0, side="tree", draft_top=DEFAULT_CHAIN),
            _checkpoint(28, side="tree", probe=(2, 1)),
        ]
        evidence = {
            28: [
                {
                    "comparison_mode": "native",
                    "verifier_decision_relevant": True,
                    "argmax_equal": True,
                },
                {
                    "comparison_mode": "same_mask",
                    "verifier_decision_relevant": True,
                    "argmax_equal": False,
                },
                {
                    "comparison_mode": "math",
                    "verifier_decision_relevant": True,
                    "argmax_equal": True,
                },
            ]
        }
        result = _compare(
            reference,
            tree,
            output_offset=28,
            evidence=evidence,
            reference_actual=[10] * 28 + [1],
            tree_actual=[10] * 28 + [2],
        )
        self.assertEqual(
            result["summary"]["failure_classification"],
            "unresolved_reference_state_divergence",
        )
        self.assertFalse(result["rows"][-1]["native_online_verifier_flip"])
        self.assertTrue(result["rows"][-1]["same_mask_control_flip"])
        self.assertFalse(result["rows"][-1]["reference_probe_faithful"])

    def test_round_segmentation_does_not_control_alignment(self) -> None:
        reference = _checkpoint(4, side="reference", round_index=1)
        tree = _checkpoint(4, side="tree", round_index=3)
        result = _compare([reference], [tree])
        row = result["rows"][0]
        self.assertEqual(row["generated_prefix_length"], 4)
        self.assertEqual(row["reference_round_index"], 1)
        self.assertEqual(row["tree_round_index"], 3)
        self.assertFalse(row["round_segmentation_equal_assumed"])
        self.assertTrue(row["semantic_prefix_token_exact"])

    def test_gold_evidence_uses_absolute_semantic_position_not_round(self) -> None:
        evidence = verifier_evidence_by_prefix(
            gold_records=[
                {
                    "round_index": 99,
                    "node_depth": 2,
                    "persistent_prefix_length": 104,
                    "predicted_token_semantic_position": 107,
                    "comparison_mode": "same_mask",
                }
            ],
            prompt_length=100,
        )
        self.assertEqual(list(evidence), [7])
        self.assertEqual(evidence[7][0]["round_index"], 99)

    def test_reference_state_schemas_accept_synthetic_outputs(self) -> None:
        result = _compare(
            [_checkpoint(0, side="reference")],
            [_checkpoint(0, side="tree")],
        )
        common = {
            "schema_version": "1.0",
            "task": "gov_report",
            "sample_id": "sample",
            "nominal_prompt_length": 1024,
            "actual_prompt_length": 1024,
            "configuration_id": "layerwise_beam-b16-w4",
        }
        row = {**result["rows"][0], **common}
        layer = {**result["layer_rows"][0], **common}
        case = {
            **result["summary"],
            **common,
            "diagnostic_runs_excluded_from_formal_timing": True,
            "reference_execution_provenance": "actual_reference_checkpoint",
            "tree_execution_provenance": "diagnostic_replay",
            "full_float32_kv_cpu_copy_created": False,
            "tensor_comparison_scope": "bounded_stable_samples",
        }
        summary = {
            "schema_version": "1.0",
            "status": "complete",
            "case_count": 1,
            "cases": [case],
            "absolute_generated_prefix_alignment_used": True,
            "round_index_alignment_assumed": False,
            "formal_timing_contaminated": False,
            "target_only_used_as_authoritative_oracle": False,
        }
        first = {
            "schema_version": "1.0",
            "status": "complete",
            "cases": [case],
        }
        schema_root = REPO_ROOT / "docs" / "schemas"
        for filename, payload in (
            ("phase2c-reference-state-differential.schema.json", row),
            ("phase2c-reference-state-layer.schema.json", layer),
            ("phase2c-reference-state-summary.schema.json", summary),
            ("phase2c-first-reference-state-divergence.schema.json", first),
        ):
            schema = json.loads((schema_root / filename).read_text())
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(schema).validate(payload)


class Phase2C4RunnerTests(unittest.TestCase):
    def test_csv_contract_includes_semantic_fix_fields(self) -> None:
        required = {
            "candidate_alignment_status",
            "native_online_verifier_flip",
            "same_mask_control_flip",
            "math_control_flip",
            "reference_actual_token_id",
            "reference_probe_faithful",
            "tree_actual_token_id",
            "tree_probe_faithful",
            "classification_evidence_prefix",
            "classification_evidence_source",
            "classification_priority",
            "downstream_consequence_only",
            "unresolved_reason",
        }
        self.assertTrue(required <= set(REFERENCE_STATE_CSV_FIELDS))

    def test_cli_enforces_complete_shadow_diagnostic_controls(self) -> None:
        args = parse_args(
            [
                "--dataset-jsonl",
                "/tmp/longbench.jsonl",
                "--materialized-draft-path",
                "/tmp/draft",
                "--tree-policy",
                "layerwise_beam",
                "--decision-stabilization",
                "shadow",
                "--enable-gold-checks",
                "--gold-comparison-mode",
                "native",
                "--gold-comparison-mode",
                "same_mask",
                "--gold-comparison-mode",
                "math",
                "--enable-reference-state-diagnostics",
            ]
        )
        self.assertTrue(args.enable_reference_state_diagnostics)
        self.assertEqual(args.decision_stabilization, "shadow")
        self.assertEqual(run_configuration(args)["phase_detail"], "2C.4.1")

    def test_dynamic_capture_is_bounded_and_timing_separated(self) -> None:
        source = (
            REPO_ROOT / "third_party/qwen3_8b_dflash_b16/instrumented/dflash.py"
        ).read_text()
        self.assertIn("actual_reference_checkpoint", source)
        self.assertIn('"checkpoint_provenance": "diagnostic_replay"', source)
        self.assertIn("native packed-tree verification and commit", source)
        self.assertIn('"full_float32_kv_cpu_copy_created": False', source)
        self.assertIn('"formal_timing_included": False', source)
        self.assertNotIn("flattened = tensor.reshape(-1)", source)
        self.assertIn("coordinates = [[] for _ in tensor.shape]", source)


if __name__ == "__main__":
    unittest.main()
