from __future__ import annotations

import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase2b.capture import build_capture_round_rows
from dflash.phase2b.tree import covered_target_prefix_depth, score_beam_tree
from dflash.phase2b1.analysis import (
    aggregate_events,
    analyze_capture_rounds,
    round_event_rows,
)
from dflash.phase2b1.confidence import _binary_metrics, draft_confidence_features
from dflash.phase2b1.policies import (
    layerwise_beam_tree,
    minimum_survival_budget,
)
from dflash.phase2b1.ranking import (
    brute_force_target_prefix_score_rank,
    capped_target_prefix_score_rank,
)
from dflash.phase2b1.validation import validate_capture_directories
from scripts.phase2b1_path_survival import (
    OUTPUT_FILENAMES,
    _validate_output,
    parse_args,
)


def _positions(depth: int = 15, width: int = 16) -> list[dict]:
    rows = []
    for position in range(1, depth + 1):
        candidates = [
            {
                "rank": rank,
                "token_id": position * 100 + rank,
                "log_probability": -0.1 * rank,
            }
            for rank in range(1, width + 1)
        ]
        rows.append(
            {
                "block_position": position,
                "baseline_token_id": candidates[0]["token_id"],
                "candidates": candidates,
                "top1_top2_log_probability_margin": 0.1,
            }
        )
    return rows


def _capture(
    *,
    reference_ranks: list[int | None],
    accepted: int = 0,
    sample_id: str = "sample-a",
    round_index: int = 0,
    terminal: bool = False,
) -> dict:
    positions = _positions()
    reference = [
        positions[index]["candidates"][rank - 1]["token_id"]
        if rank is not None
        else 999_000 + index
        for index, rank in enumerate(reference_ranks)
    ]
    return {
        "schema_version": "1.0",
        "task": "gov_report",
        "nominal_prompt_length": 32768,
        "actual_prompt_length": 32760,
        "sample_id": sample_id,
        "source_sample_id": sample_id,
        "source_text_sha256": "a" * 64,
        "generation_repeat": 0,
        "round_index": round_index,
        "terminal_round": terminal,
        "stop_reason": "max_new_tokens" if terminal else "continue",
        "reference_right_censored": len(reference) < 15,
        "full_depth_evaluation_eligible": len(reference) == 15,
        "accepted_draft_token_count": accepted,
        "positions": positions,
        "reference_target_continuation_token_ids": reference,
    }


def _built_capture(*, censored: bool = False) -> tuple[dict, dict]:
    positions = _positions()
    baseline = [position["baseline_token_id"] for position in positions]
    accepted = 1
    correction = positions[1]["candidates"][1]["token_id"]
    generated = (
        [42, baseline[0]] if censored else [42, baseline[0], correction, *baseline[2:]]
    )
    diagnostics = {
        "tree_candidate_capture": {"enabled": True},
        "draft_block_size": 16,
        "stop_reason": "max_new_tokens" if censored else "continue",
        "rounds": [
            {
                "round_index": 0,
                "context_length_before": 1000,
                "context_length_after": 1002,
                "accepted_draft_token_count": accepted,
                "rejected_draft_token_count": 14,
                "committed_token_count": 2,
                "raw_cumulative_generated_token_count": 2,
                "cumulative_generated_token_count": 2,
                "returned_token_delta": 2,
                "stop_reason": "max_new_tokens" if censored else "continue",
                "truncated_token_count": 1 if censored else 0,
                "tree_candidate_capture": {
                    "round_index": 0,
                    "root_token_id": 42,
                    "root_absolute_output_index": 1000,
                    "round_output_start_offset": 0,
                    "baseline_draft_token_ids": baseline,
                    "positions": positions,
                    "top_k": 16,
                    "accepted_draft_token_count": accepted,
                    "immediate_verification_lcp": accepted,
                    "immediate_alignment_verified": True,
                    "logit_source": "exact final draft logits used by temperature-zero sampling",
                    "score_semantics": "normalized log probability from log_softmax",
                    "tie_break_rule": "descending_logit_then_ascending_token_id",
                },
            }
        ],
    }
    metadata = {
        "task": "gov_report",
        "dataset_split": "test",
        "sample_id": "sample-a",
        "source_sample_id": "sample-a",
        "source_index": 0,
        "source_text_sha256": "a" * 64,
        "context_prefix_sha256": "b" * 64,
        "prompt_sha256": "c" * 64,
        "input_token_ids_sha256": "d" * 64,
        "nominal_prompt_length": 1024,
        "actual_prompt_length": 1020,
        "generation_repeat": 0,
        "capture_source_provenance": {
            "repository_custom_code_commit": "f" * 40,
            "draft_model_id": "z-lab/Qwen3-8B-DFlash-b16",
            "draft_model_revision": "9b41424b7109f9c5413454f481b09a82b85333f4",
            "materialization_id": "fixture",
            "actual_loaded_module_sha256": "1" * 64,
            "actual_loaded_dependency_sha256": {"phase1_observability.py": "2" * 64},
            "target_model_id": "Qwen/Qwen3-8B",
            "target_model_revision": "3" * 40,
        },
    }
    row = build_capture_round_rows(
        diagnostics=diagnostics,
        final_verified_generated_token_ids=generated,
        metadata=metadata,
    )[0]
    sample = {
        "schema_version": "1.0",
        "task": "gov_report",
        "dataset_split": "test",
        "sample_id": "sample-a",
        "source_sample_id": "sample-a",
        "source_text_sha256": "a" * 64,
        "context_prefix_sha256": "b" * 64,
        "nominal_prompt_length": 1024,
        "actual_prompt_length": 1020,
        "generated_token_count": len(generated),
        "final_verified_output_token_ids": generated,
        "control_capture_output_token_ids_equal": True,
        "capture_repeats": [{"repeat": 0}],
        "full_vocabulary_logits_saved": False,
        "target_logits_saved": False,
    }
    return row, sample


def _write_fixture(directory: Path, *, duplicate: bool = False) -> None:
    row, sample = _built_capture()
    manifest = {
        "schema_version": "1.0",
        "status": "ok",
        "configuration": {
            "phase": "2B",
            "mode": "exact_draft_candidate_capture",
            "top_k": 16,
            "block_size": 16,
            "batch_size": 1,
            "temperature": 0,
            "control_then_capture": True,
            "control_capture_token_exact_required": True,
            "full_vocabulary_logits_saved": False,
            "target_logits_saved": False,
        },
        "workload": {},
        "measurement_policy": {},
        "alignment_policy": {},
        "interpretation": {},
        "artifacts": {},
        "error": None,
    }
    provenance = {
        "schema_version": "1.0",
        "status": "ok",
        "repository_custom_code_commit": "f" * 40,
        "base_model_id": "z-lab/Qwen3-8B-DFlash-b16",
        "base_model_revision": "9b41424b7109f9c5413454f481b09a82b85333f4",
        "materialization_id": "fixture",
        "materialized_model_path": "/fixture",
        "custom_code_sha256": {},
        "actual_loaded_module_path": "/fixture/modeling_dflash.py",
        "actual_loaded_module_sha256": "1" * 64,
        "loaded_source_matches_repository": True,
        "spec_generate": {},
        "original": {},
        "error": None,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "samples.jsonl").write_text(
        json.dumps(sample) + "\n", encoding="utf-8"
    )
    round_lines = [json.dumps(row), json.dumps(row)] if duplicate else [json.dumps(row)]
    (directory / "tree-capture-rounds.jsonl").write_text(
        "\n".join(round_lines) + "\n", encoding="utf-8"
    )
    (directory / "environment.json").write_text("{}\n", encoding="utf-8")
    (directory / "source-provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    (directory / "terminal.log").write_text("fixture\n", encoding="utf-8")


class Phase2B1Tests(unittest.TestCase):
    def test_first_unavailable_and_rank_over_16(self) -> None:
        result = analyze_capture_rounds(
            [_capture(reference_ranks=[1, None, 2])],
            node_budgets=(16, 32),
            beam_widths=(2,),
            rank_cap=100,
        )
        event = next(
            item for item in result["events"] if item["policy"] == "score_beam"
        )
        self.assertEqual(event["first_candidate_unavailable_depth"], 2)
        self.assertIsNone(event["depth_events"][1]["target_token_rank"])
        self.assertEqual(event["depth_events"][1]["target_token_rank_label"], ">16")
        self.assertAlmostEqual(event["target_rank_recall_at_16"], 2 / 3)

    def test_first_pruned_depth(self) -> None:
        result = analyze_capture_rounds(
            [_capture(reference_ranks=[2, 1])],
            node_budgets=(16, 32),
            beam_widths=(2,),
            rank_cap=100,
        )
        top1 = next(item for item in result["events"] if item["policy"] == "top1_chain")
        self.assertEqual(top1["first_pruned_depth"], 1)
        self.assertEqual(
            top1["depth_events"][0]["pruning_reason"], "candidate_available_but_pruned"
        )

    def test_rank_one_through_sixteen(self) -> None:
        expected_ranks = [*range(1, 15), 16]
        result = analyze_capture_rounds(
            [_capture(reference_ranks=expected_ranks)],
            node_budgets=(16, 32),
            beam_widths=(2,),
            rank_cap=10,
        )
        event = result["events"][0]
        self.assertEqual(
            [row["target_token_rank"] for row in event["depth_events"]],
            expected_ranks,
        )

    def test_right_censored_terminal_round(self) -> None:
        capture = _capture(reference_ranks=[1, 1], accepted=2, terminal=True)
        event = analyze_capture_rounds(
            [capture], node_budgets=(16, 32), beam_widths=(2,), rank_cap=100
        )["events"][0]
        self.assertTrue(event["terminal_round"])
        self.assertTrue(event["reference_right_censored"])
        self.assertEqual(len(event["depth_events"]), 2)

    def test_off_top1_count_and_span(self) -> None:
        event = analyze_capture_rounds(
            [_capture(reference_ranks=[1, 2, 3, 1, 2])],
            node_budgets=(16, 32),
            beam_widths=(2,),
            rank_cap=100,
        )["events"][0]
        self.assertEqual(event["off_top1_decision_count"], 3)
        self.assertEqual(event["longest_consecutive_off_top1_span"], 2)
        self.assertEqual(event["first_off_top1_decision_depth"], 2)

    def test_exact_rank_matches_brute_force(self) -> None:
        positions = _positions(depth=3, width=3)
        target = [102, 202, 302]
        exact = capped_target_prefix_score_rank(positions, target, rank_cap=1000)
        self.assertEqual(
            exact["target_prefix_score_rank"],
            brute_force_target_prefix_score_rank(positions, target),
        )

    def test_rank_cap_is_lower_bound_not_rank(self) -> None:
        positions = _positions(depth=3, width=3)
        result = capped_target_prefix_score_rank(positions, [103, 203, 303], rank_cap=2)
        self.assertTrue(result["target_prefix_score_rank_censored"])
        self.assertIsNone(result["target_prefix_score_rank"])
        self.assertEqual(result["target_prefix_score_rank_lower_bound"], 3)

    def test_minimum_survival_budget(self) -> None:
        self.assertEqual(
            minimum_survival_budget({16: 0, 32: 2, 64: 2}, depth=1)[
                "minimum_total_tree_node_budget_to_preserve_target_prefix"
            ],
            32,
        )

    def test_minimum_budget_over_maximum(self) -> None:
        result = minimum_survival_budget({16: 0, 32: 0, 64: 0}, depth=1)
        self.assertTrue(result["minimum_budget_censored"])
        self.assertEqual(result["minimum_budget_label"], ">64")

    def test_nonmonotonic_survival_is_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not monotonic"):
            minimum_survival_budget({16: 1, 32: 0}, depth=1)

    def test_score_beam_replay_matches_phase2b(self) -> None:
        capture = _capture(reference_ranks=[2, 1, 1])
        result = analyze_capture_rounds(
            [capture], node_budgets=(16, 32), beam_widths=(2,), rank_cap=100
        )
        event = next(
            item
            for item in result["events"]
            if item["policy"] == "score_beam"
            and item["requested_total_tree_node_budget"] == 32
        )
        expected = covered_target_prefix_depth(
            score_beam_tree(capture["positions"], budget=32),
            capture["reference_target_continuation_token_ids"],
        )
        self.assertEqual(event["covered_target_prefix_depth"], expected)

    def test_layerwise_beam_prefix_closure_and_budget_nesting(self) -> None:
        small = layerwise_beam_tree(_positions(), budget=32, beam_width=4)
        large = layerwise_beam_tree(_positions(), budget=64, beam_width=4)
        self.assertTrue(small <= large)
        self.assertTrue(all(not node or node[:-1] in small for node in small))

    def test_layerwise_and_confidence_features_are_target_blind(self) -> None:
        self.assertNotIn("target", inspect.signature(layerwise_beam_tree).parameters)
        self.assertNotIn("reference", inspect.signature(layerwise_beam_tree).parameters)
        self.assertEqual(
            list(inspect.signature(draft_confidence_features).parameters), ["position"]
        )

    def test_topk_mass_and_truncated_entropy(self) -> None:
        features = draft_confidence_features(_positions(depth=1)[0])
        self.assertGreater(features["top_k_probability_mass"], 0)
        self.assertGreater(features["top_k_renormalized_truncated_entropy"], 0)
        self.assertLess(features["top_k_probability_mass"], 16)

    def test_auroc_auprc_and_single_class(self) -> None:
        perfect = _binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(perfect["auroc"], 1.0)
        self.assertEqual(perfect["auprc"], 1.0)
        single = _binary_metrics([0, 0], [0.1, 0.2])
        self.assertIsNone(single["auroc"])
        self.assertEqual(single["unavailable_reason"], "single_class_all_negative")

    def test_round_micro_and_sample_macro_are_explicit(self) -> None:
        analysis = analyze_capture_rounds(
            [
                _capture(reference_ranks=[1], sample_id="a", round_index=0),
                _capture(reference_ranks=[2], sample_id="a", round_index=1),
                _capture(reference_ranks=[1], sample_id="b", round_index=0),
            ],
            node_budgets=(16, 32),
            beam_widths=(2,),
            rank_cap=100,
        )
        summary = aggregate_events(analysis["events"])
        group = next(row for row in summary["groups"] if row["policy"] == "top1_chain")
        self.assertEqual(group["round_count"], 3)
        self.assertEqual(group["sample_count"], 2)
        self.assertTrue(group["sample_macro"]["sample_is_independence_unit"])

    def test_censored_round_checkpoint_denominator(self) -> None:
        analysis = analyze_capture_rounds(
            [_capture(reference_ranks=[1, 1], terminal=True)],
            node_budgets=(16, 32),
            beam_widths=(2,),
            rank_cap=100,
        )
        group = next(
            row
            for row in aggregate_events(analysis["events"])["groups"]
            if row["policy"] == "top1_chain"
        )
        self.assertEqual(group["depth_survival"]["4"]["eligible_round_count"], 0)

    def test_capture_schema_and_provenance_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _write_fixture(directory)
            result = validate_capture_directories(
                [directory], schema_dir=REPO_ROOT / "docs/schemas"
            )
            self.assertEqual(len(result["rounds"]), 1)
            self.assertEqual(result["inputs"][0]["input_git_sha"], "f" * 40)

    def test_duplicate_round_id_fails_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _write_fixture(directory, duplicate=True)
            with self.assertRaisesRegex(ValueError, "duplicate round ID"):
                validate_capture_directories(
                    [directory], schema_dir=REPO_ROOT / "docs/schemas"
                )

    def test_cli_defaults_and_overrides(self) -> None:
        defaults = parse_args(["--capture-dir", "capture"])
        self.assertEqual(defaults.node_budgets, (16, 32, 64, 128, 256))
        self.assertEqual(defaults.rank_cap, 100_000)
        changed = parse_args(
            [
                "--capture-dir",
                "a",
                "--capture-dir",
                "b",
                "--node-budgets",
                "16,32,512",
                "--beam-widths",
                "2,8",
                "--rank-cap",
                "7",
            ]
        )
        self.assertEqual(len(changed.capture_dir), 2)
        self.assertEqual(changed.rank_cap, 7)

    def test_new_schemas_are_draft_2020_12(self) -> None:
        for path in (REPO_ROOT / "docs/schemas").glob("phase2b1-*.schema.json"):
            schema = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
            )
            Draft202012Validator.check_schema(schema)

    def test_generated_json_records_validate_new_schemas(self) -> None:
        analysis = analyze_capture_rounds(
            [_capture(reference_ranks=[1, 2])],
            node_budgets=(16, 32),
            beam_widths=(2,),
            rank_cap=100,
        )
        summary = aggregate_events(analysis["events"])
        event = round_event_rows(analysis["events"])[0]
        _validate_output(event, "phase2b1-path-survival-event.schema.json")
        _validate_output(summary, "phase2b1-path-survival-summary.schema.json")
        _validate_output(
            analysis["confidence"], "phase2b1-confidence-diagnostics.schema.json"
        )

    def test_lazy_import_does_not_load_model_frameworks(self) -> None:
        code = (
            "import sys; import dflash.phase2b1.analysis; "
            "import scripts.phase2b1_path_survival; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)

    def test_expected_artifact_inventory(self) -> None:
        self.assertEqual(len(OUTPUT_FILENAMES), 12)
        self.assertIn("minimum-budget.csv", OUTPUT_FILENAMES)
        self.assertIn("long-context-summary.csv", OUTPUT_FILENAMES)


if __name__ == "__main__":
    unittest.main()
