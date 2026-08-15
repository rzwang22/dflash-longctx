from __future__ import annotations

import importlib.util
import inspect
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase2b.capture import (
    build_capture_round_rows,
    validate_capture_round,
)
from dflash.phase2b.metrics import aggregate_tree_events
from dflash.phase2b.tree import (
    brute_force_oracle_coverage,
    confidence_gated_tree,
    covered_target_prefix_depth,
    oracle_prefix_closed_tree,
    policy_event,
    score_beam_tree,
    top1_chain_tree,
    validate_prefix_closed,
)
from experiments.phase2b_tree_capture import _new_manifest as new_capture_manifest
from experiments.phase2b_tree_capture import parse_args as parse_capture_args
from scripts.phase2b_tree_replay import (
    main as replay_main,
)


def _load_observability_helper():
    path = (
        REPO_ROOT
        / "third_party/qwen3_8b_dflash_b16/instrumented/phase1_observability.py"
    )
    spec = importlib.util.spec_from_file_location("phase2b_helper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
from scripts.phase2b_tree_replay import (
    parse_args as parse_replay_args,
)
from scripts.phase2b_tree_replay import (
    replay_captures,
)


def _positions(depth: int = 15, width: int = 16) -> list[dict]:
    positions = []
    for position in range(1, depth + 1):
        candidates = [
            {
                "rank": rank,
                "token_id": position * 100 + rank,
                "log_probability": -0.1 - 0.1 * (rank - 1),
            }
            for rank in range(1, width + 1)
        ]
        positions.append(
            {
                "block_position": position,
                "baseline_token_id": candidates[0]["token_id"],
                "candidates": candidates,
                "top1_top2_log_probability_margin": 0.1,
            }
        )
    return positions


def _diagnostics(*, accepted: int, stop_reason: str = "continue") -> dict:
    positions = _positions()
    baseline = [position["baseline_token_id"] for position in positions]
    capture = {
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
    }
    return {
        "tree_candidate_capture": {"enabled": True},
        "draft_block_size": 16,
        "stop_reason": stop_reason,
        "rounds": [
            {
                "round_index": 0,
                "context_length_before": 1000,
                "context_length_after": 1001 + accepted,
                "accepted_draft_token_count": accepted,
                "rejected_draft_token_count": 15 - accepted,
                "committed_token_count": 1 + accepted,
                "raw_cumulative_generated_token_count": 2 + accepted,
                "cumulative_generated_token_count": 2 + accepted,
                "returned_token_delta": 1 + accepted,
                "stop_reason": stop_reason,
                "truncated_token_count": 0 if stop_reason == "continue" else 4,
                "tree_candidate_capture": capture,
            }
        ],
    }


def _metadata(sample_id: str = "sample-a") -> dict:
    return {
        "task": "gov_report",
        "dataset_split": "test",
        "sample_id": sample_id,
        "source_sample_id": sample_id,
        "source_index": 0,
        "source_text_sha256": "a" * 64,
        "context_prefix_sha256": "b" * 64,
        "prompt_sha256": "c" * 64,
        "input_token_ids_sha256": "d" * 64,
        "nominal_prompt_length": 1024,
        "actual_prompt_length": 1022,
        "generation_repeat": 0,
        "capture_source_provenance": {
            "repository_custom_code_commit": "f" * 40,
            "draft_model_id": "z-lab/Qwen3-8B-DFlash-b16",
            "draft_model_revision": "e" * 40,
            "materialization_id": "phase2b-test",
            "actual_loaded_module_sha256": "1" * 64,
            "actual_loaded_dependency_sha256": {
                "phase1_observability.py": "2" * 64
            },
            "target_model_id": "Qwen/Qwen3-8B",
            "target_model_revision": "3" * 40,
        },
    }


def _capture(*, accepted: int = 2, censored: bool = False, sample_id: str = "sample-a"):
    diagnostics = _diagnostics(
        accepted=accepted, stop_reason="max_new_tokens" if censored else "continue"
    )
    baseline = diagnostics["rounds"][0]["tree_candidate_capture"][
        "baseline_draft_token_ids"
    ]
    if censored:
        generated = [42, *baseline[:accepted]]
    else:
        correction = diagnostics["rounds"][0]["tree_candidate_capture"]["positions"][
            accepted
        ]["candidates"][1]["token_id"]
        tail = [
            position["candidates"][1]["token_id"]
            for position in diagnostics["rounds"][0]["tree_candidate_capture"][
                "positions"
            ][accepted + 1 :]
        ]
        generated = [42, *baseline[:accepted], correction, *tail]
    return build_capture_round_rows(
        diagnostics=diagnostics,
        final_verified_generated_token_ids=generated,
        metadata=_metadata(sample_id),
    )[0]


class Phase2BTests(unittest.TestCase):
    def test_capture_mode_validation_is_strictly_isolated(self) -> None:
        helper = _load_observability_helper()
        common = {
            "enable_tree_capture": True,
            "tree_top_k": 16,
            "enable_rank_diagnostics": False,
            "enable_verification_profiling": False,
            "enable_profiling": False,
            "return_diagnostics": True,
            "temperature": 0.0,
        }
        self.assertIsNone(helper.validate_phase2b_capture_options(**common))
        with self.assertRaisesRegex(ValueError, "return_diagnostics"):
            helper.validate_phase2b_capture_options(
                **{**common, "return_diagnostics": False}
            )
        with self.assertRaisesRegex(ValueError, "temperature=0"):
            helper.validate_phase2b_capture_options(
                **{**common, "temperature": 0.5}
            )
        with self.assertRaisesRegex(ValueError, "isolated"):
            helper.validate_phase2b_capture_options(
                **{**common, "enable_rank_diagnostics": True}
            )

    def test_uncensored_top1_lcp_equals_accepted_and_offsets_align(self) -> None:
        capture = _capture(accepted=3)
        validate_capture_round(capture)
        self.assertEqual(capture["alignment"]["final_output_lcp"], 3)
        self.assertTrue(capture["alignment"]["exact_lcp_invariant_holds"])
        self.assertEqual(capture["round_output_start_offset"], 0)
        self.assertEqual(capture["root_absolute_output_index"], 1000)

    def test_right_censoring_keeps_only_observable_prefix(self) -> None:
        capture = _capture(accepted=5, censored=True)
        validate_capture_round(capture)
        self.assertTrue(capture["reference_right_censored"])
        self.assertFalse(capture["alignment"]["exact_lcp_invariant_evaluable"])
        self.assertIsNone(capture["alignment"]["exact_lcp_invariant_holds"])
        self.assertEqual(capture["reference_observed_depth"], 5)

    def test_eos_censoring_is_distinctly_recorded(self) -> None:
        diagnostics = _diagnostics(accepted=2, stop_reason="eos")
        baseline = diagnostics["rounds"][0]["tree_candidate_capture"][
            "baseline_draft_token_ids"
        ]
        capture = build_capture_round_rows(
            diagnostics=diagnostics,
            final_verified_generated_token_ids=[42, *baseline[:2]],
            metadata=_metadata(),
        )[0]
        self.assertEqual(capture["censor_reason"], "eos")
        self.assertTrue(capture["terminal_round"])
        self.assertTrue(capture["reference_right_censored"])

    def test_censored_round_can_still_have_exact_observable_lcp(self) -> None:
        diagnostics = _diagnostics(accepted=2, stop_reason="eos")
        captured = diagnostics["rounds"][0]["tree_candidate_capture"]
        baseline = captured["baseline_draft_token_ids"]
        correction = captured["positions"][2]["candidates"][1]["token_id"]
        capture = build_capture_round_rows(
            diagnostics=diagnostics,
            final_verified_generated_token_ids=[42, *baseline[:2], correction],
            metadata=_metadata(),
        )[0]
        self.assertTrue(capture["reference_right_censored"])
        self.assertTrue(capture["alignment"]["exact_lcp_invariant_evaluable"])
        self.assertTrue(capture["alignment"]["exact_lcp_invariant_holds"])

    def test_alignment_failure_has_round_position_ids_and_offset(self) -> None:
        diagnostics = _diagnostics(accepted=2)
        with self.assertRaisesRegex(
            RuntimeError, r"round=0, position=2.*root_output_offset=0.*draft=.*reference="
        ):
            build_capture_round_rows(
                diagnostics=diagnostics,
                final_verified_generated_token_ids=[42, 101, 999],
                metadata=_metadata(),
            )

    def test_top1_chain_reproduces_baseline_accepted_depth(self) -> None:
        capture = _capture(accepted=4)
        nodes = top1_chain_tree(capture["positions"], budget=16)
        self.assertEqual(len(nodes), 16)
        self.assertEqual(
            covered_target_prefix_depth(
                nodes, capture["reference_target_continuation_token_ids"]
            ),
            4,
        )

    def test_prefix_closure_unique_nodes_and_budgets(self) -> None:
        positions = _positions()
        for budget in (16, 32, 64):
            nodes = score_beam_tree(positions, budget=budget)
            self.assertLessEqual(len(nodes), budget)
            self.assertEqual(nodes, validate_prefix_closed(nodes, budget=budget))
            self.assertEqual(len(nodes), len(set(nodes)))
            self.assertTrue(all(len(node) <= 15 for node in nodes))

    def test_score_beam_is_deterministic(self) -> None:
        positions = _positions()
        self.assertEqual(
            score_beam_tree(positions, budget=64),
            score_beam_tree(positions, budget=64),
        )

    def test_confidence_threshold_controls_branching_without_target(self) -> None:
        positions = _positions()
        low = confidence_gated_tree(
            positions, budget=32, margin_threshold=0.05, branch_width=4
        )
        high = confidence_gated_tree(
            positions, budget=32, margin_threshold=0.2, branch_width=4
        )
        self.assertEqual(len(low), 16)
        self.assertEqual(len(high), 32)
        self.assertNotIn("reference", inspect.signature(confidence_gated_tree).parameters)
        self.assertNotIn("reference", inspect.signature(score_beam_tree).parameters)

    def test_oracle_matches_bruteforce_on_synthetic_tree(self) -> None:
        positions = _positions(depth=2, width=2)
        reference = [102, 202]
        oracle = oracle_prefix_closed_tree(positions, reference, budget=3)
        self.assertEqual(covered_target_prefix_depth(oracle, reference), 2)
        self.assertEqual(
            covered_target_prefix_depth(oracle, reference),
            brute_force_oracle_coverage(positions, reference, budget=3),
        )

    def test_oracle_coverage_is_monotonic_with_budget(self) -> None:
        capture = _capture(accepted=1)
        reference = capture["reference_target_continuation_token_ids"]
        coverages = [
            covered_target_prefix_depth(
                oracle_prefix_closed_tree(capture["positions"], reference, budget=budget),
                reference,
            )
            for budget in (16, 32, 64)
        ]
        self.assertEqual(coverages, sorted(coverages))

    def test_oracle_dominates_deployable_and_labels_target_use(self) -> None:
        capture = _capture(accepted=1)
        events = replay_captures(
            [capture],
            budgets=(16, 32),
            policies=("top1_chain", "score_beam", "oracle_prefix_closed"),
            confidence_thresholds=(0.25,),
            confidence_branch_width=4,
        )
        oracle = {
            event["requested_total_tree_node_budget"]: event
            for event in events
            if event["policy_class"] == "oracle"
        }
        for event in events:
            if event["policy_class"] == "deployable":
                self.assertFalse(event["policy_uses_target_information"])
                self.assertGreaterEqual(
                    oracle[event["requested_total_tree_node_budget"]][
                        "covered_target_prefix_depth"
                    ],
                    event["covered_target_prefix_depth"],
                )
            else:
                self.assertTrue(event["policy_uses_target_information"])

    def test_event_metrics_and_terminal_aggregation(self) -> None:
        full = _capture(accepted=2, sample_id="a")
        censored = _capture(accepted=3, censored=True, sample_id="b")
        events = replay_captures(
            [full, censored],
            budgets=(16,),
            policies=("top1_chain",),
            confidence_thresholds=(0.25,),
            confidence_branch_width=4,
        )
        summary = aggregate_tree_events(events)
        group = next(iter(summary["by_task_context_budget_policy"].values()))
        self.assertEqual(group["sample_count"], 2)
        self.assertEqual(group["round_count"], 2)
        self.assertEqual(group["right_censored_round_count"], 1)
        self.assertEqual(group["full_depth_eligible_round_count"], 1)
        self.assertEqual(
            group["round_micro"]["mean_covered_target_prefix_depth"], 2.5
        )
        self.assertIn("per_sample", group["sample_macro"])

    def test_policy_event_marks_coverage_upper_bound_not_acceptance(self) -> None:
        capture = _capture(accepted=2)
        event = policy_event(
            capture=capture,
            nodes=top1_chain_tree(capture["positions"], budget=16),
            policy="top1_chain",
            policy_class="deployable",
            policy_variant="exact_baseline_chain",
            requested_budget=16,
            policy_parameters={},
        )
        self.assertEqual(event["committed_tokens_upper_bound"], 3)
        self.assertFalse(event["actual_accepted_length_derived"])
        self.assertFalse(event["latency_or_speedup_derived"])
        self.assertEqual(event["total_tree_nodes"], event["total_node_count"])
        self.assertEqual(
            event["draft_candidate_nodes"], event["candidate_node_count"]
        )

    def test_capture_and_replay_cli_defaults_and_overrides(self) -> None:
        capture = parse_capture_args(
            [
                "--dataset-jsonl",
                "fixture.jsonl",
                "--materialized-draft-path",
                "materialized",
            ]
        )
        self.assertEqual(capture.top_k, 16)
        self.assertEqual(capture.timing_repeats, 1)
        overridden = parse_capture_args(
            [
                "--dataset-jsonl",
                "fixture.jsonl",
                "--materialized-draft-path",
                "materialized",
                "--sample-id",
                "gov_report-fixed",
                "--context-lengths",
                "1K,32K",
                "--top-k",
                "8",
            ]
        )
        self.assertEqual(overridden.sample_id, "gov_report-fixed")
        self.assertEqual(overridden.context_lengths, (1024, 32768))
        self.assertEqual(overridden.top_k, 8)
        replay = parse_replay_args(["--capture-rounds", "rounds.jsonl"])
        self.assertEqual(replay.node_budgets, (16, 32, 64))
        self.assertEqual(replay.confidence_thresholds, (0.25, 0.5, 1.0))

    def test_phase2b_imports_do_not_load_model_frameworks(self) -> None:
        code = (
            "import sys; "
            "import experiments.phase2b_tree_capture; "
            "import scripts.phase2b_tree_replay; "
            "print(int('torch' in sys.modules), int('transformers' in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "0 0")

    def test_phase2b_schemas_are_draft_2020_12_and_validate_outputs(self) -> None:
        schema_root = REPO_ROOT / "docs" / "schemas"
        schemas = {}
        for path in sorted(schema_root.glob("phase2b-*.schema.json")):
            schema = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
            )
            Draft202012Validator.check_schema(schema)
            schemas[path.name] = schema
        self.assertEqual(len(schemas), 7)

        capture = _capture(accepted=2)
        Draft202012Validator(
            schemas["phase2b-capture-round.schema.json"]
        ).validate(capture)
        events = replay_captures(
            [capture],
            budgets=(16,),
            policies=("top1_chain", "oracle_prefix_closed"),
            confidence_thresholds=(0.25,),
            confidence_branch_width=4,
        )
        Draft202012Validator(schemas["phase2b-tree-event.schema.json"]).validate(
            events[0]
        )
        Draft202012Validator(
            schemas["phase2b-tree-summary.schema.json"]
        ).validate(aggregate_tree_events(events))
        capture_args = parse_capture_args(
            [
                "--dataset-jsonl",
                "fixture.jsonl",
                "--materialized-draft-path",
                "materialized",
            ]
        )
        Draft202012Validator(
            schemas["phase2b-capture-manifest.schema.json"]
        ).validate(
            new_capture_manifest(
                capture_args, {"repository": {"commit": "f" * 40}}, "capture"
            )
        )
        Draft202012Validator(
            schemas["phase2b-capture-sample.schema.json"]
        ).validate(
            {
                "schema_version": "1.0",
                "task": "gov_report",
                "sample_id": "sample-a",
                "source_sample_id": "sample-a",
                "source_text_sha256": "a" * 64,
                "nominal_prompt_length": 1024,
                "actual_prompt_length": 1022,
                "generated_token_count": 1,
                "final_verified_output_token_ids": [42],
                "control_capture_output_token_ids_equal": True,
                "capture_repeats": [{}],
                "full_vocabulary_logits_saved": False,
                "target_logits_saved": False,
            }
        )

    def test_cpu_replay_writes_expected_artifacts(self) -> None:
        capture = _capture(accepted=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture_path = root / "capture-rounds.jsonl"
            capture_path.write_text(json.dumps(capture) + "\n", encoding="utf-8")
            output = root / "replay"
            self.assertEqual(
                replay_main(
                    [
                        "--capture-rounds",
                        str(capture_path),
                        "--output-dir",
                        str(output),
                        "--node-budgets",
                        "16,32",
                    ]
                ),
                0,
            )
            expected = {
                "replay-manifest.json",
                "tree-policy-events.jsonl",
                "tree-summary.json",
                "tree-summary.csv",
                "tree-depth-coverage.csv",
                "tree-policy-config.json",
                "terminal.log",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            schema_root = REPO_ROOT / "docs" / "schemas"
            replay_manifest_schema = json.loads(
                (schema_root / "phase2b-replay-manifest.schema.json").read_text()
            )
            policy_schema = json.loads(
                (schema_root / "phase2b-tree-policy-config.schema.json").read_text()
            )
            Draft202012Validator(replay_manifest_schema).validate(
                json.loads((output / "replay-manifest.json").read_text())
            )
            Draft202012Validator(policy_schema).validate(
                json.loads((output / "tree-policy-config.json").read_text())
            )

    def test_instrumentation_source_is_opt_in_and_never_persists_full_logits(self) -> None:
        dflash_source = (
            REPO_ROOT
            / "third_party/qwen3_8b_dflash_b16/instrumented/dflash.py"
        ).read_text(encoding="utf-8")
        helper_source = (
            REPO_ROOT
            / "third_party/qwen3_8b_dflash_b16/instrumented/phase1_observability.py"
        ).read_text(encoding="utf-8")
        self.assertIn("enable_tree_capture: bool = False", dflash_source)
        self.assertIn("draft_logits=draft_logits", dflash_source)
        self.assertIn("extra_draft_forward_calls\": 0", dflash_source)
        self.assertIn("full-vocabulary logits remain transient", helper_source)
        self.assertNotIn("torch.save(", helper_source)

    def test_phase2b_repository_hygiene(self) -> None:
        candidate_files = [
            *REPO_ROOT.glob("dflash/phase2b/**/*.py"),
            REPO_ROOT / "experiments/phase2b_tree_capture.py",
            REPO_ROOT / "scripts/phase2b_tree_replay.py",
            *REPO_ROOT.glob("docs/schemas/phase2b-*.schema.json"),
        ]
        self.assertFalse(
            [
                path
                for path in REPO_ROOT.rglob("*")
                if path.is_file()
                and ".venv" not in path.parts
                and path.suffix in {".safetensors", ".bin", ".pt", ".pth"}
            ]
        )
        credential_pattern = re.compile(r"hf_[A-Za-z0-9]{20,}|BEGIN .*PRIVATE KEY")
        for path in candidate_files:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/mnt/sevenT", text, str(path))
            self.assertIsNone(credential_pattern.search(text), str(path))


if __name__ == "__main__":
    unittest.main()
