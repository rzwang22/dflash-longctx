from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from dflash.phase2d1bnr.aggregation import (
    aggregate_cohort_cases,
    aggregate_exact_reproducibility,
    build_margin_trigger_sweep,
)
from dflash.phase2d1bnr.cohort import (
    derive_matched_capture_cohort,
    subrun_identity,
)
from dflash.phase2d1bnr.decision import build_research_summary, choose_nr_decision
from dflash.phase2d1bnr.margins import extract_round_margin_events
from dflash.phase2d1bnr.parse import (
    ArtifactError,
    build_divergence_event,
    classify_subrun_artifacts,
    public_observation,
)
from experiments.phase2d1b_nr_numerical_robustness import (
    ARTIFACTS,
    _persist_results,
    parse_args,
)
from scripts.check_phase2d1b_nr_acceptance import check_output

REPO_ROOT = Path(__file__).resolve().parents[1]


def _validate(value: object, name: str) -> None:
    schema = json.loads(
        (REPO_ROOT / "docs" / "schemas" / name).read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(value)


def _hash(values: list[int]) -> str:
    payload = json.dumps(values, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _identity(
    *,
    experiment: str = "exact_case",
    mode: str = "global_fixed",
    budget: int = 16,
    repeat: int = 0,
    context: int = 1024,
) -> dict:
    return subrun_identity(
        experiment=experiment,
        task="gov_report",
        sample_id="gov_report-fd287bbfd3bdbdc9",
        source_sample_id="gov_report-fd287bbfd3bdbdc9",
        context_length=context,
        selector_mode=mode,
        node_budget=budget,
        repeat_index=repeat,
        max_new_tokens=32,
    )


def _pair(left: str, right: str, left_ids: list[int], right_ids: list[int]) -> dict:
    offset = next(
        (
            index
            for index, values in enumerate(zip(left_ids, right_ids))
            if values[0] != values[1]
        ),
        None,
    )
    return {
        "left_name": left,
        "right_name": right,
        "output_ids_equal": left_ids == right_ids,
        "left_output_ids": left_ids,
        "right_output_ids": right_ids,
        "left_output_sha256": _hash(left_ids),
        "right_output_sha256": _hash(right_ids),
        "first_divergent_generated_token_offset": offset,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_subrun(
    root: Path,
    identity: dict,
    *,
    actual_equal: bool = True,
    reference_stable: bool = True,
    structural_failure: bool = False,
    replay_mismatch: bool = False,
    return_code: int = 0,
    tree_tail: int = 50,
    source_matches: bool = True,
    downstream_gold_skip_reason: str | None = None,
    failure_class_override: str | None = None,
    hard_failure_reason_override: str | None = None,
) -> None:
    root.mkdir()
    reference = [10, 20, 30, 40]
    tree = reference if actual_equal else [10, 20, 30, tree_tail]
    target = list(reference)
    manifest_status = "ok" if return_code == 0 else "failed"
    reported_failure_class = (
        failure_class_override
        if failure_class_override is not None
        else "numerical_backend_decision_divergence"
        if not actual_equal
        else "numerical_backend_variation"
        if replay_mismatch
        else None
    )
    hard_failure_reason = hard_failure_reason_override or (
        f"gold_semantic_failure:{reported_failure_class}" if replay_mismatch else None
    )
    _write_json(
        root / "manifest.json",
        {
            "status": manifest_status,
            "error": (
                None
                if return_code == 0
                else {
                    "type": "Phase2CCorrectnessFailure",
                    "message": "expected diagnostic mismatch",
                }
            ),
            "configuration": {"frozen_selector_config_hash": "frozen"},
            "attention_backend": {
                "target": {"calls_pytorch_sdpa": True},
                "draft": {"calls_pytorch_sdpa": True},
            },
        },
    )
    _write_json(
        root / "correctness-summary.json",
        {
            "gold_status": "fail_semantic" if replay_mismatch else "pass_exact",
            "failure_class": reported_failure_class,
            "uncompleted_checks": (
                ["kv_compaction", "kv_next_step_logits"]
                if downstream_gold_skip_reason is not None
                else []
            ),
            "uncompleted_check_details": (
                [
                    {
                        "check": check,
                        "reason": downstream_gold_skip_reason,
                    }
                    for check in ("kv_compaction", "kv_next_step_logits")
                ]
                if downstream_gold_skip_reason is not None
                else []
            ),
            "actual_execution_correctness_passed": actual_equal
            and not structural_failure,
            "actual_execution_semantic_failure": not actual_equal,
            "diagnostic_numerical_variation_observed": replay_mismatch,
            "diagnostic_numerical_variation_reasons": (
                [hard_failure_reason] if hard_failure_reason is not None else []
            ),
            "replay_semantic_variation_with_actual_consequence": not actual_equal,
            "hard_correctness_failure_reasons": (
                [hard_failure_reason] if hard_failure_reason is not None else []
            ),
        },
    )
    pairwise = [
        _pair("target_only", "original_dflash_reference", target, reference),
        _pair("target_only", "layerwise_tree_control", target, tree),
        _pair("original_dflash_reference", "layerwise_tree_control", reference, tree),
    ]
    _write_json(
        root / "output-divergence.json",
        {
            "records": [
                {
                    "pairwise_comparisons": pairwise,
                    "reference_repeat_output_ids_equal": reference_stable,
                    "reference_profile_output_ids_equal": reference_stable,
                    "first_divergent_generated_token_offset": None
                    if actual_equal
                    else 3,
                    "first_divergent_absolute_sequence_position": None
                    if actual_equal
                    else 1027,
                    "divergent_tree_round_index": None if actual_equal else 1,
                    "divergent_token_kind": None if actual_equal else "accepted_draft",
                    "reference_token_id": None if actual_equal else 40,
                    "reference_token_text": None if actual_equal else " ref",
                    "tree_token_id": None if actual_equal else tree_tail,
                    "tree_token_text": None if actual_equal else " tree",
                    "tree_traversal_status": "available",
                    "classification": {
                        "failure_class": None
                        if actual_equal
                        else reported_failure_class,
                        "evidence_complete": not bool(
                            reported_failure_class
                            and reported_failure_class.startswith("unresolved_")
                        ),
                    },
                    "uncompleted_checks": [],
                }
            ]
        },
    )
    selector = (
        None
        if identity["selector_mode"] == "top1_packed_tree"
        else {
            "selected_policy_config": f"layerwise_beam-b{identity['node_budget']}-w4",
            "decision_made_before_target_verification": True,
            "target_dependent_runtime_input_used": False,
            "target_forward_count_added": 0,
        }
    )
    rounds = [
        {
            "profiling_repeat": 0,
            "round_index": index,
            "generated_root_offset": index * 2,
            "actual_node_count": identity["node_budget"],
            "configuration_id": "top1-chain" if selector is None else "global-fixed",
            "online_selector": selector,
            "accepted_draft_token_count": 1,
            "committed_token_count": 2,
            "terminal_round": index == 2,
            "truncated_token_count": 0,
        }
        for index in range(3)
    ]
    _write_jsonl(root / "tree-rounds.jsonl", rounds)
    _write_jsonl(
        root / "decision-stabilization-rounds.jsonl",
        [
            {
                "profiling_repeat": 0,
                "round_index": index,
                "minimum_path_margin": (0.2, 0.0, 0.5)[index],
                "verifier_margins": [(0.2, 0.0, 0.5)[index]],
                "first_corrected_native_margin": 0.0 if index == 1 else None,
                "native_accepted_depth": 4,
                "canonical_accepted_depth": 2,
                "native_canonical_decision_equal": index != 1,
            }
            for index in range(3)
        ],
    )
    traversal_status = "fail" if structural_failure else "pass"
    _write_jsonl(
        root / "tree-traversal-checks.jsonl",
        [
            {
                "round_index": index,
                "status": traversal_status if index == 1 else "pass",
                "structural_mask_issues": ["mask"]
                if structural_failure and index == 1
                else [],
                "checks": {
                    "structural_checks_passed": not (structural_failure and index == 1),
                    "traversal_checks_passed": True,
                    "kv_root_transition_checks_passed": True,
                    "all_checks_passed": not (structural_failure and index == 1),
                },
            }
            for index in range(3)
        ],
    )
    gold = []
    for mode in ("native", "same_mask", "math"):
        gold.append(
            {
                "round_index": 1,
                "comparison_mode": mode,
                "availability": "available",
                "status": "fail_semantic"
                if replay_mismatch and mode == "native"
                else "pass_exact",
                "structural_status": "pass",
                "semantic_status": "fail"
                if replay_mismatch and mode == "native"
                else "pass",
                "numerical_status": "outside_strict_thresholds"
                if replay_mismatch and mode == "native"
                else "within_strict_thresholds",
                "verifier_decision_changed": replay_mismatch and mode == "native",
                "argmax_equal": not (replay_mismatch and mode == "native"),
                "top8_ordered_equal": not (replay_mismatch and mode == "native"),
                "max_abs_error": 1.0 if replay_mismatch else 0.0,
                "mean_abs_error": 0.1 if replay_mismatch else 0.0,
                "thresholds": {"max_abs_error": 0.5},
                "backend": "configured_sdpa",
                "dtype": "bfloat16",
                "provenance": {"execution": "diagnostic_replay"},
            }
        )
    _write_jsonl(root / "gold-logit-checks.jsonl", gold)
    _write_jsonl(
        root / "gold-kv-checks.jsonl",
        [
            {
                "round_index": 1,
                "availability": "unavailable"
                if downstream_gold_skip_reason is not None
                else "available",
                "status": "not_run"
                if downstream_gold_skip_reason is not None
                else "fail_structural"
                if structural_failure
                else "pass_exact",
                "execution_status": "not_run_due_to_prior_failure"
                if downstream_gold_skip_reason is not None
                else "completed",
                "structural_status": "not_run"
                if downstream_gold_skip_reason is not None
                else "fail"
                if structural_failure
                else "pass",
                "semantic_status": "not_run"
                if downstream_gold_skip_reason is not None
                else "pass",
                "numerical_status": "not_run"
                if downstream_gold_skip_reason is not None
                else "within_strict_thresholds",
                "reason": downstream_gold_skip_reason,
            }
        ],
    )
    _write_jsonl(
        root / "samples.jsonl",
        [
            {
                "task": identity["task"],
                "sample_id": identity["sample_id"],
                "nominal_prompt_length": identity["nominal_prompt_length"],
                "node_budget": identity["node_budget"],
                "configuration_id": "synthetic",
            }
        ],
    )
    module_hash = "a" * 64
    dependency_hash = "b" * 64
    _write_json(
        root / "source-provenance.json",
        {
            "status": manifest_status,
            "error": (
                None
                if return_code == 0
                else {
                    "type": "Phase2CCorrectnessFailure",
                    "message": "expected diagnostic mismatch",
                }
            ),
            "repository_custom_code_commit": "commit",
            "base_model_id": "z-lab/Qwen3-8B-DFlash-b16",
            "base_model_revision": "revision",
            "materialization_id": "materialization",
            "actual_loaded_module_path": "/draft/dflash.py",
            "actual_loaded_module_sha256": module_hash,
            "custom_code_sha256": {"dflash.py": module_hash},
            "loaded_source_matches_repository": source_matches,
            "spec_generate": {"file": "/draft/dflash.py"},
            "actual_loaded_dependencies": {
                "utils.py": {
                    "matches_repository": source_matches,
                    "actual_loaded_sha256": dependency_hash,
                    "repository_sha256": dependency_hash
                    if source_matches
                    else "c" * 64,
                }
            },
        },
    )


class Phase2D1BNRTests(unittest.TestCase):
    def test_cli_defaults_and_overrides(self) -> None:
        args = parse_args(
            [
                "--target-model-path",
                "/target",
                "--materialized-draft-path",
                "/draft",
                "--longbench-root",
                "/data",
                "--frozen-selector-config",
                "/frozen.json",
                "--capture-dir",
                "/capture",
            ]
        )
        self.assertEqual(args.exact_case_repeats, 10)
        self.assertEqual(args.cohort_repeats, 1)
        self.assertEqual(args.cohort_context_lengths[-1], 32768)
        self.assertEqual(args.margin_thresholds[0], 0.0)
        overridden = parse_args(
            [
                "--target-model-path",
                "/target",
                "--materialized-draft-path",
                "/draft",
                "--longbench-root",
                "/data",
                "--frozen-selector-config",
                "/frozen.json",
                "--capture-dir",
                "/capture",
                "--exact-case-repeats",
                "3",
                "--cohort-repeats",
                "3",
                "--no-run-cohort",
            ]
        )
        self.assertEqual(overridden.exact_case_repeats, 3)
        self.assertFalse(overridden.run_cohort)

    def test_matched_cohort_and_count_semantics(self) -> None:
        rows = []
        for source in ("a", "b", "c"):
            for length in (1024, 4096):
                rows.append(
                    {
                        "task": "gov_report",
                        "sample_id": source,
                        "source_sample_id": source,
                        "nominal_prompt_length": length,
                    }
                )
        rows.append(
            {
                "task": "gov_report",
                "sample_id": "short",
                "source_sample_id": "short",
                "nominal_prompt_length": 1024,
            }
        )
        cohort = derive_matched_capture_cohort(
            rows, task="gov_report", context_lengths=(1024, 4096)
        )
        self.assertEqual([row["sample_id"] for row in cohort], ["a", "b", "c"])
        identities = [
            _identity(experiment="cohort", mode=mode, budget=budget, repeat=repeat)
            for mode, budget in (("top1_packed_tree", 16), ("global_fixed", 16))
            for repeat in range(3)
        ]
        self.assertEqual(len({row["physical_case_id"] for row in identities}), 1)
        self.assertEqual(len({row["configuration_case_id"] for row in identities}), 2)
        self.assertEqual(
            len({row["independent_repeat_observation_id"] for row in identities}), 6
        )

    def test_numerical_divergence_nonzero_is_retained(self) -> None:
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "subrun"
            _write_subrun(
                root,
                identity,
                actual_equal=False,
                replay_mismatch=True,
                return_code=1,
            )
            result = classify_subrun_artifacts(root, identity=identity, return_code=1)
        _validate(result, "phase2d1bnr-subrun.schema.json")
        self.assertEqual(
            result["outcome_category"],
            "actual_numerical_decision_divergence",
        )
        self.assertTrue(result["subprocess_nonzero_retained_as_diagnostic"])
        self.assertFalse(result["actual_reference_tree_output_ids_equal"])
        self.assertEqual(result["source_provenance_status"], "failed")
        self.assertTrue(result["source_provenance_validation_passed"])
        self.assertTrue(
            result["source_provenance_wrapper_failure_is_expected_correctness"]
        )
        event = build_divergence_event(result)
        self.assertEqual(event["minimum_verifier_margin"], 0.0)

    def test_actual_unresolved_semantic_divergence_is_retained(self) -> None:
        identity = _identity(experiment="cohort", mode="global_fixed", budget=32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "subrun"
            _write_subrun(
                root,
                identity,
                actual_equal=False,
                replay_mismatch=True,
                return_code=1,
                failure_class_override="unresolved_layerwise_output_divergence",
                hard_failure_reason_override=(
                    "gold_semantic_failure:gold_output_ids_mismatch"
                ),
            )
            result = classify_subrun_artifacts(root, identity=identity, return_code=1)
        _validate(result, "phase2d1bnr-subrun.schema.json")
        self.assertEqual(
            result["outcome_category"], "actual_unresolved_semantic_divergence"
        )
        self.assertFalse(result["actual_reference_tree_output_ids_equal"])
        self.assertTrue(result["actual_divergence_evidence_sufficient"])
        self.assertEqual(
            result["divergence_failure_class"],
            "unresolved_layerwise_output_divergence",
        )
        self.assertEqual(
            result["unresolved_attribution_reason"],
            "unresolved_layerwise_output_divergence",
        )
        self.assertEqual(result["gold_status"], "fail_semantic")
        self.assertEqual(
            result["hard_correctness_failure_reasons"],
            ["gold_semantic_failure:gold_output_ids_mismatch"],
        )
        self.assertFalse(result["genuinely_incomplete_correctness_evidence"])
        self.assertTrue(result["subprocess_nonzero_retained_as_diagnostic"])
        event = build_divergence_event(result)
        self.assertEqual(event["first_divergent_generated_offset"], 3)

    def test_correctness_exit_does_not_hide_source_provenance_mismatch(self) -> None:
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "subrun"
            _write_subrun(
                root,
                identity,
                actual_equal=False,
                replay_mismatch=True,
                return_code=1,
                source_matches=False,
            )
            result = classify_subrun_artifacts(root, identity=identity, return_code=1)
        self.assertEqual(result["outcome_category"], "infrastructure_failure")
        self.assertFalse(result["source_provenance_validation_passed"])

    def test_confirmed_divergence_retains_causally_skipped_downstream_gold(
        self,
    ) -> None:
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "subrun"
            _write_subrun(
                root,
                identity,
                actual_equal=False,
                replay_mismatch=True,
                return_code=1,
                downstream_gold_skip_reason="prior_logits_gold_failure",
            )
            result = classify_subrun_artifacts(root, identity=identity, return_code=1)
        _validate(result, "phase2d1bnr-subrun.schema.json")
        self.assertEqual(
            result["outcome_category"],
            "actual_numerical_decision_divergence",
        )
        self.assertTrue(result["gold_evidence_complete"])
        self.assertTrue(result["subprocess_nonzero_retained_as_diagnostic"])
        self.assertTrue(
            result["downstream_gold_checks_skipped_after_confirmed_divergence"]
        )
        self.assertEqual(
            result["downstream_gold_skipped_checks"],
            ["kv_compaction", "kv_next_step_logits"],
        )
        self.assertEqual(
            result["downstream_gold_skip_reasons"],
            ["prior_logits_gold_failure"],
        )
        self.assertFalse(result["genuinely_incomplete_correctness_evidence"])

    def test_unknown_gold_skip_keeps_confirmed_divergence_unresolved(self) -> None:
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "subrun"
            _write_subrun(
                root,
                identity,
                actual_equal=False,
                replay_mismatch=True,
                return_code=1,
                downstream_gold_skip_reason="unknown_failure",
            )
            result = classify_subrun_artifacts(root, identity=identity, return_code=1)
        self.assertEqual(
            result["outcome_category"], "actual_unresolved_semantic_divergence"
        )
        self.assertFalse(
            result["downstream_gold_checks_skipped_after_confirmed_divergence"]
        )
        self.assertFalse(result["gold_evidence_complete"])
        self.assertFalse(result["genuinely_incomplete_correctness_evidence"])
        self.assertEqual(
            result["unresolved_attribution_reason"],
            "causal_attribution_incomplete",
        )

    def test_output_exact_replay_only_variation_continues(self) -> None:
        identity = _identity(mode="top1_packed_tree")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "subrun"
            _write_subrun(root, identity, replay_mismatch=True)
            result = classify_subrun_artifacts(root, identity=identity, return_code=0)
        self.assertEqual(
            result["outcome_category"], "diagnostic_only_numerical_variation"
        )
        self.assertFalse(result["actual_execution_semantic_failure"])
        self.assertTrue(result["replay_only_variation_observed"])
        self.assertTrue(result["diagnostic_classification_evidence_sufficient"])
        self.assertTrue(result["diagnostic_numerical_variation_observed"])
        self.assertTrue(result["replay_semantic_mismatch_observed"])
        self.assertEqual(
            result["_replay_rows"][0]["thresholds"], {"max_abs_error": 0.5}
        )
        self.assertEqual(result["_replay_rows"][0]["backend"], "configured_sdpa")
        self.assertEqual(
            result["diagnostic_numerical_variation_reasons"],
            ["gold_semantic_failure:numerical_backend_variation"],
        )

    def test_nonzero_replay_only_exit_is_retained_and_continues(self) -> None:
        identity = _identity(mode="top1_packed_tree")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "subrun"
            _write_subrun(
                root,
                identity,
                replay_mismatch=True,
                return_code=1,
                downstream_gold_skip_reason="prior_logits_gold_failure",
            )
            result = classify_subrun_artifacts(root, identity=identity, return_code=1)
        self.assertEqual(
            result["outcome_category"], "diagnostic_only_numerical_variation"
        )
        self.assertTrue(result["subprocess_nonzero_retained_as_diagnostic"])
        self.assertTrue(
            result["downstream_gold_checks_skipped_after_confirmed_replay_variation"]
        )

    def test_real_shape_exact_output_gold_semantic_variation_continues(self) -> None:
        identity = _identity(experiment="cohort", mode="global_fixed", budget=32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "subrun"
            _write_subrun(
                root,
                identity,
                replay_mismatch=True,
                return_code=1,
                downstream_gold_skip_reason="prior_logits_gold_failure",
            )
            gold_logits = [
                json.loads(line)
                for line in (root / "gold-logit-checks.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if json.loads(line)["comparison_mode"] == "native"
            ]
            _write_jsonl(root / "gold-logit-checks.jsonl", gold_logits)
            _write_jsonl(root / "tree-traversal-checks.jsonl", [])
            result = classify_subrun_artifacts(root, identity=identity, return_code=1)
        _validate(result, "phase2d1bnr-subrun.schema.json")
        self.assertEqual(
            result["outcome_category"], "diagnostic_only_numerical_variation"
        )
        self.assertTrue(result["actual_reference_tree_output_ids_equal"])
        self.assertTrue(result["replay_only_variation_observed"])
        self.assertTrue(result["replay_structural_evidence_passed"])
        self.assertTrue(result["diagnostic_classification_evidence_sufficient"])
        self.assertFalse(result["gold_evidence_complete"])
        self.assertFalse(result["genuinely_incomplete_correctness_evidence"])
        self.assertTrue(result["subprocess_nonzero_retained_as_diagnostic"])

    def test_exact_output_unresolved_numerical_variation_continues(self) -> None:
        identity = _identity(
            experiment="cohort", mode="top1_packed_tree", budget=16, context=16384
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "subrun"
            _write_subrun(
                root,
                identity,
                replay_mismatch=True,
                return_code=1,
                failure_class_override="unresolved_numerical_variation",
                hard_failure_reason_override=(
                    "gold_semantic_failure:unresolved_numerical_variation"
                ),
            )
            _write_jsonl(root / "gold-logit-checks.jsonl", [])
            result = classify_subrun_artifacts(root, identity=identity, return_code=1)
        _validate(result, "phase2d1bnr-subrun.schema.json")
        self.assertEqual(
            result["outcome_category"], "diagnostic_only_numerical_variation"
        )
        self.assertTrue(result["actual_reference_tree_output_ids_equal"])
        self.assertTrue(result["gold_or_replay_variation_observed"])
        self.assertTrue(result["known_numerical_variation_evidence"])
        self.assertTrue(result["replay_only_variation_observed"])
        self.assertTrue(result["replay_structural_evidence_passed"])
        self.assertTrue(result["diagnostic_classification_evidence_sufficient"])
        self.assertFalse(result["gold_evidence_complete"])
        self.assertFalse(result["genuinely_incomplete_correctness_evidence"])
        self.assertEqual(
            result["hard_correctness_failure_reasons"],
            ["gold_semantic_failure:unresolved_numerical_variation"],
        )

    def test_exact_output_with_missing_replay_evidence_remains_incomplete(self) -> None:
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "subrun"
            _write_subrun(root, identity, replay_mismatch=True, return_code=1)
            rows = [
                json.loads(line)
                for line in (root / "gold-logit-checks.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if json.loads(line)["comparison_mode"] != "native"
            ]
            _write_jsonl(root / "gold-logit-checks.jsonl", rows)
            correctness = json.loads(
                (root / "correctness-summary.json").read_text(encoding="utf-8")
            )
            correctness.update(
                {
                    "gold_status": "not_run",
                    "failure_class": None,
                    "diagnostic_numerical_variation_observed": False,
                    "diagnostic_numerical_variation_reasons": [],
                    "hard_correctness_failure_reasons": [],
                }
            )
            _write_json(root / "correctness-summary.json", correctness)
            result = classify_subrun_artifacts(root, identity=identity, return_code=1)
        self.assertEqual(result["outcome_category"], "incomplete_correctness_evidence")

    def test_exact_output_with_unknown_failure_reason_remains_incomplete(self) -> None:
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "subrun"
            _write_subrun(
                root,
                identity,
                replay_mismatch=True,
                return_code=1,
                failure_class_override="unknown_gold_failure",
            )
            rows = [
                json.loads(line)
                for line in (root / "gold-logit-checks.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            for row in rows:
                row["numerical_status"] = "within_strict_thresholds"
                row["top8_ordered_equal"] = True
            _write_jsonl(root / "gold-logit-checks.jsonl", rows)
            result = classify_subrun_artifacts(root, identity=identity, return_code=1)
        self.assertEqual(result["outcome_category"], "incomplete_correctness_evidence")

    def test_missing_actual_output_relation_remains_incomplete(self) -> None:
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "subrun"
            _write_subrun(root, identity, replay_mismatch=True, return_code=1)
            artifact = json.loads(
                (root / "output-divergence.json").read_text(encoding="utf-8")
            )
            record = artifact["records"][0]
            pair = next(
                row
                for row in record["pairwise_comparisons"]
                if row["left_name"] == "original_dflash_reference"
            )
            pair.pop("output_ids_equal")
            _write_json(root / "output-divergence.json", artifact)
            result = classify_subrun_artifacts(root, identity=identity, return_code=1)
        self.assertEqual(result["outcome_category"], "incomplete_correctness_evidence")
        self.assertFalse(result["actual_output_relation_reliable"])

    def test_confirmed_divergence_with_incomplete_attribution_is_unresolved(
        self,
    ) -> None:
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "subrun"
            _write_subrun(
                root,
                identity,
                actual_equal=False,
                replay_mismatch=True,
                return_code=1,
            )
            rows = [
                json.loads(line)
                for line in (root / "gold-logit-checks.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if json.loads(line)["comparison_mode"] != "math"
            ]
            _write_jsonl(root / "gold-logit-checks.jsonl", rows)
            result = classify_subrun_artifacts(root, identity=identity, return_code=1)
        self.assertEqual(
            result["outcome_category"], "actual_unresolved_semantic_divergence"
        )
        self.assertFalse(result["gold_evidence_complete"])
        self.assertFalse(result["genuinely_incomplete_correctness_evidence"])
        self.assertEqual(
            result["unresolved_attribution_reason"],
            "causal_attribution_incomplete",
        )

    def test_structural_and_infrastructure_failures_are_distinct(self) -> None:
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "structural"
            _write_subrun(root, identity, structural_failure=True, return_code=1)
            structural = classify_subrun_artifacts(
                root, identity=identity, return_code=1
            )
            self.assertEqual(
                structural["outcome_category"], "actual_structural_failure"
            )
            unexpected = Path(temporary) / "unexpected"
            _write_subrun(
                unexpected,
                identity,
                actual_equal=False,
                replay_mismatch=True,
                return_code=1,
            )
            manifest = json.loads(
                (unexpected / "manifest.json").read_text(encoding="utf-8")
            )
            manifest["error"]["type"] = "RuntimeError"
            _write_json(unexpected / "manifest.json", manifest)
            infrastructure = classify_subrun_artifacts(
                unexpected, identity=identity, return_code=1
            )
            self.assertEqual(
                infrastructure["outcome_category"], "infrastructure_failure"
            )
            with self.assertRaises(ArtifactError):
                classify_subrun_artifacts(
                    Path(temporary) / "missing",
                    identity=identity,
                    return_code=1,
                )

    def test_reference_instability(self) -> None:
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "subrun"
            _write_subrun(root, identity, reference_stable=False, return_code=1)
            result = classify_subrun_artifacts(root, identity=identity, return_code=1)
        self.assertEqual(result["outcome_category"], "reference_instability")

    def test_reproducibility_classes_stable_stochastic_and_reference(self) -> None:
        stable = []
        for repeat in range(3):
            row = _identity(repeat=repeat)
            stable.append(
                {
                    **row,
                    "outcome_category": "actual_numerical_decision_divergence",
                    "actual_reference_output_sha256": "a" * 64,
                    "actual_tree_output_sha256": "b" * 64,
                    "actual_reference_tree_output_ids_equal": False,
                    "reference_repeat_stable": True,
                    "reference_profile_stable": True,
                    "first_divergent_generated_offset": 11,
                    "first_divergent_tree_round": 2,
                    "reference_token_id": 43282,
                    "tree_token_id": 26722,
                }
            )
        summary = aggregate_exact_reproducibility(stable)
        self.assertEqual(
            summary["configuration_strata"][0]["reproducibility_classification"],
            "stable_actual_divergence",
        )
        stable[-1] = {
            **stable[-1],
            "outcome_category": "pass_actual_exact",
            "actual_reference_tree_output_ids_equal": True,
            "actual_tree_output_sha256": "a" * 64,
        }
        summary = aggregate_exact_reproducibility(stable)
        self.assertEqual(
            summary["configuration_strata"][0]["reproducibility_classification"],
            "stochastic_actual_divergence",
        )
        stable[0]["outcome_category"] = "reference_instability"
        summary = aggregate_exact_reproducibility(stable)
        self.assertEqual(
            summary["configuration_strata"][0]["reproducibility_classification"],
            "reference_instability",
        )

    def test_margin_zero_sweep_and_post_divergence_exclusion(self) -> None:
        identity = _identity(experiment="cohort")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "subrun"
            _write_subrun(
                root,
                identity,
                actual_equal=False,
                replay_mismatch=True,
                return_code=1,
            )
            observation = classify_subrun_artifacts(
                root, identity=identity, return_code=1
            )
            events = extract_round_margin_events(observation)
        self.assertEqual([row["round_index"] for row in events], [0, 1])
        self.assertTrue(events[-1]["first_actual_divergence_round"])
        self.assertEqual(events[-1]["minimum_verifier_margin"], 0.0)
        sweep = build_margin_trigger_sweep(events, thresholds=(0.0, 0.1))
        overall = [row for row in sweep if row["scope"] == "overall"]
        self.assertEqual(overall[0]["divergence_recall"], 1.0)
        self.assertEqual(overall[0]["trigger_count"], 1)

    def test_decision_rules_and_aggregation(self) -> None:
        observations = []
        for index in range(100):
            identity = _identity(experiment="cohort", repeat=0)
            identity.update(
                {
                    "sample_id": f"sample-{index}",
                    "source_sample_id": f"source-{index}",
                    "physical_case_id": f"gov_report::source-{index}::1024",
                    "configuration_case_id": (
                        f"gov_report::source-{index}::1024::global_fixed::n16"
                    ),
                    "independent_repeat_observation_id": (
                        "cohort::gov_report::"
                        f"source-{index}::1024::global_fixed::n16::repeat-00"
                    ),
                }
            )
            observations.append(
                {
                    **identity,
                    "actual_reference_tree_output_ids_equal": index != 0,
                    "outcome_category": "actual_numerical_decision_divergence"
                    if index == 0
                    else "pass_actual_exact",
                }
            )
        cohort = aggregate_cohort_cases(observations)
        sweep = [
            {
                "experiment": "cohort",
                "selector_mode": "global_fixed",
                "node_budget": 16,
                "scope": "overall",
                "threshold": 0.05,
                "trigger_rate": 0.01,
                "divergence_recall": 1.0,
            }
        ]
        decision = choose_nr_decision(
            observations, cohort["policy_divergence_summary"], sweep
        )
        self.assertEqual(
            decision["decision"], "characterize_selective_stabilization_cost"
        )
        structural = [
            {
                **_identity(mode="top1_packed_tree"),
                "outcome_category": "actual_structural_failure",
            }
        ]
        decision = choose_nr_decision(structural, [], [])
        self.assertEqual(decision["decision"], "investigate_structural_correctness")
        infrastructure = [
            {
                **_identity(),
                "outcome_category": "infrastructure_failure",
            }
        ]
        decision = choose_nr_decision(infrastructure, [], [])
        self.assertEqual(decision["decision"], "collect_more_robustness_evidence")

    def test_generated_artifact_instances_validate(self) -> None:
        identity = _identity(experiment="cohort")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "subrun"
            _write_subrun(
                root,
                identity,
                actual_equal=False,
                replay_mismatch=True,
                return_code=1,
            )
            observation = classify_subrun_artifacts(
                root, identity=identity, return_code=1
            )
            public = public_observation(observation)
            divergence = build_divergence_event(observation)
            margins = extract_round_margin_events(observation)
        exact_identity = _identity()
        exact = {
            **public,
            **exact_identity,
        }
        exact_summary = aggregate_exact_reproducibility([exact])
        cohort_summary = aggregate_cohort_cases([public])
        sweep_rows = build_margin_trigger_sweep(margins, thresholds=(0.0, 0.1))
        sweep = {"schema_version": "1.0", "thresholds": [0.0, 0.1], "rows": sweep_rows}
        decision = choose_nr_decision(
            [public], cohort_summary["policy_divergence_summary"], sweep_rows
        )
        research = build_research_summary(
            exact_summary, cohort_summary, sweep_rows, decision
        )
        _validate(public, "phase2d1bnr-subrun.schema.json")
        _validate(divergence, "phase2d1bnr-divergence-event.schema.json")
        for row in observation["_replay_rows"]:
            _validate(row, "phase2d1bnr-replay-event.schema.json")
        for row in margins:
            _validate(row, "phase2d1bnr-round-margin.schema.json")
        _validate(exact_summary, "phase2d1bnr-exact-summary.schema.json")
        _validate(sweep, "phase2d1bnr-margin-sweep.schema.json")
        _validate(decision, "phase2d1bnr-decision.schema.json")
        _validate(research, "phase2d1bnr-research-summary.schema.json")

    def test_schemas_and_framework_lazy_import(self) -> None:
        for path in sorted(
            (REPO_ROOT / "docs" / "schemas").glob("phase2d1bnr-*.schema.json")
        ):
            Draft202012Validator.check_schema(json.loads(path.read_text()))
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import dflash.phase2d1bnr; "
                "import experiments.phase2d1b_nr_numerical_robustness; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules"
            ),
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    def test_acceptance_checker_on_complete_exact_only_study(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "nr"
            raw = root / "raw-subruns"
            raw.mkdir(parents=True)
            observations = []
            rounds = []
            divergence_events = []
            replay_events = []
            for mode in ("top1_packed_tree", "global_fixed"):
                identity = _identity(mode=mode)
                subrun = raw / mode
                divergent = mode == "global_fixed"
                replay_only = mode == "top1_packed_tree"
                _write_subrun(
                    subrun,
                    identity,
                    actual_equal=not divergent,
                    replay_mismatch=divergent or replay_only,
                    return_code=1,
                    downstream_gold_skip_reason=(
                        "prior_logits_gold_failure"
                        if divergent or replay_only
                        else None
                    ),
                    failure_class_override=(
                        "unresolved_layerwise_output_divergence" if divergent else None
                    ),
                    hard_failure_reason_override=(
                        "gold_semantic_failure:gold_output_ids_mismatch"
                        if divergent
                        else None
                    ),
                )
                if replay_only:
                    gold_logits = [
                        json.loads(line)
                        for line in (subrun / "gold-logit-checks.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()
                        if json.loads(line)["comparison_mode"] == "native"
                    ]
                    _write_jsonl(subrun / "gold-logit-checks.jsonl", gold_logits)
                    _write_jsonl(subrun / "tree-traversal-checks.jsonl", [])
                observation = classify_subrun_artifacts(
                    subrun,
                    identity=identity,
                    return_code=1,
                )
                observations.append(observation)
                rounds.extend(extract_round_margin_events(observation))
                event = build_divergence_event(observation)
                if event is not None:
                    divergence_events.append(event)
                replay_events.extend(observation["_replay_rows"])
            manifest = {
                "schema_version": "1.0",
                "phase": "2D.1B-NR",
                "status": "ok",
                "baseline_repo_sha": "d60e7db0514053ac580f2bfa9ca8cf7a534c1b49",
                "repository": {
                    "current_sha": "a" * 40,
                    "branch": "codex/phase2d1b-nr-numerical-robustness",
                    "dirty": False,
                    "protected_online_paths_changed": [],
                },
                "target_model": {"path": "/target", "revision": "target"},
                "materialized_draft": {
                    "path": "/draft",
                    "materialization_id": "materialized",
                    "materialization_manifest_sha256": "b" * 64,
                    "manifest": {},
                },
                "frozen_selector": {
                    "path": "/frozen",
                    "config_sha256": "frozen",
                    "file_sha256": "c" * 64,
                    "training_source_ids": [],
                },
                "capture_inputs": [{}],
                "longbench": {
                    "root": "/data",
                    "task": "gov_report",
                    "jsonl_path": "/data/gov_report.jsonl",
                    "jsonl_sha256": "d" * 64,
                    "matched_capture_cohort": [
                        {
                            "sample_id": "gov_report-fd287bbfd3bdbdc9",
                            "source_sample_id": "gov_report-fd287bbfd3bdbdc9",
                        }
                    ],
                },
                "environment": {},
                "configuration": {
                    "modes": ["top1_packed_tree", "global_fixed"],
                    "node_budgets": [16, 32],
                },
                "count_semantics": {
                    "physical_case": "task x source x context",
                    "configuration_case": "physical x config",
                    "independent_repeat_observation": "config x repeat",
                    "repeats_are_not_independent_source_evidence": True,
                    "requested_subrun_count": 2,
                },
                "decision_rules": {},
                "subruns": [{}, {}],
                "artifacts": list(ARTIFACTS),
                "completed_subrun_count": 2,
                "online_execution_modified": False,
                "selector_modified": False,
                "tree_modified": False,
                "kv_compaction_modified": False,
                "correctness_gate_modified": False,
                "numerical_thresholds_modified": False,
                "diagnostic_gold_numerical_summary_modified": True,
                "materialized_diagnostic_custom_code_modified": True,
                "online_execution_source_otherwise_unchanged": True,
                "materialization_required": True,
                "formal_timing_claims_made": False,
                "no_performance_claims": True,
                "error": None,
            }
            _persist_results(
                root,
                manifest,
                observations,
                divergence_events,
                replay_events,
                rounds,
                (0.0, 0.1),
            )
            (root / "terminal.log").write_text("synthetic\n", encoding="utf-8")
            result = check_output(root)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["subrun_count"], 2)

    def test_public_observation_strips_joined_raw_rows(self) -> None:
        value = {
            **_identity(),
            "_rounds": [{}],
            "outcome_category": "pass_actual_exact",
        }
        self.assertNotIn("_rounds", public_observation(value))


if __name__ == "__main__":
    unittest.main()
