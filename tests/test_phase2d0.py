from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from dflash.phase2d0.aggregation import aggregate_gap_events
from dflash.phase2d0.analysis import (
    GAP_CLASS_PRIORITY,
    analyze_captures,
    classify_gap,
)
from dflash.phase2d0.signals import (
    TARGET_ONLY_FIELDS,
    build_signal_rows,
    evaluate_signal_hypotheses,
)
from scripts.check_phase2d0_acceptance import check_output
from scripts.phase2d0_dependency_opportunity import (
    OUTPUT_FILENAMES,
    main,
    parse_args,
)
from tests.test_phase2b1 import _write_fixture

REPO_ROOT = Path(__file__).resolve().parents[1]


def _positions() -> list[dict]:
    return [
        {
            "block_position": position,
            "baseline_token_id": position * 100 + 1,
            "candidates": [
                {
                    "rank": rank,
                    "token_id": position * 100 + rank,
                    "log_probability": -0.1 * rank,
                }
                for rank in range(1, 17)
            ],
            "top1_top2_log_probability_margin": 0.1,
        }
        for position in range(1, 16)
    ]


def _capture(
    reference_ranks: list[int | None],
    *,
    accepted: int = 0,
    terminal: bool = False,
    sample_id: str = "sample-a",
    context_length: int = 1024,
    round_index: int = 0,
) -> dict:
    positions = _positions()
    reference = [
        positions[index]["candidates"][rank - 1]["token_id"]
        if rank is not None
        else 900_000 + index
        for index, rank in enumerate(reference_ranks)
    ]
    return {
        "task": "gov_report",
        "sample_id": sample_id,
        "source_sample_id": sample_id,
        "source_text_sha256": "a" * 64,
        "nominal_prompt_length": context_length,
        "actual_prompt_length": context_length - 4,
        "generation_repeat": 0,
        "round_index": round_index,
        "terminal_round": terminal,
        "reference_right_censored": len(reference) < 15,
        "stop_reason": "max_new_tokens" if terminal else "continue",
        "accepted_draft_token_count": accepted,
        "positions": positions,
        "reference_target_continuation_token_ids": reference,
        "_capture_directory": "/immutable/capture",
    }


def _classify(**overrides: bool) -> str:
    values = {
        "terminal_or_censored": False,
        "no_failure_within_horizon": False,
        "critical_token_available": True,
        "best_static_extends": False,
        "modest_budget_resolves": False,
        "oracle_extends": True,
        "significant_rank_or_budget_inflation": False,
    }
    values.update(overrides)
    return classify_gap(**values)


class Phase2D0ClassificationTests(unittest.TestCase):
    def test_candidate_unavailable_is_availability_failure(self) -> None:
        self.assertEqual(
            _classify(critical_token_available=False),
            "candidate_availability_failure",
        )

    def test_other_static_policy_success_is_selection_failure(self) -> None:
        self.assertEqual(
            _classify(best_static_extends=True),
            "selection_failure",
        )

    def test_modest_budget_success_is_budget_failure(self) -> None:
        self.assertEqual(
            _classify(modest_budget_resolves=True),
            "budget_failure",
        )

    def test_inflated_static_failure_is_dependency_opportunity(self) -> None:
        self.assertEqual(
            _classify(significant_rank_or_budget_inflation=True),
            "dependency_realization_opportunity",
        )

    def test_incomplete_selector_exclusion_is_possible_dependency(self) -> None:
        self.assertEqual(_classify(), "possible_dependency_opportunity")

    def test_terminal_or_censored_has_highest_priority(self) -> None:
        self.assertEqual(
            _classify(
                terminal_or_censored=True,
                no_failure_within_horizon=True,
                critical_token_available=False,
                best_static_extends=True,
                modest_budget_resolves=True,
                significant_rank_or_budget_inflation=True,
            ),
            "right_censored_or_terminal",
        )

    def test_no_failure_has_second_priority(self) -> None:
        self.assertEqual(
            _classify(
                no_failure_within_horizon=True,
                critical_token_available=False,
                best_static_extends=True,
            ),
            "no_failure_within_observed_horizon",
        )

    def test_unresolved_and_priority_are_mutually_exclusive(self) -> None:
        self.assertEqual(
            _classify(oracle_extends=False),
            "unresolved",
        )
        self.assertEqual(len(GAP_CLASS_PRIORITY), len(set(GAP_CLASS_PRIORITY)))


class Phase2D0AnalysisTests(unittest.TestCase):
    def test_same_round_changes_class_with_top_m(self) -> None:
        result = analyze_captures(
            [_capture([4, *([1] * 14)])],
            top_ms=(2, 4),
            node_budgets=(16, 32),
            survival_budget_cap=32,
        )
        by_key = {
            (event["top_m"], event["node_budget"]): event for event in result["events"]
        }
        self.assertEqual(by_key[(2, 16)]["gap_class"], "candidate_availability_failure")
        self.assertNotEqual(
            by_key[(4, 16)]["gap_class"], "candidate_availability_failure"
        )

    def test_candidate_unavailability_is_not_reported_as_static_pruning(self) -> None:
        event = analyze_captures(
            [_capture([1, None, *([1] * 13)], accepted=1)],
            top_ms=(16,),
            node_budgets=(16, 32),
            survival_budget_cap=32,
        )["events"][0]
        self.assertEqual(event["gap_class"], "candidate_availability_failure")
        self.assertIsNone(event["first_pruning_depth"])
        self.assertEqual(
            event["first_pruning_depth_unavailable_reason"],
            "no_observed_candidate_available_static_pruning",
        )

    def test_same_round_changes_from_budget_to_selection_with_node_budget(
        self,
    ) -> None:
        ranks = [1] * 15
        ranks[7] = 2
        result = analyze_captures(
            [_capture(ranks, accepted=7)],
            top_ms=(16,),
            node_budgets=(16, 32),
            survival_budget_cap=64,
        )
        by_budget = {event["node_budget"]: event for event in result["events"]}
        self.assertEqual(by_budget[16]["gap_class"], "budget_failure")
        self.assertEqual(by_budget[32]["gap_class"], "selection_failure")

    def test_availability_and_oracle_coverage_are_monotonic(self) -> None:
        ranks = [2, 4, 8, 16, *([1] * 11)]
        result = analyze_captures(
            [_capture(ranks)],
            top_ms=(2, 4, 8, 16),
            node_budgets=(16, 32),
            survival_budget_cap=32,
        )
        at_16 = [event for event in result["events"] if event["node_budget"] == 16]
        self.assertEqual(
            [event["candidate_availability_prefix"] for event in at_16],
            sorted(event["candidate_availability_prefix"] for event in at_16),
        )
        by_m = {}
        for event in result["events"]:
            by_m.setdefault(event["top_m"], []).append(event)
        for events in by_m.values():
            events.sort(key=lambda event: event["node_budget"])
            self.assertEqual(
                [event["oracle_coverage_depth"] for event in events],
                sorted(event["oracle_coverage_depth"] for event in events),
            )

    def test_minimum_budget_and_root_inclusive_semantics(self) -> None:
        ranks = [1] * 15
        ranks[7] = 2
        event = analyze_captures(
            [_capture(ranks, accepted=7)],
            top_ms=(16,),
            node_budgets=(16, 32),
            survival_budget_cap=64,
        )["events"][0]
        self.assertEqual(event["oracle_required_nodes"], 9)
        self.assertGreater(event["best_static_required_nodes"], 16)
        self.assertLessEqual(event["best_static_required_nodes"], 32)
        self.assertTrue(event["root_included_in_node_budget"])
        self.assertTrue(
            all(
                replay["verified_nodes"] <= event["node_budget"]
                for replay in event["static_policy_coverage"]
            )
        )

    def test_confirmed_and_possible_dependency_are_conservative(self) -> None:
        deep_ranks = [1, 1, 1, 1, 16, *([1] * 10)]
        confirmed = analyze_captures(
            [_capture(deep_ranks, accepted=4)],
            top_ms=(16,),
            node_budgets=(16, 32),
            survival_budget_cap=32,
        )["events"][0]
        possible = analyze_captures(
            [_capture([16, *([1] * 14)])],
            top_ms=(16,),
            node_budgets=(16, 32),
            survival_budget_cap=32,
            policies=("top1_chain", "score_beam"),
        )["events"][0]
        self.assertEqual(confirmed["gap_class"], "dependency_realization_opportunity")
        self.assertTrue(confirmed["significant_rank_inflation"])
        self.assertEqual(possible["gap_class"], "possible_dependency_opportunity")
        self.assertIsNone(possible["best_static_required_nodes"])

    def test_censored_rank_lower_bound_can_confirm_inflation(self) -> None:
        ranks = [1, 1, 1, 1, 16, *([1] * 10)]
        event = analyze_captures(
            [_capture(ranks, accepted=4)],
            top_ms=(16,),
            node_budgets=(16, 32),
            survival_budget_cap=32,
            rank_cap=31,
        )["events"][0]
        self.assertTrue(event["target_prefix_score_rank_censored"])
        self.assertEqual(event["target_prefix_score_rank_lower_bound"], 32)
        self.assertTrue(event["significant_rank_inflation"])
        self.assertEqual(event["gap_class"], "dependency_realization_opportunity")

    def test_no_failure_and_censoring_are_not_misclassified(self) -> None:
        no_failure = _capture([1] * 15, accepted=15)
        censored = _capture([1, 1], accepted=2, terminal=True)
        events = analyze_captures(
            [no_failure, censored],
            top_ms=(16,),
            node_budgets=(16, 32),
            survival_budget_cap=32,
        )["events"]
        self.assertEqual(events[0]["gap_class"], "no_failure_within_observed_horizon")
        self.assertEqual(events[2]["gap_class"], "right_censored_or_terminal")

    def test_target_labels_never_enter_deployable_features(self) -> None:
        event = analyze_captures(
            [_capture([2, *([1] * 14)])],
            top_ms=(16,),
            node_budgets=(16, 32),
            survival_budget_cap=32,
        )["events"][0]
        signal = build_signal_rows([event])[0]
        self.assertFalse(set(signal["deployable_features"]) & TARGET_ONLY_FIELDS)
        self.assertIn("target_token_rank", signal["labels"])
        self.assertIn("top_m_renormalized_entropy", signal["deployable_features"])
        self.assertIn("not full-vocabulary entropy", signal["top_m_entropy_semantics"])


class Phase2D0AggregationTests(unittest.TestCase):
    def test_micro_macro_denominators_and_matched_contexts(self) -> None:
        captures = [
            _capture([None, *([1] * 14)], sample_id="matched", context_length=1024),
            _capture(
                [2, *([1] * 14)],
                sample_id="matched",
                context_length=4096,
            ),
            _capture(
                [1, 1],
                accepted=2,
                terminal=True,
                sample_id="matched",
                context_length=4096,
                round_index=1,
            ),
        ]
        events = analyze_captures(
            captures,
            top_ms=(16,),
            node_budgets=(16, 32),
            survival_budget_cap=32,
        )["events"]
        summary = aggregate_gap_events(events)
        self.assertEqual(summary["raw_round_count"], 3)
        rows = [
            row
            for row in summary["task_context"]
            if row["context_length"] == 4096 and row["node_budget"] == 16
        ]
        self.assertEqual({row["raw_round_count"] for row in rows}, {2})
        self.assertEqual({row["eligible_round_count"] for row in rows}, {1})
        self.assertEqual({row["right_censored_or_terminal_count"] for row in rows}, {1})
        self.assertEqual(
            {row["context_length"] for row in summary["task_context"]},
            {1024, 4096},
        )
        self.assertAlmostEqual(
            sum(row["fraction_over_all_rounds"] for row in rows), 1.0
        )
        eligible_rows = [
            row for row in rows if row["gap_class"] != "right_censored_or_terminal"
        ]
        self.assertAlmostEqual(
            sum(row["fraction_among_eligible_rounds"] for row in eligible_rows),
            1.0,
        )
        censored_row = next(
            row for row in rows if row["gap_class"] == "right_censored_or_terminal"
        )
        self.assertIsNone(censored_row["fraction_among_eligible_rounds"])
        self.assertEqual(
            censored_row["fraction_among_eligible_rounds_unavailable_reason"],
            "class_excluded_from_eligible_denominator",
        )
        self.assertNotIn("fraction", censored_row)

    def test_signal_metrics_report_counts_auc_and_calibration(self) -> None:
        base_events = analyze_captures(
            [
                _capture([None, *([1] * 14)], sample_id="a"),
                _capture([2, *([1] * 14)], sample_id="b"),
            ],
            top_ms=(16,),
            node_budgets=(16, 32),
            survival_budget_cap=32,
        )["events"]
        rows = build_signal_rows(base_events * 10)
        results = evaluate_signal_hypotheses(rows)
        target = next(
            row
            for row in results
            if row["scope"] == "micro"
            and row["label"] == "candidate_availability_failure"
            and row["feature"] == "top1_top2_log_probability_margin"
        )
        self.assertEqual(target["observation_count"], 600)
        self.assertEqual(target["positive_count"], 20)
        self.assertIsNotNone(target["auroc"])
        self.assertIsNotNone(target["auprc"])
        self.assertTrue(target["calibration_or_binned_precision"])


class Phase2D0RunnerTests(unittest.TestCase):
    def test_cli_defaults_and_overrides(self) -> None:
        defaults = parse_args(["--capture-dir", "capture"])
        self.assertEqual(defaults.top_ms, (2, 4, 8, 16))
        self.assertEqual(defaults.node_budgets, (16, 32))
        self.assertEqual(defaults.survival_budget_cap, 256)
        changed = parse_args(
            [
                "--capture-dir",
                "capture",
                "--top-ms",
                "4,16",
                "--node-budgets",
                "16,32,64",
                "--survival-budget-cap",
                "128",
                "--policies",
                "top1_chain,score_beam",
            ]
        )
        self.assertEqual(changed.top_ms, (4, 16))
        self.assertEqual(changed.node_budgets, (16, 32, 64))

    def test_provenance_mismatch_fails_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "capture"
            output = root / "output"
            capture.mkdir()
            _write_fixture(capture, omit_provenance_field="actual_loaded_module_sha256")
            with self.assertRaises(ValueError):
                main(
                    [
                        "--capture-dir",
                        str(capture),
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertFalse(output.exists())

    def test_output_directory_cannot_overlap_immutable_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / "capture"
            capture.mkdir()
            with self.assertRaisesRegex(ValueError, "outside every immutable"):
                main(
                    [
                        "--capture-dir",
                        str(capture),
                        "--output-dir",
                        str(capture / "analysis"),
                    ]
                )

    def test_cpu_only_runner_writes_and_validates_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "capture"
            output = root / "output"
            capture.mkdir()
            _write_fixture(capture)
            result = main(
                [
                    "--capture-dir",
                    str(capture),
                    "--top-ms",
                    "2,16",
                    "--node-budgets",
                    "16,32",
                    "--survival-budget-cap",
                    "32",
                    "--policies",
                    "top1_chain,score_beam,layerwise_beam",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(result, 0)
            self.assertEqual(
                {path.name for path in output.iterdir()}, set(OUTPUT_FILENAMES)
            )
            manifest = json.loads((output / "analysis-manifest.json").read_text())
            self.assertEqual(
                manifest["baseline_commit_sha"],
                "1411c27e086cacd62425f329dcf7432cedd74654",
            )
            self.assertFalse(any(manifest["framework_modules_loaded"].values()))
            with (output / "context-gap-composition.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows)
            self.assertTrue(all(row["schema_version"] == "2.0" for row in rows))
            self.assertTrue(check_output(output)["passed"])

    def test_three_task_real_shape_captures_are_aggregated_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture_dirs = []
            for task in ("gov_report", "qasper", "hotpotqa"):
                capture = root / task
                capture.mkdir()
                _write_fixture(capture, task=task)
                capture_dirs.append(capture)
            output = root / "output"
            arguments = []
            for capture in capture_dirs:
                arguments.extend(("--capture-dir", str(capture)))
            result = main(
                [
                    *arguments,
                    "--top-ms",
                    "16",
                    "--node-budgets",
                    "16,32",
                    "--survival-budget-cap",
                    "32",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(result, 0)
            gap = json.loads((output / "gap-decomposition.json").read_text())
            self.assertEqual(gap["raw_round_count"], 3)
            self.assertEqual(
                {row["task"] for row in gap["task_context"]},
                {"gov_report", "qasper", "hotpotqa"},
            )
            self.assertTrue(check_output(output)["passed"])

    def test_schemas_are_valid_draft_2020_12(self) -> None:
        schemas = sorted((REPO_ROOT / "docs/schemas").glob("phase2d0-*.schema.json"))
        self.assertEqual(len(schemas), 5)
        for path in schemas:
            schema = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
            )
            Draft202012Validator.check_schema(schema)

    def test_lazy_import_does_not_load_frameworks_or_datasets(self) -> None:
        code = (
            "import sys; import dflash.phase2d0; "
            "import scripts.phase2d0_dependency_opportunity; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules; "
            "assert 'datasets' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
