from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from dflash.phase2d0.aggregation import _fraction_metrics, aggregate_gap_events
from dflash.phase2d0.analysis import analyze_captures
from dflash.phase2d1.aggregation import (
    _metric_row,
    policy_overlap,
    source_cluster_bootstrap,
    winner_composition,
)
from dflash.phase2d1.events import (
    FORBIDDEN_RUNTIME_FIELD_FRAGMENTS,
    RUNTIME_FEATURE_NAMES,
    build_selector_events,
    policy_map,
)
from dflash.phase2d1.preflight import validate_phase2d0_directory
from dflash.phase2d1.selectors import (
    _context_predictions,
    _events_for_fold,
    build_cross_validation_folds,
    choose_best_fixed_config,
    evaluate_policy_portfolios,
    evaluate_selectors,
    train_decision_stump,
)
from scripts.check_phase2d1_acceptance import check_output
from scripts.phase2d0_dependency_opportunity import main as phase2d0_main
from scripts.phase2d1_selector_realization import (
    OUTPUT_FILENAMES,
    parse_args,
)
from scripts.phase2d1_selector_realization import (
    main as phase2d1_main,
)
from tests.test_phase2b1 import _write_fixture
from tests.test_phase2d0 import _capture

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = REPO_ROOT / "docs/schemas"


def _source_capture(
    *,
    task: str,
    source: str,
    context: int,
    ranks: list[int | None],
    round_index: int = 0,
    terminal: bool = False,
    confidence_shift: float = 0.0,
) -> dict:
    accepted = 0
    for rank in ranks:
        if rank != 1:
            break
        accepted += 1
    row = _capture(
        ranks,
        accepted=accepted,
        terminal=terminal,
        sample_id=source,
        context_length=context,
        round_index=round_index,
    )
    row["task"] = task
    row["source_sample_id"] = source
    row["source_text_sha256"] = (source[-1] if source[-1].isalnum() else "a") * 64
    for position in row["positions"]:
        for candidate in position["candidates"]:
            candidate["log_probability"] -= confidence_shift * candidate["rank"]
    return row


def _selector_events() -> list[dict]:
    captures = []
    tasks = ("gov_report", "qasper", "hotpotqa")
    for index, task in enumerate(tasks):
        source = f"source-{index + 1}"
        captures.extend(
            [
                _source_capture(
                    task=task,
                    source=source,
                    context=1024,
                    ranks=[2, *([1] * 14)],
                    confidence_shift=0.01 * index,
                ),
                _source_capture(
                    task=task,
                    source=source,
                    context=4096,
                    ranks=[1, 2, *([1] * 13)],
                    round_index=1,
                    confidence_shift=0.02 * (index + 1),
                ),
            ]
        )
    analysis = analyze_captures(
        captures,
        top_ms=(16,),
        node_budgets=(16, 32),
        survival_budget_cap=32,
    )
    return build_selector_events(analysis["events"])


def _minimal_prediction_record(
    *, gain: int, headroom: int, source: str = "task::source"
) -> dict:
    return {
        "task": "gov_report",
        "source_cluster_id": source,
        "nominal_prompt_length": 1024,
        "node_budget_cap": 16,
        "selected_coverage_depth": gain,
        "selected_actual_node_count": 16,
        "same_actual_node_count_as_top1": True,
        "top1_coverage_depth": 0,
        "realized_gain": gain,
        "policy_oracle_headroom": headroom,
        "regret_to_policy_oracle": headroom - gain,
        "regret_to_prefix_closed_target_oracle": headroom - gain,
        "candidate_availability_prefix": 15,
        "candidate_available_for_extension": True,
    }


class Phase2D0DenominatorV2Tests(unittest.TestCase):
    def test_source_macro_namespaces_identical_ids_by_task(self) -> None:
        rows = [
            {
                "task": "gov_report",
                "sample_id": "shared",
                "source_sample_id": "shared",
                "classification_eligible": True,
                "gap_class": "selection_failure",
            },
            {
                "task": "gov_report",
                "sample_id": "shared",
                "source_sample_id": "shared",
                "classification_eligible": True,
                "gap_class": "selection_failure",
            },
            {
                "task": "qasper",
                "sample_id": "shared",
                "source_sample_id": "shared",
                "classification_eligible": True,
                "gap_class": "no_failure_within_observed_horizon",
            },
        ]
        metrics = _fraction_metrics(rows, "selection_failure")
        self.assertAlmostEqual(metrics["micro_fraction_over_all_rounds"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_fraction_over_all_rounds"], 0.5)

    def test_all_and_eligible_denominators_and_censored_null(self) -> None:
        eligible = _source_capture(
            task="gov_report",
            source="source-a",
            context=1024,
            ranks=[None, *([1] * 14)],
        )
        censored = _source_capture(
            task="gov_report",
            source="source-a",
            context=1024,
            ranks=[1, 1],
            round_index=1,
            terminal=True,
        )
        events = analyze_captures(
            [eligible, censored],
            top_ms=(16,),
            node_budgets=(16, 32),
            survival_budget_cap=32,
        )["events"]
        summary = aggregate_gap_events(events)
        for budget in (16, 32):
            rows = [
                row for row in summary["task_context"] if row["node_budget"] == budget
            ]
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
            self.assertEqual(censored_row["raw_round_count"], 2)
            self.assertEqual(censored_row["eligible_round_count"], 1)
            self.assertEqual(censored_row["right_censored_or_terminal_count"], 1)
            self.assertIsNone(censored_row["fraction_among_eligible_rounds"])
            self.assertEqual(censored_row["censoring_fraction"], 0.5)
            self.assertNotIn("fraction", censored_row)
        self.assertEqual(summary["raw_physical_round_count"], 2)
        self.assertEqual(summary["configuration_round_count"], 4)
        self.assertTrue(
            summary["cross_node_budget_counts_are_configuration_round_counts"]
        )


class Phase2D1EventTests(unittest.TestCase):
    def test_terminal_and_right_censored_rounds_are_excluded(self) -> None:
        eligible = _source_capture(
            task="gov_report",
            source="source-a",
            context=1024,
            ranks=[1] * 15,
        )
        censored = _source_capture(
            task="gov_report",
            source="source-a",
            context=1024,
            ranks=[1, 1],
            round_index=1,
            terminal=True,
        )
        phase2d0_events = analyze_captures(
            [eligible, censored],
            top_ms=(16,),
            node_budgets=(16, 32),
            survival_budget_cap=32,
        )["events"]
        selector_events = build_selector_events(phase2d0_events)
        self.assertEqual(len(phase2d0_events), 4)
        self.assertEqual(len(selector_events), 2)
        self.assertEqual({row["round_index"] for row in selector_events}, {0})

    def test_policy_and_prefix_oracles_are_distinct_and_ordered(self) -> None:
        events = _selector_events()
        for event in events:
            self.assertLessEqual(
                event["top1_coverage_depth"], event["policy_oracle_coverage_depth"]
            )
            self.assertLessEqual(
                event["policy_oracle_coverage_depth"],
                event["prefix_closed_target_oracle_coverage_depth"],
            )
            self.assertLessEqual(
                event["prefix_closed_target_oracle_coverage_depth"],
                event["candidate_availability_prefix"],
            )
            self.assertFalse(event["policy_oracle_is_deployable"])
            self.assertFalse(event["prefix_closed_target_oracle_is_deployable"])
            self.assertEqual(
                event["policy_oracle_coverage_depth"],
                max(row["coverage_depth"] for row in event["policy_configurations"]),
            )

    def test_ties_configs_and_node_accounting_are_exact(self) -> None:
        events = _selector_events()
        for event in events:
            expected_winners = sorted(
                row["config_id"]
                for row in event["policy_configurations"]
                if row["coverage_depth"] == event["policy_oracle_coverage_depth"]
            )
            self.assertEqual(event["policy_oracle_winner_config_ids"], expected_winners)
            self.assertEqual(event["policy_oracle_tie_count"], len(expected_winners))
            for policy in event["policy_configurations"]:
                self.assertLessEqual(
                    policy["actual_constructed_node_count"],
                    policy["node_budget_cap"],
                )
                self.assertTrue(policy["root_included_in_node_count"])
        top1_at_32 = [
            policy_map(event)[event["top1_config_id"]]
            for event in events
            if event["node_budget_cap"] == 32
        ]
        self.assertTrue(
            all(row["actual_constructed_node_count"] == 16 for row in top1_at_32)
        )
        self.assertTrue(all(row["budget_utilization"] == 0.5 for row in top1_at_32))

    def test_runtime_features_are_target_blind_and_entropy_is_named(self) -> None:
        event = _selector_events()[0]
        self.assertEqual(tuple(event["runtime_features"]), RUNTIME_FEATURE_NAMES)
        for field in event["runtime_features"]:
            self.assertFalse(
                any(fragment in field for fragment in FORBIDDEN_RUNTIME_FIELD_FRAGMENTS)
            )
        self.assertIn("mean_top16_renormalized_entropy", event["runtime_features"])
        self.assertIn("not full-vocabulary entropy", event["top16_entropy_semantics"])

    def test_winner_overlap_reports_ties_and_complementarity(self) -> None:
        events = _selector_events()
        winners = winner_composition(events)
        overlaps = policy_overlap(events)
        self.assertTrue(winners)
        self.assertTrue(overlaps)
        self.assertTrue(all(row["winner_entropy_nats"] >= 0 for row in winners))
        self.assertTrue(
            all(row["pairwise_complementarity_fraction"] >= 0 for row in overlaps)
        )


class Phase2D1CrossValidationTests(unittest.TestCase):
    def test_fixed_policy_tie_break_is_order_independent(self) -> None:
        events = [row for row in _selector_events() if row["node_budget_cap"] == 32]
        for event in events:
            for policy in event["policy_configurations"]:
                policy["coverage_depth"] = 1
        candidates = sorted(
            (
                sum(
                    policy_map(event)[config]["actual_constructed_node_count"]
                    for event in events
                ),
                config,
            )
            for config in policy_map(events[0])
        )
        expected = candidates[0][1]
        self.assertEqual(choose_best_fixed_config(events), expected)
        self.assertEqual(choose_best_fixed_config(list(reversed(events))), expected)

    def test_same_source_all_contexts_are_held_out_together(self) -> None:
        events = _selector_events()
        folds = build_cross_validation_folds(events)
        for fold in folds["folds"]:
            self.assertFalse(
                set(fold["training_source_ids"]) & set(fold["held_out_source_ids"])
            )
        source_fold = next(
            fold
            for fold in folds["folds"]
            if fold["evaluation_scheme"] == "leave-one-source-out"
            and "gov_report::source-1" in fold["held_out_source_ids"]
        )
        _, held_out = _events_for_fold(events, source_fold, 16)
        self.assertEqual(
            {row["nominal_prompt_length"] for row in held_out}, {1024, 4096}
        )

    def test_leave_one_task_out_holds_every_task_source(self) -> None:
        events = _selector_events()
        folds = build_cross_validation_folds(events)
        fold = next(row for row in folds["folds"] if row["fold_id"] == "loto::qasper")
        self.assertEqual(fold["held_out_task_ids"], ["qasper"])
        self.assertTrue(
            all(source.startswith("qasper::") for source in fold["held_out_source_ids"])
        )
        self.assertTrue(
            all(
                not source.startswith("qasper::")
                for source in fold["training_source_ids"]
            )
        )

    def test_fixed_and_context_rules_are_selected_from_training_only(self) -> None:
        events = _selector_events()
        folds = build_cross_validation_folds(events)
        fold = next(
            row
            for row in folds["folds"]
            if row["fold_id"] == "loso::gov_report::source-1"
        )
        training, held_out = _events_for_fold(events, fold, 16)
        selected = choose_best_fixed_config(training)
        predictions = _context_predictions(training, held_out, fold)
        self.assertIn(selected, policy_map(training[0]))
        self.assertTrue(
            all(
                row["selector_parameters"]["bin_boundaries_selected_from_training_only"]
                for row in predictions
            )
        )
        self.assertTrue(
            all(
                row["held_out_source_id"] not in row["training_source_ids"]
                for row in predictions
            )
        )

    def test_decision_stump_threshold_and_policies_use_training_only(self) -> None:
        events = _selector_events()
        folds = build_cross_validation_folds(events)
        fold = next(
            row
            for row in folds["folds"]
            if row["fold_id"] == "loso::gov_report::source-1"
        )
        training, _ = _events_for_fold(events, fold, 16)
        rule = train_decision_stump(
            training,
            minimum_leaf_support=1,
            minimum_leaf_source_support=1,
            maximum_thresholds_per_feature=8,
        )
        self.assertTrue(rule["threshold_selected_from_training_only"])
        self.assertTrue(rule["policies_selected_from_training_only"])
        if rule["stump_available"]:
            self.assertIn(rule["feature"], RUNTIME_FEATURE_NAMES)
            self.assertGreaterEqual(rule["left_training_round_count"], 1)
            self.assertGreaterEqual(rule["right_training_round_count"], 1)

    def test_single_stump_threshold_limit_is_supported(self) -> None:
        events = _selector_events()
        folds = build_cross_validation_folds(events)
        fold = next(
            row
            for row in folds["folds"]
            if row["fold_id"] == "loso::gov_report::source-1"
        )
        training, _ = _events_for_fold(events, fold, 16)
        rule = train_decision_stump(
            training,
            minimum_leaf_support=1,
            minimum_leaf_source_support=1,
            maximum_thresholds_per_feature=1,
        )
        self.assertIn("stump_available", rule)

    def test_selector_predictions_record_fold_and_no_leakage(self) -> None:
        events = _selector_events()
        folds = build_cross_validation_folds(events)
        predictions = evaluate_selectors(
            events,
            folds,
            minimum_leaf_support=1,
            minimum_leaf_source_support=1,
            maximum_thresholds_per_feature=8,
        )
        for rows in predictions.values():
            self.assertTrue(rows)
            self.assertTrue(
                all(
                    row["held_out_source_id"] not in row["training_source_ids"]
                    for row in rows
                )
            )

    def test_portfolio_is_explicit_oracle(self) -> None:
        events = _selector_events()
        folds = build_cross_validation_folds(events)
        rows = evaluate_policy_portfolios(events, folds, portfolio_sizes=(1, 2, 3))
        self.assertTrue(rows)
        self.assertTrue(all(row["portfolio_is_oracle"] for row in rows))
        self.assertTrue(all(not row["portfolio_is_deployable"] for row in rows))


class Phase2D1MetricsTests(unittest.TestCase):
    def test_zero_headroom_ratio_is_null_with_reason(self) -> None:
        row = _metric_row(
            [_minimal_prediction_record(gain=0, headroom=0)],
            identity={"scope": "overall", "node_budget_cap": 16},
            candidate_subset="all_eligible_rounds",
        )
        self.assertIsNone(row["headroom_capture_ratio_of_sums"])
        self.assertEqual(
            row["headroom_capture_ratio_of_sums_unavailable_reason"],
            "zero_policy_oracle_headroom_sum",
        )

    def test_ratio_of_sums_and_mean_round_ratio_are_distinct(self) -> None:
        records = [
            _minimal_prediction_record(gain=1, headroom=1, source="a"),
            _minimal_prediction_record(gain=1, headroom=9, source="b"),
        ]
        row = _metric_row(
            records,
            identity={"scope": "overall", "node_budget_cap": 16},
            candidate_subset="all_eligible_rounds",
        )
        self.assertAlmostEqual(row["headroom_capture_ratio_of_sums"], 0.2)
        self.assertAlmostEqual(row["mean_per_round_headroom_capture_ratio"], 5 / 9)
        self.assertEqual(row["headroom_capture_primary_metric"], "ratio_of_sums")

    def test_candidate_available_subset_uses_its_own_denominator(self) -> None:
        available = _minimal_prediction_record(gain=1, headroom=2)
        unavailable = _minimal_prediction_record(gain=0, headroom=0, source="b")
        unavailable["candidate_available_for_extension"] = False
        row = _metric_row(
            [available, unavailable],
            identity={"scope": "overall", "node_budget_cap": 16},
            candidate_subset="candidate_available_for_extension",
        )
        self.assertEqual(row["round_count"], 1)
        self.assertEqual(row["source_count"], 1)

    def test_clustered_bootstrap_samples_sources_not_rounds(self) -> None:
        rows = []
        for source in ("a", "b", "c"):
            for round_index in range(2):
                rows.append(
                    {
                        "evaluation_scheme": "leave-one-source-out",
                        "selector_family": "global_fixed",
                        "node_budget_cap": 16,
                        "held_out_task": "gov_report",
                        "held_out_source_id": source,
                        "realized_gain": round_index,
                        "policy_oracle_headroom": 2,
                    }
                )
        result = source_cluster_bootstrap(
            rows, repeats=20, seed=7, minimum_stable_clusters=3
        )
        overall = next(row for row in result if row["scope"] == "overall")
        self.assertEqual(overall["sampling_unit"], "source_document_cluster")
        self.assertEqual(overall["source_cluster_count"], 3)
        self.assertFalse(overall["round_level_bootstrap_used"])


class Phase2D1RunnerTests(unittest.TestCase):
    def test_cli_defaults_and_overrides(self) -> None:
        args = parse_args(
            [
                "--capture-dir",
                "capture",
                "--phase2d0-dir",
                "phase2d0",
            ]
        )
        self.assertEqual(args.top_m, 16)
        self.assertEqual(args.node_budgets, (16, 32))
        self.assertEqual(
            args.evaluation, ("leave-one-source-out", "leave-one-task-out")
        )
        self.assertEqual(args.portfolio_sizes, (1, 2, 3))
        self.assertEqual(args.minimum_global_fixed_selection_stability, 0.70)

    def test_preflight_failure_leaves_no_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "capture"
            capture.mkdir()
            _write_fixture(capture, task="gov_report")
            output = root / "must-not-exist"
            with self.assertRaisesRegex(ValueError, "Phase 2D.0 directory"):
                phase2d1_main(
                    [
                        "--capture-dir",
                        str(capture),
                        "--phase2d0-dir",
                        str(root / "missing-phase2d0"),
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertFalse(output.exists())

    def test_phase2d0_hash_mismatch_fails_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            captures = []
            arguments = []
            for task in ("gov_report", "qasper", "hotpotqa"):
                capture = root / task
                capture.mkdir()
                _write_fixture(capture, task=task)
                captures.append(capture)
                arguments.extend(("--capture-dir", str(capture)))
            phase2d0 = root / "phase2d0"
            self.assertEqual(
                phase2d0_main(
                    [
                        *arguments,
                        "--top-ms",
                        "16",
                        "--node-budgets",
                        "16,32",
                        "--survival-budget-cap",
                        "32",
                        "--output-dir",
                        str(phase2d0),
                    ]
                ),
                0,
            )
            validated = validate_capture_directories_for_test(captures)
            manifest_path = phase2d0 / "analysis-manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["inputs"][0]["capture_directory_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "provenance/hash mismatch"):
                validate_phase2d0_directory(
                    phase2d0,
                    validated_capture=validated,
                    schema_dir=SCHEMA_DIR,
                    top_m=16,
                    node_budgets=(16, 32),
                )

    def test_end_to_end_cpu_only_artifacts_and_checker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = []
            for task in ("gov_report", "qasper", "hotpotqa"):
                capture = root / task
                capture.mkdir()
                _write_fixture(capture, task=task)
                arguments.extend(("--capture-dir", str(capture)))
            phase2d0 = root / "phase2d0"
            self.assertEqual(
                phase2d0_main(
                    [
                        *arguments,
                        "--top-ms",
                        "16",
                        "--node-budgets",
                        "16,32",
                        "--survival-budget-cap",
                        "32",
                        "--output-dir",
                        str(phase2d0),
                    ]
                ),
                0,
            )
            output = root / "phase2d1"
            result = phase2d1_main(
                [
                    *arguments,
                    "--phase2d0-dir",
                    str(phase2d0),
                    "--minimum-leaf-support",
                    "1",
                    "--minimum-leaf-source-support",
                    "1",
                    "--stump-max-thresholds-per-feature",
                    "4",
                    "--bootstrap-repeats",
                    "10",
                    "--minimum-stable-clusters",
                    "1",
                    "--minimum-source-document-count",
                    "2",
                    "--minimum-task-count",
                    "2",
                    "--minimum-eligible-physical-rounds",
                    "1",
                    "--no-require-stable-clustered-ci",
                    "--expected-phase2d0-decision",
                    "insufficient_evidence",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(result, 0)
            self.assertEqual(
                {path.name for path in output.iterdir()}, set(OUTPUT_FILENAMES)
            )
            self.assertTrue(check_output(output)["passed"])
            manifest = json.loads((output / "analysis-manifest.json").read_text())
            self.assertFalse(any(manifest["framework_modules_loaded"].values()))
            self.assertTrue(
                manifest["count_semantics"][
                    "configuration_records_are_not_independent_physical_rounds"
                ]
            )
            decision = json.loads((output / "selector-decision.json").read_text())
            self.assertIn("global_fixed_selection_stability_by_node_budget", decision)
            self.assertEqual(
                decision["decision_rules"]["minimum_global_fixed_selection_stability"],
                0.70,
            )
            research = json.loads(
                (output / "research-question-summary.json").read_text()
            )
            self.assertEqual(
                {
                    row["node_budget_cap"]
                    for row in research["rq3"]["winner_heterogeneity_by_node_budget"]
                },
                {16, 32},
            )
            self.assertTrue(research["rq3"]["portfolio_is_non_deployable_oracle"])
            self.assertEqual(
                {
                    row["selector_level"]
                    for row in research["rq6"]["candidate_subset_comparison"]
                },
                {"global_fixed", "context_rule", "decision_stump"},
            )

    def test_schemas_and_lazy_imports(self) -> None:
        schemas = sorted(SCHEMA_DIR.glob("phase2d1-*.schema.json"))
        self.assertEqual(len(schemas), 6)
        for path in schemas:
            Draft202012Validator.check_schema(json.loads(path.read_text()))
        code = (
            "import sys; import dflash.phase2d1; "
            "import scripts.phase2d1_selector_realization; "
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


def validate_capture_directories_for_test(capture_dirs: list[Path]) -> dict:
    from dflash.phase2b1.validation import validate_capture_directories

    return validate_capture_directories(capture_dirs, schema_dir=SCHEMA_DIR)


if __name__ == "__main__":
    unittest.main()
