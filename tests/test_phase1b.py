from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase1.materialize import VENDOR_ROOT
from dflash.phase1b.metrics import (
    aggregate_sweep,
    distribution,
    summarize_profile_rounds,
)
from dflash.phase1b.workload import (
    PROMPT_PREFIX,
    build_longbench_workloads,
    build_nested_workloads,
    build_workload_inventory,
    prepare_longbench_records,
)
from experiments.phase1b_longctx import (
    SUMMARY_CSV_FIELDS,
    _write_summary_csv,
    main,
    new_manifest,
    new_summary,
    parse_args,
    run_sweep,
)
from scripts.collect_environment import collect_environment


class FakeTokenizer:
    eos_token_id = 0
    model_max_length = 100_000

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        content = messages[0]["content"]
        assert content.startswith(PROMPT_PREFIX)
        report = content[len(PROMPT_PREFIX) :]
        return list(range(4 + len(report)))

    def encode(self, text, **kwargs):
        del kwargs
        return list(range(len(text)))


class GappedTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        content = messages[0]["content"]
        report = content[len(PROMPT_PREFIX) :]
        return list(range(4 + 2 * len(report)))


class LongBenchTokenizer:
    eos_token_id = 0
    model_max_length = 100_000

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return list(range(3 + len(messages[0]["content"])))

    def encode(self, text, **kwargs):
        del kwargs
        return list(range(len(text)))


def _load_observability_helper():
    path = VENDOR_ROOT / "instrumented" / "phase1_observability.py"
    spec = importlib.util.spec_from_file_location("phase1b_helper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _round(
    *,
    round_index: int,
    accepted: int,
    stop_reason: str,
    truncated: int,
    context_length: int = 100,
) -> dict:
    proposed = 3
    return {
        "round_index": round_index,
        "context_length_before": context_length,
        "context_length_after": context_length + accepted + 1,
        "proposed_draft_token_count": proposed,
        "accepted_draft_token_count": accepted,
        "rejected_draft_token_count": proposed - accepted,
        "committed_token_count": accepted + 1,
        "returned_token_delta": accepted + 1 - truncated,
        "truncated_token_count": truncated,
        "correction_token_count": int(accepted != proposed),
        "stop_reason": stop_reason,
        "attention_calls": [
            {
                "effective_kv_length": context_length + 4,
                "theoretical_qk_score_elements": (context_length + 4) * 8,
            }
        ],
        "timing": {
            "draft_cuda_elapsed_ms": 2.0,
            "combined_attention_cuda_elapsed_ms": 1.0,
            "verification_cuda_elapsed_ms": 3.0,
            "round_cuda_elapsed_ms": 6.0,
            "combined_attention_over_draft_ratio": 0.5,
            "combined_attention_over_round_ratio": 1.0 / 6.0,
        },
    }


def _sample(nominal: int, value: float) -> dict:
    timing = distribution([value, value * 1.1, value * 0.9])
    accounting = summarize_profile_rounds(
        [
            _round(round_index=0, accepted=1, stop_reason="continue", truncated=0),
            _round(
                round_index=1, accepted=3, stop_reason="max_new_tokens", truncated=2
            ),
        ],
        generated_token_count=5,
    )
    return {
        "nominal_prompt_length": nominal,
        "actual_prompt_length": nominal,
        "generated_token_count": 5,
        "control_timing": {
            "total_cuda_elapsed_ms": timing,
            "generated_tokens_per_second": distribution([100.0]),
        },
        "profiling_timing": {
            "prefill_cuda_elapsed_ms": timing,
            "decode_cuda_elapsed_ms": timing,
            "total_cuda_elapsed_ms": timing,
            "draft_cuda_elapsed_ms": timing,
            "combined_attention_cuda_elapsed_ms": timing,
            "verification_cuda_elapsed_ms": timing,
            "combined_attention_over_draft_ratio": distribution([0.5]),
            "combined_attention_over_round_ratio": distribution([0.2]),
            "draft_over_round_ratio": distribution([1.0 / 3.0]),
            "verification_over_round_ratio": distribution([0.5]),
        },
        "memory": {"peak_allocated_bytes": 10, "peak_reserved_bytes": 20},
        "accounting": accounting,
    }


class Phase1BTests(unittest.TestCase):
    def test_three_longbench_schemas_and_task_isolation(self) -> None:
        tokenizer = LongBenchTokenizer()
        fixtures = {
            "gov_report": {
                "_id": "gov-1",
                "dataset": "gov_report",
                "context": "government report body",
                "input": "must be ignored for summarization",
                "answers": ["reference summary"],
            },
            "qasper": {
                "_id": "qasper-1",
                "dataset": "qasper",
                "context": "paper body",
                "input": "What did the paper find?",
                "answers": ["reference answer"],
            },
            "hotpotqa": {
                "_id": "hotpot-1",
                "dataset": "hotpotqa",
                "context": "multiple documents",
                "input": "Which entity connects them?",
                "answers": ["reference entity"],
            },
        }
        for task, record in fixtures.items():
            prepared = prepare_longbench_records(
                [record], tokenizer=tokenizer, task=task
            )
            self.assertEqual(prepared[0]["context"], record["context"])
            self.assertEqual(
                prepared[0]["query"], "" if task == "gov_report" else record["input"]
            )
            self.assertEqual(prepared[0]["reference_answers"], record["answers"])
        with self.assertRaisesRegex(ValueError, "tasks cannot be mixed"):
            prepare_longbench_records(
                [fixtures["qasper"]], tokenizer=tokenizer, task="hotpotqa"
            )

    def test_query_context_and_reference_answers_have_distinct_roles(self) -> None:
        context = "CONTEXT_ONLY_" + "c" * 800
        query = "QUERY_ONLY_WHAT_HAPPENED"
        answer = "REFERENCE_ANSWER_MUST_NOT_ENTER_PROMPT"
        prepared = prepare_longbench_records(
            [
                {
                    "_id": "paper-1",
                    "context": context,
                    "input": query,
                    "answers": [answer],
                }
            ],
            tokenizer=LongBenchTokenizer(),
            task="qasper",
        )
        workloads, _ = build_longbench_workloads(
            prepared,
            tokenizer=LongBenchTokenizer(),
            task="qasper",
            nominal_lengths=(256,),
            sample_count=1,
            sample_policy="matched",
            max_new_tokens=8,
            model_position_limit=512,
        )
        row = workloads[0]
        self.assertIn(query, row["prompt_text"])
        self.assertIn("CONTEXT_ONLY_", row["prompt_text"])
        self.assertNotIn(answer, row["prompt_text"])
        self.assertEqual(row["reference_answers"], [answer])
        self.assertEqual(
            row["source_text_sha256"], hashlib.sha256(context.encode()).hexdigest()
        )
        self.assertEqual(
            row["query_sha256"], hashlib.sha256(query.encode()).hexdigest()
        )

    def test_longbench_nested_context_construction(self) -> None:
        tokenizer = LongBenchTokenizer()
        prepared = prepare_longbench_records(
            [
                {
                    "_id": "paper-nested",
                    "context": "0123456789" * 100,
                    "input": "What is reported?",
                    "answers": ["metadata only"],
                }
            ],
            tokenizer=tokenizer,
            task="qasper",
        )
        workloads, cohort = build_longbench_workloads(
            prepared,
            tokenizer=tokenizer,
            task="qasper",
            nominal_lengths=(256, 384),
            sample_count=1,
            sample_policy="matched",
            max_new_tokens=8,
            model_position_limit=512,
        )
        self.assertEqual([row["actual_prompt_length"] for row in workloads], [256, 384])
        shorter = workloads[0]["context_prefix_characters"]
        longer = workloads[1]["context_prefix_characters"]
        self.assertLess(shorter, longer)
        self.assertTrue(cohort["paired_context_length_comparison"])
        self.assertEqual(
            cohort["sample_ids_by_length"]["256"],
            cohort["sample_ids_by_length"]["384"],
        )

    def test_matched_and_per_length_selection_policies(self) -> None:
        tokenizer = LongBenchTokenizer()
        records = [
            {"_id": "long-a", "context": "a" * 600, "input": "question"},
            {"_id": "long-b", "context": "b" * 600, "input": "question"},
            {"_id": "short", "context": "s" * 100, "input": "question"},
        ]
        prepared = prepare_longbench_records(
            records, tokenizer=tokenizer, task="qasper"
        )
        _, matched = build_longbench_workloads(
            prepared,
            tokenizer=tokenizer,
            task="qasper",
            nominal_lengths=(200, 400),
            sample_count=2,
            sample_policy="matched",
            max_new_tokens=8,
            model_position_limit=512,
        )
        self.assertTrue(matched["paired_context_length_comparison"])
        self.assertEqual(
            matched["sample_ids_by_length"]["200"],
            matched["sample_ids_by_length"]["400"],
        )
        _, per_length = build_longbench_workloads(
            prepared,
            tokenizer=tokenizer,
            task="qasper",
            nominal_lengths=(200, 400),
            sample_count=3,
            sample_policy="per-length",
            max_new_tokens=8,
            model_position_limit=512,
        )
        self.assertFalse(per_length["paired_context_length_comparison"])
        self.assertEqual(per_length["actual_sample_count_by_length"]["200"], 3)
        self.assertEqual(per_length["actual_sample_count_by_length"]["400"], 2)
        self.assertIn("not a strict paired", per_length["interpretation"])

    def test_insufficient_matched_sample_error_is_explicit(self) -> None:
        tokenizer = LongBenchTokenizer()
        prepared = prepare_longbench_records(
            [{"_id": "only", "context": "x" * 600, "input": "question"}],
            tokenizer=tokenizer,
            task="qasper",
        )
        with self.assertRaisesRegex(RuntimeError, "only 1 samples reach 400"):
            build_longbench_workloads(
                prepared,
                tokenizer=tokenizer,
                task="qasper",
                nominal_lengths=(200, 400),
                sample_count=2,
                sample_policy="matched",
                max_new_tokens=8,
                model_position_limit=512,
            )

    def test_inventory_counts_statistics_and_recommendations(self) -> None:
        tokenizer = LongBenchTokenizer()
        prepared = prepare_longbench_records(
            [
                {"_id": "long-a", "context": "a" * 600, "input": "question"},
                {"_id": "long-b", "context": "b" * 600, "input": "question"},
                {"_id": "short", "context": "s" * 100, "input": "question"},
            ],
            tokenizer=tokenizer,
            task="qasper",
        )
        inventory = build_workload_inventory(
            prepared,
            task="qasper",
            nominal_lengths=(200, 400),
            requested_sample_count=8,
        )
        self.assertEqual(
            inventory["eligible_by_prompt_length"]["200"]["eligible_sample_count"], 3
        )
        self.assertEqual(
            inventory["eligible_by_prompt_length"]["400"]["eligible_sample_count"], 2
        )
        self.assertEqual(inventory["recommendations"]["matched"]["sample_count"], 2)
        self.assertEqual(inventory["recommendations"]["matched"]["context_lengths"], [200, 400])
        self.assertEqual(inventory["document_token_length"]["min"], 100)

    def test_exact_nested_context_construction(self) -> None:
        records = [
            {"id": "a", "report": "a" * 100},
            {"id": "b", "report": "b" * 100},
        ]
        workloads = build_nested_workloads(
            records,
            tokenizer=FakeTokenizer(),
            nominal_lengths=(8, 16, 32),
            sample_count=2,
            split="test",
            seed=7,
            max_new_tokens=4,
            model_position_limit=64,
        )
        self.assertEqual(len(workloads), 6)
        for sample_id in {row["sample_id"] for row in workloads}:
            group = [row for row in workloads if row["sample_id"] == sample_id]
            self.assertEqual(
                [row["actual_prompt_length"] for row in group], [8, 16, 32]
            )
            self.assertTrue(all(row["exact_nominal_length"] for row in group))
            prefixes = [row["prompt_text"][len(PROMPT_PREFIX) :] for row in group]
            self.assertTrue(prefixes[1].startswith(prefixes[0]))
            self.assertTrue(prefixes[2].startswith(prefixes[1]))
            self.assertEqual(len({row["source_text_sha256"] for row in group}), 1)

    def test_model_position_limit_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds model position limit"):
            build_nested_workloads(
                [{"report": "x" * 100}],
                tokenizer=FakeTokenizer(),
                nominal_lengths=(32,),
                sample_count=1,
                max_new_tokens=4,
                model_position_limit=35,
            )

    def test_nominal_and_actual_lengths_are_both_recorded(self) -> None:
        workload = build_nested_workloads(
            [{"id": "gapped", "report": "x" * 100}],
            tokenizer=GappedTokenizer(),
            nominal_lengths=(9,),
            sample_count=1,
            max_new_tokens=4,
            model_position_limit=64,
        )[0]
        self.assertEqual(workload["nominal_prompt_length"], 9)
        self.assertEqual(workload["actual_prompt_length"], 8)
        self.assertEqual(workload["nominal_prompt_length_shortfall"], 1)
        self.assertFalse(workload["exact_nominal_length"])

    def test_kv_length_invariants(self) -> None:
        helper = _load_observability_helper()
        invariants = helper.validate_attention_kv_lengths(
            context_length_before=1024,
            cached_context_kv_length=1000,
            newly_injected_hidden_length=24,
            persistent_context_kv_length=1024,
            query_length=16,
            effective_kv_length=1040,
        )
        self.assertTrue(all(invariants.values()))
        with self.assertRaisesRegex(RuntimeError, "invariant failed"):
            helper.validate_attention_kv_lengths(
                context_length_before=1024,
                cached_context_kv_length=1000,
                newly_injected_hidden_length=23,
                persistent_context_kv_length=1024,
                query_length=16,
                effective_kv_length=1040,
            )

    def test_terminal_truncation_accounting_policy(self) -> None:
        result = summarize_profile_rounds(
            [
                _round(round_index=0, accepted=1, stop_reason="continue", truncated=0),
                _round(
                    round_index=1,
                    accepted=3,
                    stop_reason="max_new_tokens",
                    truncated=2,
                ),
            ],
            generated_token_count=5,
        )
        self.assertEqual(result["raw_all_round_accepted_length"], [1, 3])
        self.assertEqual(result["non_terminal_accepted_length"], [1])
        self.assertEqual(result["truncated_token_count"], 2)
        self.assertEqual(result["acceptance_eligible_round_count"], 1)
        self.assertEqual(result["non_terminal_acceptance_rate"], 1 / 3)
        self.assertTrue(result["acceptance_policy"]["terminal_rounds_excluded"])

    def test_aggregate_metrics(self) -> None:
        samples = [_sample(1024, 10.0), _sample(1024, 20.0)]
        rounds = []
        for sample_index in range(2):
            for row in (
                _round(round_index=0, accepted=1, stop_reason="continue", truncated=0),
                _round(
                    round_index=1,
                    accepted=3,
                    stop_reason="max_new_tokens",
                    truncated=2,
                ),
            ):
                row["nominal_prompt_length"] = 1024
                row["sample_id"] = str(sample_index)
                rounds.append(row)
        aggregate = aggregate_sweep(samples, rounds)["by_context_length"]["1024"]
        self.assertEqual(aggregate["sample_count"], 2)
        self.assertAlmostEqual(
            aggregate["sample_level"]["control_timing.total_cuda_elapsed_ms.median"][
                "median"
            ],
            15.0,
        )
        self.assertAlmostEqual(
            aggregate["micro_average"]["raw_all_round_accepted_over_proposed"],
            4 / 6,
        )
        self.assertAlmostEqual(
            aggregate["micro_average"]["non_terminal_accepted_over_proposed"],
            1 / 3,
        )
        self.assertEqual(aggregate["round_level"]["round_cuda_elapsed_ms"]["count"], 4)
        self.assertEqual(
            aggregate["round_level"]["non_terminal_accepted_length"]["count"], 2
        )
        self.assertEqual(aggregate["micro_average"]["rejected_draft_token_count"], 4)
        self.assertEqual(aggregate["micro_average"]["committed_token_count"], 12)

    def test_cli_defaults_overrides_and_schemas(self) -> None:
        defaults = parse_args(
            [
                "--materialized-draft-path",
                "/models/draft",
                "--govreport-jsonl",
                "/data/govreport.jsonl",
            ]
        )
        self.assertEqual(defaults.context_lengths, (1024, 4096, 8192, 16384, 32768))
        self.assertEqual(defaults.pilot_samples, 8)
        self.assertEqual(defaults.timing_repeats, 3)
        self.assertEqual(defaults.max_new_tokens, 128)
        overridden = parse_args(
            [
                "--materialized-draft-path",
                "/models/draft",
                "--govreport-dataset-path",
                "/data/govreport",
                "--context-lengths",
                "1K,32K",
                "--pilot-samples",
                "1",
                "--timing-repeats",
                "1",
                "--max-new-tokens",
                "8",
            ]
        )
        self.assertEqual(overridden.context_lengths, (1024, 32768))
        longbench = parse_args(
            [
                "--inspect-workload-only",
                "--longbench-root",
                "/data/LongBench",
                "--longbench-task",
                "qasper",
                "--context-field",
                "context",
                "--query-field",
                "input",
                "--answers-field",
                "answers",
                "--sample-policy",
                "per-length",
            ]
        )
        self.assertEqual(longbench.longbench_task, "qasper")
        self.assertEqual(longbench.sample_policy, "per-length")
        self.assertTrue(longbench.inspect_workload_only)
        self.assertIsNone(longbench.materialized_draft_path)
        environment = collect_environment(
            run_config={}, run_command="phase1b test", import_torch=False
        )
        manifest = new_manifest(defaults, environment, "phase1b test")
        summary = new_summary()
        for name, value in (("manifest", manifest), ("summary", summary)):
            schema = json.loads(
                (
                    REPO_ROOT / "docs" / "schemas" / f"phase1b-{name}.schema.json"
                ).read_text()
            )
            self.assertTrue(set(schema["required"]).issubset(value))
        sample_schema = json.loads(
            (REPO_ROOT / "docs/schemas/phase1b-sample.schema.json").read_text()
        )
        self.assertIn("control_timing", sample_schema["required"])
        round_schema = json.loads(
            (REPO_ROOT / "docs/schemas/phase1b-round.schema.json").read_text()
        )
        attention_required = round_schema["properties"]["attention_calls"]["items"][
            "required"
        ]
        self.assertIn("persistent_context_kv_length", attention_required)
        inventory_schema = json.loads(
            (
                REPO_ROOT
                / "docs/schemas/phase1b-workload-inventory.schema.json"
            ).read_text()
        )
        self.assertIn("models_loaded", inventory_schema["required"])

    def test_inventory_only_never_loads_cuda_or_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "qasper.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "_id": "q-1",
                        "context": "c" * 600,
                        "input": "question",
                        "answers": ["answer"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "inventory-output"
            tokenizer_source = {
                "source_kind": "local_path",
                "load_source": "/models/tokenizer",
            }
            with (
                patch(
                    "experiments.phase1b_longctx._load_tokenizer_only",
                    return_value=(LongBenchTokenizer(), tokenizer_source, "test"),
                ),
                patch("experiments.phase1b_longctx._require_cuda") as require_cuda,
                patch(
                    "experiments.phase1b_longctx._load_materialized_draft"
                ) as load_draft,
            ):
                exit_code = main(
                    [
                        "--inspect-workload-only",
                        "--dataset-jsonl",
                        str(dataset_path),
                        "--longbench-task",
                        "qasper",
                        "--context-lengths",
                        "200,400",
                        "--pilot-samples",
                        "1",
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(exit_code, 0)
            require_cuda.assert_not_called()
            load_draft.assert_not_called()
            inventory = json.loads(
                (output / "workload-inventory.json").read_text()
            )
            self.assertFalse(inventory["models_loaded"])
            self.assertFalse(inventory["cuda_required"])
            self.assertEqual(inventory["task"], "qasper")
            self.assertTrue((output / "environment.json").is_file())
            self.assertTrue((output / "terminal.log").is_file())

    def test_inventory_only_does_not_import_torch_or_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "qasper.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "_id": "q-1",
                        "context": "c" * 600,
                        "input": "question",
                        "answers": ["answer"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            code = f"""
import sys
import types

class Tokenizer:
    eos_token_id = 0
    model_max_length = 100000
    def apply_chat_template(self, messages, **kwargs):
        return list(range(3 + len(messages[0]['content'])))
    def encode(self, text, **kwargs):
        return list(range(len(text)))

class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return Tokenizer()

transformers = types.ModuleType('transformers')
transformers.__version__ = 'test'
transformers.AutoTokenizer = AutoTokenizer
sys.modules['transformers'] = transformers

from experiments.phase1b_longctx import main
result = main([
    '--inspect-workload-only',
    '--dataset-jsonl', {str(dataset_path)!r},
    '--longbench-task', 'qasper',
    '--context-lengths', '200,400',
    '--pilot-samples', '1',
    '--output-dir', {str(root / 'output')!r},
])
assert result == 0
assert 'torch' not in sys.modules, 'inventory-only imported torch'
assert 'datasets' not in sys.modules, 'JSONL inventory imported datasets'
assert not hasattr(transformers, 'AutoModel'), 'test fake exposed a model loader'
"""
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cohort_preflight_fails_before_cuda_or_model_load(self) -> None:
        args = parse_args(
            [
                "--materialized-draft-path",
                "/models/draft",
                "--dataset-jsonl",
                "/data/qasper.jsonl",
                "--longbench-task",
                "qasper",
                "--context-lengths",
                "200,400",
                "--pilot-samples",
                "8",
            ]
        )
        inventory = {
            "task": "qasper",
            "record_count": 1,
            "eligible_by_prompt_length": {},
            "document_token_length": {
                "min": 1,
                "median": 1,
                "p90": 1,
                "max": 1,
            },
            "sample_ids_reaching_maximum_length": [],
            "recommendations": {
                "matched": {"sample_count": 1, "context_lengths": [200]},
                "per_length": {"sample_count_by_length": {}},
            },
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "experiments.phase1b_longctx._load_tokenizer_only",
                return_value=(LongBenchTokenizer(), {}, "test"),
            ),
            patch(
                "experiments.phase1b_longctx._prepare_workload",
                return_value=([], inventory, None, None),
            ),
            patch(
                "experiments.phase1b_longctx.build_longbench_workloads",
                side_effect=RuntimeError("insufficient matched cohort"),
            ),
            patch("experiments.phase1b_longctx._require_cuda") as require_cuda,
            self.assertRaisesRegex(RuntimeError, "insufficient matched"),
        ):
            output = Path(directory)
            run_sweep(
                args,
                {"workload": {}},
                {},
                {},
                [],
                [],
                samples_path=output / "samples.jsonl",
                rounds_path=output / "rounds.jsonl",
            )
        require_cuda.assert_not_called()

    def test_summary_csv_includes_phase1b1_fields(self) -> None:
        samples = [_sample(1024, 10.0)]
        rounds = []
        for row in (
            _round(round_index=0, accepted=1, stop_reason="continue", truncated=0),
            _round(
                round_index=1, accepted=3, stop_reason="max_new_tokens", truncated=2
            ),
        ):
            row["nominal_prompt_length"] = 1024
            rounds.append(row)
        aggregate = aggregate_sweep(samples, rounds)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            _write_summary_csv(path, aggregate)
            with path.open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
        required = {
            "profile_draft_cuda_median_ms",
            "profile_verification_cuda_median_ms",
            "profile_round_cuda_median_ms",
            "profile_draft_over_round_median",
            "profile_verification_over_round_median",
            "mean_non_terminal_accepted_draft_tokens",
            "mean_non_terminal_committed_tokens",
            "round_count",
            "acceptance_eligible_round_count",
            "truncated_token_count",
        }
        self.assertTrue(required.issubset(SUMMARY_CSV_FIELDS))
        self.assertTrue(required.issubset(row))
        self.assertEqual(row["mean_non_terminal_accepted_draft_tokens"], "1.0")
        self.assertEqual(row["mean_non_terminal_committed_tokens"], "2.0")

    def test_import_does_not_load_model_frameworks(self) -> None:
        code = """
import sys
import dflash.phase1b
import experiments.phase1b_longctx
for name in ('torch', 'transformers', 'huggingface_hub', 'datasets'):
    assert name not in sys.modules, f'{name} was imported'
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_interrupted_run_persists_all_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "experiments.phase1b_longctx.run_sweep", side_effect=KeyboardInterrupt
            ):
                exit_code = main(
                    [
                        "--materialized-draft-path",
                        "/models/draft",
                        "--govreport-jsonl",
                        "/data/govreport.jsonl",
                        "--output-dir",
                        directory,
                    ]
                )
            output = Path(directory)
            manifest = json.loads((output / "manifest.json").read_text())
            for filename in (
                "manifest.json",
                "workload-inventory.json",
                "samples.jsonl",
                "rounds.jsonl",
                "summary.json",
                "summary.csv",
                "environment.json",
                "source-provenance.json",
                "terminal.log",
            ):
                self.assertTrue((output / filename).is_file(), filename)
        self.assertEqual(exit_code, 130)
        self.assertEqual(manifest["status"], "error")
        self.assertEqual(manifest["error"]["type"], "KeyboardInterrupt")


if __name__ == "__main__":
    unittest.main()
