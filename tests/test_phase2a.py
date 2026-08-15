from __future__ import annotations

import csv
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase1.materialize import VENDOR_ROOT
from dflash.phase1b.workload import (
    build_longbench_workloads,
    build_workload_inventory,
    prepare_longbench_records,
)
from dflash.phase2a.comparison import (
    COMPARISON_CSV_FIELDS,
    build_verification_length_comparison,
    write_verification_length_comparison,
)
from dflash.phase2a.nsight import (
    build_kernel_breakdown,
    read_nvtx_kernel_summary,
    read_nvtx_projection_summary,
    write_nsight_outputs,
)
from dflash.phase2a.rank import aggregate_rank_events, rank_bucket
from dflash.phase2a.verification import aggregate_verification_samples
from experiments.phase1b_longctx import _print_inventory
from experiments.phase2a_profile import (
    RANK_SUMMARY_FIELDS,
    RUN_MODES,
    VERIFICATION_SUMMARY_FIELDS,
    _load_inputs,
    _new_manifest,
    _rank_csv_rows,
    _require_token_exact,
    _verification_csv_rows,
    parse_args,
    run_configuration,
)
from scripts.phase2a_nsight import build_parser as build_nsight_parser
from scripts.phase2a_nsight import run_export, run_parse


class _CharacterTokenizer:
    eos_token_id = 0
    model_max_length = 100_000

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return list(range(3 + len(messages[0]["content"])))

    def encode(self, text, **kwargs):
        del kwargs
        return list(range(len(text)))


def _write_kernel_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["NVTX Range", "Kernel Name", "Kern Inst", "Total Time (ns)"])
        writer.writerow(["dflash.verify", "cutlass_gemm_kernel", 2, 100])


def _write_projection_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["Range", "Total Proj Time (ns)", "Range Instances", "Total GPU Ops"]
        )
        writer.writerow(["dflash.verify", 110, 1, 3])


def _load_observability_helper():
    path = VENDOR_ROOT / "instrumented" / "phase1_observability.py"
    spec = importlib.util.spec_from_file_location("phase2a_helper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rank_event(
    *,
    sample_id: str,
    round_index: int,
    block_position: int,
    category: str,
    rank: int,
) -> dict:
    return {
        "task": "gov_report",
        "sample_id": sample_id,
        "source_sample_id": sample_id,
        "nominal_prompt_length": 1024,
        "actual_prompt_length": 1022,
        "diagnostic_run": 0,
        "round_index": round_index,
        "block_position": block_position,
        "position_category": category,
        "accepted_prefix_length": 0,
        "target_token_id": 7,
        "draft_top1_token_id": 3 if rank > 1 else 7,
        "target_rank_in_draft": rank,
        "target_in_top_1": rank <= 1,
        "target_in_top_2": rank <= 2,
        "target_in_top_4": rank <= 4,
        "target_in_top_8": rank <= 8,
        "target_in_top_16": rank <= 16,
        "target_logit": 1.0,
        "draft_top1_logit": 2.0,
        "top1_minus_target_logit_margin": 1.0,
        "top_m_token_ids": [3, 7],
        "top_m_logits": [2.0, 1.0],
        "tree_accepted_length_derived": False,
    }


class _FakeEvent:
    def __init__(self, cuda, *, enable_timing: bool) -> None:
        self.cuda = cuda
        self.enable_timing = enable_timing
        self.timestamp = None

    def record(self) -> None:
        self.timestamp = self.cuda.clock
        self.cuda.clock += 1

    def elapsed_time(self, other) -> float:
        return float(other.timestamp - self.timestamp)


class _FakeHandle:
    def __init__(self, callbacks: list, callback) -> None:
        self.callbacks = callbacks
        self.callback = callback

    def remove(self) -> None:
        self.callbacks.remove(self.callback)


class _FakeModule:
    def __init__(self) -> None:
        self.pre_hooks = []
        self.post_hooks = []

    def register_forward_pre_hook(self, callback):
        self.pre_hooks.append(callback)
        return _FakeHandle(self.pre_hooks, callback)

    def register_forward_hook(self, callback, always_call=False):
        del always_call
        self.post_hooks.append(callback)
        return _FakeHandle(self.post_hooks, callback)

    def __call__(self):
        for callback in list(self.pre_hooks):
            callback(self, ())
        result = object()
        for callback in list(self.post_hooks):
            callback(self, (), result)
        return result


class _FakeNvtx:
    def __init__(self) -> None:
        self.stack = []

    def range_push(self, name: str) -> None:
        self.stack.append(name)

    def range_pop(self) -> None:
        self.stack.pop()


class _FakeCuda:
    def __init__(self) -> None:
        self.clock = 0
        self.synchronize_count = 0
        self.nvtx = _FakeNvtx()

    def Event(self, *, enable_timing: bool):
        return _FakeEvent(self, enable_timing=enable_timing)

    def synchronize(self, device) -> None:
        del device
        self.synchronize_count += 1


class Phase2ATests(unittest.TestCase):
    def test_rank_definition_ties_categories_and_bonus_exclusion(self) -> None:
        helper = _load_observability_helper()
        logits = [4.0, 4.0, 3.0, 4.0]
        self.assertEqual(helper.deterministic_rank_from_logits(logits, 0), 1)
        self.assertEqual(helper.deterministic_rank_from_logits(logits, 1), 2)
        self.assertEqual(helper.deterministic_rank_from_logits(logits, 3), 3)

        expected = [
            "accepted_prefix",
            "accepted_prefix",
            "first_rejection_frontier",
            "post_rejection_counterfactual",
        ]
        actual = [
            helper.rank_position_category(
                block_position=position,
                accepted_prefix_length=2,
                proposed_count=4,
            )
            for position in range(1, 5)
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(
            helper.rank_position_category(
                block_position=1,
                accepted_prefix_length=4,
                proposed_count=4,
            ),
            "full_block_accepted",
        )
        with self.assertRaisesRegex(ValueError, "proposed draft token"):
            helper.rank_position_category(
                block_position=16,
                accepted_prefix_length=15,
                proposed_count=15,
            )

    def test_rank_membership_histogram_micro_macro_and_no_tree_claim(self) -> None:
        events = [
            _rank_event(
                sample_id="a",
                round_index=0,
                block_position=1,
                category="first_rejection_frontier",
                rank=2,
            ),
            _rank_event(
                sample_id="a",
                round_index=1,
                block_position=1,
                category="first_rejection_frontier",
                rank=20,
            ),
            _rank_event(
                sample_id="b",
                round_index=0,
                block_position=1,
                category="first_rejection_frontier",
                rank=4,
            ),
            _rank_event(
                sample_id="a",
                round_index=2,
                block_position=1,
                category="full_block_accepted",
                rank=1,
            ),
        ]
        summary = aggregate_rank_events(events)
        overall = summary["overall"]
        micro = overall["first_rejection_frontier_top_m_recall_micro"]
        macro = overall["first_rejection_frontier_top_m_recall_macro"]
        self.assertAlmostEqual(micro["top_4_recall"], 2 / 3)
        self.assertAlmostEqual(macro["top_4_recall"], 0.75)
        self.assertEqual(overall["frontier_rank"]["histogram"]["rank_2"], 1)
        self.assertEqual(overall["frontier_rank"]["histogram"]["rank_3_4"], 1)
        self.assertEqual(overall["frontier_rank"]["histogram"]["rank_over_16"], 1)
        self.assertEqual(rank_bucket(8), "rank_5_8")
        self.assertFalse(summary["tree_accepted_length_derived"])
        self.assertFalse(
            overall["interpretation"]["frontier_top_1_is_recovery_metric"]
        )
        self.assertFalse(overall["interpretation"]["target_bonus_token_included"])
        csv_row = _rank_csv_rows(summary)[0]
        self.assertEqual(set(csv_row), set(RANK_SUMMARY_FIELDS))
        invalid_membership = dict(events[0], target_in_top_2=False)
        with self.assertRaisesRegex(ValueError, "target_in_top_2"):
            aggregate_rank_events([invalid_membership])

    def test_phase2a_mode_validation_and_control_token_exactness(self) -> None:
        helper = _load_observability_helper()
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            helper.validate_phase2a_mode_options(
                enable_rank_diagnostics=True,
                enable_verification_profiling=True,
                enable_profiling=True,
                return_diagnostics=True,
                temperature=0.0,
                rank_top_ms=[2, 4],
                rank_max_m=16,
            )
        with self.assertRaisesRegex(ValueError, "temperature=0"):
            helper.validate_phase2a_mode_options(
                enable_rank_diagnostics=True,
                enable_verification_profiling=False,
                enable_profiling=False,
                return_diagnostics=True,
                temperature=0.5,
                rank_top_ms=[2, 4],
                rank_max_m=16,
            )

        fake_torch = SimpleNamespace(equal=lambda left, right: left == right)
        for mode in RUN_MODES:
            _require_token_exact(fake_torch, [1, 2], [1, 2], label=mode)
            with self.assertRaisesRegex(RuntimeError, "changed output token IDs"):
                _require_token_exact(fake_torch, [1, 2], [1, 3], label=mode)

    def test_module_hooks_are_deferred_and_verification_aggregation_excludes_prefill(self) -> None:
        helper = _load_observability_helper()
        cuda = _FakeCuda()
        torch_module = SimpleNamespace(cuda=cuda)
        layer = SimpleNamespace(self_attn=_FakeModule(), mlp=_FakeModule())
        target = SimpleNamespace(
            model=SimpleNamespace(layers=[layer]), lm_head=_FakeModule()
        )
        timer = helper.DeferredCudaTimer(
            torch_module, enabled=True, device="cuda:0"
        )
        records = []
        with helper.TargetModuleCudaProfiler(
            torch_module,
            timer=timer,
            target=target,
            stage="verification",
            round_index=0,
            records=records,
            enabled=True,
        ):
            layer.self_attn()
            layer.mlp()
            target.lm_head()

        self.assertEqual(cuda.synchronize_count, 0)
        self.assertEqual({record["stage"] for record in records}, {"verification"})
        timer.resolve()
        self.assertEqual(cuda.synchronize_count, 1)
        helper.resolve_target_module_records(timer, records)
        module_summary = helper.summarize_target_module_records(
            records,
            total_cuda_ms=3.0,
            target_layer_count=1,
            invocation_count=1,
        )
        self.assertTrue(module_summary["partition_reliable"])
        self.assertEqual(module_summary["target_unattributed_cuda_ms"], 0.0)
        self.assertFalse(module_summary["pure_sdpa_isolated"])

        samples = [
            {
                "task": "gov_report",
                "sample_id": "a",
                "nominal_prompt_length": 1024,
                "partition_reliable": True,
                "verification_cuda_ms": 3.0,
                "self_attention_module_cuda_ms": 1.0,
                "mlp_module_cuda_ms": 1.0,
                "lm_head_cuda_ms": 1.0,
                "unattributed_cuda_ms": 0.0,
                "self_attention_over_verification": 1 / 3,
                "mlp_over_verification": 1 / 3,
                "lm_head_over_verification": 1 / 3,
                "draft_cuda_ms": 2.0,
                "draft_over_round": 0.4,
                "verification_over_round": 0.6,
            }
        ]
        common = {
            "task": "gov_report",
            "nominal_prompt_length": 1024,
            "layer_index": 0,
            "module_kind": "self_attn",
        }
        aggregate = aggregate_verification_samples(
            samples,
            [
                {**common, "stage": "prefill", "cuda_elapsed_ms": 99.0},
                {**common, "stage": "verification", "cuda_elapsed_ms": 1.0},
            ],
        )
        layer_profile = next(iter(aggregate["by_layer"].values()))
        self.assertEqual(layer_profile["cuda_elapsed_ms"]["mean"], 1.0)
        csv_row = _verification_csv_rows(aggregate)[0]
        self.assertEqual(set(csv_row), set(VERIFICATION_SUMMARY_FIELDS))

    def test_nsight_parser_preserves_names_rules_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kernel_csv = root / "nvtx_kern_sum.csv"
            projection_csv = root / "nvtx_gpu_proj_sum.csv"
            with kernel_csv.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["NVTX Range", "Kernel Name", "Kern Inst", "Total Time (ns)"]
                )
                writer.writerow(["dflash.verify", "cutlass_gemm_kernel", 2, 100])
                writer.writerow(
                    ["target.verify.layer.0.self_attn", "flash_fwd", 1, 60]
                )
                writer.writerow(
                    [
                        "dflash.attention.combined_context_block",
                        "scaled_dot_product_attention",
                        1,
                        40,
                    ]
                )
            with projection_csv.open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "Range",
                        "Total Proj Time (ns)",
                        "Range Instances",
                        "Total GPU Ops",
                    ]
                )
                writer.writerow(["dflash.verify", 110, 1, 3])
                writer.writerow(["target.verify.layer.0.self_attn", 60, 1, 1])

            kernel_rows = read_nvtx_kernel_summary(kernel_csv)
            nvtx_rows = read_nvtx_projection_summary(projection_csv)
            breakdown = build_kernel_breakdown(kernel_rows, nvtx_rows)
            self.assertEqual(kernel_rows[0]["raw_kernel_name"], "cutlass_gemm_kernel")
            self.assertEqual(kernel_rows[0]["classified_category"], "gemm_or_matmul")
            self.assertEqual(kernel_rows[1]["classification_rule"], "attention-name-v1")
            self.assertEqual(
                breakdown["scope_totals"]["complete_verification"][
                    "over_complete_verification_kernel_time"
                ],
                1.0,
            )
            self.assertEqual(
                breakdown["kernel_rows"][1][
                    "over_complete_verification_kernel_time"
                ],
                0.6,
            )
            self.assertTrue(
                breakdown["classification_semantics"]["requires_h100_review"]
            )
            paths = write_nsight_outputs(
                root / "parsed", kernel_rows=kernel_rows, nvtx_rows=nvtx_rows
            )
            self.assertEqual(
                {path.name for path in paths.values()},
                {
                    "nsight-kernel-summary.csv",
                    "nsight-nvtx-summary.csv",
                    "verification-kernel-breakdown.json",
                },
            )
            self.assertTrue(all(path.is_file() for path in paths.values()))

    def test_nsight_parse_kernel_only_is_truthful_and_has_all_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kernel_csv = root / "input_nvtx_kern_sum.csv"
            _write_kernel_csv(kernel_csv)
            args = build_nsight_parser().parse_args(
                [
                    "parse",
                    "--nvtx-kernel-csv",
                    str(kernel_csv),
                    "--output-dir",
                    str(root / "parsed"),
                ]
            )
            paths = run_parse(args, command="test parse")
            self.assertEqual(len(paths), 4)
            manifest = json.loads(
                paths["nsight_export_manifest_json"].read_text(encoding="utf-8")
            )
            breakdown = json.loads(
                paths["verification_kernel_breakdown_json"].read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["analysis_mode"], "kernel_only")
            self.assertEqual(manifest["projection_report_status"], "unavailable")
            self.assertEqual(manifest["source_reports"], ["nvtx_kern_sum"])
            self.assertEqual(breakdown["nvtx_projection_rows"], [])
            self.assertNotIn("projected_cuda_duration_ns", json.dumps(breakdown))
            manifest_schema = json.loads(
                (
                    REPO_ROOT
                    / "docs/schemas/phase2a-nsight-export-manifest.schema.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(set(manifest_schema["required"]).issubset(manifest))
            self.assertEqual(
                paths["nvtx_summary_csv"].read_text(encoding="utf-8").count("\n"),
                1,
            )

    def test_nsight_export_new_and_2023_report_compatibility(self) -> None:
        def run_case(*, projection_available: bool) -> dict:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            root = Path(temporary.name)
            report = root / "trace.nsys-rep"
            report.touch()
            output = root / "parsed"

            def fake_run(command):
                if command == ["nsys", "--version"]:
                    return subprocess.CompletedProcess(
                        command, 0, "NVIDIA Nsight Systems 2023.1.2\n", ""
                    )
                if command == ["nsys", "stats", "--help-reports"]:
                    reports = "  nvtx_kern_sum - kernels\n"
                    if projection_available:
                        reports += "  nvtx_gpu_proj_sum - projected ranges\n"
                    return subprocess.CompletedProcess(command, 0, reports, "")
                report_name = command[command.index("--report") + 1]
                prefix = Path(command[command.index("--output") + 1])
                csv_path = prefix.parent / f"{prefix.name}_{report_name}.csv"
                if report_name == "nvtx_kern_sum":
                    _write_kernel_csv(csv_path)
                else:
                    _write_projection_csv(csv_path)
                return subprocess.CompletedProcess(command, 0, "", "")

            args = build_nsight_parser().parse_args(
                [
                    "export",
                    "--nsys-rep",
                    str(report),
                    "--output-dir",
                    str(output),
                ]
            )
            with patch("scripts.phase2a_nsight._run_capture", side_effect=fake_run):
                paths = run_export(args, command="test export")
            return json.loads(
                paths["nsight_export_manifest_json"].read_text(encoding="utf-8")
            )

        full = run_case(projection_available=True)
        self.assertEqual(full["analysis_mode"], "full")
        self.assertEqual(
            full["succeeded_reports"], ["nvtx_kern_sum", "nvtx_gpu_proj_sum"]
        )
        self.assertEqual(len(full["report_attempts"]), 2)
        kernel_only = run_case(projection_available=False)
        self.assertEqual(kernel_only["analysis_mode"], "kernel_only")
        self.assertEqual(kernel_only["projection_report_status"], "unavailable")
        self.assertEqual(len(kernel_only["report_attempts"]), 1)

    def test_optional_projection_failure_does_not_discard_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "trace.nsys-rep"
            report.touch()

            def fake_run(command):
                if command == ["nsys", "--version"]:
                    return subprocess.CompletedProcess(command, 0, "nsys 2024.1", "")
                if command == ["nsys", "stats", "--help-reports"]:
                    return subprocess.CompletedProcess(
                        command, 0, "nvtx_kern_sum x\nnvtx_gpu_proj_sum x", ""
                    )
                report_name = command[command.index("--report") + 1]
                prefix = Path(command[command.index("--output") + 1])
                if report_name == "nvtx_kern_sum":
                    _write_kernel_csv(
                        prefix.parent / f"{prefix.name}_{report_name}.csv"
                    )
                    return subprocess.CompletedProcess(command, 0, "", "")
                return subprocess.CompletedProcess(command, 1, "", "unsupported")

            args = build_nsight_parser().parse_args(
                [
                    "export",
                    "--nsys-rep",
                    str(report),
                    "--output-dir",
                    str(root / "parsed"),
                ]
            )
            with patch("scripts.phase2a_nsight._run_capture", side_effect=fake_run):
                paths = run_export(args, command="test optional failure")
            manifest = json.loads(
                paths["nsight_export_manifest_json"].read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "ok")
            self.assertEqual(manifest["projection_report_status"], "failed")
            self.assertEqual(manifest["failed_reports"], ["nvtx_gpu_proj_sum"])

    def test_missing_required_nsight_kernel_has_diagnostic_error_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "trace.nsys-rep"
            report.touch()

            def fake_run(command):
                if command == ["nsys", "--version"]:
                    return subprocess.CompletedProcess(command, 0, "nsys 2023.1", "")
                if command == ["nsys", "stats", "--help-reports"]:
                    return subprocess.CompletedProcess(
                        command, 0, "nvtx_kern_sum x", ""
                    )
                return subprocess.CompletedProcess(command, 1, "", "export failed")

            output = root / "parsed"
            args = build_nsight_parser().parse_args(
                [
                    "export",
                    "--nsys-rep",
                    str(report),
                    "--output-dir",
                    str(output),
                ]
            )
            with (
                patch("scripts.phase2a_nsight._run_capture", side_effect=fake_run),
                self.assertRaisesRegex(
                    RuntimeError, "nsys_version=.*available_reports=.*checked_paths"
                ),
            ):
                run_export(args, command="test required failure")
            manifest = json.loads(
                (output / "nsight-export-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "error")
            self.assertIn("nvtx_kern_sum", manifest["failed_reports"])

    def test_explicit_sample_selection_is_exact_nested_and_validated(self) -> None:
        tokenizer = _CharacterTokenizer()
        prepared = prepare_longbench_records(
            [
                {"_id": "short", "context": "s" * 300, "input": "question"},
                {"_id": "long", "context": "l" * 900, "input": "question"},
            ],
            tokenizer=tokenizer,
            task="qasper",
        )
        selected_id = next(
            row["sample_id"] for row in prepared if row["source_sample_id"] == "long"
        )
        workloads, cohort = build_longbench_workloads(
            prepared,
            tokenizer=tokenizer,
            task="qasper",
            nominal_lengths=(256, 512),
            sample_count=8,
            sample_policy="matched",
            max_new_tokens=8,
            model_position_limit=1024,
            sample_id=selected_id,
        )
        self.assertEqual({row["sample_id"] for row in workloads}, {selected_id})
        self.assertEqual(cohort["explicit_sample_id"], selected_id)
        self.assertEqual(cohort["requested_sample_count"], 1)
        self.assertLess(
            workloads[0]["context_prefix_characters"],
            workloads[1]["context_prefix_characters"],
        )
        self.assertEqual(
            {row["explicit_sample_id"] for row in workloads}, {selected_id}
        )

    def test_explicit_sample_missing_and_too_short_errors(self) -> None:
        tokenizer = _CharacterTokenizer()
        prepared = prepare_longbench_records(
            [{"_id": "short", "context": "s" * 300, "input": "question"}],
            tokenizer=tokenizer,
            task="qasper",
        )
        common = {
            "prepared": prepared,
            "tokenizer": tokenizer,
            "task": "qasper",
            "nominal_lengths": (256,),
            "sample_count": 1,
            "sample_policy": "matched",
            "max_new_tokens": 8,
            "model_position_limit": 1024,
        }
        with self.assertRaisesRegex(RuntimeError, "does not exist"):
            build_longbench_workloads(**common, sample_id="qasper-missing")
        selected_id = prepared[0]["sample_id"]
        common["nominal_lengths"] = (256, 512)
        with self.assertRaisesRegex(
            RuntimeError, rf"{selected_id} reaches at most \d+.*512"
        ):
            build_longbench_workloads(**common, sample_id=selected_id)

    def test_explicit_sample_preflight_fails_before_cuda_or_model_load(self) -> None:
        tokenizer = _CharacterTokenizer()
        prepared = prepare_longbench_records(
            [{"_id": "short", "context": "s" * 300, "input": "question"}],
            tokenizer=tokenizer,
            task="qasper",
        )
        sample_id = prepared[0]["sample_id"]
        inventory = build_workload_inventory(
            prepared,
            task="qasper",
            nominal_lengths=(256, 512),
            requested_sample_count=1,
        )
        inventory.update(
            {
                "source_record_count": 1,
                "skipped_record_count": 0,
                "models_loaded": False,
                "cuda_required": False,
            }
        )
        args = parse_args(
            [
                "--dataset-jsonl",
                "/tmp/qasper.jsonl",
                "--longbench-task",
                "qasper",
                "--materialized-draft-path",
                "/tmp/draft",
                "--sample-id",
                sample_id,
                "--context-lengths",
                "256,512",
                "--pilot-samples",
                "1",
            ]
        )
        manifest = _new_manifest(
            args,
            {"repository": {"commit_sha": "abc", "branch": "test"}},
            "preflight test",
        )
        with (
            patch(
                "experiments.phase2a_profile._load_tokenizer_only",
                return_value=(tokenizer, {}, "test"),
            ),
            patch(
                "experiments.phase2a_profile._prepare_workload",
                return_value=(prepared, inventory, None, None),
            ),
            patch("experiments.phase2a_profile._require_cuda") as require_cuda,
            self.assertRaisesRegex(RuntimeError, "reaches at most"),
        ):
            _load_inputs(args, manifest, {})
        require_cuda.assert_not_called()
        selection = manifest["workload"]["inventory"]["explicit_sample_selection"]
        self.assertEqual(selection["requested_sample_id"], sample_id)
        self.assertFalse(selection["validated"])
        self.assertGreater(selection["actual_max_templated_prompt_length"], 0)

    def test_inventory_log_truncates_but_inventory_keeps_every_id(self) -> None:
        tokenizer = _CharacterTokenizer()
        prepared = prepare_longbench_records(
            [
                {"_id": f"sample-{index}", "context": "x" * 600, "input": "q"}
                for index in range(12)
            ],
            tokenizer=tokenizer,
            task="qasper",
        )
        inventory = build_workload_inventory(
            prepared,
            task="qasper",
            nominal_lengths=(256,),
            requested_sample_count=8,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            _print_inventory(inventory)
        self.assertIn("total=12 showing=10 omitted=2", output.getvalue())
        self.assertEqual(len(inventory["sample_ids_reaching_maximum_length"]), 12)
        self.assertEqual(
            len(
                json.loads(json.dumps(inventory))["sample_ids_reaching_maximum_length"]
            ),
            12,
        )

    def test_verification_length_comparison_uses_true_counts_and_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write_artifacts(length: int, source_id: str) -> Path:
                output = root / str(length)
                output.mkdir()
                metrics = {
                    field: {"median": float(length // 1024 + index + 1)}
                    for index, field in enumerate(VERIFICATION_SUMMARY_FIELDS[3:])
                }
                summary = {
                    "status": "ok",
                    "run_mode": "verification-profile",
                    "aggregates": {
                        "by_task_and_context_length": {
                            f"gov_report:{length}": {
                                "task": "gov_report",
                                "context_length": length,
                                "sample_count": 1,
                                "metrics": metrics,
                            }
                        }
                    },
                }
                summary_path = output / "verification-profile-summary.json"
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                (output / "samples.jsonl").write_text(
                    json.dumps(
                        {
                            "task": "gov_report",
                            "sample_id": f"sample-{source_id}",
                            "source_sample_id": source_id,
                            "nominal_prompt_length": length,
                            "actual_prompt_length": length - 1,
                            "generated_token_count": 8,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                rounds = [
                    {
                        "task": "gov_report",
                        "sample_id": f"sample-{source_id}",
                        "nominal_prompt_length": length,
                        "diagnostic_run": 0,
                        "round_index": index,
                        "stop_reason": "continue" if index == 0 else "max_new_tokens",
                        "truncated_token_count": 0 if index == 0 else 1,
                        "committed_token_count": 4,
                        "returned_token_delta": 4,
                    }
                    for index in range(2)
                ]
                (output / "verification-profile-rounds.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rounds),
                    encoding="utf-8",
                )
                return summary_path

            paired_paths = [
                write_artifacts(length, "same") for length in (1024, 16384, 32768)
            ]
            comparison = build_verification_length_comparison(paired_paths)
            self.assertTrue(comparison["paired_comparison"])
            comparison_schema = json.loads(
                (
                    REPO_ROOT
                    / "docs/schemas/phase2a-verification-length-comparison.schema.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(
                set(comparison_schema["required"]).issubset(comparison)
            )
            self.assertEqual(comparison["rows"][0]["round_count"], 2)
            self.assertEqual(comparison["rows"][0]["non_terminal_round_count"], 1)
            self.assertEqual(comparison["rows"][0]["total_committed_token_count"], 8)
            self.assertEqual(
                comparison["scaling_comparisons"]["1k_to_32k"]["round_count_scaling"],
                1.0,
            )
            paths = write_verification_length_comparison(
                root / "comparison", comparison
            )
            with paths["comparison_csv"].open(encoding="utf-8") as stream:
                header = next(csv.reader(stream))
            self.assertEqual(tuple(header), COMPARISON_CSV_FIELDS)

            unpaired_path = write_artifacts(4096, "different")
            unpaired = build_verification_length_comparison(
                [paired_paths[0], unpaired_path]
            )
            self.assertFalse(unpaired["paired_comparison"])
            self.assertTrue(
                any("pilot trend" in warning for warning in unpaired["warnings"])
            )

    def test_cli_defaults_overrides_schemas_and_lazy_imports(self) -> None:
        defaults = parse_args(
            [
                "--dataset-jsonl",
                "/tmp/gov_report.jsonl",
                "--materialized-draft-path",
                "/tmp/draft",
            ]
        )
        self.assertEqual(defaults.run_mode, "rank")
        self.assertEqual(defaults.rank_top_ms, (2, 4, 8, 16))
        self.assertEqual(defaults.rank_max_m, 16)
        self.assertEqual(defaults.temperature, 0.0)
        self.assertEqual(defaults.block_size, 16)
        self.assertEqual(defaults.batch_size, 1)
        self.assertEqual(run_configuration(defaults)["phase"], "2A")
        manifest = _new_manifest(
            defaults,
            {"repository": {"commit_sha": "abc", "branch": "test"}},
            "phase2a command",
        )
        self.assertEqual(
            manifest["workload"]["prompt_template"]["id"],
            "govreport-summary-v1",
        )
        self.assertFalse(
            manifest["measurement_policy"]["module_self_attention_is_pure_sdpa"]
        )

        overridden = parse_args(
            [
                "--dataset-jsonl",
                "/tmp/qasper.jsonl",
                "--longbench-task",
                "qasper",
                "--materialized-draft-path",
                "/tmp/draft",
                "--run-mode",
                "verification-profile",
                "--context-lengths",
                "1K,16K",
                "--pilot-samples",
                "1",
                "--timing-repeats",
                "2",
                "--sample-id",
                "qasper-fixed",
                "--output-dir",
                "/tmp/out",
            ]
        )
        self.assertEqual(overridden.run_mode, "verification-profile")
        self.assertEqual(overridden.context_lengths, (1024, 16384))
        self.assertEqual(overridden.timing_repeats, 2)
        self.assertEqual(overridden.sample_id, "qasper-fixed")
        self.assertEqual(run_configuration(overridden)["effective_sample_count"], 1)

        schemas = sorted((REPO_ROOT / "docs/schemas").glob("phase2a-*.schema.json"))
        self.assertEqual(len(schemas), 10)
        for path in schemas:
            schema = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertEqual(schema["type"], "object")
            self.assertTrue(schema["required"])
        manifest_schema = json.loads(
            (REPO_ROOT / "docs/schemas/phase2a-manifest.schema.json").read_text()
        )
        self.assertTrue(set(manifest_schema["required"]).issubset(manifest))
        event_schema = json.loads(
            (REPO_ROOT / "docs/schemas/phase2a-rank-event.schema.json").read_text()
        )
        self.assertTrue(
            set(event_schema["required"]).issubset(
                _rank_event(
                    sample_id="schema",
                    round_index=0,
                    block_position=1,
                    category="first_rejection_frontier",
                    rank=2,
                )
            )
        )

        code = """
import sys
import dflash.phase2a
import experiments.phase2a_profile
import scripts.phase2a_nsight
import scripts.phase2a_compare_verification
for name in ('torch', 'transformers', 'huggingface_hub'):
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


if __name__ == "__main__":
    unittest.main()
