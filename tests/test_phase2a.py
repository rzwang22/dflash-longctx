from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase1.materialize import VENDOR_ROOT
from dflash.phase2a.nsight import (
    build_kernel_breakdown,
    read_nvtx_kernel_summary,
    read_nvtx_projection_summary,
    write_nsight_outputs,
)
from dflash.phase2a.rank import aggregate_rank_events, rank_bucket
from dflash.phase2a.verification import aggregate_verification_samples
from experiments.phase2a_profile import (
    RANK_SUMMARY_FIELDS,
    RUN_MODES,
    VERIFICATION_SUMMARY_FIELDS,
    _new_manifest,
    _rank_csv_rows,
    _require_token_exact,
    _verification_csv_rows,
    parse_args,
    run_configuration,
)


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
                "--output-dir",
                "/tmp/out",
            ]
        )
        self.assertEqual(overridden.run_mode, "verification-profile")
        self.assertEqual(overridden.context_lengths, (1024, 16384))
        self.assertEqual(overridden.timing_repeats, 2)

        schemas = sorted((REPO_ROOT / "docs/schemas").glob("phase2a-*.schema.json"))
        self.assertEqual(len(schemas), 8)
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
