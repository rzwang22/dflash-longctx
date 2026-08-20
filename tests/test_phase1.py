from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase1.materialize import (
    DRAFT_REVISION,
    VENDOR_ROOT,
    dynamic_module_dependency_record,
    materialize_draft,
    verify_materialization,
)
from dflash.phase1.source import locate_callable
from experiments.phase1_observability import main, new_summary, parse_args
from scripts.collect_environment import collect_environment

OBSERVABILITY_HELPER = VENDOR_ROOT / "instrumented" / "phase1_observability.py"


def _load_observability_helper():
    spec = importlib.util.spec_from_file_location(
        "phase1_observability_test_helper", OBSERVABILITY_HELPER
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_weight_source(root: Path) -> Path:
    source = root / "weight-source"
    source.mkdir()
    provenance = json.loads(
        (VENDOR_ROOT / "provenance.json").read_text(encoding="utf-8")
    )
    for filename in provenance["original_files"]:
        content = (VENDOR_ROOT / "original" / filename).read_bytes()
        if filename == "utils.py":
            content = content.rstrip(b"\n")
        (source / filename).write_bytes(content)

    weight = source / "model.safetensors"
    expected_weight = provenance["weight_files_not_vendored"][weight.name]
    with weight.open("wb") as stream:
        stream.write(b"phase1-test-weight")
        stream.truncate(expected_weight["size_bytes"])
    metadata = source / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    (metadata / "model.safetensors.metadata").write_text(
        f"{DRAFT_REVISION}\n{expected_weight['git_lfs_oid_sha256']}\n0\n",
        encoding="utf-8",
    )
    return source


def _find_method(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f"Could not find {class_name}.{method_name}")


class Phase1Tests(unittest.TestCase):
    def test_cli_parsing(self) -> None:
        args = parse_args(
            [
                "--materialized-draft-path",
                "/models/instrumented",
                "--target-model-path",
                "/models/target",
                "--draft-model-path",
                "/models/original",
                "--warmup-runs",
                "2",
                "--measurement-runs",
                "3",
                "--enable-diagnostics",
                "--enable-profiling",
                "--run-mode",
                "ab",
            ]
        )
        self.assertEqual(args.materialized_draft_path, Path("/models/instrumented"))
        self.assertEqual(args.warmup_runs, 2)
        self.assertEqual(args.measurement_runs, 3)
        self.assertTrue(args.enable_diagnostics)
        self.assertTrue(args.enable_profiling)

    def test_materialization_is_idempotent_and_hash_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _make_weight_source(root)
            output = root / "instrumented"
            first = materialize_draft(
                source_model_path=source,
                output_dir=output,
                implementation="instrumented",
            )
            second = materialize_draft(
                source_model_path=source,
                output_dir=output,
                implementation="instrumented",
            )
            verified = verify_materialization(
                output, expected_implementation="instrumented"
            )

            self.assertEqual(first["materialization_id"], second["materialization_id"])
            self.assertEqual(first["created_at_utc"], second["created_at_utc"])
            self.assertEqual(verified["base_model_revision"], DRAFT_REVISION)
            self.assertTrue((output / "model.safetensors").is_symlink())
            self.assertEqual(
                (output / "model.safetensors").resolve(),
                (source / "model.safetensors").resolve(),
            )
            self.assertIn(
                "phase1_observability.py", verified["effective_custom_code_sha256"]
            )
            self.assertIn("phase2c_online.py", verified["effective_custom_code_sha256"])
            self.assertIn("phase2c_gold.py", verified["effective_custom_code_sha256"])
            self.assertIn(
                "phase2d2_refinement.py",
                verified["effective_custom_code_sha256"],
            )
            self.assertIn(
                "phase2d2_tensor.py", verified["effective_custom_code_sha256"]
            )
            self.assertIn(
                "phase2d2_robustness.py",
                verified["effective_custom_code_sha256"],
            )
            dependencies = verified["dynamic_module_dependencies"]
            self.assertEqual(
                dependencies,
                dynamic_module_dependency_record(output),
            )
            self.assertTrue(
                {
                    "phase2d2_refinement.py",
                    "phase2d2_tensor.py",
                    "phase2d2_robustness.py",
                }.issubset(dependencies["relative_import_files"])
            )
            self.assertNotIn("refinement.py", dependencies["relative_import_files"])
            with self.assertRaisesRegex(RuntimeError, "separate output directory"):
                materialize_draft(
                    source_model_path=source,
                    output_dir=output,
                    implementation="original",
                )

            (output / "dflash.py").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash"):
                verify_materialization(output)

    def test_dynamic_module_resolution_rejects_inactive_missing_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dflash.py").write_text(
                "from .phase2d2_tensor import helper\n", encoding="utf-8"
            )
            (root / "phase2d2_tensor.py").write_text(
                "try:\n"
                "    from .refinement import helper\n"
                "except ImportError:\n"
                "    from .phase2d2_refinement import helper\n",
                encoding="utf-8",
            )
            (root / "phase2d2_refinement.py").write_text(
                "helper = object()\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "refinement.py"):
                dynamic_module_dependency_record(root)

    def test_original_dynamic_module_closure_has_no_phase2d2_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _make_weight_source(root)
            output = root / "original"
            manifest = materialize_draft(
                source_model_path=source,
                output_dir=output,
                implementation="original",
            )
            dependencies = manifest["dynamic_module_dependencies"]
            self.assertIn("utils.py", dependencies["relative_import_files"])
            self.assertFalse(
                any(
                    filename.startswith("phase2")
                    for filename in dependencies["relative_import_files"]
                )
            )

    def test_legacy_manifest_without_dependency_record_still_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _make_weight_source(root)
            output = root / "instrumented"
            manifest = materialize_draft(
                source_model_path=source,
                output_dir=output,
                implementation="instrumented",
            )
            manifest.pop("dynamic_module_dependencies")
            (output / "materialization.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            verified = verify_materialization(
                output, expected_implementation="instrumented"
            )
            self.assertNotIn("dynamic_module_dependencies", verified)

    def test_real_auto_model_resolves_materialized_phase2d2_offline(self) -> None:
        try:
            import datasets  # noqa: F401
            import torch  # noqa: F401
            import transformers
            from transformers import AutoModel
            from transformers.modeling_utils import PreTrainedModel
        except ImportError:
            self.skipTest(
                "Torch/Transformers/datasets unavailable for offline AutoModel test"
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _make_weight_source(root)
            output = root / "instrumented"
            materialize_draft(
                source_model_path=source,
                output_dir=output,
                implementation="instrumented",
            )
            sentinel = object()
            module_cache = root / "hf-modules"
            with (
                patch.dict(
                    os.environ,
                    {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                ),
                patch.object(
                    transformers.dynamic_module_utils,
                    "HF_MODULES_CACHE",
                    str(module_cache),
                ),
                patch.object(
                    PreTrainedModel,
                    "from_pretrained",
                    return_value=sentinel,
                ) as from_pretrained,
            ):
                loaded = AutoModel.from_pretrained(
                    str(output),
                    trust_remote_code=True,
                    local_files_only=True,
                )
            self.assertIs(loaded, sentinel)
            self.assertTrue(from_pretrained.called)
            cached_files = {path.name for path in module_cache.rglob("*.py")}
            self.assertTrue(
                {
                    "phase2d2_refinement.py",
                    "phase2d2_tensor.py",
                    "phase2d2_robustness.py",
                }.issubset(cached_files)
            )

    def test_materialization_rejects_custom_code_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _make_weight_source(root)
            (source / "dflash.py").write_text("not pinned\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not match pinned"):
                materialize_draft(
                    source_model_path=source,
                    output_dir=root / "output",
                    implementation="original",
                )

    def test_unwrap_locates_definition_not_decorator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "decorator_module.py").write_text(
                "from functools import wraps\n"
                "def decorate(function):\n"
                "    @wraps(function)\n"
                "    def wrapper(*args, **kwargs):\n"
                "        return function(*args, **kwargs)\n"
                "    return wrapper\n",
                encoding="utf-8",
            )
            (root / "model_module.py").write_text(
                "from decorator_module import decorate\n"
                "class Model:\n"
                "    @decorate\n"
                "    def spec_generate(self):\n"
                "        return 1\n",
                encoding="utf-8",
            )
            sys.path.insert(0, str(root))
            try:
                spec = importlib.util.spec_from_file_location(
                    "model_module", root / "model_module.py"
                )
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
                location = locate_callable(module.Model().spec_generate)
            finally:
                sys.path.remove(str(root))
                sys.modules.pop("decorator_module", None)

        self.assertTrue(location["was_unwrapped"])
        self.assertTrue(
            location["wrapped"]["definition_file"].endswith("decorator_module.py")
        )
        self.assertTrue(
            location["unwrapped"]["definition_file"].endswith("model_module.py")
        )
        self.assertEqual(location["unwrapped"]["qualified_name"], "Model.spec_generate")

    def test_default_reference_ast_and_return_compatibility(self) -> None:
        original_tree = ast.parse(
            (VENDOR_ROOT / "original" / "dflash.py").read_text(encoding="utf-8")
        )
        instrumented_tree = ast.parse(
            (VENDOR_ROOT / "instrumented" / "dflash.py").read_text(encoding="utf-8")
        )
        original = _find_method(original_tree, "DFlashDraftModel", "spec_generate")
        reference = _find_method(
            instrumented_tree, "DFlashDraftModel", "_spec_generate_reference"
        )
        original_body = original.body
        reference_body = reference.body[1:]  # Skip the instrumentation-only docstring.
        self.assertEqual(
            ast.dump(ast.Module(body=original_body, type_ignores=[])),
            ast.dump(ast.Module(body=reference_body, type_ignores=[])),
        )

        helper = _load_observability_helper()
        output = object()
        self.assertIs(
            helper.finalize_generation_result(
                output, {"rounds": []}, return_diagnostics=False
            ),
            output,
        )
        diagnostic_result = helper.finalize_generation_result(
            output, {"rounds": []}, return_diagnostics=True
        )
        self.assertIs(diagnostic_result.output_ids, output)

    def test_token_accounting_conservation_and_boundaries(self) -> None:
        helper = _load_observability_helper()
        rejected = helper.build_round_accounting(
            round_index=0,
            context_length_before=10,
            prompt_length=10,
            max_new_tokens=32,
            block_size=16,
            accepted_draft_tokens=0,
            returned_tokens_before=1,
            eos_generated_count=None,
        )
        self.assertEqual(rejected["accepted_draft_token_count"], 0)
        self.assertEqual(rejected["committed_token_count"], 1)
        self.assertEqual(rejected["correction_token_count"], 1)
        self.assertEqual(rejected["rejected_draft_token_count"], 15)
        self.assertTrue(all(rejected["conservation"].values()))

        full = helper.build_round_accounting(
            round_index=1,
            context_length_before=41,
            prompt_length=10,
            max_new_tokens=32,
            block_size=16,
            accepted_draft_tokens=15,
            returned_tokens_before=32,
            eos_generated_count=None,
        )
        self.assertEqual(full["bonus_token_count"], 1)
        self.assertEqual(full["committed_token_count"], 16)
        self.assertEqual(full["cumulative_generated_token_count"], 32)
        self.assertGreater(full["truncated_token_count"], 0)
        self.assertEqual(full["stop_reason"], "max_new_tokens")

        eos = helper.build_round_accounting(
            round_index=0,
            context_length_before=10,
            prompt_length=10,
            max_new_tokens=32,
            block_size=16,
            accepted_draft_tokens=3,
            returned_tokens_before=1,
            eos_generated_count=3,
        )
        self.assertEqual(eos["cumulative_generated_token_count"], 3)
        self.assertEqual(eos["stop_reason"], "eos")

        round_schema = json.loads(
            (REPO_ROOT / "docs" / "schemas" / "phase1-round.schema.json").read_text(
                encoding="utf-8"
            )
        )
        eos.update(
            {
                "measurement_run": 0,
                "draft_input_shapes": {},
                "draft_output_shapes": {},
                "target_verification_input_shapes": {},
                "target_verification_output_shapes": {},
                "attention_calls": [],
                "target_attention": {},
                "timing": {},
                "gpu_memory": None,
            }
        )
        self.assertTrue(set(round_schema["required"]).issubset(eos))

    def test_summary_schema_and_interrupted_persistence(self) -> None:
        args = parse_args(["--materialized-draft-path", "/models/instrumented"])
        environment = collect_environment(
            run_config={}, run_command="phase1 test", import_torch=False
        )
        summary = new_summary(args, environment, "phase1 test")
        schema = json.loads(
            (REPO_ROOT / "docs" / "schemas" / "phase1-summary.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(schema["properties"]["schema_version"]["const"], "1.0")
        self.assertTrue(set(schema["required"]).issubset(summary))

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "experiments.phase1_observability.run_phase1",
                side_effect=KeyboardInterrupt,
            ):
                exit_code = main(
                    [
                        "--materialized-draft-path",
                        "/models/instrumented",
                        "--output-dir",
                        directory,
                    ]
                )
            persisted = json.loads(
                (Path(directory) / "summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue((Path(directory) / "rounds.jsonl").is_file())
            self.assertTrue((Path(directory) / "source-provenance.json").is_file())
        self.assertEqual(exit_code, 130)
        self.assertEqual(persisted["error"]["type"], "KeyboardInterrupt")

    def test_cuda_unavailable_is_an_explicit_persisted_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "experiments.phase1_observability._require_cuda",
                side_effect=RuntimeError("CUDA unavailable for test"),
            ):
                exit_code = main(
                    [
                        "--materialized-draft-path",
                        "/models/instrumented",
                        "--output-dir",
                        directory,
                    ]
                )
            persisted = json.loads(
                (Path(directory) / "summary.json").read_text(encoding="utf-8")
            )
        self.assertEqual(exit_code, 1)
        self.assertEqual(persisted["error"]["type"], "RuntimeError")
        self.assertIn("CUDA unavailable", persisted["error"]["message"])

    def test_imports_do_not_load_or_download_models(self) -> None:
        code = """
import sys
import dflash.phase1
import experiments.phase1_observability
import scripts.materialize_dflash
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

    def test_repository_contains_no_weights_or_server_path_literals(self) -> None:
        forbidden_suffixes = {".safetensors", ".bin", ".pt", ".pth"}
        candidate_files = [
            *REPO_ROOT.glob("dflash/phase1/**/*.py"),
            *REPO_ROOT.glob("experiments/phase1*.py"),
            *REPO_ROOT.glob("scripts/materialize_dflash.py"),
            *REPO_ROOT.glob("third_party/qwen3_8b_dflash_b16/**/*"),
        ]
        self.assertFalse(
            [path for path in candidate_files if path.suffix in forbidden_suffixes]
        )
        forbidden_literals = ("/" + "mnt/", "/" + "home/rzwang/")
        for path in candidate_files:
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="ignore")
                for literal in forbidden_literals:
                    self.assertNotIn(literal, text, str(path))


if __name__ == "__main__":
    unittest.main()
