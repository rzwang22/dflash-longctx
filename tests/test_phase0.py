from __future__ import annotations

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

from experiments.phase0_smoke import (
    DEFAULT_DRAFT_REVISION,
    DEFAULT_TARGET_REVISION,
    _pretrained_location_kwargs,
    _resolve_model_source,
    main,
    new_result_record,
    parse_args,
    run_configuration,
    validate_result_record,
)
from scripts.collect_environment import collect_environment, write_json


class Phase0SmokeTests(unittest.TestCase):
    def test_cli_defaults_and_overrides(self) -> None:
        defaults = parse_args([])
        self.assertEqual(defaults.max_new_tokens, 32)
        self.assertEqual(defaults.dtype, "bfloat16")
        self.assertEqual(defaults.device, "cuda:0")
        self.assertEqual(defaults.target_revision, DEFAULT_TARGET_REVISION)
        self.assertEqual(defaults.draft_revision, DEFAULT_DRAFT_REVISION)

        overridden = parse_args(
            [
                "--target-model",
                "target/local",
                "--target-revision",
                "target-rev",
                "--target-model-path",
                "/models/target",
                "--draft-model",
                "draft/local",
                "--draft-revision",
                "draft-rev",
                "--draft-model-path",
                "/models/draft",
                "--device",
                "cuda:2",
                "--dtype",
                "float16",
                "--max-new-tokens",
                "7",
                "--prompt",
                "test prompt",
                "--output-dir",
                "some-output",
            ]
        )
        self.assertEqual(overridden.target_revision, "target-rev")
        self.assertEqual(overridden.draft_revision, "draft-rev")
        self.assertEqual(overridden.target_model_path, Path("/models/target"))
        self.assertEqual(overridden.draft_model_path, Path("/models/draft"))
        self.assertEqual(overridden.max_new_tokens, 7)
        self.assertEqual(overridden.prompt, "test prompt")

    def test_result_schema_and_revision_persistence(self) -> None:
        args = parse_args(
            [
                "--target-revision",
                "target-pinned",
                "--draft-revision",
                "draft-pinned",
            ]
        )
        environment = collect_environment(
            repo_root=REPO_ROOT,
            run_config=run_configuration(args),
            run_command="phase0 test command",
            import_torch=False,
        )
        record = new_result_record(args, environment, "phase0 test command")
        validate_result_record(record)
        self.assertEqual(record["configuration"]["target_revision"], "target-pinned")
        self.assertEqual(record["configuration"]["draft_revision"], "draft-pinned")
        self.assertFalse(record["configuration"]["enable_thinking"])
        self.assertEqual(record["configuration"]["output_dir"], "outputs/phase0-smoke")
        self.assertEqual(
            record["models"]["target"]["requested_revision"], "target-pinned"
        )
        self.assertEqual(
            record["models"]["draft"]["requested_revision"], "draft-pinned"
        )

        schema = json.loads(
            (REPO_ROOT / "docs/schemas/phase0-smoke.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            schema["properties"]["schema_version"]["const"], record["schema_version"]
        )
        self.assertTrue(set(schema["required"]).issubset(record))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            write_json(output, record)
            persisted = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(persisted["configuration"]["target_revision"], "target-pinned")
        self.assertEqual(persisted["configuration"]["draft_revision"], "draft-pinned")

    def test_local_model_source_revision_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "target"
            metadata_path = (
                model_path
                / ".cache"
                / "huggingface"
                / "download"
                / "config.json.metadata"
            )
            metadata_path.parent.mkdir(parents=True)
            (model_path / "config.json").write_text('{"model_type": "qwen3"}\n')
            metadata_path.write_text(f"{DEFAULT_TARGET_REVISION}\netag\n0\n")

            source = _resolve_model_source(
                "target",
                "Qwen/Qwen3-8B",
                DEFAULT_TARGET_REVISION,
                model_path,
            )

            self.assertEqual(source["source_kind"], "local_path")
            self.assertEqual(source["resolved_revision"], DEFAULT_TARGET_REVISION)
            self.assertTrue(source["revision_verified"])
            self.assertEqual(
                source["revision_verification_source"],
                "local Hugging Face download metadata",
            )
            self.assertEqual(len(source["source_files_sha256"]["config.json"]), 64)
            self.assertEqual(
                _pretrained_location_kwargs(source), {"local_files_only": True}
            )

    def test_local_model_source_rejects_revision_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "draft"
            metadata_path = (
                model_path
                / ".cache"
                / "huggingface"
                / "download"
                / "config.json.metadata"
            )
            metadata_path.parent.mkdir(parents=True)
            (model_path / "config.json").write_text("{}\n")
            metadata_path.write_text(f"{DEFAULT_DRAFT_REVISION}\netag\n0\n")

            with self.assertRaisesRegex(RuntimeError, "do not uniquely match"):
                _resolve_model_source(
                    "draft",
                    "z-lab/Qwen3-8B-DFlash-b16",
                    "f" * 40,
                    model_path,
                )

    def test_keyboard_interrupt_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "experiments.phase0_smoke.run_smoke", side_effect=KeyboardInterrupt
            ):
                exit_code = main(["--output-dir", directory])
            result = json.loads((Path(directory) / "result.json").read_text())

        self.assertEqual(exit_code, 130)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "KeyboardInterrupt")

    def test_environment_collection_without_cuda_does_not_crash(self) -> None:
        snapshot = collect_environment(
            repo_root=REPO_ROOT,
            run_config={"device": "cuda:0"},
            run_command="test",
            import_torch=False,
        )
        self.assertFalse(snapshot["cuda"]["available"])
        self.assertEqual(snapshot["cuda"]["gpu_count"], 0)
        self.assertEqual(snapshot["cuda"]["gpus"], [])
        self.assertEqual(snapshot["run"]["parameters"]["device"], "cuda:0")

    def test_imports_do_not_load_model_frameworks(self) -> None:
        code = """
import sys
import experiments.phase0_smoke
import scripts.collect_environment
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
