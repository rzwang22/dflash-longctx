from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.phase0_smoke import (
    DEFAULT_DRAFT_REVISION,
    DEFAULT_TARGET_REVISION,
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
                "--draft-model",
                "draft/local",
                "--draft-revision",
                "draft-rev",
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
