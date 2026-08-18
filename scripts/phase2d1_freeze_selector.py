#!/usr/bin/env python3
"""Freeze the validated Phase 2D.1A selectors into a target-blind deployment config."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsonschema import Draft202012Validator

from dflash.phase2b1.validation import validate_capture_directories
from dflash.phase2d1.events import build_selector_events
from dflash.phase2d1.freeze import (
    freeze_selector_config,
    validate_phase2d1_freeze_inputs,
)
from dflash.phase2d1.preflight import validate_phase2d0_directory

SCHEMA_DIR = REPO_ROOT / "docs" / "schemas"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2d1-dir", required=True, type=Path)
    parser.add_argument("--phase2d0-dir", required=True, type=Path)
    parser.add_argument("--capture-dir", required=True, action="append", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/phase2d1-frozen-selector")
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git(*arguments: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    protected = [args.phase2d1_dir, args.phase2d0_dir, *args.capture_dir]
    for raw in protected:
        source = raw.expanduser().resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("freeze output must not overlap immutable inputs")
    if output.exists():
        raise ValueError("freeze output directory already exists")
    validated_capture = validate_capture_directories(args.capture_dir, schema_dir=SCHEMA_DIR)
    phase2d0 = validate_phase2d0_directory(
        args.phase2d0_dir,
        validated_capture=validated_capture,
        schema_dir=SCHEMA_DIR,
        top_m=16,
        node_budgets=(16, 32),
    )
    events = build_selector_events(phase2d0["events"])
    phase2d1 = validate_phase2d1_freeze_inputs(
        args.phase2d1_dir,
        rebuilt_events=events,
        phase2d0=phase2d0,
        validated_capture=validated_capture,
    )
    frozen = freeze_selector_config(
        events,
        phase2d1=phase2d1,
        phase2d0=phase2d0,
        validated_capture=validated_capture,
    )
    schema = json.loads(
        (SCHEMA_DIR / "phase2d1b-frozen-selector.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(frozen)
    manifest = {
        "schema_version": "1.0",
        "phase": "2D.1B",
        "mode": "cpu_only_selector_freeze",
        "status": "ok",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "commit_sha": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "config_sha256": frozen["config_sha256"],
        "training_source_count": frozen["training_source_count"],
        "framework_modules_loaded": {
            name: name in sys.modules for name in ("torch", "transformers", "datasets")
        },
        "input_artifacts_mutated": False,
        "artifacts": ["frozen-selector.json", "frozen-selector-manifest.json"],
    }
    if any(manifest["framework_modules_loaded"].values()):
        raise RuntimeError("selector freeze unexpectedly loaded a model framework")
    output.mkdir(parents=True)
    _write_json(output / "frozen-selector.json", frozen)
    _write_json(output / "frozen-selector-manifest.json", manifest)
    print(f"frozen_selector={output / 'frozen-selector.json'}")
    print(f"config_sha256={frozen['config_sha256']}")
    print(f"training_sources={frozen['training_source_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
