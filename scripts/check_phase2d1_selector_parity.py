#!/usr/bin/env python3
"""Replay Phase 2D.1 features/policies on CPU before online H100 validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase2b1.validation import validate_capture_directories
from dflash.phase2d1.events import build_selector_events
from dflash.phase2d1.freeze import (
    freeze_selector_config,
    validate_phase2d1_freeze_inputs,
)
from dflash.phase2d1.parity import check_selector_parity
from dflash.phase2d1.preflight import validate_phase2d0_directory
from dflash.phase2d1.runtime import validate_frozen_selector

SCHEMA_DIR = REPO_ROOT / "docs" / "schemas"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-selector", required=True, type=Path)
    parser.add_argument("--phase2d1-dir", required=True, type=Path)
    parser.add_argument("--phase2d0-dir", required=True, type=Path)
    parser.add_argument("--capture-dir", required=True, action="append", type=Path)
    parser.add_argument("--feature-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    frozen = validate_frozen_selector(
        json.loads(args.frozen_selector.read_text(encoding="utf-8"))
    )
    validated = validate_capture_directories(args.capture_dir, schema_dir=SCHEMA_DIR)
    phase2d0 = validate_phase2d0_directory(
        args.phase2d0_dir,
        validated_capture=validated,
        schema_dir=SCHEMA_DIR,
        top_m=16,
        node_budgets=(16, 32),
    )
    events = build_selector_events(phase2d0["events"])
    phase2d1 = validate_phase2d1_freeze_inputs(
        args.phase2d1_dir,
        rebuilt_events=events,
        phase2d0=phase2d0,
        validated_capture=validated,
    )
    if frozen["phase2d1_artifact_hashes"] != phase2d1["artifact_hashes"]:
        raise ValueError("frozen selector/Phase 2D.1A artifact hash mismatch")
    rebuilt_frozen = freeze_selector_config(
        events,
        phase2d1=phase2d1,
        phase2d0=phase2d0,
        validated_capture=validated,
    )
    if frozen != rebuilt_frozen:
        raise ValueError(
            "frozen selector differs from deterministic full-source refit"
        )
    report = check_selector_parity(
        frozen=frozen,
        selector_events=events,
        phase2d0_events=phase2d0["events"],
        captures=validated["rounds"],
        tolerance=args.feature_tolerance,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise ValueError("parity output already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
