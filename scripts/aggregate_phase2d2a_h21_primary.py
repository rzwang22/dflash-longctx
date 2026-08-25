#!/usr/bin/env python3
"""Aggregate the frozen Phase 2D.2A-H2.1 primary held-out matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.phase2d2a_h21_common import (
    EXECUTION_MANIFEST_NAME,
    EXECUTION_PLAN_NAME,
    REQUIRED_FREEZE_SHA256,
    build_aggregate,
    ensure_immutable_freeze,
    persist_aggregate,
    read_json,
    validate_h21_freeze,
    validate_h21_schema,
    validate_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", required=True, type=Path)
    parser.add_argument("--freeze-content-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def aggregate_output(
    output_root: Path, freeze_dir: Path, freeze_content_sha256: str
) -> dict:
    output_root = output_root.expanduser().resolve()
    bundle = validate_h21_freeze(
        freeze_dir,
        freeze_content_sha256,
        required_freeze_sha256=REQUIRED_FREEZE_SHA256,
    )
    plan = read_json(output_root / EXECUTION_PLAN_NAME)
    manifest = read_json(output_root / EXECUTION_MANIFEST_NAME)
    validate_plan(plan, bundle)
    validate_h21_schema(manifest)
    ensure_immutable_freeze(manifest, bundle)
    aggregate = build_aggregate(
        output_root=output_root,
        bundle=bundle,
        plan=plan,
    )
    persist_aggregate(output_root, aggregate)
    return aggregate


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    aggregate = aggregate_output(
        args.output_root, args.freeze_dir, args.freeze_content_sha256
    )
    cell = aggregate["cell_summary"]
    print(
        json.dumps(
            {
                "status": cell["status"],
                "output_root": str(args.output_root.expanduser().resolve()),
                "cell_count": cell["observed_cell_count"],
                "valid_cell_count": cell["valid_cell_count"],
                "correctness_invalid_cell_count": cell[
                    "correctness_invalid_cell_count"
                ],
                "infrastructure_failed_cell_count": cell[
                    "infrastructure_failed_cell_count"
                ],
                "family_cross_task_classifications": aggregate["primary"][
                    "family_cross_task_classifications"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
