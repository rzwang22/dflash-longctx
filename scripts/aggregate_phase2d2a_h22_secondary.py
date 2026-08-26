#!/usr/bin/env python3
"""Aggregate the frozen secondary Phase 2D.2A-H2.2 N32 matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.phase2d2a_h22_common import (
    EXECUTION_MANIFEST_NAME,
    EXECUTION_PLAN_NAME,
    build_aggregate,
    ensure_immutable_sources,
    load_h21_anchor_cells,
    load_h22_cell,
    persist_aggregate,
    read_json,
    source_bindings,
    validate_h22_freeze,
    validate_h22_schema,
    validate_plan,
)


def aggregate_output(
    *,
    freeze_dir: Path,
    freeze_content_sha256: str,
    h21_output_root: Path,
    h211_output_dir: Path,
    output_root: Path,
) -> dict:
    output_root = output_root.expanduser().resolve()
    bundle = validate_h22_freeze(freeze_dir, freeze_content_sha256)
    bindings = source_bindings(
        bundle=bundle,
        h21_output_root=h21_output_root,
        h211_output_dir=h211_output_dir,
    )
    plan = read_json(output_root / EXECUTION_PLAN_NAME)
    manifest = read_json(output_root / EXECUTION_MANIFEST_NAME)
    validate_plan(plan, bundle, bindings)
    validate_h22_schema(manifest)
    ensure_immutable_sources(manifest, bundle, bindings)
    _, h21_cells = load_h21_anchor_cells(h21_output_root)
    h22_cells = [load_h22_cell(output_root, row) for row in plan["rows"]]
    artifacts = build_aggregate(
        bundle=bundle,
        bindings=bindings,
        plan=plan,
        h21_cells=h21_cells,
        h22_cells=h22_cells,
    )
    persist_aggregate(output_root, artifacts)
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", required=True, type=Path)
    parser.add_argument("--freeze-content-sha256", required=True)
    parser.add_argument("--h21-output-root", required=True, type=Path)
    parser.add_argument("--h211-output-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = aggregate_output(
        freeze_dir=args.freeze_dir,
        freeze_content_sha256=args.freeze_content_sha256,
        h21_output_root=args.h21_output_root,
        h211_output_dir=args.h211_output_dir,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "valid_cell_count": artifacts["cell_summary"]["valid_cell_count"],
                "correctness_invalid_cell_count": artifacts["cell_summary"][
                    "correctness_invalid_cell_count"
                ],
                "infrastructure_failed_cell_count": artifacts["cell_summary"][
                    "infrastructure_failed_cell_count"
                ],
                "secondary_conclusion": artifacts["conclusion"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
