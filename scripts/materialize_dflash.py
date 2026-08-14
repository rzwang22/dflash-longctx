#!/usr/bin/env python3
"""Materialize pinned original or instrumented DFlash code with external weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase1.materialize import materialize_draft


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--implementation",
        choices=("original", "instrumented"),
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = materialize_draft(
        source_model_path=args.source_model_path,
        output_dir=args.output_dir,
        implementation=args.implementation,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Materialized model: {manifest['materialized_model_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
