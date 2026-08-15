#!/usr/bin/env python3
"""Compare existing Phase 2A verification profiles without loading models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase2a.comparison import (
    build_verification_length_comparison,
    write_verification_length_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        nargs="+",
        required=True,
        help="One or more verification-profile-summary.json paths.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    comparison = build_verification_length_comparison(args.summary)
    paths = write_verification_length_comparison(
        args.output_dir.expanduser().resolve(), comparison
    )
    print(f"paired_comparison: {str(comparison['paired_comparison']).lower()}")
    for warning in comparison["warnings"]:
        print(f"WARNING: {warning}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
