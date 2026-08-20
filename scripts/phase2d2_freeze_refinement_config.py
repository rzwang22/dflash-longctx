#!/usr/bin/env python3
"""Freeze at most three Phase 2D.2A configs from mechanism-analysis outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase2d2.freeze import freeze_refinement_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate(root: Path) -> tuple[dict, dict]:
    manifest_path = root / "manifest.json"
    summary_path = root / "refinement-summary.json"
    correctness_path = root / "correctness-summary.json"
    for path in (manifest_path, summary_path, correctness_path):
        if not path.is_file():
            raise ValueError(f"mechanism analysis lacks {path.name}: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["phase2d2"].get("evidence_role") != "mechanism_analysis":
        raise ValueError("config freezing accepts mechanism-analysis outputs only")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    correctness = json.loads(correctness_path.read_text(encoding="utf-8"))
    if len(summary["aggregates"]) != 1:
        raise ValueError("each config-freeze input must contain one aggregate row")
    aggregate = summary["aggregates"][0]
    phase = manifest["phase2d2"]
    candidate = {
        **{
            field: phase[field]
            for field in (
                "configuration_id",
                "candidate_merge_mode",
                "base_top_m",
                "conditional_top_r",
                "refine_depth",
                "refined_prefix_budget",
                "refinement_weight",
                "node_budget",
            )
        },
        "mechanism_correctness_passed": correctness["official_checker_passed"],
        "refined_only_target_recovery_count": aggregate[
            "candidate_refined_only_recovery_count"
        ],
        "accepted_tokens_per_verified_node": aggregate[
            "accepted_tokens_per_verified_node"
        ],
        "conditional_remask_cuda_median_ms": aggregate["conditional_remask_cuda_ms"][
            "median"
        ]
        or 0.0,
    }
    source = {
        "path": str(root.expanduser().resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "summary_sha256": _sha256(summary_path),
        "held_out_evaluation": False,
    }
    return candidate, source


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pairs = [_candidate(path) for path in args.analysis_dir]
    frozen = freeze_refinement_config(
        [candidate for candidate, _ in pairs], [source for _, source in pairs]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(f"Frozen {len(frozen['configurations'])} configs: {args.output}")
    print(f"SHA256: {frozen['config_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
