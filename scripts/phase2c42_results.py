#!/usr/bin/env python3
"""Extract Phase 2C.4.2 targeted-case results without importing ML frameworks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_case(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = output_dir / "reference-state-summary.json"
    rows_path = output_dir / "reference-state-differential.jsonl"
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    cases = summary_payload.get("cases") or []
    if len(cases) != 1:
        raise ValueError(f"{summary_path} must contain exactly one targeted case")
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return cases[0], rows


def extract_case(output_dir: Path) -> dict[str, Any]:
    case, rows = load_case(output_dir)
    divergence_offset = case.get("first_actual_semantic_divergence")
    divergence = next(
        (
            row
            for row in rows
            if row.get("generated_prefix_length") == divergence_offset
        ),
        None,
    )
    segmentation = [
        row["generated_prefix_length"]
        for row in rows
        if row.get("segmentation_only_difference")
    ]
    return {
        "output_dir": str(output_dir),
        "sample_id": case.get("sample_id"),
        "nominal_prompt_length": case.get("nominal_prompt_length"),
        "failure_classification": case.get("failure_classification"),
        "classification_evidence_prefix": case.get("classification_evidence_prefix"),
        "classification_evidence_complete": case.get(
            "classification_evidence_complete"
        ),
        "first_actual_semantic_divergence": divergence_offset,
        "first_trusted_state_difference": case.get(
            "first_trusted_state_difference"
        ),
        "segmentation_only_prefixes": segmentation,
        "reference_actual_token_id": (divergence or {}).get(
            "reference_actual_token_id"
        ),
        "tree_actual_token_id": (divergence or {}).get("tree_actual_token_id"),
        "packed_tree_argmax_token_id": (divergence or {}).get(
            "packed_tree_argmax_token_id"
        ),
        "native_dense_replay_token_id": (divergence or {}).get(
            "native_dense_replay_token_id"
        ),
        "same_mask_replay_token_id": (divergence or {}).get(
            "same_mask_replay_token_id"
        ),
        "math_replay_token_id": (divergence or {}).get("math_replay_token_id"),
        "unresolved_reason": case.get("unresolved_reason"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dirs", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = [extract_case(path) for path in args.output_dirs]
    payload = {"schema_version": "1.0", "cases": results}
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    print(serialized)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
