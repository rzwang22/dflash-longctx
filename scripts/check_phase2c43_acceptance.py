#!/usr/bin/env python3
"""Check a Phase 2C.4.3 aggregate for one targeted acceptance case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.aggregate_phase2c43_evidence import (
        ARTIFACT_NAME,
        LOCAL_NUMERICAL,
        UNRESOLVED,
        validate_aggregate,
    )
else:
    from aggregate_phase2c43_evidence import (
        ARTIFACT_NAME,
        LOCAL_NUMERICAL,
        UNRESOLVED,
        validate_aggregate,
    )


EXPECTED = {
    "case2": {
        "sample_id": "gov_report-4a773bcb56f65ce3",
        "nominal_prompt_length": 1024,
        "divergence_offset": 33,
        "classification": UNRESOLVED,
    },
    "case3": {
        "sample_id": "gov_report-fd287bbfd3bdbdc9",
        "nominal_prompt_length": 16384,
        "divergence_offset": 18,
        "classification": LOCAL_NUMERICAL,
    },
    "case4": {
        "sample_id": "gov_report-4a773bcb56f65ce3",
        "nominal_prompt_length": 32768,
        "divergence_offset": 28,
        "classification": UNRESOLVED,
    },
}


def _require(checks: list[dict[str, Any]], name: str, condition: bool) -> None:
    checks.append({"check": name, "passed": bool(condition)})


def _artifact_path(path: Path) -> Path:
    return path / ARTIFACT_NAME if path.is_dir() else path


def check_aggregate(label: str, artifact_or_dir: Path) -> dict[str, Any]:
    artifact = _artifact_path(artifact_or_dir)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    validate_aggregate(payload)
    expected = EXPECTED[label]
    matching = [
        group
        for group in payload["groups"]
        if group["sample_id"] == expected["sample_id"]
        and group["nominal_prompt_length"] == expected["nominal_prompt_length"]
        and group["first_actual_semantic_divergence"] == expected["divergence_offset"]
    ]
    checks: list[dict[str, Any]] = []
    _require(checks, "exactly_one_grouping_key_match", len(matching) == 1)
    group = matching[0] if len(matching) == 1 else {}
    _require(
        checks,
        "aggregate_failure_classification",
        group.get("aggregate_failure_classification") == expected["classification"],
    )
    _require(
        checks,
        "stable_actual_reference_tree_divergence",
        group.get("stable_actual_reference_tree_divergence") is True,
    )
    if label == "case3":
        _require(
            checks,
            "reproducible_numerical_support",
            group.get("numerical_control_support_rate") == 1.0,
        )
        _require(
            checks,
            "aggregate_evidence_complete",
            group.get("aggregate_evidence_complete") is True,
        )
    elif label == "case4":
        _require(checks, "run_count", group.get("run_count") == 5)
        _require(
            checks,
            "actual_reference_distribution",
            group.get("actual_reference_token_distribution") == {"8706": 5},
        )
        _require(
            checks,
            "actual_tree_distribution",
            group.get("actual_tree_token_distribution") == {"41044": 5},
        )
        _require(
            checks,
            "packed_tree_distribution",
            group.get("packed_tree_token_distribution") == {"41044": 5},
        )
        _require(
            checks,
            "native_replay_distribution",
            group.get("native_replay_token_distribution") == {"41044": 5},
        )
        _require(
            checks,
            "same_mask_replay_distribution",
            group.get("same_mask_replay_token_distribution") == {"8706": 3, "41044": 2},
        )
        _require(
            checks,
            "math_replay_distribution",
            group.get("math_replay_token_distribution") == {"41044": 5},
        )
        _require(
            checks,
            "same_mask_reference_support",
            group.get("same_mask_reference_support_count") == 3
            and group.get("same_mask_reference_support_rate") == 0.6,
        )
        _require(
            checks,
            "math_reference_support",
            group.get("math_reference_support_count") == 0
            and group.get("math_reference_support_rate") == 0.0,
        )
        _require(
            checks,
            "per_run_classification_distribution",
            group.get("per_run_classification_distribution")
            == {LOCAL_NUMERICAL: 3, UNRESOLVED: 2},
        )
        _require(
            checks,
            "nonreproducible_reason",
            group.get("aggregate_unresolved_reason")
            == "nonreproducible_numerical_control_support",
        )
    return {
        "phase": "2C.4.3",
        "case": label,
        "artifact": str(artifact),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--expected-case", required=True, choices=sorted(EXPECTED))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = check_aggregate(args.expected_case, args.artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
