#!/usr/bin/env python3
"""Check the three targeted Phase 2C.4.2 H100 acceptance cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.phase2c42_results import load_case
else:
    from phase2c42_results import load_case

EXPECTED = {
    "case2": {
        "sample_id": "gov_report-4a773bcb56f65ce3",
        "nominal_prompt_length": 1024,
        "segmentation_prefix": 0,
        "divergence_offset": 33,
        "classification": "unresolved_reference_state_divergence",
    },
    "case3": {
        "sample_id": "gov_report-fd287bbfd3bdbdc9",
        "nominal_prompt_length": 16384,
        "segmentation_prefix": 0,
        "divergence_offset": 18,
        "classification": "local_packed_verifier_numerical_divergence",
    },
    "case4": {
        "sample_id": "gov_report-4a773bcb56f65ce3",
        "nominal_prompt_length": 32768,
        "segmentation_prefix": 4,
        "divergence_offset": 28,
        "classification": "unresolved_reference_state_divergence",
    },
}


def _require(checks: list[dict[str, Any]], name: str, condition: bool) -> None:
    checks.append({"check": name, "passed": bool(condition)})


def check_case(label: str, output_dir: Path) -> dict[str, Any]:
    expected = EXPECTED[label]
    case, rows = load_case(output_dir)
    by_prefix = {int(row["generated_prefix_length"]): row for row in rows}
    segmentation = by_prefix.get(expected["segmentation_prefix"], {})
    divergence = by_prefix.get(expected["divergence_offset"], {})
    checks: list[dict[str, Any]] = []
    _require(checks, "sample_id", case.get("sample_id") == expected["sample_id"])
    _require(
        checks,
        "nominal_prompt_length",
        case.get("nominal_prompt_length") == expected["nominal_prompt_length"],
    )
    _require(
        checks,
        "first_actual_semantic_divergence",
        case.get("first_actual_semantic_divergence")
        == expected["divergence_offset"],
    )
    _require(
        checks,
        "failure_classification",
        case.get("failure_classification") == expected["classification"],
    )
    _require(
        checks,
        "segmentation_only",
        segmentation.get("segmentation_only_difference") is True,
    )
    _require(
        checks,
        "segmentation_not_semantic_mismatch",
        segmentation.get("transition_semantic_mismatch") is False
        and segmentation.get("semantic_consequence_observed") is False,
    )
    _require(
        checks,
        "segmentation_not_causal",
        segmentation.get("transition_mismatch_causal") in {None, False},
    )
    if label == "case3":
        _require(
            checks,
            "classification_evidence_prefix",
            case.get("classification_evidence_prefix") == 18,
        )
        _require(
            checks,
            "classification_evidence_complete",
            case.get("classification_evidence_complete") is True,
        )
        _require(
            checks,
            "actual_reference_tree_tokens",
            divergence.get("reference_actual_token_id") == 323
            and divergence.get("tree_actual_token_id") == 320
            and divergence.get("actual_token_divergence") is True,
        )
        _require(
            checks,
            "packed_matches_tree_not_reference",
            divergence.get("packed_tree_argmax_token_id") == 320
            and divergence.get("packed_tree_argmax_matches_actual_tree") is True
            and divergence.get("packed_tree_argmax_matches_actual_reference") is False,
        )
        _require(
            checks,
            "same_mask_math_support_reference",
            divergence.get("same_mask_replay_matches_actual_reference") is True
            and divergence.get("math_replay_matches_actual_reference") is True,
        )
        _require(
            checks,
            "replay_is_control_only",
            divergence.get("dense_replay_control_only") is True,
        )
    else:
        _require(
            checks,
            "not_commit_or_state_misattribution",
            case.get("failure_classification")
            not in {
                "commit_or_root_transition_mismatch",
                "draft_state_or_candidate_drift",
                "target_kv_state_drift",
                "target_hidden_state_drift",
            },
        )
    return {
        "case": label,
        "output_dir": str(output_dir),
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case2", required=True, type=Path)
    parser.add_argument("--case3", required=True, type=Path)
    parser.add_argument("--case4", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = [
        check_case("case2", args.case2),
        check_case("case3", args.case3),
        check_case("case4", args.case4),
    ]
    payload = {
        "phase": "2C.4.2",
        "passed": all(result["passed"] for result in results),
        "cases": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
