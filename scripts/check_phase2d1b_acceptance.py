#!/usr/bin/env python3
"""Official artifact and invariant checker for Phase 2D.1B online validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsonschema import Draft202012Validator

from dflash.phase2d1.runtime import validate_frozen_selector
from dflash.phase2d1b.metrics import DECISIONS
from experiments.phase2d1b_online_selector import (
    PHASE2C_DIAGNOSTIC_ARTIFACTS,
    ROOT_ARTIFACTS,
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate(value: dict, name: str) -> None:
    schema = _read_json(REPO_ROOT / "docs/schemas" / name)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(value)


def _validate_correctness(correctness: dict, cross_run: dict) -> None:
    if not correctness["all_checks_passed"] or not correctness[
        "target_blind_invariants_passed"
    ]:
        raise ValueError("Phase 2D.1B correctness or target-blind invariant failed")
    if not correctness.get("actual_execution_correctness_passed", False):
        raise ValueError("Phase 2D.1B actual execution correctness failed")
    if not correctness.get("actual_output_exactness_required", False):
        raise ValueError("Phase 2D.1B did not require actual output exactness")
    if any(
        not row.get("phase2d1b_online_correctness_passed", False)
        for row in correctness["records"]
    ):
        raise ValueError(
            "a selector subrun has output divergence or incomplete token-exact correctness"
        )
    if not all(
        cross_run.get(field, False)
        for field in (
            "all_control_profile_output_ids_equal",
            "all_original_dflash_reference_tree_output_ids_equal",
            "reference_output_stable_across_subruns",
            "tree_output_stable_across_subruns",
        )
    ):
        raise ValueError("cross-run/reference/control output stability failed")


def check_output(output_dir: Path) -> dict:
    root = output_dir.expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Phase 2D.1B output has no manifest.json")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "ok":
        error = manifest.get("error")
        raise ValueError(
            "Phase 2D.1B run is incomplete or unsuccessful: "
            f"status={manifest.get('status')!r} error={error!r}"
        )
    missing = [
        name
        for name in (*ROOT_ARTIFACTS, *PHASE2C_DIAGNOSTIC_ARTIFACTS)
        if not (root / name).is_file()
    ]
    if missing:
        raise ValueError(f"Phase 2D.1B output lacks artifacts: {missing}")
    frozen = validate_frozen_selector(_read_json(root / "frozen-selector.json"))
    summary = _read_json(root / "selector-summary.json")
    comparison = _read_json(root / "selector-comparison.json")
    cross_run = _read_json(root / "cross-run-comparison.json")
    correctness = _read_json(root / "correctness-summary.json")
    rounds = _read_jsonl(root / "selector-rounds.jsonl")
    samples = _read_jsonl(root / "samples.jsonl")
    decisions = _read_jsonl(root / "selector-decisions.jsonl")
    inventory = _read_json(root / "workload-inventory.json")
    alignment = _read_json(root / "offline-online-selector-summary.json")
    _validate(manifest, "phase2d1b-manifest.schema.json")
    _validate(summary, "phase2d1b-summary.schema.json")
    for row in rounds:
        _validate(row, "phase2d1b-round.schema.json")
    for row in samples:
        _validate(row, "phase2d1b-sample.schema.json")
    if manifest["selector_config_hash"] != frozen["config_sha256"]:
        raise ValueError("manifest/frozen selector hash mismatch")
    _validate_correctness(correctness, cross_run)
    for row in rounds:
        if row["committed_token_count"] != 1 + row["accepted_draft_token_count"]:
            raise ValueError("accepted/committed accounting invariant failed")
        if row["actual_constructed_node_count"] > row["node_budget_cap"]:
            raise ValueError("actual constructed nodes exceed cap")
        if not row["selector_runtime_included_in_formal_control"]:
            raise ValueError("selector runtime was excluded from formal control")
        if row["selector_mode"] == "top1_packed_tree" and row[
            "actual_constructed_node_count"
        ] != 16:
            raise ValueError("Top-1 packed tree must contain 16 root-inclusive nodes")
        if row["selector_mode"] != "top1_packed_tree" and row[
            "actual_constructed_node_count"
        ] != row["node_budget_cap"]:
            raise ValueError("frozen selector did not realize its fixed node budget")
    for row in decisions:
        if not row["decision_made_before_target_verification"]:
            raise ValueError("selector decision occurred after verification")
        if row["target_dependent_runtime_input_used"] or row[
            "target_forward_count_added"
        ]:
            raise ValueError("selector used target-dependent input/forward")
        allowed = {
            value["config_id"]
            for value in frozen["node_budget_rules"][str(row["node_budget_cap"])][
                "allowed_policy_configurations"
            ]
        }
        if row["selected_policy_config"] not in allowed:
            raise ValueError("runtime selected policy outside frozen allowlist")
    if comparison["decision"] not in DECISIONS:
        raise ValueError("automatic decision is outside registered enum")
    if inventory["independent_generalization"] and not inventory[
        "training_source_exclusion_completed_before_model_loading"
    ]:
        raise ValueError("independent source exclusion happened after model load")
    if inventory.get("training_sources_reintroduced"):
        raise ValueError("selector training source was reintroduced")
    if alignment.get("different_round_segmentation_forced_matches") != 0:
        raise ValueError("offline/online alignment forced different segmentation")
    return {
        "status": "pass",
        "selector_config_hash": frozen["config_sha256"],
        "sample_count": len(samples),
        "round_count": len(rounds),
        "selector_decision_count": len(decisions),
        "evaluation_cohort": manifest["evaluation_cohort"],
        "automatic_decision": comparison["decision"],
        "correctness_passed": True,
        "target_blind_runtime_passed": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = check_output(args.output_dir)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
