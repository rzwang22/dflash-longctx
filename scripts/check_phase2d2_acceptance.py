#!/usr/bin/env python3
"""Validate Phase 2D.2A Original-DFlash/tree and structural invariants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = REPO_ROOT / "docs" / "schemas"
REQUIRED = (
    "manifest.json",
    "environment.json",
    "source-provenance.json",
    "workload-inventory.json",
    "samples.jsonl",
    "refinement-rounds.jsonl",
    "candidate-merge-events.jsonl",
    "refined-tree-rounds.jsonl",
    "refinement-summary.json",
    "refinement-summary.csv",
    "candidate-quality-summary.csv",
    "fixed-budget-comparison.json",
    "fixed-budget-comparison.csv",
    "robust-verification-events.jsonl",
    "fallback-summary.json",
    "fallback-summary.csv",
    "correctness-summary.json",
    "output-divergence.json",
    "output-divergence.csv",
    "offline-online-comparison.json",
    "offline-online-comparison.csv",
    "timing-summary.json",
    "timing-summary.csv",
    "phase2d2-decision.json",
    "terminal.log",
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL {path.name}:{line_number}") from exc
    return rows


def _validate(payload: Any, schema_name: str) -> None:
    schema = _json(SCHEMA_ROOT / schema_name)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)


def _validate_gold_kv_shape_evidence(row: dict[str, Any]) -> None:
    """Reject absent or internally inconsistent authoritative KV shapes."""

    if row.get("availability") != "available":
        raise ValueError("KV Gold evidence is unavailable")
    required_shape_fields = {
        "candidate_shape",
        "reference_shape",
        "shape_metadata_available",
        "shape_layer_count_matches",
        "layer_shape_matches",
        "all_layer_shapes_match",
    }
    missing_shape_fields = sorted(required_shape_fields - set(row))
    if missing_shape_fields:
        raise ValueError(
            f"KV Gold shape metadata is incomplete: {missing_shape_fields}"
        )
    if not row["shape_metadata_available"]:
        raise ValueError("KV Gold shapes are unavailable")
    candidate = row["candidate_shape"]
    reference = row["reference_shape"]
    reported_matches = row["layer_shape_matches"]
    if not candidate or not reference:
        raise ValueError("KV Gold shape lists are empty")
    if len(candidate) != len(reference) or not row["shape_layer_count_matches"]:
        raise ValueError("KV Gold layer counts differ")
    if len(reported_matches) != len(candidate):
        raise ValueError("KV Gold per-layer shape evidence is incomplete")
    for layer_index, (left, right, reported) in enumerate(
        zip(candidate, reference, reported_matches)
    ):
        if set(left) != {"k", "v"} or set(right) != {"k", "v"}:
            raise ValueError(f"KV Gold layer {layer_index} lacks K/V")
        expected = {
            "layer_index": layer_index,
            "k": left["k"] == right["k"],
            "v": left["v"] == right["v"],
        }
        if reported != expected:
            raise ValueError(
                f"KV Gold layer {layer_index} shape flags disagree"
            )
        if not expected["k"] or not expected["v"]:
            raise ValueError(f"KV Gold layer {layer_index} K/V shapes differ")
    if not row["all_layer_shapes_match"]:
        raise ValueError("KV Gold shapes do not all match")
    if row.get("structural_status") != "pass":
        raise ValueError("KV Gold structural status did not pass")


def check_output(output: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED if not (output / name).is_file()]
    if missing:
        raise ValueError(f"missing Phase 2D.2A artifacts: {missing}")
    manifest = _json(output / "manifest.json")
    correctness = _json(output / "correctness-summary.json")
    samples = _jsonl(output / "samples.jsonl")
    rounds = _jsonl(output / "refinement-rounds.jsonl")
    merge_events = _jsonl(output / "candidate-merge-events.jsonl")
    tree_rounds = _jsonl(output / "refined-tree-rounds.jsonl")
    robust_events = _jsonl(output / "robust-verification-events.jsonl")
    decision = _json(output / "phase2d2-decision.json")
    output_divergence = _json(output / "output-divergence.json")
    gold_kv_path = output / "gold-kv-checks.jsonl"
    gold_kv = _jsonl(gold_kv_path) if gold_kv_path.is_file() else []
    if manifest.get("status") not in {"ok", "failed", "error"}:
        raise ValueError(
            f"incomplete Phase 2D.2A manifest status: {manifest.get('status')!r}"
        )
    if not samples or not rounds:
        raise ValueError("Phase 2D.2A run has no sample/round evidence")
    _validate(manifest, "phase2d2-manifest.schema.json")
    _validate(correctness, "phase2d2-correctness-summary.schema.json")
    _validate(decision, "phase2d2-decision.schema.json")
    _validate(_json(output / "refinement-summary.json"), "phase2d2-summary.schema.json")
    _validate(
        _json(output / "fixed-budget-comparison.json"),
        "phase2d2-fixed-budget-comparison.schema.json",
    )
    _validate(
        _json(output / "fallback-summary.json"),
        "phase2d2-fallback-summary.schema.json",
    )
    _validate(
        output_divergence,
        "phase2d2-output-divergence.schema.json",
    )
    _validate(
        _json(output / "offline-online-comparison.json"),
        "phase2d2-offline-online.schema.json",
    )
    _validate(
        _json(output / "timing-summary.json"),
        "phase2d2-timing-summary.schema.json",
    )
    for row in samples:
        _validate(row, "phase2d2-sample.schema.json")
        if (
            row["original_dflash_reference"]["generated_output_token_ids"]
            != row["generated_output_token_ids"]
        ):
            raise ValueError("sample Original-DFlash/refined-tree token IDs differ")
        if not row["correctness"]["passed"]:
            raise ValueError("sample hard correctness record did not pass")
        if row["candidate_merge_mode"] == "base_only" and not row[
            "correctness"
        ].get("base_only_reference_tree_output_ids_equal"):
            raise ValueError(
                "base_only Original-DFlash/tree outputs are not token exact"
            )
        expected_target_warnings = []
        if row["target_only"]["generated_output_token_ids"] != row[
            "original_dflash_reference"
        ]["generated_output_token_ids"]:
            expected_target_warnings.append(
                "target_only_reference_numerical_divergence"
            )
        if (
            row["target_only"]["generated_output_token_ids"]
            != row["generated_output_token_ids"]
        ):
            expected_target_warnings.append("target_only_tree_numerical_divergence")
        if (
            row["correctness"].get("target_only_auxiliary_warnings")
            != expected_target_warnings
        ):
            raise ValueError("target-only auxiliary output status is inconsistent")
    for row in rounds:
        _validate(row, "phase2d2-round.schema.json")
        phase = row["phase2d2"]
        if phase["initial_parallel_draft_forward_count"] != 1:
            raise ValueError("initial parallel draft forward count is not one")
        if phase["conditional_refinement_forward_count"] > 1:
            raise ValueError("more than one conditional forward occurred in a round")
        if (
            phase["candidate_merge_mode"] == "base_only"
            and phase["conditional_refinement_forward_count"] != 0
        ):
            raise ValueError("base_only control executed a conditional forward")
        if phase["candidate_merge_mode"] == "base_only" and (
            not phase["base_only_conditional_refinement_bypassed"]
            or phase["actual_verification_tree_source"]
            != "phase2c_frozen_selector_seed_tree"
            or phase["refined_tree_replaced_seed_tree"]
        ):
            raise ValueError("base_only did not preserve the Phase 2C seed-tree path")
        if phase["internal_branch_batch_size"] != len(
            phase.get("conditional_input_token_ids", [])
        ):
            raise ValueError("internal branch batch/input-row count mismatch")
        if (
            phase["actual_verified_node_count"]
            > phase["fixed_root_inclusive_node_budget"]
        ):
            raise ValueError("target verification exceeded fixed root-inclusive N")
        if phase["actual_verified_node_count"] != phase["selected_tree_node_count"]:
            raise ValueError("selected and actually verified node counts differ")
        if phase["root_count"] != 1 or not phase["node_budget_invariant_passed"]:
            raise ValueError("root/node-budget invariant failed")
        if phase["temporary_refinement_state_committed"]:
            raise ValueError("temporary refinement state was committed")
        if phase["fallback_replay_executed"] and any(
            value != 1 for value in phase["fallback_query_lengths"]
        ):
            raise ValueError("fallback replay used a query length other than one")
        if (
            phase["verification_mode"] == "always_replay"
            and not phase["fallback_replay_executed"]
        ):
            raise ValueError("always_replay did not execute authoritative replay")
        if phase["correction_or_bonus_committed_in_current_round"]:
            raise ValueError("correction/bonus was committed in the current round")
        if not phase["sibling_kv_removed_from_committed_cache"]:
            raise ValueError("sibling KV remained in the authoritative cache")
        layer_lengths = phase["authoritative_committed_target_kv_layer_lengths"]
        if not layer_lengths or any(
            int(length) != int(phase["authoritative_committed_cache_length"])
            for length in layer_lengths
        ):
            raise ValueError("authoritative target KV layer lengths disagree")
    for row in merge_events:
        _validate(row, "phase2d2-candidate-merge-event.schema.json")
    for row in tree_rounds:
        _validate(row, "phase2d2-refined-tree-round.schema.json")
    for row in robust_events:
        _validate(row, "phase2d2-robust-verification-event.schema.json")
    if manifest["phase2d2"].get("gold_checks_enabled"):
        if not gold_kv:
            raise ValueError("enabled Gold checks produced no KV evidence")
        for row in gold_kv:
            _validate(row, "phase2c-gold-kv-check.schema.json")
            _validate_gold_kv_shape_evidence(row)
    required_pairs = (
        ("target_only", "original_dflash_reference"),
        ("target_only", "refined_tree"),
        ("original_dflash_reference", "refined_tree"),
    )
    for left_name, right_name in required_pairs:
        matches = [
            row
            for row in output_divergence["records"]
            if row.get("left_name") == left_name
            and row.get("right_name") == right_name
        ]
        if len(matches) != len(samples):
            raise ValueError(
                f"absolute-token relation {left_name}/{right_name} is incomplete"
            )
        if (
            left_name == "original_dflash_reference"
            and right_name == "refined_tree"
            and any(not row.get("output_ids_equal") for row in matches)
        ):
            raise ValueError(
                "Original-DFlash/refined-tree absolute-token relation diverged"
            )
    if not correctness.get("all_original_dflash_reference_output_ids_equal"):
        raise ValueError("Original-DFlash/refined-tree output token mismatch")
    if not correctness.get("structural_correctness_passed"):
        raise ValueError(
            "structural correctness failed: "
            f"{correctness.get('hard_correctness_failure_reasons', [])}"
        )
    if not correctness.get("official_checker_passed"):
        raise ValueError("Phase 2D.2A hard correctness gate did not pass")
    return {
        "status": "accepted",
        "sample_count": len(samples),
        "round_count": len(rounds),
        "decision": decision["decision"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = check_output(args.output_dir.expanduser().resolve())
    except Exception as exc:  # noqa: BLE001 - clean checker failure.
        print(f"REJECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
