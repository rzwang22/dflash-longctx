#!/usr/bin/env python3
"""Validate a Phase 2D.2A-H1.1 budget and mechanism audit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

if __package__:
    from scripts.aggregate_phase2d2a_h_known import (
        EXPECTED_CONTEXT_LENGTHS,
        EXPECTED_FAMILIES,
        EXPECTED_NODE_BUDGETS,
        FROZEN_REFINEMENT_FAMILIES,
    )
    from scripts.audit_phase2d2a_h1_budget_mechanism import (
        ARTIFACTS,
        H2_ARTIFACT,
        SCHEMAS,
    )
else:
    from aggregate_phase2d2a_h_known import (
        EXPECTED_CONTEXT_LENGTHS,
        EXPECTED_FAMILIES,
        EXPECTED_NODE_BUDGETS,
        FROZEN_REFINEMENT_FAMILIES,
    )
    from audit_phase2d2a_h1_budget_mechanism import (
        ARTIFACTS,
        H2_ARTIFACT,
        SCHEMAS,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = REPO_ROOT / "docs" / "schemas"


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc


def _validate(payload: Any, schema_name: str) -> None:
    schema = _json(SCHEMA_ROOT / schema_name)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)


def _exact_inventory(
    rows: list[dict[str, Any]], expected: set[tuple[Any, ...]], *fields: str
) -> None:
    observed = [tuple(row[field] for field in fields) for row in rows]
    if set(observed) != expected or any(
        count != 1 for count in Counter(observed).values()
    ):
        raise ValueError(
            f"H1.1 artifact is not the exact {len(expected)}-row inventory"
        )


def _check_node_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        actual = row["actual_verified_node_counts"]
        selected = row["selected_tree_node_counts"]
        budget = int(row["configured_node_budget"])
        rounds = int(row["round_count"])
        total = sum(actual)
        if total != row["total_actual_verified_node_count"]:
            raise ValueError(
                "H1.1 node total was not computed from direct round counts"
            )
        if len(actual) != rounds or len(selected) != rounds:
            raise ValueError("H1.1 direct node evidence does not match round count")
        if (
            actual != selected
            or not row["selected_tree_equals_actual_verified_every_round"]
        ):
            raise ValueError("H1.1 selected/verified direct-node cross-check failed")
        expected_utilization = total / (rounds * budget)
        if not math.isclose(
            row["node_budget_utilization_rate"], expected_utilization, abs_tol=1e-12
        ):
            raise ValueError("H1.1 utilization is not based on direct node counts")
        expected_class = (
            "full_utilization" if expected_utilization == 1.0 else "underutilized"
        )
        if row["utilization_classification"] != expected_class:
            raise ValueError("H1.1 utilization classification is inconsistent")


def _check_regimes(rows: list[dict[str, Any]]) -> tuple[bool, bool]:
    for row in rows:
        if row["regime_evidence_source"] != "direct_actual_verified_node_count":
            raise ValueError("H1.1 pair regime lacks direct-work provenance")
        if row["configured_node_budget_used_to_infer_regime"]:
            raise ValueError("H1.1 pair regime was inferred from configured N")
        base = row["base_actual_verified_nodes_per_round"]
        refined = row["refined_actual_verified_nodes_per_round"]
        same = row["direct_summary_equal"] and row["normalized_distribution_equal"]
        expected = (
            "same_actual_verification_work"
            if same
            else "budget_cap_expansion"
            if refined["mean"] > base["mean"]
            else "different_actual_verification_work"
        )
        if row["regime"] != expected:
            raise ValueError("H1.1 direct-work regime classification is inconsistent")
        delta = refined["mean"] - base["mean"]
        available = delta != 0.0
        if row["marginal_yield_available"] != available:
            raise ValueError("H1.1 marginal-yield availability is inconsistent")
        if not math.isclose(
            row["delta_actual_verified_nodes_per_round"], delta, abs_tol=1e-12
        ):
            raise ValueError("H1.1 marginal yield uses the wrong work delta")
        if not available and any(
            row[field] is not None
            for field in (
                "marginal_accepted_tokens_per_additional_verified_node",
                "marginal_committed_tokens_per_additional_verified_node",
            )
        ):
            raise ValueError("H1.1 fabricated marginal yield for zero work delta")
    n16 = all(
        row["regime"] == "same_actual_verification_work"
        for row in rows
        if row["node_budget"] == 16
    )
    n32 = all(
        row["regime"] == "budget_cap_expansion"
        for row in rows
        if row["node_budget"] == 32
    )
    return n16, n32


def _check_mechanism(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if not row["unavailable_rows_excluded_from_denominators"]:
            raise ValueError("H1.1 included unavailable candidate rows")
        if row["q1_replay_used_as_target_label_source"]:
            raise ValueError("H1.1 used q1 as candidate-label authority")
        if row["causal_exclusivity_claimed"]:
            raise ValueError("H1.1 overstates post-hoc mechanism evidence")
        recovery = int(row["refined_only_target_recovery_count"])
        gain = row["selected_tree_coverage_absolute_gain_vs_base_cell"]
        expected = (
            "new_candidate_availability_signal"
            if recovery > 0
            else "reranking_or_tree_reallocation_signal"
            if gain is not None and gain > 0
            else "no_observed_candidate_mechanism_gain"
        )
        if row["mechanism_signal"] != expected:
            raise ValueError(
                "H1.1 mechanism signal contradicts recorded candidate sets"
            )
        if recovery == 0 and row["refined_only_recovery_zero_treated_as_failure"]:
            raise ValueError(
                "H1.1 incorrectly treats zero refined-only recovery as failure"
            )


def check_h11_output(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    missing = [name for name in ARTIFACTS if not (output_dir / name).is_file()]
    if missing:
        raise ValueError(f"missing Phase 2D.2A-H1.1 artifacts: {missing}")
    budget = _json(output_dir / "phase2d2a-h1-budget-utilization.json")
    mechanism = _json(output_dir / "phase2d2a-h1-mechanism-decomposition.json")
    reinterpretation = _json(output_dir / "phase2d2a-h1-reinterpretation.json")
    for name, payload in (
        ("phase2d2a-h1-budget-utilization.json", budget),
        ("phase2d2a-h1-mechanism-decomposition.json", mechanism),
        ("phase2d2a-h1-reinterpretation.json", reinterpretation),
    ):
        _validate(payload, SCHEMAS[name])

    expected_cells = {
        (family, context, budget_value)
        for family in EXPECTED_FAMILIES
        for context in EXPECTED_CONTEXT_LENGTHS
        for budget_value in EXPECTED_NODE_BUDGETS
    }
    expected_pairs = {
        (family, context, budget_value)
        for family in FROZEN_REFINEMENT_FAMILIES
        for context in EXPECTED_CONTEXT_LENGTHS
        for budget_value in EXPECTED_NODE_BUDGETS
    }
    _exact_inventory(
        budget["rows"],
        expected_cells,
        "configuration_family",
        "nominal_prompt_length",
        "configured_node_budget",
    )
    _exact_inventory(
        budget["pair_regimes"],
        expected_pairs,
        "refined_family",
        "nominal_prompt_length",
        "node_budget",
    )
    _exact_inventory(
        mechanism["rows"],
        expected_pairs,
        "refined_family",
        "nominal_prompt_length",
        "node_budget",
    )
    _check_node_rows(budget["rows"])
    n16_same, n32_expansion = _check_regimes(budget["pair_regimes"])
    _check_mechanism(mechanism["rows"])

    confirmed = n16_same and n32_expansion
    if budget["all_n16_pairs_same_actual_verification_work"] != n16_same:
        raise ValueError("H1.1 N16 direct-regime summary is inconsistent")
    if budget["all_n32_pairs_budget_cap_expansion"] != n32_expansion:
        raise ValueError("H1.1 N32 direct-regime summary is inconsistent")
    if budget["h1_direct_regime_hypothesis_confirmed"] != confirmed:
        raise ValueError("H1.1 direct-regime hypothesis flag is inconsistent")

    config_rows = reinterpretation["configuration_rows"]
    if {row["refined_family"] for row in config_rows} != set(
        FROZEN_REFINEMENT_FAMILIES
    ):
        raise ValueError("H1.1 selected or removed a frozen family")
    if any(
        not row["original_h1_classification_preserved_unchanged"]
        or row["original_cap_based_h1_classification"] != "mixed_known_sample"
        or row["frozen_family_selected_or_removed"]
        for row in config_rows
    ):
        raise ValueError("H1.1 rewrote H1 classifications or reselected families")
    if reinterpretation["winner_selected"]:
        raise ValueError("H1.1 selected a winner from one known sample")
    latency_rows = reinterpretation["latency_interpretation"]["rows"]
    _exact_inventory(
        latency_rows,
        expected_pairs,
        "refined_family",
        "nominal_prompt_length",
        "node_budget",
    )
    if any(
        row["robust_speedup_claimed"]
        or row["formal_timing_repeat_count"] != 3
        or not row["base_formal_timing"]
        or not row["refined_formal_timing"]
        or not row["formal_pair_comparison"]
        for row in latency_rows
    ):
        raise ValueError("H1.1 latency interpretation overclaims or loses raw timing")

    h2_path = output_dir / H2_ARTIFACT
    if h2_path.is_file() != confirmed:
        raise ValueError("H2 design presence does not match the direct node audit")
    if h2_path.is_file():
        h2 = _json(h2_path)
        _validate(h2, SCHEMAS[H2_ARTIFACT])
        if tuple(h2["frozen_refinement_families"]) != FROZEN_REFINEMENT_FAMILIES:
            raise ValueError("H2 design selected or removed a frozen family")
        if h2["h2_execution_authorized"]:
            raise ValueError("H1.1 incorrectly authorizes H2 execution")

    return {
        "status": "accepted",
        "phase": "2D.2A-H1.1",
        "matrix_cell_count": len(budget["rows"]),
        "pair_count": len(budget["pair_regimes"]),
        "all_n16_pairs_same_actual_verification_work": n16_same,
        "all_n32_pairs_budget_cap_expansion": n32_expansion,
        "h2_preregistered_design_present": h2_path.is_file(),
        "final_phase2d2a_decision": reinterpretation["final_phase2d2a_decision"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = check_h11_output(args.output_dir)
    except Exception as exc:  # noqa: BLE001 - clean acceptance failure.
        print(f"REJECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
