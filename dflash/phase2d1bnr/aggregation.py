"""Pure aggregation for Phase 2D.1B-NR observations and verifier margins."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter("null" if value is None else str(value) for value in values)
    return dict(sorted(counts.items()))


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def aggregate_exact_reproducibility(
    observations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize process-level repeats without asserting motivating token IDs."""

    exact = [row for row in observations if row.get("experiment") == "exact_case"]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in exact:
        grouped[(str(row["selector_mode"]), int(row["node_budget"]))].append(row)
    strata = []
    for (mode, budget), rows in sorted(grouped.items()):
        reference_hashes = _distribution(
            row.get("actual_reference_output_sha256") for row in rows
        )
        tree_hashes = _distribution(
            row.get("actual_tree_output_sha256") for row in rows
        )
        divergence = [
            row
            for row in rows
            if row.get("actual_reference_tree_output_ids_equal") is False
        ]
        incomplete = [
            row
            for row in rows
            if row.get("outcome_category")
            in {"incomplete_correctness_evidence", "infrastructure_failure"}
        ]
        reference_unstable = bool(
            any(row.get("outcome_category") == "reference_instability" for row in rows)
            or len([key for key in reference_hashes if key != "null"]) > 1
        )
        reference_stable = bool(
            rows
            and not reference_unstable
            and "null" not in reference_hashes
            and all(
                row.get("reference_repeat_stable") is True
                and row.get("reference_profile_stable") is True
                for row in rows
            )
        )
        tree_stable = bool(
            rows
            and "null" not in tree_hashes
            and len([key for key in tree_hashes if key != "null"]) == 1
        )
        if reference_unstable:
            reproducibility = "reference_instability"
        elif incomplete:
            reproducibility = "insufficient_evidence"
        elif len(divergence) == len(rows) and rows and tree_stable:
            reproducibility = "stable_actual_divergence"
        elif not divergence:
            reproducibility = "no_reproduced_actual_divergence"
        else:
            reproducibility = "stochastic_actual_divergence"
        strata.append(
            {
                "selector_mode": mode,
                "node_budget": budget,
                "repeat_count": len(rows),
                "reference_output_hash_distribution": reference_hashes,
                "tree_output_hash_distribution": tree_hashes,
                "outcome_category_distribution": _distribution(
                    row.get("outcome_category") for row in rows
                ),
                "actual_divergence_count": len(divergence),
                "actual_exact_count": len(rows) - len(divergence),
                "first_divergence_offset_distribution": _distribution(
                    row.get("first_divergent_generated_offset") for row in divergence
                ),
                "first_divergence_round_distribution": _distribution(
                    row.get("first_divergent_tree_round") for row in divergence
                ),
                "reference_token_distribution": _distribution(
                    row.get("reference_token_id") for row in divergence
                ),
                "tree_token_distribution": _distribution(
                    row.get("tree_token_id") for row in divergence
                ),
                "reference_stable_across_repeats": reference_stable,
                "tree_stable_across_repeats": tree_stable,
                "reproducibility_classification": reproducibility,
            }
        )
    return {
        "schema_version": "1.0",
        "status": "complete"
        if exact
        and all(
            row["reproducibility_classification"] != "insufficient_evidence"
            for row in strata
        )
        else "incomplete",
        "repeat_observation_count": len(exact),
        "configuration_strata": strata,
    }


def _cohort_summary_row(
    rows: Sequence[dict[str, Any]], *, extra: dict[str, Any]
) -> dict[str, Any]:
    divergences = [
        row
        for row in rows
        if row.get("actual_reference_tree_output_ids_equal") is False
    ]
    physical = {row["physical_case_id"] for row in rows}
    configuration = {row["configuration_case_id"] for row in rows}
    divergent_physical = {row["physical_case_id"] for row in divergences}
    divergent_configuration = {row["configuration_case_id"] for row in divergences}
    return {
        **extra,
        "repeat_observation_count": len(rows),
        "physical_case_count": len(physical),
        "configuration_case_count": len(configuration),
        "actual_divergence_observation_count": len(divergences),
        "actual_divergence_observation_rate": _rate(len(divergences), len(rows)),
        "actual_divergent_physical_case_count": len(divergent_physical),
        "actual_divergent_physical_case_rate": _rate(
            len(divergent_physical), len(physical)
        ),
        "actual_divergent_configuration_case_count": len(divergent_configuration),
        "actual_divergent_configuration_case_rate": _rate(
            len(divergent_configuration), len(configuration)
        ),
        "outcome_category_distribution": _distribution(
            row.get("outcome_category") for row in rows
        ),
    }


def aggregate_cohort_cases(
    observations: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    cohort = [row for row in observations if row.get("experiment") == "cohort"]
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_context: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    by_policy: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in cohort:
        by_case[str(row["configuration_case_id"])].append(row)
        by_context[
            (
                str(row["selector_mode"]),
                int(row["node_budget"]),
                int(row["nominal_prompt_length"]),
            )
        ].append(row)
        by_policy[(str(row["selector_mode"]), int(row["node_budget"]))].append(row)
    case_rows = []
    for case_id, rows in sorted(by_case.items()):
        first = rows[0]
        case_rows.append(
            _cohort_summary_row(
                rows,
                extra={
                    "configuration_case_id": case_id,
                    "physical_case_id": first["physical_case_id"],
                    "task": first["task"],
                    "sample_id": first["sample_id"],
                    "source_sample_id": first["source_sample_id"],
                    "nominal_prompt_length": first["nominal_prompt_length"],
                    "selector_mode": first["selector_mode"],
                    "node_budget": first["node_budget"],
                },
            )
        )
    context_rows = [
        _cohort_summary_row(
            rows,
            extra={
                "selector_mode": key[0],
                "node_budget": key[1],
                "nominal_prompt_length": key[2],
            },
        )
        for key, rows in sorted(by_context.items())
    ]
    policy_rows = [
        _cohort_summary_row(
            rows,
            extra={"selector_mode": key[0], "node_budget": key[1]},
        )
        for key, rows in sorted(by_policy.items())
    ]
    return {
        "cohort_case_summary": case_rows,
        "context_divergence_summary": context_rows,
        "policy_divergence_summary": policy_rows,
    }


def _sweep_row(
    rows: Sequence[dict[str, Any]],
    *,
    threshold: float,
    common: dict[str, Any],
) -> dict[str, Any]:
    analyzable = [row for row in rows if row.get("minimum_verifier_margin") is not None]
    triggered = [
        row for row in analyzable if float(row["minimum_verifier_margin"]) <= threshold
    ]
    divergence = [row for row in rows if row["first_actual_divergence_round"]]
    caught = [row for row in divergence if row in triggered]
    nondivergence = [
        row for row in analyzable if not row["first_actual_divergence_round"]
    ]
    nondivergence_triggered = [row for row in triggered if row in nondivergence]
    source_rates = []
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in analyzable:
        by_source[str(row["source_sample_id"])].append(row)
    for source_rows in by_source.values():
        source_rates.append(
            sum(
                float(row["minimum_verifier_margin"]) <= threshold
                for row in source_rows
            )
            / len(source_rows)
        )
    return {
        **common,
        "threshold": threshold,
        "analyzable_round_count": len(analyzable),
        "margin_unavailable_round_count": len(rows) - len(analyzable),
        "trigger_count": len(triggered),
        "trigger_rate": _rate(len(triggered), len(analyzable)),
        "first_actual_divergence_round_count": len(divergence),
        "divergence_rounds_caught_by_trigger": len(caught),
        "divergence_recall": _rate(len(caught), len(divergence)),
        "divergence_recall_unavailable_reason": (
            None if divergence else "no_actual_divergence_observed"
        ),
        "non_divergence_round_count": len(nondivergence),
        "non_divergence_trigger_count": len(nondivergence_triggered),
        "non_divergence_trigger_rate": _rate(
            len(nondivergence_triggered), len(nondivergence)
        ),
        "source_macro_trigger_rate": (
            statistics.fmean(source_rates) if source_rates else None
        ),
    }


def build_margin_trigger_sweep(
    round_events: Sequence[dict[str, Any]],
    *,
    thresholds: Iterable[float],
) -> list[dict[str, Any]]:
    """Compute the preregistered deterministic margin-trigger characterization."""

    clean_thresholds = tuple(float(value) for value in thresholds)
    if not clean_thresholds or any(value < 0 for value in clean_thresholds):
        raise ValueError("margin thresholds must be non-empty and non-negative")
    groups: dict[tuple[str, str, int, str, int | None], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in round_events:
        if not row.get("included_in_trigger_characterization"):
            continue
        key = (
            str(row["experiment"]),
            str(row["selector_mode"]),
            int(row["node_budget"]),
            "overall",
            None,
        )
        groups[key].append(row)
        context_key = (*key[:3], "context", int(row["nominal_prompt_length"]))
        groups[context_key].append(row)
    result = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        common = {
            "experiment": key[0],
            "selector_mode": key[1],
            "node_budget": key[2],
            "scope": key[3],
            "nominal_prompt_length": key[4],
        }
        result.extend(
            _sweep_row(rows, threshold=threshold, common=common)
            for threshold in clean_thresholds
        )
    return result
