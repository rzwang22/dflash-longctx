"""Round-level packed-native margin extraction without online intervention."""

from __future__ import annotations

from typing import Any


def _round_structural_passed(row: dict[str, Any] | None) -> bool | None:
    if row is None:
        return None
    checks = row.get("checks", {})
    return bool(
        row.get("status") == "pass"
        and checks.get("all_checks_passed") is True
        and not row.get("structural_mask_issues")
    )


def _margin(stabilization: dict[str, Any]) -> tuple[float | None, str, str | None]:
    value = stabilization.get("minimum_path_margin")
    if value is not None:
        return float(value), "minimum_path_margin", None
    vector = stabilization.get("verifier_margins")
    if vector:
        return min(float(item) for item in vector), "min(verifier_margins)", None
    return (
        None,
        "unavailable",
        stabilization.get("unavailable_reason")
        or ("minimum_path_margin_and_verifier_margins_absent"),
    )


def extract_round_margin_events(
    observation: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract only pre-divergence/boundary or clean non-terminal rounds."""

    rounds = [
        row
        for row in observation.get("_rounds", [])
        if int(row.get("profiling_repeat", 0)) == 0
    ]
    stabilization = {
        (int(row.get("profiling_repeat", 0)), int(row["round_index"])): row
        for row in observation.get("_stabilization_rounds", [])
    }
    traversal = {
        int(row["round_index"]): row for row in observation.get("_traversal_checks", [])
    }
    replay_rounds = set(observation.get("_replay_semantic_rounds", []))
    first_divergent = observation.get("first_divergent_tree_round")
    output_divergent = (
        observation.get("actual_reference_tree_output_ids_equal") is False
    )
    result = []
    for row in rounds:
        round_index = int(row["round_index"])
        if output_divergent:
            include = first_divergent is not None and round_index <= int(
                first_divergent
            )
        else:
            include = (
                not bool(row.get("terminal_round"))
                and int(row.get("truncated_token_count", 0)) == 0
            )
        if not include:
            continue
        decision = stabilization.get((0, round_index), {})
        margin, margin_source, unavailable_reason = _margin(decision)
        selector = row.get("online_selector") or {}
        is_first = bool(
            output_divergent
            and first_divergent is not None
            and round_index == int(first_divergent)
        )
        result.append(
            {
                "schema_version": "1.0",
                **{
                    field: observation[field]
                    for field in (
                        "experiment",
                        "task",
                        "sample_id",
                        "source_sample_id",
                        "nominal_prompt_length",
                        "selector_mode",
                        "node_budget",
                        "repeat_index",
                        "physical_case_id",
                        "configuration_case_id",
                        "independent_repeat_observation_id",
                    )
                },
                "round_index": round_index,
                "generated_root_offset": int(row["generated_root_offset"]),
                "actual_node_count": int(row["actual_node_count"]),
                "selected_policy_config": selector.get(
                    "selected_policy_config", row.get("configuration_id")
                ),
                "accepted_draft_depth": int(row["accepted_draft_token_count"]),
                "committed_token_count": int(row["committed_token_count"]),
                "minimum_verifier_margin": margin,
                "margin_source": margin_source,
                "margin_unavailable_reason": unavailable_reason,
                "verifier_margins": decision.get("verifier_margins"),
                "first_corrected_native_margin": decision.get(
                    "first_corrected_native_margin"
                ),
                "native_accepted_depth": decision.get("native_accepted_depth"),
                "canonical_accepted_depth": decision.get("canonical_accepted_depth"),
                "native_canonical_decision_equal": decision.get(
                    "native_canonical_decision_equal"
                ),
                "first_actual_divergence_round": is_first,
                "pre_divergence": bool(
                    output_divergent
                    and first_divergent is not None
                    and round_index < int(first_divergent)
                ),
                "included_in_trigger_characterization": True,
                "structural_checks_passed": _round_structural_passed(
                    traversal.get(round_index)
                ),
                "replay_semantic_variation_observed": round_index in replay_rounds,
                "actual_execution_divergence_occurs_here": is_first,
                "terminal_round": bool(row.get("terminal_round")),
                "truncated_token_count": int(row.get("truncated_token_count", 0)),
            }
        )
    return result
