"""Strict exact-root Phase 2D.1A/online selector alignment."""

from __future__ import annotations

from collections import Counter
from typing import Any

from dflash.phase2d1.events import policy_map
from dflash.phase2d1.runtime import select_policy


def build_offline_online_alignment(
    *,
    online_rounds: list[dict[str, Any]],
    offline_events: list[dict[str, Any]],
    captures: list[dict[str, Any]],
    frozen: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    capture_by_key = {
        (
            row["task"],
            row["source_sample_id"],
            int(row["nominal_prompt_length"]),
            int(row["generation_repeat"]),
            int(row["round_index"]),
        ): row
        for row in captures
    }
    event_by_key = {
        (
            row["task"],
            row["source_sample_id"],
            int(row["nominal_prompt_length"]),
            int(row["generation_repeat"]),
            int(row["round_index"]),
            int(row["node_budget_cap"]),
        ): row
        for row in offline_events
    }
    output = []
    for online in online_rounds:
        if online["profiling_repeat"] != 0 or online["selector_mode"] not in {
            "global_fixed",
            "context_rule",
            "decision_stump",
        }:
            continue
        base = (
            online["task"],
            online["source_sample_id"],
            int(online["nominal_prompt_length"]),
            0,
            int(online["round_index"]),
        )
        capture = capture_by_key.get(base)
        event = event_by_key.get((*base, int(online["node_budget_cap"])))
        reasons = []
        if capture is None or event is None:
            reasons.append("offline_physical_round_not_found")
        else:
            if int(capture["round_output_start_offset"]) != int(
                online["generated_root_offset"]
            ):
                reasons.append("absolute_generated_root_offset_mismatch")
            if int(capture["root_token_id"]) != int(online["root_token_id"]):
                reasons.append("root_token_id_mismatch")
            if capture["source_text_sha256"] != online["source_text_sha256"]:
                reasons.append("source_text_hash_mismatch")
        offline_decision = (
            select_policy(
                frozen,
                mode=online["selector_mode"],
                node_budget=int(online["node_budget_cap"]),
                context_length=int(online["nominal_prompt_length"]),
                runtime_features=event["runtime_features"],
            )
            if event is not None
            else None
        )
        if offline_decision is not None and offline_decision[
            "selected_policy_config"
        ] != online["selected_policy_config"]:
            reasons.append("selected_policy_mismatch")
        valid = not reasons
        offline_coverage = (
            policy_map(event)[offline_decision["selected_policy_config"]][
                "coverage_depth"
            ]
            if valid
            else None
        )
        output.append(
            {
                "schema_version": "1.0",
                "task": online["task"],
                "sample_id": online["sample_id"],
                "source_sample_id": online["source_sample_id"],
                "nominal_prompt_length": online["nominal_prompt_length"],
                "node_budget_cap": online["node_budget_cap"],
                "selector_mode": online["selector_mode"],
                "online_round_index": online["round_index"],
                "offline_round_index": event["round_index"] if event else None,
                "generated_root_offset": online["generated_root_offset"],
                "physical_round_root_alignment": valid,
                "offline_online_exact_root_match": valid,
                "alignment_valid": valid,
                "reason": None if valid else reasons,
                "offline_feature_value": (
                    event["runtime_features"].get(
                        offline_decision["feature_name"]
                    )
                    if event is not None
                    and offline_decision is not None
                    and offline_decision["feature_name"] is not None
                    else None
                ),
                "online_feature_value": online["selector_feature_value"],
                "offline_selected_policy": (
                    offline_decision["selected_policy_config"]
                    if offline_decision
                    else None
                ),
                "online_selected_policy": online["selected_policy_config"],
                "offline_target_prefix_coverage": offline_coverage,
                "online_accepted_draft_depth": (
                    online["accepted_draft_token_count"] if valid else None
                ),
                "offline_coverage_is_online_acceptance": False,
                "different_round_segmentation_forced_match": False,
            }
        )
    reasons = Counter(
        reason
        for row in output
        for reason in (row["reason"] or [])
    )
    valid_rows = [row for row in output if row["alignment_valid"]]
    summary = {
        "schema_version": "1.0",
        "status": "ok",
        "row_count": len(output),
        "alignment_valid_count": len(valid_rows),
        "alignment_invalid_count": len(output) - len(valid_rows),
        "invalid_reason_distribution": dict(reasons),
        "offline_online_policy_match_rate_on_valid_roots": (
            sum(
                row["offline_selected_policy"] == row["online_selected_policy"]
                for row in valid_rows
            )
            / len(valid_rows)
            if valid_rows
            else None
        ),
        "mean_offline_coverage_on_valid_roots": (
            sum(row["offline_target_prefix_coverage"] for row in valid_rows)
            / len(valid_rows)
            if valid_rows
            else None
        ),
        "mean_online_accepted_depth_on_valid_roots": (
            sum(row["online_accepted_draft_depth"] for row in valid_rows)
            / len(valid_rows)
            if valid_rows
            else None
        ),
        "different_round_segmentation_forced_matches": 0,
        "unmatched_rounds_are_mismatches": False,
        "offline_coverage_treated_as_online_acceptance": False,
    }
    return output, summary
