"""Exact-root alignment between Phase 2B.1 offline replay and Phase 2C online rounds."""

from __future__ import annotations

from typing import Any


def _beam_width(row: dict[str, Any]) -> int | None:
    if row.get("tree_policy", row.get("policy")) == "top1_chain":
        return 1
    value = row.get("beam_width")
    if value is None:
        value = row.get("policy_parameters", {}).get("beam_width")
    return int(value) if value is not None else None


def _configuration_key(row: dict[str, Any], *, online: bool) -> tuple[Any, ...]:
    return (
        row.get("task"),
        row.get("sample_id"),
        int(row.get("nominal_prompt_length", row.get("context_length", 0))),
        row.get("tree_policy" if online else "policy"),
        int(
            row.get(
                "requested_node_count"
                if online
                else "requested_total_tree_node_budget",
                0,
            )
        ),
        _beam_width(row),
    )


def _round_key(row: dict[str, Any], *, online: bool) -> tuple[Any, ...]:
    root = row.get("generated_root_offset" if online else "round_output_start_offset")
    return (*_configuration_key(row, online=online), int(root))


def align_offline_online(
    online_rounds: list[dict[str, Any]], offline_events: list[dict[str, Any]]
) -> dict[str, Any]:
    """Align only exact generated-root offsets; unmatched rows are not mismatches."""

    online_configurations = {
        _configuration_key(row, online=True) for row in online_rounds
    }
    relevant_offline = [
        row
        for row in offline_events
        if _configuration_key(row, online=False) in online_configurations
    ]
    offline_by_key = {_round_key(row, online=False): row for row in relevant_offline}
    online_keys = set()
    rows = []
    matched = 0
    agreements = 0
    for online in online_rounds:
        key = _round_key(online, online=True)
        online_keys.add(key)
        offline = offline_by_key.get(key)
        if offline is None:
            rows.append(
                {
                    "task": online["task"],
                    "sample_id": online["sample_id"],
                    "context_length": online["nominal_prompt_length"],
                    "generated_root_offset": online["generated_root_offset"],
                    "policy": online["tree_policy"],
                    "node_budget": online["requested_node_count"],
                    "beam_width": online["beam_width"],
                    "status": "not_comparable",
                    "offline_covered_depth": None,
                    "online_accepted_draft_token_count": online[
                        "accepted_draft_token_count"
                    ],
                    "difference": None,
                    "mismatch_reason": "no exact offline generated-root offset",
                }
            )
            continue
        matched += 1
        offline_depth = int(offline["covered_target_prefix_depth"])
        online_depth = int(online["accepted_draft_token_count"])
        difference = online_depth - offline_depth
        agreements += difference == 0
        rows.append(
            {
                "task": online["task"],
                "sample_id": online["sample_id"],
                "context_length": online["nominal_prompt_length"],
                "generated_root_offset": online["generated_root_offset"],
                "policy": online["tree_policy"],
                "node_budget": online["requested_node_count"],
                "beam_width": online["beam_width"],
                "status": "comparable",
                "offline_covered_depth": offline_depth,
                "online_accepted_draft_token_count": online_depth,
                "difference": difference,
                "mismatch_reason": None if difference == 0 else "depth disagreement",
            }
        )
    unmatched_offline = sum(
        _round_key(row, online=False) not in online_keys for row in relevant_offline
    )
    return {
        "schema_version": "1.0",
        "status": "ok",
        "exact_matched_round_count": matched,
        "unmatched_online_round_count": len(online_rounds) - matched,
        "unmatched_offline_round_count": unmatched_offline,
        "matched_round_agreement_rate": agreements / matched if matched else None,
        "unmatched_rounds_are_mismatches": False,
        "rows": rows,
    }
