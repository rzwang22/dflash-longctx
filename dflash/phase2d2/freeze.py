"""Deterministic pre-evaluation refinement-configuration freezing."""

from __future__ import annotations

import hashlib
import json
from typing import Any

FROZEN_SCHEMA_VERSION = "phase2d2a-frozen-refinement-v1"


def config_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("config_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def refinement_family_id(row: dict[str, Any]) -> str:
    weight = format(float(row["refinement_weight"]), ".8g")
    return (
        f"{row['candidate_merge_mode']}-d{int(row['refine_depth'])}-"
        f"p{int(row['refined_prefix_budget'])}-"
        f"r{int(row['conditional_top_r'])}-w{weight}"
    )


def validate_frozen_refinement(value: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "source_artifacts",
        "selection_rule",
        "configurations",
        "maximum_selected_configuration_count",
        "target_outputs_used_for_final_online_selection",
        "held_out_evaluation_results_used",
        "manual_override_used",
        "config_sha256",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"frozen refinement config lacks fields: {missing}")
    if value["schema_version"] != FROZEN_SCHEMA_VERSION:
        raise ValueError("unsupported frozen refinement schema")
    if value["config_sha256"] != config_sha256(value):
        raise ValueError("frozen refinement config SHA-256 mismatch")
    configs = value["configurations"]
    if len(configs) > 3 or value["maximum_selected_configuration_count"] != 3:
        raise ValueError("at most three Phase 2D.2A configurations may be frozen")
    if sum(row["candidate_merge_mode"] == "conditional_rerank" for row in configs) > 1:
        raise ValueError("at most one rerank configuration may be frozen")
    if sum(row["candidate_merge_mode"] == "union" for row in configs) > 2:
        raise ValueError("at most two union configurations may be frozen")
    if value["target_outputs_used_for_final_online_selection"]:
        raise ValueError("held-out online target outputs entered config freezing")
    if value["held_out_evaluation_results_used"]:
        raise ValueError("held-out evaluation results entered config freezing")
    if value["manual_override_used"]:
        raise ValueError("manual refinement configuration override is forbidden")
    identifiers = [row.get("configuration_id") for row in configs]
    if any(not identifier for identifier in identifiers) or len(identifiers) != len(
        set(identifiers)
    ):
        raise ValueError("frozen refinement configuration IDs must be unique")
    required_configuration_fields = {
        "configuration_id",
        "candidate_merge_mode",
        "base_top_m",
        "conditional_top_r",
        "refine_depth",
        "refined_prefix_budget",
        "refinement_weight",
    }
    if any(required_configuration_fields - set(row) for row in configs):
        raise ValueError("frozen refinement configuration is incomplete")
    return value


def freeze_refinement_config(
    candidates: list[dict[str, Any]], source_artifacts: list[dict[str, Any]]
) -> dict[str, Any]:
    eligible = [
        row
        for row in candidates
        if row["candidate_merge_mode"] in {"conditional_rerank", "union"}
        and row["mechanism_correctness_passed"]
    ]
    eligible.sort(
        key=lambda row: (
            -int(row["refined_only_target_recovery_count"]),
            -float(row["accepted_tokens_per_verified_node"]),
            float(row["conditional_remask_cuda_median_ms"]),
            row["configuration_id"],
        )
    )
    unique_families = {}
    for row in eligible:
        unique_families.setdefault(refinement_family_id(row), row)
    eligible = list(unique_families.values())
    selected = []
    for mode, maximum in (("conditional_rerank", 1), ("union", 2)):
        selected.extend(
            [row for row in eligible if row["candidate_merge_mode"] == mode][:maximum]
        )
    selected.sort(key=lambda row: row["configuration_id"])
    configurations = [
        {
            key: (refinement_family_id(row) if key == "configuration_id" else row[key])
            for key in (
                "configuration_id",
                "candidate_merge_mode",
                "base_top_m",
                "conditional_top_r",
                "refine_depth",
                "refined_prefix_budget",
                "refinement_weight",
            )
        }
        for row in selected
    ]
    result = {
        "schema_version": FROZEN_SCHEMA_VERSION,
        "source_artifacts": source_artifacts,
        "selection_rule": (
            "per merge family: descending single-round refined-only target recovery, "
            "descending accepted tokens per verified node, ascending conditional CUDA "
            "latency, lexicographic configuration ID; select <=1 rerank and <=2 union"
        ),
        "configurations": configurations,
        "maximum_selected_configuration_count": 3,
        "target_outputs_used_for_final_online_selection": False,
        "held_out_evaluation_results_used": False,
        "manual_override_used": False,
    }
    result["config_sha256"] = config_sha256(result)
    return validate_frozen_refinement(result)
