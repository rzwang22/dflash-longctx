"""Pure-Python frozen selector validation and target-blind runtime decisions."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

SELECTOR_CONFIG_VERSION = "phase2d1b-frozen-selector-v1"
ONLINE_SELECTOR_MODES = (
    "top1_packed_tree",
    "global_fixed",
    "context_rule",
    "decision_stump",
)
SUPPORTED_POLICIES = {
    "top1_chain",
    "score_beam",
    "confidence_gated",
    "layerwise_beam",
}
FORBIDDEN_RUNTIME_KEYS = {
    "target_token_ids",
    "target_logits",
    "target_hidden_states",
    "accepted_depth",
    "accepted_draft_token_count",
    "oracle_path",
    "gap_class",
    "posterior",
    "correction_token",
    "final_output",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def selector_config_sha256(config_without_hash: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(config_without_hash)).hexdigest()


def with_config_hash(config: dict[str, Any]) -> dict[str, Any]:
    value = dict(config)
    value.pop("config_sha256", None)
    value["config_sha256"] = selector_config_sha256(value)
    return value


def validate_frozen_selector(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "selector_config_version",
        "phase2d1_commit_sha",
        "phase2d1_artifact_hashes",
        "capture_hashes",
        "training_source_ids",
        "runtime_feature_schema",
        "top_m",
        "node_budget_rules",
        "target_blind",
        "task_id_used_as_feature",
        "target_dependent_fields_present",
        "config_sha256",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"frozen selector lacks required fields: {missing}")
    unhashed = dict(config)
    actual = unhashed.pop("config_sha256")
    expected = selector_config_sha256(unhashed)
    if actual != expected:
        raise ValueError("frozen selector config SHA-256 mismatch")
    if config["selector_config_version"] != SELECTOR_CONFIG_VERSION:
        raise ValueError("unsupported frozen selector config version")
    if config["top_m"] != 16:
        raise ValueError("frozen selector must use Top-16")
    if not config["target_blind"] or config["task_id_used_as_feature"]:
        raise ValueError("frozen selector violates target-blind/task-agnostic contract")
    if config["target_dependent_fields_present"]:
        raise ValueError("frozen selector records target-dependent runtime fields")
    if sorted(map(int, config["node_budget_rules"])) != [16, 32]:
        raise ValueError("frozen selector requires separate N=16 and N=32 rules")
    if len(config["training_source_ids"]) != len(set(config["training_source_ids"])):
        raise ValueError("duplicate frozen-selector training source ID")
    for budget_text, rule in config["node_budget_rules"].items():
        budget = int(budget_text)
        allowlist = rule["allowed_policy_configurations"]
        if not allowlist:
            raise ValueError(f"empty policy allowlist for N={budget}")
        ids = {row["config_id"] for row in allowlist}
        if len(ids) != len(allowlist):
            raise ValueError(f"duplicate policy config for N={budget}")
        for row in allowlist:
            if row["policy"] not in SUPPORTED_POLICIES:
                raise ValueError(f"unsupported frozen policy {row['policy']!r}")
            if int(row["node_budget_cap"]) != budget or int(row["top_m"]) != 16:
                raise ValueError("allowlisted policy has inconsistent budget/Top-M")
        referenced = {rule["global_fixed_rule"]["config_id"]}
        referenced.update(rule["context_conditioned_rule"]["context_length_to_config_id"].values())
        referenced.add(rule["context_conditioned_rule"]["global_fallback_config_id"])
        stump = rule["decision_stump_rule"]
        if stump["stump_available"]:
            referenced.update((stump["left_policy_config"], stump["right_policy_config"]))
        else:
            referenced.add(stump["global_fallback_config_id"])
        referenced.add(rule["fallback_rule"]["config_id"])
        unknown = referenced - ids
        if unknown:
            raise ValueError(f"frozen rules reference policies outside allowlist: {sorted(unknown)}")
    return config


def _budget_rule(config: dict[str, Any], node_budget: int) -> dict[str, Any]:
    validate_frozen_selector(config)
    try:
        return config["node_budget_rules"][str(int(node_budget))]
    except KeyError as exc:
        raise ValueError(f"no frozen selector rule for N={node_budget}") from exc


def _policy(rule: dict[str, Any], config_id: str) -> dict[str, Any]:
    matches = [
        value
        for value in rule["allowed_policy_configurations"]
        if value["config_id"] == config_id
    ]
    if len(matches) != 1:
        raise ValueError(f"selected config {config_id!r} is not uniquely allowlisted")
    return matches[0]


def select_policy(
    config: dict[str, Any],
    *,
    mode: str,
    node_budget: int,
    context_length: int,
    runtime_features: dict[str, float] | None,
    decision_boundary_epsilon: float = 1.0e-6,
) -> dict[str, Any]:
    """Select a frozen policy using deployment-known, target-blind inputs only."""

    if mode not in ONLINE_SELECTOR_MODES:
        raise ValueError(f"unsupported online selector mode: {mode}")
    rule = _budget_rule(config, node_budget)
    if runtime_features is not None:
        audit_runtime_inputs(runtime_features)
        expected_features = set(config["runtime_feature_schema"]["feature_names"])
        actual_features = set(runtime_features)
        if actual_features != expected_features:
            raise ValueError(
                "runtime feature keys do not match the frozen canonical schema: "
                f"missing={sorted(expected_features - actual_features)} "
                f"unexpected={sorted(actual_features - expected_features)}"
            )
    fallback = rule["fallback_rule"]["config_id"]
    feature_name = feature_value = threshold = distance = comparison_result = None
    fallback_used = False
    fallback_reason = None
    if mode == "top1_packed_tree":
        matches = [
            row
            for row in rule["allowed_policy_configurations"]
            if row["policy"] == "top1_chain"
        ]
        if len(matches) != 1:
            raise ValueError("frozen selector requires exactly one Top-1 policy")
        selected = matches[0]["config_id"]
    elif mode == "global_fixed":
        selected = rule["global_fixed_rule"]["config_id"]
    elif mode == "context_rule":
        mapping = rule["context_conditioned_rule"]["context_length_to_config_id"]
        selected = mapping.get(str(int(context_length)))
        if selected is None:
            selected = rule["context_conditioned_rule"]["global_fallback_config_id"]
            fallback_used = True
            fallback_reason = "unseen_context_length"
    else:
        stump = rule["decision_stump_rule"]
        if not stump["stump_available"]:
            selected = stump["global_fallback_config_id"]
            fallback_used = True
            fallback_reason = "stump_unavailable"
        else:
            feature_name = stump["feature_name"]
            threshold = float(stump["threshold"])
            raw_value = None if runtime_features is None else runtime_features.get(feature_name)
            if raw_value is None:
                selected = fallback
                fallback_used = True
                fallback_reason = "missing_feature"
            else:
                feature_value = float(raw_value)
                if not math.isfinite(feature_value):
                    selected = fallback
                    fallback_used = True
                    fallback_reason = "non_finite_feature"
                else:
                    distance = feature_value - threshold
                    comparison_result = feature_value <= threshold
                    selected = (
                        stump["left_policy_config"]
                        if comparison_result
                        else stump["right_policy_config"]
                    )
    selected_policy = _policy(rule, selected)
    return {
        "selector_mode": mode,
        "feature_name": feature_name,
        "feature_value": feature_value,
        "threshold": threshold,
        "distance_to_threshold": distance,
        "comparison_operator": "<=" if feature_name is not None else None,
        "comparison_result": comparison_result,
        "decision_boundary_nearby": (
            abs(distance) <= decision_boundary_epsilon if distance is not None else False
        ),
        "decision_boundary_epsilon": decision_boundary_epsilon,
        "selected_policy_config": selected,
        "selected_policy": selected_policy,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "selector_config_hash": config["config_sha256"],
        "decision_made_before_target_verification": True,
        "target_dependent_runtime_input_used": False,
        "runtime_input_keys": sorted(runtime_features or {}),
    }


def audit_runtime_inputs(runtime_inputs: dict[str, Any]) -> None:
    forbidden = sorted(set(runtime_inputs) & FORBIDDEN_RUNTIME_KEYS)
    if forbidden:
        raise ValueError(f"target-dependent fields entered selector runtime: {forbidden}")
