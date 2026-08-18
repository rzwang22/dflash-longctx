"""Deterministic CPU-only deployment freeze for Phase 2D.1 selectors."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from dflash.phase2b1.validation import sha256_file

from .events import RUNTIME_FEATURE_NAMES, policy_map
from .features import RUNTIME_FEATURE_SCHEMA_VERSION, feature_definitions
from .runtime import SELECTOR_CONFIG_VERSION, validate_frozen_selector, with_config_hash
from .selectors import choose_best_fixed_config, train_decision_stump

PHASE2D1_BASELINE_SHA = "e48c5cc1362b918a8bb6ceb9b6282768e5fa1166"
REQUIRED_PHASE2D1_FILES = (
    "selector-events.jsonl",
    "analysis-manifest.json",
    "selector-decision.json",
    "research-question-summary.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"line {number} is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSONL: {path}") from exc
    return rows


def phase2d1_artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def validate_phase2d1_freeze_inputs(
    phase2d1_dir: Path,
    *,
    rebuilt_events: list[dict[str, Any]],
    phase2d0: dict[str, Any],
    validated_capture: dict[str, Any],
) -> dict[str, Any]:
    directory = phase2d1_dir.expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Phase 2D.1A directory does not exist: {directory}")
    missing = [name for name in REQUIRED_PHASE2D1_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"Phase 2D.1A directory lacks required files: {missing}")
    manifest = _read_json(directory / "analysis-manifest.json")
    decision = _read_json(directory / "selector-decision.json")
    research = _read_json(directory / "research-question-summary.json")
    persisted_events = _read_jsonl(directory / "selector-events.jsonl")
    if manifest.get("status") != "ok":
        raise ValueError("Phase 2D.1A input is not successful")
    repository_sha = manifest.get("repository", {}).get("commit_sha")
    if repository_sha != PHASE2D1_BASELINE_SHA:
        raise ValueError(
            f"Phase 2D.1A artifact commit {repository_sha!r} is not {PHASE2D1_BASELINE_SHA}"
        )
    if manifest.get("decision") != "integrate_lightweight_selector_online":
        raise ValueError("Phase 2D.1A decision does not authorize lightweight selector freeze")
    if decision.get("decision") != manifest.get("decision"):
        raise ValueError("Phase 2D.1A decision/manifest mismatch")
    if research.get("decision") != manifest.get("decision"):
        raise ValueError("Phase 2D.1A research-summary/manifest mismatch")
    if persisted_events != rebuilt_events:
        raise ValueError("Phase 2D.1A selector events do not match capture/Phase 2D.0 replay")
    manifest_phase2d0 = manifest.get("phase2d0_input", {})
    if manifest_phase2d0.get("directory_sha256") != phase2d0["directory_sha256"]:
        raise ValueError("Phase 2D.1A/Phase 2D.0 directory provenance mismatch")
    expected_capture_hashes = sorted(
        row["capture_directory_sha256"] for row in validated_capture["inputs"]
    )
    manifest_capture_hashes = sorted(
        row["capture_directory_sha256"] for row in manifest.get("capture_inputs", [])
    )
    if manifest_capture_hashes != expected_capture_hashes:
        raise ValueError("Phase 2D.1A/capture provenance mismatch")
    return {
        "directory": str(directory),
        "manifest": manifest,
        "decision": decision,
        "research": research,
        "artifact_hashes": phase2d1_artifact_hashes(directory),
    }


def _deployment_policy(policy: dict[str, Any], budget: int) -> dict[str, Any]:
    return {
        "config_id": policy["config_id"],
        "policy": policy["policy"],
        "policy_variant": policy["policy_variant"],
        "policy_parameters": dict(policy["policy_parameters"]),
        "top_m": int(policy["top_m"]),
        "node_budget_cap": int(budget),
        "node_budget_includes_root": True,
        "policy_construction_uses_target_information": False,
    }


def freeze_selector_config(
    events: list[dict[str, Any]],
    *,
    phase2d1: dict[str, Any],
    phase2d0: dict[str, Any],
    validated_capture: dict[str, Any],
) -> dict[str, Any]:
    """Fit all deployment rules on all validated source documents."""

    source_ids = sorted({event["source_cluster_id"] for event in events})
    rules = {}
    for budget in (16, 32):
        rows = [event for event in events if int(event["node_budget_cap"]) == budget]
        if not rows:
            raise ValueError(f"no eligible selector events for N={budget}")
        global_config = choose_best_fixed_config(rows)
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in rows:
            grouped[int(event["nominal_prompt_length"])].append(event)
        context_mapping = {
            str(context): choose_best_fixed_config(values)
            for context, values in sorted(grouped.items())
        }
        manifest_config = phase2d1["manifest"]["configuration"]
        stump = train_decision_stump(
            rows,
            minimum_leaf_support=int(manifest_config["minimum_leaf_support"]),
            minimum_leaf_source_support=int(
                manifest_config["minimum_leaf_source_support"]
            ),
            maximum_thresholds_per_feature=int(
                manifest_config["stump_max_thresholds_per_feature"]
            ),
        )
        if stump["stump_available"]:
            frozen_stump = {
                "stump_available": True,
                "feature_name": stump["feature"],
                "feature_definition": feature_definitions()[stump["feature"]],
                "threshold": float(stump["threshold"]),
                "comparison_operator": "<=",
                "left_policy_config": stump["left_config_id"],
                "right_policy_config": stump["right_config_id"],
                "minimum_leaf_support": int(stump["minimum_leaf_support"]),
                "minimum_leaf_source_support": int(
                    stump["minimum_leaf_source_support"]
                ),
                "left_training_round_count": int(stump["left_training_round_count"]),
                "right_training_round_count": int(stump["right_training_round_count"]),
                "training_source_count": len(source_ids),
                "target_label_used_only_during_freeze": True,
            }
        else:
            frozen_stump = {
                "stump_available": False,
                "unavailable_reason": stump["unavailable_reason"],
                "global_fallback_config_id": stump["global_fallback_config_id"],
                "minimum_leaf_support": int(stump["minimum_leaf_support"]),
                "minimum_leaf_source_support": int(
                    stump["minimum_leaf_source_support"]
                ),
                "training_source_count": len(source_ids),
            }
        policies = policy_map(rows[0])
        expected_ids = set(policies)
        if any(set(policy_map(row)) != expected_ids for row in rows):
            raise ValueError(f"policy configuration family changed within N={budget}")
        rules[str(budget)] = {
            "node_budget_cap": budget,
            "top_m": 16,
            "global_fixed_rule": {
                "config_id": global_config,
                "fit_source_count": len(source_ids),
                "selection_objective": "maximum_training_sum_coverage_then_minimum_actual_nodes_then_config_id",
            },
            "context_conditioned_rule": {
                "context_length_to_config_id": context_mapping,
                "global_fallback_config_id": global_config,
                "task_id_used": False,
            },
            "decision_stump_rule": frozen_stump,
            "fallback_rule": {
                "config_id": global_config,
                "reason": "deterministic_global_fixed_fallback_for_missing_or_non_finite_feature",
                "target_information_used": False,
            },
            "allowed_policy_configurations": [
                _deployment_policy(policies[config_id], budget)
                for config_id in sorted(policies)
            ],
        }
    capture_hashes = {
        row["task"]: row["capture_directory_sha256"]
        for row in validated_capture["inputs"]
    }
    config = {
        "schema_version": "1.0",
        "selector_config_version": SELECTOR_CONFIG_VERSION,
        "phase2d1_commit_sha": PHASE2D1_BASELINE_SHA,
        "phase2d1_artifact_hashes": phase2d1["artifact_hashes"],
        "phase2d0_directory_sha256": phase2d0["directory_sha256"],
        "phase2d0_artifact_hashes": phase2d0["artifact_sha256"],
        "capture_hashes": capture_hashes,
        "training_source_ids": source_ids,
        "training_source_count": len(source_ids),
        "runtime_feature_schema": {
            "schema_version": RUNTIME_FEATURE_SCHEMA_VERSION,
            "feature_names": list(RUNTIME_FEATURE_NAMES),
            "feature_definitions": feature_definitions(),
            "top16_entropy_is_full_vocabulary_entropy": False,
        },
        "top_m": 16,
        "node_budget_cap": [16, 32],
        "node_budget_rules": rules,
        "global_fixed_rule": {
            budget: rule["global_fixed_rule"] for budget, rule in rules.items()
        },
        "context_conditioned_rule": {
            budget: rule["context_conditioned_rule"] for budget, rule in rules.items()
        },
        "decision_stump_rule": {
            budget: rule["decision_stump_rule"] for budget, rule in rules.items()
        },
        "fallback_rule": {
            budget: rule["fallback_rule"] for budget, rule in rules.items()
        },
        "allowed_policy_configurations": {
            budget: rule["allowed_policy_configurations"]
            for budget, rule in rules.items()
        },
        "target_blind": True,
        "task_id_used_as_feature": False,
        "target_dependent_fields_present": False,
        "per_round_target_labels_persisted": False,
        "runtime_may_access_target_logits": False,
        "runtime_may_access_accepted_depth": False,
    }
    return validate_frozen_selector(with_config_hash(config))
