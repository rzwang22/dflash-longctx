"""Online selector metrics, comparisons, and deterministic engineering decision."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

DECISIONS = (
    "adopt_global_fixed_n16",
    "adopt_context_rule_n16",
    "adopt_decision_stump_n16",
    "adopt_global_fixed_n32",
    "adopt_context_rule_n32",
    "adopt_decision_stump_n32",
    "offline_gain_not_realized_online",
    "correctness_failure",
    "insufficient_independent_evidence",
)


def distribution(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    ordered = sorted(finite)
    position = (len(ordered) - 1) * 0.9
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    p90 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "p90": p90,
        "min": min(finite),
        "max": max(finite),
    }


def selector_round_record(
    source: dict[str, Any], *, mode: str, node_budget: int
) -> dict[str, Any]:
    selector = source.get("online_selector")
    timing = source.get("timing", {})
    cpu = selector.get("cpu_timing", {}) if selector else {}
    accepted = int(source["accepted_draft_token_count"])
    committed = int(source["committed_token_count"])
    if committed != 1 + accepted:
        raise ValueError("committed token count must include root plus accepted draft tokens")
    actual_nodes = int(source["actual_node_count"])
    if actual_nodes > node_budget:
        raise ValueError("actual constructed node count exceeds node-budget cap")
    selector_latency = sum(
        float(cpu.get(name, 0.0) or 0.0)
        for name in (
            "draft_topk_device_to_host_sync_cpu_ms",
            "feature_extraction_cpu_ms",
            "decision_cpu_ms",
            "canonical_policy_construction_cpu_ms",
            "tree_tensor_pack_host_to_device_cpu_ms",
        )
    )
    return {
        "schema_version": "1.0",
        "task": source["task"],
        "sample_id": source["sample_id"],
        "source_sample_id": source["source_sample_id"],
        "nominal_prompt_length": int(source["nominal_prompt_length"]),
        "actual_prompt_length": int(source["actual_prompt_length"]),
        "profiling_repeat": int(source["profiling_repeat"]),
        "round_index": int(source["round_index"]),
        "generated_root_offset": int(source["generated_root_offset"]),
        "root_token_id": int(source["tree_metadata"]["nodes"][0]["token_id"]),
        "source_text_sha256": source["source_text_sha256"],
        "accepted_draft_token_count": accepted,
        "committed_token_count": committed,
        "terminal": bool(source["terminal_round"]),
        "right_censored": bool(source.get("truncated_token_count", 0)),
        "truncated_token_count": int(source.get("truncated_token_count", 0)),
        "actual_constructed_node_count": actual_nodes,
        "node_budget_cap": int(node_budget),
        "same_budget_cap": True,
        "same_actual_node_count_as_top1": actual_nodes == 16,
        "selected_policy_config": (
            selector["selected_policy_config"] if selector else "top1_chain-b16"
        ),
        "selector_mode": mode,
        "selector_feature_value": selector.get("feature_value") if selector else None,
        "selector_decision": selector,
        "fallback_used": bool(selector and selector["fallback_used"]),
        "decision_boundary_nearby": bool(
            selector and selector["decision_boundary_nearby"]
        ),
        "round_latency": timing.get("round_cuda_elapsed_ms"),
        "draft_latency": timing.get("total_draft_cuda_elapsed_ms"),
        "selector_latency": selector_latency,
        "verification_latency": timing.get("total_verification_cuda_elapsed_ms"),
        "KV_compaction_latency": timing.get("kv_compaction_cuda_elapsed_ms"),
        "draft_topk_device_to_host_sync_cpu_ms": cpu.get(
            "draft_topk_device_to_host_sync_cpu_ms", 0.0
        ),
        "host_to_device_transfer_cpu_ms": cpu.get(
            "tree_tensor_pack_host_to_device_cpu_ms", 0.0
        ),
        "selector_runtime_included_in_formal_control": True,
        "component_timing_source": "profiling_path",
    }


def aggregate_selector_results(
    samples: list[dict[str, Any]], rounds: list[dict[str, Any]]
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"samples": [], "rounds": []}
    )
    for row in samples:
        groups[(row["task"], row["nominal_prompt_length"], row["selector_mode"], row["node_budget_cap"])]["samples"].append(row)
    for row in rounds:
        groups[(row["task"], row["nominal_prompt_length"], row["selector_mode"], row["node_budget_cap"])]["rounds"].append(row)
    output = []
    for (task, context, mode, budget), values in sorted(groups.items()):
        sample_rows = values["samples"]
        round_rows = values["rounds"]
        primary_rounds = [row for row in round_rows if row["profiling_repeat"] == 0]
        eligible = [row for row in primary_rounds if not row["terminal"] and not row["right_censored"]]
        committed = sum(row["committed_token_count"] for row in eligible)
        nodes = sum(row["actual_constructed_node_count"] for row in eligible)
        control_wall = [
            repeat["cpu_wall_elapsed_ms"]
            for row in sample_rows
            for repeat in row["control_repeats"]
        ]
        returned = sum(row["generated_token_count"] for row in sample_rows)
        output.append(
            {
                "schema_version": "1.0",
                "task": task,
                "nominal_prompt_length": context,
                "selector_mode": mode,
                "node_budget_cap": budget,
                "sample_count": len(sample_rows),
                "round_count": len(primary_rounds),
                "acceptance_eligible_round_count": len(eligible),
                "mean_non_terminal_accepted_draft_tokens": (
                    statistics.fmean(row["accepted_draft_token_count"] for row in eligible)
                    if eligible else None
                ),
                "non_terminal_accepted_draft_tokens": distribution(
                    row["accepted_draft_token_count"] for row in eligible
                ),
                "mean_non_terminal_committed_tokens": (
                    statistics.fmean(row["committed_token_count"] for row in eligible)
                    if eligible else None
                ),
                "non_terminal_committed_tokens": distribution(
                    row["committed_token_count"] for row in eligible
                ),
                "actual_constructed_node_count": distribution(
                    row["actual_constructed_node_count"] for row in primary_rounds
                ),
                "accepted_draft_tokens_per_verified_node": (
                    sum(row["accepted_draft_token_count"] for row in eligible) / nodes
                    if nodes else None
                ),
                "committed_tokens_per_verified_node": committed / nodes if nodes else None,
                "verification_ms_per_round": distribution(
                    row["verification_latency"] for row in eligible
                ),
                "round_cuda_latency_ms": distribution(
                    row["round_latency"] for row in primary_rounds
                ),
                "draft_cuda_latency_ms": distribution(
                    row["draft_latency"] for row in primary_rounds
                ),
                "verification_cuda_latency_ms": distribution(
                    row["verification_latency"] for row in primary_rounds
                ),
                "kv_compaction_cuda_latency_ms": distribution(
                    row["KV_compaction_latency"] for row in primary_rounds
                ),
                "verification_ms_per_committed_token": (
                    sum(float(row["verification_latency"] or 0.0) for row in eligible)
                    / committed if committed else None
                ),
                "generation_wall_latency_ms": distribution(control_wall),
                "rounds_per_generation": distribution(
                    row.get("rounds_per_generation") for row in sample_rows
                ),
                "latency_per_returned_token_ms": (
                    sum(control_wall) / (returned * len(sample_rows[0]["control_repeats"]))
                    if returned and sample_rows else None
                ),
                "throughput_tokens_per_second": (
                    1000.0 / (sum(control_wall) / (returned * len(sample_rows[0]["control_repeats"])))
                    if returned and sample_rows and sum(control_wall) else None
                ),
                "selector_latency_cpu_ms": distribution(
                    row["selector_latency"] for row in primary_rounds
                ),
                "selector_host_sync_cpu_ms": distribution(
                    row["draft_topk_device_to_host_sync_cpu_ms"] for row in primary_rounds
                ),
                "fallback_rate": (
                    sum(row["fallback_used"] for row in primary_rounds) / len(primary_rounds)
                    if primary_rounds else None
                ),
                "threshold_nearby_rate": (
                    sum(row["decision_boundary_nearby"] for row in primary_rounds) / len(primary_rounds)
                    if primary_rounds else None
                ),
                "selected_policy_distribution": dict(
                    Counter(row["selected_policy_config"] for row in primary_rounds)
                ),
                "truncated_token_count": sum(row["truncated_token_count"] for row in primary_rounds),
                "terminal_round_count": sum(row["terminal"] for row in primary_rounds),
                "peak_allocated_bytes": max((row.get("peak_allocated_bytes", 0) for row in sample_rows), default=0),
                "peak_reserved_bytes": max((row.get("peak_reserved_bytes", 0) for row in sample_rows), default=0),
                "acceptance_scope": "non_terminal_non_truncated_rounds",
                "formal_latency_source": "unprofiled_control_cpu_wall_including_selector_sync",
                "component_timing_source": "separate_profiling_path",
            }
        )
    source_macro = _macro(rounds, key="source_sample_id")
    task_macro = _macro(rounds, key="task")
    return {
        "schema_version": "1.0",
        "status": "ok",
        "sample_level_record_count": len(samples),
        "round_level_record_count": len(rounds),
        "aggregates": output,
        "source_micro": output,
        "source_macro": source_macro,
        "task_macro": task_macro,
        "terminal_and_truncation_excluded_from_primary_acceptance": True,
    }


def _macro(rounds: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rounds:
        if row["profiling_repeat"] == 0 and not row["terminal"] and not row["right_censored"]:
            grouped[(row["task"], row["nominal_prompt_length"], row["selector_mode"], row["node_budget_cap"], str(row[key]))].append(row)
    return [
        {
            "task": group[0],
            "nominal_prompt_length": group[1],
            "selector_mode": group[2],
            "node_budget_cap": group[3],
            f"{key}_count": len({value[4] for value in grouped if value[:4] == group[:4]}),
            "macro_mean_accepted_draft_tokens": statistics.fmean(
                statistics.fmean(row["accepted_draft_token_count"] for row in values)
                for candidate, values in grouped.items() if candidate[:4] == group[:4]
            ),
        }
        for group in sorted({value[:4] for value in grouped})
    ]


def build_decision(
    summary: dict[str, Any],
    *,
    correctness_passed: bool,
    independent_evidence: bool,
    minimum_accepted_gain: float,
    minimum_latency_improvement_fraction: float,
    stump_minimum_additional_improvement_fraction: float,
) -> dict[str, Any]:
    rules = {
        "minimum_accepted_gain": minimum_accepted_gain,
        "minimum_latency_improvement_fraction": minimum_latency_improvement_fraction,
        "stump_minimum_additional_improvement_fraction": stump_minimum_additional_improvement_fraction,
        "engineering_gates_not_statistical_significance": True,
    }
    if not correctness_passed:
        return {"decision": "correctness_failure", "decision_rules": rules, "eligible_candidates": []}
    if not independent_evidence:
        return {"decision": "insufficient_independent_evidence", "decision_rules": rules, "eligible_candidates": []}
    rows = summary["aggregates"]
    by_key = {
        (
            row["selector_mode"],
            row["node_budget_cap"],
            row["task"],
            row["nominal_prompt_length"],
        ): row
        for row in rows
    }
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["selector_mode"] in {
            "global_fixed",
            "context_rule",
            "decision_stump",
        }:
            grouped[(row["selector_mode"], row["node_budget_cap"])].append(row)
    candidates = []
    for (mode, budget), config_rows in sorted(grouped.items()):
        paired = []
        for row in config_rows:
            top1 = by_key.get(
                (
                    "top1_packed_tree",
                    16,
                    row["task"],
                    row["nominal_prompt_length"],
                )
            )
            if (
                top1 is not None
                and row["mean_non_terminal_accepted_draft_tokens"] is not None
                and top1["mean_non_terminal_accepted_draft_tokens"] is not None
            ):
                paired.append((row, top1))
        if not paired:
            continue
        accepted_gain = statistics.fmean(
            row["mean_non_terminal_accepted_draft_tokens"]
            - top1["mean_non_terminal_accepted_draft_tokens"]
            for row, top1 in paired
        )
        selector_latency = statistics.fmean(
            row["latency_per_returned_token_ms"] for row, _ in paired
        )
        top1_latency = statistics.fmean(
            top1["latency_per_returned_token_ms"] for _, top1 in paired
        )
        latency_improvement = 1.0 - selector_latency / top1_latency
        if (
            accepted_gain > minimum_accepted_gain
            and latency_improvement > minimum_latency_improvement_fraction
        ):
            candidates.append(
                {
                    "selector_mode": mode,
                    "node_budget_cap": budget,
                    "paired_task_context_count": len(paired),
                    "accepted_gain_vs_top1": accepted_gain,
                    "latency_per_returned_token_ms": selector_latency,
                    "top1_latency_per_returned_token_ms": top1_latency,
                    "latency_improvement_fraction_vs_top1": latency_improvement,
                }
            )
    if not candidates:
        return {"decision": "offline_gain_not_realized_online", "decision_rules": rules, "eligible_candidates": []}
    best = min(candidates, key=lambda row: (row["latency_per_returned_token_ms"], row["node_budget_cap"], row["selector_mode"]))
    if best["selector_mode"] == "decision_stump":
        context = [row for row in candidates if row["selector_mode"] == "context_rule" and row["node_budget_cap"] == best["node_budget_cap"]]
        if context:
            simpler = min(context, key=lambda row: row["latency_per_returned_token_ms"])
            relative = 1.0 - best["latency_per_returned_token_ms"] / simpler["latency_per_returned_token_ms"]
            if relative <= stump_minimum_additional_improvement_fraction:
                best = simpler
    decision = f"adopt_{best['selector_mode']}_n{best['node_budget_cap']}"
    if decision not in DECISIONS:
        raise RuntimeError("automatic decision escaped the registered enum")
    return {"decision": decision, "decision_rules": rules, "eligible_candidates": candidates, "selected_candidate": best}
