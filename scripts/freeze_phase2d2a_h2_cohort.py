#!/usr/bin/env python3
"""Inventory and freeze the target-blind Phase 2D.2A-H2 held-out cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase1b.workload import (
    MAX_NOMINAL_SHORTFALL,
    _prefix_at_or_below,
    load_jsonl_records,
    prepare_longbench_records,
    prompt_template_metadata,
    sha256_text,
)
from scripts.aggregate_phase2d2a_h_known import (
    BASE_FAMILY,
    FROZEN_REFINEMENT_FAMILIES,
    KNOWN_TASK,
)
from scripts.check_phase2d2a_h11_acceptance import check_h11_output
from scripts.collect_environment import DEFAULT_TARGET_MODEL, DEFAULT_TARGET_REVISION

SCHEMA_VERSION = "1.0"
PHASE = "2D.2A-H2.0"
STATUS = "frozen_not_executed"
SELECTION_SEED = "phase2d2a-h2-heldout-v1"
INVENTORY_LENGTHS = (1024, 4096, 8192, 16384, 32768)
LONG_CONTEXT_ORDER = (16384, 8192, 4096)
SHORT_CONTEXT = 1024
TASKS = ("gov_report", "qasper", "hotpotqa")
INDEPENDENT_TASKS = ("qasper", "hotpotqa")
TASK_SAMPLE_COUNTS = {"gov_report": 3, "qasper": 3, "hotpotqa": 8}
TASK_ROLES = {
    "gov_report": "same_task_replication_only",
    "qasper": "independent_held_out_task",
    "hotpotqa": "independent_held_out_task",
}
FAMILIES = (BASE_FAMILY, *FROZEN_REFINEMENT_FAMILIES)
PRIMARY_NODE_BUDGET = 16
SECONDARY_NODE_BUDGET = 32
VERIFICATION_MODE = "always_fixed_shape_recheck"
CANDIDATE_LABEL_AUTHORITY = "original_dflash_reference"
CANDIDATE_LABEL_SOURCE = "original_dflash_absolute_continuation"

ARTIFACTS = (
    "phase2d2a-h2-heldout-inventory.json",
    "phase2d2a-h2-heldout-inventory.csv",
    "phase2d2a-h2-exclusion-provenance.json",
    "phase2d2a-h2-frozen-cohort.json",
    "phase2d2a-h2-primary-matrix.json",
    "phase2d2a-h2-secondary-matrix.json",
    "phase2d2a-h2-aggregation-rules.json",
    "phase2d2a-h2-freeze-manifest.json",
)
SCHEMAS = {
    "phase2d2a-h2-heldout-inventory.json": (
        "phase2d2a-h2-heldout-inventory.schema.json"
    ),
    "phase2d2a-h2-exclusion-provenance.json": (
        "phase2d2a-h2-exclusion-provenance.schema.json"
    ),
    "phase2d2a-h2-frozen-cohort.json": ("phase2d2a-h2-frozen-cohort.schema.json"),
    "phase2d2a-h2-primary-matrix.json": "phase2d2a-h2-matrix.schema.json",
    "phase2d2a-h2-secondary-matrix.json": "phase2d2a-h2-matrix.schema.json",
    "phase2d2a-h2-aggregation-rules.json": (
        "phase2d2a-h2-aggregation-rules.schema.json"
    ),
    "phase2d2a-h2-freeze-manifest.json": ("phase2d2a-h2-freeze-manifest.schema.json"),
}

EXECUTION_IDENTITY_ARTIFACT_NAMES = frozenset(
    {
        "samples.jsonl",
        "subrun-index.jsonl",
        "exact-case-repeats.jsonl",
        "cohort-subruns.jsonl",
    }
)
IDENTITY_FIELDS_READ = (
    "task",
    "longbench_task",
    "sample_id",
    "source_sample_id",
    "source_text_sha256",
)
FORBIDDEN_EXCLUSION_FIELDS = (
    "generated_output_token_ids",
    "accepted_draft_token_count",
    "candidate_quality",
    "latency",
    "target_labels",
)


class EligibilityError(RuntimeError):
    """Raised before a cohort is frozen when requested eligibility is absent."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc


def _jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL: {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"JSONL row is not an object: {path}:{line_number}")
            yield line_number, row


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_schema(payload: Any, schema_name: str) -> None:
    schema = _json(REPO_ROOT / "docs" / "schemas" / schema_name)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    clean = sorted(int(value) for value in values)
    if not clean:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    position = max(0, (9 * len(clean) + 9) // 10 - 1)
    return {
        "count": len(clean),
        "min": clean[0],
        "median": float(statistics.median(clean)),
        "p90": float(clean[position]),
        "max": clean[-1],
    }


def _infer_task(row: dict[str, Any]) -> str | None:
    task = row.get("task") or row.get("longbench_task")
    if task in TASKS:
        return str(task)
    sample_id = row.get("sample_id")
    if isinstance(sample_id, str):
        for candidate in TASKS:
            if sample_id.startswith(f"{candidate}-"):
                return candidate
    return None


def identity_from_execution_record(row: dict[str, Any]) -> dict[str, Any] | None:
    """Read only physical-identity fields from one prior execution record."""

    task = _infer_task(row)
    if task is None:
        return None
    sample_id = row.get("sample_id")
    source_sample_id = row.get("source_sample_id")
    source_hash = row.get("source_text_sha256")
    identity = {
        "task": task,
        "sample_id": str(sample_id) if sample_id not in (None, "") else None,
        "source_sample_id": (
            str(source_sample_id) if source_sample_id not in (None, "") else None
        ),
        "source_text_sha256": (
            str(source_hash) if source_hash not in (None, "") else None
        ),
    }
    if not any(identity[field] for field in identity if field != "task"):
        return None
    return identity


def collect_exclusion_provenance(result_roots: Sequence[Path]) -> dict[str, Any]:
    """Collect target-blind physical identities from prior execution artifacts."""

    roots = [path.expanduser().resolve() for path in result_roots]
    artifact_rows = []
    observed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for root in roots:
        if not root.exists():
            raise ValueError(f"previous result root does not exist: {root}")
        paths = sorted(
            path
            for path in root.rglob("*.jsonl")
            if path.name in EXECUTION_IDENTITY_ARTIFACT_NAMES
        )
        for path in paths:
            record_count = 0
            identity_count = 0
            for line_number, row in _jsonl(path):
                record_count += 1
                identity = identity_from_execution_record(row)
                if identity is None:
                    continue
                identity_count += 1
                key = (
                    identity["task"],
                    identity["sample_id"],
                    identity["source_sample_id"],
                    identity["source_text_sha256"],
                )
                entry = observed.setdefault(
                    key,
                    {
                        **identity,
                        "evidence": [],
                    },
                )
                entry["evidence"].append(
                    {"artifact_path": str(path), "line_number": line_number}
                )
            artifact_rows.append(
                {
                    "artifact_path": str(path),
                    "artifact_sha256": _sha256_file(path),
                    "record_count": record_count,
                    "identity_record_count": identity_count,
                }
            )
    identities = sorted(
        observed.values(),
        key=lambda row: (
            row["task"],
            row["sample_id"] or "",
            row["source_sample_id"] or "",
            row["source_text_sha256"] or "",
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": "complete",
        "target_blind": True,
        "selection_or_exclusion_uses_model_outputs": False,
        "identity_fields_read": list(IDENTITY_FIELDS_READ),
        "forbidden_outcome_fields_read": False,
        "forbidden_outcome_fields": list(FORBIDDEN_EXCLUSION_FIELDS),
        "execution_identity_artifact_names": sorted(EXECUTION_IDENTITY_ARTIFACT_NAMES),
        "result_roots": [str(path) for path in roots],
        "scanned_artifacts": artifact_rows,
        "scanned_artifact_count": len(artifact_rows),
        "excluded_identity_count": len(identities),
        "excluded_identities": identities,
        "exclusion_completed_before_h2_model_output_exists": True,
    }


def exclusion_index(provenance: dict[str, Any]) -> dict[str, set[tuple[str, str]]]:
    result = {
        "sample_id": set(),
        "source_sample_id": set(),
        "source_text_sha256": set(),
    }
    for row in provenance["excluded_identities"]:
        task = row["task"]
        for field, values in result.items():
            value = row.get(field)
            if value:
                values.add((task, str(value)))
    return result


def validate_exclusion_preflight(provenance: dict[str, Any]) -> None:
    if provenance["scanned_artifact_count"] <= 0:
        raise ValueError("H2.0 exclusion scan found no prior execution artifacts")
    if provenance["excluded_identity_count"] <= 0:
        raise ValueError("H2.0 exclusion scan found no prior physical identities")
    known_h1_present = any(
        row["task"] == KNOWN_TASK
        and row.get("sample_id") == "gov_report-fd287bbfd3bdbdc9"
        for row in provenance["excluded_identities"]
    )
    if not known_h1_present:
        raise ValueError("H2.0 exclusion scan did not find the frozen H1 known sample")


def exclusion_matches(
    row: dict[str, Any], index: dict[str, set[tuple[str, str]]]
) -> list[str]:
    task = row["longbench_task"]
    matches = []
    for field, values in index.items():
        if (task, str(row[field])) in values:
            matches.append(field)
    return matches


def physical_selection_hash(
    *, task: str, source_sample_id: str, source_text_sha256: str
) -> str:
    stable_identity = f"{source_sample_id}\0{source_text_sha256}"
    return sha256_text(f"{SELECTION_SEED}\0{task}\0{stable_identity}")


def _eligible(row: dict[str, Any], length: int) -> bool:
    return (
        int(row["fixed_prompt_length"]) <= length
        and int(row["source_full_prompt_length"]) >= length
    )


def _eligible_sample_record(
    *, row: dict[str, Any], tokenizer: Any, task: str, length: int
) -> dict[str, Any]:
    context_prefix, input_ids = _prefix_at_or_below(
        tokenizer,
        task=task,
        context=row["context"],
        query=row["query"],
        nominal_length=length,
    )
    shortfall = length - len(input_ids)
    if shortfall > MAX_NOMINAL_SHORTFALL:
        raise ValueError(
            f"sample {row['sample_id']} reaches only {len(input_ids)} tokens for "
            f"nominal length {length}; maximum shortfall is {MAX_NOMINAL_SHORTFALL}"
        )
    return {
        "sample_id": row["sample_id"],
        "source_sample_id": row["source_sample_id"],
        "source_text_sha256": row["source_text_sha256"],
        "source_index": row["source_index"],
        "document_token_length": row["document_token_length"],
        "source_full_prompt_length": row["source_full_prompt_length"],
        "fixed_prompt_length": row["fixed_prompt_length"],
        "selection_hash": physical_selection_hash(
            task=task,
            source_sample_id=row["source_sample_id"],
            source_text_sha256=row["source_text_sha256"],
        ),
        "actual_prompt_length": len(input_ids),
        "nominal_prompt_length_shortfall": shortfall,
        "context_prefix_characters": len(context_prefix),
        "context_prefix_sha256": sha256_text(context_prefix),
        "input_token_ids_sha256": _sha256_json(input_ids),
    }


def build_task_inventory(
    *,
    task: str,
    source_record_count: int,
    prepared: Sequence[dict[str, Any]],
    tokenizer: Any,
    exclusions: dict[str, set[tuple[str, str]]],
) -> dict[str, Any]:
    physical_keys = [
        (row["source_sample_id"], row["source_text_sha256"]) for row in prepared
    ]
    if len(set(physical_keys)) != len(physical_keys):
        raise ValueError(f"LongBench task {task} contains duplicate physical samples")
    if len({row["sample_id"] for row in prepared}) != len(prepared):
        raise ValueError(f"LongBench task {task} contains duplicate sample IDs")
    contexts = []
    for length in INVENTORY_LENGTHS:
        before = [row for row in prepared if _eligible(row, length)]
        after = [row for row in before if not exclusion_matches(row, exclusions)]
        excluded = [
            {
                "sample_id": row["sample_id"],
                "source_sample_id": row["source_sample_id"],
                "source_text_sha256": row["source_text_sha256"],
                "matched_identity_fields": exclusion_matches(row, exclusions),
            }
            for row in before
            if exclusion_matches(row, exclusions)
        ]
        eligible_samples = [
            _eligible_sample_record(
                row=row, tokenizer=tokenizer, task=task, length=length
            )
            for row in after
        ]
        eligible_samples.sort(key=lambda row: (row["selection_hash"], row["sample_id"]))
        contexts.append(
            {
                "nominal_prompt_length": length,
                "eligible_before_exclusion_count": len(before),
                "excluded_previous_sample_count": len(excluded),
                "eligible_sample_count": len(eligible_samples),
                "eligible_document_token_length_distribution": _distribution(
                    [row["document_token_length"] for row in eligible_samples]
                ),
                "eligible_sample_ids": [row["sample_id"] for row in eligible_samples],
                "eligible_source_sample_ids": [
                    row["source_sample_id"] for row in eligible_samples
                ],
                "eligible_source_text_sha256": [
                    row["source_text_sha256"] for row in eligible_samples
                ],
                "excluded_samples": excluded,
                "eligible_samples": eligible_samples,
            }
        )
    return {
        "task": task,
        "task_role": TASK_ROLES[task],
        "requested_physical_sample_count": TASK_SAMPLE_COUNTS[task],
        "source_record_count": source_record_count,
        "usable_record_count": len(prepared),
        "unusable_record_count": source_record_count - len(prepared),
        "source_document_token_length_distribution": _distribution(
            [row["document_token_length"] for row in prepared]
        ),
        "prompt_template": prompt_template_metadata(task),
        "contexts": contexts,
    }


def build_heldout_inventory(
    *,
    longbench_root: Path,
    tokenizer: Any,
    exclusion_provenance: dict[str, Any],
    split: str = "test",
    context_field: str | None = None,
    query_field: str | None = None,
    answers_field: str | None = None,
    id_field: str | None = None,
) -> dict[str, Any]:
    root = longbench_root.expanduser().resolve()
    exclusions = exclusion_index(exclusion_provenance)
    tasks = []
    for task in TASKS:
        path = root / f"{task}.jsonl"
        records = load_jsonl_records(path)
        prepared = prepare_longbench_records(
            records,
            tokenizer=tokenizer,
            task=task,
            split=split,
            seed=20260815,
            context_field=context_field,
            query_field=query_field,
            answers_field=answers_field,
            id_field=id_field,
        )
        tasks.append(
            build_task_inventory(
                task=task,
                source_record_count=len(records),
                prepared=prepared,
                tokenizer=tokenizer,
                exclusions=exclusions,
            )
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": "complete",
        "mode": "target_blind_token_length_inventory",
        "longbench_root": str(root),
        "dataset_split": split,
        "inventory_context_lengths": list(INVENTORY_LENGTHS),
        "primary_cross_task_32k_anchor_allowed": False,
        "model_forward_executed": False,
        "cuda_required": False,
        "exclusion_applied_before_selection": True,
        "selection_seed": SELECTION_SEED,
        "tasks": tasks,
    }


def _task(inventory: dict[str, Any], task: str) -> dict[str, Any]:
    return next(row for row in inventory["tasks"] if row["task"] == task)


def _context(task_inventory: dict[str, Any], length: int) -> dict[str, Any]:
    return next(
        row
        for row in task_inventory["contexts"]
        if int(row["nominal_prompt_length"]) == int(length)
    )


def _joint_eligible(
    task_inventory: dict[str, Any], short: int, long: int
) -> list[dict[str, Any]]:
    short_rows = {
        row["sample_id"]: row
        for row in _context(task_inventory, short)["eligible_samples"]
    }
    long_rows = {
        row["sample_id"]: row
        for row in _context(task_inventory, long)["eligible_samples"]
    }
    shared = [
        long_rows[sample_id] for sample_id in short_rows.keys() & long_rows.keys()
    ]
    shared.sort(key=lambda row: (row["selection_hash"], row["sample_id"]))
    return shared


def choose_common_long_context(
    inventory: dict[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    audit = []
    for length in LONG_CONTEXT_ORDER:
        counts = {
            task: len(_joint_eligible(_task(inventory, task), SHORT_CONTEXT, length))
            for task in INDEPENDENT_TASKS
        }
        qualifies = all(counts[task] >= TASK_SAMPLE_COUNTS[task] for task in counts)
        audit.append(
            {
                "nominal_prompt_length": length,
                "joint_1k_and_long_eligible_count_by_task": counts,
                "qualifies": qualifies,
            }
        )
        if qualifies:
            return length, audit
    raise EligibilityError(
        "insufficient_common_long_context_eligibility",
        "none of 16K, 8K, or 4K has 3 Qasper and 8 HotpotQA samples "
        "eligible at both 1K and the common long context",
    )


def choose_govreport_long_context(
    inventory: dict[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    task_inventory = _task(inventory, KNOWN_TASK)
    audit = []
    for length in LONG_CONTEXT_ORDER:
        count = len(_joint_eligible(task_inventory, SHORT_CONTEXT, length))
        qualifies = count >= TASK_SAMPLE_COUNTS[KNOWN_TASK]
        audit.append(
            {
                "nominal_prompt_length": length,
                "joint_1k_and_long_eligible_count": count,
                "qualifies": qualifies,
            }
        )
        if qualifies:
            return length, audit
    raise EligibilityError(
        "insufficient_govreport_replication_eligibility",
        "none of 16K, 8K, or 4K has three unused GovReport samples eligible "
        "at both 1K and the replication long context",
    )


def deterministic_select(
    candidates: Sequence[dict[str, Any]], *, count: int
) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates, key=lambda row: (row["selection_hash"], row["sample_id"])
    )
    if len(ranked) < count:
        raise EligibilityError(
            "insufficient_requested_sample_count",
            f"requested {count} physical samples but only {len(ranked)} are eligible",
        )
    return ranked[:count]


def freeze_cohort(inventory: dict[str, Any]) -> dict[str, Any]:
    common_long, common_audit = choose_common_long_context(inventory)
    gov_long, gov_audit = choose_govreport_long_context(inventory)
    task_rows = []
    for task in TASKS:
        long = gov_long if task == KNOWN_TASK else common_long
        task_inventory = _task(inventory, task)
        selected = deterministic_select(
            _joint_eligible(task_inventory, SHORT_CONTEXT, long),
            count=TASK_SAMPLE_COUNTS[task],
        )
        short_by_id = {
            row["sample_id"]: row
            for row in _context(task_inventory, SHORT_CONTEXT)["eligible_samples"]
        }
        long_by_id = {
            row["sample_id"]: row
            for row in _context(task_inventory, long)["eligible_samples"]
        }
        physical = []
        for rank, selected_row in enumerate(selected, 1):
            sample_id = selected_row["sample_id"]
            if (
                short_by_id[sample_id]["context_prefix_characters"]
                > long_by_id[sample_id]["context_prefix_characters"]
            ):
                raise ValueError(
                    f"sample {sample_id} does not have a nested short/long prefix"
                )
            physical.append(
                {
                    "selection_rank": rank,
                    "selection_hash": selected_row["selection_hash"],
                    "sample_id": sample_id,
                    "source_sample_id": selected_row["source_sample_id"],
                    "source_text_sha256": selected_row["source_text_sha256"],
                    "eligible_at_both_required_contexts": True,
                    "short_context_is_nested_prefix_of_long_context": True,
                    "contexts": {
                        str(SHORT_CONTEXT): short_by_id[sample_id],
                        str(long): long_by_id[sample_id],
                    },
                }
            )
        task_rows.append(
            {
                "task": task,
                "task_role": TASK_ROLES[task],
                "contributes_to_independent_task_count": task in INDEPENDENT_TASKS,
                "requested_physical_sample_count": TASK_SAMPLE_COUNTS[task],
                "frozen_physical_sample_count": len(physical),
                "short_context_length": SHORT_CONTEXT,
                "long_context_length": long,
                "paired_context_lengths": [SHORT_CONTEXT, long],
                "selected_physical_samples": physical,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": STATUS,
        "selection_seed": SELECTION_SEED,
        "selection_hash_formula": (
            "SHA256(seed || NUL || task || NUL || source_sample_id || NUL || "
            "source_text_sha256)"
        ),
        "selection_uses_model_outputs": False,
        "replacement_after_h2_execution_begins_allowed": False,
        "short_context_length": SHORT_CONTEXT,
        "common_independent_task_long_context_length": common_long,
        "common_long_context_selection_order": list(LONG_CONTEXT_ORDER),
        "common_long_context_selection_audit": common_audit,
        "govreport_replication_long_context_length": gov_long,
        "govreport_long_context_selection_order": list(LONG_CONTEXT_ORDER),
        "govreport_long_context_selection_audit": gov_audit,
        "independent_tasks": list(INDEPENDENT_TASKS),
        "independent_task_count": len(INDEPENDENT_TASKS),
        "govreport_contributes_to_independent_task_count": False,
        "tasks": task_rows,
        "model_forward_executed": False,
    }


def build_matrix(
    cohort: dict[str, Any], *, node_budget: int, role: str
) -> dict[str, Any]:
    rows = []
    for task in cohort["tasks"]:
        for sample in task["selected_physical_samples"]:
            for context in task["paired_context_lengths"]:
                context_record = sample["contexts"][str(context)]
                for family in FAMILIES:
                    rows.append(
                        {
                            "task": task["task"],
                            "task_role": task["task_role"],
                            "sample_id": sample["sample_id"],
                            "source_sample_id": sample["source_sample_id"],
                            "source_text_sha256": sample["source_text_sha256"],
                            "selection_rank": sample["selection_rank"],
                            "selection_hash": sample["selection_hash"],
                            "nominal_prompt_length": context,
                            "actual_prompt_length": context_record[
                                "actual_prompt_length"
                            ],
                            "context_prefix_sha256": context_record[
                                "context_prefix_sha256"
                            ],
                            "configuration_family": family,
                            "candidate_merge_mode": (
                                BASE_FAMILY if family == BASE_FAMILY else "refined_tree"
                            ),
                            "frozen_refinement_configuration_id": (
                                None if family == BASE_FAMILY else family
                            ),
                            "node_budget": node_budget,
                            "evaluation_role": role,
                            "verification_mode": VERIFICATION_MODE,
                            "online_selector_mode": "top1_packed_tree",
                            "max_new_tokens": 128,
                            "warmup_runs": 1,
                            "timing_repeats": 3,
                            "dtype": "bfloat16",
                            "external_batch_size": 1,
                            "candidate_label_authority": CANDIDATE_LABEL_AUTHORITY,
                            "candidate_label_source": CANDIDATE_LABEL_SOURCE,
                            "q1_replay_used_as_target_label_source": False,
                            "execution_authorized": False,
                        }
                    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": STATUS,
        "matrix_role": role,
        "node_budget": node_budget,
        "actual_work_regime": (
            "same_actual_verification_work"
            if node_budget == PRIMARY_NODE_BUDGET
            else "budget_cap_expansion"
        ),
        "secondary_can_override_primary_conclusion": False,
        "frozen_configuration_families": list(FAMILIES),
        "frozen_refinement_families": list(FROZEN_REFINEMENT_FAMILIES),
        "family_selected_or_removed": False,
        "row_count": len(rows),
        "rows": rows,
        "model_forward_executed": False,
        "execution_authorized": False,
    }


def build_aggregation_rules() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": STATUS,
        "primary_node_budget": PRIMARY_NODE_BUDGET,
        "primary_actual_work_regime": "same_actual_verification_work",
        "secondary_node_budget": SECONDARY_NODE_BUDGET,
        "secondary_actual_work_regime": "budget_cap_expansion",
        "secondary_can_override_primary_conclusion": False,
        "independent_tasks": list(INDEPENDENT_TASKS),
        "independent_task_count": 2,
        "same_task_replication_tasks": [KNOWN_TASK],
        "govreport_contributes_to_independent_task_count": False,
        "aggregation_unit_for_final_stability": "task_context_stratum",
        "combined_sample_weighted_global_pool_can_replace_task_level_rule": False,
        "candidate_rate_aggregation": "pooled_valid_labelled_positions_within_task",
        "online_efficiency_aggregation": {
            "accepted_draft_tokens_per_verified_node": (
                "sum(accepted_draft_token_count)/sum(actual_verified_node_count)"
            ),
            "mean_of_sample_ratios_used_as_primary": False,
        },
        "required_task_level_metrics": [
            "correctness_valid_sample_count",
            "candidate_label_available_position_count",
            "selected_tree_coverage",
            "coverage_absolute_gain",
            "accepted_draft_tokens_per_verified_node",
            "accepted_per_verified_absolute_gain",
            "accepted_per_verified_relative_gain",
            "accepted_draft_tokens_per_round",
            "rounds_per_returned_token",
            "cuda_ms_per_returned_token",
            "conditional_remask_cost",
            "always_fixed_verifier_cost",
        ],
        "per_physical_sample_paired_values_required": True,
        "sample_level_sign_distribution_required": True,
        "sample_heterogeneity_must_be_preserved": True,
        "stable_cross_task_same_work_signal_rule": {
            "classification": "stable_cross_task_same_work_signal",
            "node_budget": PRIMARY_NODE_BUDGET,
            "required_independent_tasks": list(INDEPENDENT_TASKS),
            "required_contexts": "short_and_frozen_common_long",
            "required_task_context_stratum_count_per_family": 4,
            "all_contributing_cells_correctness_valid": True,
            "coverage_gain_strictly_positive_in_every_stratum": True,
            "accepted_tokens_per_verified_node_gain_strictly_positive_in_every_stratum": True,
            "statistical_significance_threshold_used": False,
        },
        "correctness_hard_gate": {
            "original_dflash_tree_token_exactness": True,
            "formal_repeat_stability": True,
            "profile_formal_equality": True,
            "diagnostic_formal_equality": True,
            "structural_kv_correctness": True,
            "external_phase2d2_checker_accepted": True,
            "correctness_invalid_cells_contribute_to_benefit_aggregation": False,
            "correctness_invalid_treated_as_negative_candidate_quality": False,
        },
        "mechanism_decomposition_fields": [
            "base_candidate_coverage",
            "conditional_candidate_coverage",
            "merged_candidate_coverage",
            "selected_tree_coverage",
            "refined_only_target_recovery_count",
            "refined_only_target_recovery_rate",
            "target_token_rank_improvement",
            "correct_prefix_survival_depth",
        ],
        "reranking_or_tree_reallocation_signal_rule": (
            "selected_tree_coverage_gain > 0 and refined_only_target_recovery_count == 0"
        ),
        "heldout_refined_only_recovery_does_not_rewrite_h1": True,
        "cost_axes": [
            "dependency_modeling_cost",
            "correctness_preserving_verifier_cost",
        ],
        "lightweight_dependency_model_claimed_to_solve_verifier_overhead": False,
        "speedup_claim_authorized": False,
        "candidate_label_authority": CANDIDATE_LABEL_AUTHORITY,
        "candidate_label_source": CANDIDATE_LABEL_SOURCE,
        "q1_replay_used_as_target_label_source": False,
        "model_forward_executed": False,
    }


def build_freeze_artifacts(
    *,
    inventory: dict[str, Any],
    exclusions: dict[str, Any],
    tokenizer_provenance: dict[str, Any],
    h11_provenance: dict[str, Any],
) -> dict[str, Any]:
    cohort = freeze_cohort(inventory)
    primary = build_matrix(
        cohort,
        node_budget=PRIMARY_NODE_BUDGET,
        role="primary_same_actual_verification_work",
    )
    secondary = build_matrix(
        cohort,
        node_budget=SECONDARY_NODE_BUDGET,
        role="secondary_budget_cap_expansion_ablation",
    )
    rules = build_aggregation_rules()
    hash_payload = {
        "frozen_cohort": cohort,
        "primary_matrix": primary,
        "secondary_matrix": secondary,
        "aggregation_rules": rules,
    }
    component_hashes = {
        name: _sha256_json(payload) for name, payload in hash_payload.items()
    }
    freeze_hash = _sha256_json(hash_payload)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": _git_value("rev-parse", "HEAD"),
        "branch": _git_value("branch", "--show-current"),
        "tokenizer_provenance": tokenizer_provenance,
        "h11_prerequisite_provenance": h11_provenance,
        "exclusion_provenance_sha256": _sha256_json(exclusions),
        "inventory_sha256": _sha256_json(inventory),
        "frozen_content_component_sha256": component_hashes,
        "freeze_content_sha256": freeze_hash,
        "freeze_content_hash_scope": [
            "frozen_cohort",
            "primary_matrix",
            "secondary_matrix",
            "aggregation_rules",
        ],
        "h2_execution_must_require_exact_freeze_content_sha256": True,
        "model_forward_executed": False,
        "target_model_loaded": False,
        "draft_model_loaded": False,
        "cuda_required": False,
        "h2_primary_execution_authorized": False,
        "h2_secondary_execution_authorized": False,
        "no_replacement_after_execution_begins": True,
        "artifacts": list(ARTIFACTS),
        "error": None,
    }
    return {
        "inventory": inventory,
        "exclusions": exclusions,
        "cohort": cohort,
        "primary": primary,
        "secondary": secondary,
        "rules": rules,
        "manifest": manifest,
    }


def _git_value(*args: str) -> str | None:
    import subprocess

    result = subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _tokenizer_provenance(path: Path, tokenizer: Any) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    )
    return {
        "model_id": DEFAULT_TARGET_MODEL,
        "revision": DEFAULT_TARGET_REVISION,
        "local_path": str(resolved),
        "local_files_only": True,
        "trust_remote_code": True,
        "class_name": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}",
        "file_sha256": {
            name: _sha256_file(resolved / name)
            for name in names
            if (resolved / name).is_file()
        },
        "model_loaded": False,
    }


def _load_tokenizer(path: Path) -> tuple[Any, dict[str, Any]]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Transformers tokenizer support is not installed") from exc
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"target tokenizer path does not exist: {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(resolved), trust_remote_code=True, local_files_only=True
    )
    return tokenizer, _tokenizer_provenance(resolved, tokenizer)


def _h11_provenance(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    result = check_h11_output(resolved)
    design_path = resolved / "phase2d2a-h2-preregistered-design.json"
    design = _json(design_path)
    if design["h2_execution_authorized"] is not False:
        raise ValueError("H1.1 prerequisite unexpectedly authorizes H2 execution")
    names = (
        "phase2d2a-h1-budget-utilization.json",
        "phase2d2a-h1-mechanism-decomposition.json",
        "phase2d2a-h1-reinterpretation.json",
        "phase2d2a-h2-preregistered-design.json",
    )
    return {
        "directory": str(resolved),
        "checker_status": result["status"],
        "direct_regime_hypothesis_confirmed": True,
        "h2_execution_authorized_by_h11": False,
        "artifact_sha256": {name: _sha256_file(resolved / name) for name in names},
    }


def _inventory_csv_rows(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task in inventory["tasks"]:
        for context in task["contexts"]:
            rows.append(
                {
                    "task": task["task"],
                    "task_role": task["task_role"],
                    "requested_physical_sample_count": task[
                        "requested_physical_sample_count"
                    ],
                    "source_record_count": task["source_record_count"],
                    "usable_record_count": task["usable_record_count"],
                    "nominal_prompt_length": context["nominal_prompt_length"],
                    "eligible_before_exclusion_count": context[
                        "eligible_before_exclusion_count"
                    ],
                    "excluded_previous_sample_count": context[
                        "excluded_previous_sample_count"
                    ],
                    "eligible_sample_count": context["eligible_sample_count"],
                    "eligible_document_token_length_distribution": context[
                        "eligible_document_token_length_distribution"
                    ],
                    "eligible_sample_ids": context["eligible_sample_ids"],
                    "eligible_source_sample_ids": context["eligible_source_sample_ids"],
                    "eligible_source_text_sha256": context[
                        "eligible_source_text_sha256"
                    ],
                    "eligible_context_prefix_sha256": [
                        row["context_prefix_sha256"]
                        for row in context["eligible_samples"]
                    ],
                }
            )
    return rows


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for field, value in row.items()
                }
            )
    temporary.replace(path)


def persist_freeze(output_dir: Path, artifacts: dict[str, Any]) -> None:
    output = output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty H2.0 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    payloads = {
        "phase2d2a-h2-heldout-inventory.json": artifacts["inventory"],
        "phase2d2a-h2-exclusion-provenance.json": artifacts["exclusions"],
        "phase2d2a-h2-frozen-cohort.json": artifacts["cohort"],
        "phase2d2a-h2-primary-matrix.json": artifacts["primary"],
        "phase2d2a-h2-secondary-matrix.json": artifacts["secondary"],
        "phase2d2a-h2-aggregation-rules.json": artifacts["rules"],
        "phase2d2a-h2-freeze-manifest.json": artifacts["manifest"],
    }
    for name, payload in payloads.items():
        _validate_schema(payload, SCHEMAS[name])
        _write_json(output / name, payload)
    _write_csv(
        output / "phase2d2a-h2-heldout-inventory.csv",
        _inventory_csv_rows(artifacts["inventory"]),
    )


def persist_ineligible_inventory(
    output_dir: Path,
    *,
    inventory: dict[str, Any],
    exclusions: dict[str, Any],
    tokenizer_provenance: dict[str, Any],
    h11_provenance: dict[str, Any],
    status: str,
    message: str,
) -> None:
    output = output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty H2.0 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": _git_value("rev-parse", "HEAD"),
        "branch": _git_value("branch", "--show-current"),
        "tokenizer_provenance": tokenizer_provenance,
        "h11_prerequisite_provenance": h11_provenance,
        "exclusion_provenance_sha256": _sha256_json(exclusions),
        "inventory_sha256": _sha256_json(inventory),
        "frozen_content_component_sha256": {},
        "freeze_content_sha256": None,
        "freeze_content_hash_scope": [
            "frozen_cohort",
            "primary_matrix",
            "secondary_matrix",
            "aggregation_rules",
        ],
        "h2_execution_must_require_exact_freeze_content_sha256": False,
        "model_forward_executed": False,
        "target_model_loaded": False,
        "draft_model_loaded": False,
        "cuda_required": False,
        "h2_primary_execution_authorized": False,
        "h2_secondary_execution_authorized": False,
        "no_replacement_after_execution_begins": True,
        "artifacts": [
            "phase2d2a-h2-heldout-inventory.json",
            "phase2d2a-h2-heldout-inventory.csv",
            "phase2d2a-h2-exclusion-provenance.json",
            "phase2d2a-h2-freeze-manifest.json",
        ],
        "error": {"type": "EligibilityError", "status": status, "message": message},
    }
    for name, payload in (
        ("phase2d2a-h2-heldout-inventory.json", inventory),
        ("phase2d2a-h2-exclusion-provenance.json", exclusions),
        ("phase2d2a-h2-freeze-manifest.json", manifest),
    ):
        _validate_schema(payload, SCHEMAS[name])
        _write_json(output / name, payload)
    _write_csv(
        output / "phase2d2a-h2-heldout-inventory.csv",
        _inventory_csv_rows(inventory),
    )


def _print_report(artifacts: dict[str, Any]) -> None:
    print("TARGET-BLIND HELD-OUT INVENTORY")
    for task in artifacts["inventory"]["tasks"]:
        print(
            task["task"],
            f"source={task['source_record_count']}",
            f"usable={task['usable_record_count']}",
        )
        for context in task["contexts"]:
            print(
                " ",
                context["nominal_prompt_length"],
                f"before={context['eligible_before_exclusion_count']}",
                f"excluded={context['excluded_previous_sample_count']}",
                f"eligible={context['eligible_sample_count']}",
            )
    cohort = artifacts["cohort"]
    print(
        "common_independent_task_long_context_length:",
        cohort["common_independent_task_long_context_length"],
    )
    print(
        "govreport_replication_long_context_length:",
        cohort["govreport_replication_long_context_length"],
    )
    for task in cohort["tasks"]:
        print(
            task["task"],
            task["task_role"],
            [row["sample_id"] for row in task["selected_physical_samples"]],
        )
    print("primary_matrix_rows:", artifacts["primary"]["row_count"])
    print("secondary_matrix_rows:", artifacts["secondary"]["row_count"])
    print("freeze_content_sha256:", artifacts["manifest"]["freeze_content_sha256"])
    print("model_forward_executed: false")
    print("h2_primary_execution_authorized: false")
    print("h2_secondary_execution_authorized: false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--longbench-root", type=Path, required=True)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--target-revision", default=DEFAULT_TARGET_REVISION)
    parser.add_argument("--target-model-path", type=Path, required=True)
    parser.add_argument("--previous-results-root", type=Path, action="append")
    parser.add_argument("--h11-dir", type=Path, required=True)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--context-field", default=None)
    parser.add_argument("--query-field", default=None)
    parser.add_argument("--answers-field", default=None)
    parser.add_argument("--id-field", default=None)
    parser.add_argument("--selection-seed", default=SELECTION_SEED)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.target_model != DEFAULT_TARGET_MODEL:
        raise ValueError(f"H2.0 requires target model {DEFAULT_TARGET_MODEL}")
    if args.target_revision != DEFAULT_TARGET_REVISION:
        raise ValueError(f"H2.0 requires target revision {DEFAULT_TARGET_REVISION}")
    if args.selection_seed != SELECTION_SEED:
        raise ValueError(f"H2.0 selection seed is frozen to {SELECTION_SEED}")
    if not args.previous_results_root:
        raise ValueError("provide at least one --previous-results-root")
    return args


def main(argv: list[str] | None = None) -> int:
    args = validate_args(build_parser().parse_args(argv))
    exclusions = collect_exclusion_provenance(args.previous_results_root)
    validate_exclusion_preflight(exclusions)
    tokenizer, tokenizer_provenance = _load_tokenizer(args.target_model_path)
    inventory = build_heldout_inventory(
        longbench_root=args.longbench_root,
        tokenizer=tokenizer,
        exclusion_provenance=exclusions,
        split=args.dataset_split,
        context_field=args.context_field,
        query_field=args.query_field,
        answers_field=args.answers_field,
        id_field=args.id_field,
    )
    h11 = _h11_provenance(args.h11_dir)
    try:
        artifacts = build_freeze_artifacts(
            inventory=inventory,
            exclusions=exclusions,
            tokenizer_provenance=tokenizer_provenance,
            h11_provenance=h11,
        )
    except EligibilityError as exc:
        persist_ineligible_inventory(
            args.output_dir,
            inventory=inventory,
            exclusions=exclusions,
            tokenizer_provenance=tokenizer_provenance,
            h11_provenance=h11,
            status=exc.status,
            message=str(exc),
        )
        print(f"H2.0 STOP: {exc.status}: {exc}", file=sys.stderr)
        return 2
    persist_freeze(args.output_dir, artifacts)
    _print_report(artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
