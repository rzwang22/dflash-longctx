"""Framework-lazy cohort and subrun identity helpers for Phase 2D.1B-NR."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any


def physical_case_id(task: str, source_sample_id: str, context_length: int) -> str:
    return f"{task}::{source_sample_id}::{int(context_length)}"


def configuration_case_id(
    task: str,
    source_sample_id: str,
    context_length: int,
    selector_mode: str,
    node_budget: int,
) -> str:
    return (
        f"{physical_case_id(task, source_sample_id, context_length)}::"
        f"{selector_mode}::n{int(node_budget)}"
    )


def subrun_identity(
    *,
    experiment: str,
    task: str,
    sample_id: str,
    source_sample_id: str,
    context_length: int,
    selector_mode: str,
    node_budget: int,
    repeat_index: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    physical = physical_case_id(task, source_sample_id, context_length)
    configuration = configuration_case_id(
        task,
        source_sample_id,
        context_length,
        selector_mode,
        node_budget,
    )
    return {
        "experiment": experiment,
        "task": task,
        "sample_id": sample_id,
        "source_sample_id": source_sample_id,
        "nominal_prompt_length": int(context_length),
        "selector_mode": selector_mode,
        "node_budget": int(node_budget),
        "repeat_index": int(repeat_index),
        "max_new_tokens": int(max_new_tokens),
        "physical_case_id": physical,
        "configuration_case_id": configuration,
        "independent_repeat_observation_id": (
            f"{experiment}::{configuration}::repeat-{int(repeat_index):02d}"
        ),
    }


def safe_subrun_directory_name(identity: dict[str, Any]) -> str:
    raw = (
        f"{identity['experiment']}__{identity['sample_id']}__"
        f"{identity['nominal_prompt_length']}__{identity['selector_mode']}__"
        f"n{identity['node_budget']}__r{identity['repeat_index']:02d}"
    )
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw)


def derive_matched_capture_cohort(
    capture_rounds: Sequence[dict[str, Any]],
    *,
    task: str,
    context_lengths: Iterable[int],
) -> list[dict[str, str]]:
    """Derive the immutable matched source/sample cohort from capture rows."""

    requested = tuple(int(value) for value in context_lengths)
    if not requested:
        raise ValueError("at least one cohort context length is required")
    rows = [row for row in capture_rounds if row.get("task") == task]
    if not rows:
        raise ValueError(f"capture contains no rows for task {task!r}")
    by_source: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    source_order = []
    for row in rows:
        source = str(row["source_sample_id"])
        if source not in by_source:
            source_order.append(source)
        by_source[source][int(row["nominal_prompt_length"])].add(str(row["sample_id"]))
    result = []
    for source in source_order:
        lengths = by_source[source]
        if not all(length in lengths for length in requested):
            continue
        sample_ids = set().union(*(lengths[length] for length in requested))
        if len(sample_ids) != 1:
            raise ValueError(
                "matched capture source maps to multiple sample IDs: "
                f"source={source!r}, sample_ids={sorted(sample_ids)}"
            )
        result.append({"source_sample_id": source, "sample_id": next(iter(sample_ids))})
    if not result:
        raise ValueError(
            "capture has no source represented at every requested context length"
        )
    return result
