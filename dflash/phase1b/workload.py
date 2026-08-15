"""Deterministic, offline LongBench workload construction and inventory."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_CONTEXT_LENGTHS = (1024, 4096, 8192, 16384, 32768)
LONG_BENCH_TASKS = ("gov_report", "qasper", "hotpotqa")
LONG_BENCH_FILENAMES = {
    "gov_report": "gov_report.jsonl",
    "qasper": "qasper.jsonl",
    "hotpotqa": "hotpotqa.jsonl",
}
MAX_NOMINAL_SHORTFALL = 16

TASK_PROMPT_TEMPLATES = {
    "gov_report": {
        "name": "govreport-summary",
        "version": "1",
        "template": (
            "Summarize the following U.S. government report. Preserve the main "
            "findings, evidence, and recommendations.\n\nReport:\n{context}"
        ),
        "requires_query": False,
    },
    "qasper": {
        "name": "longbench-qasper-question-answering",
        "version": "1",
        "template": (
            "Answer the question using only the supplied paper context. If the "
            "context does not contain the answer, say so concisely.\n\n"
            "Question:\n{query}\n\nPaper context:\n{context}"
        ),
        "requires_query": True,
    },
    "hotpotqa": {
        "name": "longbench-hotpotqa-multidoc-question-answering",
        "version": "1",
        "template": (
            "Answer the question using only the supplied multi-document context. "
            "Use evidence across documents when needed.\n\n"
            "Question:\n{query}\n\nMulti-document context:\n{context}"
        ),
        "requires_query": True,
    },
}

# Phase 1B compatibility names.
PROMPT_TEMPLATE_ID = "govreport-summary-v1"
PROMPT_PREFIX = TASK_PROMPT_TEMPLATES["gov_report"]["template"].split("{context}")[0]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_token_ids(token_ids: Sequence[int]) -> str:
    return sha256_json(list(token_ids))


def prompt_template_metadata(task: str) -> dict[str, Any]:
    if task not in TASK_PROMPT_TEMPLATES:
        raise ValueError(f"Unsupported LongBench task: {task!r}")
    spec = TASK_PROMPT_TEMPLATES[task]
    identity = f"{spec['name']}-v{spec['version']}"
    return {
        "name": spec["name"],
        "version": spec["version"],
        "id": identity,
        "sha256": sha256_text(spec["template"]),
        "template": spec["template"],
    }


def resolve_jsonl_path(
    *,
    task: str,
    longbench_root: Path | None = None,
    dataset_jsonl: Path | None = None,
    govreport_jsonl: Path | None = None,
) -> Path | None:
    sources = [
        longbench_root is not None,
        dataset_jsonl is not None,
        govreport_jsonl is not None,
    ]
    if sum(sources) > 1:
        raise ValueError(
            "Choose only one of --longbench-root, --dataset-jsonl, or "
            "--govreport-jsonl"
        )
    if govreport_jsonl is not None and task != "gov_report":
        raise ValueError("--govreport-jsonl is compatible only with gov_report")
    if longbench_root is not None:
        return longbench_root.expanduser().resolve() / LONG_BENCH_FILENAMES[task]
    selected = dataset_jsonl or govreport_jsonl
    return selected.expanduser().resolve() if selected is not None else None


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"LongBench JSONL does not exist: {path}")
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(
                    f"LongBench JSONL line {line_number} is not a JSON object"
                )
            records.append(value)
    return records


def load_longbench_records(
    *,
    jsonl_path: Path | None = None,
    dataset_path: Path | None = None,
    dataset_cache_dir: Path | None = None,
    dataset_name: str = "ccdv/govreport-summarization",
    dataset_config: str | None = None,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Load exactly one local JSONL or backward-compatible HF dataset source."""

    choices = [
        jsonl_path is not None,
        dataset_path is not None,
        dataset_cache_dir is not None,
    ]
    if sum(choices) != 1:
        raise ValueError("Choose exactly one local dataset source")
    if jsonl_path is not None:
        return load_jsonl_records(jsonl_path)

    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        from datasets import DatasetDict, load_dataset, load_from_disk
    except ImportError as exc:
        raise RuntimeError("The datasets package is required for this source") from exc

    if dataset_path is not None:
        resolved = dataset_path.expanduser().resolve()
        if not resolved.exists():
            raise RuntimeError(f"Saved dataset does not exist: {resolved}")
        dataset = load_from_disk(str(resolved))
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise RuntimeError(f"Split {split!r} is absent from {resolved}")
            dataset = dataset[split]
    else:
        resolved = dataset_cache_dir.expanduser().resolve()
        if not resolved.is_dir():
            raise RuntimeError(f"Dataset cache does not exist: {resolved}")
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            cache_dir=str(resolved),
            download_mode="reuse_dataset_if_exists",
        )
    return [dict(record) for record in dataset]


# Backward-compatible loader name.
load_govreport_records = load_longbench_records


def render_task_prompt(task: str, context: str, query: str = "") -> list[dict[str, str]]:
    spec = TASK_PROMPT_TEMPLATES[task]
    if spec["requires_query"] and not query.strip():
        raise ValueError(f"LongBench task {task} requires a non-empty query")
    content = spec["template"].format(context=context, query=query)
    return [{"role": "user", "content": content}]


def render_prompt(report_prefix: str) -> list[dict[str, str]]:
    """Phase 1B compatibility wrapper for the GovReport prompt."""

    return render_task_prompt("gov_report", report_prefix)


def _as_token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise RuntimeError("LongBench Phase 1B.1 requires batch size 1")
        value = value[0]
    return [int(token) for token in value]


def tokenize_task_chat(
    tokenizer: Any, *, task: str, context: str, query: str = ""
) -> list[int]:
    tokenized = tokenizer.apply_chat_template(
        render_task_prompt(task, context, query),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return _as_token_ids(tokenized)


def tokenize_chat(tokenizer: Any, report_prefix: str) -> list[int]:
    """Phase 1B compatibility wrapper for GovReport chat tokenization."""

    return tokenize_task_chat(
        tokenizer, task="gov_report", context=report_prefix, query=""
    )


def _document_token_ids(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return _as_token_ids(tokenizer.encode(text, add_special_tokens=False))
    result = tokenizer(text, add_special_tokens=False)
    return _as_token_ids(result)


def _field(
    record: dict[str, Any], requested: str | None, candidates: Sequence[str]
) -> Any:
    if requested is not None:
        if requested not in record:
            raise ValueError(f"Requested field {requested!r} is absent from a record")
        return record[requested]
    for name in candidates:
        if name in record:
            return record[name]
    raise ValueError(f"Could not find any of the fields {list(candidates)}")


def prepare_longbench_records(
    records: Iterable[dict[str, Any]],
    *,
    tokenizer: Any,
    task: str,
    split: str = "test",
    seed: int = 20260815,
    context_field: str | None = None,
    text_field: str | None = None,
    query_field: str | None = None,
    answers_field: str | None = None,
    id_field: str | None = None,
) -> list[dict[str, Any]]:
    """Validate one LongBench schema and compute tokenizer-derived metadata."""

    if context_field and text_field and context_field != text_field:
        raise ValueError("--context-field and compatibility --text-field disagree")
    requested_context = context_field or text_field
    prepared = []
    for source_index, record in enumerate(records):
        declared_task = record.get("dataset")
        if (
            isinstance(declared_task, str)
            and declared_task in LONG_BENCH_TASKS
            and declared_task != task
        ):
            raise ValueError(
                f"LongBench record declares task {declared_task!r}, but this run "
                f"is restricted to {task!r}; tasks cannot be mixed"
            )
        context = _field(
            record,
            requested_context,
            ("context", "report", "document", "article", "text"),
        )
        if not isinstance(context, str) or not context.strip():
            continue
        if TASK_PROMPT_TEMPLATES[task]["requires_query"]:
            query = _field(record, query_field, ("input", "query", "question"))
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"LongBench task {task} requires a string query")
        else:
            query = ""

        if answers_field is not None:
            reference_answers = _field(record, answers_field, ())
        else:
            reference_answers = record.get("answers")
        context_hash = sha256_text(context)
        if id_field is not None:
            source_id = str(_field(record, id_field, ()))
        else:
            source_id = str(
                next(
                    (
                        record[name]
                        for name in ("_id", "id", "document_id", "report_id")
                        if record.get(name) not in (None, "")
                    ),
                    f"sha256:{context_hash[:16]}",
                )
            )
        fixed_prompt_ids = tokenize_task_chat(
            tokenizer, task=task, context="", query=query
        )
        full_prompt_ids = tokenize_task_chat(
            tokenizer, task=task, context=context, query=query
        )
        prepared.append(
            {
                "source_index": source_index,
                "source_sample_id": source_id,
                "sample_id": (
                    f"{task}-{sha256_text(source_id + chr(0) + context_hash)[:16]}"
                ),
                "longbench_task": task,
                "dataset_split": split,
                "random_seed": seed,
                "context": context,
                "query": query,
                "reference_answers": reference_answers,
                "reference_answers_sha256": sha256_json(reference_answers),
                "source_text_sha256": context_hash,
                "document_token_length": len(_document_token_ids(tokenizer, context)),
                "fixed_prompt_length": len(fixed_prompt_ids),
                "source_full_prompt_length": len(full_prompt_ids),
                "selection_key": sha256_text(
                    f"{seed}\0{split}\0{task}\0{source_id}\0{context_hash}"
                ),
            }
        )
    prepared.sort(key=lambda row: (row["selection_key"], row["source_index"]))
    return prepared


def _percentile(values: Sequence[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def build_workload_inventory(
    prepared: Sequence[dict[str, Any]],
    *,
    task: str,
    nominal_lengths: Sequence[int] = DEFAULT_CONTEXT_LENGTHS,
    requested_sample_count: int = 8,
) -> dict[str, Any]:
    lengths = tuple(int(length) for length in nominal_lengths)
    document_lengths = [row["document_token_length"] for row in prepared]
    eligible = {
        str(length): [
            row["sample_id"]
            for row in prepared
            if row["fixed_prompt_length"] <= length
            and row["source_full_prompt_length"] >= length
        ]
        for length in lengths
    }
    eligible_source_ids = {
        str(length): [
            row["source_sample_id"]
            for row in prepared
            if row["fixed_prompt_length"] <= length
            and row["source_full_prompt_length"] >= length
        ]
        for length in lengths
    }
    longest_with_any = max(
        (length for length in lengths if eligible[str(length)]), default=None
    )
    matched_lengths = [
        length for length in lengths if longest_with_any and length <= longest_with_any
    ]
    matched_count = (
        min(requested_sample_count, len(eligible[str(longest_with_any)]))
        if longest_with_any is not None
        else 0
    )
    return {
        "schema_version": "1.0",
        "task": task,
        "record_count": len(prepared),
        "prompt_template": prompt_template_metadata(task),
        "document_token_length": {
            "count": len(document_lengths),
            "min": min(document_lengths, default=None),
            "median": (
                float(statistics.median(document_lengths))
                if document_lengths
                else None
            ),
            "p90": _percentile(document_lengths, 0.90),
            "max": max(document_lengths, default=None),
        },
        "eligible_by_prompt_length": {
            str(length): {
                "eligible_sample_count": len(eligible[str(length)]),
                "sample_ids": eligible[str(length)],
                "source_sample_ids": eligible_source_ids[str(length)],
            }
            for length in lengths
        },
        "sample_ids_reaching_maximum_length": eligible[str(max(lengths))],
        "source_sample_ids_reaching_maximum_length": eligible_source_ids[
            str(max(lengths))
        ],
        "recommendations": {
            "matched": {
                "sample_count": matched_count,
                "context_lengths": matched_lengths,
                "sample_ids": (
                    eligible[str(longest_with_any)][:matched_count]
                    if longest_with_any is not None
                    else []
                ),
                "source_sample_ids": (
                    eligible_source_ids[str(longest_with_any)][:matched_count]
                    if longest_with_any is not None
                    else []
                ),
                "paired": True,
                "basis": "longest requested length with at least one eligible sample",
            },
            "per_length": {
                "paired": False,
                "sample_count_by_length": {
                    str(length): min(
                        requested_sample_count, len(eligible[str(length)])
                    )
                    for length in lengths
                },
            },
        },
    }


def _prefix_at_or_below(
    tokenizer: Any,
    *,
    task: str,
    context: str,
    query: str,
    nominal_length: int,
) -> tuple[str, list[int]]:
    empty_ids = tokenize_task_chat(tokenizer, task=task, context="", query=query)
    if len(empty_ids) > nominal_length:
        raise ValueError(
            f"Fixed instruction/query has {len(empty_ids)} tokens, above "
            f"nominal length {nominal_length}"
        )
    full_ids = tokenize_task_chat(
        tokenizer, task=task, context=context, query=query
    )
    if len(full_ids) < nominal_length:
        raise ValueError(
            f"Sample reaches only {len(full_ids)} templated tokens; "
            f"{nominal_length} requested"
        )

    low, high = 0, len(context)
    best_characters, best_ids = 0, empty_ids
    while low <= high:
        middle = (low + high) // 2
        candidate_ids = tokenize_task_chat(
            tokenizer, task=task, context=context[:middle], query=query
        )
        if len(candidate_ids) <= nominal_length:
            if len(candidate_ids) > len(best_ids) or (
                len(candidate_ids) == len(best_ids) and middle > best_characters
            ):
                best_characters, best_ids = middle, candidate_ids
            low = middle + 1
        else:
            high = middle - 1

    for characters in range(
        max(0, best_characters - 16),
        min(len(context), max(low, best_characters) + 16) + 1,
    ):
        candidate_ids = tokenize_task_chat(
            tokenizer, task=task, context=context[:characters], query=query
        )
        if len(candidate_ids) <= nominal_length and (
            len(candidate_ids) > len(best_ids)
            or (len(candidate_ids) == len(best_ids) and characters > best_characters)
        ):
            best_characters, best_ids = characters, candidate_ids
    return context[:best_characters], best_ids


def build_longbench_workloads(
    prepared: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    task: str,
    nominal_lengths: Sequence[int] = DEFAULT_CONTEXT_LENGTHS,
    sample_count: int = 8,
    sample_policy: str = "matched",
    max_new_tokens: int = 128,
    model_position_limit: int,
    sample_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a matched or explicitly unpaired LongBench cohort."""

    lengths = tuple(int(length) for length in nominal_lengths)
    if not lengths or tuple(sorted(set(lengths))) != lengths:
        raise ValueError("nominal_lengths must be unique and strictly increasing")
    if sample_count <= 0 or max_new_tokens <= 0:
        raise ValueError("sample_count and max_new_tokens must be positive")
    if max(lengths) + max_new_tokens > model_position_limit:
        raise ValueError(
            f"Largest prompt ({max(lengths)}) plus max_new_tokens "
            f"({max_new_tokens}) exceeds model position limit {model_position_limit}"
        )
    if sample_policy not in {"matched", "per-length"}:
        raise ValueError("sample_policy must be matched or per-length")
    if sample_id is not None and sample_policy != "matched":
        raise ValueError("An explicit sample_id requires sample_policy=matched")

    selected_by_length: dict[int, list[dict[str, Any]]] = {}
    if sample_id is not None:
        matches = [row for row in prepared if row["sample_id"] == sample_id]
        if not matches:
            raise RuntimeError(
                f"LongBench task {task}: requested sample ID {sample_id!r} "
                "does not exist in the prepared workload"
            )
        if len(matches) != 1:
            raise RuntimeError(
                f"LongBench task {task}: requested sample ID {sample_id!r} is "
                f"not unique ({len(matches)} prepared records)"
            )
        selected = matches[0]
        unavailable_lengths = [
            length
            for length in lengths
            if selected["fixed_prompt_length"] > length
            or selected["source_full_prompt_length"] < length
        ]
        if unavailable_lengths:
            raise RuntimeError(
                f"LongBench sample {sample_id} reaches at most "
                f"{selected['source_full_prompt_length']} templated prompt tokens; "
                f"requested lengths {unavailable_lengths} are unavailable"
            )
        selected_by_length = {length: [selected] for length in lengths}
    elif sample_policy == "matched":
        candidates = [
            row
            for row in prepared
            if row["fixed_prompt_length"] <= max(lengths)
            and row["source_full_prompt_length"] >= max(lengths)
        ]
        if len(candidates) < sample_count:
            raise RuntimeError(
                f"LongBench task {task}: only {len(candidates)} samples reach "
                f"{max(lengths)} tokens, but {sample_count} matched samples were "
                "requested. No silent reduction, filler, repetition, or "
                "concatenation is allowed. Run --inspect-workload-only first or "
                "request the inventory-recommended sample count."
            )
        cohort = candidates[:sample_count]
        selected_by_length = {length: cohort for length in lengths}
    else:
        for length in lengths:
            candidates = [
                row
                for row in prepared
                if row["fixed_prompt_length"] <= length
                and row["source_full_prompt_length"] >= length
            ]
            if not candidates:
                raise RuntimeError(
                    f"LongBench task {task}: no samples reach {length} tokens"
                )
            selected_by_length[length] = candidates[:sample_count]

    template = prompt_template_metadata(task)
    workloads = []
    prefixes_by_sample: dict[str, str] = {}
    for nominal_length in lengths:
        for selection_rank, row in enumerate(selected_by_length[nominal_length]):
            context_prefix, input_ids = _prefix_at_or_below(
                tokenizer,
                task=task,
                context=row["context"],
                query=row["query"],
                nominal_length=nominal_length,
            )
            previous = prefixes_by_sample.get(row["sample_id"], "")
            if previous and not context_prefix.startswith(previous):
                raise RuntimeError("LongBench context prefixes are not nested")
            prefixes_by_sample[row["sample_id"]] = context_prefix
            actual_length = len(input_ids)
            shortfall = nominal_length - actual_length
            if shortfall > MAX_NOMINAL_SHORTFALL:
                raise RuntimeError(
                    f"Closest prompt has {actual_length} tokens for nominal length "
                    f"{nominal_length}; allowed shortfall is {MAX_NOMINAL_SHORTFALL}"
                )
            prompt_text = render_task_prompt(
                task, context_prefix, row["query"]
            )[0]["content"]
            context_hash = sha256_text(context_prefix)
            workloads.append(
                {
                    "sample_id": row["sample_id"],
                    "source_sample_id": row["source_sample_id"],
                    "source_index": row["source_index"],
                    "selection_rank": selection_rank,
                    "longbench_task": task,
                    "dataset_split": row["dataset_split"],
                    "random_seed": row["random_seed"],
                    "sample_policy": sample_policy,
                    "paired_context_length_comparison": sample_policy == "matched",
                    "explicit_sample_id": sample_id,
                    "prompt_template_id": template["id"],
                    "prompt_template_name": template["name"],
                    "prompt_template_version": template["version"],
                    "prompt_template_sha256": template["sha256"],
                    "source_text_sha256": row["source_text_sha256"],
                    "source_full_prompt_length": row["source_full_prompt_length"],
                    "document_token_length": row["document_token_length"],
                    "fixed_prompt_length": row["fixed_prompt_length"],
                    "query_sha256": sha256_text(row["query"]),
                    "reference_answers": row["reference_answers"],
                    "reference_answers_sha256": row["reference_answers_sha256"],
                    "nominal_prompt_length": nominal_length,
                    "actual_prompt_length": actual_length,
                    "exact_nominal_length": actual_length == nominal_length,
                    "nominal_prompt_length_shortfall": shortfall,
                    "context_prefix_characters": len(context_prefix),
                    "context_prefix_sha256": context_hash,
                    # Compatibility aliases for Phase 1B consumers.
                    "report_prefix_characters": len(context_prefix),
                    "report_prefix_sha256": context_hash,
                    "prompt_sha256": sha256_text(prompt_text),
                    "input_token_ids_sha256": sha256_token_ids(input_ids),
                    "prompt_text": prompt_text,
                    "input_ids": input_ids,
                }
            )

    sample_ids_by_length = {
        str(length): [row["sample_id"] for row in selected_by_length[length]]
        for length in lengths
    }
    cohort = {
        "sample_policy": sample_policy,
        "paired_context_length_comparison": sample_policy == "matched",
        "explicit_sample_id": sample_id,
        "requested_sample_count": 1 if sample_id is not None else sample_count,
        "actual_sample_count_by_length": {
            length: len(ids) for length, ids in sample_ids_by_length.items()
        },
        "sample_ids_by_length": sample_ids_by_length,
        "interpretation": (
            "strict paired context-length comparison"
            if sample_policy == "matched"
            else "unpaired cohorts; not a strict paired context-length comparison"
        ),
    }
    return workloads, cohort


def build_nested_workloads(
    records: Iterable[dict[str, Any]],
    *,
    tokenizer: Any,
    nominal_lengths: Sequence[int] = DEFAULT_CONTEXT_LENGTHS,
    sample_count: int = 8,
    split: str = "train",
    seed: int = 20260815,
    max_new_tokens: int = 128,
    model_position_limit: int,
    text_field: str | None = None,
    id_field: str | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible matched GovReport builder."""

    prepared = prepare_longbench_records(
        records,
        tokenizer=tokenizer,
        task="gov_report",
        split=split,
        seed=seed,
        text_field=text_field,
        id_field=id_field,
    )
    workloads, _ = build_longbench_workloads(
        prepared,
        tokenizer=tokenizer,
        task="gov_report",
        nominal_lengths=nominal_lengths,
        sample_count=sample_count,
        sample_policy="matched",
        max_new_tokens=max_new_tokens,
        model_position_limit=model_position_limit,
    )
    return workloads
