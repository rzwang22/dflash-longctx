"""Deterministic, offline GovReport workload construction.

This module deliberately imports neither Transformers nor datasets at import time.
The caller supplies the actual target tokenizer after model provenance is checked.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_CONTEXT_LENGTHS = (1024, 4096, 8192, 16384, 32768)
PROMPT_TEMPLATE_ID = "govreport-summary-v1"
MAX_NOMINAL_SHORTFALL = 16
PROMPT_PREFIX = (
    "Summarize the following U.S. government report. Preserve the main findings, "
    "evidence, and recommendations.\n\nReport:\n"
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_token_ids(token_ids: Sequence[int]) -> str:
    canonical = json.dumps(list(token_ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"GovReport JSONL does not exist: {path}")
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(
                    f"GovReport JSONL line {line_number} is not a JSON object"
                )
            records.append(value)
    return records


def load_govreport_records(
    *,
    jsonl_path: Path | None = None,
    dataset_path: Path | None = None,
    dataset_cache_dir: Path | None = None,
    dataset_name: str = "ccdv/govreport-summarization",
    dataset_config: str | None = None,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Load one explicitly local GovReport source without network fallback.

    ``dataset_path`` is a directory produced by ``Dataset.save_to_disk``.
    ``dataset_cache_dir`` uses the Hugging Face datasets cache in strict offline
    mode and fails if the requested builder/config/split is not already cached.
    """

    choices = [
        jsonl_path is not None,
        dataset_path is not None,
        dataset_cache_dir is not None,
    ]
    if sum(choices) != 1:
        raise ValueError(
            "Choose exactly one of jsonl_path, dataset_path, or dataset_cache_dir"
        )
    if jsonl_path is not None:
        return _jsonl_records(jsonl_path)

    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        from datasets import DatasetDict, load_dataset, load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required for a local Hugging Face dataset source"
        ) from exc

    if dataset_path is not None:
        resolved = dataset_path.expanduser().resolve()
        if not resolved.exists():
            raise RuntimeError(f"Saved GovReport dataset does not exist: {resolved}")
        dataset = load_from_disk(str(resolved))
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise RuntimeError(
                    f"Split {split!r} is absent from saved dataset {resolved}"
                )
            dataset = dataset[split]
    else:
        resolved = dataset_cache_dir.expanduser().resolve()
        if not resolved.is_dir():
            raise RuntimeError(f"GovReport dataset cache does not exist: {resolved}")
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            cache_dir=str(resolved),
            download_mode="reuse_dataset_if_exists",
        )
    return [dict(record) for record in dataset]


def render_prompt(report_prefix: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": PROMPT_PREFIX + report_prefix}]


def _as_token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise RuntimeError("Phase 1B requires batch size 1")
        value = value[0]
    return [int(token) for token in value]


def tokenize_chat(tokenizer: Any, report_prefix: str) -> list[int]:
    tokenized = tokenizer.apply_chat_template(
        render_prompt(report_prefix),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return _as_token_ids(tokenized)


def _prefix_at_or_below(
    tokenizer: Any, report: str, nominal_length: int
) -> tuple[str, list[int]]:
    """Find the longest character prefix whose templated prompt fits the target."""

    empty_ids = tokenize_chat(tokenizer, "")
    if len(empty_ids) > nominal_length:
        raise ValueError(
            f"Prompt template alone has {len(empty_ids)} tokens, above {nominal_length}"
        )
    full_ids = tokenize_chat(tokenizer, report)
    if len(full_ids) < nominal_length:
        raise ValueError(
            f"Report reaches only {len(full_ids)} templated tokens; "
            f"{nominal_length} requested"
        )

    low, high = 0, len(report)
    best_characters, best_ids = 0, empty_ids
    while low <= high:
        middle = (low + high) // 2
        candidate_ids = tokenize_chat(tokenizer, report[:middle])
        if len(candidate_ids) <= nominal_length:
            if len(candidate_ids) > len(best_ids) or (
                len(candidate_ids) == len(best_ids) and middle > best_characters
            ):
                best_characters, best_ids = middle, candidate_ids
            low = middle + 1
        else:
            high = middle - 1

    # BPE boundary effects can make adjacent counts locally non-monotonic. Inspect
    # the small boundary region without scanning or padding the report.
    boundary_start = max(0, best_characters - 16)
    boundary_end = min(len(report), max(low, best_characters) + 16)
    for characters in range(boundary_start, boundary_end + 1):
        candidate_ids = tokenize_chat(tokenizer, report[:characters])
        if len(candidate_ids) <= nominal_length and (
            len(candidate_ids) > len(best_ids)
            or (len(candidate_ids) == len(best_ids) and characters > best_characters)
        ):
            best_characters, best_ids = characters, candidate_ids
    return report[:best_characters], best_ids


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
    """Build deterministic nested report prefixes using actual templated token IDs."""

    lengths = tuple(int(length) for length in nominal_lengths)
    if not lengths or tuple(sorted(set(lengths))) != lengths:
        raise ValueError("nominal_lengths must be unique and strictly increasing")
    if min(lengths) <= 0 or sample_count <= 0 or max_new_tokens <= 0:
        raise ValueError("lengths, sample_count, and max_new_tokens must be positive")
    if max(lengths) + max_new_tokens > model_position_limit:
        raise ValueError(
            f"Largest prompt ({max(lengths)}) plus max_new_tokens "
            f"({max_new_tokens}) exceeds model position limit {model_position_limit}"
        )

    candidates = []
    for source_index, record in enumerate(records):
        report = _field(
            record, text_field, ("report", "document", "article", "input", "text")
        )
        if not isinstance(report, str) or not report.strip():
            continue
        text_hash = sha256_text(report)
        if id_field is not None:
            source_id = str(_field(record, id_field, ()))
        else:
            source_id = str(
                next(
                    (
                        record[name]
                        for name in ("id", "report_id", "document_id")
                        if name in record
                    ),
                    f"sha256:{text_hash[:16]}",
                )
            )
        selection_key = sha256_text(f"{seed}\0{split}\0{source_id}\0{text_hash}")
        candidates.append((selection_key, source_index, source_id, report, text_hash))

    candidates.sort(key=lambda value: (value[0], value[1]))
    selected = []
    for candidate in candidates:
        full_prompt_length = len(tokenize_chat(tokenizer, candidate[3]))
        if full_prompt_length >= max(lengths):
            selected.append((*candidate, full_prompt_length))
            if len(selected) == sample_count:
                break
    if len(selected) < sample_count:
        raise RuntimeError(
            f"Only {len(selected)} reports reach {max(lengths)} tokens; "
            f"{sample_count} required. No filler or repeated text was used."
        )

    workloads = []
    for selection_rank, (
        _,
        source_index,
        source_id,
        report,
        text_hash,
        full_length,
    ) in enumerate(selected):
        previous_prefix = ""
        sample_id = f"govreport-{selection_rank:03d}-{text_hash[:12]}"
        for nominal_length in lengths:
            report_prefix, input_ids = _prefix_at_or_below(
                tokenizer, report, nominal_length
            )
            if not report_prefix.startswith(previous_prefix):
                raise RuntimeError("GovReport prefixes are not nested")
            actual_length = len(input_ids)
            nominal_shortfall = nominal_length - actual_length
            if nominal_shortfall > MAX_NOMINAL_SHORTFALL:
                raise RuntimeError(
                    f"Closest nested prompt has {actual_length} tokens for nominal "
                    f"length {nominal_length}, exceeding the allowed shortfall of "
                    f"{MAX_NOMINAL_SHORTFALL}"
                )
            if actual_length + max_new_tokens > model_position_limit:
                raise ValueError(
                    f"Prompt {actual_length} plus max_new_tokens {max_new_tokens} "
                    f"exceeds model position limit {model_position_limit}"
                )
            prompt_text = PROMPT_PREFIX + report_prefix
            workloads.append(
                {
                    "sample_id": sample_id,
                    "source_sample_id": source_id,
                    "source_index": source_index,
                    "selection_rank": selection_rank,
                    "dataset_split": split,
                    "random_seed": seed,
                    "prompt_template_id": PROMPT_TEMPLATE_ID,
                    "source_text_sha256": text_hash,
                    "source_full_prompt_length": full_length,
                    "nominal_prompt_length": nominal_length,
                    "actual_prompt_length": actual_length,
                    "exact_nominal_length": actual_length == nominal_length,
                    "nominal_prompt_length_shortfall": nominal_shortfall,
                    "report_prefix_characters": len(report_prefix),
                    "report_prefix_sha256": sha256_text(report_prefix),
                    "prompt_sha256": sha256_text(prompt_text),
                    "input_token_ids_sha256": sha256_token_ids(input_ids),
                    "prompt_text": prompt_text,
                    "input_ids": input_ids,
                }
            )
            previous_prefix = report_prefix
    return workloads
