"""Pure-Python comparison of existing Phase 2A verification artifacts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

TOTAL_FIELDS = (
    "verification_cuda_ms",
    "self_attention_module_cuda_ms",
    "mlp_module_cuda_ms",
    "lm_head_cuda_ms",
    "unattributed_cuda_ms",
    "self_attention_over_verification",
    "mlp_over_verification",
    "lm_head_over_verification",
    "draft_cuda_ms",
    "draft_over_round",
    "verification_over_round",
)

COMPARISON_CSV_FIELDS = (
    "task",
    "context_length",
    "actual_prompt_length",
    "sample_id",
    "source_sample_id",
    *TOTAL_FIELDS,
    "unattributed_over_verification",
    "round_count",
    "non_terminal_round_count",
    "returned_token_count",
    "total_committed_token_count",
    "mean_non_terminal_committed_tokens",
    "verification_cuda_ms_per_round",
    "self_attention_module_cuda_ms_per_round",
    "mlp_module_cuda_ms_per_round",
    "draft_cuda_ms_per_round",
    "verification_cuda_ms_per_returned_token",
    "self_attention_cuda_ms_per_returned_token",
    "verification_cuda_ms_per_committed_token",
    "verification_total_vs_1k",
    "verification_per_round_vs_1k",
    "self_attention_total_vs_1k",
    "self_attention_per_round_vs_1k",
    "round_count_vs_1k",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def _divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _summary_value(group: dict[str, Any], field: str) -> float | None:
    metric = group["metrics"].get(field)
    if not metric:
        return None
    value = metric.get("median")
    return float(value) if value is not None else None


def _consistent_repeat_count(
    counts: dict[int, int], *, label: str, artifact: Path
) -> int:
    values = set(counts.values())
    if len(values) != 1:
        raise ValueError(
            f"{label} differs across diagnostic repeats in {artifact}: {counts}"
        )
    return next(iter(values))


def _artifact_counts(
    rounds: list[dict[str, Any]],
    *,
    rounds_path: Path,
    warnings: list[str],
) -> dict[str, int | float | None]:
    by_repeat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rounds:
        by_repeat[int(row.get("diagnostic_run", 0))].append(row)
    round_count = _consistent_repeat_count(
        {repeat: len(rows) for repeat, rows in by_repeat.items()},
        label="round_count",
        artifact=rounds_path,
    )

    accounting_fields = (
        "stop_reason",
        "truncated_token_count",
        "committed_token_count",
        "returned_token_delta",
    )
    if any(row.get(field) is None for row in rounds for field in accounting_fields):
        warnings.append(
            f"{rounds_path}: exact terminal/commit accounting fields are absent; "
            "non-terminal and committed-token metrics are null, not inferred"
        )
        return {
            "round_count": round_count,
            "non_terminal_round_count": None,
            "total_committed_token_count": None,
            "mean_non_terminal_committed_tokens": None,
        }

    per_repeat = {}
    for repeat, repeat_rows in by_repeat.items():
        non_terminal = [
            row
            for row in repeat_rows
            if row["stop_reason"] == "continue"
            and int(row["truncated_token_count"]) == 0
        ]
        per_repeat[repeat] = {
            "non_terminal_round_count": len(non_terminal),
            "total_committed_token_count": sum(
                int(row["committed_token_count"]) for row in repeat_rows
            ),
            "non_terminal_committed_sum": sum(
                int(row["committed_token_count"]) for row in non_terminal
            ),
        }
    output: dict[str, int | float | None] = {"round_count": round_count}
    for field in ("non_terminal_round_count", "total_committed_token_count"):
        output[field] = _consistent_repeat_count(
            {repeat: int(values[field]) for repeat, values in per_repeat.items()},
            label=field,
            artifact=rounds_path,
        )
    committed_sum = _consistent_repeat_count(
        {
            repeat: int(values["non_terminal_committed_sum"])
            for repeat, values in per_repeat.items()
        },
        label="non_terminal_committed_sum",
        artifact=rounds_path,
    )
    output["mean_non_terminal_committed_tokens"] = _divide(
        committed_sum, output["non_terminal_round_count"]
    )
    return output


def _one_comparison_row(
    *,
    group: dict[str, Any],
    summary_path: Path,
    warnings: list[str],
) -> dict[str, Any]:
    task = group["task"]
    context_length = int(group["context_length"])
    samples_path = summary_path.parent / "samples.jsonl"
    rounds_path = summary_path.parent / "verification-profile-rounds.jsonl"
    samples = [
        row
        for row in _read_jsonl(samples_path)
        if row.get("task") == task
        and int(row.get("nominal_prompt_length", -1)) == context_length
    ]
    if len(samples) != 1:
        raise ValueError(
            f"Comparison requires exactly one sample artifact for {task} "
            f"at {context_length}; found {len(samples)} in {samples_path}"
        )
    sample = samples[0]
    rounds = [
        row
        for row in _read_jsonl(rounds_path)
        if row.get("task") == task
        and row.get("sample_id") == sample["sample_id"]
        and int(row.get("nominal_prompt_length", -1)) == context_length
    ]
    if not rounds:
        raise ValueError(
            f"No true round records for {sample['sample_id']} at {context_length} "
            f"in {rounds_path}"
        )
    counts = _artifact_counts(rounds, rounds_path=rounds_path, warnings=warnings)
    row = {
        "task": task,
        "context_length": context_length,
        "actual_prompt_length": sample.get("actual_prompt_length"),
        "sample_id": sample["sample_id"],
        "source_sample_id": sample["source_sample_id"],
        **{field: _summary_value(group, field) for field in TOTAL_FIELDS},
        **counts,
        "returned_token_count": int(sample["generated_token_count"]),
        "source_summary": str(summary_path),
        "source_samples": str(samples_path),
        "source_rounds": str(rounds_path),
    }
    row["unattributed_over_verification"] = _divide(
        row["unattributed_cuda_ms"], row["verification_cuda_ms"]
    )
    row["verification_cuda_ms_per_round"] = _divide(
        row["verification_cuda_ms"], row["round_count"]
    )
    row["self_attention_module_cuda_ms_per_round"] = _divide(
        row["self_attention_module_cuda_ms"], row["round_count"]
    )
    row["mlp_module_cuda_ms_per_round"] = _divide(
        row["mlp_module_cuda_ms"], row["round_count"]
    )
    row["draft_cuda_ms_per_round"] = _divide(row["draft_cuda_ms"], row["round_count"])
    row["verification_cuda_ms_per_returned_token"] = _divide(
        row["verification_cuda_ms"], row["returned_token_count"]
    )
    row["self_attention_cuda_ms_per_returned_token"] = _divide(
        row["self_attention_module_cuda_ms"], row["returned_token_count"]
    )
    row["verification_cuda_ms_per_committed_token"] = _divide(
        row["verification_cuda_ms"], row["total_committed_token_count"]
    )
    return row


def build_verification_length_comparison(
    summary_paths: list[Path],
) -> dict[str, Any]:
    if not summary_paths:
        raise ValueError("At least one verification summary is required")
    warnings: list[str] = []
    rows = []
    for raw_path in summary_paths:
        path = raw_path.expanduser().resolve()
        summary = json.loads(path.read_text(encoding="utf-8"))
        if (
            summary.get("status") != "ok"
            or summary.get("run_mode") != "verification-profile"
        ):
            raise ValueError(f"Not a successful verification-profile summary: {path}")
        groups = summary["aggregates"]["by_task_and_context_length"].values()
        rows.extend(
            _one_comparison_row(group=group, summary_path=path, warnings=warnings)
            for group in groups
        )
    rows.sort(key=lambda row: (row["task"], row["context_length"]))
    keys = {(row["task"], row["context_length"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("Duplicate task/context-length summaries were supplied")
    if len({row["task"] for row in rows}) != 1:
        raise ValueError("Do not mix LongBench tasks in one length comparison")

    source_ids_by_length = {
        str(row["context_length"]): row["source_sample_id"] for row in rows
    }
    paired = len(set(source_ids_by_length.values())) == 1
    if not paired:
        warnings.append(
            "Different source sample IDs were used across context lengths; this is "
            "an unpaired pilot trend, not a paired context-length comparison."
        )

    baseline = next((row for row in rows if row["context_length"] == 1024), None)
    scaling_comparisons = {}
    for row in rows:
        for field in (
            "verification_total_vs_1k",
            "verification_per_round_vs_1k",
            "self_attention_total_vs_1k",
            "self_attention_per_round_vs_1k",
            "round_count_vs_1k",
        ):
            row[field] = None
        if baseline is None:
            continue
        row["verification_total_vs_1k"] = _divide(
            row["verification_cuda_ms"], baseline["verification_cuda_ms"]
        )
        row["verification_per_round_vs_1k"] = _divide(
            row["verification_cuda_ms_per_round"],
            baseline["verification_cuda_ms_per_round"],
        )
        row["self_attention_total_vs_1k"] = _divide(
            row["self_attention_module_cuda_ms"],
            baseline["self_attention_module_cuda_ms"],
        )
        row["self_attention_per_round_vs_1k"] = _divide(
            row["self_attention_module_cuda_ms_per_round"],
            baseline["self_attention_module_cuda_ms_per_round"],
        )
        row["round_count_vs_1k"] = _divide(row["round_count"], baseline["round_count"])
        if row["context_length"] in (16384, 32768):
            scaling_comparisons[f"1k_to_{row['context_length'] // 1024}k"] = {
                "from_context_length": 1024,
                "to_context_length": row["context_length"],
                "verification_total_scaling": row["verification_total_vs_1k"],
                "verification_per_round_scaling": row["verification_per_round_vs_1k"],
                "self_attention_total_scaling": row["self_attention_total_vs_1k"],
                "self_attention_per_round_scaling": row[
                    "self_attention_per_round_vs_1k"
                ],
                "round_count_scaling": row["round_count_vs_1k"],
            }
    if baseline is None:
        warnings.append("No 1K baseline was supplied; 1K-relative scaling is null")

    return {
        "schema_version": "1.0",
        "status": "ok",
        "paired_comparison": paired,
        "source_sample_ids_by_context_length": source_ids_by_length,
        "warnings": warnings,
        "measurement_semantics": {
            "generation_total_metrics": "median component time from the profile path",
            "per_round_denominator": "true verification-profile round artifact count",
            "per_returned_token_denominator": "generated_token_count from samples.jsonl",
            "per_committed_token_denominator": (
                "exact committed_token_count from round artifacts; null when absent"
            ),
            "self_attention_scope": (
                "target module including QKV projections, attention/SDPA path, "
                "and output projection; not pure SDPA or pure KV-access time"
            ),
        },
        "rows": rows,
        "scaling_comparisons": scaling_comparisons,
    }


def write_verification_length_comparison(
    output_dir: Path, comparison: dict[str, Any]
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "verification-length-comparison.json"
    csv_path = output_dir / "verification-length-comparison.csv"
    json_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COMPARISON_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in COMPARISON_CSV_FIELDS}
            for row in comparison["rows"]
        )
    return {"comparison_json": json_path, "comparison_csv": csv_path}
