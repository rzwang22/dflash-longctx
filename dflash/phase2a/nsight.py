"""Parse official Nsight Systems NVTX/kernel CSV reports transparently."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

KERNEL_CLASSIFICATION_RULES = (
    {
        "id": "attention-name-v1",
        "pattern": r"(?i)(flash|fmha|fused.*attn|attention|scaled.*dot.*product|sdpa)",
        "category": "sdpa_or_attention",
    },
    {
        "id": "gemm-name-v1",
        "pattern": r"(?i)(gemm|matmul|cublas|cutlass|mma)",
        "category": "gemm_or_matmul",
    },
    {
        "id": "normalization-name-v1",
        "pattern": r"(?i)(layer.?norm|rms.?norm|normalization|norm_kernel)",
        "category": "normalization",
    },
    {
        "id": "elementwise-name-v1",
        "pattern": r"(?i)(elementwise|vectorized|pointwise|unrolled_elementwise|copy|cast)",
        "category": "elementwise_or_copy",
    },
)


def classify_kernel(raw_kernel_name: str) -> dict[str, str]:
    for rule in KERNEL_CLASSIFICATION_RULES:
        if re.search(rule["pattern"], raw_kernel_name):
            return {
                "classification_rule": rule["id"],
                "classification_pattern": rule["pattern"],
                "classified_category": rule["category"],
            }
    return {
        "classification_rule": "unmatched-v1",
        "classification_pattern": "no configured pattern matched",
        "classified_category": "other",
    }


def _normalized(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _column(fieldnames: Sequence[str], *candidates: str) -> str:
    normalized = {_normalized(name): name for name in fieldnames}
    for candidate in candidates:
        candidate_normalized = _normalized(candidate)
        if candidate_normalized in normalized:
            return normalized[candidate_normalized]
    for candidate in candidates:
        candidate_normalized = _normalized(candidate)
        for normalized_name, original in normalized.items():
            if candidate_normalized in normalized_name:
                return original
    raise ValueError(f"Could not find columns {candidates}; got {list(fieldnames)}")


def read_nvtx_kernel_summary(path: Path) -> list[dict[str, Any]]:
    """Read ``nsys stats --report nvtx_kern_sum`` CSV output."""

    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"Nsight kernel CSV has no header: {path}")
        range_column = _column(reader.fieldnames, "NVTX Range")
        kernel_column = _column(reader.fieldnames, "Kernel Name")
        count_column = _column(
            reader.fieldnames, "Kern Inst", "Kernel Instances", "Instances"
        )
        duration_column = _column(reader.fieldnames, "Total Time")
        rows = []
        for source in reader:
            range_name = source[range_column].strip()
            raw_kernel_name = source[kernel_column].strip()
            if not range_name or not raw_kernel_name:
                continue
            rows.append(
                {
                    "nvtx_range": range_name,
                    "raw_kernel_name": raw_kernel_name,
                    "kernel_count": int(float(source[count_column].replace(",", ""))),
                    "total_duration_ns": float(
                        source[duration_column].replace(",", "")
                    ),
                    **classify_kernel(raw_kernel_name),
                }
            )
    return rows


def read_nvtx_projection_summary(path: Path) -> list[dict[str, Any]]:
    """Read ``nsys stats --report nvtx_gpu_proj_sum`` CSV output."""

    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"Nsight NVTX CSV has no header: {path}")
        range_column = _column(reader.fieldnames, "Range")
        duration_column = _column(reader.fieldnames, "Total Proj Time")
        instance_column = _column(reader.fieldnames, "Range Instances")
        operations_column = _column(reader.fieldnames, "Total GPU Ops")
        rows = []
        for source in reader:
            range_name = source[range_column].strip()
            if not range_name:
                continue
            rows.append(
                {
                    "nvtx_range": range_name,
                    "projected_cuda_duration_ns": float(
                        source[duration_column].replace(",", "")
                    ),
                    "range_instance_count": int(
                        float(source[instance_column].replace(",", ""))
                    ),
                    "gpu_operation_count": int(
                        float(source[operations_column].replace(",", ""))
                    ),
                }
            )
    return rows


def _scope(range_name: str) -> str | None:
    if range_name == "dflash.verify":
        return "complete_verification"
    if re.fullmatch(r"target\.verify\.layer\.\d+\.self_attn", range_name):
        return "target_self_attention_modules"
    if re.fullmatch(r"target\.verify\.layer\.\d+\.mlp", range_name):
        return "target_mlp_modules"
    if range_name == "target.verify.lm_head":
        return "target_lm_head"
    if range_name == "dflash.attention.combined_context_block":
        return "draft_combined_attention"
    return None


def build_kernel_breakdown(
    kernel_rows: Sequence[dict[str, Any]], nvtx_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    scoped_rows = []
    for row in kernel_rows:
        scope = _scope(row["nvtx_range"])
        if scope is not None:
            scoped_rows.append({**row, "scope": scope})

    scope_category: dict[tuple[str, str], dict[str, Any]] = {}
    for row in scoped_rows:
        key = (row["scope"], row["classified_category"])
        aggregate = scope_category.setdefault(
            key,
            {
                "scope": row["scope"],
                "classified_category": row["classified_category"],
                "kernel_count": 0,
                "total_duration_ns": 0.0,
            },
        )
        aggregate["kernel_count"] += row["kernel_count"]
        aggregate["total_duration_ns"] += row["total_duration_ns"]

    scope_totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"kernel_count": 0, "total_duration_ns": 0.0}
    )
    for row in scoped_rows:
        scope_totals[row["scope"]]["kernel_count"] += row["kernel_count"]
        scope_totals[row["scope"]]["total_duration_ns"] += row["total_duration_ns"]
    verification_duration = scope_totals.get("complete_verification", {}).get(
        "total_duration_ns", 0.0
    )
    if not verification_duration:
        raise ValueError(
            "Nsight kernel report has no nonzero dflash.verify kernel duration"
        )
    for scope, totals in scope_totals.items():
        totals["over_complete_verification_kernel_time"] = (
            totals["total_duration_ns"] / verification_duration
        )
    for aggregate in scope_category.values():
        aggregate["over_complete_verification_kernel_time"] = (
            aggregate["total_duration_ns"] / verification_duration
        )
    for row in scoped_rows:
        row["over_complete_verification_kernel_time"] = (
            row["total_duration_ns"] / verification_duration
        )

    return {
        "schema_version": "1.0",
        "status": "parsed",
        "source_reports": ["nvtx_kern_sum", "nvtx_gpu_proj_sum"],
        "scope_totals": dict(scope_totals),
        "scope_category_totals": list(scope_category.values()),
        "kernel_rows": scoped_rows,
        "nvtx_projection_rows": [
            row for row in nvtx_rows if _scope(row["nvtx_range"]) is not None
        ],
        "classification_rules": list(KERNEL_CLASSIFICATION_RULES),
        "classification_semantics": {
            "transparent_name_based_heuristic": True,
            "raw_kernel_name_retained": True,
            "rule_retained_per_row": True,
            "requires_h100_review": True,
            "pure_sdpa_fraction_requires_kernel_evidence": True,
            "nested_nvtx_scopes_may_count_the_same_kernel_in_multiple_scopes": True,
        },
    }


def write_nsight_outputs(
    output_dir: Path,
    *,
    kernel_rows: Sequence[dict[str, Any]],
    nvtx_rows: Sequence[dict[str, Any]],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    breakdown = build_kernel_breakdown(kernel_rows, nvtx_rows)
    kernel_path = output_dir / "nsight-kernel-summary.csv"
    nvtx_path = output_dir / "nsight-nvtx-summary.csv"
    breakdown_path = output_dir / "verification-kernel-breakdown.json"

    kernel_fields = (
        "scope",
        "nvtx_range",
        "raw_kernel_name",
        "classification_rule",
        "classification_pattern",
        "classified_category",
        "kernel_count",
        "total_duration_ns",
        "over_complete_verification_kernel_time",
    )
    with kernel_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=kernel_fields)
        writer.writeheader()
        writer.writerows(breakdown["kernel_rows"])

    nvtx_fields = (
        "nvtx_range",
        "projected_cuda_duration_ns",
        "range_instance_count",
        "gpu_operation_count",
    )
    with nvtx_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=nvtx_fields)
        writer.writeheader()
        writer.writerows(breakdown["nvtx_projection_rows"])
    breakdown_path.write_text(
        json.dumps(breakdown, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "kernel_summary_csv": kernel_path,
        "nvtx_summary_csv": nvtx_path,
        "verification_kernel_breakdown_json": breakdown_path,
    }
