"""Strict read-only Phase 2D.0 v2 preflight for Phase 2D.1A."""

from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from dflash.phase2b1.validation import sha256_file

REQUIRED_PHASE2D0_FILES = (
    "opportunity-events.jsonl",
    "gap-decomposition-v2.json",
    "context-gap-composition-v2.csv",
    "research-question-summary-v2.json",
    "analysis-manifest.json",
    "terminal.log",
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
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"line {line_number} is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"cannot read valid JSONL: {path}") from exc
    return rows


@cache
def _validator(schema_path: str) -> Draft202012Validator:
    schema = _read_json(Path(schema_path))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate(value: Any, schema_path: Path) -> None:
    _validator(str(schema_path.resolve())).validate(value)


def _round_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["task"],
        row["sample_id"],
        int(row["nominal_prompt_length"]),
        int(row["generation_repeat"]),
        int(row["round_index"]),
    )


def _input_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["task"],
        int(row["sample_count"]),
        int(row["round_count"]),
        int(row["top_k"]),
        row["capture_directory_sha256"],
    )


def validate_phase2d0_directory(
    phase2d0_dir: Path,
    *,
    validated_capture: dict[str, Any],
    schema_dir: Path,
    top_m: int,
    node_budgets: tuple[int, ...],
) -> dict[str, Any]:
    directory = phase2d0_dir.expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Phase 2D.0 directory does not exist: {directory}")
    paths = {name: directory / name for name in REQUIRED_PHASE2D0_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"Phase 2D.0 directory lacks required v2 files: {missing}")

    manifest = _read_json(paths["analysis-manifest.json"])
    gap = _read_json(paths["gap-decomposition-v2.json"])
    rq = _read_json(paths["research-question-summary-v2.json"])
    events = _read_jsonl(paths["opportunity-events.jsonl"])
    _validate(manifest, schema_dir / "phase2d0-analysis-manifest.schema.json")
    _validate(gap, schema_dir / "phase2d0-gap-decomposition.schema.json")
    _validate(rq, schema_dir / "phase2d0-research-question-summary.schema.json")
    for event in events:
        _validate(event, schema_dir / "phase2d0-opportunity-event.schema.json")

    if manifest["status"] != "ok" or any(manifest["framework_modules_loaded"].values()):
        raise ValueError("Phase 2D.0 input is not a successful CPU-only analysis")
    configuration = manifest["configuration"]
    if top_m not in configuration["top_ms"]:
        raise ValueError(f"Phase 2D.0 input does not contain Top-{top_m}")
    if not set(node_budgets) <= set(configuration["node_budgets"]):
        raise ValueError("Phase 2D.0 input lacks requested node-budget strata")
    required_policies = {
        "top1_chain",
        "score_beam",
        "confidence_gated",
        "layerwise_beam",
    }
    if not required_policies <= set(configuration["policies"]):
        raise ValueError("Phase 2D.0 input lacks the canonical policy family")

    capture_signatures = sorted(
        _input_signature(row) for row in validated_capture["inputs"]
    )
    manifest_signatures = sorted(_input_signature(row) for row in manifest["inputs"])
    if capture_signatures != manifest_signatures:
        raise ValueError("Phase 2D.0 capture provenance/hash mismatch")

    capture_keys = {_round_key(row) for row in validated_capture["rounds"]}
    selected = [
        event
        for event in events
        if int(event["top_m"]) == top_m and int(event["node_budget"]) in node_budgets
    ]
    expected_event_keys = {
        (*round_key, node_budget)
        for round_key in capture_keys
        for node_budget in node_budgets
    }
    actual_event_keys = {
        (*_round_key(event), int(event["node_budget"])) for event in selected
    }
    if len(actual_event_keys) != len(selected):
        raise ValueError("Phase 2D.0 input contains duplicate selector event keys")
    if actual_event_keys != expected_event_keys:
        missing_keys = sorted(expected_event_keys - actual_event_keys)[:3]
        extra_keys = sorted(actual_event_keys - expected_event_keys)[:3]
        raise ValueError(
            "Phase 2D.0/capture round alignment mismatch: "
            f"missing={missing_keys} extra={extra_keys}"
        )
    if gap["configuration_round_count"] != len(events):
        raise ValueError("Phase 2D.0 gap/event count mismatch")
    if rq["decision"] != manifest["decision"]:
        raise ValueError("Phase 2D.0 research-summary/manifest decision mismatch")

    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }
    canonical = "".join(
        f"{name}:{artifact_hashes[name]}\n" for name in sorted(artifact_hashes)
    )
    return {
        "directory": str(directory),
        "manifest": manifest,
        "gap": gap,
        "research_summary": rq,
        "events": selected,
        "artifact_sha256": artifact_hashes,
        "directory_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "validated_capture_signatures": capture_signatures,
    }
