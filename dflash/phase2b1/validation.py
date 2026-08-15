"""Strict preflight for immutable Phase 2B capture directories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from dflash.phase2b.capture import validate_capture_round

REQUIRED_ARTIFACTS = (
    "manifest.json",
    "samples.jsonl",
    "tree-capture-rounds.jsonl",
    "environment.json",
    "source-provenance.json",
    "terminal.log",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("row is not an object")
                rows.append(row)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"cannot read valid JSONL: {path}") from exc
    return rows


def _validate_schema(instance: Any, path: Path) -> None:
    schema = _read_json(path)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(instance),
        key=lambda error: list(error.path),
    )
    if errors:
        error = errors[0]
        location = "/".join(str(value) for value in error.absolute_path) or "<root>"
        raise ValueError(f"schema validation failed at {location}: {error.message}")


def _sample_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["task"]),
        str(row["sample_id"]),
        int(row["nominal_prompt_length"]),
    )


def validate_capture_directories(
    capture_dirs: list[Path], *, schema_dir: Path
) -> dict[str, Any]:
    """Validate every input completely before a caller creates output files."""

    if not capture_dirs:
        raise ValueError("at least one --capture-dir is required")
    all_rounds: list[dict[str, Any]] = []
    inputs = []
    global_round_ids: set[tuple[Any, ...]] = set()
    for raw_dir in capture_dirs:
        directory = raw_dir.expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"capture directory does not exist: {directory}")
        paths = {name: directory / name for name in REQUIRED_ARTIFACTS}
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise ValueError(f"capture directory lacks required files: {missing}")

        manifest = _read_json(paths["manifest.json"])
        samples = _read_jsonl(paths["samples.jsonl"])
        rounds = _read_jsonl(paths["tree-capture-rounds.jsonl"])
        provenance = _read_json(paths["source-provenance.json"])
        _validate_schema(manifest, schema_dir / "phase2b-capture-manifest.schema.json")
        _validate_schema(
            provenance,
            schema_dir / "phase2b1-capture-source-provenance.schema.json",
        )
        for sample in samples:
            _validate_schema(sample, schema_dir / "phase2b-capture-sample.schema.json")
        for row in rounds:
            _validate_schema(row, schema_dir / "phase2b-capture-round.schema.json")

        configuration = manifest["configuration"]
        required = {
            "phase": "2B",
            "mode": "exact_draft_candidate_capture",
            "block_size": 16,
            "batch_size": 1,
        }
        for field, expected in required.items():
            if configuration.get(field) != expected:
                raise ValueError(
                    f"capture configuration {field} must equal {expected!r}"
                )
        if float(configuration.get("temperature", -1)) != 0.0:
            raise ValueError("capture temperature must equal zero")
        if int(configuration.get("top_k", 0)) < 16:
            raise ValueError("capture Top-K must be at least 16")
        if manifest.get("status") != "ok" or provenance.get("status") != "ok":
            raise ValueError("capture manifest and source provenance must be ok")
        if not provenance.get("loaded_source_matches_repository"):
            raise ValueError("capture source provenance did not pass source matching")

        sample_map: dict[tuple[str, str, int], dict[str, Any]] = {}
        for sample in samples:
            key = _sample_key(sample)
            if key in sample_map:
                raise ValueError(f"duplicate capture sample key: {key}")
            if not sample.get("control_capture_output_token_ids_equal"):
                raise ValueError(
                    "capture sample failed control/capture token alignment"
                )
            sample_map[key] = sample
        if not sample_map or not rounds:
            raise ValueError("capture samples and rounds must be non-empty")
        tasks = {key[0] for key in sample_map}
        if len(tasks) != 1:
            raise ValueError("one capture directory must contain exactly one task")
        manifest_task = manifest.get("workload", {}).get("longbench_task")
        if manifest_task is not None and manifest_task not in tasks:
            raise ValueError("manifest workload task does not match sample task")

        seen_local: set[tuple[Any, ...]] = set()
        for row in rounds:
            validate_capture_round(row)
            if int(row["top_k"]) < 16 or int(row["top_k"]) != int(
                configuration["top_k"]
            ):
                raise ValueError("round Top-K must match manifest and be at least 16")
            if row.get("score_semantics") != (
                "normalized log probability from log_softmax"
            ):
                raise ValueError(
                    "Phase 2B.1 probability diagnostics require captured "
                    "log_softmax log probabilities"
                )
            if row.get("tie_break_rule") != (
                "descending_logit_then_ascending_token_id"
            ):
                raise ValueError("capture tie-break rule is unsupported")
            key = _sample_key(row)
            if key not in sample_map:
                raise ValueError(f"capture round has no matching sample: {key}")
            sample = sample_map[key]
            for field in (
                "task",
                "sample_id",
                "source_sample_id",
                "source_text_sha256",
                "nominal_prompt_length",
                "actual_prompt_length",
            ):
                if row.get(field) != sample.get(field):
                    raise ValueError(f"round/sample mismatch for {field}: {key}")
            if (
                row["final_verified_output_token_ids"]
                != sample["final_verified_output_token_ids"]
            ):
                raise ValueError(f"round/sample final output mismatch: {key}")
            reference = row["reference_target_continuation_token_ids"]
            observed = int(row["reference_observed_depth"])
            right_censored = bool(row["reference_right_censored"])
            if len(reference) != observed:
                raise ValueError("reference_observed_depth must equal reference length")
            if right_censored != (observed < 15):
                raise ValueError(
                    "right-censored flag must exactly mean observable depth < 15"
                )
            if bool(row["full_depth_evaluation_eligible"]) != (observed == 15):
                raise ValueError("full-depth eligibility is inconsistent")
            if right_censored and not bool(row["terminal_round"]):
                raise ValueError("right-censored capture must be terminal")
            if right_censored and not row.get("censor_reason"):
                raise ValueError("right-censored capture must include censor_reason")
            if (
                row["alignment"].get("immediate_verification_invariant_holds")
                is not True
            ):
                raise ValueError("immediate baseline alignment did not pass")
            if (
                row["alignment"].get("exact_lcp_invariant_evaluable")
                and row["alignment"].get("exact_lcp_invariant_holds") is not True
            ):
                raise ValueError("final-output baseline alignment did not pass")
            row_provenance = row.get("capture_source_provenance", {})
            for row_field, provenance_field in (
                ("repository_custom_code_commit", "repository_custom_code_commit"),
                ("materialization_id", "materialization_id"),
                ("actual_loaded_module_sha256", "actual_loaded_module_sha256"),
            ):
                if row_provenance.get(row_field) != provenance.get(provenance_field):
                    raise ValueError(
                        f"round capture provenance mismatch for {row_field}: {key}"
                    )
            expected_dependency_hashes = {
                filename: record["actual_loaded_sha256"]
                for filename, record in provenance["actual_loaded_dependencies"].items()
            }
            if row_provenance.get("actual_loaded_dependency_sha256") != (
                expected_dependency_hashes
            ):
                raise ValueError(f"round capture dependency provenance mismatch: {key}")
            round_id = (*key, int(row["generation_repeat"]), int(row["round_index"]))
            if round_id in seen_local or round_id in global_round_ids:
                raise ValueError(
                    f"duplicate round ID across capture inputs: {round_id}"
                )
            seen_local.add(round_id)
            global_round_ids.add(round_id)
            row = dict(row)
            row["_capture_directory"] = str(directory)
            all_rounds.append(row)

        file_hashes = {name: sha256_file(path) for name, path in paths.items()}
        canonical = "".join(
            f"{name}:{file_hashes[name]}\n" for name in sorted(file_hashes)
        )
        inputs.append(
            {
                "capture_directory": str(directory),
                "capture_directory_sha256": hashlib.sha256(
                    canonical.encode("utf-8")
                ).hexdigest(),
                "artifact_sha256": file_hashes,
                "capture_schema_version": manifest["schema_version"],
                "source_provenance_schema": "phase2b1-capture-source-provenance.schema.json",
                "source_provenance_schema_version": provenance["schema_version"],
                "input_git_sha": provenance["repository_custom_code_commit"],
                "top_k": int(configuration["top_k"]),
                "task": next(iter(sample_map))[0],
                "sample_count": len(samples),
                "round_count": len(rounds),
            }
        )
    return {"rounds": all_rounds, "inputs": inputs}
