#!/usr/bin/env python3
"""Aggregate independent Phase 2C.4.2 reference-state evidence runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

if __package__:
    from scripts.phase2c42_results import load_case
else:
    from phase2c42_results import load_case


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "docs/schemas/phase2c43-cross-run-evidence.schema.json"
ARTIFACT_NAME = "cross-run-evidence.json"
GROUPING_KEY_FIELDS = [
    "sample_id",
    "nominal_prompt_length",
    "first_actual_semantic_divergence",
]
LOCAL_NUMERICAL = "local_packed_verifier_numerical_divergence"
UNRESOLVED = "unresolved_reference_state_divergence"
SUPPORTED_CLASSIFICATIONS = {
    LOCAL_NUMERICAL,
    "target_kv_state_drift",
    "target_hidden_state_drift",
    "draft_state_or_candidate_drift",
    "commit_or_root_transition_mismatch",
    UNRESOLVED,
}
REPLAY_MODES = ("native_dense", "same_mask", "math")


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _optional_integer(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _integer(value, field)


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ValueError(f"{field} must equal {expected!r}, got {actual!r}")


def _distribution(values: Iterable[int | str | None]) -> dict[str, int]:
    counts = Counter("null" if value is None else str(value) for value in values)

    def sort_key(item: tuple[str, int]) -> tuple[int, int | str]:
        token = item[0]
        if token == "null":
            return (2, token)
        try:
            return (0, int(token))
        except ValueError:
            return (1, token)

    return dict(sorted(counts.items(), key=sort_key))


def _stable_token(values: list[int | None]) -> bool:
    return (
        bool(values)
        and all(value is not None for value in values)
        and len(set(values)) == 1
    )


def _validated_replay(
    row: dict[str, Any], mode: str, reference_token: int, source: Path
) -> tuple[int | None, bool | None]:
    token_field = f"{mode}_replay_token_id"
    match_field = f"{mode}_replay_matches_actual_reference"
    if token_field not in row or match_field not in row:
        raise ValueError(f"{source}: missing {token_field} or {match_field}")
    token = _optional_integer(row[token_field], f"{source}:{token_field}")
    stored_match = row[match_field]
    expected_match = None if token is None else token == reference_token
    if stored_match is not expected_match:
        raise ValueError(
            f"{source}: {match_field} contradicts {token_field} and "
            "reference_actual_token_id"
        )
    return token, expected_match


def load_run_evidence(output_dir: Path) -> dict[str, Any]:
    """Load and strictly normalize one existing Phase 2C.4.2 output directory."""

    output_dir = output_dir.expanduser().resolve()
    case, rows = load_case(output_dir)
    summary_path = output_dir / "reference-state-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _require_equal(summary.get("schema_version"), "1.0", f"{summary_path}:schema")
    _require_equal(summary.get("status"), "complete", f"{summary_path}:status")
    _require_equal(
        summary.get("actual_execution_is_semantic_oracle"),
        True,
        f"{summary_path}:actual_execution_is_semantic_oracle",
    )
    _require_equal(
        summary.get("dense_replay_controls_are_attribution_only"),
        True,
        f"{summary_path}:dense_replay_controls_are_attribution_only",
    )
    _require_equal(
        summary.get("formal_timing_contaminated"),
        False,
        f"{summary_path}:formal_timing_contaminated",
    )

    sample_id = case.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{summary_path}: sample_id must be a non-empty string")
    nominal_length = _integer(
        case.get("nominal_prompt_length"),
        f"{summary_path}:nominal_prompt_length",
    )
    divergence_offset = _integer(
        case.get("first_actual_semantic_divergence"),
        f"{summary_path}:first_actual_semantic_divergence",
    )
    divergence_rows = [
        row for row in rows if row.get("generated_prefix_length") == divergence_offset
    ]
    if len(divergence_rows) != 1:
        raise ValueError(
            f"{output_dir}: expected exactly one differential row at actual "
            f"divergence offset {divergence_offset}, got {len(divergence_rows)}"
        )
    row = divergence_rows[0]
    row_path = output_dir / "reference-state-differential.jsonl"
    _require_equal(row.get("sample_id"), sample_id, f"{row_path}:sample_id")
    _require_equal(
        row.get("nominal_prompt_length"),
        nominal_length,
        f"{row_path}:nominal_prompt_length",
    )
    _require_equal(
        row.get("actual_token_divergence"),
        True,
        f"{row_path}:actual_token_divergence",
    )
    _require_equal(
        row.get("actual_decision_oracle_source"),
        "actual_original_dflash_output",
        f"{row_path}:actual_decision_oracle_source",
    )
    _require_equal(
        row.get("dense_replay_control_only"),
        True,
        f"{row_path}:dense_replay_control_only",
    )

    reference_token = _integer(
        row.get("reference_actual_token_id"),
        f"{row_path}:reference_actual_token_id",
    )
    tree_token = _integer(
        row.get("tree_actual_token_id"),
        f"{row_path}:tree_actual_token_id",
    )
    packed_token = _integer(
        row.get("packed_tree_argmax_token_id"),
        f"{row_path}:packed_tree_argmax_token_id",
    )
    if reference_token == tree_token:
        raise ValueError(
            f"{row_path}: first actual semantic divergence has equal actual tokens"
        )
    _require_equal(
        row.get("packed_tree_argmax_matches_actual_reference"),
        packed_token == reference_token,
        f"{row_path}:packed_tree_argmax_matches_actual_reference",
    )
    _require_equal(
        row.get("packed_tree_argmax_matches_actual_tree"),
        packed_token == tree_token,
        f"{row_path}:packed_tree_argmax_matches_actual_tree",
    )

    replay_tokens: dict[str, int | None] = {}
    replay_matches: dict[str, bool | None] = {}
    for mode in REPLAY_MODES:
        token, matches = _validated_replay(row, mode, reference_token, row_path)
        replay_tokens[mode] = token
        replay_matches[mode] = matches

    numerical_support = row.get("local_packed_verifier_numerical_divergence_supported")
    if not isinstance(numerical_support, bool):
        raise TypeError(
            f"{row_path}: "
            "local_packed_verifier_numerical_divergence_supported must be boolean"
        )
    if numerical_support and not (
        replay_matches["same_mask"] is True or replay_matches["math"] is True
    ):
        raise ValueError(
            f"{row_path}: numerical attribution support lacks a same-mask or "
            "math replay matching the actual reference"
        )

    classification = case.get("failure_classification")
    if classification not in SUPPORTED_CLASSIFICATIONS:
        raise ValueError(
            f"{summary_path}: unsupported failure_classification {classification!r}"
        )
    classification_complete = case.get("classification_evidence_complete")
    if not isinstance(classification_complete, bool):
        raise TypeError(
            f"{summary_path}: classification_evidence_complete must be boolean"
        )
    _require_equal(
        row.get("classification_evidence_complete"),
        classification_complete,
        f"{row_path}:classification_evidence_complete",
    )

    return {
        "input_directory": str(output_dir),
        "sample_id": sample_id,
        "nominal_prompt_length": nominal_length,
        "first_actual_semantic_divergence": divergence_offset,
        "actual_reference_token_id": reference_token,
        "actual_tree_token_id": tree_token,
        "packed_tree_token_id": packed_token,
        "native_replay_token_id": replay_tokens["native_dense"],
        "same_mask_replay_token_id": replay_tokens["same_mask"],
        "math_replay_token_id": replay_tokens["math"],
        "native_matches_actual_reference": replay_matches["native_dense"],
        "same_mask_matches_actual_reference": replay_matches["same_mask"],
        "math_matches_actual_reference": replay_matches["math"],
        "numerical_attribution_supported": numerical_support,
        "failure_classification": classification,
        "classification_evidence_complete": classification_complete,
        "unresolved_reason": case.get("unresolved_reason"),
    }


def _support_fields(runs: list[dict[str, Any]], mode: str) -> dict[str, int | float]:
    field = f"{mode}_matches_actual_reference"
    values = [run[field] for run in runs]
    count = sum(value is True for value in values)
    available = sum(value is not None for value in values)
    return {
        f"{mode}_reference_support_count": count,
        f"{mode}_reference_support_rate": count / len(runs),
        f"{mode}_replay_available_count": available,
    }


def aggregate_group(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one absolute-semantic-divergence cohort."""

    if not runs:
        raise ValueError("cannot aggregate an empty run group")
    first = runs[0]
    key = tuple(first[field] for field in GROUPING_KEY_FIELDS)
    for run in runs[1:]:
        if tuple(run[field] for field in GROUPING_KEY_FIELDS) != key:
            raise ValueError("aggregate_group received more than one grouping key")

    reference_tokens = [run["actual_reference_token_id"] for run in runs]
    tree_tokens = [run["actual_tree_token_id"] for run in runs]
    packed_tokens = [run["packed_tree_token_id"] for run in runs]
    reference_stable = _stable_token(reference_tokens)
    tree_stable = _stable_token(tree_tokens)
    packed_stable = _stable_token(packed_tokens)
    stable_actual_divergence = (
        reference_stable
        and tree_stable
        and packed_stable
        and all(
            reference != tree for reference, tree in zip(reference_tokens, tree_tokens)
        )
        and all(packed == tree for packed, tree in zip(packed_tokens, tree_tokens))
    )

    numerical_support_count = sum(
        run["numerical_attribution_supported"] for run in runs
    )
    classification_complete_count = sum(
        run["classification_evidence_complete"] for run in runs
    )
    classifications = _distribution(run["failure_classification"] for run in runs)
    all_local = classifications == {LOCAL_NUMERICAL: len(runs)}
    all_complete = classification_complete_count == len(runs)

    if (
        stable_actual_divergence
        and numerical_support_count == len(runs)
        and all_local
        and all_complete
    ):
        aggregate_classification = LOCAL_NUMERICAL
        aggregate_reason = None
        aggregate_evidence_complete = True
    elif stable_actual_divergence and 0 < numerical_support_count < len(runs):
        aggregate_classification = UNRESOLVED
        aggregate_reason = "nonreproducible_numerical_control_support"
        aggregate_evidence_complete = False
    elif stable_actual_divergence and numerical_support_count == 0:
        aggregate_classification = UNRESOLVED
        aggregate_reason = "no_numerical_control_support"
        aggregate_evidence_complete = False
    elif stable_actual_divergence:
        aggregate_classification = UNRESOLVED
        aggregate_reason = "inconsistent_single_run_attribution"
        aggregate_evidence_complete = False
    else:
        aggregate_classification = UNRESOLVED
        aggregate_reason = "unstable_actual_or_packed_decision_evidence"
        aggregate_evidence_complete = False

    group = {
        "sample_id": first["sample_id"],
        "nominal_prompt_length": first["nominal_prompt_length"],
        "first_actual_semantic_divergence": first["first_actual_semantic_divergence"],
        "run_count": len(runs),
        "actual_reference_token_distribution": _distribution(reference_tokens),
        "actual_tree_token_distribution": _distribution(tree_tokens),
        "packed_tree_token_distribution": _distribution(packed_tokens),
        "native_replay_token_distribution": _distribution(
            run["native_replay_token_id"] for run in runs
        ),
        "same_mask_replay_token_distribution": _distribution(
            run["same_mask_replay_token_id"] for run in runs
        ),
        "math_replay_token_distribution": _distribution(
            run["math_replay_token_id"] for run in runs
        ),
        "actual_reference_stability": reference_stable,
        "actual_tree_stability": tree_stable,
        "packed_tree_stability": packed_stable,
        "stable_actual_reference_tree_divergence": stable_actual_divergence,
        **_support_fields(runs, "native"),
        **_support_fields(runs, "same_mask"),
        **_support_fields(runs, "math"),
        "numerical_control_support_count": numerical_support_count,
        "numerical_control_support_rate": numerical_support_count / len(runs),
        "per_run_classification_distribution": classifications,
        "classification_evidence_complete_count": classification_complete_count,
        "classification_evidence_completeness_rate": (
            classification_complete_count / len(runs)
        ),
        "aggregate_failure_classification": aggregate_classification,
        "aggregate_unresolved_reason": aggregate_reason,
        "aggregate_evidence_complete": aggregate_evidence_complete,
        "runs": sorted(runs, key=lambda run: run["input_directory"]),
    }
    return group


def build_aggregate(output_dirs: list[Path]) -> dict[str, Any]:
    """Load, group, and aggregate independent Phase 2C.4.2 directories."""

    if not output_dirs:
        raise ValueError("at least one Phase 2C.4.2 output directory is required")
    resolved = [path.expanduser().resolve() for path in output_dirs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate input directories are not independent runs")
    runs = [load_run_evidence(path) for path in resolved]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        key = tuple(run[field] for field in GROUPING_KEY_FIELDS)
        grouped[key].append(run)
    groups = [aggregate_group(grouped[key]) for key in sorted(grouped)]
    return {
        "schema_version": "1.0",
        "phase": "2C.4.3",
        "input_run_count": len(runs),
        "group_count": len(groups),
        "grouping_key_fields": GROUPING_KEY_FIELDS,
        "input_directories": [str(path) for path in resolved],
        "single_run_correctness_logic_modified": False,
        "instrumented_dflash_modified": False,
        "formal_timing_contaminated": False,
        "actual_execution_is_semantic_oracle": True,
        "dense_replay_controls_are_attribution_only": True,
        "groups": groups,
    }


def validate_aggregate(payload: dict[str, Any]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)


def write_aggregate(
    output_dirs: list[Path], output_dir: Path
) -> tuple[Path, dict[str, Any]]:
    resolved_output = output_dir.expanduser().resolve()
    for input_dir in output_dirs:
        resolved_input = input_dir.expanduser().resolve()
        if (
            resolved_output == resolved_input
            or resolved_input in resolved_output.parents
        ):
            raise ValueError(
                "aggregate output directory must be outside every capture directory"
            )
    payload = build_aggregate(output_dirs)
    validate_aggregate(payload)
    resolved_output.mkdir(parents=True, exist_ok=True)
    artifact = resolved_output / ARTIFACT_NAME
    artifact.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dirs",
        nargs="+",
        type=Path,
        help="Independent Phase 2C.4.2 output directories",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact, payload = write_aggregate(args.output_dirs, args.output_dir)
    print(f"wrote {artifact}")
    for group in payload["groups"]:
        key = (
            group["sample_id"],
            group["nominal_prompt_length"],
            group["first_actual_semantic_divergence"],
        )
        print(
            f"key={key} runs={group['run_count']} "
            f"verdict={group['aggregate_failure_classification']} "
            f"reason={group['aggregate_unresolved_reason']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
