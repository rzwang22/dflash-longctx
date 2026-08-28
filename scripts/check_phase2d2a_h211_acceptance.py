#!/usr/bin/env python3
"""Validate the no-model Phase 2D.2A-H2.1.1 final-route audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_phase2d2a_h211_final_route import (
    AUDIT_CSV_NAME,
    AUDIT_NAME,
    MANIFEST_NAME,
    _sha256_file,
    _validate_schema,
    audit_csv_rows,
    build_h211_artifacts,
)

CREATION_PROVENANCE_FIELDS = ("repository_commit", "branch")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _expected_manifest_with_persisted_creation_provenance(
    expected: dict, observed: dict
) -> dict:
    normalized = dict(expected)
    for field in CREATION_PROVENANCE_FIELDS:
        value = observed.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"persisted H2.1.1 creation provenance is invalid: {field}"
            )
        normalized[field] = value
    return normalized


def check_h211_output(
    output_dir: Path,
    freeze_dir: Path,
    freeze_content_sha256: str,
    h21_output_root: Path,
) -> dict:
    output_dir = output_dir.expanduser().resolve()
    for name in (AUDIT_NAME, AUDIT_CSV_NAME, MANIFEST_NAME):
        if not (output_dir / name).is_file():
            raise ValueError(f"missing H2.1.1 artifact: {name}")
    expected = build_h211_artifacts(
        freeze_dir=freeze_dir,
        freeze_content_sha256=freeze_content_sha256,
        h21_output_root=h21_output_root,
    )
    observed_audit = json.loads((output_dir / AUDIT_NAME).read_text(encoding="utf-8"))
    observed_manifest = json.loads(
        (output_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    _validate_schema(observed_audit)
    _validate_schema(observed_manifest)
    if _canonical(observed_audit) != _canonical(expected["audit"]):
        raise ValueError(
            "persisted H2.1.1 route audit differs from accepted H2.1 evidence"
        )
    expected_manifest = _expected_manifest_with_persisted_creation_provenance(
        expected["manifest"], observed_manifest
    )
    if _canonical(observed_manifest) != _canonical(expected_manifest):
        raise ValueError("persisted H2.1.1 manifest differs from immutable inputs")
    with (output_dir / AUDIT_CSV_NAME).open(encoding="utf-8", newline="") as stream:
        observed_csv = list(csv.DictReader(stream))
    expected_csv = [
        {field: "" if value is None else str(value) for field, value in row.items()}
        for row in audit_csv_rows(expected["audit"])
    ]
    if observed_csv != expected_csv:
        raise ValueError("persisted H2.1.1 CSV differs from accepted H2.1 evidence")
    audit = observed_audit
    route = audit["route"]
    if audit["formal_primary_evidence"]["stable_cross_task_same_work_signal_exists"]:
        raise ValueError("H2.1.1 rewrote the negative primary classification")
    if audit["secondary_efficiency_observation"]["primary_criterion_rewritten"]:
        raise ValueError("secondary efficiency rewrote the primary criterion")
    if (
        route["proceed_to_phase2d2b_lightweight_adapter"]
        or route["select_refinement_winner"]
        or route["increase_dependency_model_complexity"]
        or route["enter_phase2d3"]
    ):
        raise ValueError("H2.1.1 authorized a prohibited route")
    if route["h22_can_override_h21_primary"]:
        raise ValueError("H2.1.1 permits H2.2 to override H2.1")
    if not route["h22_secondary_characterization_allowed"]:
        raise ValueError("H2.1.1 did not retain secondary H2.2 characterization")
    if audit["model_forward_executed"] or observed_manifest["model_forward_executed"]:
        raise ValueError("H2.1.1 claims a model forward")
    current_h21_hashes = {
        name: _sha256_file(h21_output_root.expanduser().resolve() / name)
        for name in observed_manifest["h21_artifact_file_sha256"]
    }
    if current_h21_hashes != observed_manifest["h21_artifact_file_sha256"]:
        raise ValueError("accepted H2.1 artifacts changed after route audit")
    return {
        "status": "accepted",
        "phase": "2D.2A-H2.1.1",
        "freeze_content_sha256": freeze_content_sha256,
        "h21_primary_accepted": True,
        "stable_cross_task_same_work_signal_exists": False,
        "proceed_to_phase2d2b_lightweight_adapter": False,
        "select_refinement_winner": False,
        "h22_secondary_characterization_allowed": True,
        "h22_can_override_h21_primary": False,
        "model_forward_executed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", required=True, type=Path)
    parser.add_argument("--freeze-content-sha256", required=True)
    parser.add_argument("--h21-output-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = check_h211_output(
            args.output_dir,
            args.freeze_dir,
            args.freeze_content_sha256,
            args.h21_output_root,
        )
    except Exception as exc:  # noqa: BLE001 - clean acceptance rejection.
        print(f"REJECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
