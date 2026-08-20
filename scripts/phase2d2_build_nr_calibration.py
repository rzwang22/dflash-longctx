#!/usr/bin/env python3
"""Freeze a provenance-bound margin guard from an independent Phase 2D.1B-NR cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase2d2.robustness import (
    CALIBRATION_SCHEMA_VERSION,
    calibration_sha256,
    validate_nr_calibration,
)
from scripts.check_phase2d1b_nr_acceptance import check_output as check_nr_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nr-dir", required=True, type=Path)
    parser.add_argument("--decision-margin-threshold", required=True, type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--target-revision",
        default="b968826d9c46dd6066d109eabc6255188de91218",
    )
    parser.add_argument(
        "--draft-revision",
        default="9b41424b7109f9c5413454f481b09a82b85333f4",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "Torch and Transformers are required on the calibration host"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("NR calibration must be frozen on its matching CUDA host")
    manifest_path = args.nr_dir / "nr-manifest.json"
    sweep_path = args.nr_dir / "margin-trigger-sweep.json"
    for path in (manifest_path, sweep_path):
        if not path.is_file():
            raise ValueError(f"missing independent NR artifact: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    if manifest.get("phase") != "2D.1B-NR" or manifest.get("status") != "ok":
        raise ValueError("NR calibration source must be a completed Phase 2D.1B-NR run")
    nr_checker_result = check_nr_output(args.nr_dir)
    if nr_checker_result.get("status") != "pass":
        raise ValueError("independent Phase 2D.1B-NR acceptance checker did not pass")
    recorded_target_revision = manifest.get("target_model", {}).get("revision")
    if recorded_target_revision not in {None, args.target_revision}:
        raise ValueError("NR target revision differs from calibration revision")
    recorded_draft_revision = (
        manifest.get("materialized_draft", {})
        .get("manifest", {})
        .get("base_model_revision")
    )
    if recorded_draft_revision not in {None, args.draft_revision}:
        raise ValueError("NR draft revision differs from calibration revision")
    threshold = float(args.decision_margin_threshold)
    rows = [
        row
        for row in sweep.get("rows", [])
        if float(row["threshold"]) == threshold
        and row.get("scope") == "overall"
        and row.get("experiment") == "cohort"
    ]
    if not rows:
        raise ValueError(
            "requested threshold is absent from the independent cohort sweep"
        )
    required_sweep_fields = {
        "first_actual_divergence_round_count",
        "divergence_rounds_caught_by_trigger",
        "divergence_recall",
    }
    if any(required_sweep_fields - set(row) for row in rows):
        raise ValueError("NR sweep lacks fields required to establish false negatives")
    divergence_count = sum(
        int(row["first_actual_divergence_round_count"]) for row in rows
    )
    false_negative_count = sum(
        int(row["first_actual_divergence_round_count"])
        - int(row["divergence_rounds_caught_by_trigger"])
        for row in rows
    )
    if divergence_count < 1:
        raise ValueError("NR calibration has no actual divergence event to calibrate")
    if false_negative_count != 0:
        raise ValueError("requested NR threshold has a cohort false negative")
    index = torch.device(args.device).index or 0
    capability = torch.cuda.get_device_capability(index)
    calibration = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "source_phase": "2D.1B-NR",
        "independent_calibration_cohort": True,
        "decision_margin_threshold": threshold,
        "source_artifacts": {
            "nr_directory": str(args.nr_dir.expanduser().resolve()),
            "nr_manifest_sha256": _sha256(manifest_path),
            "margin_trigger_sweep_sha256": _sha256(sweep_path),
            "qualifying_rows": rows,
            "nr_acceptance_checker_result": nr_checker_result,
            "observed_actual_divergence_round_count": divergence_count,
            "observed_false_negative_count": false_negative_count,
        },
        "provenance": {
            "gpu_name": torch.cuda.get_device_name(index),
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "dtype": args.dtype,
            "target_model_revision": args.target_revision,
            "draft_model_revision": args.draft_revision,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "target_attention_implementation": "sdpa",
        },
        "evaluation_outputs_used_to_choose_threshold": False,
        "formal_error_bound_claimed": False,
        "empirically_validated_guard_only": True,
    }
    calibration["calibration_sha256"] = calibration_sha256(calibration)
    validate_nr_calibration(calibration, calibration["provenance"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "nr-calibration.json"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(f"NR calibration: {output}")
    print(f"SHA256: {calibration['calibration_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
