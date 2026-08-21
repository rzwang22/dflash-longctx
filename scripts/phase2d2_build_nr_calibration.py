#!/usr/bin/env python3
"""Freeze a provenance-bound margin guard from an independent Phase 2D.1B-NR cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase2d2.robustness import (
    CALIBRATION_SCHEMA_VERSION,
    FROZEN_DECISION_MARGIN_THRESHOLD,
    calibration_sha256,
    validate_nr_calibration,
)
from scripts.check_phase2d1b_nr_acceptance import check_output as check_nr_output

EXPECTED_NR_CHECKER_COUNTS = {
    "subrun_count": 45,
    "physical_case_count": 15,
    "configuration_case_count": 45,
    "actual_divergence_event_count": 4,
}
ACTUAL_DIVERGENCE_OUTCOMES = (
    "actual_numerical_decision_divergence",
    "actual_unresolved_semantic_divergence",
)
EXPECTED_FROZEN_DIVERGENCE_OUTCOMES = {
    "actual_numerical_decision_divergence": 1,
    "actual_unresolved_semantic_divergence": 3,
}


def _same_threshold(left: Any, right: float) -> bool:
    try:
        return math.isclose(float(left), right, rel_tol=0.0, abs_tol=1.0e-12)
    except (TypeError, ValueError):
        return False


def _validated_frozen_sweep_evidence(
    *,
    sweep: dict[str, Any],
    observations: list[dict[str, Any]],
    checker_result: dict[str, Any],
    requested_threshold: float,
) -> dict[str, Any]:
    """Validate the frozen independent cohort without selecting a new threshold."""

    if not _same_threshold(requested_threshold, FROZEN_DECISION_MARGIN_THRESHOLD):
        raise ValueError(
            "Phase 2D.2A uses the frozen NR decision margin threshold 0.25"
        )
    if not any(
        _same_threshold(value, FROZEN_DECISION_MARGIN_THRESHOLD)
        for value in sweep.get("thresholds", [])
    ):
        raise ValueError("frozen threshold 0.25 is absent from the NR sweep")
    if checker_result.get("status") != "pass":
        raise ValueError("independent Phase 2D.1B-NR acceptance checker did not pass")
    for field, expected in EXPECTED_NR_CHECKER_COUNTS.items():
        if int(checker_result.get(field, -1)) != expected:
            raise ValueError(
                f"frozen NR checker {field} differs: "
                f"{checker_result.get(field)!r} != {expected}"
            )

    if not observations or any(
        row.get("experiment") != "cohort" for row in observations
    ):
        raise ValueError("NR calibration observations must be an independent cohort")
    outcome_counts = Counter(str(row.get("outcome_category")) for row in observations)
    observed_divergence_counts = {
        outcome: outcome_counts.get(outcome, 0)
        for outcome in ACTUAL_DIVERGENCE_OUTCOMES
    }
    if observed_divergence_counts != EXPECTED_FROZEN_DIVERGENCE_OUTCOMES:
        raise ValueError(
            "frozen NR actual-divergence outcome distribution changed: "
            f"{observed_divergence_counts}"
        )
    actual_divergence_observation_count = sum(observed_divergence_counts.values())
    if actual_divergence_observation_count < 1:
        raise ValueError("NR calibration has no actual divergence event to calibrate")

    overall = [
        row
        for row in sweep.get("rows", [])
        if row.get("experiment") == "cohort" and row.get("scope") == "overall"
    ]
    if not overall:
        raise ValueError("NR sweep has no cohort overall selector/budget strata")
    all_strata = {
        (str(row.get("selector_mode")), int(row.get("node_budget"))) for row in overall
    }
    selected = [
        row
        for row in overall
        if _same_threshold(row.get("threshold"), FROZEN_DECISION_MARGIN_THRESHOLD)
    ]
    selected_strata = {
        (str(row.get("selector_mode")), int(row.get("node_budget"))) for row in selected
    }
    if selected_strata != all_strata or len(selected) != len(all_strata):
        raise ValueError("frozen threshold does not cover every cohort overall stratum")

    required = {
        "first_actual_divergence_round_count",
        "divergence_rounds_caught_by_trigger",
        "divergence_recall",
    }
    if any(required - set(row) for row in selected):
        raise ValueError("NR sweep lacks fields required to establish false negatives")
    divergence_count = sum(
        int(row["first_actual_divergence_round_count"]) for row in selected
    )
    caught_count = sum(
        int(row["divergence_rounds_caught_by_trigger"]) for row in selected
    )
    false_negative_count = divergence_count - caught_count
    if divergence_count != actual_divergence_observation_count:
        raise ValueError(
            "NR sweep divergence count does not include every numerical and "
            "unresolved actual divergence"
        )
    if divergence_count != EXPECTED_NR_CHECKER_COUNTS["actual_divergence_event_count"]:
        raise ValueError("frozen NR threshold must cover exactly four divergences")
    if false_negative_count != 0:
        raise ValueError("requested NR threshold has a cohort false negative")

    zero_false_negative_thresholds = []
    for threshold in sweep.get("thresholds", []):
        rows = [
            row
            for row in overall
            if _same_threshold(row.get("threshold"), float(threshold))
        ]
        strata = {
            (str(row.get("selector_mode")), int(row.get("node_budget"))) for row in rows
        }
        if strata != all_strata or len(rows) != len(all_strata):
            continue
        total = sum(
            int(row.get("first_actual_divergence_round_count", 0)) for row in rows
        )
        caught = sum(
            int(row.get("divergence_rounds_caught_by_trigger", 0)) for row in rows
        )
        if total == actual_divergence_observation_count and caught == total:
            zero_false_negative_thresholds.append(float(threshold))
    if not zero_false_negative_thresholds or not _same_threshold(
        min(zero_false_negative_thresholds), FROZEN_DECISION_MARGIN_THRESHOLD
    ):
        raise ValueError(
            "0.25 is no longer the minimum preregistered zero-FN cohort threshold"
        )

    return {
        "qualifying_rows": selected,
        "cohort_overall_strata": [
            {"selector_mode": selector, "node_budget": budget}
            for selector, budget in sorted(all_strata)
        ],
        "observed_actual_divergence_outcome_counts": observed_divergence_counts,
        "observed_actual_divergence_round_count": divergence_count,
        "observed_divergence_rounds_caught_by_trigger": caught_count,
        "observed_false_negative_count": false_negative_count,
        "minimum_preregistered_zero_false_negative_threshold": min(
            zero_false_negative_thresholds
        ),
    }


def _source_runtime_provenance(
    root: Path,
    manifest: dict[str, Any],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recover one exact runtime identity from the accepted raw NR subruns."""

    records = []
    for observation in observations:
        raw = Path(str(observation["raw_subrun_dir"])).expanduser().resolve()
        if root not in raw.parents:
            raise ValueError(f"NR raw subrun is outside the accepted artifact: {raw}")
        environment = json.loads((raw / "environment.json").read_text(encoding="utf-8"))
        subrun_manifest = json.loads(
            (raw / "manifest.json").read_text(encoding="utf-8")
        )
        device_text = str(manifest.get("environment", {}).get("device", "cuda:0"))
        device_index = int(device_text.split(":", 1)[1]) if ":" in device_text else 0
        devices = environment.get("cuda", {}).get("gpus", [])
        gpu = next(
            (row for row in devices if int(row.get("index", -1)) == device_index),
            None,
        )
        if gpu is None:
            raise ValueError(f"NR environment lacks CUDA device {device_index}")
        records.append(
            {
                "gpu_name": gpu.get("name"),
                "compute_capability": gpu.get("compute_capability"),
                "torch_version": environment.get("packages", {})
                .get("torch", {})
                .get("version"),
                "transformers_version": environment.get("packages", {})
                .get("transformers", {})
                .get("version"),
                "target_attention_implementation": subrun_manifest.get(
                    "attention_backend", {}
                )
                .get("target", {})
                .get("configured_attention_implementation"),
            }
        )
    canonical_records = {
        json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records
    }
    if len(canonical_records) != 1:
        raise ValueError("NR subruns do not share one exact runtime provenance")
    runtime = records[0]
    runtime.update(
        {
            "dtype": manifest.get("configuration", {}).get("dtype"),
            "target_model_revision": manifest.get("target_model", {}).get("revision"),
            "draft_model_revision": manifest.get("materialized_draft", {})
            .get("manifest", {})
            .get("base_model_revision"),
        }
    )
    missing = sorted(field for field, value in runtime.items() if value in {None, ""})
    if missing:
        raise ValueError(f"NR runtime provenance is incomplete: {missing}")
    if runtime["target_attention_implementation"] != "sdpa":
        raise ValueError("NR target attention implementation was not SDPA")
    return runtime


def _current_runtime_provenance(
    args: argparse.Namespace, torch: Any, transformers: Any
) -> dict[str, Any]:
    index = torch.device(args.device).index or 0
    capability = torch.cuda.get_device_capability(index)
    return {
        "gpu_name": torch.cuda.get_device_name(index),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "dtype": args.dtype,
        "target_model_revision": args.target_revision,
        "draft_model_revision": args.draft_revision,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "target_attention_implementation": "sdpa",
    }


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
    threshold = float(args.decision_margin_threshold)
    observations = [
        json.loads(line)
        for line in (args.nr_dir / "subrun-index.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    evidence = _validated_frozen_sweep_evidence(
        sweep=sweep,
        observations=observations,
        checker_result=nr_checker_result,
        requested_threshold=threshold,
    )
    source_provenance = _source_runtime_provenance(
        args.nr_dir.expanduser().resolve(), manifest, observations
    )
    runtime_provenance = _current_runtime_provenance(args, torch, transformers)
    mismatches = {
        field: {"nr_source": source_provenance[field], "runtime": value}
        for field, value in runtime_provenance.items()
        if source_provenance.get(field) != value
    }
    if mismatches:
        raise ValueError(f"NR calibration/runtime provenance mismatch: {mismatches}")
    calibration = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "source_phase": "2D.1B-NR",
        "independent_calibration_cohort": True,
        "decision_margin_threshold": threshold,
        "source_artifacts": {
            "nr_directory": str(args.nr_dir.expanduser().resolve()),
            "nr_manifest_sha256": _sha256(manifest_path),
            "margin_trigger_sweep_sha256": _sha256(sweep_path),
            "nr_acceptance_checker_result": nr_checker_result,
            **evidence,
        },
        "provenance": source_provenance,
        "evaluation_outputs_used_to_choose_threshold": False,
        "formal_error_bound_claimed": False,
        "empirically_validated_guard_only": True,
    }
    calibration["calibration_sha256"] = calibration_sha256(calibration)
    validate_nr_calibration(calibration, runtime_provenance)
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
