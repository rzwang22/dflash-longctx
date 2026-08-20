#!/usr/bin/env python3
"""Validate Phase 2D.1B-NR artifacts and diagnostic-only invariants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsonschema import Draft202012Validator

from dflash.phase2d1bnr.aggregation import (
    aggregate_cohort_cases,
    aggregate_exact_reproducibility,
    build_margin_trigger_sweep,
)
from dflash.phase2d1bnr.decision import build_research_summary, choose_nr_decision
from dflash.phase2d1bnr.parse import (
    CAUSALLY_SKIPPABLE_DOWNSTREAM_GOLD_CHECKS,
    CONFIRMED_DIVERGENCE_SKIP_REASONS,
    REQUIRED_DIVERGENCE_GOLD_LOGIT_MODES,
    UNRESOLVED_SEMANTIC_ATTRIBUTIONS,
    read_json,
    read_jsonl,
)
from experiments.phase2d1b_nr_numerical_robustness import ARTIFACTS


def _schema(value: Any, name: str) -> None:
    schema = read_json(REPO_ROOT / "docs" / "schemas" / name)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: list(error.path),
    )
    if errors:
        first = errors[0]
        location = "/".join(str(item) for item in first.absolute_path) or "<root>"
        raise ValueError(f"{name} validation failed at {location}: {first.message}")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            parsed = {}
            for key, value in row.items():
                if value == "":
                    parsed[key] = None
                    continue
                try:
                    parsed[key] = json.loads(value)
                except json.JSONDecodeError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _token_hash(values: list[int]) -> str:
    payload = json.dumps(values, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical(rows: list[dict[str, Any]]) -> str:
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def _validate_same_case_controls(
    observations: list[dict[str, Any]], manifest: dict[str, Any]
) -> None:
    expected = []
    modes = manifest["configuration"]["modes"]
    budgets = manifest["configuration"]["node_budgets"]
    if "top1_packed_tree" in modes:
        expected.append(("top1_packed_tree", 16))
    if "global_fixed" in modes:
        expected.extend(("global_fixed", int(value)) for value in budgets)
    grouped: dict[tuple[Any, ...], set[tuple[str, int]]] = {}
    for row in observations:
        key = (
            row["experiment"],
            row["sample_id"],
            row["source_sample_id"],
            row["nominal_prompt_length"],
            row["repeat_index"],
            row["max_new_tokens"],
        )
        grouped.setdefault(key, set()).add(
            (row["selector_mode"], int(row["node_budget"]))
        )
    for key, actual in grouped.items():
        expected_for_experiment = set(expected)
        if key[0] == "exact_case":
            expected_for_experiment.discard(("global_fixed", 32))
        if actual != expected_for_experiment:
            raise ValueError(
                f"case does not contain matched configuration controls: {key} {actual}"
            )


def _validate_observations(
    root: Path, manifest: dict[str, Any], observations: list[dict[str, Any]]
) -> None:
    requested = int(manifest["count_semantics"]["requested_subrun_count"])
    if len(observations) != requested:
        raise ValueError(
            f"requested/accounted subrun mismatch: {requested} != {len(observations)}"
        )
    identities = [row["independent_repeat_observation_id"] for row in observations]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate independent subrun identity")
    if len(manifest["subruns"]) != len(observations):
        raise ValueError("manifest/subrun-index count mismatch")
    _validate_same_case_controls(observations, manifest)
    frozen_hash = manifest["frozen_selector"]["config_sha256"]
    for row in observations:
        _schema(row, "phase2d1bnr-subrun.schema.json")
        raw = Path(row["raw_subrun_dir"])
        if not raw.is_dir() or root not in raw.parents:
            raise ValueError(f"raw subrun directory is absent/outside root: {raw}")
        raw_manifest = read_json(raw / "manifest.json")
        configured_hash = raw_manifest.get("configuration", {}).get(
            "frozen_selector_config_hash"
        )
        if configured_hash != frozen_hash:
            raise ValueError("raw subrun did not use the frozen selector hash")
        if not row["selector_target_blind"] or row["target_forward_count_added"] != 0:
            raise ValueError("selector target-blind/forward-count invariant failed")
        if row.get("target_attention_calls_pytorch_sdpa") is not True:
            raise ValueError("target attention was not configured as PyTorch SDPA")
        if row.get("draft_attention_calls_pytorch_sdpa") is not True:
            raise ValueError("draft attention was not configured as PyTorch SDPA")
        if row.get("source_provenance_validation_passed") is not True:
            raise ValueError("subrun loaded-source provenance did not validate")
        if row.get("source_provenance_status") not in {"ok", "failed"}:
            raise ValueError("subrun source-provenance wrapper status is invalid")
        if row.get("source_provenance_status") == "failed" and not row.get(
            "source_provenance_wrapper_failure_is_expected_correctness"
        ):
            raise ValueError("source-provenance failure was not a correctness exit")
        if any(
            int(value) > int(row["node_budget"]) for value in row["actual_node_counts"]
        ):
            raise ValueError("actual constructed nodes exceed requested budget")
        if row["selector_mode"] == "top1_packed_tree" and any(
            int(value) != 16 for value in row["actual_node_counts"]
        ):
            raise ValueError("top1 packed-tree did not construct exactly 16 nodes")
        output_equal = row["actual_reference_tree_output_ids_equal"]
        outcome = row["outcome_category"]
        if not row.get("actual_output_relation_reliable"):
            raise ValueError("subrun lacks a reliable actual output relation")
        if output_equal is not True and outcome in {
            "pass_actual_exact",
            "diagnostic_only_numerical_variation",
        }:
            raise ValueError("actual output divergence was relabeled as output-exact")
        if (
            outcome == "actual_numerical_decision_divergence"
            and not row["structural_correctness_passed"]
        ):
            raise ValueError("structural failure was relabeled as numerical")
        if outcome == "actual_numerical_decision_divergence" and not row.get(
            "gold_evidence_complete"
        ):
            raise ValueError(
                "numerical divergence lacks complete classification evidence"
            )
        if outcome == "actual_unresolved_semantic_divergence":
            if output_equal is not False:
                raise ValueError("unresolved actual divergence lacks output mismatch")
            if row.get("actual_divergence_evidence_sufficient") is not True:
                raise ValueError("unresolved actual divergence lacks token evidence")
            if row.get("structural_correctness_passed") is not True:
                raise ValueError("structural failure was relabeled as unresolved")
            reason = row.get("unresolved_attribution_reason")
            if reason not in UNRESOLVED_SEMANTIC_ATTRIBUTIONS:
                raise ValueError("unresolved actual divergence has unknown attribution")
            if row.get("divergence_failure_class") != reason:
                raise ValueError(
                    "unresolved attribution did not preserve failure class"
                )
            if row.get("genuinely_incomplete_correctness_evidence") is not False:
                raise ValueError(
                    "unresolved actual divergence was also marked incomplete"
                )
        if outcome == "diagnostic_only_numerical_variation":
            if row.get("replay_only_variation_observed") is not True:
                raise ValueError("diagnostic-only variation lacks replay evidence")
            if row.get("gold_or_replay_variation_observed") is not True:
                raise ValueError("diagnostic-only variation lacks Gold/replay evidence")
            if row.get("unclassified_gold_or_replay_failure_observed") is not False:
                raise ValueError("diagnostic-only variation has unknown attribution")
            if row.get("replay_structural_evidence_passed") is not True:
                raise ValueError(
                    "diagnostic-only variation lacks passing replay structure"
                )
            if row.get("diagnostic_numerical_variation_observed") is not True:
                raise ValueError("diagnostic-only variation lacks numerical evidence")
            if row.get("known_numerical_variation_evidence") is not True:
                raise ValueError(
                    "diagnostic-only variation lacks recognized numerical evidence"
                )
            if row.get("diagnostic_classification_evidence_sufficient") is not True:
                raise ValueError(
                    "diagnostic-only variation lacks sufficient classification evidence"
                )
            if row.get("structural_correctness_passed") is not True:
                raise ValueError("diagnostic-only variation has a structural failure")
            if row.get("genuinely_incomplete_correctness_evidence") is not False:
                raise ValueError(
                    "diagnostic-only variation was also marked genuinely incomplete"
                )
        if row.get("downstream_gold_checks_skipped_after_confirmed_divergence"):
            if outcome != "actual_numerical_decision_divergence":
                raise ValueError("causal downstream Gold skip lacks actual divergence")
            if row.get("genuinely_incomplete_correctness_evidence"):
                raise ValueError("causal downstream skip was also marked incomplete")
            skipped_checks = set(row.get("downstream_gold_skipped_checks", []))
            skip_reasons = set(row.get("downstream_gold_skip_reasons", []))
            if not skipped_checks or not skipped_checks <= (
                CAUSALLY_SKIPPABLE_DOWNSTREAM_GOLD_CHECKS
            ):
                raise ValueError("causal downstream skip contains an unrelated check")
            if (
                not skip_reasons
                or not skip_reasons <= CONFIRMED_DIVERGENCE_SKIP_REASONS
            ):
                raise ValueError("causal downstream skip has an unknown reason")
            if not REQUIRED_DIVERGENCE_GOLD_LOGIT_MODES <= set(
                row.get("first_divergence_gold_comparison_modes_observed", [])
            ):
                raise ValueError("first divergence lacks required Gold-logit modes")
            if not row.get("first_divergence_replay_decision_mismatch_captured"):
                raise ValueError("first divergence lacks replay decision evidence")
            if not row.get("structural_traversal_complete_through_first_divergence"):
                raise ValueError("first divergence lacks complete traversal evidence")
        if row.get("downstream_gold_checks_skipped_after_confirmed_replay_variation"):
            if outcome != "diagnostic_only_numerical_variation":
                raise ValueError("causal replay Gold skip lacks diagnostic variation")
            if row.get("genuinely_incomplete_correctness_evidence"):
                raise ValueError("causal replay Gold skip was also marked incomplete")
            skipped_checks = set(row.get("downstream_gold_skipped_checks", []))
            skip_reasons = set(row.get("downstream_gold_skip_reasons", []))
            if not skipped_checks or not skipped_checks <= (
                CAUSALLY_SKIPPABLE_DOWNSTREAM_GOLD_CHECKS
            ):
                raise ValueError("causal replay Gold skip contains an unrelated check")
            if (
                not skip_reasons
                or not skip_reasons <= CONFIRMED_DIVERGENCE_SKIP_REASONS
            ):
                raise ValueError("causal replay Gold skip has an unknown reason")
            if not REQUIRED_DIVERGENCE_GOLD_LOGIT_MODES <= set(
                row.get("first_replay_variation_gold_comparison_modes_observed", [])
            ):
                raise ValueError("first replay variation lacks Gold-logit modes")
            if not row.get("first_replay_variation_evidence_captured"):
                raise ValueError("first replay variation lacks replay evidence")
            if not row.get(
                "structural_traversal_complete_through_first_replay_variation"
            ):
                raise ValueError("first replay variation lacks traversal evidence")
        reference_stable = bool(
            row.get("reference_repeat_stable") is True
            and row.get("reference_profile_stable") is True
        )
        if not reference_stable and outcome != "reference_instability":
            raise ValueError("reference instability was not classified separately")
        if outcome in {
            "incomplete_correctness_evidence",
            "infrastructure_failure",
        }:
            raise ValueError(f"NR study contains invalidating outcome: {outcome}")


def _validate_divergences(
    observations: list[dict[str, Any]], events: list[dict[str, Any]]
) -> None:
    divergent = [
        row
        for row in observations
        if row["actual_reference_tree_output_ids_equal"] is False
    ]
    if len(events) != len(divergent):
        raise ValueError("actual divergence/event count mismatch")
    by_id = {row["independent_repeat_observation_id"]: row for row in divergent}
    for event in events:
        _schema(event, "phase2d1bnr-divergence-event.schema.json")
        source = by_id.get(event["independent_repeat_observation_id"])
        if source is None:
            raise ValueError("divergence event has no source observation")
        for kind in ("reference", "tree"):
            values = event[f"actual_{kind}_output_ids"]
            if _token_hash(values) != event[f"actual_{kind}_output_sha256"]:
                raise ValueError(f"{kind} output token IDs/hash mismatch")
        if event["actual_reference_output_ids"] == event["actual_tree_output_ids"]:
            raise ValueError("divergence event preserves equal output IDs")


def _validate_margins(
    observations: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
    sweep_artifact: dict[str, Any],
) -> None:
    by_id = {row["independent_repeat_observation_id"]: row for row in observations}
    for row in rounds:
        _schema(row, "phase2d1bnr-round-margin.schema.json")
        source = by_id[row["independent_repeat_observation_id"]]
        first = source.get("first_divergent_tree_round")
        if source["actual_reference_tree_output_ids_equal"] is False and (
            first is None or int(row["round_index"]) > int(first)
        ):
            raise ValueError("post-divergence round entered trigger characterization")
    recomputed = build_margin_trigger_sweep(
        rounds, thresholds=sweep_artifact["thresholds"]
    )
    if _canonical(recomputed) != _canonical(sweep_artifact["rows"]):
        raise ValueError("margin-trigger sweep is not exactly recomputable")


def check_output(output_dir: Path) -> dict[str, Any]:
    root = output_dir.expanduser().resolve()
    missing = [name for name in ARTIFACTS if not (root / name).is_file()]
    if missing:
        raise ValueError(f"NR output lacks required artifacts: {missing}")
    if not (root / "raw-subruns").is_dir():
        raise ValueError("NR output lacks raw-subruns directory")
    manifest = read_json(root / "nr-manifest.json")
    _schema(manifest, "phase2d1bnr-manifest.schema.json")
    if manifest["status"] != "ok" or manifest["error"] is not None:
        raise ValueError(
            f"NR study is incomplete: status={manifest['status']} error={manifest['error']}"
        )
    observations = read_jsonl(root / "subrun-index.jsonl")
    exact = read_jsonl(root / "exact-case-repeats.jsonl")
    cohort = read_jsonl(root / "cohort-subruns.jsonl")
    expected_exact = [row for row in observations if row["experiment"] == "exact_case"]
    expected_cohort = [row for row in observations if row["experiment"] == "cohort"]
    if _canonical(exact) != _canonical(expected_exact) or _canonical(
        cohort
    ) != _canonical(expected_cohort):
        raise ValueError("exact/cohort partition does not cover subrun index")
    _validate_observations(root, manifest, observations)

    exact_summary = read_json(root / "exact-case-reproducibility-summary.json")
    _schema(exact_summary, "phase2d1bnr-exact-summary.schema.json")
    if _canonical(exact_summary) != _canonical(
        aggregate_exact_reproducibility(observations)
    ):
        raise ValueError("exact-case summary is not recomputable")
    cohort_summary = aggregate_cohort_cases(observations)
    for filename, key in (
        ("cohort-case-summary.csv", "cohort_case_summary"),
        ("context-divergence-summary.csv", "context_divergence_summary"),
        ("policy-divergence-summary.csv", "policy_divergence_summary"),
    ):
        if _canonical(_read_csv(root / filename)) != _canonical(cohort_summary[key]):
            raise ValueError(f"{filename} is not recomputable")

    divergences = read_jsonl(root / "divergence-events.jsonl")
    _validate_divergences(observations, divergences)
    for replay in read_jsonl(root / "replay-semantic-events.jsonl"):
        _schema(replay, "phase2d1bnr-replay-event.schema.json")
    rounds = read_jsonl(root / "round-margin-events.jsonl")
    sweep = read_json(root / "margin-trigger-sweep.json")
    _schema(sweep, "phase2d1bnr-margin-sweep.schema.json")
    _validate_margins(observations, rounds, sweep)
    decision = read_json(root / "nr-decision.json")
    research = read_json(root / "research-question-summary.json")
    _schema(decision, "phase2d1bnr-decision.schema.json")
    _schema(research, "phase2d1bnr-research-summary.schema.json")
    expected_decision = choose_nr_decision(
        observations,
        cohort_summary["policy_divergence_summary"],
        sweep["rows"],
        rules=manifest["decision_rules"],
    )
    if _canonical(decision) != _canonical(expected_decision):
        raise ValueError("NR decision is not exactly recomputable")
    expected_research = build_research_summary(
        exact_summary,
        cohort_summary,
        sweep["rows"],
        expected_decision,
    )
    if _canonical(research) != _canonical(expected_research):
        raise ValueError("research-question summary is not exactly recomputable")
    if not research["no_performance_claims"] or not manifest["no_performance_claims"]:
        raise ValueError("NR artifacts emitted a performance-claim contract")
    return {
        "status": "pass",
        "subrun_count": len(observations),
        "physical_case_count": len({row["physical_case_id"] for row in cohort}),
        "configuration_case_count": len(
            {row["configuration_case_id"] for row in cohort}
        ),
        "actual_divergence_event_count": len(divergences),
        "decision": decision["decision"],
        "materialization_required": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = check_output(args.output_dir)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
