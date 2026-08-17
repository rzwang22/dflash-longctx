"""Target-blind signal table and single-feature diagnostics for Phase 2D.0."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

LABEL_FIELDS = (
    "candidate_availability_failure",
    "all_tested_static_path_failure",
    "dependency_realization_opportunity",
    "first_critical_depth_position",
)
TARGET_ONLY_FIELDS = {
    "target_token_rank",
    "target_token_in_top_m",
    "accepted_draft_depth",
    "gap_class",
    "first_critical_depth",
    *LABEL_FIELDS,
}
FEATURE_RISK_DIRECTIONS = {
    "position_index": 1.0,
    "top1_log_probability": -1.0,
    "top1_top2_log_probability_margin": -1.0,
    "top_m_probability_mass": -1.0,
    "top_m_renormalized_entropy": 1.0,
    "adjacent_position_confidence_change": -1.0,
    "static_beam_score": -1.0,
    "surviving_prefix_count": 1.0,
    "tree_depth": 1.0,
    "context_length": 1.0,
    "top_m": 1.0,
    "node_budget": 1.0,
}


def build_signal_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        for features in event["deployable_features_by_position"]:
            overlap = set(features) & TARGET_ONLY_FIELDS
            if overlap:
                raise ValueError(
                    "target-dependent fields entered deployable features: "
                    f"{sorted(overlap)}"
                )
            if "top_m_renormalized_entropy" not in features:
                raise ValueError(
                    "signal table must name Top-M renormalized entropy exactly"
                )
            depth = int(features["position_index"])
            observed = depth <= int(event["observed_target_horizon"])
            eligible = bool(event["classification_eligible"] and observed)
            membership = (
                bool(event["target_membership_top_m"][depth - 1]) if observed else None
            )
            prefix_available = bool(event["candidate_availability_prefix"] >= depth)
            labels = {
                "candidate_availability_failure": (
                    not membership if eligible else None
                ),
                "all_tested_static_path_failure": (
                    prefix_available and event["best_static_coverage_depth"] < depth
                    if eligible
                    else None
                ),
                "dependency_realization_opportunity": (
                    event["gap_class"] == "dependency_realization_opportunity"
                    and event["critical_depth"] == depth
                    if eligible
                    else None
                ),
                "first_critical_depth_position": (
                    event["critical_depth"] == depth if eligible else None
                ),
                "first_critical_depth": event["critical_depth"],
                "target_token_rank": (
                    event["target_candidate_ranks_top16"][depth - 1]
                    if observed
                    else None
                ),
                "target_token_in_top_m": membership,
                "accepted_draft_depth": event["baseline_accepted_draft_depth_a0"],
                "gap_class": event["gap_class"],
            }
            rows.append(
                {
                    "schema_version": "1.0",
                    **{
                        key: event.get(key)
                        for key in (
                            "task",
                            "sample_id",
                            "source_sample_id",
                            "source_text_sha256",
                            "nominal_prompt_length",
                            "actual_prompt_length",
                            "generation_repeat",
                            "round_index",
                            "top_m",
                            "node_budget",
                        )
                    },
                    "position_index": depth,
                    "deployable_features": dict(features),
                    "labels": labels,
                    "label_eligible": eligible,
                    "label_unavailable_reason": (
                        None
                        if eligible
                        else "terminal_or_right_censored"
                        if not event["classification_eligible"]
                        else "position_beyond_observed_horizon"
                    ),
                    "feature_contract": "draft_side_target_blind_at_deployment",
                    "label_contract": "target_dependent_offline_diagnostic_only",
                    "top_m_entropy_semantics": (
                        "entropy after renormalizing captured Top-M probability "
                        "mass; not full-vocabulary entropy"
                    ),
                }
            )
    return rows


def _binary_metrics(labels: list[int], scores: list[float]) -> dict[str, Any]:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not labels or positives == 0 or negatives == 0:
        reason = (
            "no_observations"
            if not labels
            else "single_class_all_negative"
            if positives == 0
            else "single_class_all_positive"
        )
        return {"auroc": None, "auprc": None, "unavailable_reason": reason}

    grouped: dict[float, list[int]] = defaultdict(list)
    for label, score in zip(labels, scores):
        grouped[float(score)].append(int(label))
    wins = 0.0
    seen_negatives = 0
    for score in sorted(grouped):
        current = grouped[score]
        current_positive = sum(current)
        current_negative = len(current) - current_positive
        wins += (
            current_positive * seen_negatives
            + 0.5 * current_positive * current_negative
        )
        seen_negatives += current_negative
    auroc = wins / (positives * negatives)

    true_positive = false_positive = 0
    previous_recall = 0.0
    auprc = 0.0
    for score in sorted(grouped, reverse=True):
        current = grouped[score]
        true_positive += sum(current)
        false_positive += len(current) - sum(current)
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        auprc += (recall - previous_recall) * precision
        previous_recall = recall
    return {"auroc": auroc, "auprc": auprc, "unavailable_reason": None}


def _calibration_bins(
    labels: list[int], scores: list[float], *, bucket_count: int = 5
) -> list[dict[str, Any]]:
    ordered = sorted(zip(scores, labels), key=lambda value: value[0])
    rows = []
    for bucket_index in range(bucket_count):
        start = len(ordered) * bucket_index // bucket_count
        end = len(ordered) * (bucket_index + 1) // bucket_count
        bucket = ordered[start:end]
        if not bucket:
            continue
        rows.append(
            {
                "risk_bucket": bucket_index + 1,
                "count": len(bucket),
                "mean_risk_score": mean(score for score, _ in bucket),
                "positive_count": sum(label for _, label in bucket),
                "precision": mean(label for _, label in bucket),
            }
        )
    return rows


def _evaluate_group(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    group: str,
) -> list[dict[str, Any]]:
    results = []
    sample_count = len({(row["task"], row["sample_id"]) for row in rows})
    for label_field in LABEL_FIELDS:
        for feature, direction in FEATURE_RISK_DIRECTIONS.items():
            pairs = [
                (
                    int(row["labels"][label_field]),
                    float(row["deployable_features"][feature]) * direction,
                )
                for row in rows
                if row["labels"].get(label_field) is not None
                and row["deployable_features"].get(feature) is not None
            ]
            labels = [label for label, _ in pairs]
            scores = [score for _, score in pairs]
            metrics = _binary_metrics(labels, scores)
            positives = sum(labels)
            negatives = len(labels) - positives
            unstable_reasons = []
            if len(labels) < 30:
                unstable_reasons.append("fewer_than_30_observations")
            if min(positives, negatives) < 5:
                unstable_reasons.append("fewer_than_5_observations_in_one_class")
            results.append(
                {
                    "schema_version": "1.0",
                    "scope": scope,
                    "group": group,
                    "label": label_field,
                    "feature": feature,
                    "single_feature_only": True,
                    "risk_score_direction": direction,
                    "observation_count": len(labels),
                    "sample_count": sample_count,
                    "positive_count": positives,
                    "negative_count": negatives,
                    "positive_rate": positives / len(labels) if labels else None,
                    **metrics,
                    "calibration_or_binned_precision": _calibration_bins(
                        labels, scores
                    ),
                    "result_unstable": bool(unstable_reasons),
                    "instability_reasons": unstable_reasons,
                    "training_performed": False,
                    "refinement_gain_predicted": False,
                }
            )
    return results


def evaluate_signal_hypotheses(
    signal_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Report untrained single-feature diagnostics by context and task."""

    results = _evaluate_group(signal_rows, scope="micro", group="all")
    groupers = (
        (
            "context_length",
            lambda row: str(row["nominal_prompt_length"]),
        ),
        ("task", lambda row: str(row["task"])),
        (
            "task_context",
            lambda row: f"{row['task']}:{row['nominal_prompt_length']}",
        ),
    )
    for scope, key_fn in groupers:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in signal_rows:
            grouped[key_fn(row)].append(row)
        for group, rows in sorted(grouped.items()):
            results.extend(_evaluate_group(rows, scope=scope, group=group))
    return results
