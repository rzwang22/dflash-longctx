"""Untrained draft-confidence diagnostics for position-level Top-1 errors."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any


def draft_confidence_features(position: dict[str, Any]) -> dict[str, float]:
    """Return target-blind features computed only from captured draft Top-K."""

    log_probabilities = [
        float(candidate["log_probability"]) for candidate in position["candidates"]
    ]
    probabilities = [math.exp(value) for value in log_probabilities]
    mass = sum(probabilities)
    normalized = [value / mass for value in probabilities]
    entropy = -sum(value * math.log(value) for value in normalized if value > 0)
    margin = log_probabilities[0] - log_probabilities[1]
    return {
        "top1_top2_log_probability_margin": margin,
        "negative_margin_risk_score": -margin,
        "top_k_probability_mass": mass,
        "top_k_renormalized_truncated_entropy": entropy,
        "top1_probability": probabilities[0],
        "negative_top1_probability_risk_score": -probabilities[0],
    }


def _binary_metrics(labels: list[int], scores: list[float]) -> dict[str, Any]:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not labels or positives == 0 or negatives == 0:
        return {
            "auroc": None,
            "auprc": None,
            "unavailable_reason": (
                "no_observations"
                if not labels
                else "single_class_all_negative"
                if positives == 0
                else "single_class_all_positive"
            ),
        }

    grouped: dict[float, list[int]] = defaultdict(list)
    for label, score in zip(labels, scores):
        grouped[float(score)].append(int(label))
    wins = 0.0
    seen_negatives = 0
    for score in sorted(grouped):
        group = grouped[score]
        group_pos = sum(group)
        group_neg = len(group) - group_pos
        wins += group_pos * seen_negatives + 0.5 * group_pos * group_neg
        seen_negatives += group_neg
    auroc = wins / (positives * negatives)

    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    auprc = 0.0
    for score in sorted(grouped, reverse=True):
        group = grouped[score]
        true_positive += sum(group)
        false_positive += len(group) - sum(group)
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        auprc += (recall - previous_recall) * precision
        previous_recall = recall
    return {"auroc": auroc, "auprc": auprc, "unavailable_reason": None}


def _calibration(labels: list[int], scores: list[float]) -> list[dict[str, Any]]:
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    if not ordered:
        return []
    rows = []
    for decile in range(10):
        start = len(ordered) * decile // 10
        end = len(ordered) * (decile + 1) // 10
        bucket = ordered[start:end]
        if not bucket:
            continue
        rows.append(
            {
                "risk_decile": decile + 1,
                "count": len(bucket),
                "mean_risk_score": mean(value for value, _ in bucket),
                "observed_top1_error_rate": mean(label for _, label in bucket),
            }
        )
    return rows


def confidence_diagnostics(position_rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictors = {
        "negative_margin": "negative_margin_risk_score",
        "top_k_truncated_entropy": "top_k_renormalized_truncated_entropy",
        "lower_top1_probability": "negative_top1_probability_risk_score",
    }
    groupers: list[tuple[str, Any]] = [
        ("micro", lambda row: "all"),
        ("task", lambda row: row["task"]),
        ("context_length", lambda row: str(row["nominal_prompt_length"])),
        (
            "task_context",
            lambda row: f"{row['task']}:{row['nominal_prompt_length']}",
        ),
    ]
    result_rows = []
    for scope, key_fn in groupers:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in position_rows:
            groups[key_fn(row)].append(row)
        for group, rows in sorted(groups.items()):
            labels = [int(row["top1_incorrect"]) for row in rows]
            for predictor, field in predictors.items():
                scores = [float(row[field]) for row in rows]
                result_rows.append(
                    {
                        "scope": scope,
                        "group": group,
                        "predictor": predictor,
                        "observation_count": len(rows),
                        "positive_error_count": sum(labels),
                        **_binary_metrics(labels, scores),
                        "calibration_by_risk_decile": _calibration(labels, scores),
                    }
                )

    macro_groupers: list[tuple[str, Any]] = [
        ("sample_macro", lambda row: "all"),
        ("sample_macro_by_task", lambda row: row["task"]),
        (
            "sample_macro_by_context",
            lambda row: str(row["nominal_prompt_length"]),
        ),
        (
            "sample_macro_by_task_context",
            lambda row: f"{row['task']}:{row['nominal_prompt_length']}",
        ),
    ]
    for scope, group_fn in macro_groupers:
        scoped: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in position_rows:
            scoped[group_fn(row)][(row["task"], row["sample_id"])].append(row)
        for group, sample_groups in sorted(scoped.items()):
            for predictor, field in predictors.items():
                sample_metrics = [
                    _binary_metrics(
                        [int(row["top1_incorrect"]) for row in rows],
                        [float(row[field]) for row in rows],
                    )
                    for rows in sample_groups.values()
                ]
                evaluable = [
                    metric for metric in sample_metrics if metric["auroc"] is not None
                ]
                result_rows.append(
                    {
                        "scope": scope,
                        "group": group,
                        "predictor": predictor,
                        "sample_count": len(sample_groups),
                        "evaluable_sample_count": len(evaluable),
                        "auroc": mean(metric["auroc"] for metric in evaluable)
                        if evaluable
                        else None,
                        "auprc": mean(metric["auprc"] for metric in evaluable)
                        if evaluable
                        else None,
                        "unavailable_reason": None
                        if evaluable
                        else "no_sample_contains_both_classes",
                        "calibration_by_risk_decile": [],
                    }
                )
    return {
        "schema_version": "1.0",
        "label": "captured_target_token_differs_from_draft_top1",
        "predictors_are_target_blind": True,
        "target_score_gap_role": "post_hoc_explanation_only_not_a_predictor",
        "top_k_entropy_semantics": "entropy after renormalizing captured full-softmax Top-K probability mass; not full-vocabulary entropy",
        "training_performed": False,
        "threshold_generalization_claimed": False,
        "confidence_intervals": None,
        "confidence_interval_reason": "not computed; positions and rounds are not treated as independent samples",
        "results": result_rows,
    }
