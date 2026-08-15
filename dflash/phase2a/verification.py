"""Pure-Python aggregation for target verification module profiles."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from dflash.phase1b.metrics import distribution

VERIFICATION_FIELDS = (
    "verification_cuda_ms",
    "self_attention_module_cuda_ms",
    "mlp_module_cuda_ms",
    "lm_head_cuda_ms",
    "unattributed_cuda_ms",
    "self_attention_over_verification",
    "mlp_over_verification",
    "lm_head_over_verification",
    "draft_cuda_ms",
    "draft_over_round",
    "verification_over_round",
)


def aggregate_verification_samples(
    samples: Sequence[dict[str, Any]], module_events: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    by_context: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_context[(sample["task"], int(sample["nominal_prompt_length"]))].append(
            sample
        )
    by_layer: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    for event in module_events:
        if event["stage"] != "verification" or event["layer_index"] is None:
            continue
        by_layer[
            (
                event["task"],
                int(event["nominal_prompt_length"]),
                int(event["layer_index"]),
                event["module_kind"],
            )
        ].append(event["cuda_elapsed_ms"])

    context_output = {}
    for (task, context_length), group in sorted(by_context.items()):
        context_output[f"{task}:{context_length}"] = {
            "task": task,
            "context_length": context_length,
            "sample_count": len({sample["sample_id"] for sample in group}),
            "metrics": {
                field: distribution(sample.get(field) for sample in group)
                for field in VERIFICATION_FIELDS
            },
            "partition_reliable_sample_count": sum(
                sample["partition_reliable"] for sample in group
            ),
            "partition_unreliable_sample_count": sum(
                not sample["partition_reliable"] for sample in group
            ),
        }

    layer_output = {
        f"{task}:{context_length}:{layer_index}:{module_kind}": {
            "task": task,
            "context_length": context_length,
            "layer_index": layer_index,
            "module_kind": module_kind,
            "cuda_elapsed_ms": distribution(values),
        }
        for (
            task,
            context_length,
            layer_index,
            module_kind,
        ), values in sorted(by_layer.items())
    }
    return {
        "by_task_and_context_length": context_output,
        "by_layer": layer_output,
        "measurement_semantics": {
            "formal_latency_source": "instrumented_default_control",
            "component_timing_source": "verification_profile_path",
            "self_attention_includes": "QKV projections, SDPA path, and output projection",
            "self_attention_is_pure_sdpa": False,
            "prefill_excluded_from_primary_statistics": True,
            "unattributed_reported_only_when_partition_reliable": True,
        },
    }
