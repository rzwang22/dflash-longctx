"""Pure-Python helpers for opt-in DFlash Phase 1A diagnostics.

This file is first-party instrumentation carried alongside the pinned third-party
model code.  It deliberately does not import torch so accounting can be tested on
CPU-only development machines.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any, NamedTuple


class DFlashGenerationResult(NamedTuple):
    """Explicit result used only when ``return_diagnostics=True``."""

    output_ids: Any
    diagnostics: dict[str, Any]


def finalize_generation_result(
    output_ids: Any, diagnostics: dict[str, Any], *, return_diagnostics: bool
) -> Any:
    """Preserve the original output object unless diagnostics were requested."""

    if return_diagnostics:
        return DFlashGenerationResult(output_ids, diagnostics)
    return output_ids


def tensor_shape(value: Any) -> list[int] | None:
    """Return a JSON-compatible tensor shape without moving tensor data."""

    shape = getattr(value, "shape", None)
    return [int(dimension) for dimension in shape] if shape is not None else None


def nvtx_range(torch_module: Any, name: str, enabled: bool):
    """Return an NVTX range context, or a no-op context when profiling is off."""

    if enabled:
        return torch_module.cuda.nvtx.range(name)
    return nullcontext()


class DeferredCudaTimer:
    """Collect CUDA event pairs and resolve them at one measurement boundary."""

    def __init__(self, torch_module: Any, *, enabled: bool, device: Any) -> None:
        self.torch = torch_module
        self.enabled = enabled
        self.device = device
        self._pairs: list[dict[str, Any]] = []

    def start(self) -> dict[str, Any]:
        pair: dict[str, Any] = {
            "cpu_started_at": time.perf_counter(),
            "cpu_elapsed_ms": None,
            "cuda_elapsed_ms": None,
            "start_event": None,
            "end_event": None,
        }
        if self.enabled:
            pair["start_event"] = self.torch.cuda.Event(enable_timing=True)
            pair["start_event"].record()
        self._pairs.append(pair)
        return pair

    def stop(self, pair: dict[str, Any]) -> None:
        if self.enabled:
            pair["end_event"] = self.torch.cuda.Event(enable_timing=True)
            pair["end_event"].record()
        pair["cpu_elapsed_ms"] = (time.perf_counter() - pair["cpu_started_at"]) * 1000.0

    def resolve(self) -> None:
        if not self.enabled:
            return
        self.torch.cuda.synchronize(self.device)
        for pair in self._pairs:
            if pair["start_event"] is not None and pair["end_event"] is not None:
                pair["cuda_elapsed_ms"] = pair["start_event"].elapsed_time(
                    pair["end_event"]
                )

    @staticmethod
    def values(pair: dict[str, Any]) -> dict[str, float | None]:
        return {
            "cpu_elapsed_ms": pair["cpu_elapsed_ms"],
            "cuda_elapsed_ms": pair["cuda_elapsed_ms"],
        }


def gpu_memory_snapshot(torch_module: Any, device: Any) -> dict[str, int]:
    """Read allocator counters without introducing a device synchronization."""

    return {
        "allocated_bytes": int(torch_module.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch_module.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved(device)),
    }


def build_round_accounting(
    *,
    round_index: int,
    context_length_before: int,
    prompt_length: int,
    max_new_tokens: int,
    block_size: int,
    accepted_draft_tokens: int,
    returned_tokens_before: int,
    eos_generated_count: int | None,
) -> dict[str, Any]:
    """Describe one round using the exact accounting in pinned ``dflash.py``.

    The block's first token is a target-produced seed carried into the round.
    ``accepted_draft_tokens`` counts only matching positions 1..B-1.  The target
    always writes one new correction token, or a bonus token when all drafts match;
    that token is the seed for the next round and is not included in this round's
    target-cache crop.
    """

    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    proposed = block_size - 1
    if not 0 <= accepted_draft_tokens <= proposed:
        raise ValueError(
            "accepted_draft_tokens must be between zero and block_size - 1"
        )
    if context_length_before < prompt_length:
        raise ValueError("context_length_before cannot precede the prompt")

    rejected = proposed - accepted_draft_tokens
    committed = accepted_draft_tokens + 1
    context_length_after = context_length_before + committed
    raw_generated_before = context_length_before - prompt_length + 1
    raw_generated_after = context_length_after - prompt_length + 1
    capped_generated_after = min(raw_generated_after, max_new_tokens)
    returned_tokens_after = (
        min(capped_generated_after, eos_generated_count)
        if eos_generated_count is not None
        else capped_generated_after
    )
    returned_token_delta = max(0, returned_tokens_after - returned_tokens_before)
    max_length = prompt_length + max_new_tokens

    if eos_generated_count is not None:
        stop_reason = "eos"
    elif context_length_after >= max_length:
        stop_reason = "max_new_tokens"
    else:
        stop_reason = "continue"

    all_drafts_accepted = accepted_draft_tokens == proposed
    accounting = {
        "round_index": round_index,
        "context_length_before": context_length_before,
        "context_length_after": context_length_after,
        "draft_block_size": block_size,
        "proposed_draft_token_count": proposed,
        "accepted_draft_token_count": accepted_draft_tokens,
        "rejected_draft_token_count": rejected,
        "pending_target_seed_token_count": 1,
        "committed_token_count": committed,
        "target_only_or_bonus_token_count": 1,
        "bonus_token_count": 1 if all_drafts_accepted else 0,
        "correction_token_count": 0 if all_drafts_accepted else 1,
        "target_token_kind": "bonus" if all_drafts_accepted else "correction",
        "raw_cumulative_generated_token_count": raw_generated_after,
        "cumulative_generated_token_count": returned_tokens_after,
        "returned_token_delta": returned_token_delta,
        "truncated_token_count": raw_generated_after - returned_tokens_after,
        "remaining_output_budget_before": max_length - context_length_before,
        "stop_reason": stop_reason,
    }
    accounting["conservation"] = {
        "proposal_partition": proposed == accepted_draft_tokens + rejected,
        "commit_partition": committed == 1 + accepted_draft_tokens,
        "target_emission_partition": 1
        == accounting["bonus_token_count"] + accounting["correction_token_count"],
        "context_progress": context_length_after == context_length_before + committed,
        "raw_generation_progress": raw_generated_after
        == raw_generated_before + committed,
        "returned_within_budget": 0 <= returned_tokens_after <= max_new_tokens,
    }
    return accounting
