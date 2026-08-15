"""Pure-Python helpers for opt-in DFlash Phase 1A diagnostics.

This file is first-party instrumentation carried alongside the pinned third-party
model code.  It deliberately does not import torch so accounting can be tested on
CPU-only development machines.
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from types import TracebackType
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


def rank_position_category(
    *, block_position: int, accepted_prefix_length: int, proposed_count: int
) -> str:
    """Classify one draft position without inferring a candidate-tree outcome."""

    if not 1 <= block_position <= proposed_count:
        raise ValueError("block_position must address a proposed draft token")
    if not 0 <= accepted_prefix_length <= proposed_count:
        raise ValueError("accepted_prefix_length is outside the draft block")
    if accepted_prefix_length == proposed_count:
        return "full_block_accepted"
    zero_based_position = block_position - 1
    if zero_based_position < accepted_prefix_length:
        return "accepted_prefix"
    if zero_based_position == accepted_prefix_length:
        return "first_rejection_frontier"
    return "post_rejection_counterfactual"


def deterministic_rank_from_logits(logits: list[float], target_token_id: int) -> int:
    """Return 1-based descending-logit rank, breaking ties by token ID."""

    if not 0 <= target_token_id < len(logits):
        raise ValueError("target_token_id is outside the vocabulary")
    target_logit = logits[target_token_id]
    return 1 + sum(
        value > target_logit
        or (value == target_logit and token_id < target_token_id)
        for token_id, value in enumerate(logits)
    )


def validate_phase2a_mode_options(
    *,
    enable_rank_diagnostics: bool,
    enable_verification_profiling: bool,
    enable_profiling: bool,
    return_diagnostics: bool,
    temperature: float,
    rank_top_ms: list[int] | None,
    rank_max_m: int,
) -> list[int]:
    """Validate opt-in Phase 2A modes without importing a model framework."""

    if enable_rank_diagnostics and enable_verification_profiling:
        raise ValueError(
            "rank diagnostics and verification profiling are mutually exclusive"
        )
    if enable_rank_diagnostics and enable_profiling:
        raise ValueError("rank diagnostics cannot run with CUDA profiling")
    if enable_verification_profiling and not enable_profiling:
        raise ValueError("verification module profiling requires CUDA profiling")
    if (enable_rank_diagnostics or enable_verification_profiling) and (
        not return_diagnostics
    ):
        raise ValueError("Phase 2A modes require return_diagnostics=True")
    if enable_rank_diagnostics and temperature != 0.0:
        raise ValueError("rank diagnostics require temperature=0")

    configured = [2, 4, 8, 16] if rank_top_ms is None else list(rank_top_ms)
    if enable_rank_diagnostics:
        if (
            not configured
            or configured != sorted(set(configured))
            or configured[0] < 2
        ):
            raise ValueError("rank Top-M values must be unique, increasing, and >= 2")
        if rank_max_m < 16 or rank_max_m < max(configured):
            raise ValueError("rank max_m must be >= 16 and every requested Top-M")
    return configured


def validate_phase2b_capture_options(
    *,
    enable_tree_capture: bool,
    tree_top_k: int,
    enable_rank_diagnostics: bool,
    enable_verification_profiling: bool,
    enable_profiling: bool,
    return_diagnostics: bool,
    temperature: float,
) -> None:
    """Validate the isolated Phase 2B candidate capture mode."""

    if not enable_tree_capture:
        return
    if not return_diagnostics:
        raise ValueError("Phase 2B tree capture requires return_diagnostics=True")
    if temperature != 0.0:
        raise ValueError("Phase 2B tree capture requires temperature=0")
    if enable_rank_diagnostics or enable_verification_profiling or enable_profiling:
        raise ValueError(
            "Phase 2B tree capture is isolated from Phase 2A rank/profiling modes"
        )
    if not 2 <= tree_top_k <= 16:
        raise ValueError("Phase 2B tree_top_k must be between 2 and 16")


def build_tree_candidate_capture(
    torch_module: Any,
    *,
    draft_logits: Any,
    block_output_ids: Any,
    target_verification_token_ids: Any,
    round_index: int,
    context_length_before: int,
    prompt_length: int,
    accepted_draft_tokens: int,
    top_k: int,
) -> dict[str, Any]:
    """Extract bounded candidates from the exact logits used by draft sampling.

    Only ``P * top_k`` token IDs and normalized log probabilities cross to CPU.
    The full-vocabulary logits remain transient GPU tensors owned by generation.
    """

    if len(draft_logits.shape) != 3 or int(draft_logits.shape[0]) != 1:
        raise ValueError("tree capture requires draft logits shaped [1, P, V]")
    proposed_count = int(draft_logits.shape[1])
    vocabulary_size = int(draft_logits.shape[2])
    if proposed_count <= 0 or not 2 <= top_k <= min(16, vocabulary_size):
        raise ValueError("tree capture top_k must be in [2, min(16, vocab_size)]")
    if (
        len(block_output_ids.shape) != 2
        or int(block_output_ids.shape[0]) != 1
        or int(block_output_ids.shape[1]) != proposed_count + 1
    ):
        raise ValueError("tree capture block IDs do not align with draft logits")
    if (
        len(target_verification_token_ids.shape) != 2
        or int(target_verification_token_ids.shape[0]) != 1
        or int(target_verification_token_ids.shape[1]) < proposed_count
    ):
        raise ValueError("tree capture verification IDs do not align with drafts")
    if not 0 <= accepted_draft_tokens <= proposed_count:
        raise ValueError("accepted draft length is outside the proposed block")

    draft = draft_logits[0]
    ordered_token_ids = torch_module.argsort(
        draft, dim=-1, descending=True, stable=True
    )[:, :top_k]
    normalized_log_probabilities = torch_module.log_softmax(
        draft.float(), dim=-1
    )
    ordered_log_probabilities = torch_module.gather(
        normalized_log_probabilities, 1, ordered_token_ids
    )

    baseline_ids = block_output_ids[0, 1:].detach()
    top1_ids = ordered_token_ids[:, 0]
    if not bool(torch_module.equal(baseline_ids, top1_ids)):
        raise RuntimeError(
            "Phase 2B alignment failed: sampled draft IDs differ from exact-logit "
            f"Top-1 at round={round_index}, context={context_length_before}, "
            f"draft={baseline_ids.cpu().tolist()}, top1={top1_ids.cpu().tolist()}"
        )

    verification_ids = target_verification_token_ids[0, :proposed_count]
    verification_lcp = int(
        (baseline_ids == verification_ids).cumprod(dim=0).sum().item()
    )
    if verification_lcp != accepted_draft_tokens:
        raise RuntimeError(
            "Phase 2B verification alignment failed: "
            f"round={round_index}, context={context_length_before}, "
            f"accepted={accepted_draft_tokens}, lcp={verification_lcp}, "
            f"draft={baseline_ids.cpu().tolist()}, "
            f"verification={verification_ids.detach().cpu().tolist()}"
        )

    top_ids_cpu = ordered_token_ids.detach().cpu().tolist()
    top_log_probs_cpu = ordered_log_probabilities.detach().cpu().tolist()
    baseline_ids_cpu = baseline_ids.cpu().tolist()
    positions = []
    for position_index in range(proposed_count):
        candidates = []
        for candidate_index in range(top_k):
            log_probability = float(top_log_probs_cpu[position_index][candidate_index])
            if not math.isfinite(log_probability):
                raise RuntimeError(
                    "Phase 2B capture encountered a non-finite normalized score at "
                    f"round={round_index}, position={position_index + 1}, "
                    f"rank={candidate_index + 1}"
                )
            candidates.append(
                {
                    "rank": candidate_index + 1,
                    "token_id": int(top_ids_cpu[position_index][candidate_index]),
                    "log_probability": log_probability,
                }
            )
        positions.append(
            {
                "block_position": position_index + 1,
                "baseline_token_id": int(baseline_ids_cpu[position_index]),
                "candidates": candidates,
                "top1_top2_log_probability_margin": float(
                    candidates[0]["log_probability"]
                    - candidates[1]["log_probability"]
                ),
            }
        )

    root_token_id = int(block_output_ids[0, 0].detach().cpu().item())
    return {
        "round_index": int(round_index),
        "root_token_id": root_token_id,
        "root_absolute_output_index": int(context_length_before),
        "round_output_start_offset": int(context_length_before - prompt_length),
        "baseline_draft_token_ids": [int(value) for value in baseline_ids_cpu],
        "positions": positions,
        "top_k": int(top_k),
        "proposed_draft_token_count": proposed_count,
        "accepted_draft_token_count": int(accepted_draft_tokens),
        "immediate_verification_lcp": verification_lcp,
        "immediate_alignment_verified": True,
        "logit_source": "exact final draft logits used by temperature-zero sampling",
        "score_semantics": "normalized log probability from log_softmax",
        "tie_break_rule": "descending_logit_then_ascending_token_id",
        "full_vocabulary_logits_saved": False,
        "target_verification_token_ids_saved": False,
    }


def build_target_rank_events(
    torch_module: Any,
    *,
    draft_logits: Any,
    target_verification_logits: Any,
    round_index: int,
    accepted_prefix_length: int,
    top_ms: list[int],
    max_m: int,
) -> list[dict[str, Any]]:
    """Compute exact rank records from already-produced GPU logits.

    The vocabulary is ordered by descending draft logit and then ascending token
    ID for exact ties. Only scalar metadata and the leading ``max_m`` values are
    transferred off device; full-vocabulary logits are never retained.
    """

    if len(draft_logits.shape) != 3 or int(draft_logits.shape[0]) != 1:
        raise ValueError("rank diagnostics require draft logits shaped [1, P, V]")
    proposed_count = int(draft_logits.shape[1])
    vocabulary_size = int(draft_logits.shape[2])
    if (
        len(target_verification_logits.shape) != 3
        or int(target_verification_logits.shape[0]) != 1
        or int(target_verification_logits.shape[1]) < proposed_count
        or int(target_verification_logits.shape[2]) != vocabulary_size
    ):
        raise ValueError("target verification logits do not align with draft logits")
    if not top_ms or any(value < 2 for value in top_ms):
        raise ValueError("rank Top-M values must be at least 2")
    if max_m < 16 or max_m < max(top_ms) or max_m > vocabulary_size:
        raise ValueError("rank max_m must be >= 16, >= every Top-M, and <= vocab")

    draft = draft_logits[0]
    target = target_verification_logits[0, :proposed_count]
    target_token_ids = torch_module.argmax(target, dim=-1)
    draft_top1_token_ids = torch_module.argmax(draft, dim=-1)
    target_draft_logits = torch_module.gather(
        draft, 1, target_token_ids.unsqueeze(-1)
    ).squeeze(-1)
    draft_top1_logits = torch_module.gather(
        draft, 1, draft_top1_token_ids.unsqueeze(-1)
    ).squeeze(-1)
    target_verification_values = torch_module.gather(
        target, 1, target_token_ids.unsqueeze(-1)
    ).squeeze(-1)

    vocabulary_ids = torch_module.arange(vocabulary_size, device=draft.device)
    greater_count = (draft > target_draft_logits.unsqueeze(-1)).sum(dim=-1)
    tied_lower_id_count = (
        (draft == target_draft_logits.unsqueeze(-1))
        & (vocabulary_ids.unsqueeze(0) < target_token_ids.unsqueeze(-1))
    ).sum(dim=-1)
    ranks = greater_count + tied_lower_id_count + 1

    ordered_token_ids = torch_module.argsort(
        draft, dim=-1, descending=True, stable=True
    )[:, :max_m]
    ordered_logits = torch_module.gather(draft, 1, ordered_token_ids)

    target_ids_cpu = target_token_ids.detach().cpu().tolist()
    draft_top1_ids_cpu = draft_top1_token_ids.detach().cpu().tolist()
    target_draft_logits_cpu = target_draft_logits.detach().float().cpu().tolist()
    draft_top1_logits_cpu = draft_top1_logits.detach().float().cpu().tolist()
    target_verification_cpu = (
        target_verification_values.detach().float().cpu().tolist()
    )
    ranks_cpu = ranks.detach().cpu().tolist()
    top_ids_cpu = ordered_token_ids.detach().cpu().tolist()
    top_logits_cpu = ordered_logits.detach().float().cpu().tolist()

    events = []
    configured_top_ms = sorted({int(value) for value in top_ms})
    for position_index in range(proposed_count):
        rank = int(ranks_cpu[position_index])
        target_token_id = int(target_ids_cpu[position_index])
        draft_top1_token_id = int(draft_top1_ids_cpu[position_index])
        category = rank_position_category(
            block_position=position_index + 1,
            accepted_prefix_length=accepted_prefix_length,
            proposed_count=proposed_count,
        )
        if category == "first_rejection_frontier" and (
            target_token_id == draft_top1_token_id
        ):
            raise RuntimeError("rank frontier unexpectedly matches draft Top-1")
        event = {
            "round_index": int(round_index),
            # Position zero is the target-produced seed. Draft positions are 1..15.
            "block_position": position_index + 1,
            "position_category": category,
            "accepted_prefix_length": int(accepted_prefix_length),
            "target_token_id": target_token_id,
            "draft_top1_token_id": draft_top1_token_id,
            "target_rank_in_draft": rank,
            "target_in_top_1": rank <= 1,
            "target_in_top_2": rank <= 2,
            "target_in_top_4": rank <= 4,
            "target_in_top_8": rank <= 8,
            "target_in_top_16": rank <= 16,
            # This is the target token's value in the draft distribution.
            "target_logit": float(target_draft_logits_cpu[position_index]),
            "target_verification_logit": float(
                target_verification_cpu[position_index]
            ),
            "draft_top1_logit": float(draft_top1_logits_cpu[position_index]),
            "top1_minus_target_logit_margin": float(
                draft_top1_logits_cpu[position_index]
                - target_draft_logits_cpu[position_index]
            ),
            "top_m_token_ids": [
                int(value) for value in top_ids_cpu[position_index]
            ],
            "top_m_logits": [
                float(value) for value in top_logits_cpu[position_index]
            ],
            "configured_top_ms": configured_top_ms,
            "rank_max_m": int(max_m),
            "tie_break_rule": "descending_logit_then_ascending_token_id",
            "single_frontier_recoverability_only": True,
            "tree_accepted_length_derived": False,
        }
        events.append(event)
    return events


class TargetModuleCudaProfiler:
    """Install temporary target-module hooks for one explicitly named stage."""

    def __init__(
        self,
        torch_module: Any,
        *,
        timer: DeferredCudaTimer,
        target: Any,
        stage: str,
        round_index: int | None,
        records: list[dict[str, Any]],
        enabled: bool,
    ) -> None:
        if stage not in {"prefill", "verification"}:
            raise ValueError("target module stage must be prefill or verification")
        self.torch = torch_module
        self.timer = timer
        self.target = target
        self.stage = stage
        self.round_index = round_index
        self.records = records
        self.enabled = enabled
        self._handles: list[Any] = []
        self._active: dict[int, tuple[dict[str, Any], str, str, int | None]] = {}

    def _register(self, module: Any, kind: str, layer_index: int | None) -> None:
        range_stage = "verify" if self.stage == "verification" else "prefill"
        range_name = (
            f"target.{range_stage}.lm_head"
            if kind == "lm_head"
            else f"target.{range_stage}.layer.{layer_index}.{kind}"
        )

        def pre_hook(hooked_module: Any, inputs: Any) -> None:
            del inputs
            pair = self.timer.start()
            self.torch.cuda.nvtx.range_push(range_name)
            self._active[id(hooked_module)] = (
                pair,
                range_name,
                kind,
                layer_index,
            )

        def post_hook(hooked_module: Any, inputs: Any, output: Any) -> None:
            del inputs, output
            active = self._active.pop(id(hooked_module), None)
            if active is None:
                raise RuntimeError(f"Missing target module timing start: {range_name}")
            pair, active_name, active_kind, active_layer = active
            self.timer.stop(pair)
            self.torch.cuda.nvtx.range_pop()
            self.records.append(
                {
                    "stage": self.stage,
                    "round_index": self.round_index,
                    "module_kind": active_kind,
                    "layer_index": active_layer,
                    "nvtx_range": active_name,
                    "cpu_elapsed_ms": None,
                    "cuda_elapsed_ms": None,
                    "_timing_pair": pair,
                }
            )

        self._handles.append(module.register_forward_pre_hook(pre_hook))
        try:
            handle = module.register_forward_hook(post_hook, always_call=True)
        except TypeError:
            handle = module.register_forward_hook(post_hook)
        self._handles.append(handle)

    def __enter__(self):
        if not self.enabled:
            return self
        layers = getattr(getattr(self.target, "model", None), "layers", None)
        lm_head = getattr(self.target, "lm_head", None)
        if layers is None or lm_head is None:
            raise RuntimeError("Target model does not expose model.layers and lm_head")
        for layer_index, layer in enumerate(layers):
            self._register(layer.self_attn, "self_attn", layer_index)
            self._register(layer.mlp, "mlp", layer_index)
        self._register(lm_head, "lm_head", None)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        # ``always_call`` handles normal target exceptions. This fallback prevents
        # an unmatched NVTX push on older torch hook APIs.
        while self._active:
            _, (pair, _, _, _) = self._active.popitem()
            self.timer.stop(pair)
            self.torch.cuda.nvtx.range_pop()


def resolve_target_module_records(
    timer: DeferredCudaTimer, records: list[dict[str, Any]]
) -> None:
    for record in records:
        pair = record.pop("_timing_pair")
        record.update(timer.values(pair))


def summarize_target_module_records(
    records: list[dict[str, Any]],
    *,
    total_cuda_ms: float | None,
    target_layer_count: int,
    invocation_count: int,
) -> dict[str, Any]:
    """Summarize non-overlapping module hooks and validate the remainder."""

    totals = {
        kind: sum(
            float(record["cuda_elapsed_ms"] or 0.0)
            for record in records
            if record["module_kind"] == kind
        )
        for kind in ("self_attn", "mlp", "lm_head")
    }
    counts = {
        kind: sum(record["module_kind"] == kind for record in records)
        for kind in totals
    }
    expected_counts = {
        "self_attn": target_layer_count * invocation_count,
        "mlp": target_layer_count * invocation_count,
        "lm_head": invocation_count,
    }
    count_invariants = {
        kind: counts[kind] == expected_counts[kind] for kind in totals
    }
    module_sum = sum(totals.values())
    raw_unattributed = (
        float(total_cuda_ms) - module_sum if total_cuda_ms is not None else None
    )
    partition_reliable = (
        total_cuda_ms is not None
        and raw_unattributed is not None
        and raw_unattributed >= 0.0
        and all(count_invariants.values())
    )
    unattributed = raw_unattributed if partition_reliable else None

    per_layer = []
    for layer_index in range(target_layer_count):
        self_attention = sum(
            float(record["cuda_elapsed_ms"] or 0.0)
            for record in records
            if record["module_kind"] == "self_attn"
            and record["layer_index"] == layer_index
        )
        mlp = sum(
            float(record["cuda_elapsed_ms"] or 0.0)
            for record in records
            if record["module_kind"] == "mlp"
            and record["layer_index"] == layer_index
        )
        per_layer.append(
            {
                "layer_index": layer_index,
                "self_attention_module_cuda_ms": self_attention,
                "mlp_module_cuda_ms": mlp,
            }
        )

    denominator = float(total_cuda_ms) if total_cuda_ms else None
    return {
        "target_self_attention_module_cuda_ms": totals["self_attn"],
        "target_mlp_module_cuda_ms": totals["mlp"],
        "target_lm_head_cuda_ms": totals["lm_head"],
        "target_verification_cuda_ms": total_cuda_ms,
        "target_module_sum_cuda_ms": module_sum,
        "target_unattributed_cuda_ms": unattributed,
        "target_raw_unattributed_cuda_ms": raw_unattributed,
        "target_self_attention_over_verification": (
            totals["self_attn"] / denominator if denominator else None
        ),
        "target_mlp_over_verification": (
            totals["mlp"] / denominator if denominator else None
        ),
        "target_lm_head_over_verification": (
            totals["lm_head"] / denominator if denominator else None
        ),
        "partition_reliable": partition_reliable,
        "unattributed_reason": (
            None
            if partition_reliable
            else "negative remainder, missing total, or unexpected hook call count"
        ),
        "module_call_counts": counts,
        "expected_module_call_counts": expected_counts,
        "module_call_count_invariants": count_invariants,
        "per_layer": per_layer,
        "self_attention_scope": "QKV projections, rotary/SDPA path, and output projection",
        "pure_sdpa_isolated": False,
    }


def validate_attention_kv_lengths(
    *,
    context_length_before: int,
    cached_context_kv_length: int,
    newly_injected_hidden_length: int,
    persistent_context_kv_length: int,
    query_length: int,
    effective_kv_length: int,
) -> dict[str, bool]:
    """Validate the context/KV meanings at the actual combined attention call.

    The draft cache contains context committed before the current round.  Projecting
    ``target_hidden`` appends the newly committed context features.  The SDPA K/V
    tensors then contain that complete persistent context followed by the current
    speculative query block.
    """

    invariants = {
        "persistent_equals_context_before": (
            persistent_context_kv_length == context_length_before
        ),
        "persistent_partition": (
            persistent_context_kv_length
            == cached_context_kv_length + newly_injected_hidden_length
        ),
        "effective_partition": (
            effective_kv_length == persistent_context_kv_length + query_length
        ),
    }
    if not all(invariants.values()):
        raise RuntimeError(
            "DFlash attention KV length invariant failed: "
            f"context_length_before={context_length_before}, "
            f"cached_context_kv_length={cached_context_kv_length}, "
            f"newly_injected_hidden_length={newly_injected_hidden_length}, "
            f"persistent_context_kv_length={persistent_context_kv_length}, "
            f"query_length={query_length}, effective_kv_length={effective_kv_length}"
        )
    return invariants


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
