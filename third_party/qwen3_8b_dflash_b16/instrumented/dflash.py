import time
from typing import Optional, Callable
from typing_extensions import Unpack, Tuple
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from .phase1_observability import (
    DeferredCudaTimer,
    TargetModuleCudaProfiler,
    build_tree_candidate_capture,
    build_target_rank_events,
    build_round_accounting,
    finalize_generation_result,
    gpu_memory_snapshot,
    nvtx_range,
    resolve_target_module_records,
    summarize_target_module_records,
    tensor_shape,
    validate_attention_kv_lengths,
    validate_phase2a_mode_options,
    validate_phase2b_capture_options,
)
from .phase2c_online import (
    build_online_tree_tensors,
    build_tree_attention_mask,
    clone_dynamic_cache,
    compact_dynamic_cache,
    dynamic_cache_lengths,
    extract_topk_candidates,
    greedy_tree_traversal_tensors,
    tensor_tree_metadata,
    tree_position_tensors,
    validate_online_tree_options,
)
from .utils import build_target_layer_ids, extract_context_feature, sample


class _Phase2CNoopTimer:
    """Avoid host timing work on the formal Phase 2C control path."""

    @staticmethod
    def start():
        return None

    @staticmethod
    def stop(_pair):
        return None

    @staticmethod
    def resolve():
        return None

    @staticmethod
    def values(_pair):
        return {"cpu_elapsed_ms": None, "cuda_elapsed_ms": None}


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        phase1_attention_records = kwargs.pop("phase1_attention_records", None)
        phase1_round_index = kwargs.pop("phase1_round_index", None)
        phase1_profiling = kwargs.pop("phase1_profiling", False)
        phase1_attention_timer = kwargs.pop("phase1_attention_timer", None)
        phase1_attention_timing_pairs = kwargs.pop(
            "phase1_attention_timing_pairs", None
        )
        phase1_context_length_before = kwargs.pop(
            "phase1_context_length_before", None
        )
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        cached_context_kv_length = 0
        if past_key_values is not None:
            cached_context_kv_length = int(
                past_key_values.get_seq_length(self.layer_idx)
            )
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        effective_kv_length = int(k.shape[-2])
        persistent_context_kv_length = effective_kv_length - int(q_len)
        kv_length_invariants = None
        if phase1_context_length_before is not None:
            kv_length_invariants = validate_attention_kv_lengths(
                context_length_before=int(phase1_context_length_before),
                cached_context_kv_length=cached_context_kv_length,
                newly_injected_hidden_length=int(ctx_len),
                persistent_context_kv_length=persistent_context_kv_length,
                query_length=int(q_len),
                effective_kv_length=effective_kv_length,
            )
        attention_record = None
        if phase1_attention_records is not None:
            attention_record = {
                "round_index": phase1_round_index,
                "layer_index": self.layer_idx,
                "configured_attention_implementation": self.config._attn_implementation,
                "calls_pytorch_sdpa": self.config._attn_implementation == "sdpa",
                "dtype": str(q.dtype),
                "batch_size": int(bsz),
                "query_length": int(q_len),
                # Compatibility alias: this has always meant the number of
                # target hidden states newly supplied in this round.
                "injected_context_length": int(ctx_len),
                "newly_injected_hidden_length": int(ctx_len),
                "cached_context_kv_length_before_injection": (
                    cached_context_kv_length
                ),
                "persistent_context_kv_length": persistent_context_kv_length,
                "current_block_kv_length": int(ctx_len + q_len),
                "effective_kv_length": effective_kv_length,
                "query_head_count": int(self.config.num_attention_heads),
                "key_value_head_count": int(self.config.num_key_value_heads),
                "head_dimension": int(self.head_dim),
                "query_shape": tensor_shape(q),
                "key_shape": tensor_shape(k),
                "value_shape": tensor_shape(v),
                "theoretical_qk_score_elements": int(
                    bsz * self.config.num_attention_heads * q_len * k.shape[-2]
                ),
                "kv_length_invariants": kv_length_invariants,
                "combined_attention_cuda_elapsed_ms": None,
                "cuda_region_isolated": True,
                "kernel_latency_isolated": False,
                "timing_scope": (
                    "combined persistent-context and current-block attention"
                ),
            }
            phase1_attention_records.append(attention_record)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attention_pair = (
            phase1_attention_timer.start()
            if phase1_profiling and phase1_attention_timer is not None
            else None
        )
        try:
            with nvtx_range(
                torch, "dflash.attention.combined_context_block", phase1_profiling
            ):
                attn_output, attn_weights = attn_fn(
                    self,
                    q,
                    k,
                    v,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    sliding_window=self.sliding_window,
                    **kwargs,
                )
        finally:
            if attention_pair is not None:
                phase1_attention_timer.stop(attention_pair)
        if (
            attention_pair is not None
            and attention_record is not None
            and phase1_attention_timing_pairs is not None
        ):
            phase1_attention_timing_pairs.append((attention_record, attention_pair))
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.target_layer_ids = self.config.dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = self.config.dflash_config.get("mask_token_id", None)
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        *,
        return_diagnostics: bool = False,
        enable_profiling: bool = False,
        enable_rank_diagnostics: bool = False,
        rank_top_ms: list[int] | None = None,
        rank_max_m: int = 16,
        enable_verification_profiling: bool = False,
        enable_tree_capture: bool = False,
        tree_top_k: int = 16,
        enable_online_tree: bool = False,
        online_tree_policy: str = "top1_chain",
        online_tree_node_budget: int = 16,
        online_tree_beam_width: int = 4,
        online_tree_draft_top_k: int = 16,
        enable_online_tree_profiling: bool = False,
        enable_online_tree_gold_checks: bool = False,
    ):
        """Generate with the pinned behavior; diagnostics are strictly opt-in."""

        configured_rank_top_ms = validate_phase2a_mode_options(
            enable_rank_diagnostics=enable_rank_diagnostics,
            enable_verification_profiling=enable_verification_profiling,
            enable_profiling=enable_profiling,
            return_diagnostics=return_diagnostics,
            temperature=temperature,
            rank_top_ms=rank_top_ms,
            rank_max_m=rank_max_m,
        )
        validate_phase2b_capture_options(
            enable_tree_capture=enable_tree_capture,
            tree_top_k=tree_top_k,
            enable_rank_diagnostics=enable_rank_diagnostics,
            enable_verification_profiling=enable_verification_profiling,
            enable_profiling=enable_profiling,
            return_diagnostics=return_diagnostics,
            temperature=temperature,
        )
        phase2c_profiling = enable_online_tree_profiling or enable_profiling
        validate_online_tree_options(
            enabled=enable_online_tree,
            policy=online_tree_policy,
            node_budget=online_tree_node_budget,
            beam_width=online_tree_beam_width,
            draft_top_k=online_tree_draft_top_k,
            temperature=temperature,
            block_size=self.block_size,
            batch_size=int(input_ids.shape[0]),
            return_diagnostics=return_diagnostics,
            enable_profiling=phase2c_profiling,
            enable_gold_checks=enable_online_tree_gold_checks,
        )
        if enable_online_tree:
            if any(
                (
                    enable_rank_diagnostics,
                    enable_verification_profiling,
                    enable_tree_capture,
                )
            ):
                raise ValueError(
                    "Phase 2C online tree is isolated from Phase 1/2A/2B modes"
                )
            return self._spec_generate_online_tree(
                target=target,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                stop_token_ids=stop_token_ids,
                temperature=temperature,
                return_diagnostics=return_diagnostics,
                enable_profiling=phase2c_profiling,
                enable_gold_checks=enable_online_tree_gold_checks,
                tree_policy=online_tree_policy,
                node_budget=online_tree_node_budget,
                beam_width=online_tree_beam_width,
                draft_top_k=online_tree_draft_top_k,
            )
        if not any(
            (
                return_diagnostics,
                enable_profiling,
                enable_rank_diagnostics,
                enable_verification_profiling,
                enable_tree_capture,
            )
        ):
            return self._spec_generate_reference(
                target=target,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                stop_token_ids=stop_token_ids,
                temperature=temperature,
            )
        return self._spec_generate_observed(
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            return_diagnostics=return_diagnostics,
            enable_profiling=enable_profiling,
            enable_rank_diagnostics=enable_rank_diagnostics,
            rank_top_ms=configured_rank_top_ms,
            rank_max_m=rank_max_m,
            enable_verification_profiling=enable_verification_profiling,
            enable_tree_capture=enable_tree_capture,
            tree_top_k=tree_top_k,
        )

    @staticmethod
    def _phase2c_compare_logits(
        torch_module,
        packed_logits,
        reference_logits,
        *,
        top_k=8,
        maximum_absolute_tolerance=0.5,
        mean_absolute_tolerance=0.02,
    ):
        packed = packed_logits.float()
        reference = reference_logits.float()
        difference = (packed - reference).abs()
        packed_top = torch_module.argsort(
            packed, descending=True, stable=True
        )[:top_k]
        reference_top = torch_module.argsort(
            reference, descending=True, stable=True
        )[:top_k]
        result = {
            "greedy_argmax_equal": bool(
                (torch_module.argmax(packed) == torch_module.argmax(reference)).item()
            ),
            "top_k": int(top_k),
            "top_k_token_ids_equal": bool(
                torch_module.equal(packed_top, reference_top)
            ),
            "maximum_absolute_error": float(difference.max().item()),
            "mean_absolute_error": float(difference.mean().item()),
            "maximum_absolute_tolerance": float(maximum_absolute_tolerance),
            "mean_absolute_tolerance": float(mean_absolute_tolerance),
        }
        result["within_bfloat16_tolerance"] = (
            result["maximum_absolute_error"] <= maximum_absolute_tolerance
            and result["mean_absolute_error"] <= mean_absolute_tolerance
        )
        result["passed"] = (
            result["greedy_argmax_equal"]
            and result["top_k_token_ids_equal"]
            and result["within_bfloat16_tolerance"]
        )
        return result

    def _phase2c_packed_logits_gold_check(
        self,
        *,
        target,
        prefix_cache,
        tree,
        packed_logits,
        context_length_before,
    ):
        node_results = []
        root = tree["token_ids"][0].reshape(1)
        for node_index in range(tree["actual_total_node_count"]):
            depth = int(tree["depths"][node_index].item())
            path = tree["path_tokens"][node_index, :depth]
            linear_ids = torch.cat([root, path]).unsqueeze(0)
            semantic_positions = torch.arange(
                context_length_before,
                context_length_before + depth + 1,
                dtype=torch.long,
                device=linear_ids.device,
            )
            storage_positions = torch.arange(
                context_length_before,
                context_length_before + depth + 1,
                dtype=torch.long,
                device=linear_ids.device,
            )
            linear_output = target(
                linear_ids,
                position_ids=semantic_positions.unsqueeze(0),
                cache_position=storage_positions,
                past_key_values=clone_dynamic_cache(prefix_cache),
                use_cache=True,
                logits_to_keep=1,
            )
            comparison = self._phase2c_compare_logits(
                torch,
                packed_logits[0, node_index],
                linear_output.logits[0, -1],
            )
            node_results.append({"node_index": node_index, **comparison})
        passed = all(record["passed"] for record in node_results)
        result = {
            "status": "passed" if passed else "failed",
            "checked_node_count": len(node_results),
            "all_tree_nodes_checked": len(node_results)
            == tree["actual_total_node_count"],
            "greedy_argmax_all_equal": all(
                record["greedy_argmax_equal"] for record in node_results
            ),
            "top_k_token_ids_all_equal": all(
                record["top_k_token_ids_equal"] for record in node_results
            ),
            "maximum_absolute_error": max(
                record["maximum_absolute_error"] for record in node_results
            ),
            "mean_absolute_error": sum(
                record["mean_absolute_error"] for record in node_results
            )
            / len(node_results),
            "maximum_absolute_tolerance": 0.5,
            "mean_absolute_tolerance": 0.02,
            "nodes": node_results,
        }
        if not passed:
            raise RuntimeError(
                "Phase 2C packed-tree logits failed dense path gold validation: "
                f"max_error={result['maximum_absolute_error']}, "
                f"mean_error={result['mean_absolute_error']}"
            )
        return result

    def _phase2c_kv_gold_check(
        self,
        *,
        target,
        compacted_cache,
        output_ids,
        next_root_index,
        compaction,
    ):
        next_root = output_ids[:, next_root_index : next_root_index + 1]
        compacted_check_cache = clone_dynamic_cache(compacted_cache)
        compacted_output = target(
            next_root,
            position_ids=torch.tensor(
                [[next_root_index]], dtype=torch.long, device=next_root.device
            ),
            cache_position=torch.tensor(
                [next_root_index], dtype=torch.long, device=next_root.device
            ),
            past_key_values=compacted_check_cache,
            use_cache=True,
            logits_to_keep=1,
        )
        dense_ids = output_ids[:, : next_root_index + 1]
        dense_output = target(
            dense_ids,
            position_ids=torch.arange(
                dense_ids.shape[1], dtype=torch.long, device=dense_ids.device
            ).unsqueeze(0),
            cache_position=torch.arange(
                dense_ids.shape[1], dtype=torch.long, device=dense_ids.device
            ),
            past_key_values=DynamicCache(),
            use_cache=True,
            logits_to_keep=1,
        )
        comparison = self._phase2c_compare_logits(
            torch,
            compacted_output.logits[0, -1],
            dense_output.logits[0, -1],
        )
        compacted_lengths = dynamic_cache_lengths(compacted_cache)
        next_step_lengths = dynamic_cache_lengths(compacted_check_cache)
        dense_lengths = dynamic_cache_lengths(dense_output.past_key_values)
        expected_compacted = int(compaction["expected_compacted_length"])
        expected_next = next_root_index + 1
        length_checks = {
            "all_compacted_layers_equal_expected": bool(compacted_lengths)
            and all(length == expected_compacted for length in compacted_lengths),
            "all_next_step_layers_equal_expected": bool(next_step_lengths)
            and all(length == expected_next for length in next_step_lengths),
            "all_dense_layers_equal_expected": bool(dense_lengths)
            and all(length == expected_next for length in dense_lengths),
        }
        sibling_nodes_discarded = (
            compaction["discarded_tree_node_count"] >= 0
            and expected_compacted
            < compaction["temporary_layer_lengths"][0]
            if compaction["discarded_tree_node_count"]
            else True
        )
        passed = (
            comparison["passed"]
            and all(length_checks.values())
            and sibling_nodes_discarded
            and not compaction["correction_or_bonus_written"]
        )
        result = {
            "status": "passed" if passed else "failed",
            **comparison,
            "compacted_layer_lengths": compacted_lengths,
            "next_step_layer_lengths": next_step_lengths,
            "dense_reference_layer_lengths": dense_lengths,
            "length_checks": length_checks,
            "sibling_nodes_discarded": sibling_nodes_discarded,
            "correction_or_bonus_absent_from_persistent_cache": not compaction[
                "correction_or_bonus_written"
            ],
        }
        if not passed:
            raise RuntimeError(
                "Phase 2C compacted cache failed dense recomputation gold validation"
            )
        return result

    @torch.inference_mode()
    def _spec_generate_online_tree(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        *,
        return_diagnostics: bool,
        enable_profiling: bool,
        enable_gold_checks: bool,
        tree_policy: str,
        node_budget: int,
        beam_width: int,
        draft_top_k: int,
    ):
        """Run opt-in online marginal-tree drafting with packed target verification."""

        self.eval()
        target_attention_implementation = getattr(
            target.config, "_attn_implementation", None
        )
        if target_attention_implementation != "sdpa":
            raise RuntimeError(
                "Phase 2C custom 4D tree mask currently requires the PyTorch SDPA "
                f"target backend, found {target_attention_implementation!r}"
            )
        if enable_profiling and (
            not input_ids.is_cuda or not torch.cuda.is_available()
        ):
            raise RuntimeError("Phase 2C profiling requires CUDA tensors")

        num_input_tokens = int(input_ids.shape[1])
        max_length = num_input_tokens + max_new_tokens
        block_size = int(self.block_size)
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        linear_position_ids = torch.arange(
            output_ids.shape[1], device=target.device
        ).unsqueeze(0)
        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()
        if enable_profiling:
            torch.cuda.reset_peak_memory_stats(target.device)
        timer = (
            DeferredCudaTimer(torch, enabled=True, device=target.device)
            if enable_profiling
            else _Phase2CNoopTimer()
        )
        total_pair = timer.start()
        rounds = []
        timing_pairs = []
        accepted_lengths = []
        cache_commit_lengths = []
        packed_gold_records = []
        kv_gold_records = []

        with nvtx_range(torch, "phase2c.total", enable_profiling):
            prefill_pair = timer.start()
            with nvtx_range(torch, "phase2c.prefill", enable_profiling):
                output = target(
                    input_ids,
                    position_ids=linear_position_ids[:, :num_input_tokens],
                    cache_position=linear_position_ids[0, :num_input_tokens],
                    past_key_values=past_key_values_target,
                    use_cache=True,
                    logits_to_keep=1,
                    output_hidden_states=True,
                )
            timer.stop(prefill_pair)
            output_ids[:, :num_input_tokens] = input_ids
            output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
                output.logits, temperature
            )
            target_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )

            start = num_input_tokens
            round_index = 0
            returned_tokens_before = min(1, max_new_tokens)
            while start < max_length:
                context_length_before = int(start)
                if past_key_values_target.get_seq_length() != context_length_before:
                    raise RuntimeError(
                        "Phase 2C target cache does not equal round prefix length"
                    )
                round_pair = timer.start()
                with nvtx_range(torch, "phase2c.round", enable_profiling):
                    block_output_ids = output_ids[
                        :, start : start + block_size
                    ].clone()
                    noise_embedding = target.model.embed_tokens(block_output_ids)
                    draft_position_ids = linear_position_ids[
                        :,
                        past_key_values_draft.get_seq_length() : start + block_size,
                    ]

                    total_draft_pair = timer.start()
                    draft_pair = timer.start()
                    with nvtx_range(torch, "phase2c.draft", enable_profiling):
                        draft_hidden = self(
                            target_hidden=target_hidden,
                            noise_embedding=noise_embedding,
                            position_ids=draft_position_ids,
                            past_key_values=past_key_values_draft,
                            use_cache=True,
                            is_causal=False,
                        )
                        draft_logits = target.lm_head(
                            draft_hidden[:, -block_size + 1 :, :]
                        )
                        past_key_values_draft.crop(start)
                    timer.stop(draft_pair)

                    topk_pair = timer.start()
                    with nvtx_range(torch, "phase2c.topk", enable_profiling):
                        candidates = extract_topk_candidates(
                            torch, draft_logits, top_k=draft_top_k
                        )
                    timer.stop(topk_pair)
                    # This equivalence assertion is diagnostic-only.  Avoid the
                    # redundant argmax/comparison on the formal control path.
                    top1_matches_sampling = None
                    if return_diagnostics:
                        baseline_ids = sample(draft_logits, temperature)
                        top1_matches_sampling = (
                            baseline_ids[0] == candidates["token_ids"][:, 0]
                        ).all()

                    tree_build_pair = timer.start()
                    with nvtx_range(torch, "phase2c.tree_build", enable_profiling):
                        tree = build_online_tree_tensors(
                            torch,
                            candidates=candidates,
                            root_token_id=block_output_ids[0, 0],
                            policy=tree_policy,
                            node_budget=node_budget,
                            beam_width=beam_width,
                        )
                    timer.stop(tree_build_pair)

                    mask_pair = timer.start()
                    with nvtx_range(torch, "phase2c.mask_build", enable_profiling):
                        tree_mask = build_tree_attention_mask(
                            torch,
                            tree=tree,
                            prefix_length=context_length_before,
                            dtype=target.model.embed_tokens.weight.dtype,
                        )
                        semantic_positions, cache_positions = tree_position_tensors(
                            torch,
                            tree=tree,
                            context_length_before=context_length_before,
                        )
                    timer.stop(mask_pair)
                    timer.stop(total_draft_pair)

                    gold_prefix_cache = (
                        clone_dynamic_cache(past_key_values_target)
                        if enable_gold_checks and round_index == 0
                        else None
                    )
                    total_verification_pair = timer.start()
                    verify_pair = timer.start()
                    with nvtx_range(torch, "phase2c.verify", enable_profiling):
                        packed_output = target(
                            tree["token_ids"].unsqueeze(0),
                            attention_mask=tree_mask,
                            position_ids=semantic_positions,
                            cache_position=cache_positions,
                            past_key_values=past_key_values_target,
                            use_cache=True,
                            output_hidden_states=True,
                        )
                    timer.stop(verify_pair)

                    traverse_pair = timer.start()
                    with nvtx_range(torch, "phase2c.traverse", enable_profiling):
                        traversal = greedy_tree_traversal_tensors(
                            torch, tree=tree, target_logits=packed_output.logits
                        )
                    timer.stop(traverse_pair)
                    accepted_draft_tokens = traversal[
                        "accepted_draft_token_count"
                    ]
                    committed_node_indices = torch.cat(
                        [
                            torch.zeros(
                                1,
                                dtype=torch.long,
                                device=tree["token_ids"].device,
                            ),
                            traversal["accepted_node_indices"],
                        ]
                    )
                    committed_tokens = tree["token_ids"].index_select(
                        0, committed_node_indices
                    )
                    committed_token_count = int(committed_tokens.shape[0])
                    output_ids[
                        :, start : start + committed_token_count
                    ] = committed_tokens.unsqueeze(0)
                    output_ids[:, start + committed_token_count] = traversal[
                        "next_root_token_id"
                    ]

                    compact_pair = timer.start()
                    with nvtx_range(torch, "phase2c.kv_compact", enable_profiling):
                        compaction = compact_dynamic_cache(
                            torch,
                            cache=past_key_values_target,
                            prefix_length=context_length_before,
                            actual_tree_node_count=tree[
                                "actual_total_node_count"
                            ],
                            accepted_node_indices=traversal[
                                "accepted_node_indices"
                            ],
                        )
                    timer.stop(compact_pair)
                    timer.stop(total_verification_pair)

                    start += committed_token_count
                    target_hidden_all = extract_context_feature(
                        packed_output.hidden_states, self.target_layer_ids
                    )
                    target_hidden = target_hidden_all.index_select(
                        1, committed_node_indices
                    )
                    if past_key_values_target.get_seq_length() != start:
                        raise RuntimeError(
                            "Phase 2C compacted target cache length does not equal "
                            "context_length_after"
                        )

                    if gold_prefix_cache is not None:
                        packed_gold = self._phase2c_packed_logits_gold_check(
                            target=target,
                            prefix_cache=gold_prefix_cache,
                            tree=tree,
                            packed_logits=packed_output.logits,
                            context_length_before=context_length_before,
                        )
                        packed_gold["round_index"] = round_index
                        packed_gold_records.append(packed_gold)
                        kv_gold = self._phase2c_kv_gold_check(
                            target=target,
                            compacted_cache=past_key_values_target,
                            output_ids=output_ids,
                            next_root_index=start,
                            compaction=compaction,
                        )
                        kv_gold["round_index"] = round_index
                        kv_gold_records.append(kv_gold)

                    stop_detected = stop_token_ids is not None and any(
                        stop_token_id in output_ids[:, num_input_tokens:]
                        for stop_token_id in stop_token_ids
                    )
                timer.stop(round_pair)

                eos_generated_count = None
                if stop_detected:
                    stop_tensor = torch.tensor(
                        stop_token_ids, device=output_ids.device
                    )
                    stop_indices = torch.isin(
                        output_ids[0][num_input_tokens : start + 1], stop_tensor
                    ).nonzero(as_tuple=True)[0]
                    if stop_indices.numel() > 0:
                        eos_generated_count = int(stop_indices[0].item()) + 1

                accounting = build_round_accounting(
                    round_index=round_index,
                    context_length_before=context_length_before,
                    prompt_length=num_input_tokens,
                    max_new_tokens=max_new_tokens,
                    block_size=block_size,
                    accepted_draft_tokens=accepted_draft_tokens,
                    returned_tokens_before=returned_tokens_before,
                    eos_generated_count=eos_generated_count,
                )
                returned_tokens_before = accounting[
                    "cumulative_generated_token_count"
                ]
                accepted_lengths.append(accepted_draft_tokens)
                cache_commit_lengths.append(committed_token_count)
                if return_diagnostics:
                    round_record = {
                        "schema_version": "1.0",
                        **accounting,
                        "generated_root_offset": context_length_before
                        - num_input_tokens,
                        "prefix_cache_length_before_round": context_length_before,
                        "requested_node_count": node_budget,
                        "actual_node_count": tree["actual_total_node_count"],
                        "beam_width": beam_width,
                        "maximum_tree_depth": tree["maximum_depth"],
                        "per_depth_node_counts": tree["per_depth_node_counts"],
                        "terminal_round": accounting["stop_reason"] != "continue",
                        "tree_policy": tree_policy,
                        "tree_metadata": tree,
                        "tree_attention": {
                            "mask_shape": [int(value) for value in tree_mask.shape],
                            "mask_semantics": "prefix plus self and ancestors only",
                            "semantic_position_rule": "context_length_before + tree_depth",
                            "cache_position_rule": "prefix_length + packed_node_index",
                            "packed_kv_storage": True,
                            "configured_target_attention_implementation": target_attention_implementation,
                            "actual_cuda_kernel": None,
                        },
                        "cache_compaction": compaction,
                        "gold_check_status": (
                            "passed"
                            if gold_prefix_cache is not None
                            else "not_run"
                        ),
                        "output_ids_equal_reference": None,
                        "offline_comparison_status": "not_requested",
                        "_top1_matches_sampling": top1_matches_sampling,
                        "_next_root_is_bonus": traversal["next_root_is_bonus"],
                        "timing": {},
                    }
                    rounds.append(round_record)
                    timing_pairs.append(
                        {
                            "record": round_record,
                            "draft": draft_pair,
                            "topk": topk_pair,
                            "tree_build": tree_build_pair,
                            "mask_build": mask_pair,
                            "verify": verify_pair,
                            "traverse": traverse_pair,
                            "kv_compact": compact_pair,
                            "total_draft": total_draft_pair,
                            "total_verification": total_verification_pair,
                            "round": round_pair,
                        }
                    )
                round_index += 1
                if stop_detected:
                    break

            output_ids = output_ids[:, :max_length]
            output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
            final_stop_reason = "max_new_tokens"
            if stop_token_ids is not None:
                stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
                stop_indices = torch.isin(
                    output_ids[0][num_input_tokens:], stop_tensor
                ).nonzero(as_tuple=True)[0]
                if stop_indices.numel() > 0:
                    output_ids = output_ids[
                        :, : num_input_tokens + stop_indices[0] + 1
                    ]
                    final_stop_reason = "eos"
        timer.stop(total_pair)
        timer.resolve()
        if not return_diagnostics:
            return output_ids

        component_totals = {
            "draft_hidden_logits_cuda_elapsed_ms": 0.0,
            "topk_cuda_elapsed_ms": 0.0,
            "tree_construction_cuda_elapsed_ms": 0.0,
            "mask_construction_cuda_elapsed_ms": 0.0,
            "verification_cuda_elapsed_ms": 0.0,
            "traversal_cuda_elapsed_ms": 0.0,
            "kv_compaction_cuda_elapsed_ms": 0.0,
            "total_draft_cuda_elapsed_ms": 0.0,
            "total_verification_cuda_elapsed_ms": 0.0,
            "round_cuda_elapsed_ms": 0.0,
        }
        pair_to_field = {
            "draft": "draft_hidden_logits_cuda_elapsed_ms",
            "topk": "topk_cuda_elapsed_ms",
            "tree_build": "tree_construction_cuda_elapsed_ms",
            "mask_build": "mask_construction_cuda_elapsed_ms",
            "verify": "verification_cuda_elapsed_ms",
            "traverse": "traversal_cuda_elapsed_ms",
            "kv_compact": "kv_compaction_cuda_elapsed_ms",
            "total_draft": "total_draft_cuda_elapsed_ms",
            "total_verification": "total_verification_cuda_elapsed_ms",
            "round": "round_cuda_elapsed_ms",
        }
        for timing in timing_pairs:
            record = timing["record"]
            if not bool(record.pop("_top1_matches_sampling").item()):
                raise RuntimeError(
                    "Phase 2C deterministic Top-1 differs from DFlash sampling"
                )
            next_root_is_bonus = bool(record.pop("_next_root_is_bonus").item())
            record.update(
                {
                    "bonus_token_count": 1 if next_root_is_bonus else 0,
                    "correction_token_count": 0 if next_root_is_bonus else 1,
                    "target_token_kind": (
                        "bonus" if next_root_is_bonus else "correction"
                    ),
                }
            )
            record["conservation"]["target_emission_partition"] = True
            for pair_name, field in pair_to_field.items():
                value = timer.values(timing[pair_name])["cuda_elapsed_ms"]
                record["timing"][field] = value
                component_totals[field] += value or 0.0
            round_ms = record["timing"]["round_cuda_elapsed_ms"]
            record["timing"]["draft_over_round_ratio"] = (
                record["timing"]["total_draft_cuda_elapsed_ms"] / round_ms
                if round_ms
                else None
            )
            record["timing"]["verification_over_round_ratio"] = (
                record["timing"]["total_verification_cuda_elapsed_ms"]
                / round_ms
                if round_ms
                else None
            )
            record["tree_metadata"] = tensor_tree_metadata(
                record["tree_metadata"]
            )
            compaction = record["cache_compaction"]
            record["cache_compaction"] = {
                key: (
                    value.detach().cpu().tolist()
                    if hasattr(value, "detach")
                    else value
                )
                for key, value in compaction.items()
            }

        diagnostics = {
            "schema_version": "1.0",
            "prompt_token_count": num_input_tokens,
            "generated_token_count": int(
                output_ids.shape[1] - num_input_tokens
            ),
            "max_new_tokens": int(max_new_tokens),
            "draft_block_size": block_size,
            "prefill_seed_token_count": 1,
            "accepted_draft_token_lengths": accepted_lengths,
            "cache_commit_lengths": cache_commit_lengths,
            "stop_reason": final_stop_reason,
            "rounds": rounds,
            "phase2c": {
                "enabled": True,
                "tree_policy": tree_policy,
                "requested_node_budget": node_budget,
                "beam_width": beam_width,
                "draft_top_k": draft_top_k,
                "maximum_draft_depth": 15,
                "target_information_used_for_tree_construction": False,
                "full_vocabulary_logits_saved": False,
                "target_logits_saved": False,
                "gold_checks_enabled": enable_gold_checks,
                "packed_logits_gold_checks": packed_gold_records,
                "kv_compaction_gold_checks": kv_gold_records,
            },
            "profiling": {
                "enabled": enable_profiling,
                "nvtx_ranges": [
                    "phase2c.draft",
                    "phase2c.topk",
                    "phase2c.tree_build",
                    "phase2c.mask_build",
                    "phase2c.verify",
                    "phase2c.traverse",
                    "phase2c.kv_compact",
                    "phase2c.round",
                ],
                "explicit_cuda_synchronize_count": 1 if enable_profiling else 0,
                "prefill": timer.values(prefill_pair),
                "total": timer.values(total_pair),
                "components": {
                    field: value if enable_profiling else None
                    for field, value in component_totals.items()
                },
                "memory": gpu_memory_snapshot(torch, target.device)
                if enable_profiling
                else None,
                "custom_4d_tree_mask": True,
                "configured_target_attention_implementation": target_attention_implementation,
                "actual_cuda_kernel": None,
                "kernel_confirmation_requires_nsight": True,
            },
        }
        return finalize_generation_result(
            output_ids, diagnostics, return_diagnostics=return_diagnostics
        )

    def _spec_generate_reference(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
    ):
        """AST-equivalent algorithmic transcription of pinned ``spec_generate``."""

        self.eval()
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(
            output_ids.shape[1], device=target.device
        ).unsqueeze(0)

        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # Prefill stage
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        target_hidden = extract_context_feature(
            output.hidden_states, self.target_layer_ids
        )

        # Decode stage
        acceptance_lengths = []
        start = input_ids.shape[1]
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(
                self(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[
                        :, past_key_values_draft.get_seq_length() : start + block_size
                    ],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )[:, -block_size + 1 :, :]
            )
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)

            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )

            posterior = sample(output.logits, temperature)
            acceptance_length = (
                (block_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[
                :, acceptance_length
            ]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )[:, : acceptance_length + 1, :]
            acceptance_lengths.append(acceptance_length + 1)
            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break
        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_token_ids
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        return output_ids

    def _spec_generate_observed(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        *,
        return_diagnostics: bool,
        enable_profiling: bool,
        enable_rank_diagnostics: bool,
        rank_top_ms: list[int],
        rank_max_m: int,
        enable_verification_profiling: bool,
        enable_tree_capture: bool,
        tree_top_k: int,
    ):
        """Run the same algorithm with optional accounting, events, and NVTX."""

        self.eval()
        if enable_profiling and (
            not input_ids.is_cuda or not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "DFlash profiling requires CUDA tensors and CUDA PyTorch"
            )

        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(
            output_ids.shape[1], device=target.device
        ).unsqueeze(0)
        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        if enable_profiling:
            torch.cuda.reset_peak_memory_stats(target.device)
        timer = DeferredCudaTimer(torch, enabled=enable_profiling, device=target.device)
        attention_records = []
        attention_timing_pairs = []
        target_module_records = []
        round_timing_pairs = []
        rounds = []
        rank_events = []
        accepted_draft_lengths = []
        cache_commit_lengths = []
        total_pair = timer.start()

        with nvtx_range(torch, "dflash.total", enable_profiling):
            prefill_pair = timer.start()
            with (
                nvtx_range(torch, "dflash.prefill", enable_profiling),
                TargetModuleCudaProfiler(
                    torch,
                    timer=timer,
                    target=target,
                    stage="prefill",
                    round_index=None,
                    records=target_module_records,
                    enabled=enable_verification_profiling,
                ),
            ):
                output = target(
                    input_ids,
                    position_ids=position_ids[:, :num_input_tokens],
                    past_key_values=past_key_values_target,
                    use_cache=True,
                    logits_to_keep=1,
                    output_hidden_states=True,
                )
            timer.stop(prefill_pair)

            output_ids[:, :num_input_tokens] = input_ids
            output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
                output.logits, temperature
            )
            target_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )

            start = input_ids.shape[1]
            round_index = 0
            returned_tokens_before = min(1, max_new_tokens)
            while start < max_length:
                context_length_before = int(start)
                attention_record_start = len(attention_records)
                round_pair = timer.start()
                with nvtx_range(torch, "dflash.round", enable_profiling):
                    block_output_ids = output_ids[:, start : start + block_size].clone()
                    block_position_ids = position_ids[:, start : start + block_size]
                    noise_embedding = target.model.embed_tokens(block_output_ids)
                    draft_target_hidden_shape = tensor_shape(target_hidden)
                    draft_position_ids = position_ids[
                        :,
                        past_key_values_draft.get_seq_length() : start + block_size,
                    ]

                    draft_pair = timer.start()
                    with nvtx_range(torch, "dflash.draft", enable_profiling):
                        draft_hidden = self(
                            target_hidden=target_hidden,
                            noise_embedding=noise_embedding,
                            position_ids=draft_position_ids,
                            past_key_values=past_key_values_draft,
                            use_cache=True,
                            is_causal=False,
                            phase1_attention_records=attention_records,
                            phase1_round_index=round_index,
                            phase1_profiling=enable_profiling,
                            phase1_attention_timer=timer,
                            phase1_attention_timing_pairs=attention_timing_pairs,
                            phase1_context_length_before=context_length_before,
                        )
                        draft_logits = target.lm_head(
                            draft_hidden[:, -block_size + 1 :, :]
                        )
                        past_key_values_draft.crop(start)
                        block_output_ids[:, 1:] = sample(draft_logits)
                    timer.stop(draft_pair)

                    verify_pair = timer.start()
                    with (
                        nvtx_range(torch, "dflash.verify", enable_profiling),
                        TargetModuleCudaProfiler(
                            torch,
                            timer=timer,
                            target=target,
                            stage="verification",
                            round_index=round_index,
                            records=target_module_records,
                            enabled=enable_verification_profiling,
                        ),
                    ):
                        output = target(
                            block_output_ids,
                            position_ids=block_position_ids,
                            past_key_values=past_key_values_target,
                            use_cache=True,
                            output_hidden_states=True,
                        )
                    timer.stop(verify_pair)

                    commit_started_at = time.perf_counter()
                    with nvtx_range(torch, "dflash.commit", enable_profiling):
                        posterior = sample(output.logits, temperature)
                        accepted_draft_tokens = int(
                            (block_output_ids[:, 1:] == posterior[:, :-1])
                            .cumprod(dim=1)
                            .sum(dim=1)[0]
                            .item()
                        )
                        current_tree_capture = None
                        if enable_tree_capture:
                            current_tree_capture = build_tree_candidate_capture(
                                torch,
                                draft_logits=draft_logits,
                                block_output_ids=block_output_ids,
                                target_verification_token_ids=posterior[:, :-1],
                                round_index=round_index,
                                context_length_before=context_length_before,
                                prompt_length=num_input_tokens,
                                accepted_draft_tokens=accepted_draft_tokens,
                                top_k=tree_top_k,
                            )
                        if enable_rank_diagnostics:
                            current_rank_events = build_target_rank_events(
                                torch,
                                draft_logits=draft_logits,
                                target_verification_logits=output.logits,
                                round_index=round_index,
                                accepted_prefix_length=accepted_draft_tokens,
                                top_ms=rank_top_ms,
                                max_m=rank_max_m,
                            )
                            rank_events.extend(current_rank_events)
                        output_ids[:, start : start + accepted_draft_tokens + 1] = (
                            block_output_ids[:, : accepted_draft_tokens + 1]
                        )
                        output_ids[:, start + accepted_draft_tokens + 1] = posterior[
                            :, accepted_draft_tokens
                        ]
                        start += accepted_draft_tokens + 1
                        past_key_values_target.crop(start)
                        target_hidden = extract_context_feature(
                            output.hidden_states, self.target_layer_ids
                        )[:, : accepted_draft_tokens + 1, :]
                    commit_cpu_elapsed_ms = (
                        time.perf_counter() - commit_started_at
                    ) * 1000.0
                    accepted_draft_lengths.append(accepted_draft_tokens)
                    cache_commit_lengths.append(accepted_draft_tokens + 1)
                    stop_detected = stop_token_ids is not None and any(
                        stop_token_id in output_ids[:, num_input_tokens:]
                        for stop_token_id in stop_token_ids
                    )
                timer.stop(round_pair)

                eos_generated_count = None
                if stop_detected:
                    stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
                    stop_indices = torch.isin(
                        output_ids[0][num_input_tokens : start + 1], stop_tensor
                    ).nonzero(as_tuple=True)[0]
                    if stop_indices.numel() > 0:
                        eos_generated_count = int(stop_indices[0].item()) + 1

                accounting = build_round_accounting(
                    round_index=round_index,
                    context_length_before=context_length_before,
                    prompt_length=num_input_tokens,
                    max_new_tokens=max_new_tokens,
                    block_size=block_size,
                    accepted_draft_tokens=accepted_draft_tokens,
                    returned_tokens_before=returned_tokens_before,
                    eos_generated_count=eos_generated_count,
                )
                returned_tokens_before = accounting["cumulative_generated_token_count"]
                accounting.update(
                    {
                        "draft_input_shapes": {
                            "target_hidden": draft_target_hidden_shape,
                            "noise_embedding": tensor_shape(noise_embedding),
                            "position_ids": tensor_shape(draft_position_ids),
                        },
                        "draft_output_shapes": {
                            "hidden_states": tensor_shape(draft_hidden),
                            "logits": tensor_shape(draft_logits),
                        },
                        "target_verification_input_shapes": {
                            "input_ids": tensor_shape(block_output_ids),
                            "position_ids": tensor_shape(block_position_ids),
                        },
                        "target_verification_output_shapes": {
                            "logits": tensor_shape(output.logits),
                            "selected_context_features": tensor_shape(target_hidden),
                        },
                        "attention_calls": attention_records[attention_record_start:],
                        "timing": {
                            "draft_cpu_elapsed_ms": None,
                            "draft_cuda_elapsed_ms": None,
                            "verification_cpu_elapsed_ms": None,
                            "verification_cuda_elapsed_ms": None,
                            "combined_attention_cuda_elapsed_ms": None,
                            "combined_attention_over_draft_ratio": None,
                            "combined_attention_over_round_ratio": None,
                            "commit_cpu_elapsed_ms": commit_cpu_elapsed_ms,
                            "round_cpu_elapsed_ms": None,
                            "round_cuda_elapsed_ms": None,
                        },
                        "gpu_memory": gpu_memory_snapshot(torch, target.device)
                        if enable_profiling
                        else None,
                        "tree_candidate_capture": current_tree_capture,
                    }
                )
                rounds.append(accounting)
                round_timing_pairs.append(
                    (accounting, draft_pair, verify_pair, round_pair)
                )
                round_index += 1
                if stop_detected:
                    break

            output_ids = output_ids[:, :max_length]
            output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
            final_stop_reason = "max_new_tokens"
            if stop_token_ids is not None:
                stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
                stop_token_indices = torch.isin(
                    output_ids[0][num_input_tokens:], stop_tensor
                ).nonzero(as_tuple=True)[0]
                if stop_token_indices.numel() > 0:
                    output_ids = output_ids[
                        :, : num_input_tokens + stop_token_indices[0] + 1
                    ]
                    final_stop_reason = "eos"
        timer.stop(total_pair)
        timer.resolve()
        if enable_verification_profiling:
            resolve_target_module_records(timer, target_module_records)

        for attention_record, attention_pair in attention_timing_pairs:
            attention_record["combined_attention_cuda_elapsed_ms"] = timer.values(
                attention_pair
            )["cuda_elapsed_ms"]

        for accounting, draft_pair, verify_pair, round_pair in round_timing_pairs:
            draft_timing = timer.values(draft_pair)
            verify_timing = timer.values(verify_pair)
            round_timing = timer.values(round_pair)
            combined_attention_cuda_elapsed_ms = sum(
                record["combined_attention_cuda_elapsed_ms"] or 0.0
                for record in accounting["attention_calls"]
            )
            draft_cuda_elapsed_ms = draft_timing["cuda_elapsed_ms"]
            round_cuda_elapsed_ms = round_timing["cuda_elapsed_ms"]
            accounting["timing"].update(
                {
                    "draft_cpu_elapsed_ms": draft_timing["cpu_elapsed_ms"],
                    "draft_cuda_elapsed_ms": draft_timing["cuda_elapsed_ms"],
                    "verification_cpu_elapsed_ms": verify_timing["cpu_elapsed_ms"],
                    "verification_cuda_elapsed_ms": verify_timing["cuda_elapsed_ms"],
                    "round_cpu_elapsed_ms": round_timing["cpu_elapsed_ms"],
                    "round_cuda_elapsed_ms": round_timing["cuda_elapsed_ms"],
                    "combined_attention_cuda_elapsed_ms": (
                        combined_attention_cuda_elapsed_ms
                        if enable_profiling
                        else None
                    ),
                    "combined_attention_over_draft_ratio": (
                        combined_attention_cuda_elapsed_ms / draft_cuda_elapsed_ms
                        if draft_cuda_elapsed_ms
                        else None
                    ),
                    "combined_attention_over_round_ratio": (
                        combined_attention_cuda_elapsed_ms / round_cuda_elapsed_ms
                        if round_cuda_elapsed_ms
                        else None
                    ),
                }
            )
            if enable_verification_profiling:
                round_module_records = [
                    record
                    for record in target_module_records
                    if record["stage"] == "verification"
                    and record["round_index"] == accounting["round_index"]
                ]
                accounting["target_module_profile"] = {
                    "records": round_module_records,
                    "summary": summarize_target_module_records(
                        round_module_records,
                        total_cuda_ms=verify_timing["cuda_elapsed_ms"],
                        target_layer_count=len(target.model.layers),
                        invocation_count=1,
                    ),
                }
            else:
                accounting["target_module_profile"] = None

        combined_attention_total_ms = sum(
            record["combined_attention_cuda_elapsed_ms"] or 0.0
            for record in attention_records
        )
        draft_total_ms = sum(
            record["timing"]["draft_cuda_elapsed_ms"] or 0.0 for record in rounds
        )
        round_total_ms = sum(
            record["timing"]["round_cuda_elapsed_ms"] or 0.0 for record in rounds
        )
        verification_total_ms = sum(
            record["timing"]["verification_cuda_elapsed_ms"] or 0.0
            for record in rounds
        )
        if enable_verification_profiling:
            prefill_module_records = [
                record
                for record in target_module_records
                if record["stage"] == "prefill"
            ]
            verification_module_records = [
                record
                for record in target_module_records
                if record["stage"] == "verification"
            ]
            target_module_profiling = {
                "enabled": True,
                "timing_scope": "whole target modules; self-attention includes projections",
                "pure_sdpa_isolated": False,
                "prefill": {
                    "records": prefill_module_records,
                    "summary": summarize_target_module_records(
                        prefill_module_records,
                        total_cuda_ms=timer.values(prefill_pair)["cuda_elapsed_ms"],
                        target_layer_count=len(target.model.layers),
                        invocation_count=1,
                    ),
                },
                "verification": {
                    "records": verification_module_records,
                    "summary": summarize_target_module_records(
                        verification_module_records,
                        total_cuda_ms=verification_total_ms,
                        target_layer_count=len(target.model.layers),
                        invocation_count=len(rounds),
                    ),
                },
            }
        else:
            target_module_profiling = {
                "enabled": False,
                "timing_scope": "not measured",
                "pure_sdpa_isolated": False,
                "prefill": None,
                "verification": None,
            }

        diagnostics = {
            "schema_version": "1.0",
            "prompt_token_count": int(num_input_tokens),
            "generated_token_count": int(output_ids.shape[1] - num_input_tokens),
            "max_new_tokens": int(max_new_tokens),
            "draft_block_size": int(block_size),
            "prefill_seed_token_count": 1,
            "accepted_draft_token_lengths": accepted_draft_lengths,
            "cache_commit_lengths": cache_commit_lengths,
            "stop_reason": final_stop_reason,
            "rounds": rounds,
            "rank_diagnostics": {
                "enabled": enable_rank_diagnostics,
                "top_ms": rank_top_ms if enable_rank_diagnostics else [],
                "max_m": rank_max_m if enable_rank_diagnostics else None,
                "events": rank_events,
                "full_vocabulary_logits_saved": False,
                "target_bonus_token_ranked": False,
                "interpretation": "single-frontier recoverability only",
                "tree_accepted_length_derived": False,
            },
            "tree_candidate_capture": {
                "enabled": enable_tree_capture,
                "top_k": tree_top_k if enable_tree_capture else None,
                "round_count": len(rounds) if enable_tree_capture else 0,
                "logit_source": (
                    "exact final draft logits used by temperature-zero sampling"
                    if enable_tree_capture
                    else None
                ),
                "score_semantics": (
                    "normalized log probability from log_softmax"
                    if enable_tree_capture
                    else None
                ),
                "full_vocabulary_logits_saved": False,
                "target_logits_saved": False,
                "extra_draft_forward_calls": 0,
            },
            "target_module_profiling": target_module_profiling,
            "profiling": {
                "enabled": enable_profiling,
                "nvtx_ranges": [
                    "dflash.total",
                    "dflash.prefill",
                    "dflash.round",
                    "dflash.draft",
                    "dflash.verify",
                    "dflash.commit",
                    "dflash.attention.combined_context_block",
                    "target.prefill.layer.{i}.self_attn",
                    "target.prefill.layer.{i}.mlp",
                    "target.prefill.lm_head",
                    "target.verify.layer.{i}.self_attn",
                    "target.verify.layer.{i}.mlp",
                    "target.verify.lm_head",
                ],
                "explicit_cuda_synchronize_count": 1 if enable_profiling else 0,
                "prefill": timer.values(prefill_pair),
                "total": timer.values(total_pair),
                "combined_attention": {
                    "cuda_elapsed_ms": combined_attention_total_ms
                    if enable_profiling
                    else None,
                    "over_draft_ratio": (
                        combined_attention_total_ms / draft_total_ms
                        if draft_total_ms
                        else None
                    ),
                    "over_round_ratio": (
                        combined_attention_total_ms / round_total_ms
                        if round_total_ms
                        else None
                    ),
                    "call_count": len(attention_records),
                    "scope": "sum of per-layer combined SDPA calls",
                },
                "memory": gpu_memory_snapshot(torch, target.device)
                if enable_profiling
                else None,
            },
        }
        return finalize_generation_result(
            output_ids, diagnostics, return_diagnostics=return_diagnostics
        )
