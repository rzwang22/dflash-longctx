import hashlib
import json
import time
from itertools import pairwise
from typing import Callable, Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack

from .phase1_observability import (
    DeferredCudaTimer,
    TargetModuleCudaProfiler,
    build_round_accounting,
    build_target_rank_events,
    build_tree_candidate_capture,
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
from .phase2c_gold import (
    classify_comparison_matrix,
    expected_node_alignment,
    structural_alignment_issues,
    summarize_comparison_mode,
    validate_gold_modes,
    validate_gold_thresholds,
    validate_visibility_matrix,
)
from .phase2c_online import (
    build_online_tree_tensors,
    build_online_tree_tensors_from_nodes,
    build_tree_attention_mask,
    build_tree_attention_visibility,
    clone_dynamic_cache,
    compact_dynamic_cache,
    dynamic_cache_lengths,
    extract_topk_candidates,
    greedy_tree_traversal_tensors,
    layerwise_beam_tree,
    tensor_tree_metadata,
    tree_position_tensors,
    validate_online_tree_options,
)
from .phase2c_stabilization import (
    canonical_selected_path_decision,
    evenly_spaced_sample_indices,
    should_recheck,
    stabilization_commit_source,
    validate_decision_stabilization,
)
from .phase2d1_features import (
    aggregate_runtime_features,
    build_deployable_feature_rows,
    candidate_positions,
)
from .phase2d1_selector import select_policy, validate_frozen_selector
from .phase2d1_tree_policies import (
    confidence_gated_tree,
    score_beam_tree,
    top1_chain_tree,
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
        phase1_context_length_before = kwargs.pop("phase1_context_length_before", None)
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
                "cached_context_kv_length_before_injection": (cached_context_kv_length),
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
        online_tree_gold_comparison_modes: list[str] | None = None,
        online_tree_gold_layer_localization: bool = False,
        online_tree_gold_max_rounds: int = 1,
        online_tree_gold_record_all_nodes: bool = False,
        online_tree_gold_thresholds: dict[str, float] | None = None,
        enable_online_tree_divergence_diagnostics: bool = False,
        enable_reference_state_diagnostics: bool = False,
        reference_state_generated_prefixes: list[int] | None = None,
        decision_stabilization: str = "off",
        decision_margin_threshold: float = 0.1,
        online_selector_mode: str | None = None,
        online_selector_config: dict | None = None,
        online_selector_config_hash: str | None = None,
        online_selector_context_length: int | None = None,
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
        configured_gold_modes = validate_gold_modes(
            online_tree_gold_comparison_modes or ["native"]
        )
        configured_gold_thresholds = validate_gold_thresholds(
            online_tree_gold_thresholds
            or {
                "maximum_absolute_error": 0.5,
                "mean_absolute_error": 0.02,
                "relative_error_epsilon": 1.0e-6,
                "layer_localization_p99_error": 0.1,
            }
        )
        if online_tree_gold_max_rounds < 1:
            raise ValueError("online_tree_gold_max_rounds must be positive")
        configured_stabilization, configured_margin_threshold = (
            validate_decision_stabilization(
                decision_stabilization, decision_margin_threshold
            )
        )
        configured_selector = None
        if online_selector_mode is not None:
            if not enable_online_tree:
                raise ValueError("online selector requires enable_online_tree=True")
            if online_selector_config is None:
                raise ValueError("online selector requires a frozen selector config")
            configured_selector = validate_frozen_selector(online_selector_config)
            if configured_selector["config_sha256"] != online_selector_config_hash:
                raise ValueError("online selector expected config hash mismatch")
            if online_selector_context_length is None:
                raise ValueError("online selector requires nominal context length")
        if online_tree_gold_layer_localization and not enable_online_tree_gold_checks:
            raise ValueError("gold layer localization requires gold checks")
        if enable_online_tree_divergence_diagnostics and not (
            enable_online_tree and return_diagnostics
        ):
            raise ValueError(
                "tree divergence diagnostics require online tree diagnostics"
            )
        if enable_reference_state_diagnostics and (
            not return_diagnostics or enable_profiling
        ):
            raise ValueError(
                "reference-state diagnostics require an unprofiled diagnostic run"
            )
        configured_reference_state_prefixes = sorted(
            {int(value) for value in (reference_state_generated_prefixes or [])}
        )
        if any(value < 0 for value in configured_reference_state_prefixes):
            raise ValueError("reference-state generated prefixes cannot be negative")
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
                gold_comparison_modes=configured_gold_modes,
                gold_layer_localization=online_tree_gold_layer_localization,
                gold_max_rounds=online_tree_gold_max_rounds,
                gold_record_all_nodes=online_tree_gold_record_all_nodes,
                gold_thresholds=configured_gold_thresholds,
                enable_divergence_diagnostics=enable_online_tree_divergence_diagnostics,
                enable_reference_state_diagnostics=(
                    enable_reference_state_diagnostics
                ),
                reference_state_generated_prefixes=(
                    configured_reference_state_prefixes
                ),
                decision_stabilization=configured_stabilization,
                decision_margin_threshold=configured_margin_threshold,
                selector_mode=online_selector_mode,
                selector_config=configured_selector,
                selector_context_length=online_selector_context_length,
            )
        if not any(
            (
                return_diagnostics,
                enable_profiling,
                enable_rank_diagnostics,
                enable_verification_profiling,
                enable_tree_capture,
                enable_reference_state_diagnostics,
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
            enable_reference_state_diagnostics=enable_reference_state_diagnostics,
            reference_state_generated_prefixes=(
                configured_reference_state_prefixes
            ),
        )

    @staticmethod
    def _phase2c_compare_logits(
        torch_module,
        packed_logits,
        reference_logits,
        *,
        selected_token_id,
        top_k=8,
        relative_error_epsilon=1.0e-6,
    ):
        packed = packed_logits.float()
        reference = reference_logits.float()
        difference = (packed - reference).abs()
        relative = difference / reference.abs().clamp_min(relative_error_epsilon)
        packed_top = torch_module.argsort(packed, descending=True, stable=True)[:top_k]
        reference_top = torch_module.argsort(reference, descending=True, stable=True)[
            :top_k
        ]
        maximum_index = int(difference.argmax().item())
        packed_argmax = int(torch_module.argmax(packed).item())
        reference_argmax = int(torch_module.argmax(reference).item())
        selected_token_id = int(selected_token_id)
        packed_top_values = torch_module.topk(packed, k=2).values
        reference_top_values = torch_module.topk(reference, k=2).values
        packed_top_list = [int(value) for value in packed_top.detach().cpu().tolist()]
        reference_top_list = [
            int(value) for value in reference_top.detach().cpu().tolist()
        ]
        result = {
            "max_abs_error": float(difference[maximum_index].item()),
            "mean_abs_error": float(difference.mean().item()),
            "p50_abs_error": float(torch_module.quantile(difference, 0.50).item()),
            "p95_abs_error": float(torch_module.quantile(difference, 0.95).item()),
            "p99_abs_error": float(torch_module.quantile(difference, 0.99).item()),
            "max_relative_error": float(relative.max().item()),
            "mean_relative_error": float(relative.mean().item()),
            "relative_error_epsilon": float(relative_error_epsilon),
            "maximum_error_vocabulary_token_id": maximum_index,
            "packed_logit_at_maximum_error": float(packed[maximum_index].item()),
            "dense_logit_at_maximum_error": float(reference[maximum_index].item()),
            "packed_argmax_token_id": packed_argmax,
            "dense_argmax_token_id": reference_argmax,
            "argmax_equal": packed_argmax == reference_argmax,
            "top_k": int(top_k),
            "packed_top8_token_ids": packed_top_list,
            "dense_top8_token_ids": reference_top_list,
            "top8_set_overlap_count": len(
                set(packed_top_list) & set(reference_top_list)
            ),
            "top8_ordered_equal": packed_top_list == reference_top_list,
            "selected_token_id": selected_token_id,
            "selected_token_packed_rank": int(
                (packed > packed[selected_token_id]).sum().item()
            )
            + 1,
            "selected_token_dense_rank": int(
                (reference > reference[selected_token_id]).sum().item()
            )
            + 1,
            "packed_top1_top2_margin": float(
                (packed_top_values[0] - packed_top_values[1]).item()
            ),
            "dense_top1_top2_margin": float(
                (reference_top_values[0] - reference_top_values[1]).item()
            ),
            "verifier_decision_changed": packed_argmax != reference_argmax,
            "logit_element_count": int(difference.numel()),
        }
        return result

    @staticmethod
    def _phase2c_linear_attention_mask(*, query_length, prefix_length, dtype, device):
        causal = torch.tril(
            torch.ones((query_length, query_length), dtype=torch.bool, device=device)
        )
        visible = torch.cat(
            [
                torch.ones(
                    (query_length, prefix_length), dtype=torch.bool, device=device
                ),
                causal,
            ],
            dim=1,
        )
        mask = torch.full(
            (1, 1, query_length, prefix_length + query_length),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=device,
        )
        return mask.masked_fill(visible[None, None, :, :], 0.0)

    @staticmethod
    def _phase2c_math_sdpa_context():
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "torch.nn.attention.sdpa_kernel with SDPBackend.MATH is unavailable"
            ) from exc
        return sdpa_kernel(backends=[SDPBackend.MATH])

    @staticmethod
    def _phase2c_tensor_error_stats(left, right):
        difference = (left.float() - right.float()).abs().reshape(-1)
        return {
            "max_abs_error": float(difference.max().item()),
            "mean_abs_error": float(difference.mean().item()),
            "p99_abs_error": float(torch.quantile(difference, 0.99).item()),
            "element_count": int(difference.numel()),
        }

    def _phase2c_forward_with_layer_capture(self, target, **forward_kwargs):
        captures = {}
        handles = []

        def capture(name):
            def hook(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                captures[name] = value

            return hook

        handles.append(
            target.model.embed_tokens.register_forward_hook(capture("embedding"))
        )
        for index, layer in enumerate(target.model.layers):
            handles.append(
                layer.register_forward_hook(capture(f"target_layer_{index}"))
            )
        handles.append(target.model.norm.register_forward_hook(capture("final_norm")))
        try:
            output = target(**forward_kwargs)
        finally:
            for handle in handles:
                handle.remove()
        return output, captures

    def _phase2c_layer_localization(
        self,
        *,
        target,
        prefix_cache,
        tree,
        tree_mask,
        semantic_positions,
        cache_positions,
        context_length_before,
        node_indices,
        thresholds,
    ):
        threshold = thresholds["layer_localization_p99_error"]
        packed_output, packed_captures = self._phase2c_forward_with_layer_capture(
            target,
            input_ids=tree["token_ids"].unsqueeze(0),
            attention_mask=tree_mask,
            position_ids=semantic_positions,
            cache_position=cache_positions,
            past_key_values=clone_dynamic_cache(prefix_cache),
            use_cache=True,
        )
        root = tree["token_ids"][0].reshape(1)
        records = []
        for node_index in node_indices:
            depth = int(tree["depths"][node_index].item())
            path = tree["path_tokens"][node_index, :depth]
            linear_ids = torch.cat([root, path]).unsqueeze(0)
            positions = torch.arange(
                context_length_before,
                context_length_before + depth + 1,
                dtype=torch.long,
                device=linear_ids.device,
            )
            dense_mask = self._phase2c_linear_attention_mask(
                query_length=depth + 1,
                prefix_length=context_length_before,
                dtype=tree_mask.dtype,
                device=linear_ids.device,
            )
            dense_output, dense_captures = self._phase2c_forward_with_layer_capture(
                target,
                input_ids=linear_ids,
                attention_mask=dense_mask,
                position_ids=positions.unsqueeze(0),
                cache_position=positions,
                past_key_values=clone_dynamic_cache(prefix_cache),
                use_cache=True,
            )
            first_exceeding = None
            for stage, packed_hidden in packed_captures.items():
                stats = self._phase2c_tensor_error_stats(
                    packed_hidden[0, node_index], dense_captures[stage][0, -1]
                )
                exceeds = stats["p99_abs_error"] > threshold
                if exceeds and first_exceeding is None:
                    first_exceeding = stage
                records.append(
                    {
                        "availability": "available",
                        "status": "diagnostic",
                        "comparison_mode": "same_mask",
                        "backend": "configured_sdpa_same_mask",
                        "dtype": str(packed_hidden.dtype).removeprefix("torch."),
                        "thresholds": thresholds,
                        "flattened_node_index": int(node_index),
                        "node_depth": depth,
                        "stage": stage,
                        **stats,
                        "diagnostic_p99_threshold": float(threshold),
                        "exceeds_diagnostic_threshold": exceeds,
                        "is_first_stage_exceeding_threshold": exceeds
                        and first_exceeding == stage,
                        "first_stage_exceeding_threshold": first_exceeding,
                        "discrete_metric_applicable": False,
                        "warning": None,
                        "reason": None,
                        "provenance": {
                            "packed_source": "same-mask packed rerun with hooks",
                            "dense_source": "same-mask dense-path rerun with hooks",
                            "configured_attention_implementation": getattr(
                                target.config, "_attn_implementation", None
                            ),
                            "actual_cuda_kernel": None,
                        },
                    }
                )
            logits_stats = self._phase2c_tensor_error_stats(
                packed_output.logits[0, node_index], dense_output.logits[0, -1]
            )
            exceeds = logits_stats["p99_abs_error"] > threshold
            if exceeds and first_exceeding is None:
                first_exceeding = "lm_head_logits"
            records.append(
                {
                    "availability": "available",
                    "status": "diagnostic",
                    "comparison_mode": "same_mask",
                    "backend": "configured_sdpa_same_mask",
                    "dtype": str(packed_output.logits.dtype).removeprefix("torch."),
                    "thresholds": thresholds,
                    "flattened_node_index": int(node_index),
                    "node_depth": depth,
                    "stage": "lm_head_logits",
                    **logits_stats,
                    "diagnostic_p99_threshold": float(threshold),
                    "exceeds_diagnostic_threshold": exceeds,
                    "is_first_stage_exceeding_threshold": exceeds
                    and first_exceeding == "lm_head_logits",
                    "first_stage_exceeding_threshold": first_exceeding,
                    "discrete_metric_applicable": True,
                    "warning": None,
                    "reason": None,
                    "provenance": {
                        "packed_source": "same-mask packed rerun with hooks",
                        "dense_source": "same-mask dense-path rerun with hooks",
                        "configured_attention_implementation": getattr(
                            target.config, "_attn_implementation", None
                        ),
                        "actual_cuda_kernel": None,
                    },
                }
            )
        return records

    def _phase2c_packed_logits_gold_check(
        self,
        *,
        target,
        prefix_cache,
        tree,
        packed_logits,
        tree_mask,
        semantic_positions,
        cache_positions,
        context_length_before,
        comparison_modes,
        thresholds,
        verifier_node_indices,
        verifier_next_root_token_id,
        packed_temporary_layer_lengths,
        record_all_nodes,
        layer_localization,
    ):
        metadata = tensor_tree_metadata(tree)
        nodes = metadata["nodes"]
        node_indices = (
            list(range(tree["actual_total_node_count"]))
            if record_all_nodes
            else sorted({int(value) for value in verifier_node_indices})
        )
        verifier_set = {int(value) for value in verifier_node_indices}
        verifier_path = [int(value) for value in verifier_node_indices]
        verifier_decisions = {}
        for offset, node_index in enumerate(verifier_path):
            if offset + 1 < len(verifier_path):
                child_index = verifier_path[offset + 1]
                verifier_decisions[node_index] = {
                    "kind": "accepted_child",
                    "selected_token_id": int(tree["token_ids"][child_index].item()),
                    "selected_child_node_index": child_index,
                }
            else:
                verifier_decisions[node_index] = {
                    "kind": "correction_or_bonus",
                    "selected_token_id": int(verifier_next_root_token_id.item()),
                    "selected_child_node_index": None,
                }
        actual_visibility = (tree_mask[0, 0] == 0).detach().cpu().tolist()
        expected_visibility = build_tree_attention_visibility(
            [node["parent_index"] for node in nodes],
            prefix_length=context_length_before,
        )
        shared_structural_issues = validate_visibility_matrix(
            actual_visibility, expected_visibility
        )
        if any(
            decision["selected_child_node_index"] is not None
            and int(
                tree["parent_indices"][decision["selected_child_node_index"]].item()
            )
            != node_index
            for node_index, decision in verifier_decisions.items()
        ):
            shared_structural_issues.append("verifier_path_parent_child_alignment")
        if dynamic_cache_lengths(prefix_cache) and any(
            length != context_length_before
            for length in dynamic_cache_lengths(prefix_cache)
        ):
            shared_structural_issues.append("persistent_prefix_cache_length")
        if semantic_positions[0].detach().cpu().tolist() != [
            context_length_before + node["depth"] for node in nodes
        ]:
            shared_structural_issues.append("semantic_position_mapping")
        if cache_positions.detach().cpu().tolist() != list(
            range(
                context_length_before,
                context_length_before + tree["actual_total_node_count"],
            )
        ):
            shared_structural_issues.append("packed_cache_position_mapping")
        expected_packed_kv_length = (
            context_length_before + tree["actual_total_node_count"]
        )
        if not packed_temporary_layer_lengths or any(
            length != expected_packed_kv_length
            for length in packed_temporary_layer_lengths
        ):
            shared_structural_issues.append("packed_actual_kv_length")

        all_records = []
        mode_summaries = []
        for mode in comparison_modes:
            mode_records = []
            backend = {
                "native": "configured_sdpa_packed_4d_dense_implicit_causal",
                "equal_shape": "configured_sdpa_equal_shape_explicit_4d",
                "same_mask": "configured_sdpa_explicit_4d_both_sides",
                "math": "forced_math_sdpa_explicit_4d_both_sides",
            }[mode]
            unavailable_reason = None
            try:
                if mode == "equal_shape" and tree["policy"] != "top1_chain":
                    raise RuntimeError(
                        "equal_shape control is defined only for top1_chain"
                    )
                packed_mode_logits = packed_logits
                context = self._phase2c_math_sdpa_context() if mode == "math" else None
                if context is not None:
                    with context:
                        packed_mode_output = target(
                            tree["token_ids"].unsqueeze(0),
                            attention_mask=tree_mask,
                            position_ids=semantic_positions,
                            cache_position=cache_positions,
                            past_key_values=clone_dynamic_cache(prefix_cache),
                            use_cache=True,
                        )
                        packed_mode_logits = packed_mode_output.logits
                        mode_records = self._phase2c_dense_mode_records(
                            target=target,
                            prefix_cache=prefix_cache,
                            tree=tree,
                            nodes=nodes,
                            packed_logits=packed_mode_logits,
                            context_length_before=context_length_before,
                            node_indices=node_indices,
                            verifier_set=verifier_set,
                            verifier_decisions=verifier_decisions,
                            mode=mode,
                            thresholds=thresholds,
                            shared_structural_issues=shared_structural_issues,
                            packed_temporary_layer_lengths=packed_temporary_layer_lengths,
                        )
                else:
                    mode_records = self._phase2c_dense_mode_records(
                        target=target,
                        prefix_cache=prefix_cache,
                        tree=tree,
                        nodes=nodes,
                        packed_logits=packed_mode_logits,
                        context_length_before=context_length_before,
                        node_indices=node_indices,
                        verifier_set=verifier_set,
                        verifier_decisions=verifier_decisions,
                        mode=mode,
                        thresholds=thresholds,
                        shared_structural_issues=shared_structural_issues,
                        packed_temporary_layer_lengths=packed_temporary_layer_lengths,
                    )
            except (ImportError, RuntimeError, AttributeError) as exc:
                unavailable_reason = f"{type(exc).__name__}: {exc}"
                mode_records = [
                    {
                        "availability": "unavailable",
                        "status": "not_run",
                        "comparison_mode": mode,
                        "backend": backend,
                        "dtype": str(packed_logits.dtype).removeprefix("torch."),
                        "structural_status": "not_run",
                        "semantic_status": "not_run",
                        "numerical_status": "not_run",
                        "thresholds": thresholds,
                        "warning": None,
                        "reason": unavailable_reason,
                        "flattened_node_index": None,
                        "node_depth": None,
                        "max_abs_error": None,
                        "mean_abs_error": None,
                        "p50_abs_error": None,
                        "p95_abs_error": None,
                        "p99_abs_error": None,
                        "max_relative_error": None,
                        "mean_relative_error": None,
                        "maximum_error_vocabulary_token_id": None,
                        "packed_logit_at_maximum_error": None,
                        "dense_logit_at_maximum_error": None,
                        "packed_argmax_token_id": None,
                        "dense_argmax_token_id": None,
                        "argmax_equal": None,
                        "packed_top8_token_ids": None,
                        "dense_top8_token_ids": None,
                        "top8_set_overlap_count": None,
                        "top8_ordered_equal": None,
                        "selected_token_packed_rank": None,
                        "selected_token_dense_rank": None,
                        "packed_top1_top2_margin": None,
                        "dense_top1_top2_margin": None,
                        "verifier_decision_changed": None,
                        "verifier_decision_relevant": None,
                        "provenance": {
                            "packed_source": "target packed-tree forward",
                            "dense_source": "independent target reference forward",
                            "actual_cuda_kernel": None,
                        },
                    }
                ]
            summary = summarize_comparison_mode(
                mode=mode,
                backend=backend,
                dtype=str(packed_logits.dtype).removeprefix("torch."),
                nodes=mode_records,
                structural_issues=shared_structural_issues,
                thresholds=thresholds,
                unavailable_reason=unavailable_reason,
            )
            for record in mode_records:
                record["status"] = summary["status"]
                if record.get("warning") is None:
                    record["warning"] = summary.get("warning")
                record["mode_summary"] = summary
            all_records.extend(mode_records)
            mode_summaries.append(summary)
        matrix = classify_comparison_matrix(
            mode_summaries,
            output_ids_equal=True,
            round_semantics_equal=True,
        )
        for record in all_records:
            record["matrix_status"] = matrix["status"]
            record["matrix_failure_class"] = matrix["failure_class"]
            record["classification_evidence_modes"] = matrix[
                "classification_evidence_modes"
            ]
        localization_records = []
        if layer_localization:
            try:
                localization_records = self._phase2c_layer_localization(
                    target=target,
                    prefix_cache=prefix_cache,
                    tree=tree,
                    tree_mask=tree_mask,
                    semantic_positions=semantic_positions,
                    cache_positions=cache_positions,
                    context_length_before=context_length_before,
                    node_indices=node_indices,
                    thresholds=thresholds,
                )
            except (RuntimeError, AttributeError) as exc:
                localization_records = [
                    {
                        "availability": "unavailable",
                        "status": "not_run",
                        "comparison_mode": "same_mask",
                        "backend": "configured_sdpa_explicit_4d_both_sides",
                        "dtype": str(packed_logits.dtype).removeprefix("torch."),
                        "thresholds": thresholds,
                        "flattened_node_index": None,
                        "node_depth": None,
                        "stage": None,
                        "max_abs_error": None,
                        "mean_abs_error": None,
                        "p99_abs_error": None,
                        "diagnostic_p99_threshold": thresholds[
                            "layer_localization_p99_error"
                        ],
                        "exceeds_diagnostic_threshold": None,
                        "is_first_stage_exceeding_threshold": None,
                        "first_stage_exceeding_threshold": None,
                        "discrete_metric_applicable": None,
                        "warning": None,
                        "reason": f"{type(exc).__name__}: {exc}",
                        "provenance": {
                            "packed_source": "unavailable",
                            "dense_source": "unavailable",
                            "configured_attention_implementation": getattr(
                                target.config, "_attn_implementation", None
                            ),
                            "actual_cuda_kernel": None,
                        },
                    }
                ]
        return {
            **matrix,
            "availability": "available",
            "comparison_modes": list(comparison_modes),
            "mode_summaries": mode_summaries,
            "records": all_records,
            "layer_localization_records": localization_records,
            "completed_checks": [
                "mask_visibility",
                "position_mapping",
                "cache_position_mapping",
                "logit_query_alignment",
                "logit_comparison_matrix",
            ]
            + (
                ["layer_localization"]
                if localization_records
                and all(
                    record["availability"] == "available"
                    for record in localization_records
                )
                else []
            ),
            "uncompleted_checks": (
                ["layer_localization"]
                if layer_localization
                and any(
                    record["availability"] == "unavailable"
                    for record in localization_records
                )
                else []
            ),
        }

    def _phase2c_dense_mode_records(
        self,
        *,
        target,
        prefix_cache,
        tree,
        nodes,
        packed_logits,
        context_length_before,
        node_indices,
        verifier_set,
        verifier_decisions,
        mode,
        thresholds,
        shared_structural_issues,
        packed_temporary_layer_lengths,
    ):
        root = tree["token_ids"][0].reshape(1)
        equal_shape_output = None
        if mode == "equal_shape":
            query_length = tree["actual_total_node_count"]
            equal_mask = self._phase2c_linear_attention_mask(
                query_length=query_length,
                prefix_length=context_length_before,
                dtype=target.model.embed_tokens.weight.dtype,
                device=root.device,
            )
            positions = torch.arange(
                context_length_before,
                context_length_before + query_length,
                dtype=torch.long,
                device=root.device,
            )
            equal_shape_output = target(
                tree["token_ids"].unsqueeze(0),
                attention_mask=equal_mask,
                position_ids=positions.unsqueeze(0),
                cache_position=positions,
                past_key_values=clone_dynamic_cache(prefix_cache),
                use_cache=True,
            )
        records = []
        for node_index in node_indices:
            node = nodes[node_index]
            alignment = expected_node_alignment(
                node=node,
                persistent_prefix_length=context_length_before,
                packed_query_length=tree["actual_total_node_count"],
            )
            depth = int(node["depth"])
            if equal_shape_output is not None:
                reference_logits = equal_shape_output.logits[0, node_index]
                dense_actual_lengths = dynamic_cache_lengths(
                    equal_shape_output.past_key_values
                )
                alignment.update(
                    {
                        "dense_reference_query_length": tree["actual_total_node_count"],
                        "dense_reference_kv_length": context_length_before
                        + tree["actual_total_node_count"],
                        "dense_reference_position_ids": list(
                            range(
                                context_length_before,
                                context_length_before + tree["actual_total_node_count"],
                            )
                        ),
                        "dense_reference_cache_positions": list(
                            range(
                                context_length_before,
                                context_length_before + tree["actual_total_node_count"],
                            )
                        ),
                        "dense_logits_query_index": node_index,
                    }
                )
            else:
                path = tree["path_tokens"][node_index, :depth]
                linear_ids = torch.cat([root, path]).unsqueeze(0)
                positions = torch.arange(
                    context_length_before,
                    context_length_before + depth + 1,
                    dtype=torch.long,
                    device=linear_ids.device,
                )
                attention_mask = None
                if mode in {"same_mask", "math"}:
                    attention_mask = self._phase2c_linear_attention_mask(
                        query_length=depth + 1,
                        prefix_length=context_length_before,
                        dtype=target.model.embed_tokens.weight.dtype,
                        device=linear_ids.device,
                    )
                linear_output = target(
                    linear_ids,
                    attention_mask=attention_mask,
                    position_ids=positions.unsqueeze(0),
                    cache_position=positions,
                    past_key_values=clone_dynamic_cache(prefix_cache),
                    use_cache=True,
                    logits_to_keep=1,
                )
                reference_logits = linear_output.logits[0, -1]
                dense_actual_lengths = dynamic_cache_lengths(
                    linear_output.past_key_values
                )
            decision = verifier_decisions.get(node_index)
            selected_token = (
                decision["selected_token_id"]
                if decision is not None
                else int(torch.argmax(packed_logits[0, node_index]).item())
            )
            structural_issues = structural_alignment_issues(alignment)
            if mode == "equal_shape":
                # Equal-shape intentionally changes the dense query-index and
                # full reference position vectors while preserving the compared
                # node's semantic position.
                structural_issues = [
                    issue
                    for issue in structural_issues
                    if issue
                    not in {
                        "dense_logits_query_index",
                        "dense_semantic_positions",
                        "dense_cache_positions",
                    }
                ]
            if not dense_actual_lengths or any(
                length != alignment["dense_reference_kv_length"]
                for length in dense_actual_lengths
            ):
                structural_issues.append("dense_actual_kv_length")
            comparison = self._phase2c_compare_logits(
                torch,
                packed_logits[0, node_index],
                reference_logits,
                selected_token_id=selected_token,
                relative_error_epsilon=thresholds["relative_error_epsilon"],
            )
            records.append(
                {
                    "availability": "available",
                    "status": "diagnostic",
                    "comparison_mode": mode,
                    "backend": {
                        "native": "configured_sdpa_packed_4d_dense_implicit_causal",
                        "equal_shape": "configured_sdpa_equal_shape_explicit_4d",
                        "same_mask": "configured_sdpa_explicit_4d_both_sides",
                        "math": "forced_math_sdpa_explicit_4d_both_sides",
                    }[mode],
                    "dtype": str(packed_logits.dtype).removeprefix("torch."),
                    "structural_status": "fail"
                    if structural_issues or shared_structural_issues
                    else "pass",
                    "semantic_status": "pass" if comparison["argmax_equal"] else "fail",
                    "numerical_status": "within_strict_thresholds"
                    if comparison["max_abs_error"]
                    <= thresholds["maximum_absolute_error"]
                    and comparison["mean_abs_error"]
                    <= thresholds["mean_absolute_error"]
                    else "outside_strict_thresholds",
                    "thresholds": thresholds,
                    "warning": None,
                    "reason": None,
                    **alignment,
                    "packed_actual_layer_kv_lengths": list(
                        packed_temporary_layer_lengths
                    ),
                    "dense_reference_actual_layer_kv_lengths": dense_actual_lengths,
                    "persistent_cache_snapshot_semantics": (
                        "both sides deep-copy the same pre-packed-forward cache"
                    ),
                    "structural_issues": sorted(
                        set(structural_issues + shared_structural_issues)
                    ),
                    "verifier_decision_relevant": node_index in verifier_set,
                    "verifier_decision_kind": (
                        decision["kind"] if decision is not None else "not_on_path"
                    ),
                    "verifier_selected_child_node_index": (
                        decision["selected_child_node_index"]
                        if decision is not None
                        else None
                    ),
                    **comparison,
                    "provenance": {
                        "packed_source": "target packed-tree forward",
                        "dense_source": (
                            "equal-shape linear explicit-mask target forward"
                            if mode == "equal_shape"
                            else "per-path target forward"
                        ),
                        "configured_attention_implementation": getattr(
                            target.config, "_attn_implementation", None
                        ),
                        "actual_cuda_kernel": None,
                    },
                }
            )
        return records

    def _phase2c_kv_gold_check(
        self,
        *,
        target,
        compacted_cache,
        output_ids,
        next_root_index,
        compaction,
        tree,
        thresholds,
    ):
        committed_dense_ids = output_ids[:, :next_root_index]
        committed_dense_output = target(
            committed_dense_ids,
            position_ids=torch.arange(
                committed_dense_ids.shape[1],
                dtype=torch.long,
                device=committed_dense_ids.device,
            ).unsqueeze(0),
            cache_position=torch.arange(
                committed_dense_ids.shape[1],
                dtype=torch.long,
                device=committed_dense_ids.device,
            ),
            past_key_values=DynamicCache(),
            use_cache=True,
            logits_to_keep=1,
        )
        dense_committed_cache = committed_dense_output.past_key_values
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
            selected_token_id=int(torch.argmax(compacted_output.logits[0, -1]).item()),
            relative_error_epsilon=thresholds["relative_error_epsilon"],
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
        canonical_source = (
            compaction.get("committed_cache_source") == "canonical_recheck"
        )
        sibling_nodes_discarded = (
            compaction["discarded_tree_node_count"] >= 0
            and expected_compacted < compaction["temporary_layer_lengths"][0]
            if compaction["discarded_tree_node_count"]
            else True
        )
        committed_indices = [
            int(value)
            for value in compaction["committed_node_indices"].detach().cpu().tolist()
        ]
        expected_gather = (
            list(range(next_root_index))
            if canonical_source
            else [
                *range(next_root_index - len(committed_indices)),
                *(
                    next_root_index - len(committed_indices) + index
                    for index in committed_indices
                ),
            ]
        )
        actual_gather = [
            int(value) for value in compaction["gather_indices"].detach().cpu().tolist()
        ]
        committed_path_is_valid = bool(committed_indices) and committed_indices[0] == 0
        committed_path_is_valid = committed_path_is_valid and all(
            int(tree["parent_indices"][child].item()) == parent
            for parent, child in pairwise(committed_indices)
        )
        layer_count_matches = (
            len(compacted_cache.layers) == len(dense_committed_cache.layers)
            and len(compacted_cache.layers) > 0
        )
        structural_status = (
            all(length_checks.values())
            and sibling_nodes_discarded
            and not compaction["correction_or_bonus_written"]
            and actual_gather == expected_gather
            and committed_path_is_valid
            and layer_count_matches
        )
        layer_records = []
        for layer_index, (packed_layer, dense_layer) in enumerate(
            zip(compacted_cache.layers, dense_committed_cache.layers)
        ):
            layer_records.append(
                {
                    "layer_index": layer_index,
                    "cache_sequence_length": int(packed_layer.get_seq_length()),
                    "expected_committed_length": next_root_index,
                    "k": self._phase2c_tensor_error_stats(
                        packed_layer.keys, dense_layer.keys
                    ),
                    "v": self._phase2c_tensor_error_stats(
                        packed_layer.values, dense_layer.values
                    ),
                }
            )
        numerical_exact = (
            comparison["max_abs_error"] <= thresholds["maximum_absolute_error"]
            and comparison["mean_abs_error"] <= thresholds["mean_absolute_error"]
            and all(
                record[kind]["max_abs_error"] <= thresholds["maximum_absolute_error"]
                and record[kind]["mean_abs_error"] <= thresholds["mean_absolute_error"]
                for record in layer_records
                for kind in ("k", "v")
            )
        )
        status = (
            "fail_structural"
            if not structural_status
            else "fail_semantic"
            if not comparison["argmax_equal"]
            else "pass_exact"
            if numerical_exact and comparison["top8_ordered_equal"]
            else "pass_semantic_with_numerical_variation"
        )
        result = {
            "availability": "available",
            "status": status,
            "comparison_mode": "dense_committed_path",
            "backend": "configured_sdpa",
            "dtype": str(compacted_output.logits.dtype).removeprefix("torch."),
            "structural_status": "pass" if structural_status else "fail",
            "semantic_status": "pass" if comparison["argmax_equal"] else "fail",
            "numerical_status": "within_strict_thresholds"
            if numerical_exact
            else "outside_strict_thresholds",
            "thresholds": thresholds,
            "warning": None,
            "reason": None,
            **comparison,
            "compacted_layer_lengths": compacted_lengths,
            "next_step_layer_lengths": next_step_lengths,
            "dense_reference_layer_lengths": dense_lengths,
            "length_checks": length_checks,
            "sibling_nodes_discarded": sibling_nodes_discarded,
            "persistent_prefix_fully_retained": actual_gather[
                : next_root_index - len(committed_indices)
            ]
            == list(range(next_root_index - len(committed_indices))),
            "root_and_accepted_path_order_valid": committed_path_is_valid,
            "target_layer_count_matches_dense_reference": layer_count_matches,
            "correction_or_bonus_absent_from_persistent_cache": not compaction[
                "correction_or_bonus_written"
            ],
            "selected_flattened_node_indices": committed_indices,
            "selected_storage_indices": actual_gather,
            "expected_storage_indices": expected_gather,
            "retained_token_ids": [
                int(tree["token_ids"][index].item()) for index in committed_indices
            ],
            "retained_depths": [
                int(tree["depths"][index].item()) for index in committed_indices
            ],
            "discarded_sibling_or_descendant_count": int(
                compaction["discarded_tree_node_count"]
            ),
            "layers": layer_records,
            "provenance": {
                "compacted_source": (
                    "canonical linear selected-path cache crop"
                    if canonical_source
                    else "path-aware gathered packed-tree cache"
                ),
                "dense_source": "full committed token sequence recomputation",
                "configured_attention_implementation": getattr(
                    target.config, "_attn_implementation", None
                ),
                "actual_cuda_kernel": None,
            },
        }
        return result

    @staticmethod
    def _phase2c_tensor_comparable_summary(tensor):
        numel = int(tensor.numel())
        sample_indices = evenly_spaced_sample_indices(numel)
        if not sample_indices:
            return {
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": [int(value) for value in tensor.shape],
                "numel": 0,
                "sample_indices": [],
                "sample_values": [],
                "sample_sum": 0.0,
            }
        if tensor.ndim == 0:
            sampled = tensor.unsqueeze(0)
        else:
            coordinates = [[] for _ in tensor.shape]
            for flat_index in sample_indices:
                remainder = flat_index
                current = [0] * tensor.ndim
                for dimension in range(tensor.ndim - 1, -1, -1):
                    current[dimension] = remainder % int(tensor.shape[dimension])
                    remainder //= int(tensor.shape[dimension])
                for dimension, coordinate in enumerate(current):
                    coordinates[dimension].append(coordinate)
            sampled = tensor[
                tuple(
                    torch.tensor(
                        dimension_coordinates,
                        dtype=torch.long,
                        device=tensor.device,
                    )
                    for dimension_coordinates in coordinates
                )
            ]
        values = sampled.float().detach().cpu().tolist()
        return {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(value) for value in tensor.shape],
            "numel": numel,
            "sample_indices": sample_indices,
            "sample_values": [float(value) for value in values],
            "sample_sum": float(sum(values)),
        }

    @classmethod
    def _phase2c_cache_comparable_summary(cls, cache, sequence_length=None):
        requested_length = (
            None if sequence_length is None else int(sequence_length)
        )

        records = []
        for layer_index, layer in enumerate(cache.layers):
            actual_length = int(layer.get_seq_length())
            if (
                requested_length is not None
                and not 0 <= requested_length <= actual_length
            ):
                raise RuntimeError(
                    "reference-state checkpoint length is outside the cache"
                )
            key = (
                layer.keys
                if requested_length is None
                else layer.keys[..., :requested_length, :]
            )
            value = (
                layer.values
                if requested_length is None
                else layer.values[..., :requested_length, :]
            )
            records.append(
                {
                    "layer_index": layer_index,
                    "sequence_length": int(key.shape[-2]),
                    "actual_cache_sequence_length": actual_length,
                    "key_shape": [int(item) for item in key.shape],
                    "value_shape": [int(item) for item in value.shape],
                    "key_comparable_sample": cls._phase2c_tensor_comparable_summary(
                        key
                    ),
                    "value_comparable_sample": cls._phase2c_tensor_comparable_summary(
                        value
                    ),
                }
            )
        return records

    @staticmethod
    def _phase2c_logits_comparable_summary(logits, top_k=8):
        flattened = logits.float().reshape(-1)
        count = min(int(top_k), int(flattened.numel()))
        values, indices = torch.topk(flattened, k=count)
        margin = (
            float((values[0] - values[1]).item()) if count >= 2 else None
        )
        return {
            "dtype": str(logits.dtype).removeprefix("torch."),
            "shape": [int(value) for value in logits.shape],
            "top_k": count,
            "top_token_ids": [int(value) for value in indices.detach().cpu().tolist()],
            "top_values": [float(value) for value in values.detach().cpu().tolist()],
            "top1_token_id": int(indices[0].item()) if count else None,
            "top1_top2_margin": margin,
            "full_vocabulary_logits_saved": False,
        }

    @classmethod
    def _phase2c_draft_logits_comparable_summary(cls, logits):
        return {
            "shape": [int(value) for value in logits.shape],
            "positions": [
                {
                    "draft_position": index + 1,
                    **cls._phase2c_logits_comparable_summary(logits[0, index]),
                }
                for index in range(int(logits.shape[1]))
            ],
        }

    @classmethod
    def _phase2c_update_reference_state_checkpoint(
        cls,
        records,
        *,
        execution_path,
        provenance,
        generated_prefix_length,
        prompt_length,
        output_ids,
        target_cache=None,
        target_hidden=None,
        target_hidden_provenance=None,
        semantic_probe_logits=None,
        semantic_probe_provenance=None,
        updates=None,
    ):
        prefix = int(generated_prefix_length)
        absolute_length = int(prompt_length) + prefix
        prefix_ids = [
            int(value)
            for value in output_ids[0, :absolute_length].detach().cpu().tolist()
        ]
        digest = hashlib.sha256(
            json.dumps(prefix_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        record = records.setdefault(
            prefix,
            {
                "generated_prefix_length": prefix,
                "absolute_sequence_prefix_length": absolute_length,
                "execution_path": execution_path,
                "checkpoint_provenance": provenance,
                "semantic_prefix_token_ids_sha256": digest,
                "semantic_prefix_tail_token_ids": prefix_ids[-16:],
                "persistent_target_kv": None,
                "target_hidden": None,
                "target_hidden_provenance": "not_directly_comparable",
                "semantic_probe": None,
                "semantic_probe_provenance": "not_directly_comparable",
                "draft_state": None,
                "draft_state_provenance": "not_directly_comparable",
                "round_entry": None,
                "commit_transition": None,
            },
        )
        if target_cache is not None:
            record["persistent_target_kv"] = cls._phase2c_cache_comparable_summary(
                target_cache, sequence_length=absolute_length
            )
        if target_hidden is not None:
            record["target_hidden"] = cls._phase2c_tensor_comparable_summary(
                target_hidden
            )
            record["target_hidden_provenance"] = target_hidden_provenance
        if semantic_probe_logits is not None:
            record["semantic_probe"] = cls._phase2c_logits_comparable_summary(
                semantic_probe_logits
            )
            record["semantic_probe_provenance"] = semantic_probe_provenance
        if updates:
            record.update(updates)
        return record

    def _phase2c_tree_traversal_record(
        self,
        *,
        round_index,
        tree,
        tree_mask,
        semantic_positions,
        cache_positions,
        packed_logits,
        traversal,
        compaction,
        context_length_before,
        context_length_after,
        output_ids,
        expected_round_root_token_id,
        expected_round_root_position,
        cache_before_summary,
        cache_after_summary,
    ):
        metadata = tensor_tree_metadata(tree)
        nodes = metadata["nodes"]
        children = {index: [] for index in range(len(nodes))}
        for node in nodes[1:]:
            children[node["parent_index"]].append(node["node_index"])
        top8 = torch.topk(packed_logits[0].float(), k=8, dim=-1).indices
        top2_values = torch.topk(packed_logits[0].float(), k=2, dim=-1).values
        target_tokens = traversal["target_greedy_token_ids"]
        accepted = [
            int(value)
            for value in traversal["accepted_node_indices"].detach().cpu().tolist()
        ]
        committed = [0, *accepted]
        terminal_parent = int(traversal["terminal_parent_index"].item())
        node_records = []
        for node in nodes:
            index = node["node_index"]
            argmax_token = int(target_tokens[index].item())
            matching_children = [
                child
                for child in children[index]
                if int(tree["token_ids"][child].item()) == argmax_token
            ]
            selected_child = matching_children[0] if matching_children else None
            node_records.append(
                {
                    **node,
                    "children_indices": children[index],
                    "semantic_position": int(semantic_positions[0, index].item()),
                    "packed_storage_position": int(cache_positions[index].item()),
                    "target_logits_row_index": index,
                    "target_argmax_token_id": argmax_token,
                    "target_top8_token_ids": [
                        int(value) for value in top8[index].detach().cpu().tolist()
                    ],
                    "target_top1_top2_margin": float(
                        (top2_values[index, 0] - top2_values[index, 1]).item()
                    ),
                    "selected_child_index": selected_child,
                    "selected_child_token_id": (
                        int(tree["token_ids"][selected_child].item())
                        if selected_child is not None
                        else None
                    ),
                    "on_accepted_path": index in committed,
                }
            )
        expected_visibility = build_tree_attention_visibility(
            [node["parent_index"] for node in nodes],
            prefix_length=context_length_before,
        )
        actual_visibility = (tree_mask[0, 0] == 0).detach().cpu().tolist()
        mask_issues = validate_visibility_matrix(actual_visibility, expected_visibility)
        transition_checks = []
        parent = 0
        for child in accepted:
            transition_checks.append(
                {
                    "parent_index": parent,
                    "child_index": child,
                    "parent_argmax_token_id": int(target_tokens[parent].item()),
                    "child_token_id": int(tree["token_ids"][child].item()),
                    "parent_relationship_valid": int(
                        tree["parent_indices"][child].item()
                    )
                    == parent,
                    "parent_logit_predicts_child": int(target_tokens[parent].item())
                    == int(tree["token_ids"][child].item()),
                }
            )
            parent = child
        next_root = int(traversal["next_root_token_id"].item())
        next_root_position = context_length_after
        expected_gather = [
            *range(context_length_before),
            *(context_length_before + index for index in committed),
        ]
        sibling_tokens_unique = all(
            len({nodes[child]["token_id"] for child in node_children})
            == len(node_children)
            for node_children in children.values()
        )
        checks = {
            "root_excluded_from_accepted_draft_count": 0 not in accepted,
            "accepted_count_matches_path": len(accepted)
            == int(traversal["accepted_draft_token_count"]),
            "accepted_path_prefix_closed": all(
                row["parent_relationship_valid"] for row in transition_checks
            ),
            "parent_logits_predict_each_child": all(
                row["parent_logit_predicts_child"] for row in transition_checks
            ),
            "terminal_parent_matches_path": terminal_parent == parent,
            "correction_or_bonus_uses_terminal_parent_logits": next_root
            == int(target_tokens[terminal_parent].item()),
            "correction_or_bonus_kind_matches_terminal_children": bool(
                traversal["next_root_is_bonus"].item()
            )
            == (not children[terminal_parent]),
            "duplicate_sibling_tokens_absent": sibling_tokens_unique,
            "committed_nodes_are_root_plus_accepted_path": [
                int(value)
                for value in compaction["committed_node_indices"]
                .detach()
                .cpu()
                .tolist()
            ]
            == committed,
            "correction_or_bonus_not_in_current_kv": not compaction[
                "correction_or_bonus_written"
            ],
            "kv_gather_discards_unaccepted_siblings": [
                int(value)
                for value in compaction["gather_indices"].detach().cpu().tolist()
            ]
            == expected_gather,
            "next_root_matches_output_buffer": int(
                output_ids[0, next_root_position].item()
            )
            == next_root,
            "round_root_matches_previous_target": int(tree["token_ids"][0].item())
            == int(expected_round_root_token_id),
            "round_root_absolute_position_matches": context_length_before
            == int(expected_round_root_position),
            "round_root_appended_exactly_once": all(
                row["sequence_length"] == context_length_before
                for row in cache_before_summary
            ),
            "semantic_positions_follow_depth": semantic_positions[0]
            .detach()
            .cpu()
            .tolist()
            == [context_length_before + node["depth"] for node in nodes],
            "packed_storage_positions_are_contiguous": cache_positions.detach()
            .cpu()
            .tolist()
            == list(
                range(
                    context_length_before,
                    context_length_before + tree["actual_total_node_count"],
                )
            ),
        }
        structural_names = {
            "semantic_positions_follow_depth",
            "packed_storage_positions_are_contiguous",
        }
        traversal_names = {
            "root_excluded_from_accepted_draft_count",
            "accepted_count_matches_path",
            "accepted_path_prefix_closed",
            "parent_logits_predict_each_child",
            "terminal_parent_matches_path",
            "correction_or_bonus_uses_terminal_parent_logits",
            "correction_or_bonus_kind_matches_terminal_children",
            "duplicate_sibling_tokens_absent",
            "committed_nodes_are_root_plus_accepted_path",
        }
        kv_root_names = {
            "correction_or_bonus_not_in_current_kv",
            "kv_gather_discards_unaccepted_siblings",
            "next_root_matches_output_buffer",
            "round_root_matches_previous_target",
            "round_root_absolute_position_matches",
            "round_root_appended_exactly_once",
        }
        checks["structural_checks_passed"] = not mask_issues and all(
            checks[name] for name in structural_names
        )
        checks["traversal_checks_passed"] = all(
            checks[name] for name in traversal_names
        )
        checks["kv_root_transition_checks_passed"] = all(
            checks[name] for name in kv_root_names
        )
        checks["all_checks_passed"] = all(checks.values()) and not mask_issues
        return {
            "schema_version": "1.0",
            "round_index": int(round_index),
            "status": "pass" if checks["all_checks_passed"] else "fail",
            "root_token_id": int(tree["token_ids"][0].item()),
            "nodes": node_records,
            "accepted_path_node_indices": accepted,
            "accepted_depth": len(accepted),
            "terminal_parent_index": terminal_parent,
            "correction_or_bonus_token_id": next_root,
            "correction_or_bonus_kind": (
                "bonus"
                if bool(traversal["next_root_is_bonus"].item())
                else "correction"
            ),
            "committed_node_indices": committed,
            "kv_gather_indices": [
                int(value)
                for value in compaction["gather_indices"].detach().cpu().tolist()
            ],
            "next_round_root_token_id": next_root,
            "next_round_root_absolute_position": next_root_position,
            "context_length_before": int(context_length_before),
            "context_length_after": int(context_length_after),
            "persistent_kv_length_before": int(context_length_before),
            "persistent_kv_length_after": int(context_length_after),
            "transition_checks": transition_checks,
            "structural_mask_issues": mask_issues,
            "allowed_attention_positions_by_node": [
                [index for index, visible in enumerate(row) if visible]
                for row in actual_visibility
            ],
            "checks": checks,
            "cache_before_comparable_summary": cache_before_summary,
            "cache_after_comparable_summary": cache_after_summary,
        }

    def _phase2c_canonical_selected_path_recheck(
        self,
        *,
        target,
        prefix_cache,
        tree,
        native_traversal,
        context_length_before,
        block_size,
    ):
        """Recheck the native-selected path with the original linear block shape."""

        native_indices = native_traversal["accepted_node_indices"]
        native_tokens = tree["token_ids"].index_select(0, native_indices)
        linear_ids = torch.full(
            (1, block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=tree["token_ids"].device,
        )
        linear_ids[0, 0] = tree["token_ids"][0]
        if int(native_tokens.shape[0]):
            linear_ids[0, 1 : 1 + int(native_tokens.shape[0])] = native_tokens
        positions = torch.arange(
            context_length_before,
            context_length_before + block_size,
            dtype=torch.long,
            device=linear_ids.device,
        ).unsqueeze(0)
        canonical_output = target(
            linear_ids,
            position_ids=positions,
            past_key_values=prefix_cache,
            use_cache=True,
            output_hidden_states=True,
        )
        canonical_argmax = torch.argmax(canonical_output.logits[0], dim=-1)
        decision = canonical_selected_path_decision(
            selected_draft_token_ids=native_tokens.detach().cpu().tolist(),
            canonical_parent_argmax_token_ids=canonical_argmax.detach().cpu().tolist(),
            native_next_root_token_id=int(
                native_traversal["next_root_token_id"].item()
            ),
            native_next_root_is_bonus=bool(
                native_traversal["next_root_is_bonus"].item()
            ),
        )
        accepted_count = int(decision["accepted_draft_token_count"])
        accepted_indices = native_indices[:accepted_count]
        committed_node_indices = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=torch.long,
                    device=tree["token_ids"].device,
                ),
                accepted_indices,
            ]
        )
        committed_tokens = tree["token_ids"].index_select(
            0, committed_node_indices
        )
        temporary_lengths = dynamic_cache_lengths(canonical_output.past_key_values)
        expected_length = context_length_before + 1 + accepted_count
        hidden_all = extract_context_feature(
            canonical_output.hidden_states, self.target_layer_ids
        )
        target_hidden = hidden_all[:, : 1 + accepted_count, :]
        compaction = {
            "committed_cache_source": "canonical_recheck",
            "prefix_length": context_length_before,
            "temporary_layer_lengths": temporary_lengths,
            "compacted_layer_lengths": None,
            "expected_compacted_length": expected_length,
            "committed_node_indices": committed_node_indices,
            "gather_indices": torch.arange(
                expected_length, dtype=torch.long, device=linear_ids.device
            ),
            "discarded_tree_node_count": max(
                0, int(tree["actual_total_node_count"]) - 1 - accepted_count
            ),
            "discarded_canonical_tail_count": block_size - 1 - accepted_count,
            "correction_or_bonus_written": False,
            "canonical_linear_query_length": block_size,
            "canonical_cache_length": expected_length,
        }
        return {
            **decision,
            "accepted_node_indices": accepted_indices,
            "committed_node_indices": committed_node_indices,
            "committed_tokens": committed_tokens,
            "committed_token_count": int(committed_tokens.shape[0]),
            "past_key_values": canonical_output.past_key_values,
            "target_hidden": target_hidden,
            "compaction": compaction,
            "canonical_parent_argmax_token_ids": canonical_argmax[
                : int(native_tokens.shape[0]) + 1
            ],
            "next_root_token_tensor": canonical_argmax[accepted_count],
            "next_root_is_bonus_tensor": torch.tensor(
                decision["next_root_is_bonus"],
                dtype=torch.bool,
                device=linear_ids.device,
            ),
            "input_token_ids": linear_ids,
            "query_length": block_size,
            "round_entry_kv_length": context_length_before,
            "effective_kv_length": context_length_before + block_size,
        }

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
        gold_comparison_modes: tuple[str, ...],
        gold_layer_localization: bool,
        gold_max_rounds: int,
        gold_record_all_nodes: bool,
        gold_thresholds: dict[str, float],
        enable_divergence_diagnostics: bool,
        enable_reference_state_diagnostics: bool,
        reference_state_generated_prefixes: list[int],
        decision_stabilization: str,
        decision_margin_threshold: float,
        selector_mode: str | None,
        selector_config: dict | None,
        selector_context_length: int | None,
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
        gold_layer_localization_records = []
        gold_round_results = []
        tree_traversal_checks = []
        requested_state_prefixes = set(reference_state_generated_prefixes)
        reference_state_checkpoints = {}
        if selector_mode is not None and selector_config is None:
            raise RuntimeError("online selector config disappeared after validation")

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
            if enable_reference_state_diagnostics and 0 in requested_state_prefixes:
                self._phase2c_update_reference_state_checkpoint(
                    reference_state_checkpoints,
                    execution_path="layerwise_tree_shadow",
                    provenance="diagnostic_replay",
                    generated_prefix_length=0,
                    prompt_length=num_input_tokens,
                    output_ids=output_ids,
                    target_cache=past_key_values_target,
                    target_hidden=target_hidden,
                    target_hidden_provenance="actual_tree_target_hidden_used_by_draft",
                    semantic_probe_logits=output.logits[0, -1],
                    semantic_probe_provenance="actual_tree_prefill_next_token_logits",
                )

            start = num_input_tokens
            round_index = 0
            returned_tokens_before = min(1, max_new_tokens)
            expected_round_root_token_id = int(output_ids[0, start].item())
            expected_round_root_position = start
            while start < max_length:
                context_length_before = int(start)
                if past_key_values_target.get_seq_length() != context_length_before:
                    raise RuntimeError(
                        "Phase 2C target cache does not equal round prefix length"
                    )
                generated_prefix_before = context_length_before - num_input_tokens
                if (
                    enable_reference_state_diagnostics
                    and generated_prefix_before in requested_state_prefixes
                ):
                    self._phase2c_update_reference_state_checkpoint(
                        reference_state_checkpoints,
                        execution_path="layerwise_tree_shadow",
                        provenance="diagnostic_replay",
                        generated_prefix_length=generated_prefix_before,
                        prompt_length=num_input_tokens,
                        output_ids=output_ids,
                        target_cache=past_key_values_target,
                        updates={
                            "target_hidden_used_by_draft": (
                                self._phase2c_tensor_comparable_summary(target_hidden)
                            ),
                            "target_hidden_used_by_draft_provenance": (
                                "actual_tree_target_hidden_used_by_draft"
                            ),
                            "round_entry": {
                                "round_index": round_index,
                                "generated_root_offset": generated_prefix_before,
                                "root_token_id": int(output_ids[0, start].item()),
                                "persistent_target_kv_length": context_length_before,
                                "draft_kv_layer_lengths_before": dynamic_cache_lengths(
                                    past_key_values_draft
                                ),
                            }
                        },
                    )
                round_pair = timer.start()
                with nvtx_range(torch, "phase2c.round", enable_profiling):
                    block_output_ids = output_ids[:, start : start + block_size].clone()
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
                    if (
                        enable_reference_state_diagnostics
                        and generated_prefix_before in requested_state_prefixes
                    ):
                        checkpoint = reference_state_checkpoints[
                            generated_prefix_before
                        ]
                        checkpoint["draft_state"] = {
                            "cache": self._phase2c_cache_comparable_summary(
                                past_key_values_draft
                            ),
                            "cache_layer_lengths": dynamic_cache_lengths(
                                past_key_values_draft
                            ),
                            "logits": self._phase2c_draft_logits_comparable_summary(
                                draft_logits
                            ),
                            "candidate_top1_token_ids": [
                                int(value)
                                for value in candidates["token_ids"][:, 0]
                                .detach()
                                .cpu()
                                .tolist()
                            ],
                            "candidate_topk_token_ids": [
                                [int(value) for value in row]
                                for row in candidates["token_ids"]
                                .detach()
                                .cpu()
                                .tolist()
                            ],
                            "candidate_tree_input": {
                                "policy": tree_policy,
                                "node_budget": node_budget,
                                "beam_width": beam_width,
                            },
                        }
                        checkpoint["draft_state_provenance"] = (
                            "actual_tree_draft_round_entry"
                        )
                    # This equivalence assertion is diagnostic-only.  Avoid the
                    # redundant argmax/comparison on the formal control path.
                    top1_matches_sampling = None
                    if return_diagnostics:
                        baseline_ids = sample(draft_logits, temperature)
                        top1_matches_sampling = (
                            baseline_ids[0] == candidates["token_ids"][:, 0]
                        ).all()

                    selector_record = None
                    selected_nodes = None
                    effective_tree_policy = tree_policy
                    effective_beam_width = beam_width
                    selector_cpu_timing = {
                        "draft_topk_device_to_host_sync_cpu_ms": 0.0,
                        "feature_extraction_cpu_ms": 0.0,
                        "decision_cpu_ms": 0.0,
                        "canonical_policy_construction_cpu_ms": 0.0,
                        "tree_tensor_pack_host_to_device_cpu_ms": 0.0,
                        "device_to_host_synchronization_occurred": False,
                        "host_to_device_transfer_occurred": False,
                    }
                    if selector_mode is not None:
                        selector_runtime_features = None
                        selector_feature_rows = []
                        if selector_mode != "decision_stump":
                            decision_started = time.perf_counter()
                            selector_record = select_policy(
                                selector_config,
                                mode=selector_mode,
                                node_budget=node_budget,
                                context_length=selector_context_length,
                                runtime_features=None,
                            )
                            selector_cpu_timing["decision_cpu_ms"] = (
                                time.perf_counter() - decision_started
                            ) * 1000.0
                        sync_started = time.perf_counter()
                        selector_token_ids = candidates["token_ids"].detach().cpu().tolist()
                        selector_log_probabilities = (
                            candidates["log_probabilities"].detach().cpu().tolist()
                        )
                        selector_cpu_timing[
                            "draft_topk_device_to_host_sync_cpu_ms"
                        ] = (time.perf_counter() - sync_started) * 1000.0
                        selector_cpu_timing[
                            "device_to_host_synchronization_occurred"
                        ] = bool(candidates["token_ids"].is_cuda)
                        if selector_mode == "decision_stump":
                            feature_started = time.perf_counter()
                            selector_positions = candidate_positions(
                                selector_token_ids, selector_log_probabilities
                            )
                            selector_canonical_nodes = score_beam_tree(
                                selector_positions, budget=node_budget
                            )
                            selector_feature_rows = build_deployable_feature_rows(
                                positions=selector_positions,
                                canonical_nodes=selector_canonical_nodes,
                                context_length=selector_context_length,
                                top_m=draft_top_k,
                                node_budget=node_budget,
                            )
                            selector_runtime_features = aggregate_runtime_features(
                                selector_feature_rows,
                                context_length=selector_context_length,
                                node_budget=node_budget,
                                policy_configuration_count=len(
                                    selector_config["node_budget_rules"][
                                        str(node_budget)
                                    ]["allowed_policy_configurations"]
                                ),
                            )
                            selector_cpu_timing["feature_extraction_cpu_ms"] = (
                                time.perf_counter() - feature_started
                            ) * 1000.0
                            decision_started = time.perf_counter()
                            selector_record = select_policy(
                                selector_config,
                                mode=selector_mode,
                                node_budget=node_budget,
                                context_length=selector_context_length,
                                runtime_features=selector_runtime_features,
                            )
                            selector_cpu_timing["decision_cpu_ms"] = (
                                time.perf_counter() - decision_started
                            ) * 1000.0
                        selected_policy = selector_record["selected_policy"]
                        effective_tree_policy = selected_policy["policy"]
                        policy_parameters = selected_policy["policy_parameters"]
                        effective_beam_width = int(policy_parameters.get("beam_width", 1))
                        construction_started = time.perf_counter()
                        if selector_mode != "decision_stump":
                            selector_positions = candidate_positions(
                                selector_token_ids, selector_log_probabilities
                            )
                        if effective_tree_policy == "top1_chain":
                            selected_nodes = top1_chain_tree(
                                selector_positions, budget=node_budget
                            )
                        elif effective_tree_policy == "score_beam":
                            selected_nodes = score_beam_tree(
                                selector_positions, budget=node_budget
                            )
                        elif effective_tree_policy == "confidence_gated":
                            selected_nodes = confidence_gated_tree(
                                selector_positions,
                                budget=node_budget,
                                margin_threshold=float(
                                    policy_parameters["margin_threshold"]
                                ),
                                branch_width=int(policy_parameters["branch_width"]),
                            )
                        elif effective_tree_policy == "layerwise_beam":
                            selected_nodes = layerwise_beam_tree(
                                selector_positions,
                                budget=node_budget,
                                beam_width=effective_beam_width,
                            )
                        else:
                            raise RuntimeError(
                                f"frozen selector chose unsupported policy {effective_tree_policy}"
                            )
                        selector_cpu_timing[
                            "canonical_policy_construction_cpu_ms"
                        ] = (time.perf_counter() - construction_started) * 1000.0

                    tree_build_pair = timer.start()
                    with nvtx_range(torch, "phase2c.tree_build", enable_profiling):
                        if selector_mode is None:
                            tree = build_online_tree_tensors(
                                torch,
                                candidates=candidates,
                                root_token_id=block_output_ids[0, 0],
                                policy=tree_policy,
                                node_budget=node_budget,
                                beam_width=beam_width,
                            )
                        else:
                            pack_started = time.perf_counter()
                            tree = build_online_tree_tensors_from_nodes(
                                torch,
                                positions=selector_positions,
                                nodes=selected_nodes,
                                root_token_id=block_output_ids[0, 0],
                                policy=effective_tree_policy,
                                node_budget=node_budget,
                                beam_width=effective_beam_width,
                                device=target.device,
                            )
                            selector_cpu_timing[
                                "tree_tensor_pack_host_to_device_cpu_ms"
                            ] = (time.perf_counter() - pack_started) * 1000.0
                            selector_cpu_timing[
                                "host_to_device_transfer_occurred"
                            ] = bool(input_ids.is_cuda)
                    timer.stop(tree_build_pair)
                    if selector_record is not None:
                        selector_record["runtime_features"] = selector_runtime_features
                        selector_record["feature_rows"] = selector_feature_rows
                        selector_record["cpu_timing"] = selector_cpu_timing
                        selector_record["actual_constructed_node_count"] = int(
                            tree["actual_total_node_count"]
                        )
                        selector_record["node_budget_cap"] = node_budget
                        selector_record["selected_policy_matches_tree"] = (
                            tree["policy"] == effective_tree_policy
                        )
                        selector_record["target_forward_count_added"] = 0
                    if (
                        enable_reference_state_diagnostics
                        and generated_prefix_before in requested_state_prefixes
                    ):
                        tree_input = reference_state_checkpoints[
                            generated_prefix_before
                        ]["draft_state"]["candidate_tree_input"]
                        tree_input.update(
                            {
                                "actual_total_node_count": int(
                                    tree["actual_total_node_count"]
                                ),
                                "flattened_token_ids": [
                                    int(value)
                                    for value in tree["token_ids"]
                                    .detach()
                                    .cpu()
                                    .tolist()
                                ],
                                "parent_indices": [
                                    int(value)
                                    for value in tree["parent_indices"]
                                    .detach()
                                    .cpu()
                                    .tolist()
                                ],
                                "depths": [
                                    int(value)
                                    for value in tree["depths"]
                                    .detach()
                                    .cpu()
                                    .tolist()
                                ],
                            }
                        )

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
                        if enable_gold_checks and round_index < gold_max_rounds
                        else None
                    )
                    snapshot_pair = None
                    stabilization_prefix_cache = None
                    if decision_stabilization != "off":
                        snapshot_pair = timer.start()
                        with nvtx_range(
                            torch, "phase2c.stabilization_cache_snapshot", enable_profiling
                        ):
                            stabilization_prefix_cache = clone_dynamic_cache(
                                past_key_values_target
                            )
                        timer.stop(snapshot_pair)
                    record_divergence_round = (
                        enable_divergence_diagnostics and round_index < gold_max_rounds
                    )
                    divergence_cache_before = (
                        self._phase2c_cache_comparable_summary(past_key_values_target)
                        if record_divergence_round
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
                    packed_temporary_layer_lengths = dynamic_cache_lengths(
                        past_key_values_target
                    )

                    traverse_pair = timer.start()
                    with nvtx_range(torch, "phase2c.traverse", enable_profiling):
                        traversal = greedy_tree_traversal_tensors(
                            torch, tree=tree, target_logits=packed_output.logits
                        )
                    timer.stop(traverse_pair)
                    native_accepted_draft_tokens = traversal[
                        "accepted_draft_token_count"
                    ]
                    native_committed_node_indices = torch.cat(
                        [
                            torch.zeros(
                                1,
                                dtype=torch.long,
                                device=tree["token_ids"].device,
                            ),
                            traversal["accepted_node_indices"],
                        ]
                    )
                    native_committed_tokens = tree["token_ids"].index_select(
                        0, native_committed_node_indices
                    )
                    native_path_decision_indices = torch.cat(
                        [
                            torch.zeros(
                                1,
                                dtype=torch.long,
                                device=tree["token_ids"].device,
                            ),
                            traversal["accepted_node_indices"],
                        ]
                    )
                    path_margins_tensor = None
                    path_margins = None
                    minimum_path_margin = None
                    if return_diagnostics or decision_stabilization == "margin":
                        path_logits = packed_output.logits[0].index_select(
                            0, native_path_decision_indices
                        )
                        path_top2 = torch.topk(path_logits.float(), k=2, dim=-1).values
                        path_margins_tensor = path_top2[:, 0] - path_top2[:, 1]
                        path_margins = [
                            float(value)
                            for value in path_margins_tensor.detach().cpu().tolist()
                        ]
                        minimum_path_margin = min(path_margins)

                    recheck_triggered = should_recheck(
                        decision_stabilization,
                        minimum_path_margin,
                        decision_margin_threshold,
                    )
                    trigger_reason = (
                        "mode_always"
                        if decision_stabilization == "always"
                        else "shadow_oracle"
                        if decision_stabilization == "shadow"
                        else "minimum_path_margin_at_or_below_threshold"
                        if recheck_triggered
                        else "minimum_path_margin_above_threshold"
                        if decision_stabilization == "margin"
                        else "disabled"
                    )
                    canonical_pair = None
                    canonical = None
                    if recheck_triggered:
                        canonical_pair = timer.start()
                        with nvtx_range(
                            torch, "phase2c.canonical_recheck", enable_profiling
                        ):
                            canonical = self._phase2c_canonical_selected_path_recheck(
                                target=target,
                                prefix_cache=stabilization_prefix_cache,
                                tree=tree,
                                native_traversal=traversal,
                                context_length_before=context_length_before,
                                block_size=block_size,
                            )
                        timer.stop(canonical_pair)

                    selected_commit_source = stabilization_commit_source(
                        decision_stabilization, recheck_triggered
                    )
                    use_canonical_commit = bool(
                        canonical is not None
                        and selected_commit_source == "canonical_recheck"
                    )
                    compact_pair = None
                    kv_replacement_pair = None
                    if use_canonical_commit:
                        kv_replacement_pair = timer.start()
                        with nvtx_range(
                            torch, "phase2c.canonical_kv_commit", enable_profiling
                        ):
                            canonical["past_key_values"].crop(
                                canonical["compaction"]["expected_compacted_length"]
                            )
                            compacted_lengths = dynamic_cache_lengths(
                                canonical["past_key_values"]
                            )
                            if not compacted_lengths or any(
                                length
                                != canonical["compaction"][
                                    "expected_compacted_length"
                                ]
                                for length in compacted_lengths
                            ):
                                raise RuntimeError(
                                    "canonical recheck cache crop produced a wrong length"
                                )
                            canonical["compaction"][
                                "compacted_layer_lengths"
                            ] = compacted_lengths
                            past_key_values_target = canonical["past_key_values"]
                            compaction = canonical["compaction"]
                            committed_node_indices = canonical[
                                "committed_node_indices"
                            ]
                            committed_tokens = canonical["committed_tokens"]
                            committed_token_count = canonical[
                                "committed_token_count"
                            ]
                            accepted_draft_tokens = canonical[
                                "accepted_draft_token_count"
                            ]
                            next_root_token_id = canonical["next_root_token_id"]
                            next_root_is_bonus = canonical["next_root_is_bonus"]
                            target_hidden = canonical["target_hidden"]
                        timer.stop(kv_replacement_pair)
                    else:
                        committed_node_indices = native_committed_node_indices
                        committed_tokens = native_committed_tokens
                        committed_token_count = int(committed_tokens.shape[0])
                        accepted_draft_tokens = native_accepted_draft_tokens
                        next_root_token_id = traversal["next_root_token_id"]
                        next_root_is_bonus = traversal["next_root_is_bonus"]
                        compact_pair = timer.start()
                        with nvtx_range(torch, "phase2c.kv_compact", enable_profiling):
                            compaction = compact_dynamic_cache(
                                torch,
                                cache=past_key_values_target,
                                prefix_length=context_length_before,
                                actual_tree_node_count=tree["actual_total_node_count"],
                                accepted_node_indices=traversal[
                                    "accepted_node_indices"
                                ],
                            )
                            compaction["committed_cache_source"] = "packed_native"
                        timer.stop(compact_pair)

                    output_ids[:, start : start + committed_token_count] = (
                        committed_tokens.unsqueeze(0)
                    )
                    output_ids[:, start + committed_token_count] = next_root_token_id
                    effective_traversal = traversal
                    if use_canonical_commit:
                        effective_traversal = {
                            **traversal,
                            "accepted_node_indices": canonical[
                                "accepted_node_indices"
                            ],
                            "accepted_draft_token_count": accepted_draft_tokens,
                            "next_root_token_id": canonical[
                                "next_root_token_tensor"
                            ],
                            "next_root_is_bonus": canonical[
                                "next_root_is_bonus_tensor"
                            ],
                            "terminal_parent_index": committed_node_indices[-1],
                        }
                    timer.stop(total_verification_pair)

                    start += committed_token_count
                    if not use_canonical_commit:
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
                    if enable_reference_state_diagnostics:
                        committed_node_list = [
                            int(value)
                            for value in committed_node_indices.detach().cpu().tolist()
                        ]
                        for committed_offset, node_index in enumerate(
                            committed_node_list, start=1
                        ):
                            generated_prefix = (
                                generated_prefix_before + committed_offset
                            )
                            if generated_prefix not in requested_state_prefixes:
                                continue
                            self._phase2c_update_reference_state_checkpoint(
                                reference_state_checkpoints,
                                execution_path="layerwise_tree_shadow",
                                provenance="diagnostic_replay",
                                generated_prefix_length=generated_prefix,
                                prompt_length=num_input_tokens,
                                output_ids=output_ids,
                                target_cache=past_key_values_target,
                                target_hidden=target_hidden[
                                    :, committed_offset - 1 : committed_offset, :
                                ],
                                target_hidden_provenance=(
                                    "actual_tree_committed_token_hidden"
                                ),
                                semantic_probe_logits=(
                                    packed_output.logits[0, node_index]
                                    if not use_canonical_commit
                                    else None
                                ),
                                semantic_probe_provenance=(
                                    "actual_packed_tree_verifier_logits"
                                    if not use_canonical_commit
                                    else None
                                ),
                                updates={
                                    "tree_round_index": round_index,
                                    "committed_token_kind": (
                                        "root"
                                        if committed_offset == 1
                                        else "accepted_draft"
                                    ),
                                    "committed_token_id": int(
                                        committed_tokens[committed_offset - 1].item()
                                    ),
                                    "packed_tree_node_index": node_index,
                                },
                            )
                        generated_prefix_after = start - num_input_tokens
                        if generated_prefix_after in requested_state_prefixes:
                            checkpoint = reference_state_checkpoints[
                                generated_prefix_after
                            ]
                            checkpoint["commit_transition"] = {
                                "round_index": round_index,
                                "generated_prefix_before": generated_prefix_before,
                                "generated_prefix_after": generated_prefix_after,
                                "accepted_draft_token_count": int(
                                    accepted_draft_tokens
                                ),
                                "committed_token_ids": [
                                    int(value)
                                    for value in committed_tokens.detach()
                                    .cpu()
                                    .tolist()
                                ],
                                "next_root_token_id": int(
                                    next_root_token_id.item()
                                    if hasattr(next_root_token_id, "item")
                                    else next_root_token_id
                                ),
                                "next_root_kind": (
                                    "bonus"
                                    if bool(
                                        next_root_is_bonus.item()
                                        if hasattr(next_root_is_bonus, "item")
                                        else next_root_is_bonus
                                    )
                                    else "correction"
                                ),
                                "committed_cache_source": compaction.get(
                                    "committed_cache_source"
                                ),
                                "persistent_target_kv_length_after": start,
                            }

                    if record_divergence_round and not use_canonical_commit:
                        divergence_cache_after = self._phase2c_cache_comparable_summary(
                            past_key_values_target
                        )
                        tree_traversal_checks.append(
                            self._phase2c_tree_traversal_record(
                                round_index=round_index,
                                tree=tree,
                                tree_mask=tree_mask,
                                semantic_positions=semantic_positions,
                                cache_positions=cache_positions,
                                packed_logits=packed_output.logits,
                                traversal=effective_traversal,
                                compaction=compaction,
                                context_length_before=context_length_before,
                                context_length_after=start,
                                output_ids=output_ids,
                                expected_round_root_token_id=expected_round_root_token_id,
                                expected_round_root_position=expected_round_root_position,
                                cache_before_summary=divergence_cache_before,
                                cache_after_summary=divergence_cache_after,
                            )
                        )
                    if record_divergence_round:
                        expected_round_root_token_id = int(
                            effective_traversal["next_root_token_id"].item()
                        )
                        expected_round_root_position = start

                    if gold_prefix_cache is not None:
                        packed_gold = self._phase2c_packed_logits_gold_check(
                            target=target,
                            prefix_cache=gold_prefix_cache,
                            tree=tree,
                            packed_logits=packed_output.logits,
                            tree_mask=tree_mask,
                            semantic_positions=semantic_positions,
                            cache_positions=cache_positions,
                            context_length_before=context_length_before,
                            comparison_modes=gold_comparison_modes,
                            thresholds=gold_thresholds,
                            verifier_node_indices=native_committed_node_indices,
                            verifier_next_root_token_id=traversal["next_root_token_id"],
                            packed_temporary_layer_lengths=packed_temporary_layer_lengths,
                            record_all_nodes=gold_record_all_nodes,
                            layer_localization=gold_layer_localization,
                        )
                        for record in packed_gold.pop("records"):
                            record["round_index"] = round_index
                            record["decision_stabilization"] = decision_stabilization
                            record["canonical_commit_replaced_native"] = (
                                use_canonical_commit
                            )
                            packed_gold_records.append(record)
                        for record in packed_gold.pop("layer_localization_records"):
                            record["round_index"] = round_index
                            gold_layer_localization_records.append(record)
                        if packed_gold["all_checks_passed"] or use_canonical_commit:
                            try:
                                kv_gold = self._phase2c_kv_gold_check(
                                    target=target,
                                    compacted_cache=past_key_values_target,
                                    output_ids=output_ids,
                                    next_root_index=start,
                                    compaction=compaction,
                                    tree=tree,
                                    thresholds=gold_thresholds,
                                )
                            except (RuntimeError, AttributeError) as exc:
                                kv_gold = {
                                    "availability": "unavailable",
                                    "status": "not_run",
                                    "execution_status": "not_run_due_to_execution_error",
                                    "comparison_mode": "dense_committed_path",
                                    "backend": "configured_sdpa",
                                    "dtype": str(
                                        packed_output.logits.dtype
                                    ).removeprefix("torch."),
                                    "structural_status": "not_run",
                                    "semantic_status": "not_run",
                                    "numerical_status": "not_run",
                                    "thresholds": gold_thresholds,
                                    "warning": None,
                                    "reason": f"{type(exc).__name__}: {exc}",
                                    "max_abs_error": None,
                                    "mean_abs_error": None,
                                    "p50_abs_error": None,
                                    "p95_abs_error": None,
                                    "p99_abs_error": None,
                                    "argmax_equal": None,
                                    "top8_set_overlap_count": None,
                                    "top8_ordered_equal": None,
                                    "provenance": {
                                        "packed_source": "path-aware gathered packed-tree cache",
                                        "dense_source": "dense committed-path recomputation did not complete",
                                        "actual_cuda_kernel": None,
                                    },
                                }
                        else:
                            kv_gold = {
                                "availability": "unavailable",
                                "status": "not_run",
                                "execution_status": "not_run_due_to_prior_failure",
                                "comparison_mode": "dense_committed_path",
                                "backend": "configured_sdpa",
                                "dtype": str(packed_output.logits.dtype).removeprefix(
                                    "torch."
                                ),
                                "structural_status": "not_run",
                                "semantic_status": "not_run",
                                "numerical_status": "not_run",
                                "thresholds": gold_thresholds,
                                "warning": None,
                                "reason": "prior_logits_gold_failure",
                                "max_abs_error": None,
                                "mean_abs_error": None,
                                "p50_abs_error": None,
                                "p95_abs_error": None,
                                "p99_abs_error": None,
                                "argmax_equal": None,
                                "top8_set_overlap_count": None,
                                "top8_ordered_equal": None,
                                "provenance": {
                                    "packed_source": "not_run",
                                    "dense_source": "not_run",
                                    "actual_cuda_kernel": None,
                                },
                            }
                        kv_gold["round_index"] = round_index
                        kv_gold_records.append(kv_gold)
                        if use_canonical_commit and kv_gold["status"] in {
                            "pass_exact",
                            "pass_semantic_with_numerical_variation",
                        }:
                            current_gold_status = kv_gold["status"]
                            failure_class = None
                            packed_gold["native_packed_divergence_corrected"] = not packed_gold[
                                "all_checks_passed"
                            ]
                        elif packed_gold["status"] in {
                            "fail_structural",
                            "fail_semantic",
                        }:
                            current_gold_status = packed_gold["status"]
                            failure_class = packed_gold["failure_class"]
                        elif kv_gold["status"] in {
                            "fail_structural",
                            "fail_semantic",
                        }:
                            current_gold_status = kv_gold["status"]
                            failure_class = (
                                "kv_structural_mismatch"
                                if kv_gold["status"] == "fail_structural"
                                else "kv_semantic_mismatch"
                            )
                        elif (
                            packed_gold["status"] == "not_run"
                            or kv_gold["status"] == "not_run"
                        ):
                            current_gold_status = "not_run"
                            failure_class = (
                                packed_gold.get("failure_class")
                                or "kv_gold_unavailable"
                            )
                        elif (
                            packed_gold["status"]
                            == "pass_semantic_with_numerical_variation"
                            or kv_gold["status"]
                            == "pass_semantic_with_numerical_variation"
                        ):
                            current_gold_status = (
                                "pass_semantic_with_numerical_variation"
                            )
                            failure_class = "numerical_backend_variation"
                        else:
                            current_gold_status = "pass_exact"
                            failure_class = None
                        gold_round_results.append(
                            {
                                "round_index": round_index,
                                "status": current_gold_status,
                                "failure_class": failure_class,
                                "logits": packed_gold,
                                "kv": {
                                    "status": kv_gold["status"],
                                    "reason": kv_gold.get("reason"),
                                },
                                "completed_checks": packed_gold["completed_checks"]
                                + (
                                    ["kv_compaction", "kv_next_step_logits"]
                                    if kv_gold["status"] != "not_run"
                                    else []
                                ),
                                "uncompleted_checks": (
                                    []
                                    if kv_gold["status"] != "not_run"
                                    else ["kv_compaction", "kv_next_step_logits"]
                                ),
                            }
                        )
                    else:
                        current_gold_status = "not_run"

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
                accepted_lengths.append(accepted_draft_tokens)
                cache_commit_lengths.append(committed_token_count)
                if return_diagnostics:
                    native_path_token_ids = [
                        int(value)
                        for value in native_committed_tokens.detach().cpu().tolist()
                    ]
                    canonical_available = canonical is not None
                    native_canonical_equal = (
                        not canonical["decision_changed"]
                        if canonical_available
                        else None
                    )
                    false_negative = (
                        bool(
                            canonical["decision_changed"]
                            and minimum_path_margin is not None
                            and minimum_path_margin > decision_margin_threshold
                        )
                        if canonical_available
                        and decision_stabilization == "shadow"
                        else None
                    )
                    stabilization_record = {
                        "stabilization_mode": decision_stabilization,
                        "decision_margin_threshold": decision_margin_threshold,
                        "recheck_triggered": recheck_triggered,
                        "trigger_reason": trigger_reason,
                        "minimum_path_margin": minimum_path_margin,
                        "verifier_margins": path_margins,
                        "native_canonical_decision_equal": native_canonical_equal,
                        "native_selected_path_token_ids": native_path_token_ids,
                        "canonical_accepted_path_token_ids": (
                            [
                                int(value)
                                for value in canonical["committed_tokens"]
                                .detach()
                                .cpu()
                                .tolist()
                            ]
                            if canonical_available
                            else None
                        ),
                        "native_accepted_depth": native_accepted_draft_tokens,
                        "canonical_accepted_depth": (
                            canonical["accepted_draft_token_count"]
                            if canonical_available
                            else None
                        ),
                        "first_corrected_depth": (
                            canonical["first_corrected_depth"]
                            if canonical_available
                            else None
                        ),
                        "first_corrected_native_margin": (
                            path_margins[
                                canonical["first_corrected_depth"] - 1
                            ]
                            if canonical_available
                            and canonical["first_corrected_depth"] is not None
                            and path_margins is not None
                            else None
                        ),
                        "native_next_root_token_id": int(
                            traversal["next_root_token_id"].item()
                        ),
                        "native_next_root_kind": (
                            "bonus"
                            if bool(traversal["next_root_is_bonus"].item())
                            else "correction"
                        ),
                        "canonical_next_root_token_id": (
                            canonical["next_root_token_id"]
                            if canonical_available
                            else None
                        ),
                        "canonical_next_root_kind": (
                            canonical["next_root_kind"]
                            if canonical_available
                            else None
                        ),
                        "corrected_decision_count": (
                            1
                            if canonical_available and canonical["decision_changed"]
                            else 0
                            if canonical_available
                            else None
                        ),
                        "false_negative": false_negative,
                        "recheck_input_shape": (
                            [int(value) for value in canonical["input_token_ids"].shape]
                            if canonical_available
                            else None
                        ),
                        "recheck_query_length": (
                            canonical["query_length"] if canonical_available else None
                        ),
                        "recheck_round_entry_kv_length": (
                            canonical["round_entry_kv_length"]
                            if canonical_available
                            else None
                        ),
                        "recheck_effective_kv_length": (
                            canonical["effective_kv_length"]
                            if canonical_available
                            else None
                        ),
                        "canonical_cache_length": (
                            canonical["compaction"]["canonical_cache_length"]
                            if canonical_available
                            else None
                        ),
                        "canonical_cache_committed": (
                            use_canonical_commit if canonical_available else None
                        ),
                        "committed_cache_source": compaction[
                            "committed_cache_source"
                        ],
                        "unavailable_reason": (
                            None
                            if canonical_available
                            else "canonical_recheck_not_triggered"
                        ),
                    }
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
                        "tree_policy": tree["policy"],
                        "online_selector": selector_record,
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
                        "decision_stabilization": stabilization_record,
                        "gold_check_status": current_gold_status,
                        "output_ids_equal_reference": None,
                        "offline_comparison_status": "not_requested",
                        "_top1_matches_sampling": top1_matches_sampling,
                        "_next_root_is_bonus": effective_traversal[
                            "next_root_is_bonus"
                        ],
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
                            "stabilization_cache_snapshot": snapshot_pair,
                            "canonical_recheck": canonical_pair,
                            "kv_replacement": kv_replacement_pair,
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
                    output_ids = output_ids[:, : num_input_tokens + stop_indices[0] + 1]
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
            "stabilization_cache_snapshot_cuda_elapsed_ms": 0.0,
            "canonical_recheck_cuda_elapsed_ms": 0.0,
            "kv_replacement_cuda_elapsed_ms": 0.0,
            "total_draft_cuda_elapsed_ms": 0.0,
            "total_verification_cuda_elapsed_ms": 0.0,
            "round_cuda_elapsed_ms": 0.0,
        }
        component_observed = {field: 0 for field in component_totals}
        pair_to_field = {
            "draft": "draft_hidden_logits_cuda_elapsed_ms",
            "topk": "topk_cuda_elapsed_ms",
            "tree_build": "tree_construction_cuda_elapsed_ms",
            "mask_build": "mask_construction_cuda_elapsed_ms",
            "verify": "verification_cuda_elapsed_ms",
            "traverse": "traversal_cuda_elapsed_ms",
            "kv_compact": "kv_compaction_cuda_elapsed_ms",
            "stabilization_cache_snapshot": "stabilization_cache_snapshot_cuda_elapsed_ms",
            "canonical_recheck": "canonical_recheck_cuda_elapsed_ms",
            "kv_replacement": "kv_replacement_cuda_elapsed_ms",
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
                pair = timing[pair_name]
                value = (
                    timer.values(pair)["cuda_elapsed_ms"]
                    if pair is not None
                    else None
                )
                record["timing"][field] = value
                if value is not None:
                    component_totals[field] += value
                    component_observed[field] += 1
            round_ms = record["timing"]["round_cuda_elapsed_ms"]
            record["timing"]["draft_over_round_ratio"] = (
                record["timing"]["total_draft_cuda_elapsed_ms"] / round_ms
                if round_ms
                else None
            )
            record["timing"]["verification_over_round_ratio"] = (
                record["timing"]["total_verification_cuda_elapsed_ms"] / round_ms
                if round_ms
                else None
            )
            canonical_ms = record["timing"][
                "canonical_recheck_cuda_elapsed_ms"
            ]
            snapshot_ms = record["timing"][
                "stabilization_cache_snapshot_cuda_elapsed_ms"
            ]
            replacement_ms = record["timing"]["kv_replacement_cuda_elapsed_ms"]
            stabilized_round_ms = (
                None
                if record["decision_stabilization"]["stabilization_mode"]
                == "shadow"
                else round_ms
            )
            record["timing"]["stabilized_round_cuda_elapsed_ms"] = (
                stabilized_round_ms
            )
            record["timing"]["diagnostic_round_cuda_elapsed_ms"] = round_ms
            record["timing"]["recheck_over_round_ratio"] = (
                (canonical_ms + (snapshot_ms or 0.0) + (replacement_ms or 0.0))
                / round_ms
                if canonical_ms is not None and round_ms
                else None
            )
            total_recheck_overhead_ms = (
                canonical_ms + (snapshot_ms or 0.0) + (replacement_ms or 0.0)
                if canonical_ms is not None
                else None
            )
            record["timing"]["total_recheck_overhead_cuda_elapsed_ms"] = (
                total_recheck_overhead_ms
            )
            record["timing"]["unavailable_reasons"] = {
                "canonical_recheck_cuda_elapsed_ms": (
                    None
                    if canonical_ms is not None
                    else "canonical_recheck_not_triggered"
                ),
                "kv_replacement_cuda_elapsed_ms": (
                    None
                    if replacement_ms is not None
                    else "canonical_kv_not_committed"
                ),
            }
            stabilization = record["decision_stabilization"]
            stabilization.update(
                {
                    "packed_verification_cuda_elapsed_ms": record["timing"][
                        "verification_cuda_elapsed_ms"
                    ],
                    "canonical_recheck_cuda_elapsed_ms": canonical_ms,
                    "kv_replacement_cuda_elapsed_ms": replacement_ms,
                    "total_recheck_overhead_cuda_elapsed_ms": (
                        total_recheck_overhead_ms
                    ),
                    "stabilized_round_cuda_elapsed_ms": stabilized_round_ms,
                    "stabilized_round_unavailable_reason": (
                        "shadow_oracle_is_not_an_online_stabilized_timing"
                        if stabilized_round_ms is None
                        else None
                    ),
                    "diagnostic_round_cuda_elapsed_ms": round_ms,
                    "recheck_over_round_ratio": record["timing"][
                        "recheck_over_round_ratio"
                    ],
                    "generated_token_count": record["returned_token_delta"],
                    "committed_token_count": record["committed_token_count"],
                }
            )
            record["tree_metadata"] = tensor_tree_metadata(record["tree_metadata"])
            compaction = record["cache_compaction"]
            record["cache_compaction"] = {
                key: (
                    value.detach().cpu().tolist() if hasattr(value, "detach") else value
                )
                for key, value in compaction.items()
            }

        diagnostics = {
            "schema_version": "1.0",
            "prompt_token_count": num_input_tokens,
            "generated_token_count": int(output_ids.shape[1] - num_input_tokens),
            "max_new_tokens": int(max_new_tokens),
            "draft_block_size": block_size,
            "prefill_seed_token_count": 1,
            "accepted_draft_token_lengths": accepted_lengths,
            "cache_commit_lengths": cache_commit_lengths,
            "stop_reason": final_stop_reason,
            "reference_state": {
                "enabled": enable_reference_state_diagnostics,
                "execution_path": "layerwise_tree_shadow",
                "checkpoint_provenance": "diagnostic_replay",
                "replay_semantics": "native packed-tree verification and commit",
                "requested_generated_prefixes": reference_state_generated_prefixes,
                "checkpoints": [
                    reference_state_checkpoints[prefix]
                    for prefix in sorted(reference_state_checkpoints)
                ],
                "formal_timing_included": False,
                "full_float32_kv_cpu_copy_created": False,
                "tensor_comparison_scope": "bounded_stable_samples",
            },
            "rounds": rounds,
            "phase2c": {
                "enabled": True,
                "tree_policy": (
                    "dynamic_frozen_selector"
                    if selector_mode is not None
                    else tree_policy
                ),
                "online_selector_mode": selector_mode,
                "online_selector_config_hash": (
                    selector_config["config_sha256"]
                    if selector_config is not None
                    else None
                ),
                "requested_node_budget": node_budget,
                "beam_width": beam_width,
                "draft_top_k": draft_top_k,
                "decision_stabilization": decision_stabilization,
                "decision_margin_threshold": decision_margin_threshold,
                "canonical_recheck_scope": "native-selected root-to-terminal path",
                "canonical_recheck_query_length": block_size,
                "canonical_attention_mask": "implicit causal, matching original DFlash",
                "maximum_draft_depth": 15,
                "target_information_used_for_tree_construction": False,
                "full_vocabulary_logits_saved": False,
                "target_logits_saved": False,
                "gold_checks_enabled": enable_gold_checks,
                "gold_comparison_modes": list(gold_comparison_modes),
                "gold_layer_localization_enabled": gold_layer_localization,
                "gold_max_rounds": gold_max_rounds,
                "gold_record_all_nodes": gold_record_all_nodes,
                "gold_thresholds": gold_thresholds,
                "packed_logits_gold_checks": packed_gold_records,
                "kv_compaction_gold_checks": kv_gold_records,
                "gold_layer_localization": gold_layer_localization_records,
                "gold_round_results": gold_round_results,
                "tree_traversal_checks": tree_traversal_checks,
                "gold_overall_status": (
                    "not_run"
                    if not gold_round_results
                    else next(
                        (
                            status
                            for status in ("fail_structural", "fail_semantic")
                            if any(
                                row["status"] == status for row in gold_round_results
                            )
                        ),
                        "pass_semantic_with_numerical_variation"
                        if any(
                            row["status"] == "pass_semantic_with_numerical_variation"
                            for row in gold_round_results
                        )
                        else "not_run"
                        if any(row["status"] == "not_run" for row in gold_round_results)
                        else "pass_exact",
                    )
                ),
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
                    "phase2c.stabilization_cache_snapshot",
                    "phase2c.canonical_recheck",
                    "phase2c.canonical_kv_commit",
                    "phase2c.round",
                ],
                "explicit_cuda_synchronize_count": 1 if enable_profiling else 0,
                "prefill": timer.values(prefill_pair),
                "total": timer.values(total_pair),
                "components": {
                    field: (
                        value
                        if enable_profiling and component_observed[field]
                        else None
                    )
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
        enable_reference_state_diagnostics: bool,
        reference_state_generated_prefixes: list[int],
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
        requested_state_prefixes = set(reference_state_generated_prefixes)
        reference_state_checkpoints = {}
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
            if enable_reference_state_diagnostics and 0 in requested_state_prefixes:
                self._phase2c_update_reference_state_checkpoint(
                    reference_state_checkpoints,
                    execution_path="original_dflash_reference",
                    provenance="actual_reference_checkpoint",
                    generated_prefix_length=0,
                    prompt_length=num_input_tokens,
                    output_ids=output_ids,
                    target_cache=past_key_values_target,
                    target_hidden=target_hidden,
                    target_hidden_provenance=(
                        "actual_reference_target_hidden_used_by_draft"
                    ),
                    semantic_probe_logits=output.logits[0, -1],
                    semantic_probe_provenance=(
                        "actual_reference_prefill_next_token_logits"
                    ),
                )

            start = input_ids.shape[1]
            round_index = 0
            returned_tokens_before = min(1, max_new_tokens)
            while start < max_length:
                context_length_before = int(start)
                generated_prefix_before = context_length_before - num_input_tokens
                if (
                    enable_reference_state_diagnostics
                    and generated_prefix_before in requested_state_prefixes
                ):
                    self._phase2c_update_reference_state_checkpoint(
                        reference_state_checkpoints,
                        execution_path="original_dflash_reference",
                        provenance="actual_reference_checkpoint",
                        generated_prefix_length=generated_prefix_before,
                        prompt_length=num_input_tokens,
                        output_ids=output_ids,
                        target_cache=past_key_values_target,
                        updates={
                            "target_hidden_used_by_draft": (
                                self._phase2c_tensor_comparable_summary(target_hidden)
                            ),
                            "target_hidden_used_by_draft_provenance": (
                                "actual_reference_target_hidden_used_by_draft"
                            ),
                            "round_entry": {
                                "round_index": round_index,
                                "generated_root_offset": generated_prefix_before,
                                "root_token_id": int(output_ids[0, start].item()),
                                "persistent_target_kv_length": context_length_before,
                                "draft_kv_layer_lengths_before": dynamic_cache_lengths(
                                    past_key_values_draft
                                ),
                            }
                        },
                    )
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
                    if (
                        enable_reference_state_diagnostics
                        and generated_prefix_before in requested_state_prefixes
                    ):
                        checkpoint = reference_state_checkpoints[
                            generated_prefix_before
                        ]
                        draft_summary = self._phase2c_draft_logits_comparable_summary(
                            draft_logits
                        )
                        checkpoint["draft_state"] = {
                            "cache": self._phase2c_cache_comparable_summary(
                                past_key_values_draft
                            ),
                            "cache_layer_lengths": dynamic_cache_lengths(
                                past_key_values_draft
                            ),
                            "logits": draft_summary,
                            "candidate_top1_token_ids": [
                                int(row["top1_token_id"])
                                for row in draft_summary["positions"]
                            ],
                            "candidate_topk_token_ids": [
                                [int(value) for value in row["top_token_ids"]]
                                for row in draft_summary["positions"]
                            ],
                            "candidate_tree_input": {
                                "status": (
                                    "not_applicable_original_dflash_linear_proposal"
                                ),
                                "actual_proposed_token_ids": [
                                    int(value)
                                    for value in block_output_ids[0, 1:]
                                    .detach()
                                    .cpu()
                                    .tolist()
                                ],
                            },
                        }
                        checkpoint["draft_state_provenance"] = (
                            "actual_reference_draft_round_entry"
                        )

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
                    if enable_reference_state_diagnostics:
                        committed_token_count = accepted_draft_tokens + 1
                        for committed_offset in range(1, committed_token_count + 1):
                            generated_prefix = (
                                generated_prefix_before + committed_offset
                            )
                            if generated_prefix not in requested_state_prefixes:
                                continue
                            self._phase2c_update_reference_state_checkpoint(
                                reference_state_checkpoints,
                                execution_path="original_dflash_reference",
                                provenance="actual_reference_checkpoint",
                                generated_prefix_length=generated_prefix,
                                prompt_length=num_input_tokens,
                                output_ids=output_ids,
                                target_cache=past_key_values_target,
                                target_hidden=target_hidden[
                                    :, committed_offset - 1 : committed_offset, :
                                ],
                                target_hidden_provenance=(
                                    "actual_reference_committed_token_hidden"
                                ),
                                semantic_probe_logits=output.logits[
                                    0, committed_offset - 1
                                ],
                                semantic_probe_provenance=(
                                    "actual_original_dflash_linear_verifier_logits"
                                ),
                                updates={
                                    "reference_round_index": round_index,
                                    "committed_token_kind": (
                                        "root"
                                        if committed_offset == 1
                                        else "accepted_draft"
                                    ),
                                    "committed_token_id": int(
                                        block_output_ids[
                                            0, committed_offset - 1
                                        ].item()
                                    ),
                                    "linear_verifier_query_index": (
                                        committed_offset - 1
                                    ),
                                },
                            )
                        generated_prefix_after = start - num_input_tokens
                        if generated_prefix_after in requested_state_prefixes:
                            checkpoint = reference_state_checkpoints[
                                generated_prefix_after
                            ]
                            checkpoint["commit_transition"] = {
                                "round_index": round_index,
                                "generated_prefix_before": generated_prefix_before,
                                "generated_prefix_after": generated_prefix_after,
                                "accepted_draft_token_count": accepted_draft_tokens,
                                "committed_token_ids": [
                                    int(value)
                                    for value in block_output_ids[
                                        0, :committed_token_count
                                    ]
                                    .detach()
                                    .cpu()
                                    .tolist()
                                ],
                                "next_root_token_id": int(
                                    posterior[0, accepted_draft_tokens].item()
                                ),
                                "next_root_kind": (
                                    "bonus"
                                    if accepted_draft_tokens == block_size - 1
                                    else "correction"
                                ),
                                "committed_cache_source": (
                                    "original_dflash_linear_verification_crop"
                                ),
                                "persistent_target_kv_length_after": start,
                            }
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
                        combined_attention_cuda_elapsed_ms if enable_profiling else None
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
            record["timing"]["verification_cuda_elapsed_ms"] or 0.0 for record in rounds
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
            "reference_state": {
                "enabled": enable_reference_state_diagnostics,
                "execution_path": "original_dflash_reference",
                "checkpoint_provenance": "actual_reference_checkpoint",
                "requested_generated_prefixes": reference_state_generated_prefixes,
                "checkpoints": [
                    reference_state_checkpoints[prefix]
                    for prefix in sorted(reference_state_checkpoints)
                ],
                "formal_timing_included": False,
                "full_float32_kv_cpu_copy_created": False,
                "tensor_comparison_scope": "bounded_stable_samples",
            },
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
