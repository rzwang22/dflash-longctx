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
    build_round_accounting,
    finalize_generation_result,
    gpu_memory_snapshot,
    nvtx_range,
    tensor_shape,
)
from .utils import build_target_layer_ids, extract_context_feature, sample


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
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        if phase1_attention_records is not None:
            phase1_attention_records.append(
                {
                    "round_index": phase1_round_index,
                    "layer_index": self.layer_idx,
                    "configured_attention_implementation": self.config._attn_implementation,
                    "calls_pytorch_sdpa": self.config._attn_implementation == "sdpa",
                    "dtype": str(q.dtype),
                    "batch_size": int(bsz),
                    "query_length": int(q_len),
                    "injected_context_length": int(ctx_len),
                    "current_block_kv_length": int(ctx_len + q_len),
                    "effective_kv_length": int(k.shape[-2]),
                    "query_head_count": int(self.config.num_attention_heads),
                    "key_value_head_count": int(self.config.num_key_value_heads),
                    "head_dimension": int(self.head_dim),
                    "query_shape": tensor_shape(q),
                    "key_shape": tensor_shape(k),
                    "value_shape": tensor_shape(v),
                    "theoretical_qk_score_elements": int(
                        bsz * self.config.num_attention_heads * q_len * k.shape[-2]
                    ),
                    "kernel_latency_isolated": False,
                }
            )
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
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
    ):
        """Generate with the pinned behavior; diagnostics are strictly opt-in."""

        if not return_diagnostics and not enable_profiling:
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
        round_timing_pairs = []
        rounds = []
        accepted_draft_lengths = []
        cache_commit_lengths = []
        total_pair = timer.start()

        with nvtx_range(torch, "dflash.total", enable_profiling):
            prefill_pair = timer.start()
            with nvtx_range(torch, "dflash.prefill", enable_profiling):
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
                        )
                        draft_logits = target.lm_head(
                            draft_hidden[:, -block_size + 1 :, :]
                        )
                        past_key_values_draft.crop(start)
                        block_output_ids[:, 1:] = sample(draft_logits)
                    timer.stop(draft_pair)

                    verify_pair = timer.start()
                    with nvtx_range(torch, "dflash.verify", enable_profiling):
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
                            "commit_cpu_elapsed_ms": commit_cpu_elapsed_ms,
                            "round_cpu_elapsed_ms": None,
                            "round_cuda_elapsed_ms": None,
                        },
                        "gpu_memory": gpu_memory_snapshot(torch, target.device)
                        if enable_profiling
                        else None,
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

        for accounting, draft_pair, verify_pair, round_pair in round_timing_pairs:
            draft_timing = timer.values(draft_pair)
            verify_timing = timer.values(verify_pair)
            round_timing = timer.values(round_pair)
            accounting["timing"].update(
                {
                    "draft_cpu_elapsed_ms": draft_timing["cpu_elapsed_ms"],
                    "draft_cuda_elapsed_ms": draft_timing["cuda_elapsed_ms"],
                    "verification_cpu_elapsed_ms": verify_timing["cpu_elapsed_ms"],
                    "verification_cuda_elapsed_ms": verify_timing["cuda_elapsed_ms"],
                    "round_cpu_elapsed_ms": round_timing["cpu_elapsed_ms"],
                    "round_cuda_elapsed_ms": round_timing["cuda_elapsed_ms"],
                }
            )

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
                ],
                "explicit_cuda_synchronize_count": 1 if enable_profiling else 0,
                "prefill": timer.values(prefill_pair),
                "total": timer.values(total_pair),
                "memory": gpu_memory_snapshot(torch, target.device)
                if enable_profiling
                else None,
            },
        }
        return finalize_generation_result(
            output_ids, diagnostics, return_diagnostics=return_diagnostics
        )
