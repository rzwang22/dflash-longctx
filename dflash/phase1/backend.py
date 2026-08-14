"""Attention-backend provenance without changing backend selection."""

from __future__ import annotations

from typing import Any


def _call_flag(owner: Any, name: str) -> bool | None:
    function = getattr(owner, name, None)
    if function is None:
        return None
    try:
        return bool(function())
    except Exception:  # noqa: BLE001 - backend probes are best-effort provenance.
        return None


def _parameter_dtype(model: Any) -> str | None:
    try:
        return str(next(model.parameters()).dtype)
    except (AttributeError, StopIteration):
        return None


def model_attention_snapshot(model: Any, *, role: str) -> dict[str, Any]:
    config = model.config
    implementation = getattr(config, "_attn_implementation", None)
    head_dimension = getattr(config, "head_dim", None)
    if head_dimension is None and getattr(config, "num_attention_heads", 0):
        head_dimension = config.hidden_size // config.num_attention_heads
    return {
        "role": role,
        "configured_attention_implementation": implementation,
        "calls_pytorch_sdpa": implementation == "sdpa",
        "configured_eager_path": implementation == "eager",
        "math_or_eager_fallback_status": (
            "configured_eager"
            if implementation == "eager"
            else "unconfirmed_without_kernel_trace"
        ),
        "dtype": _parameter_dtype(model),
        "query_head_count": getattr(config, "num_attention_heads", None),
        "key_value_head_count": getattr(config, "num_key_value_heads", None),
        "head_dimension": head_dimension,
        "actual_cuda_kernel": None,
        "actual_cuda_kernel_status": (
            "Not identifiable from Python configuration alone; confirm with Nsight "
            "or an equivalent kernel trace."
        ),
    }


def attention_backend_snapshot(
    torch_module: Any, *, target: Any, draft: Any
) -> dict[str, Any]:
    cuda_backend = getattr(getattr(torch_module, "backends", None), "cuda", None)
    return {
        "target": model_attention_snapshot(target, role="target"),
        "draft": model_attention_snapshot(draft, role="draft"),
        "pytorch_sdpa_allowed_backends": {
            "flash": _call_flag(cuda_backend, "flash_sdp_enabled"),
            "memory_efficient": _call_flag(cuda_backend, "mem_efficient_sdp_enabled"),
            "math": _call_flag(cuda_backend, "math_sdp_enabled"),
            "cudnn": _call_flag(cuda_backend, "cudnn_sdp_enabled"),
        },
        "flash_attn_python_package_required_for_sdpa": False,
        "backend_changed_by_phase1": False,
        "interpretation": (
            "Allowed SDPA backends are policy flags, not proof of the kernel selected "
            "for a particular tensor shape."
        ),
    }
