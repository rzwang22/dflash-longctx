# Phase 0: Qwen3-8B Transformers execution path

This audit covers repository commit `94e4abc` as the starting baseline and the
Qwen3-8B pair below. It does not change DFlash generation or attention behavior.

- Target: `Qwen/Qwen3-8B`
- Target revision pinned for Phase 0: `b968826d9c46dd6066d109eabc6255188de91218`
- Draft: `z-lab/Qwen3-8B-DFlash-b16`
- Draft revision pinned for Phase 0: `9b41424b7109f9c5413454f481b09a82b85333f4`
- Backend: Transformers

## Source-confirmed facts

### There are two distinct Transformers paths

The repository currently exposes two paths that should not be conflated.

1. The README quick start calls
   `AutoModel.from_pretrained(..., trust_remote_code=True)` and then
   `draft.spec_generate(...)`.
2. `python -m dflash.benchmark --backend transformers` enters
   `dflash.benchmark._run_transformers`. It imports
   `DFlashDraftModel` and `dflash_generate` from the local
   `dflash/model.py`, loads draft weights with
   `DFlashDraftModel.from_pretrained`, and invokes local
   `dflash_generate(..., return_stats=True)`. It does not call `AutoModel` or
   the remote `spec_generate` method.

The Phase 0 smoke test deliberately follows path 1 because the requested audit
is for `trust_remote_code=True`. The existing multi-process benchmark follows
path 2 and is not the first smoke test.

### What `AutoModel` loads

At the pinned draft revision, the model `config.json` contains:

```json
"auto_map": {"AutoModel": "dflash.DFlashDraftModel"}
```

Consequently, Transformers downloads the small custom-code files from the
model repository, places them in its dynamic-module cache, and imports
`DFlashDraftModel` from that cached `dflash.py`. It does **not** import this
repository's `dflash/model.py`. The exact on-disk cache path depends on the
host's Hugging Face cache settings; the smoke test records it with
`inspect.getfile`.

The target config declares native architecture `Qwen3ForCausalLM` and has no
`auto_map`. Therefore `AutoModelForCausalLM` uses the Qwen3 implementation from
the installed Transformers package (normally
`transformers/models/qwen3/modeling_qwen3.py`), while target configuration and
weights reside in the Hugging Face cache. Passing `trust_remote_code=True` to
the target does not replace its native class.

This behavior is also why immutable model revisions matter: Hugging Face
recommends pinning a commit when executing custom model code.

### Origin and role of each similarly named file

| File | Origin in the remote-code path | Loaded at the pinned revision? | Role |
|---|---|---:|---|
| `dflash.py` | Draft model repository, then Transformers dynamic-module cache | Yes | Defines `DFlashDraftModel`, `Qwen3DFlashAttention`, and the active `spec_generate` method. |
| `utils.py` | Draft model repository, then the same dynamic-module cache | Yes | Imported by remote `dflash.py`; provides `build_target_layer_ids`, `extract_context_feature`, and `sample` (plus dataset helpers unused by the smoke test). |
| `modeling_dflash.py` | Draft model repository | No | A second/older standalone implementation. It is neither named by `auto_map` nor imported by active `dflash.py`. |
| `dflash/model.py` | This Git repository | No, not for `AutoModel` | Version-controlled local implementation used by `dflash.benchmark._run_transformers`. |

### Active functions in the remote-code quick-start path

`spec_generate` is defined by
`DFlashDraftModel.spec_generate` in the draft repository's `dflash.py`. The
active stages are:

1. **Target prefill:** `spec_generate` calls `target(...)` with
   `output_hidden_states=True`, initializes the target `DynamicCache`, and
   samples the first token from target logits.
2. **Context feature extraction:** `utils.extract_context_feature` selects
   target hidden states at `target_layer_ids` and concatenates them on the
   hidden dimension.
3. **Draft forward:** `DFlashDraftModel.forward` projects the concatenated
   target features through `fc` and `hidden_norm`, embeds the masked/proposed
   block using the target embedding table, and runs its draft decoder layers.
4. **Injected context K/V attention:** every
   `Qwen3DFlashAttention.forward` projects `target_hidden` to `k_ctx` and
   `v_ctx`, projects draft/noise hidden states to `k_noise` and `v_noise`,
   concatenates context and draft K/V, updates the draft `DynamicCache`, and
   dispatches to the configured Transformers attention function. The injected
   values are selected target *hidden states projected by draft K/V layers*,
   not the target model's own KV tensors.
5. **Target verification:** inside the decode loop, `spec_generate` calls
   `target(block_output_ids, ...)`, samples the target posterior, counts the
   matching prefix between proposed tokens and shifted target tokens, commits
   the accepted prefix plus one target token, and crops the target cache.

The local path implements the same responsibilities in
`dflash.model.dflash_generate`, `DFlashDraftModel.forward`, and
`Qwen3DFlashAttention.forward`. Unlike remote `spec_generate`, local
`dflash_generate(return_stats=True)` returns `acceptance_lengths` directly.

### Acceptance-length availability

The pinned remote `spec_generate` builds an internal `acceptance_lengths` list
but returns only the `output_ids` tensor. There is no supported return flag or
public field for that list. The Phase 0 smoke test therefore records acceptance
length as unavailable; it does not infer acceptance from generated output.

The local benchmark does have direct access through
`dflash_generate(..., return_stats=True)`, but substituting that API would
change which execution path the remote-code smoke test validates.

## Questions that require an NVIDIA GPU run

- The actual dynamic-module cache path for remote `dflash.py`/`utils.py`, the
  target class file path, and the `spec_generate` definition path on the GPU
  server. The smoke test prints and saves all three.
- Whether repository-pinned Transformers `4.57.1` is fully compatible with the
  draft revision whose config/model card references Transformers `4.57.3`.
- The concrete attention implementation/kernel selected on the server and
  whether optional FlashAttention installation changes the baseline.
- Single-GPU memory fit, CUDA/PyTorch compatibility, successful model download,
  generated text, token counts, and elapsed generation time.
- Numerical parity between the remote quick-start implementation and the local
  benchmark implementation for the same pinned weights and prompt. Phase 0
  does not claim parity until it is measured.
- Runtime tensor shapes/cache lengths for injected context K/V. Their source
  operations are confirmed above; instrumentation is deferred to Phase 1.

## Phase 1 modification recommendation

For version-controlled injected-KV/context-attention changes, use the existing
local path and modify these symbols in `dflash/model.py`:

- `Qwen3DFlashAttention.forward` for context K/V projection, concatenation,
  masking, cache update, or attention dispatch;
- `DFlashDraftModel.forward` for context feature projection and draft-layer
  data flow;
- `extract_context_feature` for selection/assembly of target layer features;
- `dflash_generate` for draft/verify scheduling and measurement only when the
  research change genuinely belongs outside the attention layer.

Do not edit files inside the Hugging Face cache. Such edits are host-local,
unreviewable, and easy to lose. If Phase 1 must preserve the exact `AutoModel`
interface, add a repository-owned custom model module (or register the existing
local class with the relevant AutoClass) and load the pinned Hub weights into
that local class. Keep only Python/configuration code under Git; continue to
reference weights by model ID and immutable revision. Before experiments,
establish a GPU parity test between that version-controlled class and the
pinned remote baseline.

## Audited primary sources

- Local: `README.md`, `pyproject.toml`, `dflash/benchmark.py`, and
  `dflash/model.py` at baseline commit `94e4abc`.
- Draft model: [`config.json`](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16/blob/9b41424b7109f9c5413454f481b09a82b85333f4/config.json),
  [`dflash.py`](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16/blob/9b41424b7109f9c5413454f481b09a82b85333f4/dflash.py),
  [`modeling_dflash.py`](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16/blob/9b41424b7109f9c5413454f481b09a82b85333f4/modeling_dflash.py), and
  [`utils.py`](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16/blob/9b41424b7109f9c5413454f481b09a82b85333f4/utils.py).
- Target model: [`config.json`](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json).
- Transformers guidance on [loading and pinning custom model code](https://huggingface.co/docs/transformers/v4.57.0/en/models#custom-models).
