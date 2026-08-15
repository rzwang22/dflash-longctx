# Phase 1A: version-controlled DFlash observability

## Scope and frozen baseline

Phase 0 established a token-generating Transformers baseline on one NVIDIA H100
PCIe 80GB. Its immutable reference is commit
`a2e5e7353f38c443abaf2f2f8170b5b8c5a0b685`, tagged
`phase0-gpu-validated-h100-a2e5e73`.

Phase 1A makes the actually executed draft code version-controlled and adds
opt-in diagnostics, CUDA events, and NVTX ranges. It does not implement sparse
attention, change attention masks or backend selection, alter candidate
distributions or verification semantics, train weights, or make long-context
performance claims.

## Actual custom-code loading mechanism

The pinned draft configuration maps `AutoModel` to
`dflash.DFlashDraftModel`. With `trust_remote_code=True`, Transformers reads the
Python files from a local model directory and imports a copied module from its
dynamic-module cache. Consequently the executed `dflash.py` path is normally
under `~/.cache/huggingface/modules/transformers_modules/`, not this repository.

Phase 1A materialization closes that provenance gap:

1. `third_party/qwen3_8b_dflash_b16/original/` retains the four files from
   `z-lab/Qwen3-8B-DFlash-b16` revision
   `9b41424b7109f9c5413454f481b09a82b85333f4`.
2. `instrumented/dflash.py` contains the opt-in implementation; its default
   path delegates to an AST-equivalent transcription of the pinned function.
3. `scripts/materialize_dflash.py` validates local source files, Hugging Face
   revision metadata, weight size, and the pinned Git LFS SHA-256/etag.
4. It copies repository-controlled configuration/code into an external output
   directory and symlinks, rather than copies, the external weight.
5. `AutoModel.from_pretrained(..., local_files_only=True)` loads that directory.
6. The runner hashes the actual dynamic-cache `dflash.py` and requires it to
   equal the repository-controlled hash before accepting a run.

The materializer is idempotent: rerunning the same command preserves the
materialization ID and creation time, repairs controlled code files, verifies
the weight link, and refuses to replace a conflicting weight file or link. No
server path is encoded in repository code; paths are supplied at runtime.

The model card declares MIT. The retained license, upstream hashes, model ID,
revision, and the non-vendored weight's LFS identity are in
`third_party/qwen3_8b_dflash_b16/provenance.json`. The pinned upstream `utils.py`
lacks a final newline; the vendored reference adds one, with both hashes and the
normalization explicitly recorded.

Weight identity is checked by exact byte size plus equality between the pinned
Git LFS SHA-256 and the local Hugging Face metadata etag at the pinned revision.
The materializer does not reread the full 2.097 GB file to recompute that digest;
this limitation is recorded as `full_weight_hash_recomputed: false`.

## Exact acceptance and commit accounting

For block size `B=16`, every decode round begins with one target-produced token
already present at block position zero. The draft proposes `B-1=15` tokens at
positions 1 through 15. The target verifies the complete 16-token block.

- `accepted_draft_token_count` is the length of the matching prefix between
  draft positions 1..15 and target posterior positions 0..14. It excludes the
  block's target seed and excludes the next target token.
- `rejected_draft_token_count = proposed_draft_token_count -
  accepted_draft_token_count`. “Rejected” here means not accepted by the prefix
  rule; positions after the first mismatch are also counted as unaccepted.
- `committed_token_count = 1 + accepted_draft_token_count`. This is precisely
  the amount by which `start` advances and to which target KV is cropped: the
  carried target seed plus accepted draft tokens.
- The target always writes one additional token at the new `start`. It is a
  correction when fewer than 15 drafts match, or a bonus when all 15 match. It
  is the pending seed for the next round and is not part of the current round's
  target-cache crop.

Therefore accepted length zero commits the existing target seed, rejects all 15
draft proposals, and writes one target correction as the next pending seed.
The principal conservation checks are:

```text
proposed drafts = accepted drafts + rejected drafts
cache committed = one pending target seed + accepted drafts
context after   = context before + cache committed
```

The source always drafts and verifies a full block, including the final round
when fewer than 16 returned-token slots remain. Diagnostics separately report
raw generated count, returned count, and truncation. `max_new_tokens` slicing
can discard accepted or newly written target tokens beyond the return budget.
EOS detection occurs after a round; returned output is truncated at the first
EOS, so work later in that round can also be excluded from returned counts.

The upstream local variable `acceptance_lengths` stores
`accepted_draft_token_count + 1`, not draft acceptance alone. Phase 1A avoids
exposing that ambiguous value as accepted length; it records separate
`accepted_draft_token_lengths` and `cache_commit_lengths`.

## Diagnostics, timing, and NVTX

The original five `spec_generate` arguments remain valid. With both new keyword
options omitted or false, the method returns the original `torch.Tensor` and
executes the reference path. `return_diagnostics=True` returns an object with
`output_ids` and `diagnostics`; `enable_profiling=True` adds CUDA-event timing,
memory counters, and NVTX.

Stable ranges are:

- `dflash.total`
- `dflash.prefill`
- `dflash.round`
- `dflash.draft`
- `dflash.verify`
- `dflash.commit`
- `dflash.attention.combined_context_block`

CUDA event pairs are resolved with one explicit synchronization at the complete
generation measurement boundary. The source algorithm itself still calls
`.item()` on acceptance each round; that algorithmic host dependency can imply
a synchronization and is not hidden. No layer performs an explicit
`torch.cuda.synchronize()`.

Per-round output contains draft/verification CUDA and CPU time, commit CPU time,
total round time, allocator memory, tensor shapes, and attention dimensions.
The runner also measures CPU wall time. These values are labeled separately;
CPU wall time is never presented as CUDA execution time. Warmup runs finish
before paired control/instrumented measurement runs begin. The summary reports
median absolute and percentage instrumentation overhead, including the
measurement-boundary synchronization behavior.

`Qwen3DFlashAttention` concatenates injected target context K/V with block-local
K/V and dispatches one attention function. Phase 1A does not pretend those two
parts have independently measured latency. It records Q/K/V shapes, query and
KV lengths, head counts, head dimension, theoretical QK score elements, and an
NVTX range around the combined call. Separating costs requires a later
context-length sweep or counterfactual execution.

Phase 1B preserves `injected_context_length` as a compatibility alias and
clarifies that it counts newly supplied `target_hidden` positions, not all
persistent context KV. New records use `newly_injected_hidden_length`,
`persistent_context_kv_length`, and `effective_kv_length`; their exact meanings
and invariants are documented in `phase1b-longctx-baseline.md`.

## Attention backend provenance

The summary records target and draft `_attn_implementation`, whether that value
selects the PyTorch SDPA registry path, dtype, head counts, and head dimension.
It also records whether PyTorch currently allows flash, memory-efficient, math,
and cuDNN SDPA backends.

Absence of the `flash_attn` Python package does not disable fused PyTorch SDPA.
Conversely, an enabled SDPA policy flag does not prove which CUDA kernel a given
shape used or whether it fell back to math/eager. `actual_cuda_kernel` therefore
remains unknown until an Nsight or equivalent kernel trace confirms it. Phase 1A
does not change backend configuration.

## Mac checks

Do not materialize production weights or run generation on a Mac. The CPU-safe
checks do not import Transformers or download models:

```bash
uvx ruff check experiments scripts dflash/phase1 tests
python -m compileall -q experiments scripts dflash/phase1 tests third_party/qwen3_8b_dflash_b16
python -m unittest discover -s tests -v
uvx --python 3.12 pytest -q
git diff --check
```

## H100 setup and materialization

Activate the Phase 0 environment, then define machine-local paths in the shell.
The output directories must be outside Git and on a filesystem that can reach
the original weight file:

```bash
cd "$DFLASH_REPO"
git switch codex/phase1-observability
git pull --ff-only origin codex/phase1-observability
git status --short --branch

export TARGET_MODEL_DIR="/path/to/Qwen3-8B-b968826d"
export DRAFT_WEIGHT_DIR="/path/to/Qwen3-8B-DFlash-b16"
export DFLASH_ORIGINAL_DIR="/path/outside/git/dflash-original-9b41424"
export DFLASH_INSTRUMENTED_DIR="/path/outside/git/dflash-instrumented-9b41424"
```

Materialize both paths from the same external weights:

```bash
python scripts/materialize_dflash.py \
  --source-model-path "$DRAFT_WEIGHT_DIR" \
  --output-dir "$DFLASH_ORIGINAL_DIR" \
  --implementation original

python scripts/materialize_dflash.py \
  --source-model-path "$DRAFT_WEIGHT_DIR" \
  --output-dir "$DFLASH_INSTRUMENTED_DIR" \
  --implementation instrumented
```

The commands fail rather than fall back if source code, revision metadata,
weight identity, an output hash, or a symlink target is wrong.

## Token-exact original/instrumented A/B

This one command compares the same input IDs across original, instrumented
default, diagnostics-on, and profiling-on paths. It requires equality of output
IDs and decoded text and requires the original/default return types to match:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python experiments/phase1_observability.py \
  --run-mode ab \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path "$TARGET_MODEL_DIR" \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --draft-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
  --draft-model-path "$DFLASH_ORIGINAL_DIR" \
  --materialized-draft-path "$DFLASH_INSTRUMENTED_DIR" \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-new-tokens 32 \
  --prompt "How many positive whole-number divisors does 196 have?" \
  --output-dir outputs/phase1-ab
```

Decoded-text equality is reported in addition to, never instead of, exact
`torch.equal` token-ID checks.

## Observability measurement

Run warmups separately from paired default-control and instrumented
measurements:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python experiments/phase1_observability.py \
  --run-mode observe \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path "$TARGET_MODEL_DIR" \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --draft-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
  --materialized-draft-path "$DFLASH_INSTRUMENTED_DIR" \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-new-tokens 32 \
  --warmup-runs 2 \
  --measurement-runs 5 \
  --enable-diagnostics \
  --enable-profiling \
  --prompt "How many positive whole-number divisors does 196 have?" \
  --output-dir outputs/phase1-observability
```

Expected outputs are:

- `summary.json`: configuration, output IDs/text, timings, overhead, equality,
  and attention backend provenance;
- `rounds.jsonl`: one validated record per speculative round and measurement;
- `environment.json`: Git, software, CUDA, GPU, and run configuration;
- `source-provenance.json`: repository commit, all custom-code hashes, weight
  provenance, materialized path, actual module path/hash, and wrapped/unwrapped
  `spec_generate` definition, line, qualified name, and source hash;
- `terminal.log`: mirrored stdout/stderr, including explicit failures.

Interrupted and failed runs still write all five artifacts.

## Optional Nsight Systems capture

Nsight is GPU-server-only. Raw traces belong outside Git (the repository ignores
Nsight outputs):

```bash
export DFLASH_NSYS_DIR="/path/outside/git/nsight"
mkdir -p "$DFLASH_NSYS_DIR"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output "$DFLASH_NSYS_DIR/dflash-phase1" \
  python experiments/phase1_observability.py \
    --run-mode observe \
    --target-model-path "$TARGET_MODEL_DIR" \
    --materialized-draft-path "$DFLASH_INSTRUMENTED_DIR" \
    --device cuda:0 \
    --dtype bfloat16 \
    --max-new-tokens 32 \
    --warmup-runs 1 \
    --measurement-runs 1 \
    --enable-diagnostics \
    --enable-profiling \
    --output-dir outputs/phase1-nsys
```

## H100-only acceptance criteria and limitations

The following cannot be established on the Mac and remain H100 validation work:

- materialization against the real 2.097 GB weight and its local metadata;
- successful Transformers dynamic-module loading of both materializations;
- actual wrapped/unwrapped source paths and source-hash equality on the server;
- token-ID equality for all four A/B paths;
- round accounting against real generation, CUDA-event timing, memory counters,
  NVTX visibility, and instrumentation overhead;
- the concrete target/draft attention implementations selected in that runtime;
- the actual SDPA/eager CUDA kernel, which requires a kernel trace.

Single-request timings are observability checks, not performance conclusions.
They should not be compared with the un-warmed Phase 0 4.05-second run. Injected
context and block-local attention remain fused in one attention dispatch, so
Phase 1A cannot independently attribute their latency.
