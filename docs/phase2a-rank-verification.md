# Phase 2A: target-rank retention and verification breakdown

Phase 2A adds two mutually exclusive, opt-in diagnostic modes to the pinned
DFlash baseline. It does not change sampling, attention masks, KV-cache crops,
target verification, model weights, or the default `spec_generate` return type.
With every diagnostic flag off, `spec_generate` still dispatches to the
token-exact Phase 1B.1 reference body.

This phase measures two questions only:

1. At the first draft Top-1 mismatch, where is the target-selected token ranked
   in the existing draft distribution?
2. Within target verification, how much CUDA time is spent in whole
   self-attention modules, MLP modules, the LM head, and the remainder?

It does not implement candidate trees or infer their accepted length, and it
does not implement any optimization.

## Rank definition and interpretation

Rank mode is restricted to batch size 1, temperature 0, and the pinned block
size 16:

```text
draft_top1_j = argmax(draft_logits_j)
target_token_j = argmax(target_verification_logits_j)
rank_j = 1-based position of target_token_j in draft_logits_j
```

Vocabulary entries are ordered by descending logit. Exact ties are ordered by
ascending token ID, making rank deterministic. The implementation computes the
exact rank against the full on-device vocabulary but transfers and saves only
scalars plus the leading `--rank-max-m` token IDs and logits. It never persists
full-vocabulary logits. `target_logit` means the target token's logit in the
draft distribution; `target_verification_logit` is retained separately.

Each round's 15 draft positions are classified as:

- `accepted_prefix`: positions before the first mismatch;
- `first_rejection_frontier`: the single first mismatch in a rejected round;
- `post_rejection_counterfactual`: later logits conditioned on the erroneous,
  uncommitted draft prefix;
- `full_block_accepted`: every draft position in a round with all 15 accepted.

The target correction/bonus token is not one of the 15 draft positions and is
never ranked. The primary metric is first-rejection-frontier Top-M recall for
M=2, 4, 8, and 16. Top-1 at that frontier is zero by definition and is explicitly
marked as not being a recovery metric. Micro statistics pool frontier events;
sample-level macro statistics average each sample's frontier recall with equal
weight. Rank summaries also group by task/context, task, sample, and block
position and retain frontier rank percentiles and buckets.

These measurements support only a `single-frontier recoverability` statement.
Independent ranks after a mismatch cannot be combined into a candidate-tree
accepted length because those positions would have to be recomputed after a
corrected token. Every event and summary records
`tree_accepted_length_derived=false`.

Rank instrumentation consumes the draft and target logits already produced in
the current round; it does not run another draft or target forward. Each run
pairs an uninstrumented control generation with a rank generation and requires
complete output-token-ID equality. Rank-path elapsed time is not used as formal
latency.

## Verification module timing boundary

`--run-mode verification-profile` is separate from rank mode. Temporary target
forward hooks create CUDA event pairs and stable NVTX ranges for:

```text
dflash.verify
target.verify.layer.{i}.self_attn
target.verify.layer.{i}.mlp
target.verify.lm_head
```

Prefill uses corresponding `target.prefill.*` ranges and records, but it is kept
out of the primary verification aggregates. Existing `dflash.draft` and
`dflash.attention.combined_context_block` ranges remain available.

The self-attention module interval includes Q/K/V projections, rotary/cache and
SDPA work, and the output projection. It is not pure SDPA time. MLP and LM-head
hooks are sibling intervals, so the profiler compares their sum with complete
verification time. A nonnegative remainder with the expected hook call counts
is emitted as `target_unattributed_cuda_ms`; otherwise the public remainder is
`null`, the raw difference is retained, and `partition_reliable=false` records
why the partition must not be trusted. The remainder can include layer norms,
residual/embedding/final-normalization work, cache bookkeeping, and other work
outside the three hooked module types.

No layer or round synchronizes CUDA. All module, round, and generation event
pairs are resolved after generation using the one existing generation-boundary
synchronization. Raw records provide each layer invocation; round records
include per-layer and per-round aggregates; each sample's diagnostic repeat
contains per-generation totals and per-layer aggregates. Formal end-to-end
latency still comes from the paired diagnostic-free control generation.

Python module timing can answer whether the whole target self-attention module
or whole MLP module dominates. It cannot separate SDPA kernels from projections.
Only the Nsight/NCU kernel evidence described below can support a pure-SDPA
fraction.

## H100 setup and mandatory rematerialization

Phase 2A changes both repository-controlled files under
`third_party/qwen3_8b_dflash_b16/instrumented/`. An older Phase 1A/1B
materialization will intentionally fail provenance validation. Materialize the
Phase 2A code into a new directory; do not overwrite or reuse the validated
older directory.

```bash
cd ~/data/dflash-longctx
git fetch origin
git switch codex/phase2a-rank-verification-profile
git pull --ff-only origin codex/phase2a-rank-verification-profile
git status --short --branch

source .venv/bin/activate
python -m pip install -e '.[transformers]'

export TARGET_MODEL_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-b968826d
export DRAFT_WEIGHT_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16
export DFLASH_PHASE2A_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16-phase2a-instrumented
export LONGBENCH_ROOT=/mnt/sevenT/wrz_data/datasets/LongBench

python scripts/materialize_dflash.py \
  --source-model-path "$DRAFT_WEIGHT_DIR" \
  --output-dir "$DFLASH_PHASE2A_DIR" \
  --implementation instrumented
```

Run the Phase 1B.1 inventory-only command for each task before generation. The
commands and fields are documented in `docs/phase1b1-longbench-workloads.md`.
The sample counts below are prior recommendations, not hard-coded assumptions;
the actual inventory must still confirm them.

## Minimum rank smoke

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
python experiments/phase2a_profile.py \
  --run-mode rank \
  --rank-top-ms 2,4,8,16 \
  --rank-max-m 16 \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path "$TARGET_MODEL_DIR" \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --draft-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
  --materialized-draft-path "$DFLASH_PHASE2A_DIR" \
  --longbench-root "$LONGBENCH_ROOT" \
  --longbench-task gov_report \
  --sample-policy matched \
  --context-lengths 1K \
  --pilot-samples 1 \
  --timing-repeats 1 \
  --warmup-runs 1 \
  --max-new-tokens 32 \
  --device cuda:0 \
  --dtype bfloat16 \
  --output-dir outputs/phase2a-rank-smoke
```

## Formal rank runs for the three tasks

After each inventory confirms eligibility, run one task per output directory.
The common pinned model arguments are repeated to keep each command auditable.

GovReport, recommended matched three through 32K:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python experiments/phase2a_profile.py --run-mode rank \
  --rank-top-ms 2,4,8,16 --rank-max-m 16 \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path "$TARGET_MODEL_DIR" \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --draft-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
  --materialized-draft-path "$DFLASH_PHASE2A_DIR" \
  --longbench-root "$LONGBENCH_ROOT" --longbench-task gov_report \
  --sample-policy matched --context-lengths 1K,4K,8K,16K,32K \
  --pilot-samples 3 --timing-repeats 1 --warmup-runs 1 \
  --max-new-tokens 128 --device cuda:0 --dtype bfloat16 \
  --output-dir outputs/phase2a-rank-gov-report-matched-3
```

Qasper, recommended matched three through 16K:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python experiments/phase2a_profile.py --run-mode rank \
  --rank-top-ms 2,4,8,16 --rank-max-m 16 \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path "$TARGET_MODEL_DIR" \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --draft-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
  --materialized-draft-path "$DFLASH_PHASE2A_DIR" \
  --longbench-root "$LONGBENCH_ROOT" --longbench-task qasper \
  --sample-policy matched --context-lengths 1K,4K,8K,16K \
  --pilot-samples 3 --timing-repeats 1 --warmup-runs 1 \
  --max-new-tokens 128 --device cuda:0 --dtype bfloat16 \
  --output-dir outputs/phase2a-rank-qasper-matched-3
```

HotpotQA, recommended matched eight through 16K:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python experiments/phase2a_profile.py --run-mode rank \
  --rank-top-ms 2,4,8,16 --rank-max-m 16 \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path "$TARGET_MODEL_DIR" \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --draft-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
  --materialized-draft-path "$DFLASH_PHASE2A_DIR" \
  --longbench-root "$LONGBENCH_ROOT" --longbench-task hotpotqa \
  --sample-policy matched --context-lengths 1K,4K,8K,16K \
  --pilot-samples 8 --timing-repeats 1 --warmup-runs 1 \
  --max-new-tokens 128 --device cuda:0 --dtype bfloat16 \
  --output-dir outputs/phase2a-rank-hotpotqa-matched-8
```

## Minimum verification-profile run

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
python experiments/phase2a_profile.py \
  --run-mode verification-profile \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path "$TARGET_MODEL_DIR" \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --draft-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
  --materialized-draft-path "$DFLASH_PHASE2A_DIR" \
  --longbench-root "$LONGBENCH_ROOT" \
  --longbench-task gov_report \
  --sample-policy matched \
  --context-lengths 1K \
  --pilot-samples 1 \
  --timing-repeats 1 \
  --warmup-runs 1 \
  --max-new-tokens 32 \
  --device cuda:0 \
  --dtype bfloat16 \
  --output-dir outputs/phase2a-verification-smoke
```

## Nsight Systems traces at 1K, 16K, and 32K

Nsight traces are kernel-characterization runs only: one sample, 32 output
tokens, one repeat, and one warmup. They are not acceptance experiments. Create
one trace per context length:

```bash
mkdir -p outputs/phase2a-nsight/{1k,16k,32k}

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true \
  -o outputs/phase2a-nsight/1k/trace \
  python experiments/phase2a_profile.py --run-mode verification-profile \
  --target-model-path "$TARGET_MODEL_DIR" \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --materialized-draft-path "$DFLASH_PHASE2A_DIR" \
  --longbench-root "$LONGBENCH_ROOT" --longbench-task gov_report \
  --sample-policy matched --context-lengths 1K --pilot-samples 1 \
  --timing-repeats 1 --warmup-runs 1 --max-new-tokens 32 \
  --device cuda:0 --dtype bfloat16 \
  --output-dir outputs/phase2a-nsight/1k/run

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true \
  -o outputs/phase2a-nsight/16k/trace \
  python experiments/phase2a_profile.py --run-mode verification-profile \
  --target-model-path "$TARGET_MODEL_DIR" \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --materialized-draft-path "$DFLASH_PHASE2A_DIR" \
  --longbench-root "$LONGBENCH_ROOT" --longbench-task gov_report \
  --sample-policy matched --context-lengths 16K --pilot-samples 1 \
  --timing-repeats 1 --warmup-runs 1 --max-new-tokens 32 \
  --device cuda:0 --dtype bfloat16 \
  --output-dir outputs/phase2a-nsight/16k/run

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true \
  -o outputs/phase2a-nsight/32k/trace \
  python experiments/phase2a_profile.py --run-mode verification-profile \
  --target-model-path "$TARGET_MODEL_DIR" \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --materialized-draft-path "$DFLASH_PHASE2A_DIR" \
  --longbench-root "$LONGBENCH_ROOT" --longbench-task gov_report \
  --sample-policy matched --context-lengths 32K --pilot-samples 1 \
  --timing-repeats 1 --warmup-runs 1 --max-new-tokens 32 \
  --device cuda:0 --dtype bfloat16 \
  --output-dir outputs/phase2a-nsight/32k/run
```

Export and parse each report:

```bash
python scripts/phase2a_nsight.py export \
  --nsys-rep outputs/phase2a-nsight/1k/trace.nsys-rep \
  --output-dir outputs/phase2a-nsight/1k/parsed
python scripts/phase2a_nsight.py export \
  --nsys-rep outputs/phase2a-nsight/16k/trace.nsys-rep \
  --output-dir outputs/phase2a-nsight/16k/parsed
python scripts/phase2a_nsight.py export \
  --nsys-rep outputs/phase2a-nsight/32k/trace.nsys-rep \
  --output-dir outputs/phase2a-nsight/32k/parsed
```

The exporter uses the official Nsight Systems `nvtx_kern_sum` and
`nvtx_gpu_proj_sum` reports. NVIDIA documents that the first associates kernels
with enclosing NVTX ranges based on the kernel launch API, while the second
projects CPU NVTX ranges onto GPU operations. Nested ranges may therefore list
the same kernel under both a parent and a child scope; the parser records this
explicitly and does not add parent and child totals together. See the
[Nsight Systems Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html).

Kernel categories use published regular-expression rules and always retain the
raw kernel name, rule ID, pattern, and resulting category. The rules are a
transparent first pass, not proof of kernel semantics. Actual H100 kernel names
and every classification require review on the server.

## Output files

Rank mode writes:

```text
manifest.json
samples.jsonl
rank-events.jsonl
rank-summary.json
rank-summary.csv
environment.json
source-provenance.json
terminal.log
```

Verification-profile mode writes:

```text
manifest.json
samples.jsonl
verification-module-events.jsonl
verification-profile-rounds.jsonl
verification-profile-summary.json
verification-profile-summary.csv
environment.json
source-provenance.json
terminal.log
```

Nsight parsing writes:

```text
nsight-kernel-summary.csv
nsight-nvtx-summary.csv
verification-kernel-breakdown.json
```

All generation modes rerun LongBench inventory/cohort preflight before CUDA or
8B model loading, persist the selected sample IDs and prompt hashes, and fail
instead of silently shrinking an insufficient matched cohort.

## H100 validation still required

Mac tests validate pure rank/accounting logic, mode isolation, hooks with fake
modules/events, schemas, lazy imports, and a synthetic Nsight parser fixture.
They make no GPU claim. H100 must still confirm:

- Phase 2A materialization and loaded-source hashes;
- default/rank and default/profile output-token equality with real models;
- exact alignment of draft positions with target verification logits;
- hook call-count and nonnegative-remainder invariants for real Qwen3 layers;
- one internal generation-boundary synchronization and valid CUDA event times;
- real LongBench eligible cohorts and position limits;
- rank distributions and their context-length trend;
- module-level verification fractions and repeat variability;
- actual NVTX nesting, kernel names, category review, and the pure-SDPA share at
  1K, 16K, and 32K.

No rank, latency, acceptance, or kernel-dominance conclusion is claimed until
those H100 runs are complete.
