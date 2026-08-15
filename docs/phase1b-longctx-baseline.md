# Phase 1B: DFlash long-context baseline profiling

Phase 1B.1 generalizes the workload and adds model-free inventory preflight.
For current LongBench commands and cohort semantics, see
`docs/phase1b1-longbench-workloads.md`. The CUDA instrumentation described here
is unchanged by Phase 1B.1.

## Scope

Phase 1B is a measurement-only extension of the H100-validated Phase 1A baseline
at commit `213250cb0fef2caa088e24fa467fd7affaa9a6f4`. It adds a deterministic
GovReport workload, combined draft-attention CUDA timing, and sweep aggregation.
It does not change generation, attention masks, sampling, target verification,
model weights, or cache crops. It does not implement sparse KV, tree candidates,
multi-candidate sampling, or hardware simulation.

The experiment is designed to answer three empirical questions after the H100
pilot has run:

1. whether draft latency grows materially with persistent context KV length;
2. what fraction of draft and round latency the combined attention call occupies
   at 16K and 32K;
3. how acceptance, effectively returned tokens, and formal end-to-end latency
   vary together at long context.

No answer to those questions is claimed from Mac tests or from the code alone.

## KV length semantics and invariants

`Qwen3DFlashAttention.forward` projects two sources into the K/V tensors passed
to a single attention dispatcher call:

- KV already present in `past_key_values_draft` before the layer update;
- K/V projected from this round's `target_hidden`;
- K/V projected from the current speculative query block.

The first two together are the persistent context visible to the block query.
On the first round the draft cache is empty and `target_hidden` contains the
prompt features. On later rounds the cache contains earlier committed context,
while `target_hidden` contains the newly committed features selected from the
previous target verification. Consequently the following fields mean:

- `injected_context_length`: compatibility alias for the old Phase 1A field. It
  is the number of newly supplied target hidden positions, not accumulated KV.
- `newly_injected_hidden_length`: exactly `target_hidden.shape[1]`.
- `cached_context_kv_length_before_injection`: per-layer draft-cache length
  before appending this call's projected K/V.
- `persistent_context_kv_length`: cached context plus newly injected context;
  equivalently, K/V length at the actual attention call excluding the query
  block.
- `query_length`: current speculative block length (16 for the pinned model).
- `effective_kv_length`: complete K/V length passed to the attention call.
- `current_block_kv_length`: retained Phase 1A field for newly injected hidden
  plus query length. It must not be interpreted as accumulated context.

Each recorded layer call is checked at runtime:

```text
persistent_context_kv_length == context_length_before
persistent_context_kv_length == cached_context_kv_length_before_injection
                                + newly_injected_hidden_length
effective_kv_length == persistent_context_kv_length + query_length
newly_injected_hidden_length == target_hidden.shape[1]
```

The final equality is true by construction at the point where the shape is
read. A violation of the other relations raises an error instead of emitting a
plausible-looking record. `kv_length_invariants` stores the successfully checked
relations. `theoretical_qk_score_elements` is
`batch_size * query_heads * query_length * effective_kv_length`; Q/K/V shapes,
KV head count, and head dimension remain in every attention-call record.

## Combined-attention timing boundary

Persistent-context KV and current-block KV enter the same `attn_fn(...)` call.
There is no source-grounded injected-KV-only boundary. Phase 1B therefore records
only `combined_attention_cuda_elapsed_ms`:

- a CUDA event pair directly surrounds the actual attention dispatcher/SDPA
  call in every draft layer;
- the existing `dflash.attention.combined_context_block` NVTX range surrounds
  the same call;
- projections, rotary embedding, cache update, output projection, MLP, target
  LM head, and verification are outside this per-layer boundary;
- no layer or round calls `torch.cuda.synchronize()`;
- all pending event pairs are resolved after generation with one explicit
  synchronization at that generation boundary.

The per-round value is the sum of its layer calls. Generation-level combined
attention is the sum across all layers and rounds. Ratios are reported as
combined attention divided by draft CUDA time and by round CUDA time. These are
nested CUDA-event regions, not a claim that SDPA launches exactly one kernel.
The sweep fails explicitly if the loaded draft configuration does not select the
PyTorch SDPA registry path.
The original acceptance `.item()` remains an algorithmic host dependency and
can imply a synchronization; Phase 1B does not alter it. Nsight is still needed
to identify the concrete SDPA kernel selected for each shape.

## Deterministic offline GovReport construction

`dflash.phase1b.workload` accepts exactly one local source:

- a local JSONL file;
- a directory previously written by `Dataset.save_to_disk`;
- an already populated Hugging Face datasets cache.

The cache path runs with `HF_DATASETS_OFFLINE=1` and `reuse_dataset_if_exists`;
missing data raises an error. JSONL and `save_to_disk` are the least ambiguous
offline inputs. Raw reports, Arrow/Parquet data, model files, and run outputs are
ignored by Git and must not be added to the repository.

The fixed template ID is `govreport-summary-v1`. The actual Qwen3 tokenizer is
called with the Qwen chat template, `add_generation_prompt=True`, and
`enable_thinking=False`. For each selected report, increasing character prefixes
are searched independently for 1K, 4K, 8K, 16K, and 32K templated tokens. The
same source report is used at every length and the report prefixes are checked
to be nested. Tokenizer boundary effects can make an exact nominal count
unreachable; the longest prefix at or below the nominal length is used, and both
`nominal_prompt_length` and `actual_prompt_length` plus
`exact_nominal_length` and the shortfall are recorded. A shortfall above 16
tokens fails workload construction instead of silently turning a nominal length
into a materially shorter prompt.

Reports are selected by a stable SHA-256 ordering derived from seed, split,
source sample ID, and full-text hash. Only reports whose original text is long
enough for the maximum requested length qualify. The builder never repeats text
and never appends filler. The manifest records template ID, split, seed, sample
IDs, source/prefix/prompt hashes, actual lengths, prefix character counts, and a
hash of the actual templated input-token IDs, but omits report text and
input-token arrays. Every actual prompt length plus
`max_new_tokens` is checked against the target model's
`max_position_embeddings`.

## Paired measurement protocol

For each sample/length/repeat, the runner executes in fixed order:

1. `instrumented_default_control`, with diagnostics and internal profiling off;
2. `diagnostics_profiling_path`, with accounting, NVTX, memory, and component
   CUDA events enabled;
3. an exact `torch.equal` comparison of the complete output token IDs.

Greedy repeats must also be token-exact. A mismatch fails the run. Formal
end-to-end CUDA latency and generated tokens/s come only from the default control
path. Prefill, decode, draft, combined attention, verification, round timing,
and peak memory are explicitly labeled as profiling-path measurements; the
profile total is retained for diagnosis but is not substituted for formal
latency. One default-path warmup is performed per context length unless
`--warmup-runs 0` is requested.

## Accounting and aggregation

`samples.jsonl` has one row per sample/context-length pair and retains all timing
repeats plus their mean, median, P90, standard deviation, and coefficient of
variation. `rounds.jsonl` has every profiling-path round for every repeat and
contains per-layer attention calls.

Raw all-round proposed, accepted, rejected, committed, and accepted-length
values are always retained. The primary acceptance rate, correction rate, and
full-block acceptance rate exclude terminal rounds and any round with returned
token truncation. This avoids treating a full speculative block executed beyond
the return budget as ordinary acceptance. The output states this policy in both
sample and summary records. Effective returned tokens per round uses returned
tokens over all executed rounds. Per-length output includes sample-level metric
distributions, round-level distributions, count-weighted micro averages, and
equal-weight sample-level macro averages.

## H100 setup

Start from the validated Phase 1A commit and activate the existing Transformers
environment:

```bash
cd "$DFLASH_REPO"
git fetch origin
git switch codex/phase1b-longctx-baseline
git pull --ff-only origin codex/phase1b-longctx-baseline
git status --short --branch

source .venv/bin/activate
python -m pip install -e '.[transformers]'

export TARGET_MODEL_DIR="/path/to/Qwen3-8B-b968826d"
export DRAFT_WEIGHT_DIR="/path/to/Qwen3-8B-DFlash-b16"
export DFLASH_INSTRUMENTED_DIR="/path/outside/git/dflash-instrumented-phase1b"
export GOVREPORT_JSONL="/path/outside/git/govreport.jsonl"
```

Phase 1B changes repository-controlled instrumentation, so rematerialize even if
a Phase 1A instrumented directory already exists:

```bash
python scripts/materialize_dflash.py \
  --source-model-path "$DRAFT_WEIGHT_DIR" \
  --output-dir "$DFLASH_INSTRUMENTED_DIR" \
  --implementation instrumented
```

### Minimal smoke: one sample at 1K and 32K

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
python experiments/phase1b_longctx.py \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path "$TARGET_MODEL_DIR" \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --draft-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
  --materialized-draft-path "$DFLASH_INSTRUMENTED_DIR" \
  --govreport-jsonl "$GOVREPORT_JSONL" \
  --dataset-split test \
  --context-lengths 1K,32K \
  --pilot-samples 1 \
  --timing-repeats 1 \
  --warmup-runs 1 \
  --max-new-tokens 128 \
  --device cuda:0 \
  --dtype bfloat16 \
  --output-dir outputs/phase1b-smoke
```

Use the split name corresponding to the local source. JSONL records should have
`report` and preferably `id`; otherwise pass `--text-field` and `--id-field`.

### Pilot: eight samples at all five lengths

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
python experiments/phase1b_longctx.py \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path "$TARGET_MODEL_DIR" \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --draft-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
  --materialized-draft-path "$DFLASH_INSTRUMENTED_DIR" \
  --govreport-jsonl "$GOVREPORT_JSONL" \
  --dataset-split test \
  --context-lengths 1K,4K,8K,16K,32K \
  --pilot-samples 8 \
  --timing-repeats 3 \
  --warmup-runs 1 \
  --max-new-tokens 128 \
  --device cuda:0 \
  --dtype bfloat16 \
  --output-dir outputs/phase1b-pilot
```

To use a saved dataset directory, replace `--govreport-jsonl ...` with
`--govreport-dataset-path /path/to/saved/govreport`. To use an existing datasets
cache, use `--govreport-dataset-cache /path/to/hf/datasets/cache` and, if needed,
override `--dataset-name` and `--dataset-config`.

## Output files

Every run creates all required paths, including on a caught error:

- `manifest.json`: frozen configuration, workload hashes/IDs, measurement policy,
  model position limit, attention backend, and artifact paths;
- `workload-inventory.json`: tokenizer-derived eligibility and pilot
  recommendations, produced before model loading;
- `samples.jsonl`: sample-level control timing, profiling components, output-ID
  equality, accounting, memory, and theoretical shapes;
- `rounds.jsonl`: repeat-level round accounting, component timing, and layer calls;
- `summary.json`: per-length and overall sample/round, micro/macro aggregates;
- `summary.csv`: compact per-length medians, means, P90s, and acceptance metrics;
- `environment.json`: Git, OS, Python, package, CUDA, GPU, and run configuration;
- `source-provenance.json`: pinned model provenance and actually loaded dynamic
  module paths/hashes;
- `terminal.log`: mirrored terminal output and explicit tracebacks.

Schemas live under `docs/schemas/phase1b-*.schema.json`. Partial results are
atomically checkpointed after each completed sample/length pair.

## Mac checks and H100-only validation

Mac checks must not load model frameworks or download data/models:

```bash
uvx ruff check experiments scripts dflash tests
python -m compileall -q experiments scripts dflash tests third_party/qwen3_8b_dflash_b16
python -m unittest discover -s tests -v
uvx --python 3.12 pytest -q
git diff --check
```

The following remain H100-only and are not asserted by CPU tests:

- rematerialization and dynamic-module hash/provenance equality with real weights;
- successful selection of enough genuine reports reaching a 32K templated prompt;
- actual nominal-versus-tokenizer lengths and position-limit validation;
- token-exact control/profile equality for every sample, length, and repeat;
- CUDA-event values, per-layer/round/generation combined-attention sums and ratios;
- peak allocator memory and latency repeat variability;
- NVTX visibility and concrete SDPA kernel identity in Nsight;
- the three empirical Phase 1B research conclusions stated at the start.
