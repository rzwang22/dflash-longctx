# Phase 2C: Online Bounded-Tree Verification

Phase 2C converts the Phase 2B/2B.1 marginal candidate-tree replay into an
explicitly enabled online DDTree/OPT-Tree-style baseline. It is a measurement
baseline, not the project's final algorithmic contribution. Sparse KV, sparse
target attention, branch-conditioned refinement, learned selection, stochastic
acceptance, custom kernels, and hardware simulation remain out of scope.

## Source-confirmed reference round semantics

The repository-controlled dynamic module is materialized from
`third_party/qwen3_8b_dflash_b16/instrumented/dflash.py`. Its default
`spec_generate` still dispatches to the AST-equivalent reference implementation
when no diagnostic or Phase 2C option is enabled.

The actual reference code establishes the following sequence:

1. Target prefill stores only the prompt in persistent target KV. Target prefill
   logits produce the first root, which is written to the output buffer but is
   not yet in persistent KV.
2. At a round start, `block_output_ids[:, 0]` is that pending target-produced
   root. Positions 1 through 15 are the draft proposals.
3. Target verification temporarily appends the root and all 15 proposals.
4. `accepted_draft_token_count` is the matching draft prefix and excludes root.
   The round commits exactly `root + accepted draft prefix`, so
   `committed_token_count = 1 + accepted_draft_token_count`. Accepted zero still
   commits root.
5. Target logits emit a correction after the first mismatch or a bonus after a
   fully accepted chain. That token is written to the output buffer as the next
   round's root. It is not part of the current round's committed target KV.
6. Reference `DynamicCache.crop(start)` retains prompt/previous commits plus the
   current root and accepted prefix. EOS and `max_new_tokens` are applied to the
   returned output after raw round accounting, so a terminal round can contain
   truncated raw tokens.

Phase 2C preserves the same state machine. The only difference is that target
verification temporarily appends a packed tree instead of one 16-token chain.
After traversal, path-aware KV gather retains the persistent prefix, root, and
accepted path in generated order. Correction/bonus remains pending until it is
input as the next root.

## Shared marginal policy

`dflash.phase2c.core.layerwise_beam_tree` is the canonical CPU definition used
by Phase 2B.1 replay and Phase 2C tests. The same source file is copied into the
materialized model as `phase2c_online.py`; its device-resident builder implements
the same layerwise expansion, score, and tie rules and is tested against the
canonical policy contract.

- `top1_chain`: root plus the exact 15 marginal Top-1 tokens, budget 16.
- `layerwise_beam`: fixed width 2, 4, 8, 16, or 32 and root-inclusive budget 16,
  32, 64, or 128.
- Maximum draft depth: 15.
- Score: mean cumulative marginal draft log probability. At one depth this is
  equivalent to cumulative score.
- Stable priority: higher normalized score, higher cumulative score, shorter
  depth, then lexicographically ascending token IDs.
- The builder is target-blind. Full-vocabulary logits are transient and are not
  saved; only Top-K IDs/probabilities and optional tree metadata are retained.

The tree is prefix-closed, deduplicates siblings by token ID, never exceeds the
requested root-inclusive budget, and records its actual node count without
padding.

## Flattening, mask, positions, and cache storage

Flattening uses root first and then deterministic depth/policy order. Every
parent precedes its children; parent indices, paths, scores, and per-depth counts
are recorded.

Packed target verification uses a four-dimensional additive tree mask. Each
query sees every persistent-prefix key, itself, and its ancestors. Siblings,
cousins, other branches, and future nodes are masked. A normal linear causal mask
is not substituted.

Transformers 4.57.1 Qwen3 treats three concepts separately:

- `position_ids` select RoPE semantics. Phase 2C uses
  `context_length_before + depth`, so nodes at the same depth share a position.
- `cache_position` describes packed storage indices. Phase 2C uses the contiguous
  range `prefix_length .. prefix_length + actual_node_count - 1`.
- Dynamic KV storage appends tensors along sequence dimension `-2`; its packed
  index does not define RoPE position.

Qwen3 returns a caller-provided 4D mask unchanged from mask preparation and passes
it to its configured attention interface. Phase 2C currently requires the target
to be configured for PyTorch SDPA and records that backend. The custom mask can
change the actual CUDA kernel; the runner records the kernel as unknown until an
Nsight run confirms it. Implementation references are the official Transformers
4.57.1 [Qwen3 source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py),
[masking utilities](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/masking_utils.py),
and [DynamicCache source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/cache_utils.py).

## Greedy traversal and KV compaction

Target argmax after the current parent selects a matching child if present. A
missing child produces a correction. Reaching a leaf produces a bonus from that
leaf's target logits. The result always commits root plus the accepted draft path.

Temporary packed KV is not cropped by length. For every target layer, Phase 2C
gathers:

```text
[0 .. prefix_length-1]
+ [prefix_length + root_node_index]
+ [prefix_length + each accepted flattened node index]
```

Sibling and rejected branch KV is discarded. All layers must have temporary
length `prefix + actual_tree_nodes` and compacted length
`prefix + 1 + accepted_draft_tokens`. Correction/bonus is not included.

## Gold checks

`--enable-gold-checks` is limited to budgets 16 and 32 and is excluded from
formal performance. On the first tree round it checks every packed node against
an independently run root-to-node linear path:

- greedy argmax equality;
- exact stable Top-8 target IDs;
- maximum absolute bf16 error at most 0.5;
- mean absolute bf16 error at most 0.02.

It also advances a clone of the compacted cache with the pending next root and
compares it with dense recomputation of the exact generated sequence. Greedy,
Top-K, numeric tolerance, every layer length, sibling removal, and absence of the
pending root from persistent KV are checked.

For `top1_chain,budget=16`, the runner additionally requires exact equality with
reference DFlash for output IDs, every accepted length, correction/bonus kind,
generated root offset, and committed KV length. Failure stops the run.

## Control, profiling, and interpretation

The formal control path disables diagnostics, gold checks, and tree metadata
serialization. An external pair of CUDA events measures the whole generation.
The separate profiling path records deferred CUDA events for:

```text
phase2c.draft
phase2c.topk
phase2c.tree_build
phase2c.mask_build
phase2c.verify
phase2c.traverse
phase2c.kv_compact
phase2c.round
```

There is one CUDA synchronization after generation to resolve all internal
events. Output IDs must match between reference, control, profiling, repeats, and
gold paths. Tree construction and traversal use device tensors; no full logits
are moved to CPU in the formal control path. JSON conversion occurs only after
profiling timing resolves.

Acceptance, verified-node count, verification overhead, round reduction,
end-to-end latency, throughput, and speedup are reported separately. Acceptance
alone is not evidence of speedup. Summary acceptance/accounting uses profiling
repeat zero so deterministic token counts are not multiplied by timing repeats;
component latency distributions use every profiling repeat and report mean,
median, and P90.

## Offline/online alignment

Optional `--phase2b-capture-dir` inputs are strictly preflighted, and
`--phase2b1-analysis-dir` supplies offline events after online generation. Neither
is visible to tree construction. Alignment requires exact equality of task,
sample, nominal context length, generated root offset, policy, node budget, and
beam width. Unmatched rounds are `not_comparable`, never mismatches. The output
reports matched/unmatched counts and agreement only over exact matches.

## Artifacts

Every run writes:

```text
manifest.json
workload-inventory.json
samples.jsonl
tree-rounds.jsonl
tree-summary.json
tree-summary.csv
correctness-summary.json
offline-online-comparison.json
offline-online-comparison.csv
environment.json
source-provenance.json
terminal.log
gold-logit-checks.jsonl
gold-kv-checks.jsonl
```

The two gold JSONL files are empty when gold checks are disabled. New JSON and
JSONL structures have Draft 2020-12 schemas under `docs/schemas/`.

## H100 reproduction

Phase 2C modifies instrumented custom code and adds a materialized dependency.
Create a new directory; do not overwrite the validated Phase 2B model:

```bash
cd ~/data/dflash-longctx
git fetch origin
git switch codex/phase2c-online-bounded-tree
git pull --ff-only origin codex/phase2c-online-bounded-tree
source .venv/bin/activate

python scripts/materialize_dflash.py \
  --source-model-path /mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16 \
  --output-dir /mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16-phase2c-instrumented \
  --implementation instrumented
```

Top-1 gold smoke:

```bash
python experiments/phase2c_online_tree.py \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path /mnt/sevenT/wrz_data/models/Qwen3-8B-b968826d \
  --materialized-draft-path /mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16-phase2c-instrumented \
  --longbench-root /mnt/sevenT/wrz_data/datasets/LongBench \
  --longbench-task gov_report \
  --sample-policy matched \
  --sample-id gov_report-fd287bbfd3bdbdc9 \
  --context-lengths 1K \
  --pilot-samples 1 \
  --max-new-tokens 32 \
  --warmup-runs 0 \
  --timing-repeats 1 \
  --tree-policy top1_chain \
  --node-budget 16 \
  --enable-gold-checks \
  --output-dir outputs/phase2c-gold-top1-1k
```

Layerwise gold smoke:

```bash
python experiments/phase2c_online_tree.py \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path /mnt/sevenT/wrz_data/models/Qwen3-8B-b968826d \
  --materialized-draft-path /mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16-phase2c-instrumented \
  --longbench-root /mnt/sevenT/wrz_data/datasets/LongBench \
  --longbench-task gov_report \
  --sample-policy matched \
  --sample-id gov_report-fd287bbfd3bdbdc9 \
  --context-lengths 1K \
  --pilot-samples 1 \
  --max-new-tokens 32 \
  --warmup-runs 0 \
  --timing-repeats 1 \
  --tree-policy layerwise_beam \
  --node-budget 16 --node-budget 32 \
  --beam-width 4 --beam-width 8 --beam-width 16 \
  --enable-gold-checks \
  --output-dir outputs/phase2c-gold-layerwise-1k
```

Formal pilots use `--warmup-runs 1 --timing-repeats 3`, omit gold checks, and
request both policies, budgets 16/32/64/128, and widths 4/8/16/32. Run one task
per output directory:

```bash
COMMON=(
  --target-model Qwen/Qwen3-8B
  --target-revision b968826d9c46dd6066d109eabc6255188de91218
  --target-model-path /mnt/sevenT/wrz_data/models/Qwen3-8B-b968826d
  --materialized-draft-path /mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16-phase2c-instrumented
  --longbench-root /mnt/sevenT/wrz_data/datasets/LongBench
  --sample-policy matched
  --max-new-tokens 128
  --warmup-runs 1
  --timing-repeats 3
  --tree-policy top1_chain
  --tree-policy layerwise_beam
  --node-budget 16 --node-budget 32 --node-budget 64 --node-budget 128
  --beam-width 4 --beam-width 8 --beam-width 16 --beam-width 32
)

python experiments/phase2c_online_tree.py "${COMMON[@]}" \
  --longbench-task gov_report \
  --context-lengths 1K,4K,8K,16K,32K \
  --pilot-samples 3 \
  --output-dir outputs/phase2c-pilot-gov-report-matched-3

python experiments/phase2c_online_tree.py "${COMMON[@]}" \
  --longbench-task qasper \
  --context-lengths 1K,4K,8K,16K \
  --pilot-samples 3 \
  --output-dir outputs/phase2c-pilot-qasper-matched-3

python experiments/phase2c_online_tree.py "${COMMON[@]}" \
  --longbench-task hotpotqa \
  --context-lengths 1K,4K,8K,16K \
  --pilot-samples 8 \
  --output-dir outputs/phase2c-pilot-hotpotqa-matched-8
```

Stop before the formal pilots if top1 round equality, packed logits, compacted
cache, final output equality, cache length, source hash, control/profile equality,
or the custom-mask backend check fails. Nsight is still required to identify the
actual target verification kernel selected for the custom 4D mask.
