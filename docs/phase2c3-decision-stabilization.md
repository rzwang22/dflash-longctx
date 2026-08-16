# Phase 2C.3: Online Tree Decision Stabilization

Phase 2C.3 addresses the H100 BF16 packed-tree verifier decision flip confirmed
in Phase 2C.2. It does not change the draft tree policy, attention mask, target
weights, sampling rule, or original DFlash control. The H100 result that motivated
this phase is a native packed argmax of `Actions` where the original linear
verification argmax is `Activities`.

## Modes and commit semantics

`off` preserves the Phase 2C.2 packed-tree output and packed-path KV commit.
`always` performs a canonical linear recheck before every commit. `margin` first
measures the native selected-path Top-1/Top-2 margins and rechecks when the
minimum is at or below `--decision-margin-threshold`. `shadow` performs the same
recheck for evidence but commits the native result.

The canonical pass starts from a clone of the round-entry target cache. Its query
always has 16 positions: root, the native-selected draft path, and safe tail
padding. It uses contiguous original-DFlash position IDs, implicit causal masking,
and the configured native target attention backend. Causality prevents tail
padding from affecting the root or selected path. A candidate mismatch emits the
canonical parent argmax as correction; full selected-path acceptance emits the
canonical leaf argmax as bonus.

When `always` or triggered `margin` commits the canonical result, the persistent
cache length is exactly:

```text
round_entry_prefix_length + 1 + canonical_accepted_draft_count
```

The root and accepted draft KV come from the canonical linear pass. Packed
siblings, any native over-accepted nodes, and the padded canonical tail are
discarded. Correction and bonus remain the next-round root and are not written
to the current persistent cache. Gold KV validation is allowed to run after a
native packed-logit mismatch when a canonical commit replaced that native path.

`always` is the selected-path token-exact correctness mode. `margin` is only an
empirical performance/correctness tradeoff until shadow evaluation observes zero
false negatives on the intended cohort. It must not be described as strictly
lossless from a single sample or threshold.

## Timing policy

CUDA events and NVTX ranges separately cover the packed verification, cache
snapshot, canonical recheck, native compaction, and canonical KV commit. Events
are resolved once after generation. The formal `always` and `margin` control
latencies include their actual online work.

For `shadow`, the formal timing control explicitly runs `off`; a separate
profiling path runs the shadow oracle. Thus shadow recheck time is reported but
is not hidden inside or subtracted from a native end-to-end measurement.

Target-only generation now passes an all-one attention mask for the batch-1,
unpadded prompt and clears inactive sampling-only generation knobs while keeping
`do_sample=False`. H100 must confirm token IDs are unchanged before and after this
warning-only fix.

## Artifacts

In addition to the Phase 2C.2 artifacts, every run creates:

```text
decision-stabilization-rounds.jsonl
decision-stabilization-summary.json
decision-stabilization-summary.csv
decision-margin-frontier.csv
canonical-recheck-events.jsonl
```

Round records contain native and canonical paths/depths, correction or bonus,
margins, trigger reason, decision equality, recheck shapes, canonical cache
length, committed-cache source, component timing, and explicit null/reason pairs
when no recheck ran. The margin frontier evaluates thresholds
`0.01,0.02,0.05,0.1,0.2,0.5,1.0` without selecting one from the known sample.

## H100 materialization and known-sample acceptance

The instrumented code and its materialized dependencies changed. Use a new
directory; do not overwrite any Phase 2C.2 materialization:

```bash
cd ~/data/dflash-longctx
git fetch origin
git switch codex/phase2c3-decision-stabilization
git pull --ff-only origin codex/phase2c3-decision-stabilization
source .venv/bin/activate

python scripts/materialize_dflash.py \
  --source-model-path /mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16 \
  --output-dir /mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16-phase2c3-instrumented \
  --implementation instrumented
```

Define the fixed known-sample command once:

```bash
COMMON=(
  --target-model Qwen/Qwen3-8B
  --target-revision b968826d9c46dd6066d109eabc6255188de91218
  --target-model-path /mnt/sevenT/wrz_data/models/Qwen3-8B-b968826d
  --materialized-draft-path /mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16-phase2c3-instrumented
  --longbench-root /mnt/sevenT/wrz_data/datasets/LongBench
  --longbench-task gov_report
  --sample-policy matched
  --sample-id gov_report-fd287bbfd3bdbdc9
  --context-lengths 1K
  --pilot-samples 1
  --max-new-tokens 32
  --warmup-runs 0
  --timing-repeats 1
  --tree-policy layerwise_beam
  --node-budget 16
  --beam-width 4
)
```

Run native, canonical, and shadow:

```bash
python experiments/phase2c_online_tree.py "${COMMON[@]}" \
  --decision-stabilization off \
  --output-dir outputs/phase2c3-known-off

python experiments/phase2c_online_tree.py "${COMMON[@]}" \
  --decision-stabilization always \
  --enable-gold-checks --gold-max-rounds 8 \
  --output-dir outputs/phase2c3-known-always

python experiments/phase2c_online_tree.py "${COMMON[@]}" \
  --decision-stabilization shadow \
  --output-dir outputs/phase2c3-known-shadow

for THRESHOLD in 0.05 0.1 0.2 0.5; do
  python experiments/phase2c_online_tree.py "${COMMON[@]}" \
    --decision-stabilization margin \
    --decision-margin-threshold "$THRESHOLD" \
    --output-dir "outputs/phase2c3-known-margin-${THRESHOLD}"
done
```

Do not proceed to performance pilots until `always` shows target-only, original
DFlash, and stabilized token IDs equal; canonical commit lengths and next-step KV
gold pass; and the `Activities`/`Actions` round is recorded as a corrected native
decision.

## Shadow cohort and later configurations

The matched-3 shadow threshold inventory is:

```bash
python experiments/phase2c_online_tree.py \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path /mnt/sevenT/wrz_data/models/Qwen3-8B-b968826d \
  --materialized-draft-path /mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16-phase2c3-instrumented \
  --longbench-root /mnt/sevenT/wrz_data/datasets/LongBench \
  --longbench-task gov_report \
  --sample-policy matched \
  --context-lengths 1K,4K,8K,16K,32K \
  --pilot-samples 3 \
  --max-new-tokens 64 \
  --warmup-runs 1 \
  --timing-repeats 1 \
  --tree-policy layerwise_beam \
  --node-budget 16 \
  --beam-width 4 \
  --decision-stabilization shadow \
  --output-dir outputs/phase2c3-shadow-gov-report-matched-3
```

Choose a margin threshold only from this frontier, then run it with the same
cohort and require output-token equality and zero observed false negatives. After
that, extend `always` and the selected empirical margin mode to `b16-w8`,
`b16-w16`, `b32-w4`, and `b32-w8`. Formal three-task/five-length pilots remain
blocked until these correctness gates pass.

Mac tests validate pure decisions, cache invariants, schemas, CLI behavior,
artifact persistence, lazy imports, and aggregation. They do not establish H100
token equality, CUDA timing, BF16 margins, actual attention kernels, or KV/logit
gold results.
