# Phase 2B: bounded candidate-tree offline feasibility

Phase 2B asks whether the 15 static per-position candidate distributions already
produced by DFlash can cover a longer verified target path under a bounded,
prefix-closed tree. It does not implement target tree verification, a tree
attention mask, sparse KV, sequential reconditioning, hardware simulation, or
any optimization. Consequently, all reported tree results are offline coverage
statistics, not accepted lengths, latency, round reductions, or speedups.

## Source-confirmed token alignment

The pinned `DFlashDraftModel.spec_generate` and
`DFlashDraftModel._spec_generate_observed` in
`third_party/qwen3_8b_dflash_b16/instrumented/dflash.py` have the following exact
semantics.

1. Target prefill selects one token at absolute output index `prompt_length`.
   That token is the first round's root/seed.
2. A block has size 16. `block_output_ids[:, 0]` is the already-produced target
   root; `block_output_ids[:, 1:16]` contains 15 draft proposals.
3. The final logits used for those 15 proposals are produced by
   `target.lm_head(draft_hidden[:, -block_size + 1:, :])`. At temperature zero,
   the 15 sampled IDs are the Top-1 IDs of these exact logits.
4. Target verification selects `posterior` from the block target forward. The
   acceptance comparison is `block_output_ids[:, 1:] == posterior[:, :-1]` and
   `accepted_draft_token_count` is its longest matching prefix, from 0 through
   15. The seed is excluded.
5. Before a round, `start` is both the root's absolute output index and
   `context_length_before`. Its index in the generated-only output is
   `start - prompt_length`; the reference continuation therefore starts at
   `start - prompt_length + 1`.
6. Commit writes the root plus the accepted proposals, then writes the target
   correction/bonus at the next position. That correction/bonus becomes the
   next round's root.

The capture checks two alignment gates. Inside the same generation, the LCP of
the 15 captured Top-1 IDs and `posterior[:, :-1]` must equal the accepted count.
After generation, an uncensored round must satisfy:

```text
LCP(Top-1 draft IDs, final verified continuation after root)
  == accepted_draft_token_count
```

Any failure includes the round, mismatch position, draft IDs, reference IDs,
and root offset and aborts the run. For a terminal round ending inside its
accepted prefix, final output cannot reveal the later mismatch. Such a round is
marked `right_censored_observed_prefix_consistent`; every observable token must
match, no hidden token is imputed, and the round is excluded from full-depth
denominators. This is the only accurate interpretation of the invariant under
EOS or `max_new_tokens` censoring.

## Capture boundary and artifacts

`experiments/phase2b_tree_capture.py` runs each workload in this order:

1. the diagnostics-free reference control;
2. the isolated candidate capture;
3. `torch.equal` over the complete output IDs.

The capture sorts the exact final draft logits by descending logit and ascending
token ID for exact ties, applies `log_softmax` for normalized scores, and moves
only 15 × Top-K IDs/scores to CPU. It introduces no draft or target forward.
Full-vocabulary logits, target logits, hidden states, weights, and KV-cache
contents are never persisted. A mismatch fails the run and leaves only error
and partial evidence, not a successful replay summary.

Successful capture writes:

```text
manifest.json
samples.jsonl
tree-capture-rounds.jsonl
environment.json
source-provenance.json
terminal.log
```

The manifest/environment record the Git SHA and model revisions. Provenance
records the materialized and actually imported Python paths and SHA-256 hashes.
Each round records task/context/sample identity and hashes, repeat/round/root
offsets, the root, 15 baseline IDs, Top-K IDs/ranks/log probabilities, exact
round accounting, final verified generated IDs, reference/censoring metadata,
and both alignment checks.

## Tree and budget definition

The tree has one root and at most 15 draft depths. A non-root node is identified
by its complete token-ID prefix, so equal prefixes are stored once. Every parent
must exist before a child. Budgets are maximum total nodes including the root:

| Total nodes | Candidate nodes | Meaning |
| ---: | ---: | --- |
| 16 | 15 | exact Top-1 baseline chain |
| 32 | 31 | root plus at most 31 candidate-prefix nodes |
| 64 | 63 | root plus at most 63 candidate-prefix nodes |

No `16^15` Cartesian product is materialized. The recorded per-position logits
are static parallel DFlash outputs: after a branch differs from Top-1, later
position distributions have not been reconditioned on that branch. This is the
central limitation the oracle/deployable gap is intended to expose.

## Offline policies

`scripts/phase2b_tree_replay.py` imports neither Torch nor Transformers and can
replay one capture under many configurations.

- `top1_chain` is deployable and uses no target information. It always retains
  the root plus the 15 Top-1 prefixes; unused budget stays unused.
- `score_beam` is deployable and uses no target information. It first retains
  the Top-1 backbone, then performs deterministic best-first expansion ranked
  by mean cumulative draft log probability. Ties use cumulative score, depth,
  then lexicographically ascending token IDs.
- `confidence_gated` is deployable and uses no target/rejection information. It
  starts from the Top-1 backbone and permits extra ranks only where the draft
  Top-1 minus Top-2 log-probability margin is at most the configured threshold.
  Eligible nodes use the same deterministic length-normalized best-first
  allocation as `score_beam`. Thresholds and branch width are replay-time
  parameters.
- `oracle_prefix_closed` is non-deployable and target-informed. Under the same
  budget it retains the longest candidate-available reference prefix. It is a
  structural coverage upper bound only and is never pooled with deployable
  policy averages. Tiny synthetic instances are checked against exhaustive
  prefix-closed enumeration.

Each policy event reports coverage depth, the coverage-derived
`committed_tokens_upper_bound = 1 + depth`, improvement over Top-1, first
rejection recovery, depth 1/2/4/8/15 recall, full-depth coverage, tree shape,
both node-count conventions, prefix sharing, target coverage per candidate,
oracle gap, and censoring. Aggregates contain round micro and sample macro
statistics, mean/median/P90, task/context/budget/policy breakdowns, improvement
fractions, depth recall, oracle-only opportunities, and explicit terminal/full-
depth exclusion counts. No round-independent confidence interval is emitted.

Replay writes:

```text
tree-policy-events.jsonl
tree-summary.json
tree-summary.csv
tree-depth-coverage.csv
tree-policy-config.json
replay-manifest.json
terminal.log
```

Schemas are under `docs/schemas/phase2b-*.schema.json` and use JSON Schema Draft
2020-12.

## H100 setup and mandatory rematerialization

Phase 2B minimally changes the two repository-controlled instrumented files to
expose candidates from the exact final draft logits. Default diagnostics still
dispatches to the unchanged reference body, but a Phase 2A materialization has
old source hashes. Materialize this branch into a new directory; do not
overwrite the validated Phase 2A directory.

```bash
cd ~/data/dflash-longctx
git fetch origin
git switch codex/phase2b-bounded-tree-oracle
git pull --ff-only origin codex/phase2b-bounded-tree-oracle
git status --short --branch

source .venv/bin/activate
python -m pip install -e '.[transformers]'

export TARGET_MODEL_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-b968826d
export DRAFT_WEIGHT_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16
export DFLASH_PHASE2B_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16-phase2b-instrumented
export LONGBENCH_ROOT=/mnt/sevenT/wrz_data/datasets/LongBench

python scripts/materialize_dflash.py \
  --source-model-path "$DRAFT_WEIGHT_DIR" \
  --output-dir "$DFLASH_PHASE2B_DIR" \
  --implementation instrumented
```

### Fixed-sample smoke: GovReport 1K

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python experiments/phase2b_tree_capture.py \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path "$TARGET_MODEL_DIR" \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --draft-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
  --materialized-draft-path "$DFLASH_PHASE2B_DIR" \
  --longbench-root "$LONGBENCH_ROOT" --longbench-task gov_report \
  --sample-policy matched --sample-id gov_report-fd287bbfd3bdbdc9 \
  --context-lengths 1K --pilot-samples 1 \
  --timing-repeats 1 --warmup-runs 1 --max-new-tokens 32 --top-k 16 \
  --device cuda:0 --dtype bfloat16 \
  --output-dir outputs/phase2b-capture-smoke-gov-report-1k

python scripts/phase2b_tree_replay.py \
  --capture-rounds outputs/phase2b-capture-smoke-gov-report-1k/tree-capture-rounds.jsonl \
  --capture-manifest outputs/phase2b-capture-smoke-gov-report-1k/manifest.json \
  --node-budgets 16,32,64 \
  --confidence-thresholds 0.25,0.5,1.0 \
  --confidence-branch-width 4 \
  --output-dir outputs/phase2b-replay-smoke-gov-report-1k
```

### Formal one-time captures

GovReport matched three, 1K through 32K:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python experiments/phase2b_tree_capture.py \
  --target-model Qwen/Qwen3-8B --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path "$TARGET_MODEL_DIR" \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --draft-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
  --materialized-draft-path "$DFLASH_PHASE2B_DIR" \
  --longbench-root "$LONGBENCH_ROOT" --longbench-task gov_report \
  --sample-policy matched --context-lengths 1K,4K,8K,16K,32K \
  --pilot-samples 3 --timing-repeats 1 --warmup-runs 1 \
  --max-new-tokens 128 --top-k 16 --device cuda:0 --dtype bfloat16 \
  --output-dir outputs/phase2b-capture-gov-report-matched-3

python scripts/phase2b_tree_replay.py \
  --capture-rounds outputs/phase2b-capture-gov-report-matched-3/tree-capture-rounds.jsonl \
  --capture-manifest outputs/phase2b-capture-gov-report-matched-3/manifest.json \
  --node-budgets 16,32,64 --confidence-thresholds 0.25,0.5,1.0 \
  --confidence-branch-width 4 \
  --output-dir outputs/phase2b-replay-gov-report-matched-3
```

Qasper matched three, 1K through 16K:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python experiments/phase2b_tree_capture.py \
  --target-model Qwen/Qwen3-8B --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path "$TARGET_MODEL_DIR" \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --draft-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
  --materialized-draft-path "$DFLASH_PHASE2B_DIR" \
  --longbench-root "$LONGBENCH_ROOT" --longbench-task qasper \
  --sample-policy matched --context-lengths 1K,4K,8K,16K \
  --pilot-samples 3 --timing-repeats 1 --warmup-runs 1 \
  --max-new-tokens 128 --top-k 16 --device cuda:0 --dtype bfloat16 \
  --output-dir outputs/phase2b-capture-qasper-matched-3

python scripts/phase2b_tree_replay.py \
  --capture-rounds outputs/phase2b-capture-qasper-matched-3/tree-capture-rounds.jsonl \
  --capture-manifest outputs/phase2b-capture-qasper-matched-3/manifest.json \
  --node-budgets 16,32,64 --confidence-thresholds 0.25,0.5,1.0 \
  --confidence-branch-width 4 \
  --output-dir outputs/phase2b-replay-qasper-matched-3
```

HotpotQA matched eight, 1K through 16K:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python experiments/phase2b_tree_capture.py \
  --target-model Qwen/Qwen3-8B --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path "$TARGET_MODEL_DIR" \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --draft-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
  --materialized-draft-path "$DFLASH_PHASE2B_DIR" \
  --longbench-root "$LONGBENCH_ROOT" --longbench-task hotpotqa \
  --sample-policy matched --context-lengths 1K,4K,8K,16K \
  --pilot-samples 8 --timing-repeats 1 --warmup-runs 1 \
  --max-new-tokens 128 --top-k 16 --device cuda:0 --dtype bfloat16 \
  --output-dir outputs/phase2b-capture-hotpotqa-matched-8

python scripts/phase2b_tree_replay.py \
  --capture-rounds outputs/phase2b-capture-hotpotqa-matched-8/tree-capture-rounds.jsonl \
  --capture-manifest outputs/phase2b-capture-hotpotqa-matched-8/manifest.json \
  --node-budgets 16,32,64 --confidence-thresholds 0.25,0.5,1.0 \
  --confidence-branch-width 4 \
  --output-dir outputs/phase2b-replay-hotpotqa-matched-8
```

These commands run the two 8B models only once per capture configuration. Any
additional budget, score, or confidence-threshold experiment must reuse the
same `tree-capture-rounds.jsonl` through CPU replay.

## H100 checks still required

- Confirm the new materialization hashes and actual imported file paths match
  the Phase 2B branch.
- Confirm complete control/capture token equality at 1K and then every formal
  task/length/sample.
- Confirm the immediate verification LCP and final-output alignment gates for
  every uncensored round, plus honest EOS/max-token censoring on terminal rounds.
- Confirm Top-16 extraction fits H100 memory and does not accidentally retain
  full-vocabulary tensors after generation.
- Validate the workload inventory counts and requested matched cohorts before
  model loading, actual Qwen3 chat-templated prompt lengths, and position limits.
- Inspect the resulting context-length/budget/policy coverage and oracle gaps.
  Only those measurements can answer whether static logits degrade with context
  length or whether sequential reconditioning deserves a later experiment.

Mac unit tests validate semantics, schemas, bounded search, and aggregation but
do not establish any GPU result or performance conclusion.

Run the CPU-only validation environment with:

```bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
python -m compileall -q dflash experiments scripts tests \
  third_party/qwen3_8b_dflash_b16/instrumented
```
