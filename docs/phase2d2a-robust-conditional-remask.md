# Phase 2D.2A: numerically robust conditional-remask refinement

Phase 2D.2A is an opt-in diagnostic mechanism built from baseline
`1ba29e8ce4a165c03af997c2f3b3b595ab79c1d2`. It does not train an adapter,
change target greedy decoding, enlarge the target node budget, or implement
sparse KV/hardware simulation. Results are not evidence of acceptance or speedup
until the H100 gates and held-out pilot pass.

## Audited DFlash semantics

The pinned draft config defines mask token `151669` and block size 16. At every
round, original DFlash embeds `[root, MASK x 15]`; the draft logits from the last
15 hidden positions predict draft positions 1 through 15. The root is block
position zero and a refinement prefix begins at position one.

The DFlash draft attention is explicitly non-causal (`is_causal=False`) and the
normal draft call supplies no restrictive block mask. Each suffix MASK therefore
sees the filled prefix and the other block positions. Attention keys/values are
the concatenation of newly injected target hidden states and all current block
states, following the existing persistent draft cache. The RoPE position tensor
covers the prior draft-cache offset through the newly injected target-hidden span
and the 16-token block.

A draft forward mutates its supplied cache. The conditional pass consequently
uses a deep copy of the round-entry draft cache, expands only that copy from
external batch 1 to the selected internal branch batch, performs at most one
batched forward, and discards the result. It never mutates target KV and never
commits conditional draft KV. The initial parallel draft still runs exactly once.

These source-confirmed relations need H100 confirmation against Transformers
4.57.1 tensors: DynamicCache batch-axis expansion, the exact conditional logits
position alignment, EOS/partial final-block behavior, and no persistent cache
aliasing after the conditional forward.

## Target-blind selection and candidate merge

The frozen Phase 2D.1 selector first constructs its normal seed tree. At the
configured depth (1, 2, or 4), eligible seed prefixes are ordered by mean
cumulative base draft log probability, cumulative log probability, shorter depth,
then lexicographic token-ID prefix. At most 1, 2, or 4 are retained. No target
token, logits, accepted depth, reference output, oracle label, or Gold result is
accepted by the selection API.

The `base_only` CLI control defaults to the previously validated Phase 2C
`top1_packed_tree` selector mode. `global_fixed`, `context_rule`, and
`decision_stump` remain available as explicit frozen-selector experiments, but
are not silently substituted for that base control. Target-only generation uses
an explicit all-ones attention mask and greedy `generate` with inactive sampling
arguments suppressed; Original DFlash uses the unchanged
`spec_generate(temperature=0)` call. Both receive the same chat-templated input
IDs, EOS ID, maximum token count, target instance, dtype, and cache semantics.
Every run records all three pairwise relations among target-only, Original
DFlash, and the tree output. Original DFlash actual speculative execution is the
token-exact oracle for the tree. Target-only is an auxiliary numerical control:
its reference/tree mismatches are retained as explicit warnings but are not hard
failures without an Original-DFlash/tree divergence.

For each selected prefix, one internal-batch conditional pass computes full
vocabulary logits. Only bounded IDs are retained, but both the base and
conditional full-vocabulary log probability are gathered for every retained ID.
Missing values are never replaced with zero.

- `base_only`: exact frozen-selector seed tree; conditional forward count is zero.
- `conditional_rerank`: token availability remains base Top-16; conditional
  logits can only reorder those tokens.
- `union`: base Top-16 is unioned with conditional Top-2 or Top-4.

For a refined descendant, the positional score is
`(1-lambda)*base_logprob + lambda*conditional_logprob`. Outside a refined
descendant it is the base log probability. Prefix scores are cumulative; stable
ties use mean cumulative score, cumulative score, depth, and token-ID prefix.
The final tree is prefix-closed, contains exactly one root, and never exceeds the
same root-inclusive N=16 or N=32 target budget. Available, merged, selected,
verified, refined-only, and unused nodes are recorded separately.

## Robust verification

Evaluation of frozen refinements is temporarily paused for verifier-oracle
characterization. The names `always_replay` and `margin_guarded_replay` describe
the existing online mechanisms; they do not establish that q1 replay is an
authoritative numerical oracle. Original DFlash actual speculative execution
remains the hard token-exact reference. Packed-versus-q1 disagreement alone is
diagnostic and cannot be called a correctness failure or a prevented failure.

- `packed_native` commits the existing packed traversal and cache. It is the risk
  and performance baseline, not an equality claim.
- `always_replay` runs packed verification for candidate statistics, then starts
  from the pre-round persistent target KV and traverses the candidate tree with
  sequential target forwards of `q_len=1`. The replay target argmax controls
  accept/correct/bonus decisions. Only root plus accepted draft path remains in
  the replay cache; correction/bonus becomes the next root and is not committed
  in the current round. Packed temporary KV is discarded.
- `margin_guarded_replay` applies the same q1 replay when any
  verifier-relevant packed margin is at or below a frozen threshold or is
  unavailable. The calibration must come from an independent Phase 2D.1B-NR
  cohort and exactly match GPU, compute capability, dtype, model revisions,
  PyTorch, Transformers, and target attention backend. No calibration means a
  preflight failure; zero is not silently assumed.
- `always_fixed_shape_recheck` runs one Original-DFlash-compatible fixed
  16-token linear block from an independent clone of the exact round-entry KV
  snapshot on every round. It rechecks only the packed-selected path, may
  shorten that path at the first mismatch, crops the temporary KV to the root
  plus accepted prefix, and commits with source
  `phase2d2_fixed_shape_recheck`. The correction/bonus remains the next root.
- `margin_guarded_fixed_shape_recheck` uses that same q=16 executor only when
  the frozen packed margin is at most 0.25 or unavailable. A non-triggered
  round is the existing packed-native commit. The provenance-matched NR
  calibration is mandatory and the threshold is not recalibrated here.

The two fixed-shape modes are new opt-in modes. They do not redefine the
historical q=1 behavior or artifact meaning of `always_replay` and
`margin_guarded_replay`. Their neutral accounting uses `fallback_executed` and
`fallback_executor=fixed_shape_recheck`; the legacy `fallback_replay_*` fields
remain false. Each online fixed fallback takes a second independent clone from
the preserved pre-packed snapshot, so packed temporary KV cannot become its
input. Diagnostic records prove snapshot hash stability, storage independence,
the full temporary prefix+16 layer lengths, final prefix+root+accepted crop,
MASK-tail removal, packed-cache discard, and exact target-hidden commit length.

The diagnostic run can execute a non-authoritative shadow replay above the guard
threshold to measure false positives/negatives. Formal latency, component
profiling, and Gold/shadow diagnostics are separate generations. The guard is an
empirical H100 control, not a formal numerical error bound.

Candidate-quality tables still have their historical q1-labelled provenance.
Fixed-shape F1/F2/F3 qualification therefore does not use coverage or
refined-only recovery as a decision criterion and makes no new refinement
benefit claim. Original DFlash speculative execution is the hard token-exact
oracle; target-only and q1 disagreement remain auxiliary diagnostics.

### Three-way verifier-oracle diagnostic

`--enable-verifier-oracle-diagnostics` adds a diagnostics-generation-only
comparison at every round-entry semantic prefix. It records the actual packed
decision, q1 replay, and the original-shape fixed linear-block recheck. Q1 and
fixed-shape execution receive independent deep clones of one cache snapshot;
sampled cache summaries and hashes establish the shared source, while storage
independence establishes that neither replay consumes the other's mutated
cache. Neither diagnostic cache can be committed. The runner aligns a
candidate committed sequence to Original DFlash using absolute generated-prefix
length and the token-prefix hash, not raw round number.

| Property | Packed native | Q1 replay | Fixed-shape recheck |
|---|---|---|---|
| Target query | all packed tree nodes | one token per forward | one 16-token block |
| Mask | explicit 4D prefix/self/ancestor tree mask | explicit all-ones linear prefix | implicit causal mask |
| Position IDs | semantic tree depths | one absolute position per forward | 16 consecutive absolute positions |
| Cache position | explicit packed storage positions | explicit, advanced after every token | implicit argument; semantic positions recorded |
| Cache update | one packed append then gather | sequential append per token | one 16-token append, cropped only by the existing online mode |
| Target hidden | gathered committed packed nodes | concatenated q1 hidden steps | leading root/accepted part of the fixed block |
| Correction/bonus | terminal packed parent logits | terminal q1 logits | corresponding fixed-block parent logits |
| Subsequent draft state if committed | packed gathered KV/hidden | q1 KV/hidden | fixed-block cropped KV/hidden |

These query shapes and cache schedules can select different SDPA kernels and
produce different BF16 argmax results. Sequential q1 is therefore not assumed
to be more correct. The fixed-shape recheck is a candidate diagnostic oracle,
not a newly promoted online commit policy. Formal execution, selector,
candidate tree, acceptance, threshold, and timing remain unchanged.

The new artifacts are `verifier-oracle-comparisons.jsonl`,
`verifier-oracle-summary.json`, and `tree-traversal-checks.jsonl`. Per-repeat,
profile, and diagnostic output hashes expose stability without adding work to
the timed generation. Neutral robust-event fields report packed/q1 disagreement
and guard response; legacy positive/negative fields remain only for backward
schema compatibility.

When selected by an existing replay mode, the Q1 cache is its linear online
commit source, not a gathered packed-cache source. Gold KV validation checks its storage indices as the full
contiguous committed prefix and requires the packed temporary KV to be marked
discarded. Each Gold row records non-null candidate and dense-reference
K/V shapes for every target layer, layer-count/shape equality, the existing
length invariants, and numerical error reductions. Missing or incompatible shape
metadata remains a structural failure.

The external checker permits one narrow exception for unavailable downstream
KV Gold: `availability=unavailable`, `status=not_run`, and reason exactly
`prior_logits_gold_failure`. Acceptance additionally requires exact and stable
Original-DFlash/tree execution, an error-free manifest, a passing traversal at
that round, every configured Gold-logit mode present and structurally valid,
and an actual prior semantic logit failure. Runtime errors, missing modes,
unrelated unavailable records, or absent traversal evidence remain hard
failures; the skipped row is not rewritten as an exact Gold pass.

Gold p99 reductions use native `torch.quantile` through its supported indexing
range. Larger tensors are transferred in full to CPU and evaluated exactly from
the two required `kthvalue` order statistics with linear interpolation. This is
not a sample or approximation. Each result records the backend, input element
count, dtype, device, probability, interpolation, and fallback status. The
reduction remains diagnostic-only and outside formal timing.

Original DFlash actual speculative execution is the Phase 2D.2A hard token-exact
oracle. Target-only greedy output remains a recorded auxiliary numerical control.
Refinement may alter accepted depth, correction rounds, and round segmentation;
the Original-DFlash/tree absolute generated-token relation and structural/cache
invariants are hard-gated.

## Artifacts and automatic decision

The runner writes the full artifact inventory listed by `--help`, including
sample/round records, candidate merge provenance, fixed-N comparison, robust
verification and fallback events, formal/component timing, correctness and Gold
diagnostics. Every new JSON/JSONL format has a Draft 2020-12 schema under
`docs/schemas/`; CSV rows contain `schema_version`. Unavailable measurements use
null plus a reason.

`phase2d2-decision.json` applies the preregistered priority order. Correctness or
guard false negatives dominate benefit. Fewer than two independent tasks or
missing stability/cost evidence produces `insufficient_evidence`; smoke runs
cannot be promoted manually. Per-task runs therefore remain
`insufficient_evidence`. After held-out runs, `aggregate_phase2d2_decision.py`
requires at least two tasks with positive per-task mean and cross-task macro
gain. Conditional plus authoritative-fallback cost is acceptable only when its
median share of profiled total CUDA time is at most 25%; a conditional pass at
least as expensive as the initial parallel draft is classified as a full second
pass that is too heavy. These are preregistered characterization rules, not a
speedup claim.

## H100 setup and new materialization

The instrumented `dflash.py` changed. Use a new SHA-bound directory; in
particular, do not overwrite the current
`Qwen3-8B-DFlash-b16-phase2d2a-instrumented-804234b14133` directory. The change
adds two opt-in fixed-shape online verifier modes and their records; it does not
change refinement, tree construction, or any legacy verification mode.

```bash
cd ~/data/dflash-longctx
git fetch origin
git checkout codex/phase2d2a-robust-conditional-remask
git pull --ff-only origin codex/phase2d2a-robust-conditional-remask
git merge-base --is-ancestor 1ba29e8ce4a165c03af997c2f3b3b595ab79c1d2 HEAD
test -z "$(git status --porcelain)"

source .venv/bin/activate
export TARGET_MODEL_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-b968826d
export DRAFT_WEIGHT_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16
export LONGBENCH_ROOT=/mnt/sevenT/wrz_data/datasets/LongBench/data
export FROZEN_SELECTOR=outputs/phase2d1b-frozen-selector/frozen-selector.json
export SHORT_SHA=$(git rev-parse --short=12 HEAD)
export DRAFT_MODEL_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16-phase2d2a-instrumented-$SHORT_SHA

test ! -e "$DRAFT_MODEL_DIR"
python scripts/materialize_dflash.py \
  --source-model-path "$DRAFT_WEIGHT_DIR" \
  --output-dir "$DRAFT_MODEL_DIR" \
  --implementation instrumented
```

## Earlier ordered H100 gates (paused)

The commands below document the preregistered evaluation sequence, but frozen
refinement, N=32, held-out, and threshold runs are paused during verifier-oracle
characterization. Do not resume them from this document until the temporary
characterization stage is closed.

Base control smoke:

```bash
python experiments/phase2d2_robust_conditional_refinement.py \
  --target-model-path "$TARGET_MODEL_DIR" \
  --materialized-draft-path "$DRAFT_MODEL_DIR" \
  --longbench-root "$LONGBENCH_ROOT" --longbench-task gov_report \
  --frozen-selector-config "$FROZEN_SELECTOR" \
  --sample-id gov_report-fd287bbfd3bdbdc9 \
  --context-lengths 1K --pilot-samples 1 --max-new-tokens 32 \
  --warmup-runs 0 --timing-repeats 1 \
  --candidate-merge-mode base_only --node-budget 16 \
  --verification-mode always_replay \
  --enable-gold-checks --gold-max-rounds 32 \
  --output-dir outputs/phase2d2a-base-n16-1k-always-gold

python scripts/check_phase2d2_acceptance.py \
  outputs/phase2d2a-base-n16-1k-always-gold
```

Conditional-remask mechanics smoke:

```bash
python experiments/phase2d2_robust_conditional_refinement.py \
  --target-model-path "$TARGET_MODEL_DIR" \
  --materialized-draft-path "$DRAFT_MODEL_DIR" \
  --longbench-root "$LONGBENCH_ROOT" --longbench-task gov_report \
  --frozen-selector-config "$FROZEN_SELECTOR" \
  --sample-id gov_report-fd287bbfd3bdbdc9 \
  --context-lengths 1K --pilot-samples 1 --max-new-tokens 32 \
  --warmup-runs 0 --timing-repeats 1 \
  --candidate-merge-mode union --refine-depth 2 \
  --refined-prefix-budget 2 --conditional-top-r 2 \
  --refinement-weight 0.5 --node-budget 16 \
  --verification-mode always_replay \
  --enable-gold-checks --gold-max-rounds 32 \
  --output-dir outputs/phase2d2a-union-d2-p2-r2-n16-1k-always-gold

python scripts/check_phase2d2_acceptance.py \
  outputs/phase2d2a-union-d2-p2-r2-n16-1k-always-gold
```

Build the guard calibration only from the frozen independent NR cohort. The
threshold is fixed at 0.25; Phase 2D.2A outputs cannot retune it. The builder
reruns the final Phase 2D.1B-NR checker, requires its 45 observations / 15
physical cases / 45 configuration cases / four actual divergences, and checks
every cohort-level selector/budget stratum. Both the one attributed numerical
divergence and the three unresolved actual semantic divergences contribute to
the false-negative count. The total at 0.25 must be four caught and zero missed,
and 0.25 must remain the minimum preregistered zero-FN threshold:

```bash
export NR_DIR=outputs/phase2d1b-nr-full-cohort-45dd261c928a
export NR_CALIBRATION=outputs/phase2d2a-nr-calibration

python scripts/check_phase2d1b_nr_acceptance.py --output-dir "$NR_DIR"

python scripts/phase2d2_build_nr_calibration.py \
  --nr-dir "$NR_DIR" \
  --decision-margin-threshold 0.25 \
  --output-dir "$NR_CALIBRATION"
```

Gate 3B uses the same mechanism configuration already validated by Gate 2 and
changes only the verification mode to the calibrated guard:

```bash
python experiments/phase2d2_robust_conditional_refinement.py \
  --target-model-path "$TARGET_MODEL_DIR" \
  --materialized-draft-path "$DRAFT_MODEL_DIR" \
  --longbench-root "$LONGBENCH_ROOT" --longbench-task gov_report \
  --frozen-selector-config "$FROZEN_SELECTOR" \
  --sample-id gov_report-fd287bbfd3bdbdc9 \
  --context-lengths 1K --pilot-samples 1 --max-new-tokens 32 \
  --warmup-runs 0 --timing-repeats 1 \
  --candidate-merge-mode union --refine-depth 2 \
  --refined-prefix-budget 2 --conditional-top-r 2 \
  --refinement-weight 0.5 --node-budget 16 \
  --verification-mode margin_guarded_replay \
  --nr-calibration-dir "$NR_CALIBRATION" \
  --enable-gold-checks --gold-max-rounds 32 \
  --output-dir outputs/phase2d2a-gate3b-margin-guarded

python scripts/check_phase2d2_acceptance.py \
  outputs/phase2d2a-gate3b-margin-guarded
```

Do not run the following preregistered one-round mechanism analyses until Gate
3B has passed and a separate continuation is authorized. They remain documented
for the later stage only. `max_new_tokens=1` deliberately limits each generation
to one tree round; it is not a latency result.

```bash
run_mechanism () {
  LABEL="$1"
  shift
  python experiments/phase2d2_robust_conditional_refinement.py \
    --target-model-path "$TARGET_MODEL_DIR" \
    --materialized-draft-path "$DRAFT_MODEL_DIR" \
    --longbench-root "$LONGBENCH_ROOT" --longbench-task gov_report \
    --frozen-selector-config "$FROZEN_SELECTOR" \
    --sample-id gov_report-fd287bbfd3bdbdc9 \
    --context-lengths 1K --pilot-samples 1 --max-new-tokens 1 \
    --warmup-runs 0 --timing-repeats 1 --node-budget 16 \
    --verification-mode always_replay \
    --enable-gold-checks --gold-max-rounds 1 \
    "$@" --output-dir "outputs/phase2d2a-mechanism-$LABEL"
  python scripts/check_phase2d2_acceptance.py \
    "outputs/phase2d2a-mechanism-$LABEL"
}

run_mechanism rerank-d1-p1-r2 \
  --candidate-merge-mode conditional_rerank \
  --refine-depth 1 --refined-prefix-budget 1 \
  --conditional-top-r 2 --refinement-weight 0.5
run_mechanism union-d2-p2-r2 \
  --candidate-merge-mode union \
  --refine-depth 2 --refined-prefix-budget 2 \
  --conditional-top-r 2 --refinement-weight 0.5
run_mechanism union-d4-p2-r4 \
  --candidate-merge-mode union \
  --refine-depth 4 --refined-prefix-budget 2 \
  --conditional-top-r 4 --refinement-weight 0.5

export FROZEN_REFINEMENT=outputs/phase2d2a-frozen-refinement.json
python scripts/phase2d2_freeze_refinement_config.py \
  --analysis-dir outputs/phase2d2a-mechanism-rerank-d1-p1-r2 \
  --analysis-dir outputs/phase2d2a-mechanism-union-d2-p2-r2 \
  --analysis-dir outputs/phase2d2a-mechanism-union-d4-p2-r4 \
  --output "$FROZEN_REFINEMENT"

mapfile -t FROZEN_REFINEMENT_IDS < <(
  python -c 'import json,sys; [print(r["configuration_id"]) for r in json.load(open(sys.argv[1]))["configurations"]]' "$FROZEN_REFINEMENT"
)
```

The frozen refinement family deliberately excludes N, so the same frozen
mechanism is evaluated at both N=16 and N=32. Run the known-sample nested
1K/16K base controls first, then each frozen family. These are mechanism
validation runs, not held-out evidence, because the sample ID is prescribed by
this gate.

```bash
for N in 16 32; do
  BASE_DIR="outputs/phase2d2a-paired-base-n$N"
  python experiments/phase2d2_robust_conditional_refinement.py \
    --target-model-path "$TARGET_MODEL_DIR" \
    --materialized-draft-path "$DRAFT_MODEL_DIR" \
    --longbench-root "$LONGBENCH_ROOT" --longbench-task gov_report \
    --frozen-selector-config "$FROZEN_SELECTOR" \
    --sample-id gov_report-fd287bbfd3bdbdc9 \
    --context-lengths 1K,16K --pilot-samples 1 --max-new-tokens 128 \
    --warmup-runs 1 --timing-repeats 3 \
    --candidate-merge-mode base_only --node-budget "$N" \
    --verification-mode margin_guarded_replay \
    --nr-calibration-dir "$NR_CALIBRATION" \
    --enable-gold-checks --gold-max-rounds 128 \
    --output-dir "$BASE_DIR"
  python scripts/check_phase2d2_acceptance.py "$BASE_DIR"

  for CONFIG_ID in "${FROZEN_REFINEMENT_IDS[@]}"; do
    REFINED_DIR="outputs/phase2d2a-paired-${CONFIG_ID}-n$N"
    python experiments/phase2d2_robust_conditional_refinement.py \
      --target-model-path "$TARGET_MODEL_DIR" \
      --materialized-draft-path "$DRAFT_MODEL_DIR" \
      --longbench-root "$LONGBENCH_ROOT" --longbench-task gov_report \
      --frozen-selector-config "$FROZEN_SELECTOR" \
      --frozen-refinement-config "$FROZEN_REFINEMENT" \
      --frozen-refinement-config-id "$CONFIG_ID" \
      --sample-id gov_report-fd287bbfd3bdbdc9 \
      --context-lengths 1K,16K --pilot-samples 1 --max-new-tokens 128 \
      --warmup-runs 1 --timing-repeats 3 --node-budget "$N" \
      --verification-mode margin_guarded_replay \
      --nr-calibration-dir "$NR_CALIBRATION" \
      --enable-gold-checks --gold-max-rounds 128 \
      --base-result-dir "$BASE_DIR" \
      --output-dir "$REFINED_DIR"
    python scripts/check_phase2d2_acceptance.py "$REFINED_DIR"
  done
done
```

Cross-task pilots are last. Held-out mode excludes every frozen-selector
training source before checking matched eligibility and before loading either 8B
model. It errors rather than reducing the requested cohort.

```bash
for TASK_SPEC in gov_report:3 qasper:3 hotpotqa:8; do
  TASK="${TASK_SPEC%%:*}"
  COUNT="${TASK_SPEC##*:}"
  for N in 16 32; do
    BASE_DIR="outputs/phase2d2a-pilot-${TASK}-base-n$N"
    python experiments/phase2d2_robust_conditional_refinement.py \
      --target-model-path "$TARGET_MODEL_DIR" \
      --materialized-draft-path "$DRAFT_MODEL_DIR" \
      --longbench-root "$LONGBENCH_ROOT" --longbench-task "$TASK" \
      --frozen-selector-config "$FROZEN_SELECTOR" \
      --evidence-role held_out_evaluation --sample-policy matched \
      --context-lengths 1K,4K,8K,16K,32K --pilot-samples "$COUNT" \
      --max-new-tokens 128 --warmup-runs 1 --timing-repeats 3 \
      --candidate-merge-mode base_only --node-budget "$N" \
      --verification-mode margin_guarded_replay \
      --nr-calibration-dir "$NR_CALIBRATION" \
      --output-dir "$BASE_DIR"
    python scripts/check_phase2d2_acceptance.py "$BASE_DIR"

    for CONFIG_ID in "${FROZEN_REFINEMENT_IDS[@]}"; do
      REFINED_DIR="outputs/phase2d2a-pilot-${TASK}-${CONFIG_ID}-n$N"
      python experiments/phase2d2_robust_conditional_refinement.py \
        --target-model-path "$TARGET_MODEL_DIR" \
        --materialized-draft-path "$DRAFT_MODEL_DIR" \
        --longbench-root "$LONGBENCH_ROOT" --longbench-task "$TASK" \
        --frozen-selector-config "$FROZEN_SELECTOR" \
        --frozen-refinement-config "$FROZEN_REFINEMENT" \
        --frozen-refinement-config-id "$CONFIG_ID" \
        --evidence-role held_out_evaluation --sample-policy matched \
        --context-lengths 1K,4K,8K,16K,32K --pilot-samples "$COUNT" \
        --max-new-tokens 128 --warmup-runs 1 --timing-repeats 3 \
        --node-budget "$N" --verification-mode margin_guarded_replay \
        --nr-calibration-dir "$NR_CALIBRATION" \
        --base-result-dir "$BASE_DIR" --output-dir "$REFINED_DIR"
      python scripts/check_phase2d2_acceptance.py "$REFINED_DIR"
    done
  done
done
```

Finally, produce one cross-task automatic decision per frozen refinement family.
Do not mix refinement families in one decision artifact.

```bash
for CONFIG_ID in "${FROZEN_REFINEMENT_IDS[@]}"; do
  DECISION_ARGS=()
  for TASK in gov_report qasper hotpotqa; do
    for N in 16 32; do
      DECISION_ARGS+=(
        --result-dir "outputs/phase2d2a-pilot-${TASK}-${CONFIG_ID}-n$N"
      )
    done
  done
  python scripts/aggregate_phase2d2_decision.py \
    "${DECISION_ARGS[@]}" \
    --output "outputs/phase2d2a-decision-${CONFIG_ID}.json"
done
```

Proceeding to Phase 2D.2B requires Original-DFlash/tree exactness, all structural/KV
checks, no held-out guard false negative, and stable same-N coverage plus online
accepted-token-efficiency improvement. Abandon dependency refinement if fixed-N
coverage and online efficiency show no stable gain; prefer a lightweight adapter
only if candidate gain is stable but the full conditional pass is demonstrably
too costly.
