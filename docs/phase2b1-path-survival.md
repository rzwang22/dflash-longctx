# Phase 2B.1: target-path survival and pruning analysis

Phase 2B.1 is a CPU-only, offline diagnostic. It reads immutable Phase 2B
Top-16 capture artifacts and asks why a reference target prefix does or does not
survive a bounded, prefix-closed candidate tree. It does not load Torch,
Transformers, either 8B model, or CUDA. It does not implement tree verification,
sparse KV, training, or any change to DFlash generation.

## Strict preflight

All capture directories are completely validated before the output directory is
created. The preflight validates the Phase 2B manifest, sample, round, and Phase
1 source-provenance schemas, and then checks:

- successful capture/provenance status and repository-source matching;
- batch size 1, temperature 0, block size 16, draft depth 15, and Top-K 16;
- unique `(task, sample, nominal length, repeat, round)` identifiers across all
  input directories;
- one task per directory and exact sample/round task, source hash, context
  length, final-output, and provenance agreement;
- control/capture token equality and both immediate and observable baseline
  alignment invariants;
- consecutive positions/ranks, unique token IDs, Top-1 baseline identity, and
  score/tie ordering;
- exact observable-depth, terminal, full-depth, and right-censoring semantics.

No missing target token is filled or inferred. `candidate_unavailable` means only
that the observed target token is absent from the captured candidate set at that
static position. It is not a claim that the model could never recover it under a
different candidate-generation procedure.

The manifest records SHA-256 for every required input artifact and a canonical
directory digest. The analyzer opens inputs read-only and never changes them.

## Scores, rank, and probability fields

The captures store log probabilities from `log_softmax`, not unnormalized
logits. Thus the analyzer reports:

- `top_k_probability_mass = sum(exp(log_probability))`, the actual
  full-vocabulary probability mass represented by captured Top-K;
- `top_k_renormalized_truncated_entropy`, entropy after renormalizing only that
  captured mass. This is explicitly not full-vocabulary entropy;
- the cumulative log probability of an available target-consistent prefix;
- the Top-1 minus target log-probability gap as a post-hoc label-aware field.

`target_prefix_score_rank` is the target-consistent prefix's rank among all
captured Cartesian prefixes at the same depth under Phase 2B's draft-only score.
At equal depth, mean cumulative log probability and cumulative log probability
give the same ordering. Equal scores use lexicographically ascending token IDs.

The implementation does not materialize `16^15` paths. A branch-and-bound search
uses suffix score bounds to count whole subtrees that certainly outrank or trail
the target, and descends only across score/tie boundaries. The default cap is
100,000. When at least 100,000 prefixes outrank the target, the exact rank is
`null`, the lower bound is `100001`, and `target_prefix_score_rank_censored` is
true. The cap is never reported as the actual rank. If any target token is absent
from captured Top-K, score rank is unavailable rather than estimated.

## Policies and target-label use

All total-node budgets include the root.

- `top1_chain` is the exact root-plus-Top-1 baseline.
- `score_beam` directly calls the existing Phase 2B
  `score_beam_tree`: Top-1 backbone plus length-normalized best-first branches,
  with the existing prefix closure and tie break.
- `confidence_gated` directly calls the existing Phase 2B implementation at
  fixed margin thresholds 0.25, 0.5, and 1.0 with branch width 4.
- `layerwise_beam` is a diagnostic, target-blind fixed-width beam. It scores
  same-depth prefixes identically to score-beam and inserts complete layers in
  deterministic score order until the total budget is exhausted. This makes
  smaller-budget trees strict subsets of larger-budget trees.
- `oracle_prefix_closed` reads the target and is an explicitly non-deployable
  structural upper bound.

The four deployable/diagnostic policies receive positions and budgets only;
they never receive the target. Target labels are used after construction to
query membership, classify the failure, compute target score rank and minimum
survival budget, and construct the oracle. The target-vs-Top-1 score gap is
post-hoc only and is excluded from confidence predictors.

For a fixed policy and depth,
`minimum_total_tree_node_budget_to_preserve_target_prefix` is the first
configured root-inclusive budget whose actual replayed tree contains that target
prefix. It is not approximated from score rank. If no configured budget works,
the exact value is `null` and the label is `>max_budget`. Survival is checked for
monotonicity across budgets; any decrease is a runtime error.

## Confidence and aggregation semantics

The untrained confidence diagnostics predict whether captured draft Top-1 is
wrong using only negative Top-1/Top-2 margin, Top-K truncated entropy, and lower
Top-1 probability. They report AUROC, AUPRC, and equal-frequency risk deciles for
micro, task, context, task-context, and sample-macro scopes. A single-class group
reports `null` with a reason. No threshold is fitted and no generalization claim
is made. Confidence intervals are intentionally absent: positions and rounds
within a sample are not treated as independent units.

Summary artifacts separately record round micro and sample macro quantities,
mean/median/P90, full-depth and right-censored counts, absolute extra covered
target tokens, and relative fractions. A censored round contributes only at
depths actually observed. The “best” layerwise configuration in the concise
long-context table is selected descriptively on the same data and is labelled as
such.

These are offline coverage metrics, not actual accepted lengths. In particular,
the analyzer makes no claim about tree verification latency, end-to-end speedup,
round reduction, a real tree mask, KV sharing, sequential refinement, or learned
selection.

## Server command

Run from the repository root in the Phase 2B.1 environment:

```bash
python scripts/phase2b1_path_survival.py \
  --capture-dir outputs/phase2b-capture-gov-report-matched-3 \
  --capture-dir outputs/phase2b-capture-qasper-matched-3 \
  --capture-dir outputs/phase2b-capture-hotpotqa-matched-8 \
  --node-budgets 16,32,64,128,256 \
  --beam-widths 2,4,8,16,32 \
  --rank-cap 100000 \
  --output-dir outputs/phase2b1-path-survival
```

This command needs CPU and sufficient host memory only. It does not need model
paths, materialized custom code, Hugging Face access, a GPU, or CUDA.

## Artifacts

- `path-survival-events.jsonl`: one row per unique capture round, with shared
  depth features and nested policy/budget replay results.
- `path-survival-summary.json` and `.csv`: task/context/policy/budget aggregates.
- `depth-survival.csv`: each depth's candidate availability and path survival.
- `pruning-reasons.csv`: survived, unavailable, and available-but-pruned counts.
- `minimum-budget.csv`: per-round, per-policy, per-depth replayed minimum budget.
- `confidence-diagnostics.json` and `.csv`: predictor metrics and calibration.
- `budget-frontier.csv`: absolute coverage and 16/32/64/128/256 marginal returns.
- `long-context-summary.csv`: GovReport 32K, Qasper 16K, and HotpotQA 16K focus.
- `analysis-manifest.json`: immutable input hashes, Git/source versions, exact
  algorithms, target-use classes, command, and interpretation constraints.
- `terminal.log`: mirrored terminal progress.

Draft 2020-12 schemas for all new JSON/JSONL records live under `docs/schemas/`.

## How results guide the next phase

The evidence should be interpreted conditionally:

1. If target tokens remain available, target-prefix score ranks are moderate,
   and a target-blind policy improves materially with budget or allocation,
   continue improving a static selector.
2. If tokens are individually available but static prefix ranks explode,
   off-Top-1 decisions cluster, and layerwise allocation also fails, investigate
   branch-conditioned sequential refinement. Phase 2B.1 itself does not measure
   that method.
3. If failure is dominated by target tokens absent from captured Top-16 at the
   first lost depth, capture a larger Top-K in a separately approved experiment;
   current data cannot infer ranks beyond 16.
4. If deployable coverage saturates from 64 to 128 to 256 with poor marginal
   tokens per node even while the oracle gap remains, avoid a large static tree
   and investigate another acceptance-recovery direction.
