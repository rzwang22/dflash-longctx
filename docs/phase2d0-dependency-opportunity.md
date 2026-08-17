# Phase 2D.0: Dependency Opportunity Characterization

Phase 2D.0 is a CPU-only, offline analysis of immutable Phase 2B captures. Its
baseline is Phase 2C.4.3 commit
`1411c27e086cacd62425f329dcf7432cedd74654`. It does not load Torch,
Transformers, datasets, CUDA, tokenizers, models, or weights. It does not modify
the online tree, verifier, acceptance, KV commit, sampling, Phase 2C
correctness, or instrumented custom code, and requires no materialization.

The runner validates every capture directory before creating its output
directory. Validation covers the Phase 2B manifest, source provenance and
loaded-source hashes, round/sample alignment, immutable file hashes, exact
15-position block shape, Top-16 ranks, root-inclusive accounting, batch size
one, temperature zero, and log-softmax score semantics. Phase 2B.1 artifacts
are not trusted as an intermediate input; policies are replayed directly from
the validated captures.

## Unit of analysis and classification

For each round, `a0` is Original DFlash's accepted Top-1 draft depth, excluding
the root. `A_M` is the longest prefix for which every target continuation token
is present in its positional Top-M candidate set. Static and oracle coverage
are longest prefix depths, never online accepted lengths. Every event is keyed
by round, `M ∈ {2,4,8,16}`, and root-inclusive node budget `N`.

The mutually exclusive priority order is:

1. `right_censored_or_terminal`: the round is terminal or its reference is
   right-censored.
2. `no_failure_within_observed_horizon`: `a0` reaches the complete observed
   horizon.
3. `candidate_availability_failure`: the target token at depth `a0+1` is not in
   positional Top-M. More tree nodes cannot recover it.
4. `selection_failure`: at least one tested target-blind static policy covers
   `a0+1` at the same `N`, while the Top-1 baseline does not.
5. `budget_failure`: every tested policy fails at `N`, but at least one covers
   `a0+1` by the configured modest-budget limit, default `2N`.
6. `dependency_realization_opportunity`: the token is available, the
   root-inclusive oracle covers `a0+1` at the same `N`, every tested static
   policy fails, modest budget does not resolve the failure, and an exact
   target-prefix score rank or an observed finite static survival budget exceeds
   its configured `2×` inflation threshold.
7. `possible_dependency_opportunity`: candidate and oracle conditions hold but
   rank/budget evidence is censored or does not conservatively exclude a better
   target-blind selector.
8. `unresolved`: no prior condition applies.

The policy family contains `top1_chain`, `score_beam`, three fixed
`confidence_gated` variants, and four fixed `layerwise_beam` widths. All are
constructed without target data. The prefix-closed oracle uses the target only
as a structural upper bound. A score-rank or minimum-budget value above its cap
is `null` with an explicit censor reason; it is not converted to zero or treated
as confirmed evidence.

The default frontier is 16, 32, 64, 128, and 256 root-inclusive nodes. Core
composition is reported only for 16 and 32 nodes. Coverage per verified node
and marginal coverage per added node are offline verification-efficiency
proxies, not hardware efficiency, latency, acceptance, round reduction, or
speedup.

## Signal/label separation

`signal-labels.jsonl` contains one row for every captured draft position and
configuration. Every position is emitted independently of the target outcome;
this prevents selecting the feature row using the target-dependent critical
depth. The nested `deployable_features` object contains only captured
draft-side or static structural values: position, Top-1 log probability,
Top-1/Top-2 margin, Top-M probability mass, Top-M renormalized entropy,
adjacent-margin change, fixed score-beam state, prefix count, context length,
M, and N.

Target rank/membership, `a0`, first critical position, static-path failure,
oracle/dependency labels, and gap class are kept in the separate `labels`
object. Top-M entropy is explicitly the entropy after renormalizing captured
Top-M mass; it is not full-vocabulary entropy. The analysis reports untrained
single-feature AUROC, AUPRC, five-bin precision, counts, and per-context trends.
Results with fewer than 30 observations or fewer than five observations in one
class are marked unstable. No refinement-gain label such as `ΔA_i` is defined.

## Aggregation and Phase 2D.1 decision

The context table reports every mutually exclusive class, including censored
and unresolved events. Round micro fractions use configuration-rounds as
observations. Macro fractions first aggregate within each sample. Task-level
and cross-task macro rows are separate, and no GovReport result is extrapolated
to another task.

`research-question-summary.json` answers RQ1–RQ6 mechanically. The decision is
not selected by hand. With at least 100 eligible Top-16 configuration-rounds
from at least two tasks:

- candidate availability share at least 0.5 selects
  `prioritize_candidate_generation`;
- selection plus modest-budget share at least 0.5 selects
  `prioritize_selector`;
- confirmed dependency share at least 0.15 plus a stable single feature with
  AUROC at least 0.65 and AUPRC at least prevalence plus 0.05 selects
  `continue_dependency_refinement`;
- two or more families at share 0.15 select `mixed_problem`;
- otherwise the result is `insufficient_evidence`.

These rules may be revised only in a later phase, not after seeing Phase 2D.0
results. Before real capture aggregation, no dependency-dominance claim is
supported.

## CPU-only run and acceptance

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
python scripts/phase2d0_dependency_opportunity.py \
  --capture-dir outputs/phase2b-capture-gov-report-matched-3 \
  --capture-dir outputs/phase2b-capture-qasper-matched-3 \
  --capture-dir outputs/phase2b-capture-hotpotqa-matched-8 \
  --top-ms 2,4,8,16 \
  --node-budgets 16,32 \
  --survival-budget-cap 256 \
  --policies top1_chain,score_beam,confidence_gated,layerwise_beam \
  --output-dir outputs/phase2d0-dependency-opportunity
```

Validate existence, JSON schemas, mutually exclusive counts, CSV versions, and
the no-framework provenance contract with:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
python scripts/check_phase2d0_acceptance.py \
  --output-dir outputs/phase2d0-dependency-opportunity
```

The output contains:

```text
opportunity-events.jsonl
gap-decomposition.json
gap-decomposition.csv
context-gap-composition.csv
availability-by-depth.csv
static-survival-frontier.csv
minimum-survival-budget.csv
dependency-opportunity.csv
signal-labels.jsonl
signal-hypotheses.csv
research-question-summary.json
analysis-manifest.json
terminal.log
```

This phase does not implement selective refinement, remasking, training,
adapter inference, a second model pass, sparse KV, hardware simulation, or any
online optimization.
