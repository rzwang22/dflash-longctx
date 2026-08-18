# Phase 2D.1A: Fixed-Budget Selector Realization

Phase 2D.1A is a CPU-only offline analysis over immutable Phase 2B captures and
a validated Phase 2D.0 v2 directory. Its exact code baseline is Phase 2D.0
commit `26b451dbe55737aeabdccb51871d562208694320`, which descends from the
validated Phase 2C.4.3 commit
`1411c27e086cacd62425f329dcf7432cedd74654`. The analysis does not import
Torch, Transformers, or datasets; load CUDA, tokenizers, models, or weights;
download data; modify capture inputs; or modify instrumented custom code.
Materialization is not required.

The question is deliberately narrower than online speculative decoding: how
much of the per-round policy-oracle target-prefix coverage observed in Phase
2D.0 can a target-blind fixed policy or transparent selector recover at the
same root-inclusive node-budget cap? Coverage is an offline
verification-efficiency proxy. It is not actual accepted tokens, fewer rounds,
latency improvement, or speedup.

## Records, policies, and two distinct oracles

The core analysis fixes Top-M at 16 and emits one configuration-round record
for every eligible physical round at each of N=16 and N=32. The two records are
not two independent physical rounds. Every concrete configuration recorded by
Phase 2D.0 is retained, including the policy variant, confidence threshold or
beam width, Top-M, and N. No unobserved policy is invented.

For each configuration, the event contains coverage depth, actual constructed
nodes, root-inclusive budget cap, budget utilization, coverage per actual node,
gain over Top-1, and regrets. `same_budget_cap_comparison` and
`same_actual_node_count_comparison` are separate. In particular, a 16-node
Top-1 chain under the N=32 cap is not described as the same actual target
compute as a 32-node tree.

The two ceilings have different semantics:

```text
Top-1 coverage
<= per-round policy-oracle coverage
<= prefix-closed target-oracle coverage
<= candidate-availability prefix
```

The policy oracle selects after the fact among tested target-blind policy
configurations. Its construction policies are target-blind, but its per-round
winner label uses the target and is not deployable. The prefix-closed target
oracle directly follows the target path and is a still higher, nondeployable
structural bound. Preflight and the acceptance checker fail on any violation of
this ordering.

## Deployable selector definitions

The held-out selector families are:

1. **Global fixed policy.** For each N and training fold, select the one
   concrete configuration with maximum training coverage sum, then minimum
   actual-node sum, then lexicographic configuration ID. Apply it unchanged to
   the held-out source.
2. **Context-conditioned fixed rule.** For each N, select one configuration per
   exact context-length bucket using training events only. A context absent in
   training uses the training-fold global fixed configuration. The primary rule
   never uses task ID.
3. **Draft-feature decision stump.** Search deterministic rules of the form
   `feature <= threshold ? policy A : policy B`. Features, candidate thresholds,
   leaf policies, and bin boundaries come only from the training fold. Each
   leaf must meet both minimum round and source-document support.

Allowed runtime inputs are deployment-known context/N and bounded draft-side
summaries: Top-1 log probability, Top-1/Top-2 margin, captured Top-16 mass,
Top-16-renormalized entropy, adjacent confidence change, surviving-prefix
count, static score, and structural counts. Top-16 entropy is not
full-vocabulary entropy. Target tokens/ranks/membership, acceptance, oracle
path, gap class, winners, per-policy target coverage, and gain labels cannot
enter runtime features.

The primary validation is leave-one-source-document-out (LOSO). Every context,
round, and N from one source stays in its held-out fold; every output prediction
records the training and held-out source IDs. Leave-one-task-out (LOTO) is a
supplemental cross-task stress test. There is no random round split. Confidence
intervals resample source-document clusters, and are marked unstable when the
configured minimum cluster count is not met.

K=1,2,3 portfolios are also fitted inside each training fold. Their held-out
best-member result is explicitly a nondeployable portfolio oracle used only to
measure whether policy complementarity is concentrated in a small set.

## Metrics and deterministic decision gates

Each selector is summarized separately for N=16 and N=32 over all eligible
rounds and the candidate-available subset. Outputs include mean/median and the
coverage distribution, coverage per actual node, absolute and relative gain,
regret to both oracles, source/task macro metrics, and candidate ceiling.

The primary headroom realization metric is the ratio of sums:

```text
sum(selector coverage - Top-1 coverage)
------------------------------------------------
sum(policy-oracle coverage - Top-1 coverage)
```

The mean of nonzero per-round ratios is reported separately. If aggregate
policy-oracle headroom is zero, the ratio of sums is null with an explicit
reason. It is never coerced to zero or one.

Default engineering gates are exposed through CLI arguments and copied to the
manifest: at least 8 source documents, 2 tasks, 500 eligible physical rounds,
0.10 held-out mean gain, 0.25 headroom capture, 0.05 additional gain for a more
complex selector, at most 0.10 regression in supported context/task-context
strata, 50 rounds and 2 sources per stump leaf, stable source-clustered CIs, and
0.70 modal global-policy selection stability across LOSO folds. GovReport and
HotpotQA source-macro gain must both be nonnegative for fixed-policy
integration; Qasper is reported separately because of its censoring and small
source cohort. Context/stump integration must independently meet the absolute
gain and headroom gates as well as outperforming the simpler selector. These
are preregistered engineering go/no-go rules, not statistical-significance
theorems.

The resulting decision is one of:

```text
integrate_fixed_policy_online
integrate_context_rule_online
integrate_lightweight_selector_online
selector_headroom_not_realizable
deprioritize_selector_prioritize_candidate_generation
insufficient_evidence
```

An integration decision only nominates a candidate for a later Phase 2D.1B
online validation. It is not itself evidence of acceptance, round, latency, or
speedup improvement.

## CPU-only execution

First regenerate Phase 2D.0 with corrected v2 denominator semantics in a new
directory:

```bash
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" \
python scripts/phase2d0_dependency_opportunity.py \
  --capture-dir outputs/phase2b-capture-gov-report-matched-3 \
  --capture-dir outputs/phase2b-capture-qasper-matched-3 \
  --capture-dir outputs/phase2b-capture-hotpotqa-matched-8 \
  --top-ms 2,4,8,16 \
  --node-budgets 16,32 \
  --survival-budget-cap 256 \
  --policies top1_chain,score_beam,confidence_gated,layerwise_beam \
  --output-dir outputs/phase2d0-dependency-opportunity-v2
```

Then run the selector analysis and its acceptance checker:

```bash
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" \
python scripts/phase2d1_selector_realization.py \
  --capture-dir outputs/phase2b-capture-gov-report-matched-3 \
  --capture-dir outputs/phase2b-capture-qasper-matched-3 \
  --capture-dir outputs/phase2b-capture-hotpotqa-matched-8 \
  --phase2d0-dir outputs/phase2d0-dependency-opportunity-v2 \
  --top-m 16 \
  --node-budgets 16,32 \
  --evaluation leave-one-source-out,leave-one-task-out \
  --selector-families global_fixed,context_rule,decision_stump \
  --portfolio-sizes 1,2,3 \
  --output-dir outputs/phase2d1-selector-realization

env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" \
python scripts/check_phase2d1_acceptance.py \
  --output-dir outputs/phase2d1-selector-realization
```

The 18 artifacts are the selector events; fixed-policy JSON/CSV; winner,
overlap, and portfolio tables; fold document; three held-out prediction tables;
headroom, regret, clustered-bootstrap, and context tables; RQ1-RQ7 summary;
decision; manifest; and terminal log. Input hashes, source membership, exact
configuration, repository provenance, and framework-load state are retained in
those outputs. A preflight failure happens before output-directory creation.
