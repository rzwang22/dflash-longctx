# Phase 2C.2: Layerwise Output Correctness

Phase 2C.2 diagnoses the confirmed `layerwise_beam-b16-w4` output-token
divergence without changing tree policy, greedy verification, acceptance,
correction/bonus, or KV commit semantics.

## Source-confirmed failure path

The Phase 2C.1 runner called `_require_equal` immediately after the formal tree
control. A mismatch therefore raised before an observed tree run, gold replay,
round mapping, or artifact flush. That explains the empty evidence but does not
identify the underlying H100 mismatch.

The new flow retains formal timing, then performs a separate diagnostic run.
It compares target-only greedy, original DFlash, and the formal tree output,
aligns different round segmentations by absolute generated-token offset, runs
selected-path native/math gold checks, writes all artifacts atomically, and only
then returns a non-zero correctness result.

Diagnostic traversal and cache summaries are bounded by `--gold-max-rounds`
even when gold checks are disabled. Set that value high enough to cover the
first divergent round; the artifact records the applied limit. If the original
DFlash and tree outputs are equal but the target-only control differs,
`round_differential_status` explicitly says that a reference/tree round
differential is not applicable rather than leaving an unexplained empty JSONL.

## Evidence and classification

`tree-traversal-checks.jsonl` records every node, parent/children, path and
score, semantic and packed positions, target argmax/Top-8/margin, selected
child, accepted path, terminal correction/bonus, committed nodes, gather
indices, mask visibility, and comparable before/after cache summaries.
The comparable cache snapshot uses fixed tensor samples and shapes so it does
not materialize full float32 KV copies. The gold KV check remains the source of
full per-layer K/V max/mean/P99 error against dense recomputation.

Structural mask/position checks, parent-to-child traversal checks, and
KV/root-transition checks have separate booleans. A failed gather or root
transition is therefore not mislabeled as a traversal failure.

The first divergent absolute token is mapped independently into the reference
and tree rounds. Round indices are never assumed equal. Classification is:

- `numerical_backend_decision_divergence`: native verifier argmax differs but
  forced-math packed/dense selected-path decisions agree;
- `branch_mask_or_position_mismatch`: structural visibility fails or math still
  has a verifier decision mismatch;
- `tree_traversal_mismatch`: packed logits are available but standard greedy
  parent-to-child traversal checks fail;
- `kv_compaction_or_root_transition_mismatch`: traversal is valid but compacted
  next-token/root evidence fails;
- `unresolved_layerwise_output_divergence`: required H100 evidence is missing.

No numerical classification is made merely from BF16 logit magnitude.

## Phase 2C.1 status correction

Ordered Top-8 differences with unchanged verifier decisions are warnings.
Numerical-threshold-only differences use
`pass_semantic_with_numerical_variation`, never `fail_semantic` with
`semantic_status=pass`. Summaries separately identify the first numerical,
Top-8, semantic, and structural node.

## Output artifacts

Phase 2C outputs now additionally include:

```text
output-divergence.json
output-divergence.csv
round-differential.jsonl
tree-traversal-checks.jsonl
```

On an expected equality failure, `correctness-summary.json` has
`status=failed`; the manifest, traceback, completed checks, uncompleted checks,
and their reasons remain available. Unexpected execution errors retain
`status=error`.

The dynamic instrumented module changed and must be materialized into a new,
independent directory. H100 must determine the first divergent token and root
cause before any correction to online verification is justified.
