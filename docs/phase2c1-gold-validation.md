# Phase 2C.1: Packed-Tree Gold Validation

Phase 2C.1 is a correctness-diagnostic change. It does not change tree
construction, packed verification, greedy traversal, acceptance,
correction/bonus handling, KV commit semantics, or the default DFlash path.

## Source audit of the H100 failure

The Phase 2C exception combined three independent predicates into one `passed`
boolean: greedy argmax equality, ordered Top-8 equality, and the strict numerical
thresholds. The reported H100 values, maximum error `0.3125` and mean error
`0.003911118...`, are both below the unchanged thresholds `0.5` and `0.02`.
Consequently the old message does not establish that numerical tolerance caused
the failure: either argmax or ordered Top-8 differed for at least one node. The
old artifact flow then raised inside the dynamic module before returning the
record, leaving every gold JSONL empty.

This is the source-confirmed diagnostic defect. Whether the underlying H100
difference is harmless backend variation or a semantic/structural error remains
unknown until the comparison matrix below runs on H100.

## Logit alignment semantics

For flattened node `i` at depth `d`:

- its input token is at semantic position `prefix_length + d`;
- packed storage uses `cache_position = prefix_length + i`;
- `packed_logits[0, i]` and the last dense-path logit both predict semantic
  position `prefix_length + d + 1`;
- a parent logit predicts its accepted child, or correction/bonus when traversal
  stops at that parent;
- dense input is exactly root plus the node's draft-token path, so root is added
  once;
- both sides clone the same persistent cache snapshot taken before packed
  verification.

Every record includes token path, parent/depth, query/KV lengths, semantic and
storage positions, query indices, prediction positions, and whether the node is
an actual verifier decision. Mask visibility is compared with the independently
constructed prefix-plus-ancestor specification. Sibling, cousin, or future
visibility is a structural failure.

## Comparison modes

- `native`: existing packed 4D tree mask versus per-path implicit causal
  reference. This preserves the original failing comparison and records that
  mask/query shapes differ.
- `equal_shape`: Top-1 only. Packed root-plus-15 and a linear root-plus-15
  reference use the same query/KV length, dtype, cache snapshot, positions,
  cache positions, and explicit causal 4D mask.
- `same_mask`: packed tree versus per-path reference with explicit additive 4D
  masks on both sides. Allowed semantic key sets are identical, while physical
  query shapes can differ.
- `math`: reruns packed and same-mask dense references inside
  `torch.nn.attention.sdpa_kernel([SDPBackend.MATH])`. If that API is unavailable,
  the mode is recorded as unavailable; it never silently claims math execution.

Python records the configured/forced interface only. The actual CUDA kernel is
`null` until Nsight confirms it.

## Per-node diagnostics and classification

Each available node records maximum/mean/P50/P95/P99 absolute error, explicit
relative-error epsilon, maximum-error vocabulary ID and both values, argmax and
ordered Top-8 details, Top-8 set overlap, selected-token ranks, Top-1/Top-2
margins, and verifier-decision relevance.

Statuses are:

- `pass_exact`: structural checks, strict numerical thresholds, argmax, and
  ordered Top-8 all agree.
- `pass_semantic_with_numerical_variation`: all structural and actual verifier
  decisions agree, final/round semantics agree, native differs, and at least one
  equal-shape/same-mask/math control passes exactly.
- `fail_semantic`: verifier decision, accepted/correction/bonus, next-step, or
  output IDs disagree; unresolved native mismatch without control evidence also
  stops rather than being relabeled numerical variation.
- `fail_structural`: mask, position, query/logit alignment, cache length, gather,
  or retained path is invalid.
- `not_run`: an unavailable mode or a check gated by a prior failure.

Ordered Top-8 mismatch is reported separately and is not automatically a greedy
generation failure. The original `0.5` maximum and `0.02` mean thresholds remain
unchanged, are CLI-visible, and are written to manifest and every record.

## Layer and KV diagnostics

Optional layer localization reruns a same-mask packed/dense pair with temporary
forward hooks. It records embedding output, every target decoder-layer output,
final norm, and LM-head errors without persisting hidden tensors. The named P99
threshold identifies the first stage that exceeds it; hidden-state argmax is not
invented.

KV validation runs only after logit structural/semantic validation passes. It
compares every compacted K/V layer with dense committed-sequence recomputation,
records max/mean/P99 error, cache lengths, flattened/storage indices, retained
tokens/depths and discarded count, then compares next-root logits and Top-8.
Cache/gather/sibling errors and next-step argmax mismatch are hard failures. If
logits fail first, KV has `status=not_run`,
`execution_status=not_run_due_to_prior_failure`, and
`reason=prior_logits_gold_failure`.

## Failure-safe artifacts

The runner atomically flushes accumulated JSONL and summaries before raising a
classified gold failure. A failed run therefore retains:

```text
gold-logit-checks.jsonl
gold-kv-checks.jsonl
gold-layer-localization.jsonl
gold-comparison-summary.json
gold-comparison-summary.csv
correctness-summary.json
manifest.json
terminal.log
```

`correctness-summary.json` includes gold status, failure class, exception,
completed checks, and uncompleted checks. `manifest.json` is set to `error` by
the outer failure handler.

## H100 order of execution

Instrumented code and materialized dependencies changed. Materialize into a new
directory and never overwrite Phase 2B or Phase 2C directories. Run the 1K
Top-1 four-mode diagnostic first. Continue to layerwise and 16K only if mask,
alignment, verifier decisions, KV next-step decision, final IDs, and persisted
artifacts pass. Complete copyable commands are provided in the implementation
handoff for the committed short SHA.
