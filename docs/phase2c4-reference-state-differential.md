# Phase 2C.4.2: Actual-Decision & Segmentation-Aware Diagnostic Fix

Phase 2C.4 is diagnostic-only. It does not alter tree construction, packed
verification, traversal, acceptance, root selection, KV commit, sampling,
decision stabilization, thresholds, or formal timing. Target-only remains an
observational control and is not used as the reference-state oracle.

Phase 2C.4.2 corrects only diagnostic semantics and causal attribution. It does
not change the dynamic model or any online execution path, so an existing
Phase 2C.4 materialization may be reused.

## Alignment and provenance

All checkpoints are keyed by the number of generated tokens already present in
the semantic prefix. Reference and tree round indices are metadata only and are
never used as the join key. A prefix can therefore map to different reference
and tree rounds.

The unprofiled Original DFlash diagnostic run defines the token reference and
records `actual_reference_checkpoint` state. For every requested prefix within
a committed linear-verification segment, its persistent target KV is a bounded
view of the actual cropped cache, not a dense full-prefix reconstruction. The
tree side is a separate `diagnostic_replay` using unchanged native packed-tree
verification and commit. Its output IDs must equal the formal tree control or
the evidence is marked partial. No missing side is silently represented as an
actual checkpoint; it is `not_directly_comparable`.

Reference-state and tree diagnostic work occurs after formal reference/tree
latency measurements. It is never included in performance fields.

Actual execution is the semantic oracle. At every absolute generated-token
offset, the runner joins the actual Original DFlash output token with the actual
tree diagnostic output token. Original DFlash is authoritative for speculative
verification correctness. Native dense, same-mask, math, and semantic-probe
replays are attribution controls only; their argmax may change with execution
shape, BF16 rounding, or SDPA backend and can never override actual output.

## Segmentation-aware transitions

A round transition is normalized into absolute semantic token events. If a
transition enters at generated prefix `p`, committed token `i` is attached to
offset `p + i`, and its next root is attached to offset `p + committed_count`.
Adjacent rounds may therefore describe the same token as a previous next root
and a later committed root. The duplicate evidence must agree, but its round
boundary and source kind need not match.

Reference and tree events are compared only at the same absolute offset. A
different accepted count, exit prefix, next-root kind, or next-root token at a
different offset is a segmentation difference, not a semantic mismatch. A row
is `segmentation_only_difference` when its round boundaries differ but every
overlapping absolute token event is exact. Such a row cannot establish semantic
consequence or `commit_or_root_transition_mismatch`.

## Captured state

At each requested absolute generated-token prefix, the diagnostic records:

- every target K/V layer's sequence length and shape;
- up to 16 deterministic, endpoint-covering values per K/V tensor;
- bounded-sample max, mean, and P99 absolute error plus stable hashes;
- the target hidden state for the committed semantic token;
- captured verifier Top-1, Top-8, and Top-1/Top-2 margin for that token;
- reference/tree round mapping and root/commit transition metadata.

At an actual round entry it additionally records draft-cache lengths and bounded
samples, the target-hidden chunk injected into the draft, draft-logit Top-8 and
margins at all 15 positions, Original DFlash proposals, tree Top-K candidates,
and the flattened candidate-tree input. When only one execution has a round
boundary at a prefix, draft-entry state is explicitly not directly comparable.

The runner pairs every captured semantic probe with the token actually returned
by the same Original DFlash or tree diagnostic execution. A probe is
`faithful` only when its argmax equals that actual token at the same generated
position. An unfaithful reference probe is retained as evidence about the
capture, but cannot establish target-KV or target-hidden causality and lowers
evidence completeness to `partial`. No differently shaped post-hoc forward is
substituted for an actual reference decision.

Candidate chains always describe 15 root-excluding draft positions: candidate
position 1 corresponds to the token immediately after the round root. The
artifact separately records the Original DFlash actual proposed chain, the
captured reference Top-1 chain, and the tree positional Top-1 chain. Different
lower-rank Top-K entries are diagnostic variation only. If the actual proposal
agrees with tree Top-1 while the captured reference Top-1 does not, the row is
`candidate_capture_alignment_inconsistent`; it cannot support causal
attribution and lowers evidence completeness.

The diagnostic never creates a complete float32 CPU copy of target or draft KV.
Layer errors are explicitly labelled `bounded_stable_samples`; they are evidence
of sampled numerical drift, not a claim about the full tensor maximum.

## Semantic consequence and classification

Raw numerical inequality is recorded separately from semantic consequence. A
KV or hidden sample difference, Top-8 ordering difference, margin variation, or
lower-rank candidate difference with unchanged actual decisions and transitions
is `mere_numerical_variation_only` and cannot by itself become a state-drift
root cause.

The supported evidence-based classifications are:

- `local_packed_verifier_numerical_divergence`;
- `target_kv_state_drift`;
- `target_hidden_state_drift`;
- `draft_state_or_candidate_drift`;
- `commit_or_root_transition_mismatch`;
- `unresolved_reference_state_divergence`.

Packed-verifier modes have distinct meanings:

- the actual packed-tree argmax is joined to the actual tree and Original
  DFlash tokens at the predicted absolute semantic position;
- `native_dense_replay_token_id`, `same_mask_replay_token_id`, and
  `math_replay_token_id` are explicitly replay controls;
- `same_mask` and `math` may support numerical attribution when they recover
  the actual Original DFlash token, but no replay is a correctness oracle.

`local_packed_verifier_numerical_divergence` requires an actual
Original-DFlash/tree token difference at one absolute offset, a structurally
valid packed verifier decision that matches the actual tree token rather than
the actual Original DFlash token, and supporting same-mask/math numerical
control evidence. The native dense replay may equal either side without
changing the classification.

Classification then considers a same-offset transition token mismatch only when
structurally valid packed-verifier evidence matches the actual reference token
but not the token the tree committed; a transition mismatch that merely repeats
an unexplained output difference is downstream evidence, not a cause. Later
priorities cover a faithful target-decision divergence preceded by KV/hidden
drift and a strictly aligned actual proposal drift that changes the selected
path. Everything else is
`unresolved_reference_state_divergence`. Each summary records the selected
evidence prefix, source, priority, unresolved reason, and downstream-only
consequences.

## Artifacts

An enabled run writes:

```text
reference-state-differential.jsonl
reference-state-differential.csv
reference-state-summary.json
first-reference-state-divergence.json
reference-state-layer-differential.jsonl
```

Each case summary includes the first output divergence, first aligned state
difference, first semantically consequential difference, affected component,
reference/tree round mapping, evidence completeness, and failure
classification. The files are also created with `not_run`/empty content on an
early failure.

`first_raw_numerical_state_difference` reports sampled numerical/structural
variation. `first_trusted_state_difference` excludes unaligned candidate
captures and is also the backward-compatible meaning of
`first_state_divergent_aligned_prefix`.
`first_candidate_capture_inconsistency` is an evidence-quality issue, not state
drift. `first_actual_semantic_divergence` comes only from actual output IDs.
Global diagnostic completeness is reported separately from local
`classification_evidence_complete`, so unrelated missing/candidate evidence
does not invalidate a complete Case 3 causal proof.

The framework-lazy helpers `scripts/phase2c42_results.py` and
`scripts/check_phase2c42_acceptance.py` extract the three targeted cases and
enforce their Phase 2C.4.2 acceptance criteria without importing Torch or
Transformers.

This phase diagnoses the three targeted Shadow divergences only. It does not
authorize periodic refresh, margin tuning, a state-refresh heuristic, or any
new stabilization algorithm.
