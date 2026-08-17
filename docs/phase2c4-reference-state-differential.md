# Phase 2C.4: Original-DFlash-vs-Tree Reference-State Differential

Phase 2C.4 is diagnostic-only. It does not alter tree construction, packed
verification, traversal, acceptance, root selection, KV commit, sampling,
decision stabilization, thresholds, or formal timing. Target-only remains an
observational control and is not used as the reference-state oracle.

Phase 2C.4.1 corrects only diagnostic semantics and causal attribution. It does
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

- `native_online_verifier_flip` is actual online evidence and alone can support
  `local_packed_verifier_numerical_divergence`;
- `same_mask_control_flip` and `math_control_flip` are counterfactual backend or
  query-shape controls and cannot independently become the online root cause.

Classification searches all aligned prefixes in this order: any actual native
online verifier flip; the first same-entry actual commit/root mismatch not
already explained by an earlier cause; a faithful target-decision divergence
preceded by KV/hidden drift; and a strictly aligned actual proposal drift that
changes the selected/accepted path. Everything else is
`unresolved_reference_state_divergence`. Each summary records the selected
evidence prefix, source, priority, unresolved reason, and downstream-only
consequences. This global priority prevents an early weak Top-K variation from
masking a later actual native verifier flip.

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

`first_state_divergent_aligned_prefix` may identify an early numerical or
capture variation. `first_semantically_consequential_state_divergence` is more
restrictive and uses only fidelity-gated actual semantic evidence.

This phase diagnoses the three targeted Shadow divergences only. It does not
authorize periodic refresh, margin tuning, a state-refresh heuristic, or any
new stabilization algorithm.
