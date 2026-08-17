# Phase 2C.4.3: Cross-Run Evidence Aggregation

Phase 2C.4.3 is an offline aggregation step. It consumes existing, independent
Phase 2C.4.2 output directories and does not load a model, run CUDA, change the
single-run classifier, or alter formal timing. The input capture directories
are read-only; the aggregate output directory must be outside them.

Runs are grouped by `sample_id`, `nominal_prompt_length`, and the first actual
semantic divergence offset. Evidence is read only from that aligned divergence
row. The loader rejects contradictory replay match flags, missing actual-token
provenance, duplicate input directories, and evidence that treats replay as an
oracle.

The artifact preserves distributions for actual Original DFlash, actual tree,
packed tree, native replay, same-mask replay, and math replay tokens. It also
records reference-support rates, per-run classification counts, and the rate of
complete single-run classification evidence.

Actual Original DFlash remains the semantic oracle. Replay controls are only
causal-attribution evidence. Stable actual reference/tree/packed divergence is
therefore reported independently of replay reproducibility. A
`local_packed_verifier_numerical_divergence` aggregate requires every run to
carry complete single-run numerical attribution supported by same-mask or math
replay. Partial support is downgraded to
`unresolved_reference_state_divergence` with
`nonreproducible_numerical_control_support`; this preserves Case 3 when its
support is reproducible and prevents Case 4's 3/5 replay outcome from becoming
a confirmed numerical cause. No support leaves Case 2 unresolved.

Aggregate existing directories with:

```bash
python scripts/aggregate_phase2c43_evidence.py \
  /path/to/case4-run-1 \
  /path/to/case4-run-2 \
  /path/to/case4-run-3 \
  /path/to/case4-run-4 \
  /path/to/case4-run-5 \
  --output-dir outputs/phase2c43-case4-aggregate
```

This writes `cross-run-evidence.json`. Validate its schema and Case 4 acceptance
contract with:

```bash
python scripts/check_phase2c43_acceptance.py \
  --artifact outputs/phase2c43-case4-aggregate \
  --expected-case case4
```

The checker also accepts `case2` and `case3`. It can receive either the
aggregate directory or the JSON artifact path.
