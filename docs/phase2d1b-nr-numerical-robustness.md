# Phase 2D.1B-NR: Numerical Robustness Characterization

Phase 2D.1B-NR is a diagnostic-only wrapper around the existing Phase 2C and
Phase 2D.1B execution path. It starts one fresh process for every
sample/context/configuration/repeat observation. It does not alter selector
decisions, tree construction, packed verification, KV compaction, BF16/SDPA,
decision stabilization, correctness gates, or target-forward counts.

The wrapper deliberately runs `subprocess.run(..., check=False)`. A non-zero
Phase 2C exit is retained as a valid observation only when persisted artifacts
show a complete actual numerical decision divergence, a structural failure, or
reference instability. Missing/malformed artifacts, model/CUDA failures,
unexpected non-zero exits, and incomplete correctness evidence invalidate the
study. Actual output divergence is never relabeled as an online pass.

Phase 2C gives `manifest.json`, `correctness-summary.json`, and
`source-provenance.json` the shared wrapper status `failed` for an expected
`Phase2CCorrectnessFailure`. NR accepts that exact wrapper transition only;
the loaded module hash, materialized `dflash.py` hash, dependency hashes,
materialization identity, and source-match flags must still validate. Any real
source provenance mismatch remains an infrastructure failure.

## Count and margin semantics

- `physical_case` is task × source/sample × context.
- `configuration_case` is physical case × selector/config × node budget.
- `independent_repeat_observation` is configuration case × fresh process
  repeat.
- N=16 and N=32 are configuration strata, not independent physical samples.
- Repeats of one source remain clustered under that source and are not treated
  as independent source evidence.

The canonical round signal is the existing shadow diagnostic
`minimum_path_margin`; if absent but `verifier_margins` exists, its minimum is
used with explicit provenance. Missing margins remain null. For divergent
outputs, only rounds through and including the first actual divergent tree
round enter the offline threshold sweep. For exact outputs, only
non-terminal/non-truncated rounds enter it.

Thresholds are fixed at `0,0.01,0.05,0.1,0.25,0.5,1.0`. The offline predicate
is `minimum_verifier_margin <= threshold`. It is an opportunity
characterization, not an online stabilization implementation or learned
classifier.

## Outcome categories

Every process receives exactly one category:

1. `pass_actual_exact`
2. `actual_numerical_decision_divergence`
3. `actual_unresolved_semantic_divergence`
4. `diagnostic_only_numerical_variation`
5. `actual_structural_failure`
6. `reference_instability`
7. `incomplete_correctness_evidence`
8. `infrastructure_failure`

Categories 2–6 remain study data and execution continues. Categories 7–8 stop
the current stage and leave the NR root explicitly incomplete.

`actual_unresolved_semantic_divergence` means stable, hashed reference/tree
token lists establish an actual mismatch and its first divergent offset, while
the available controls do not establish a numerical verifier attribution. The
raw failure class is retained as `unresolved_attribution_reason`; this outcome
remains diagnostic study data and does not abort the cohort. Missing output
relations, missing first-divergence evidence, and unknown failure classes remain
`incomplete_correctness_evidence`.

`diagnostic_only_numerical_variation` means the actual Original-DFlash and tree
outputs remain token exact while a replay/Gold comparison records semantic or
strict numerical variation. The observation explicitly records
`actual_execution_semantic_failure=false` and
`replay_only_variation_observed=true`; Gold comparison mode, verifier decision,
threshold, backend, dtype, and numerical evidence remain in
`replay-semantic-events.jsonl`. This diagnostic result does not abort a cohort.
`diagnostic_classification_evidence_sufficient` is separate from the stricter
`gold_evidence_complete`: an output-exact observation may be conclusively
diagnostic even when downstream Gold checks were skipped. It still requires a
known numerical attribution, an actual replay record with passing structural
status, stable actual-output evidence, round/stabilization records, no actual
semantic consequence, and no unknown or unrelated Gold skip.

Gold incompleteness is classified causally rather than by treating every
`uncompleted_checks` entry alike. An actual numerical divergence may remain a
complete NR observation when `kv_compaction`/`kv_next_step_logits` were
explicitly skipped with `prior_logits_gold_failure`, but only if the actual
reference/tree mismatch and divergence attribution are complete, native,
same-mask, and math Gold-logit records establish the first verifier-decision
divergence, and traversal/structural checks pass through that round. The NR
observation preserves the skipped checks and reasons in
`downstream_gold_skipped_checks` and `downstream_gold_skip_reasons`. Unknown
reasons, unrelated skipped checks, absent comparison modes, or incomplete
traversal remain `incomplete_correctness_evidence` and stop the study.
The native/same-mask/math and traversal-through-variation requirements remain
the stricter test for declaring the Gold evidence itself complete. They are not
used to mislabel otherwise sufficient output-exact diagnostic evidence as
genuinely incomplete.

## Artifacts

The output root contains:

- `nr-manifest.json`
- `exact-case-repeats.jsonl`
- `exact-case-reproducibility-summary.json`
- `cohort-subruns.jsonl`
- `cohort-case-summary.csv`
- `context-divergence-summary.csv`
- `policy-divergence-summary.csv`
- `divergence-events.jsonl`
- `replay-semantic-events.jsonl`
- `round-margin-events.jsonl`
- `margin-trigger-sweep.csv` and `margin-trigger-sweep.json`
- `research-question-summary.json`
- `nr-decision.json`
- `subrun-index.jsonl`
- `terminal.log`
- every untouched Phase 2C process directory under `raw-subruns/`

`--resume` appends a disjoint stage to an existing output. It rejects changed
repository, dataset, materialization, frozen-selector, threshold, or generation
provenance and rejects duplicate subrun identities. It never overwrites an
existing raw subrun.

No latency, speedup, throughput, acceptance-performance, or stabilization-cost
claim is emitted by this phase.

## Gate NR-0: checkout and provenance

```bash
cd ~/data/dflash-longctx
git fetch origin
git checkout codex/phase2d1b-nr-numerical-robustness
git pull --ff-only origin codex/phase2d1b-nr-numerical-robustness
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/codex/phase2d1b-nr-numerical-robustness)"
test "$(git merge-base HEAD d60e7db0514053ac580f2bfa9ca8cf7a534c1b49)" = "d60e7db0514053ac580f2bfa9ca8cf7a534c1b49"
test -z "$(git status --porcelain)"

source .venv/bin/activate

export TARGET_MODEL_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-b968826d
export DRAFT_MODEL_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16-phase2d1b-instrumented-3339d83dc214
export LONGBENCH_ROOT=/mnt/sevenT/wrz_data/datasets/LongBench
export FROZEN_SELECTOR=outputs/phase2d1b-frozen-selector/frozen-selector.json
export CAPTURE_GOV=outputs/phase2b-capture-gov-report-matched-3
export NR_OUT=outputs/phase2d1b-nr-numerical-robustness

test -d "$TARGET_MODEL_DIR"
test -f "$DRAFT_MODEL_DIR/materialization.json"
test -f "$LONGBENCH_ROOT/gov_report.jsonl"
test -f "$FROZEN_SELECTOR"
test -d "$CAPTURE_GOV"
```

No new materialization is required.

## Gate NR-1: CPU-only sanity

```bash
python -m unittest tests.test_phase2d1b_nr -v
python -m pytest tests/test_phase2d1b_nr.py -q
python experiments/phase2d1b_nr_numerical_robustness.py --help
python scripts/check_phase2d1b_nr_acceptance.py --help

python - <<'PY'
import json
from pathlib import Path
from jsonschema import Draft202012Validator

schemas = sorted(Path("docs/schemas").glob("phase2d1bnr-*.schema.json"))
for path in schemas:
    Draft202012Validator.check_schema(json.loads(path.read_text()))
print(f"validated {len(schemas)} Phase 2D.1B-NR schemas")
PY
```

## Gate NR-1.5: one-repeat exact-case smoke

A confirmed global-fixed numerical divergence is retained as data even when
Phase 2C returns non-zero. Causally skipped downstream Gold-KV checks remain
visible in `exact-case-repeats.jsonl`.

```bash
export NR_SMOKE_OUT=outputs/phase2d1b-nr-smoke-1repeat-fixed
test ! -e "$NR_SMOKE_OUT"

python experiments/phase2d1b_nr_numerical_robustness.py \
  --target-model-path "$TARGET_MODEL_DIR" \
  --materialized-draft-path "$DRAFT_MODEL_DIR" \
  --longbench-root "$LONGBENCH_ROOT" \
  --frozen-selector-config "$FROZEN_SELECTOR" \
  --capture-dir "$CAPTURE_GOV" \
  --exact-case-sample-id gov_report-fd287bbfd3bdbdc9 \
  --exact-case-context-length 1K \
  --exact-case-repeats 1 \
  --modes top1_packed_tree,global_fixed \
  --node-budgets 16,32 \
  --max-new-tokens 32 \
  --no-run-cohort \
  --output-dir "$NR_SMOKE_OUT"

python scripts/check_phase2d1b_nr_acceptance.py \
  --output-dir "$NR_SMOKE_OUT"
```

## Gate NR-2: exact known-case reproducibility

This creates the output root and runs 10 fresh processes for each of Top-1 N=16
and global-fixed N=16.

```bash
test ! -e "$NR_OUT"

python experiments/phase2d1b_nr_numerical_robustness.py \
  --target-model-path "$TARGET_MODEL_DIR" \
  --materialized-draft-path "$DRAFT_MODEL_DIR" \
  --longbench-root "$LONGBENCH_ROOT" \
  --frozen-selector-config "$FROZEN_SELECTOR" \
  --capture-dir "$CAPTURE_GOV" \
  --exact-case-sample-id gov_report-fd287bbfd3bdbdc9 \
  --exact-case-context-length 1K \
  --exact-case-repeats 10 \
  --modes top1_packed_tree,global_fixed \
  --node-budgets 16,32 \
  --max-new-tokens 32 \
  --no-run-cohort \
  --output-dir "$NR_OUT"
```

## Gate NR-3: matched-3 cohort

This resumes the same root and appends, without rerunning Gate NR-2, the 45
default cohort configuration cases: 3 sources × 5 contexts × Top-1 N=16,
global-fixed N=16, and global-fixed N=32. Each has one fresh-process repeat.

```bash
python experiments/phase2d1b_nr_numerical_robustness.py \
  --target-model-path "$TARGET_MODEL_DIR" \
  --materialized-draft-path "$DRAFT_MODEL_DIR" \
  --longbench-root "$LONGBENCH_ROOT" \
  --frozen-selector-config "$FROZEN_SELECTOR" \
  --capture-dir "$CAPTURE_GOV" \
  --cohort-context-lengths 1K,4K,8K,16K,32K \
  --cohort-repeats 1 \
  --modes top1_packed_tree,global_fixed \
  --node-budgets 16,32 \
  --max-new-tokens 32 \
  --no-run-exact-case \
  --resume \
  --output-dir "$NR_OUT"
```

## Gate NR-4: acceptance and compact results

```bash
python scripts/check_phase2d1b_nr_acceptance.py --output-dir "$NR_OUT"

python - <<'PY'
import csv, json, os
from pathlib import Path

root = Path(os.environ["NR_OUT"])
print("\nEXACT REPRODUCIBILITY")
print(json.dumps(json.loads((root / "exact-case-reproducibility-summary.json").read_text()), indent=2))

print("\nPOLICY DIVERGENCE RATES")
with (root / "policy-divergence-summary.csv").open() as stream:
    for row in csv.DictReader(stream):
        print(row)

print("\nCONTEXT TRENDS")
with (root / "context-divergence-summary.csv").open() as stream:
    for row in csv.DictReader(stream):
        print(row)

print("\nCOHORT OVERALL MARGIN SWEEP")
sweep = json.loads((root / "margin-trigger-sweep.json").read_text())
for row in sweep["rows"]:
    if row["experiment"] == "cohort" and row["scope"] == "overall":
        print(row)

print("\nRQ1-RQ7")
print(json.dumps(json.loads((root / "research-question-summary.json").read_text()), indent=2))
print("\nFINAL NR DECISION")
print(json.dumps(json.loads((root / "nr-decision.json").read_text()), indent=2))
PY
```

Do not run Phase 2D.1B Gates 2–7 as part of this characterization.
