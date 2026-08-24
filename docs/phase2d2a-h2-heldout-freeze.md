# Phase 2D.2A-H2.0 held-out cohort freeze

H2.0 freezes the held-out identities, context strata, execution matrices, and
aggregation rules before any H2 model output exists. It loads only the pinned
Qwen3-8B tokenizer from a local directory. It does not load target or draft
weights, require CUDA, execute H2.1/H2.2, or authorize either execution phase.

The successful status is `frozen_not_executed`; all model-loaded and execution
authorization fields are false.

## Target-blind exclusions

The freeze tool recursively scans explicitly supplied previous-result roots.
Only execution identity artifacts with these names are considered:

- `samples.jsonl`;
- `subrun-index.jsonl`;
- `exact-case-repeats.jsonl`;
- `cohort-subruns.jsonl`.

From each record it reads only `task`/`longbench_task`, `sample_id`,
`source_sample_id`, and `source_text_sha256`. It never reads generation output,
acceptance, candidate quality, latency, or target labels for exclusion or
selection. Duplicate observations are retained as provenance but collapse to
one exclusion identity. Matching any available task-qualified sample ID,
source ID, or source-text hash excludes that physical source.

This covers raw Phase 2C/2D execution directories and the NR index artifacts
without treating inventory-only eligible lists as previously executed samples.
The exclusion artifact records every scanned file and its SHA256.

## Inventory and deterministic selection

The inventory uses the same prompt templates, Qwen chat template,
`enable_thinking=False`, tokenizer-derived eligibility, and prefix construction
as the Phase 2D.2 runner. It reports 1K, 4K, 8K, 16K, and descriptive 32K
eligibility for GovReport, Qasper, and HotpotQA after exclusions. Each eligible
row includes the physical identity, source-text hash, actual templated prompt
length, context-prefix hash, and input-token hash. A prefix may fall at most 16
tokens below its nominal length, matching the existing workload contract.

The short anchor is always 1K. The common independent-task long anchor is the
first qualifying length in this exact order:

```text
16K -> 8K -> 4K
```

A length qualifies only when the post-exclusion intersection eligible at both
1K and that length contains at least three Qasper and eight HotpotQA physical
samples. A task-specific long anchor is forbidden. Descriptive 32K inventory
cannot become the primary cross-task anchor.

GovReport is selected independently as same-task replication. Its long anchor
uses the same `16K -> 8K -> 4K` order and requires three unused samples jointly
eligible at 1K and the candidate long length. It never increases
`independent_task_count`, which remains two.

Physical samples are ranked with the frozen seed
`phase2d2a-h2-heldout-v1`:

```text
SHA256(seed || NUL || task || NUL || source_sample_id
       || NUL || source_text_sha256)
```

The lowest hashes are selected: three Qasper, eight HotpotQA, and three
GovReport. The same physical identities are frozen at short and long context.
There is no manual selection, replacement, or post-output reselection.

If no common independent-task length or GovReport replication length meets the
frozen counts, the tool stops with an explicit `insufficient_*_eligibility`
status. It writes the complete inventory, CSV, exclusion provenance, and stop
manifest; it does not silently reduce a cohort.

## Frozen execution matrices

H2.1 is the primary N16 same-actual-verification-work matrix. H2.2 is the
secondary N32 budget-cap-expansion ablation. Both contain the exact same 14
physical samples, two task-specific context strata, and four configurations:

- `base_only`;
- `conditional_rerank-d1-p1-r2-w0.5`;
- `union-d2-p2-r2-w0.5`;
- `union-d4-p2-r4-w0.5`.

Each matrix therefore contains 112 frozen rows. Both retain
`top1_packed_tree`, `always_fixed_shape_recheck`, 128 new tokens, one warmup,
three formal repeats, BF16, and external batch one. Candidate labels remain the
Original-DFlash absolute continuation; q1 is excluded. H2.2 cannot override a
primary N16 same-work conclusion.

## Frozen aggregation semantics

Candidate coverage is pooled over valid labelled positions within each task.
Accepted tokens per verified node is computed from pooled counts:

```text
sum(accepted_draft_token_count) / sum(actual_verified_node_count)
```

It is not the mean of per-sample ratios. Per-physical-sample pairs and sign
distributions remain required so heterogeneous negative samples are visible.

`stable_cross_task_same_work_signal` requires all four independent
task/context strata—Qasper short/long and HotpotQA short/long—to be
correctness-valid and jointly positive for both coverage gain and accepted
tokens per verified node. A combined sample-weighted global pool cannot replace
this task-level rule. No significance threshold or speedup claim is introduced.

Correctness-invalid cells do not enter benefit aggregation and are not counted
as negative candidate-quality examples. Mechanism output preserves candidate
set coverage, refined-only recovery, rank improvement, and prefix survival.
Dependency-modeling and correctness-preserving verifier costs remain separate.

The freeze manifest hashes the canonical frozen cohort, primary matrix,
secondary matrix, and aggregation rules. Future H2.1 execution must supply and
match the exact `freeze_content_sha256` before it can run.

## H100 tokenizer-only freeze commands

These commands reuse existing artifacts and perform no model forward. They are
non-destructive and intentionally avoid `set -e`, `exit 1`, and shell-killing
assertions.

```bash
cd ~/data/dflash-longctx
git fetch origin
git checkout codex/phase2d2a-robust-conditional-remask
git pull --ff-only origin codex/phase2d2a-robust-conditional-remask

source .venv/bin/activate
export TARGET_MODEL_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-b968826d
export LONGBENCH_ROOT=/mnt/sevenT/wrz_data/datasets/LongBench/data
export PREVIOUS_RESULTS_ROOT=outputs
export H11_PARENT=outputs/phase2d2a-h1-known-d3c21f67310e

find "$H11_PARENT" -maxdepth 1 -type d \
  -name 'h1.1-budget-mechanism-audit-*' -print | sort

export H11_DIR="$(find "$H11_PARENT" -maxdepth 1 -type d \
  -name 'h1.1-budget-mechanism-audit-*' -print | sort | tail -n 1)"
export H20_OUT="outputs/phase2d2a-h2-heldout-freeze-$(date +%Y%m%d-%H%M%S)"

python scripts/freeze_phase2d2a_h2_cohort.py \
  --longbench-root "$LONGBENCH_ROOT" \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path "$TARGET_MODEL_DIR" \
  --previous-results-root "$PREVIOUS_RESULTS_ROOT" \
  --h11-dir "$H11_DIR" \
  --dataset-split test \
  --selection-seed phase2d2a-h2-heldout-v1 \
  --output-dir "$H20_OUT"
echo "H20_FREEZE_RC=$?"

python scripts/check_phase2d2a_h2_freeze.py "$H20_OUT"
echo "H20_CHECKER_RC=$?"

python - "$H20_OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads(
    (root / "phase2d2a-h2-freeze-manifest.json").read_text()
)
print("status:", manifest["status"])
print("freeze_content_sha256:", manifest["freeze_content_sha256"])
print("model_forward_executed:", manifest["model_forward_executed"])
print(
    "h2_primary_execution_authorized:",
    manifest["h2_primary_execution_authorized"],
)
print(
    "h2_secondary_execution_authorized:",
    manifest["h2_secondary_execution_authorized"],
)
PY
```

Stop after the H2.0 checker. Do not execute H2.1 or H2.2.

## Expected successful artifacts

Under `$H20_OUT`:

- `phase2d2a-h2-heldout-inventory.json`;
- `phase2d2a-h2-heldout-inventory.csv`;
- `phase2d2a-h2-exclusion-provenance.json`;
- `phase2d2a-h2-frozen-cohort.json`;
- `phase2d2a-h2-primary-matrix.json`;
- `phase2d2a-h2-secondary-matrix.json`;
- `phase2d2a-h2-aggregation-rules.json`;
- `phase2d2a-h2-freeze-manifest.json`.

No `third_party/`, online execution, selector, refinement, verifier, KV, or NR
file changes are required. No rematerialization is required.
