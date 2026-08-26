# Phase 2D.2A-H2.2 secondary N32 budget-expansion characterization

H2.2 executes only the immutable 112-row H2.0 secondary matrix at `N=32` and
uses accepted H2.1 N16 results as an immutable cross-budget anchor. Its role is
secondary characterization: it cannot change the H2.1 primary classification,
select a winner, authorize Phase 2D.2B/2D.3, or make a speedup claim.

## Frozen analysis and execution

Before execution the runner writes an analysis contract and exact execution
plan. Without `--execute-frozen-secondary`, it is plan-only and performs zero
model forwards. The contract defines no outcome-dependent or arbitrary
low-yield threshold. Marginal yield always uses measured per-round
`actual_verified_node_count` from `refinement-rounds.jsonl`, never configured
N. A zero actual-node delta makes marginal yield unavailable rather than zero.

The existing Phase2D2 experiment accepts one physical sample, one context, and
one family per invocation. H2.2 therefore retains the qualified H2.1 grouping:
112 deterministic cell directories and 112 target/draft model-load groups for
the full matrix. The first frozen Qasper 1K base/conditional pair is the
sentinel and remains part of the 112 rows. Virtualenv symlink preservation,
child import preflight, safe resume, exact row identity, base dependency,
infrastructure circuit breaker, and scientific-invalid classification are
reused from H2.1. No scientific result directory is created before preflight
passes.

## Comparison semantics

H2.2 reports two separate axes:

- Within N32, each refined family is compared with its exact N32 base cell. If
  measured work differs, it is labelled
  `combined_refinement_plus_actual_work_difference`, never same-work.
- Across budgets, exact task/sample/context/family pairs compare accepted H2.1
  N16 with H2.2 N32. Additional-node accepted and committed yields divide by
  the measured actual-node delta.

The paired incremental-budget decomposition reports the difference between
the observed N32 combined gain and N16 same-work refinement gain, as well as
the corresponding incremental actual work. It is descriptive observational
evidence, not a causal estimator. Base-only N16/N32 work distributions and
output equality are retained as a descriptive cap-consumption sanity check.

Qasper and HotpotQA are aggregated separately by task and context. GovReport
is same-task replication only. Candidate coverage and accepted/verified rates
use pooled counts within each task/context; per-sample distributions remain
available. Correctness-invalid cells remain in inventory but do not enter
benefit or marginal-yield aggregation. Infrastructure failures make execution
incomplete.

## Artifacts

The H2.2 root contains:

- `phase2d2a-h22-analysis-contract.json`;
- `phase2d2a-h22-execution-manifest.json`;
- `phase2d2a-h22-execution-plan.json`;
- `phase2d2a-h22-subrun-index.jsonl`;
- `phase2d2a-h22-sentinel-gate.json` after the sentinel;
- cell, within-N32, cross-budget, decomposition, task/context, mechanism, and
  cost JSON/CSV artifacts;
- `phase2d2a-h22-secondary-conclusion.json`;
- 112 unchanged Phase2D2 cell directories under `runs/` after full execution.

Every artifact binds the exact H2.0 SHA, accepted H2.1 artifact hashes, and
accepted H2.1.1 route hashes.

## H100 qualification workflow: stop after sentinel

The commands are non-destructive and do not use fail-fast shell settings.

```bash
cd ~/data/dflash-longctx
git switch codex/phase2d2a-robust-conditional-remask
git pull --ff-only origin codex/phase2d2a-robust-conditional-remask
git status --short --branch
git rev-parse HEAD
source .venv/bin/activate

export FREEZE_DIR="$PWD/outputs/phase2d2a-h2-heldout-freeze-82919fa2978d"
export FREEZE_SHA="d5fdc816b183c19bd5683fb89a3c5e1ace641a15d2ac68da7b6dc0e56a678856"
export H21_OUTPUT_ROOT="$PWD/outputs/phase2d2a-h21-primary-n16-2e2bc7c121d6"
export H211_OUTPUT_DIR="$PWD/outputs/phase2d2a-h211-final-route-$(git rev-parse --short=12 HEAD)"
export H22_OUTPUT_ROOT="$PWD/outputs/phase2d2a-h22-secondary-n32-$(git rev-parse --short=12 HEAD)"
export TARGET_MODEL_DIR="/mnt/sevenT/wrz_data/models/Qwen3-8B-b968826d"
export DRAFT_MODEL_DIR="/mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16-phase2d2a-instrumented-33e7159c3d50"
export LONGBENCH_ROOT="/mnt/sevenT/wrz_data/datasets/LongBench/data"
export FROZEN_SELECTOR="$PWD/outputs/phase2d1b-frozen-selector/frozen-selector.json"
export FROZEN_REFINEMENT="$PWD/outputs/phase2d2a-frozen-refinement-66ce72888b34.json"
export CHILD_PYTHON="$PWD/.venv/bin/python"

python scripts/check_phase2d2a_h2_freeze.py "$FREEZE_DIR"
echo "H20_REVALIDATION_RC=$?"

python scripts/check_phase2d2a_h21_acceptance.py \
  --freeze-dir "$FREEZE_DIR" \
  --freeze-content-sha256 "$FREEZE_SHA" \
  --output-root "$H21_OUTPUT_ROOT"
echo "H21_BACKCOMPAT_ACCEPTANCE_RC=$?"

python scripts/audit_phase2d2a_h211_final_route.py \
  --freeze-dir "$FREEZE_DIR" \
  --freeze-content-sha256 "$FREEZE_SHA" \
  --h21-output-root "$H21_OUTPUT_ROOT" \
  --output-dir "$H211_OUTPUT_DIR"
echo "H211_AUDIT_RC=$?"

python scripts/check_phase2d2a_h211_acceptance.py \
  --freeze-dir "$FREEZE_DIR" \
  --freeze-content-sha256 "$FREEZE_SHA" \
  --h21-output-root "$H21_OUTPUT_ROOT" \
  --output-dir "$H211_OUTPUT_DIR"
echo "H211_ACCEPTANCE_RC=$?"

python scripts/run_phase2d2a_h22_secondary.py \
  --freeze-dir "$FREEZE_DIR" \
  --freeze-content-sha256 "$FREEZE_SHA" \
  --h21-output-root "$H21_OUTPUT_ROOT" \
  --h211-output-dir "$H211_OUTPUT_DIR" \
  --output-root "$H22_OUTPUT_ROOT" \
  --target-model-path "$TARGET_MODEL_DIR" \
  --materialized-draft-path "$DRAFT_MODEL_DIR" \
  --longbench-root "$LONGBENCH_ROOT" \
  --frozen-selector-config "$FROZEN_SELECTOR" \
  --frozen-refinement-config "$FROZEN_REFINEMENT" \
  --python-executable "$CHILD_PYTHON"
echo "H22_PLAN_ONLY_RC=$?"

python scripts/run_phase2d2a_h22_secondary.py \
  --freeze-dir "$FREEZE_DIR" \
  --freeze-content-sha256 "$FREEZE_SHA" \
  --h21-output-root "$H21_OUTPUT_ROOT" \
  --h211-output-dir "$H211_OUTPUT_DIR" \
  --output-root "$H22_OUTPUT_ROOT" \
  --target-model-path "$TARGET_MODEL_DIR" \
  --materialized-draft-path "$DRAFT_MODEL_DIR" \
  --longbench-root "$LONGBENCH_ROOT" \
  --frozen-selector-config "$FROZEN_SELECTOR" \
  --frozen-refinement-config "$FROZEN_REFINEMENT" \
  --python-executable "$CHILD_PYTHON" \
  --execute-frozen-secondary \
  --sentinel-pair-only
H22_SENTINEL_RC=$?
echo "H22_SENTINEL_RC=$H22_SENTINEL_RC"

python - "$H22_OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
gate = json.loads((root / "phase2d2a-h22-sentinel-gate.json").read_text())
manifest = json.loads((root / "phase2d2a-h22-execution-manifest.json").read_text())
print("canonical_status:", manifest["status"])
print("cell_counts:", {
    "completed": manifest["completed_cell_count"],
    "correctness_invalid": manifest["scientific_correctness_invalid_cell_count"],
    "infrastructure_failed": manifest["infrastructure_failed_cell_count"],
    "not_started": manifest["not_started_cell_count"],
})
print("sentinel_passed:", gate["passed"])
print("cell_correctness:", [row["correctness"] for row in gate["cell_gates"]])
print("direct_actual_verification_work:",
      gate["direct_actual_verification_work"])
PY
echo "H22_SENTINEL_INSPECTION_RC=$?"
```

Stop here for the first H100 qualification. Do not start the remaining matrix
until the sentinel has been inspected and separately authorized.

## Later, separately authorized full/resume workflow

The same output root safely skips the two completed sentinel cells.

```bash
python scripts/run_phase2d2a_h22_secondary.py \
  --freeze-dir "$FREEZE_DIR" \
  --freeze-content-sha256 "$FREEZE_SHA" \
  --h21-output-root "$H21_OUTPUT_ROOT" \
  --h211-output-dir "$H211_OUTPUT_DIR" \
  --output-root "$H22_OUTPUT_ROOT" \
  --target-model-path "$TARGET_MODEL_DIR" \
  --materialized-draft-path "$DRAFT_MODEL_DIR" \
  --longbench-root "$LONGBENCH_ROOT" \
  --frozen-selector-config "$FROZEN_SELECTOR" \
  --frozen-refinement-config "$FROZEN_REFINEMENT" \
  --python-executable "$CHILD_PYTHON" \
  --execute-frozen-secondary
H22_EXECUTOR_RC=$?
echo "H22_EXECUTOR_RC=$H22_EXECUTOR_RC"

if [ "$H22_EXECUTOR_RC" -eq 0 ]; then
  python scripts/aggregate_phase2d2a_h22_secondary.py \
    --freeze-dir "$FREEZE_DIR" \
    --freeze-content-sha256 "$FREEZE_SHA" \
    --h21-output-root "$H21_OUTPUT_ROOT" \
    --h211-output-dir "$H211_OUTPUT_DIR" \
    --output-root "$H22_OUTPUT_ROOT"
  H22_AGGREGATOR_RC=$?
  echo "H22_AGGREGATOR_RC=$H22_AGGREGATOR_RC"
else
  H22_AGGREGATOR_RC=1
  echo "H2.2 aggregation skipped because execution is incomplete"
fi

if [ "$H22_AGGREGATOR_RC" -eq 0 ]; then
  python scripts/check_phase2d2a_h22_acceptance.py \
    --freeze-dir "$FREEZE_DIR" \
    --freeze-content-sha256 "$FREEZE_SHA" \
    --h21-output-root "$H21_OUTPUT_ROOT" \
    --h211-output-dir "$H211_OUTPUT_DIR" \
    --output-root "$H22_OUTPUT_ROOT"
  H22_ACCEPTANCE_RC=$?
  echo "H22_ACCEPTANCE_RC=$H22_ACCEPTANCE_RC"
else
  echo "H2.2 acceptance skipped because aggregation did not pass"
fi

python - "$H22_OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = (
    "phase2d2a-h22-n32-paired-comparison.json",
    "phase2d2a-h22-cross-budget-paired-comparison.json",
    "phase2d2a-h22-incremental-budget-decomposition.json",
    "phase2d2a-h22-task-context-summary.json",
    "phase2d2a-h22-mechanism-summary.json",
    "phase2d2a-h22-cost-summary.json",
    "phase2d2a-h22-secondary-conclusion.json",
)
for name in names:
    payload = json.loads((root / name).read_text())
    print(f"\n=== {name} ===")
    print(json.dumps(payload, indent=2, sort_keys=True))
PY
echo "H22_EVIDENCE_PRINT_RC=$?"
```
