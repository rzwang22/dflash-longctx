# Phase 2D.2A-H2.1 primary held-out N16 evaluation

H2.1 executes only the immutable H2.0 primary matrix at `N=16` under
`same_actual_verification_work`. It does not regenerate the cohort, authorize
H2.2/N32, select a family winner, or enter Phase 2D.2B/2D.3.

## Execution grouping audit

The existing `experiments/phase2d2_robust_conditional_refinement.py` runner
accepts one explicit `--sample-id` and one fixed configuration family. It can
accept more than one context length, but its correctness summary and external
Phase2D2 checker operate on the whole output directory. Grouping the short and
long context into one directory could therefore make a scientific failure in
one context contaminate the other context's cell status.

H2.1 consequently uses one unchanged existing-runner invocation per frozen
matrix row: 112 logical rows, 112 target/draft model-load groups. This is the
smallest safe grouping with independent per-cell artifacts and checker status.
Each refined cell resolves `--base-result-dir` to the exact same task, physical
sample, context prefix, N16, verifier, and protocol base cell. No online or
instrumented model source is changed.

The frozen H2.0 `candidate_merge_mode` has matrix-level semantics. It is
`base_only` for the base execution category and `refined_tree` for every
refined execution category; `refined_tree` is not an algorithmic merge-mode
string. The Phase2D2 result manifest instead records the selected frozen
configuration's runtime algorithm: the conditional family records
`conditional_rerank`, while both frozen union families record `union`. H2.1
loads the canonical frozen-refinement artifact with the existing
`validate_frozen_refinement()` parser and validates the exact family ID,
artifact hash, algorithmic merge mode, and refinement parameters against the
selected entry. It never derives runtime mode by parsing a family name and
never compares the H2.0 `refined_tree` category directly with the runtime mode.

## Freeze and execution contract

`scripts/run_phase2d2a_h21_primary.py` validates the H2.0 directory with the
existing checker and requires the exact frozen SHA
`d5fdc816b183c19bd5683fb89a3c5e1ace641a15d2ac68da7b6dc0e56a678856`
before checking runtime model paths or launching a subprocess. It also checks
the manifest status, canonical component hashes, the exact 112-row N16 matrix,
row/cohort identities, four-family inventory, Original-DFlash label
provenance, and q1 exclusion.

Without `--execute-frozen-primary`, the command is plan-only: it writes and
prints the exact 112-row plan, reports task/family counts and 112 expected model
loads, and executes no model forward. The explicit flag authorizes only the
frozen primary matrix; the immutable H2.0 authorization fields remain false.

Results use:

```text
<output-root>/runs/<task>/<sample-id>/<nominal-context>/<family>/
```

Each completed subrun contains `h21-external-status.json`, binding its frozen
row identity, freeze SHA, repository commit, command, runner return code, and
external checker return code. A matching completed cell is skipped on resume.
An incomplete non-empty directory is preserved and reported, never deleted or
overwritten. A conflicting freeze/identity is a hard infrastructure rejection.
Scientific checker failures with complete, identity-valid artifacts are kept
as `correctness_invalid`; broken or missing evidence is
`infrastructure_failed` and makes the matrix incomplete.

The configured child Python entry point is normalized with absolute-path
semantics without dereferencing symlinks. In particular, a uv-backed
`.venv/bin/python` remains `.venv/bin/python`; converting it with
`Path.resolve()`/`realpath()` would instead launch the bare uv CPython and lose
the virtualenv site-packages. Execution mode runs a child-process preflight
with that exact entry point before creating any scientific result directory.
The preflight records the parent and child executable, symlink/realpath
diagnostics, child Python version, and import results for `jsonschema`, `torch`,
and `transformers` in the execution manifest. Plan-only mode does not run this
execution preflight.

An execution preflight failure stops with nonzero status and leaves all 112
rows `not_started`. After scheduling begins, an incomplete artifact set,
identity failure, dependency/import/model-load failure, or other
infrastructure failure is persisted for the current row and trips a circuit
breaker; no later row is launched. Complete identity-valid scientific
artifacts rejected by the Phase2D2 correctness checker remain
`correctness_invalid` and do not trip this infrastructure circuit breaker.

`--sentinel-pair-only` writes
`phase2d2a-h21-sentinel-gate.json`. Its return code is zero only when both
frozen cells completed with runner/checker RC 0, retained their exact frozen
identity, contain every required Phase2D2 artifact, and pass the hard
Original-DFlash/tree correctness checks. Infrastructure failure, incomplete
evidence, blocked dependency, or scientific correctness rejection makes the
sentinel return nonzero without deleting evidence.

The sentinel retains the canonical loader classification. An
`infrastructure_failed` cell reports
`cell_status_infrastructure_failed:<concrete reason>` and is not collapsed into
`hard_correctness_not_accepted`; that latter reason is reserved for a
`correctness_invalid` scientific result. After a requested execution scope
finishes, the manifest counters are reconciled once with canonical `load_cell`
classification. This keeps `completed_cell_count`,
`infrastructure_failed_cell_count`, and
`scientific_correctness_invalid_cell_count` consistent without repeatedly
reloading every completed artifact after every row.

## Aggregation semantics

The aggregator retains all 112 cells and all 84 exact refined/base pairs.
Correctness-invalid pairs contribute neither positive nor negative benefit
metrics. Candidate-unavailable positions do not enter coverage denominators.

Within each task/context/family stratum, primary candidate coverage is:

```text
sum(candidate membership count) / sum(valid labelled reference positions)
```

Primary online efficiency is:

```text
sum(accepted draft token count) / sum(actual verified node count)
```

Per-sample ratios and gain signs remain in the paired and heterogeneity
artifacts, but a mean of sample ratios cannot replace either pooled primary
metric. Qasper 1K/8K and HotpotQA 1K/8K remain four independent strata. A
family is `stable_cross_task_same_work_signal` only when all four strata are
correctness-valid and have strictly positive selected-tree coverage gain and
accepted-per-verified-node gain. GovReport 1K/16K is reported separately as
same-task replication and cannot make or break that classification.

Timing retains raw formal repeats and reports only nominal latency change. Cost
aggregation keeps conditional-remask dependency-modeling work separate from
the always-fixed correctness-preserving verifier path, its round-entry
snapshot, and KV replacement. Only non-overlapping conditional and fixed
recheck timers are summed.

## Output artifacts

The output root contains:

- `phase2d2a-h21-execution-manifest.json`;
- `phase2d2a-h21-execution-plan.json`;
- `phase2d2a-h21-subrun-index.jsonl`;
- optional `phase2d2a-h21-sentinel-gate.json` after a sentinel invocation;
- `phase2d2a-h21-cell-summary.json` and `.csv`;
- `phase2d2a-h21-sample-paired-comparison.json` and `.csv`;
- `phase2d2a-h21-task-context-summary.json` and `.csv`;
- `phase2d2a-h21-family-summary.json` and `.csv`;
- `phase2d2a-h21-mechanism-summary.json` and `.csv`;
- `phase2d2a-h21-cost-summary.json` and `.csv`;
- `phase2d2a-h21-primary-conclusion.json`;
- the 112 unchanged Phase2D2 result directories under `runs/`.

The execution manifest records byte hashes for every H2.0 artifact and
canonical hashes for the frozen cohort, primary matrix, and aggregation rules.
The acceptance checker recomputes both the raw aggregates and these immutable
bindings.

## H100 commands

All commands below are non-destructive and intentionally avoid fail-fast shell
control and shell-killing assertions. Inspect each printed return code before
moving to the next scientific stage.

### H2.1-A — sync and validate the prerequisite

```bash
cd ~/data/dflash-longctx
git switch codex/phase2d2a-robust-conditional-remask
git pull --ff-only origin codex/phase2d2a-robust-conditional-remask
git status --short --branch
git rev-parse HEAD

source .venv/bin/activate
export FREEZE_DIR="$PWD/outputs/phase2d2a-h2-heldout-freeze-82919fa2978d"
export FREEZE_SHA="d5fdc816b183c19bd5683fb89a3c5e1ace641a15d2ac68da7b6dc0e56a678856"
export TARGET_MODEL_DIR="/mnt/sevenT/wrz_data/models/Qwen3-8B-b968826d"
export DRAFT_MODEL_DIR="/mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16-phase2d2a-instrumented-33e7159c3d50"
export LONGBENCH_ROOT="/mnt/sevenT/wrz_data/datasets/LongBench/data"
export FROZEN_SELECTOR="$PWD/outputs/phase2d1b-frozen-selector/frozen-selector.json"
export FROZEN_REFINEMENT="$PWD/outputs/phase2d2a-frozen-refinement-66ce72888b34.json"
export H21_CODE_SHA="$(git rev-parse --short=12 HEAD)"
export H21_OUTPUT_ROOT="$PWD/outputs/phase2d2a-h21-primary-n16-${H21_CODE_SHA}"
export CHILD_PYTHON="$PWD/.venv/bin/python"

python scripts/check_phase2d2a_h2_freeze.py "$FREEZE_DIR"
echo "H20_CHECKER_RC=$?"
```

Do not reuse or delete the failed
`outputs/phase2d2a-h21-primary-n16-495a6a819cd9` root. It is immutable failure
provenance. The new root is bound to the new repository commit while the H2.0
freeze directory and content SHA remain unchanged.

Likewise, do not resume the validator-failed sentinel root
`outputs/phase2d2a-h21-primary-n16-028651c3b356`. Its scientific artifacts are
retained, but its execution manifest is repository-commit-bound. Validation of
a newer repository commit must use a new commit-derived output root.

### H2.1-B — verify the exact child interpreter

```bash
python - "$CHILD_PYTHON" <<'PY'
import os
import sys
from pathlib import Path

child = Path(sys.argv[1])
print("parent_sys_executable:", sys.executable)
print("configured_child_executable:", os.path.abspath(child))
print("configured_child_is_symlink:", child.is_symlink())
print("configured_child_realpath:", os.path.realpath(child))
print("symlink_entry_preserved:", os.path.abspath(child) != os.path.realpath(child))
PY
echo "H21_PARENT_CHILD_PATH_RC=$?"

"$CHILD_PYTHON" -c 'import jsonschema, torch, transformers, sys; print("child_sys_executable:", sys.executable); print("imports: jsonschema/torch/transformers OK")'
echo "H21_CHILD_IMPORT_RC=$?"
```

### H2.1-C — plan-only validation (zero model forwards)

```bash
python scripts/run_phase2d2a_h21_primary.py \
  --freeze-dir "$FREEZE_DIR" \
  --freeze-content-sha256 "$FREEZE_SHA" \
  --output-root "$H21_OUTPUT_ROOT" \
  --target-model-path "$TARGET_MODEL_DIR" \
  --materialized-draft-path "$DRAFT_MODEL_DIR" \
  --longbench-root "$LONGBENCH_ROOT" \
  --frozen-selector-config "$FROZEN_SELECTOR" \
  --frozen-refinement-config "$FROZEN_REFINEMENT" \
  --python-executable "$CHILD_PYTHON"
echo "H21_PLAN_RC=$?"
```

Expected plan counts are GovReport 24, Qasper 24, HotpotQA 64, total 112,
with 112 model-load groups.

### H2.1-D — frozen sentinel pair

The sentinel is the first frozen Qasper 1K base cell plus the first frozen
refinement family. Both cells remain part of the final matrix; their result may
not be used to tune, select, remove, or replace anything.

```bash
python scripts/run_phase2d2a_h21_primary.py \
  --freeze-dir "$FREEZE_DIR" \
  --freeze-content-sha256 "$FREEZE_SHA" \
  --output-root "$H21_OUTPUT_ROOT" \
  --target-model-path "$TARGET_MODEL_DIR" \
  --materialized-draft-path "$DRAFT_MODEL_DIR" \
  --longbench-root "$LONGBENCH_ROOT" \
  --frozen-selector-config "$FROZEN_SELECTOR" \
  --frozen-refinement-config "$FROZEN_REFINEMENT" \
  --python-executable "$CHILD_PYTHON" \
  --execute-frozen-primary \
  --sentinel-pair-only
H21_SENTINEL_RC=$?
echo "H21_SENTINEL_RC=$H21_SENTINEL_RC"

python - "$H21_OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "phase2d2a-h21-sentinel-gate.json"
print(json.dumps(json.loads(path.read_text()), indent=2, sort_keys=True))
PY
echo "H21_SENTINEL_GATE_PRINT_RC=$?"
```

Do not proceed unless `H21_SENTINEL_RC=0` and the printed gate has
`"passed": true` for both frozen cells.

For the repository-side runtime-identity validator revalidation, stop here
after confirming `completed_cell_count=2`,
`infrastructure_failed_cell_count=0`, `not_started_cell_count=110`, both cell
gates `canonical_cell_status=valid`, sentinel `passed=true`, and RC 0. Do not
start the 112-row matrix as part of that validator validation task. The
remaining sections describe the later primary execution only after separate
authorization.

### H2.1-E — conditionally resume/execute the complete frozen primary matrix

```bash
if [ "$H21_SENTINEL_RC" -eq 0 ]; then
  python scripts/run_phase2d2a_h21_primary.py \
    --freeze-dir "$FREEZE_DIR" \
    --freeze-content-sha256 "$FREEZE_SHA" \
    --output-root "$H21_OUTPUT_ROOT" \
    --target-model-path "$TARGET_MODEL_DIR" \
    --materialized-draft-path "$DRAFT_MODEL_DIR" \
    --longbench-root "$LONGBENCH_ROOT" \
    --frozen-selector-config "$FROZEN_SELECTOR" \
    --frozen-refinement-config "$FROZEN_REFINEMENT" \
    --python-executable "$CHILD_PYTHON" \
    --execute-frozen-primary
  H21_PRIMARY_EXECUTOR_RC=$?
  echo "H21_PRIMARY_EXECUTOR_RC=$H21_PRIMARY_EXECUTOR_RC"
else
  H21_PRIMARY_EXECUTOR_RC=1
  echo "H21 primary skipped because sentinel did not pass"
fi
```

### H2.1-F — aggregate only after complete execution

```bash
if [ "$H21_PRIMARY_EXECUTOR_RC" -eq 0 ]; then
  python scripts/aggregate_phase2d2a_h21_primary.py \
    --freeze-dir "$FREEZE_DIR" \
    --freeze-content-sha256 "$FREEZE_SHA" \
    --output-root "$H21_OUTPUT_ROOT"
  H21_AGGREGATOR_RC=$?
  echo "H21_AGGREGATOR_RC=$H21_AGGREGATOR_RC"
else
  echo "H2.1 aggregation skipped because primary execution is incomplete"
fi
```

### H2.1-G — acceptance checker

```bash
if [ "${H21_AGGREGATOR_RC:-1}" -eq 0 ]; then
  python scripts/check_phase2d2a_h21_acceptance.py \
    --freeze-dir "$FREEZE_DIR" \
    --freeze-content-sha256 "$FREEZE_SHA" \
    --output-root "$H21_OUTPUT_ROOT"
  echo "H21_ACCEPTANCE_RC=$?"
else
  echo "H2.1 acceptance skipped because aggregation did not pass"
fi
```

### H2.1-H — print the primary evidence

```bash
python - "$H21_OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
load = lambda name: json.loads((root / name).read_text())
cells = load("phase2d2a-h21-cell-summary.json")
tasks = load("phase2d2a-h21-task-context-summary.json")
families = load("phase2d2a-h21-family-summary.json")
mechanism = load("phase2d2a-h21-mechanism-summary.json")
cost = load("phase2d2a-h21-cost-summary.json")
conclusion = load("phase2d2a-h21-primary-conclusion.json")

print("execution_complete:", conclusion["execution_complete"])
print("cell_counts:", {
    "valid": cells["valid_cell_count"],
    "correctness_invalid": cells["correctness_invalid_cell_count"],
    "infrastructure_failed": cells["infrastructure_failed_cell_count"],
})
for row in tasks["rows"]:
    print("task_context:", row["task"], row["nominal_prompt_length"],
          row["refined_family"], row["stratum_status"],
          row["candidate_coverage"], row["accepted_per_verified"],
          row["sample_heterogeneity"])
for row in families["rows"]:
    print("family:", row["refined_family"],
          row["cross_task_classification"],
          "govreport=", row["same_task_replication_classification"])
for row in mechanism["rows"]:
    print("mechanism:", row["task"], row["nominal_prompt_length"],
          row["refined_family"], row["mechanism_signal"])
for row in cost["rows"]:
    print("cost:", row["task"], row["nominal_prompt_length"],
          row["refined_family"],
          "verifier_gt_dependency=",
          row["h1_cost_hierarchy_generalizes_for_all_comparable_samples"])
print("primary_conclusion:", conclusion)
PY
echo "H21_PRINT_RC=$?"
```

Stop after inspection. Do not run H2.2/N32, choose a winner, retune the frozen
families/verifier, train an adapter, or enter Phase 2D.2B/2D.3.
