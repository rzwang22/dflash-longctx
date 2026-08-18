# Phase 2D.1B: Online Fixed-Budget Selector Validation

Phase 2D.1B is based exactly on Phase 2D.1A commit
`e48c5cc1362b918a8bb6ceb9b6282768e5fa1166`. It freezes the LOSO-validated
training procedure on all validated source documents, then evaluates global
fixed, context-conditioned, and decision-stump selectors through the existing
Phase 2C.4.3 packed-tree verification path. Offline target-prefix coverage is
kept separate from measured online acceptance.

The selector is target-blind at runtime. Its only inputs are the nominal prompt
length, root-inclusive node budget, and canonical summaries of the current
round's draft Top-16 log probabilities. The shared feature code is used by
Phase 2D.0, Phase 2D.1A, the CPU parity replay, and the materialized dynamic
module. The decision is made after draft Top-K extraction and before target
verification. Target logits, target hidden state, posterior, accepted depth,
correction token, and final output cannot enter the selector. Missing or
non-finite stump features use the frozen global fixed fallback. Equality uses
`feature_value <= threshold`, selecting the left leaf.

The online implementation copies the bounded 15x16 draft candidates to the
host so the exact canonical pure-Python features and policy constructors can be
used. This synchronization is not hidden: device-to-host wait/transfer, feature
calculation, decision, canonical tree construction, and tensor packing are
recorded separately. Unprofiled control CPU wall time is the formal
end-to-end measure and therefore includes this overhead. CUDA components come
only from a separate profiling path. Control/profile output IDs must agree.

`reference_dflash` is the original DFlash measurement produced beside every
Phase 2C tree subrun. `top1_packed_tree` isolates packed-engine cost. N=16 is a
same-actual-node comparison because all paths verify 16 root-inclusive nodes.
At N=32, Top-1 still has 16 actual nodes, so it is only a same-cap reference;
global/context/stump N=32 are mutually same-actual-node comparisons. The final
choice is based on end-to-end latency per returned token, not acceptance alone.

## Frozen configuration and parity

The freeze step strictly revalidates the three immutable Phase 2B capture
directories, Phase 2D.0 provenance, the persisted Phase 2D.1A selector events,
and every input hash. It then fits, separately for N=16 and N=32:

- a global fixed rule over all source documents;
- an exact-context rule with the global rule as fallback;
- a deterministic one-level stump with the original support constraints;
- a complete allowlist of canonical Top-16 policy configurations.

`frozen-selector.json` has no timestamp and is deterministic for identical
inputs. Its SHA-256 covers the semantic configuration. The adjacent manifest
records execution provenance. `check_phase2d1_selector_parity.py` reconstructs
every feature from the captures, replays all three selectors, and exercises
every allowlisted canonical constructor without Torch or Transformers.

## Cohorts and alignment

An in-sample mechanics run may reuse the 14 training sources to verify feature,
policy, root, KV, timing-pipeline, and 32K mechanics. It is explicitly marked
`independent_generalization=false` and is not generalization evidence.

With `--exclude-selector-training-sources`, all frozen training source IDs are
removed before model loading. Inventory reports total eligible, excluded, and
remaining source IDs at every context length. Nested prefixes remain attached
to one source. No training source is reintroduced. If GovReport has no
remaining 32K source, the inventory records
`independent_32k_evidence_unavailable=true`; a separate training-source 32K
mechanics run must not be relabeled as independent evidence.

Offline/online alignment is optional and requires both `--phase2d1-dir` and all
three `--capture-dir` arguments. A row is comparable only when task, source,
context, generation repeat, round index, absolute generated root offset, root
token, and source hash match. Different round segmentation is never forced into
alignment.

## Artifacts and decision

The root online directory always retains manifest, inventory, frozen config,
selector decisions, round and sample records, JSON/CSV summaries and
comparisons (including repeat/control-profile cross-run comparison), correctness,
exact-root alignment, environment, source provenance, terminal log, and Phase 2C
gold/divergence artifacts. Each underlying Phase 2C subrun is preserved under
`phase2c-subruns/`.

The automatic decision is one of the registered Phase 2D.1B values. Correctness
or target-blind failure has first priority. Phase 2C shadow-mode diagnostics
remain available, but any observed Original-DFlash/tree output divergence is a
Phase 2D.1B correctness failure and can never yield an adoption decision.
For Phase 2D.1B subruns, actual Original-DFlash/tree execution is the semantic
oracle. Gold logit/KV runs are replay-side attribution controls. A replay
verifier mismatch, strict numerical threshold exceedance, or Top-K ordering
change remains fully recorded, but is diagnostic when actual outputs and Top-1
accepted/committed round semantics are exact and structural, mask, position,
KV-transition, traversal, and evidence-completeness checks pass. Such a record
is classified as `replay_semantic_variation_without_actual_consequence`.

A replay mismatch becomes hard only when connected to an actual output,
accepted/committed-round, committed-path, or KV-transition consequence.
Structural/mask/position/traversal failures and incomplete Gold evidence remain
independent hard failures; they are never whitelisted by a replay label.
`correctness-summary.json` explicitly records
`actual_execution_correctness_passed`, `actual_execution_semantic_failure`,
`replay_semantic_variation_observed`, its round indices, whether it has an
actual consequence, `diagnostic_numerical_variation_observed`, and the mandatory
actual-output exactness contract.
Inadequate independent evidence is
reported explicitly. A configuration is not adoptable unless online
accepted/committed tokens improve and formal latency per returned token falls.
A stump must beat the context rule by its configured extra margin; otherwise
the simpler context rule wins. N=32 must beat the best N=16 end-to-end result.
These thresholds are engineering gates, not statistical significance claims.

No online acceptance, round reduction, latency benefit, or speedup is claimed
until the following H100 gates produce accepted artifacts.

## H100 setup and Gate 0

```bash
git fetch origin
git checkout codex/phase2d1b-online-selector-validation
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/codex/phase2d1b-online-selector-validation)"

uv sync --extra transformers
source .venv/bin/activate

export TARGET_MODEL_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-b968826d
export DRAFT_WEIGHT_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16
export LONGBENCH_ROOT=/mnt/sevenT/wrz_data/datasets/LongBench
export PHASE2D0_DIR=outputs/phase2d0-dependency-opportunity-v2
export PHASE2D1_DIR=outputs/phase2d1-selector-realization
export CAPTURE_GOV=outputs/phase2b-capture-gov-report-matched-3
export CAPTURE_QAS=outputs/phase2b-capture-qasper-matched-3
export CAPTURE_HOT=outputs/phase2b-capture-hotpotqa-matched-8
export FROZEN_DIR=outputs/phase2d1b-frozen-selector
export FROZEN_SELECTOR=$FROZEN_DIR/frozen-selector.json

python scripts/phase2d1_freeze_selector.py \
  --phase2d1-dir "$PHASE2D1_DIR" \
  --phase2d0-dir "$PHASE2D0_DIR" \
  --capture-dir "$CAPTURE_GOV" \
  --capture-dir "$CAPTURE_QAS" \
  --capture-dir "$CAPTURE_HOT" \
  --output-dir "$FROZEN_DIR"

python scripts/check_phase2d1_frozen_selector.py \
  --frozen-selector-dir "$FROZEN_DIR"

python scripts/check_phase2d1_selector_parity.py \
  --frozen-selector "$FROZEN_SELECTOR" \
  --phase2d1-dir "$PHASE2D1_DIR" \
  --phase2d0-dir "$PHASE2D0_DIR" \
  --capture-dir "$CAPTURE_GOV" \
  --capture-dir "$CAPTURE_QAS" \
  --capture-dir "$CAPTURE_HOT" \
  --output outputs/phase2d1b-selector-parity.json
```

The instrumented code changed, so a new materialization directory is required.
Do not overwrite any validated Phase 2B/2C/2D directory.

```bash
export SHORT_SHA=$(git rev-parse --short=12 HEAD)
export DRAFT_MODEL_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16-phase2d1b-instrumented-$SHORT_SHA

python scripts/materialize_dflash.py \
  --source-model-path "$DRAFT_WEIGHT_DIR" \
  --output-dir "$DRAFT_MODEL_DIR" \
  --implementation instrumented
```

## Gates 1–4: mechanics and gold

Gate 1, known-sample reference/Top-1/global N=16:

```bash
python experiments/phase2d1b_online_selector.py \
  --target-model-path "$TARGET_MODEL_DIR" \
  --materialized-draft-path "$DRAFT_MODEL_DIR" \
  --longbench-root "$LONGBENCH_ROOT" --longbench-task gov_report \
  --frozen-selector-config "$FROZEN_SELECTOR" \
  --sample-id gov_report-fd287bbfd3bdbdc9 \
  --context-lengths 1K --pilot-samples 1 --max-new-tokens 32 \
  --warmup-runs 0 --timing-repeats 1 \
  --online-selector-mode reference_dflash \
  --online-selector-mode top1_packed_tree \
  --online-selector-mode global_fixed --selector-node-budget 16 \
  --enable-gold-checks --gold-max-rounds 32 \
  --output-dir outputs/phase2d1b-gate1-n16-fixed-gold
```

Gate 2, context and stump N=16:

```bash
python experiments/phase2d1b_online_selector.py \
  --target-model-path "$TARGET_MODEL_DIR" \
  --materialized-draft-path "$DRAFT_MODEL_DIR" \
  --longbench-root "$LONGBENCH_ROOT" --longbench-task gov_report \
  --frozen-selector-config "$FROZEN_SELECTOR" \
  --sample-id gov_report-fd287bbfd3bdbdc9 \
  --context-lengths 1K --pilot-samples 1 --max-new-tokens 32 \
  --warmup-runs 0 --timing-repeats 1 \
  --online-selector-mode context_rule \
  --online-selector-mode decision_stump --selector-node-budget 16 \
  --enable-gold-checks --gold-max-rounds 32 \
  --output-dir outputs/phase2d1b-gate2-n16-context-stump-gold
```

Gate 3, all N=32 selectors:

```bash
python experiments/phase2d1b_online_selector.py \
  --target-model-path "$TARGET_MODEL_DIR" \
  --materialized-draft-path "$DRAFT_MODEL_DIR" \
  --longbench-root "$LONGBENCH_ROOT" --longbench-task gov_report \
  --frozen-selector-config "$FROZEN_SELECTOR" \
  --sample-id gov_report-fd287bbfd3bdbdc9 \
  --context-lengths 1K --pilot-samples 1 --max-new-tokens 32 \
  --warmup-runs 0 --timing-repeats 1 \
  --online-selector-mode global_fixed \
  --online-selector-mode context_rule \
  --online-selector-mode decision_stump --selector-node-budget 32 \
  --enable-gold-checks --gold-max-rounds 32 \
  --output-dir outputs/phase2d1b-gate3-n32-gold
```

Gate 4, paired nested 1K/16K mechanics with exact-root alignment:

```bash
python experiments/phase2d1b_online_selector.py \
  --target-model-path "$TARGET_MODEL_DIR" \
  --materialized-draft-path "$DRAFT_MODEL_DIR" \
  --longbench-root "$LONGBENCH_ROOT" --longbench-task gov_report \
  --frozen-selector-config "$FROZEN_SELECTOR" \
  --phase2d1-dir "$PHASE2D1_DIR" \
  --capture-dir "$CAPTURE_GOV" --capture-dir "$CAPTURE_QAS" --capture-dir "$CAPTURE_HOT" \
  --sample-id gov_report-fd287bbfd3bdbdc9 \
  --context-lengths 1K,16K --pilot-samples 1 --max-new-tokens 32 \
  --warmup-runs 0 --timing-repeats 1 \
  --online-selector-mode reference_dflash \
  --online-selector-mode top1_packed_tree \
  --online-selector-mode global_fixed \
  --online-selector-mode context_rule \
  --online-selector-mode decision_stump \
  --selector-node-budget 16 --selector-node-budget 32 \
  --output-dir outputs/phase2d1b-gate4-mechanics-1k-16k
```

Run the acceptance checker after each generation gate by changing the directory:

```bash
python scripts/check_phase2d1b_acceptance.py \
  --output-dir outputs/phase2d1b-gate4-mechanics-1k-16k
```

## Gates 5–7: independent evidence

Gate 5 inventories only; tokenizer and JSONL are loaded, but no 8B model is
materialized:

```bash
for TASK in gov_report qasper hotpotqa; do
  python experiments/phase2d1b_online_selector.py \
    --target-model-path "$TARGET_MODEL_DIR" \
    --longbench-root "$LONGBENCH_ROOT" --longbench-task "$TASK" \
    --frozen-selector-config "$FROZEN_SELECTOR" \
    --exclude-selector-training-sources --inspect-workload-only \
    --context-lengths 1K,4K,8K,16K,32K --pilot-samples 8 \
    --output-dir "outputs/phase2d1b-gate5-inventory-$TASK"
done
```

Gate 6 runs one independent GovReport and HotpotQA source at 1K/8K/16K:

```bash
for TASK in gov_report hotpotqa; do
  INV="outputs/phase2d1b-gate5-inventory-$TASK/workload-inventory.json"
  AVAILABLE=$(python - "$INV" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
print(value["selector_source_inventory_by_prompt_length"]["16384"]["remaining_independent_sources"])
PY
)
  if test "$AVAILABLE" -lt 1; then
    echo "skip $TASK: no independent 16K source"
    continue
  fi
  python experiments/phase2d1b_online_selector.py \
    --target-model-path "$TARGET_MODEL_DIR" \
    --materialized-draft-path "$DRAFT_MODEL_DIR" \
    --longbench-root "$LONGBENCH_ROOT" --longbench-task "$TASK" \
    --frozen-selector-config "$FROZEN_SELECTOR" \
    --exclude-selector-training-sources \
    --context-lengths 1K,8K,16K --pilot-samples 1 --max-new-tokens 32 \
    --warmup-runs 0 --timing-repeats 1 \
    --online-selector-mode reference_dflash \
    --online-selector-mode top1_packed_tree \
    --online-selector-mode global_fixed \
    --online-selector-mode context_rule \
    --online-selector-mode decision_stump \
    --selector-node-budget 16 --selector-node-budget 32 \
    --output-dir "outputs/phase2d1b-gate6-independent-$TASK"
done
```

Gate 7 uses the sample counts recommended by Gate 5. GovReport 32K remains a
separate independent result only if its inventory count is nonzero. The shell
below removes an unavailable 32K bucket before choosing the matched independent
cohort; it never reintroduces a training source.

```bash
for SPEC in 'gov_report:1K,4K,8K,16K,32K' 'qasper:1K,4K,8K,16K' 'hotpotqa:1K,4K,8K,16K'; do
  TASK=${SPEC%%:*}
  LENGTHS=${SPEC#*:}
  INV="outputs/phase2d1b-gate5-inventory-$TASK/workload-inventory.json"
  read -r LENGTHS PILOT < <(python - "$INV" "$LENGTHS" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
requested = sys.argv[2].split(",")
counts = value["selector_source_inventory_by_prompt_length"]
numeric = {"1K": "1024", "4K": "4096", "8K": "8192", "16K": "16384", "32K": "32768"}
kept = [name for name in requested if counts[numeric[name]]["remaining_independent_sources"] > 0]
pilot = min(counts[numeric[name]]["remaining_independent_sources"] for name in kept) if kept else 0
print(",".join(kept), min(pilot, 8))
PY
)
  test "$PILOT" -gt 0
  python experiments/phase2d1b_online_selector.py \
    --target-model-path "$TARGET_MODEL_DIR" \
    --materialized-draft-path "$DRAFT_MODEL_DIR" \
    --longbench-root "$LONGBENCH_ROOT" --longbench-task "$TASK" \
    --frozen-selector-config "$FROZEN_SELECTOR" \
    --exclude-selector-training-sources \
    --context-lengths "$LENGTHS" --pilot-samples "$PILOT" \
    --max-new-tokens 128 --warmup-runs 1 --timing-repeats 3 \
    --online-selector-mode reference_dflash \
    --online-selector-mode top1_packed_tree \
    --online-selector-mode global_fixed \
    --online-selector-mode context_rule \
    --online-selector-mode decision_stump \
    --selector-node-budget 16 --selector-node-budget 32 \
    --output-dir "outputs/phase2d1b-gate7-independent-$TASK"
  python scripts/check_phase2d1b_acceptance.py \
    --output-dir "outputs/phase2d1b-gate7-independent-$TASK"
done
```

## Final artifact and target-blind audits

```bash
OUT=outputs/phase2d1b-gate7-independent-hotpotqa

python - "$OUT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
required = json.load(open(root / "manifest.json"))["artifacts"]
missing = [name for name in required if not (root / name).is_file()]
assert not missing, missing
comparison = json.load(open(root / "selector-comparison.json"))
correctness = json.load(open(root / "correctness-summary.json"))
print("decision:", comparison["decision"])
print("correctness:", correctness["all_checks_passed"])
PY

python - "$OUT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for line in (root / "selector-decisions.jsonl").read_text().splitlines():
    row = json.loads(line)
    assert row["decision_made_before_target_verification"]
    assert not row["target_dependent_runtime_input_used"]
    assert row["target_forward_count_added"] == 0
print("selector target-blind audit: pass")
PY

python - "$OUT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
samples = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
assert all(row.get("control_profile_output_ids_equal", True) for row in samples)
print("control/profile token check: pass")
PY
```

The H100 validation must establish policy-node realization, gold/KV/traversal
correctness, control/profile equality, BF16 threshold proximity, selector sync
cost, independent source availability, actual acceptance/commit gains, round
count, end-to-end wall latency, per-token latency, throughput, and memory. Only
those real results can decide whether a selector is adopted or Phase 3 moves
directly to candidate availability under sparse context access.
