# Phase 2D.2A-H1 known-sample paired benefit/cost evaluation

Phase 2D.2A-G is complete. H1 is a mechanism and cost study on one physical
GovReport sample; it is not held-out cross-task evidence and cannot select or
re-freeze a refinement family. The final Phase 2D.2A decision remains
`insufficient_evidence` until H2 supplies independent-task evidence.

## Frozen matrix and pairing

The matrix contains exactly 16 cells:

- sample: `gov_report-fd287bbfd3bdbdc9`;
- prompt lengths: 1K and 16K;
- root-inclusive target-node budgets: N16 and N32;
- control: `base_only`;
- refinements: `conditional_rerank-d1-p1-r2-w0.5`,
  `union-d2-p2-r2-w0.5`, and `union-d4-p2-r4-w0.5`;
- selector: `top1_packed_tree`;
- verifier: `always_fixed_shape_recheck`;
- `max_new_tokens=128`, one warmup, three formal timing repeats, external
  batch one, BF16, CUDA device 0.

Every refinement is paired only with `base_only` at the same
`(task, sample_id, context_length, node_budget)`. A different N, context
prefix, physical source, materialization, model revision, selector hash, or
timing protocol is rejected rather than treated as a pair.

Original DFlash actual speculative execution is the hard token-exact oracle.
Target-only remains auxiliary. Candidate-quality labels come only from the
aligned Original-DFlash absolute continuation; q1, target-only, packed logits,
and fixed-shape logits are not label authorities and never enter runtime
selection.

## Timer audit

The existing timers have the following source-confirmed scopes:

| Field | Scope | Nesting/interpretation |
|---|---|---|
| `conditional_remask_cuda_elapsed_ms` | Conditional input/tensor preparation, batched temporary cache construction, the second DFlash draft forward, LM head, and conditional-candidate extraction. It excludes the earlier round-entry cache snapshot, prefix selection, refined-tree construction, and target verification. | Nested inside `total_draft_cuda_elapsed_ms`. |
| `total_draft_cuda_elapsed_ms` | Enclosing draft-side interval from the optional refinement-cache snapshot through the initial draft pass, candidate/selector/refinement work, tree packing, and mask construction. | Contains the conditional timer; never add the two. |
| `packed_verification_cuda_elapsed_ms` | The packed target-model forward over the selected tree. | Nested inside total verification and round time. |
| `fallback_cuda_elapsed_ms` | For H1 always-fixed runs, the fixed-shape cache clone and q_len=16 selected-path target recheck. It excludes the earlier round-entry stabilization snapshot and the later authoritative KV replacement timer. | Nested inside total verification and round time. It is robust-verification cost, not conditional-remask cost. |
| `round_cuda_elapsed_ms` | Enclosing diagnostic round interval containing draft construction, packed verification, traversal, fixed-shape recheck, and commit work. | Denominator for component shares; enclosing components are not summed into it. |
| `formal_target_forward_count` | Logical target-forward count, `1 + fallback_forward_count`; always-fixed normally records two target forwards per round. | A count, not a latency timer. |

Profile component totals come from the separate profiling generation.
Per-round medians come from the separate diagnostic generation. Formal
end-to-end latency comes only from the unprofiled formal repeats.

`conditional_remask + fixed_shape_recheck` is a valid non-overlapping
additional-work sum. `conditional_remask / initial_parallel_draft` is not
reported as an exact full-pass ratio: although the intervals do not overlap,
the conditional timer includes input/cache/candidate work that the initial
draft-forward timer does not. Both raw costs remain visible, with the ratio
marked unavailable instead of implying equivalent scopes.
The cost artifact also reports the separately timed round-entry snapshot and
KV replacement, plus their non-overlapping sum with the fixed-shape recheck;
these adjacent costs are not silently folded into `fallback_cuda_elapsed_ms`.

## Aggregation and interpretation

The aggregator reports candidate-quality availability before computing any
rate. Unavailable terminal-boundary rounds remain counted by reason but their
zero placeholders do not enter candidate denominators. A cell with no labelled
positions has insufficient candidate-benefit evidence; no threshold is
invented for H1.

A correctness-invalid cell is listed in the matrix but contributes no paired
benefit or cost conclusion. The hard cell conditions are:

- Original-DFlash/tree token exactness;
- formal repeat stability;
- profile/formal and diagnostic/formal output equality;
- actual-execution correctness;
- structural, traversal, cache, and fixed-shape commit invariants accepted by
  the existing Phase 2D.2A checker.

For each refinement, the descriptive known-sample classification is:

- `consistent_positive_known_sample`: coverage gain and
  accepted-draft-tokens/verified-node gain are both positive in all four valid
  context/N pairs;
- `mixed_known_sample`: the joint signal is positive in some but not all valid
  pairs;
- `no_positive_known_sample_signal`: no valid pair has both gains positive;
- `invalid_due_to_correctness`: at least one of the four pairs is correctness
  invalid;
- `insufficient_candidate_label_evidence`: at least one otherwise-valid pair
  has no labelled candidate positions.

Signs use exact measured direction (`+`, `0`, `-`; `faster`, `neutral`,
`slower`) without an H1 significance threshold. These classifications do not
choose a winner. The legacy margin `guard_false_negative_count` is preserved in
source artifacts but is not an H1 gate because H1 always uses the always-fixed
trajectory.

## H100 execution

The following commands create 16 distinct cell directories, record runner and
external-checker return codes without terminating the shell, aggregate the
matrix, run the H1 checker, and stop. They reuse the validated Phase F
materialization; no rematerialization is required.

```bash
cd ~/data/dflash-longctx
git fetch origin
git checkout codex/phase2d2a-robust-conditional-remask
git pull --ff-only origin codex/phase2d2a-robust-conditional-remask

source .venv/bin/activate
export TARGET_MODEL_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-b968826d
export DRAFT_MODEL_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16-phase2d2a-instrumented-33e7159c3d50
export LONGBENCH_ROOT=/mnt/sevenT/wrz_data/datasets/LongBench/data
export FROZEN_SELECTOR=outputs/phase2d1b-frozen-selector/frozen-selector.json
export FROZEN_REFINEMENT=outputs/phase2d2a-frozen-refinement-66ce72888b34.json
export H1_ROOT=outputs/phase2d2a-h1-known-$(date +%Y%m%d-%H%M%S)
export H1_RUN_ROOT="$H1_ROOT/runs"
export H1_AGGREGATE_OUT="$H1_ROOT/aggregate"

run_h1_cell() {
  family="$1"
  context="$2"
  budget="$3"
  output_dir="$H1_RUN_ROOT/${family}-${context}-n${budget}"
  base_dir="$H1_RUN_ROOT/base_only-${context}-n${budget}"

  refinement_args=()
  if [ "$family" = "base_only" ]; then
    refinement_args+=(--candidate-merge-mode base_only)
  else
    refinement_args+=(
      --frozen-refinement-config "$FROZEN_REFINEMENT"
      --frozen-refinement-config-id "$family"
      --base-result-dir "$base_dir"
    )
  fi

  python experiments/phase2d2_robust_conditional_refinement.py \
    --target-model Qwen/Qwen3-8B \
    --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
    --target-model-path "$TARGET_MODEL_DIR" \
    --draft-model z-lab/Qwen3-8B-DFlash-b16 \
    --draft-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
    --materialized-draft-path "$DRAFT_MODEL_DIR" \
    --longbench-root "$LONGBENCH_ROOT" \
    --longbench-task gov_report \
    --sample-policy matched \
    --sample-id gov_report-fd287bbfd3bdbdc9 \
    --context-lengths "$context" \
    --pilot-samples 1 \
    --max-new-tokens 128 \
    --warmup-runs 1 \
    --timing-repeats 3 \
    --batch-size 1 \
    --temperature 0 \
    --block-size 16 \
    --dtype bfloat16 \
    --device cuda:0 \
    --frozen-selector-config "$FROZEN_SELECTOR" \
    --online-selector-mode top1_packed_tree \
    --node-budget "$budget" \
    --verification-mode always_fixed_shape_recheck \
    --evidence-role mechanism_analysis \
    "${refinement_args[@]}" \
    --output-dir "$output_dir"
  runner_rc=$?

  python scripts/check_phase2d2_acceptance.py "$output_dir"
  checker_rc=$?

  python - "$output_dir" "$runner_rc" "$checker_rc" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
payload = {
    "runner_return_code": int(sys.argv[2]),
    "external_checker_return_code": int(sys.argv[3]),
}
(root / "h1-external-status.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(root, payload)
PY
}

for context in 1K 16K; do
  run_h1_cell base_only "$context" 16
done
for context in 1K 16K; do
  run_h1_cell conditional_rerank-d1-p1-r2-w0.5 "$context" 16
done
for context in 1K 16K; do
  run_h1_cell union-d2-p2-r2-w0.5 "$context" 16
done
for context in 1K 16K; do
  run_h1_cell union-d4-p2-r4-w0.5 "$context" 16
done

for context in 1K 16K; do
  run_h1_cell base_only "$context" 32
done
for context in 1K 16K; do
  run_h1_cell conditional_rerank-d1-p1-r2-w0.5 "$context" 32
done
for context in 1K 16K; do
  run_h1_cell union-d2-p2-r2-w0.5 "$context" 32
done
for context in 1K 16K; do
  run_h1_cell union-d4-p2-r4-w0.5 "$context" 32
done

python scripts/aggregate_phase2d2a_h_known.py \
  --result-root "$H1_RUN_ROOT" \
  --output-dir "$H1_AGGREGATE_OUT"
echo "H1_AGGREGATOR_RC=$?"

python scripts/check_phase2d2a_h_known_acceptance.py "$H1_AGGREGATE_OUT"
echo "H1_CHECKER_RC=$?"

python - "$H1_AGGREGATE_OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "phase2d2a-h-known-manifest.json").read_text())
summary = json.loads((root / "phase2d2a-h-known-config-summary.json").read_text())
decision = json.loads((root / "phase2d2a-h-known-decision.json").read_text())
print("matrix_complete:", manifest["matrix_complete"])
print("cell_count:", manifest["matrix_observed_cell_count"])
print("invalid_cell_count:", manifest["invalid_cell_count"])
print("known_sample_classifications:")
for row in summary["rows"]:
    print(" ", row["refined_family"], row["known_sample_stability_classification"])
print("final_phase2d2a_decision:", decision["final_phase2d2a_decision"])
PY
```

Stop after the H1 checker and inspection. Do not run Qasper, HotpotQA,
additional GovReport samples, H2, Phase 2D.2B, or Phase 2D.3.

## Expected aggregate artifacts

Under `$H1_AGGREGATE_OUT`:

- `phase2d2a-h-known-manifest.json`;
- `phase2d2a-h-known-paired-comparison.json` and `.csv`;
- `phase2d2a-h-known-config-summary.json` and `.csv`;
- `phase2d2a-h-known-cost-summary.json` and `.csv`;
- `phase2d2a-h-known-decision.json`.
