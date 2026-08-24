# Phase 2D.2A-H1.1 budget-utilization and mechanism audit

H1.1 is a repository-side analysis of the completed H1 known-sample matrix. It
does not rerun a model, alter online execution, choose a refinement family, or
authorize H2. The original H1 decision remains `insufficient_evidence`.

## Direct work accounting

The audit reads every round's raw
`refinement-rounds.jsonl -> phase2d2.actual_verified_node_count`. It separately
records `selected_tree_node_count` and requires the existing Phase 2D.2
selected/verified-node invariant. Configured `node_budget` is only the capacity
denominator:

```text
node_budget_utilization_rate =
    sum(actual_verified_node_count) / (round_count * configured_node_budget)
```

The configured budget is never substituted for actual work. Each of the 16
cells reports the raw per-round counts, total, mean, median, minimum, maximum,
selected-node histogram, and utilization. The raw total is also checked against
the already accepted H1 aggregate's `verified_target_node_count`.

A base/refinement pair is `same_actual_verification_work` only when its direct
per-round summaries and normalized count distributions agree. It is
`budget_cap_expansion` when the refined path directly records more verified
nodes per round. The N16/N32 label alone cannot determine either regime.

For a nonzero work delta, H1.1 additionally reports:

```text
marginal accepted yield =
    delta accepted draft tokens per round / delta verified nodes per round

marginal committed yield =
    delta committed tokens per round / delta verified nodes per round
```

Both values are explicitly unavailable when the verified-node delta is zero.

## Mechanism interpretation

For each of the 12 refined pairs, H1.1 carries forward base, conditional,
merged, and selected-tree coverage; refined-only target recovery; target-token
rank improvement; and correct-prefix survival depth. Candidate-label authority
remains the aligned Original-DFlash continuation. Unavailable candidate rows
are recorded by reason and excluded from denominators; q1 is not a label
source.

The post-hoc mechanism labels are descriptive:

- `new_candidate_availability_signal` requires a positive refined-only target
  recovery count;
- `reranking_or_tree_reallocation_signal` requires positive selected-tree
  coverage gain and zero refined-only recovery;
- otherwise the row is `no_observed_candidate_mechanism_gain`.

These labels do not claim causal exclusivity. In particular, zero refined-only
recovery is a mechanism result, not a correctness or qualification failure.

## H1 reinterpretation and costs

The preregistered four-cell cap-based `mixed_known_sample` classifications are
preserved unchanged. H1.1 adds a second
`primary_same_work_classification`, using only directly identified same-work
pairs. `consistent_positive_same_work` requires positive coverage gain and
positive accepted-tokens-per-verified-node gain in every valid same-work pair;
no significance threshold is introduced.

Cost output keeps dependency modeling (`conditional_remask`) separate from the
correctness-preserving fixed-shape verifier. It recomputes each share and their
non-overlapping combined share from accepted H1 artifacts, and reports which
component is larger. A lightweight dependency model is not claimed to remove
fixed-shape verifier cost.

Formal latency values and deltas are preserved, but H1.1 labels them only as
`nominal_latency_change`. One physical sample and three formal repeats cannot
support a robust speedup claim.

## Conditional H2 design artifact

Only if direct evidence shows all N16 refined pairs are same-work and all N32
pairs are budget expansions does H1.1 emit
`phase2d2a-h2-preregistered-design.json`. It is explicitly non-executing and
retains all three frozen families:

- `conditional_rerank-d1-p1-r2-w0.5`;
- `union-d2-p2-r2-w0.5`;
- `union-d4-p2-r4-w0.5`.

The design uses N16 as the primary same-work mechanism evaluation and N32 as a
secondary budget-expansion ablation. It preserves
`always_fixed_shape_recheck` and Original-DFlash candidate labels, selects no
winner, and does not authorize H2 execution.

## Analyze the existing H100 result

These commands are non-destructive, require no GPU/model loading, and do not
rerun any H1 cell. They intentionally avoid `set -e`, `exit 1`, and
shell-killing assertions.

```bash
cd ~/data/dflash-longctx
git fetch origin
git checkout codex/phase2d2a-robust-conditional-remask
git pull --ff-only origin codex/phase2d2a-robust-conditional-remask

source .venv/bin/activate
export H1_ROOT=outputs/phase2d2a-h1-known-d3c21f67310e
export H11_OUT="$H1_ROOT/h1.1-budget-mechanism-audit-$(date +%Y%m%d-%H%M%S)"

python scripts/audit_phase2d2a_h1_budget_mechanism.py \
  --h1-root "$H1_ROOT" \
  --output-dir "$H11_OUT"
echo "H11_AUDIT_RC=$?"

python scripts/check_phase2d2a_h11_acceptance.py "$H11_OUT"
echo "H11_CHECKER_RC=$?"

python - "$H11_OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
budget = json.loads(
    (root / "phase2d2a-h1-budget-utilization.json").read_text()
)
mechanism = json.loads(
    (root / "phase2d2a-h1-mechanism-decomposition.json").read_text()
)
view = json.loads(
    (root / "phase2d2a-h1-reinterpretation.json").read_text()
)

print("direct_node_utilization")
for row in budget["rows"]:
    stats = row["actual_verified_nodes_per_round"]
    print(
        row["configuration_family"],
        row["nominal_prompt_length"],
        row["configured_node_budget"],
        row["round_count"],
        row["total_actual_verified_node_count"],
        stats["mean"], stats["median"], stats["min"], stats["max"],
        row["selected_tree_node_count_distribution"],
        row["node_budget_utilization_rate"],
    )

print("regime_and_marginal_yield")
for row in budget["pair_regimes"]:
    print(
        row["refined_family"], row["nominal_prompt_length"], row["node_budget"],
        row["regime"], row["delta_actual_verified_nodes_per_round"],
        row["marginal_accepted_tokens_per_additional_verified_node"],
        row["marginal_committed_tokens_per_additional_verified_node"],
    )

print("mechanism_decomposition")
for row in mechanism["rows"]:
    print(
        row["refined_family"], row["nominal_prompt_length"], row["node_budget"],
        row["base_candidate_coverage_rate"],
        row["conditional_candidate_coverage_rate"],
        row["merged_candidate_coverage_rate"],
        row["selected_tree_coverage_rate"],
        row["refined_only_target_recovery_count"],
        row["mechanism_signal"],
    )

print("classifications")
for row in view["configuration_rows"]:
    print(
        row["refined_family"],
        row["original_cap_based_h1_classification"],
        row["primary_same_work_classification"],
    )
print("cost_decomposition", view["cost_decomposition"])
print("final_phase2d2a_decision", view["final_phase2d2a_decision"])
print(
    "h2_preregistered_design_present",
    (root / "phase2d2a-h2-preregistered-design.json").is_file(),
)
PY
```

Stop after the H1.1 checker and inspection. Do not execute H2 or begin Phase
2D.2B/2D.3.

## Output artifacts

The audit writes a new, non-overwriting output directory containing:

- `phase2d2a-h1-budget-utilization.json` and `.csv`;
- `phase2d2a-h1-mechanism-decomposition.json` and `.csv`;
- `phase2d2a-h1-reinterpretation.json` and `.csv`;
- conditionally, `phase2d2a-h2-preregistered-design.json`.

No online source or `third_party/` file changes, so no materialization is
required.
