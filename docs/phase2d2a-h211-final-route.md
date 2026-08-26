# Phase 2D.2A-H2.1.1 final route and evidence reconciliation

H2.1.1 is a repository-side, no-model-forward audit of the accepted H2.1
result. It revalidates the immutable H2.0 freeze and the complete H2.1 output,
then derives its scientific counts from the accepted artifacts. It never
rewrites H2.0 or H2.1.

## Evidence levels and frozen route

The audit keeps three evidence levels separate:

1. The formal primary evidence is the preregistered cross-task joint
   same-work classification. No frozen family produced a stable signal;
   conditional-rerank, union-d2, and union-d4 remain mixed at 3/4, 3/4, and
   2/4 independent strata, respectively.
2. The secondary efficiency evidence retains per-stratum accepted/verified
   signs, sample-level sign heterogeneity, and pooled accepted/verified counts
   and gain. Conditional-rerank and union-d2 were positive in all four
   independent task/context strata. This does not alter the joint primary
   criterion.
3. Mechanism and cost interpretation retains coverage signs, mechanism-signal
   distributions, refined-only recovery, GovReport replication, and the
   verifier-versus-dependency cost hierarchy. New-candidate availability is
   not declared dominant and dependency information is not declared useless.

The formal route is therefore:

- do not authorize a Phase 2D.2B lightweight adapter;
- do not select a refinement winner;
- do not increase dependency-model complexity as the primary path;
- permit H2.2 only as secondary budget-expansion characterization;
- never let H2.2 rescue or override the accepted H2.1 primary result.

Budget-constrained candidate allocation is recorded as a project route
recommendation, not an experimentally proven conclusion. Sparse-KV and
long-context traffic reduction remain a later system-level direction. H2.1.1
makes no speedup claim.

## Inputs and artifacts

The commands require the exact H2.0 freeze SHA and the accepted H2.1 output
root. The audit hashes every H2.1 artifact used by its checker and writes:

- `phase2d2a-h211-final-route-audit.json`;
- `phase2d2a-h211-final-route-audit.csv`;
- `phase2d2a-h211-route-manifest.json`.

The checker rebuilds the audit from immutable inputs and compares canonical
content. Any changed count, classification, source hash, route authorization,
or model-forward claim fails closed.

## H100 no-model commands

```bash
cd ~/data/dflash-longctx
git switch codex/phase2d2a-robust-conditional-remask
git pull --ff-only origin codex/phase2d2a-robust-conditional-remask
source .venv/bin/activate

export FREEZE_DIR="$PWD/outputs/phase2d2a-h2-heldout-freeze-82919fa2978d"
export FREEZE_SHA="d5fdc816b183c19bd5683fb89a3c5e1ace641a15d2ac68da7b6dc0e56a678856"
export H21_OUTPUT_ROOT="$PWD/outputs/phase2d2a-h21-primary-n16-2e2bc7c121d6"
export H211_OUTPUT_DIR="$PWD/outputs/phase2d2a-h211-final-route-$(git rev-parse --short=12 HEAD)"

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
```

These commands load no target or draft model and execute no model forward.
