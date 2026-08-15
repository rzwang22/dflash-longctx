# Phase 1B.1: LongBench workload adaptation and inventory

Phase 1B.1 generalizes the Phase 1B workload layer from GovReport to one
LongBench task per run: `gov_report`, `qasper`, or `hotpotqa`. It does not change
DFlash generation, draft attention, KV cache behavior, sampling, target
verification, or model weights. The instrumented custom-code tree is unchanged
from Phase 1B, so an existing Phase 1B materialization does not need to be
rebuilt. Source-provenance checks still apply when generation starts.

## Dataset schema and prompt contract

The preferred local layout is:

```text
/mnt/sevenT/wrz_data/datasets/LongBench/
├── gov_report.jsonl
├── qasper.jsonl
└── hotpotqa.jsonl
```

Pass that directory with `--longbench-root` and select exactly one task with
`--longbench-task`. A single file may instead be passed with `--dataset-jsonl`.
LongBench's default field mapping is `context` for source text, `input` for the
Qasper/HotpotQA question, `answers` for reference-only metadata, and `_id` for
the source ID. Override these with `--context-field`, `--query-field`,
`--answers-field`, and `--id-field`. The old `--govreport-jsonl` and
`--text-field` names remain accepted for GovReport only.

The three prompt templates are versioned independently:

- GovReport asks for a report summary.
- Qasper asks the `input` question using only the paper context.
- HotpotQA asks the `input` question using only the multi-document context.

`answers` is copied only to reference metadata and is never interpolated into
the prompt. For a source sample, the instruction and query stay byte-identical
at every requested length; only a nested prefix of `context` grows. The
manifest records the template name, version, canonical template text, and its
SHA-256 hash. `actual_prompt_length` is measured only after applying the full
Qwen3 chat template with `add_generation_prompt=True` and thinking disabled.

Raw LongBench files and run outputs are ignored by Git and must remain outside
the repository.

## Inventory before model loading

`--inspect-workload-only` loads the target tokenizer and the selected local
JSONL, but it does not require CUDA and does not instantiate either the target
or draft model. It writes `workload-inventory.json`, `environment.json`, and
`terminal.log`. The terminal table and JSON report:

- tokenizer-derived eligible sample counts at each requested prompt length;
- document-token min, median, P90, and max;
- all stable sample IDs and original source IDs that reach the largest requested
  length;
- a recommended matched pilot length set and sample count;
- independent per-length sample-count recommendations.

A sample is eligible only when its fixed instruction/query fits below the
requested length and its complete chat-templated source reaches that length.
Selected prefixes are then constructed and shortfall/nesting constraints are
validated, still before any model load.

Run all three inventories on the H100 server before any generation:

```bash
cd "$DFLASH_REPO"
source .venv/bin/activate
export TARGET_MODEL_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-b968826d
export LONGBENCH_ROOT=/mnt/sevenT/wrz_data/datasets/LongBench

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python experiments/phase1b_longctx.py \
  --inspect-workload-only \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path "$TARGET_MODEL_DIR" \
  --longbench-root "$LONGBENCH_ROOT" \
  --longbench-task gov_report \
  --context-lengths 1K,4K,8K,16K,32K \
  --pilot-samples 8 \
  --output-dir outputs/phase1b1-inventory-gov-report

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python experiments/phase1b_longctx.py \
  --inspect-workload-only \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path "$TARGET_MODEL_DIR" \
  --longbench-root "$LONGBENCH_ROOT" \
  --longbench-task qasper \
  --context-lengths 1K,4K,8K,16K,32K \
  --pilot-samples 8 \
  --output-dir outputs/phase1b1-inventory-qasper

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python experiments/phase1b_longctx.py \
  --inspect-workload-only \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path "$TARGET_MODEL_DIR" \
  --longbench-root "$LONGBENCH_ROOT" \
  --longbench-task hotpotqa \
  --context-lengths 1K,4K,8K,16K,32K \
  --pilot-samples 8 \
  --output-dir outputs/phase1b1-inventory-hotpotqa
```

The automatic recommendation is available at
`recommendations.matched.{context_lengths,sample_count,sample_ids}`. The
per-length alternative is at
`recommendations.per_length.sample_count_by_length`. Do not infer Qasper or
HotpotQA counts from GovReport; use each task's inventory result.

## Cohort policy

`--sample-policy matched` is the default. It selects the requested number of
documents that all reach the largest requested prompt length, then uses those
same IDs and nested prefixes at every shorter length. If eight are requested
and only three qualify, preflight raises an error before CUDA initialization or
8B model loading. It never reduces the cohort silently, concatenates samples,
repeats text, or adds filler.

`--sample-policy per-length` independently selects up to `--pilot-samples` at
each length. Its manifest sets `paired_context_length_comparison` to false and
records actual counts and IDs for every length. Such results are explicitly
unpaired and must not be described as a strict paired context-length
comparison.

## GovReport matched-three five-length run

The previously observed GovReport inventory has three samples eligible at 32K.
After checking the new inventory, run the matched cohort as follows:

```bash
export DFLASH_INSTRUMENTED_DIR=/path/outside/git/dflash-instrumented-phase1b

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
python experiments/phase1b_longctx.py \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --target-model-path "$TARGET_MODEL_DIR" \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --draft-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
  --materialized-draft-path "$DFLASH_INSTRUMENTED_DIR" \
  --longbench-root "$LONGBENCH_ROOT" \
  --longbench-task gov_report \
  --sample-policy matched \
  --context-lengths 1K,4K,8K,16K,32K \
  --pilot-samples 3 \
  --timing-repeats 3 \
  --warmup-runs 1 \
  --max-new-tokens 128 \
  --device cuda:0 \
  --dtype bfloat16 \
  --output-dir outputs/phase1b1-gov-report-matched-3
```

Generation mode repeats inventory and cohort validation before `_require_cuda`
or either model constructor. It then writes the usual Phase 1B artifacts plus
`workload-inventory.json`:

```text
manifest.json
workload-inventory.json
samples.jsonl
rounds.jsonl
summary.json
summary.csv
environment.json
source-provenance.json
terminal.log
```

## Accounting and CSV additions

Accepted draft tokens exclude the round seed. Committed tokens include the seed
and therefore equal accepted draft tokens plus one before terminal return-budget
truncation. Raw all-round counts remain available, while the primary acceptance
statistics exclude terminal or truncated rounds. Terminal counts and truncated
returned tokens remain explicit rather than being discarded.

`summary.csv` now also includes draft, verification, and round CUDA medians;
draft/round and verification/round medians; mean non-terminal accepted-draft
and committed-token counts; executed round and acceptance-eligible round counts;
and truncated-token count. Existing effective-returned-token, acceptance, and
combined-attention fields remain unchanged. Component latency remains sourced
from the profiling path; formal end-to-end latency remains sourced from the
instrumented-default control path.

## H100-only checks still required

- Run the three inventories with the actual Qwen3 tokenizer and preserve their
  reported eligible counts, IDs, and recommendations.
- Confirm the existing materialized Phase 1B custom-code provenance still
  matches the repository-controlled source hashes.
- Run the GovReport matched-three cohort and token-exact control/profile checks.
- Choose Qasper and HotpotQA pilot lengths/sample counts from their own inventory
  outputs, then validate token equality, position limits, CUDA timing, memory,
  and repeat variability on H100.
- Use Nsight/NVTX if the concrete SDPA kernel at the observed shapes must be
  confirmed.

No Mac test or inventory-only run makes a GPU performance claim.
