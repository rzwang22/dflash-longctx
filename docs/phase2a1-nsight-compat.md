# Phase 2A.1: Nsight compatibility and paired length comparison

Phase 2A.1 is an analysis and workload-selection patch on top of Phase 2A. It
does not alter DFlash generation, target verification, attention, KV caches,
sampling, model weights, or output tokens. No file under `third_party/` changes,
so a Phase 2A materialized draft remains valid and does not need to be rebuilt.

## Nsight export compatibility

`scripts/phase2a_nsight.py export` records `nsys --version` and
`nsys stats --help-reports`, then invokes each report separately.
`nvtx_kern_sum` is required. `nvtx_gpu_proj_sum` is optional:

- supported and successfully parsed: `analysis_mode=full`;
- absent from the report inventory: `analysis_mode=kernel_only` and
  `projection_report_status=unavailable`;
- advertised but failing: `analysis_mode=kernel_only` and
  `projection_report_status=failed`.

The optional failure never discards a valid kernel report. `nvtx_sum` is not
used as a substitute for `nvtx_gpu_proj_sum`. Missing projection values are not
filled with zero. The header-only `nsight-nvtx-summary.csv` in kernel-only mode
means “not measured,” not zero GPU time.

Every successful parse writes:

```text
nsight-kernel-summary.csv
nsight-nvtx-summary.csv
verification-kernel-breakdown.json
nsight-export-manifest.json
```

The manifest records available, requested, required, optional, generated,
reused, succeeded, unavailable, and failed reports; commands and return codes;
checked paths; the Nsight version; and the actual `source_reports`. Kernel rows
retain raw names and name-based classification rules. Nested NVTX kernel totals
must not be added as disjoint time because one kernel can be associated with
multiple enclosing ranges. See NVIDIA's
[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/).

Existing data can be reused without profiling a model:

```bash
# Existing .nsys-rep
python scripts/phase2a_nsight.py export \
  --nsys-rep path/to/trace.nsys-rep --output-dir path/to/parsed

# Existing exported SQLite
python scripts/phase2a_nsight.py export \
  --sqlite path/to/trace.sqlite --output-dir path/to/parsed

# Existing required CSV, with optional projection CSV omitted
python scripts/phase2a_nsight.py parse \
  --nvtx-kernel-csv path/to/nsight_nvtx_kern_sum.csv \
  --output-dir path/to/parsed
```

If `nvtx_kern_sum` cannot be found or generated, the error includes the Nsight
version, probed reports, and every checked output path.

## Fixed LongBench sample

Phase 2A accepts `--sample-id`. It selects only that stable LongBench sample,
requires `--sample-policy matched`, validates existence and every requested
length before CUDA or either 8B model is loaded, and reports the full templated
maximum prompt length on failure. The sample ID and source ID are saved in the
run configuration, workload inventory, cohort, manifest sample list, and sample
records. Deterministic construction uses nested prefixes of the same source
document at all lengths.

Inventory terminal output shows at most ten maximum-length eligible IDs by
default, with total/shown/omitted counts. The JSON retains the complete list.
Use `--show-all-eligible-sample-ids` to print every ID.

## Re-export the existing 1K, 16K, and 32K traces

These commands do not load a tokenizer or model and do not require
materialization:

```bash
cd ~/data/dflash-longctx
source .venv/bin/activate

for label in 1k 16k 32k; do
  python scripts/phase2a_nsight.py export \
    --nsys-rep "outputs/phase2a-nsight/${label}/trace.nsys-rep" \
    --output-dir "outputs/phase2a-nsight/${label}/parsed"
done
```

On Nsight Systems 2023.1.2, each directory is expected to report
`analysis_mode=kernel_only`, with `nvtx_kern_sum` in `source_reports` and
`nvtx_gpu_proj_sum` in `unavailable_reports`.

## Compare existing verification summaries

The comparison script imports no model framework. It automatically reads each
summary's sibling `samples.jsonl` and `verification-profile-rounds.jsonl`:

```bash
python scripts/phase2a_compare_verification.py \
  --summary \
    outputs/phase2a-nsight/1k/run/verification-profile-summary.json \
    outputs/phase2a-nsight/16k/run/verification-profile-summary.json \
    outputs/phase2a-nsight/32k/run/verification-profile-summary.json \
  --output-dir outputs/phase2a-nsight/verification-length-comparison
```

It writes `verification-length-comparison.json` and `.csv`. Generation-total,
per-round, per-returned-token, and per-committed-token metrics are separate.
Round counts come only from actual round artifacts. Phase 2A artifacts produced
before this patch did not save exact terminal/commit accounting in verification
round rows; for those fields the comparison writes `null` plus a warning rather
than inferring values. New runs save the exact accounting fields.

Different source IDs produce `paired_comparison=false` and a pilot-trend
warning. Identical source IDs at every length produce `paired_comparison=true`.
Whole self-attention-module time includes QKV projections, the attention/SDPA
path, and output projection; it is not pure SDPA or pure KV-access latency.

## Formal paired H100 traces

Use the already materialized Phase 2A draft. The fixed sample below is known to
reach 32K in the current GovReport inventory:

```bash
cd ~/data/dflash-longctx
source .venv/bin/activate

export TARGET_MODEL_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-b968826d
export DFLASH_PHASE2A_DIR=/mnt/sevenT/wrz_data/models/Qwen3-8B-DFlash-b16-phase2a-instrumented
export LONGBENCH_ROOT=/mnt/sevenT/wrz_data/datasets/LongBench
export PAIRED_SAMPLE_ID=gov_report-fd287bbfd3bdbdc9

for label in 1k 16k 32k; do
  case "$label" in
    1k) length=1K ;;
    16k) length=16K ;;
    32k) length=32K ;;
  esac
  mkdir -p "outputs/phase2a-nsight-paired/${label}"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0 \
  nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true \
    -o "outputs/phase2a-nsight-paired/${label}/trace" \
    python experiments/phase2a_profile.py \
      --run-mode verification-profile \
      --target-model Qwen/Qwen3-8B \
      --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
      --target-model-path "$TARGET_MODEL_DIR" \
      --draft-model z-lab/Qwen3-8B-DFlash-b16 \
      --draft-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
      --materialized-draft-path "$DFLASH_PHASE2A_DIR" \
      --longbench-root "$LONGBENCH_ROOT" --longbench-task gov_report \
      --sample-policy matched --sample-id "$PAIRED_SAMPLE_ID" \
      --context-lengths "$length" --pilot-samples 1 \
      --timing-repeats 1 --warmup-runs 1 --max-new-tokens 32 \
      --device cuda:0 --dtype bfloat16 \
      --output-dir "outputs/phase2a-nsight-paired/${label}/run"
done
```

Then export and compare without rerunning the models:

```bash
for label in 1k 16k 32k; do
  python scripts/phase2a_nsight.py export \
    --nsys-rep "outputs/phase2a-nsight-paired/${label}/trace.nsys-rep" \
    --output-dir "outputs/phase2a-nsight-paired/${label}/parsed"
done

python scripts/phase2a_compare_verification.py \
  --summary \
    outputs/phase2a-nsight-paired/1k/run/verification-profile-summary.json \
    outputs/phase2a-nsight-paired/16k/run/verification-profile-summary.json \
    outputs/phase2a-nsight-paired/32k/run/verification-profile-summary.json \
  --output-dir outputs/phase2a-nsight-paired/verification-length-comparison
```

H100 validation must still confirm the 2023.1 report inventory and filenames,
kernel classifications/NVTX nesting, fixed-sample prompt hashes and nested
prefixes, control/profile token equality, true round accounting, and the paired
1K-to-16K/32K scaling values. No performance conclusion is claimed by Mac tests.
