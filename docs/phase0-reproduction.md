# Phase 0 DFlash baseline reproduction

This procedure creates a single-GPU, Transformers-based Qwen3-8B DFlash smoke
test. It does not run sparse attention, long-context experiments, or the
official multi-process benchmark.

## Pinned inputs

- Repository branch: `codex/phase0-reproduction`
- Target: `Qwen/Qwen3-8B` at
  `b968826d9c46dd6066d109eabc6255188de91218`
- Draft: `z-lab/Qwen3-8B-DFlash-b16` at
  `9b41424b7109f9c5413454f481b09a82b85333f4`
- Batch size: 1
- Temperature: 0
- Maximum new tokens: 32

The model revisions were the respective Hub `main` commits observed on
2026-08-15. Override them only deliberately and retain the resulting JSON.

## 1. Fetch the branch on the GPU server

For a new checkout:

```bash
git clone https://github.com/rzwang22/dflash-longctx.git
cd dflash-longctx
git remote add upstream https://github.com/z-lab/dflash.git
git fetch origin codex/phase0-reproduction
git switch --track origin/codex/phase0-reproduction
git status --short --branch
git remote -v
```

For an existing checkout:

```bash
cd dflash-longctx
git fetch origin
git switch codex/phase0-reproduction
git pull --ff-only origin codex/phase0-reproduction
git status --short --branch
git remote -v
```

Do not switch, merge, or commit on `main` or
`baseline-dflash-20260815` for this work.

## 2. Create the GPU environment

The repository requires Python 3.10 or newer. Python 3.11 is the recommended
Phase 0 environment.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
python --version
nvidia-smi
```

Install the repository's Transformers backend. `--torch-backend=auto` lets `uv`
select a PyTorch build appropriate for the detected NVIDIA/CUDA environment.

```bash
uv pip install --torch-backend=auto -e ".[transformers]"
python -c "import accelerate, torch, transformers; print(torch.__version__, transformers.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The repository pins Transformers `4.57.1`. The selected draft revision was
published with `4.57.3`; successful compatibility is one of the Phase 0 GPU
checks, not an assumption. FlashAttention is not required by this smoke test.
If it is installed later for performance work, record that as a separate
environment/run because it may change attention dispatch.

## 3. Collect the pre-run environment

```bash
python scripts/collect_environment.py \
  --output outputs/phase0-smoke/preflight-environment.json \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --draft-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-new-tokens 32
```

This collector is safe on a CPU-only/Mac host: missing PyTorch, missing CUDA,
and partial CUDA probes are represented in JSON rather than causing a crash.
It does not load a tokenizer or model.

## 4. Run the single-GPU smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/phase0_smoke.py \
  --target-model Qwen/Qwen3-8B \
  --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --draft-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-new-tokens 32 \
  --prompt "How many positive whole-number divisors does 196 have?" \
  --output-dir outputs/phase0-smoke
```

The script refuses CPU/MPS fallback and performs its CUDA check before importing
Transformers or loading model files. Do not run this generation command on a
Mac. On a Mac, limit validation to `--help`, imports, argument/schema tests, and
the environment collector.

Do not use the README's `torchrun --nproc_per_node=8` benchmark as the first
smoke test. It adds distributed initialization, datasets, multiple GPUs, and a
different local-code path before the basic remote-code path is known to work.

## Expected files

The smoke-test output directory contains:

- `result.json`: structured run configuration, requested and resolved model
  revisions, loaded classes/files, token counts, timing, generated text,
  acceptance availability, and any error;
- `environment.json`: environment snapshot embedded in the result as a
  standalone file;
- `terminal.log`: mirrored Python stdout/stderr, including explicit failures;
- optionally `preflight-environment.json` from the separate collector.

The result format is documented by
`docs/schemas/phase0-smoke.schema.json`. The entire `outputs/` tree is ignored
by Git.

## Acceptance-length policy

At the pinned draft revision, remote `spec_generate` returns only an output ID
tensor even though it builds an acceptance list internally. The smoke test
records `acceptance.available=false` and a reason. It never guesses acceptance
from output IDs. If a deliberately selected future API directly returns
`acceptance_lengths`, the script records that explicit field.

## Common failures

### `accelerate` is missing

Install the complete Transformers extra again:

```bash
uv pip install --torch-backend=auto -e ".[transformers]"
```

The smoke test uses an explicit single-GPU `device_map`, which requires
`accelerate`.

### CUDA is unavailable

Check both the driver and the PyTorch build:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

An NVIDIA driver problem or a CPU-only PyTorch wheel must be fixed on the GPU
server. The script intentionally will not fall back to CPU or Apple MPS.

### Model/config/custom-code download fails

Confirm outbound access to Hugging Face, disk space in the cache location, and
the immutable revision. For gated/proxied environments, configure the normal
Hugging Face authentication mechanism outside the repository. Never save a
token in tracked files or command output. Retry the exact same pinned command
after fixing connectivity; do not silently replace a model or revision.

### Draft has no `spec_generate`, or custom code is rejected

Confirm that `--draft-model`, `--draft-revision`, and
`trust_remote_code=True` are unchanged. Inspect `result.json`/`terminal.log`.
The pinned draft config maps `AutoModel` to `dflash.DFlashDraftModel`.

### Transformers compatibility error

Keep the complete traceback and environment JSON. First test the repository
pin (`4.57.1`) as specified. If it is incompatible with the pinned remote code,
test `4.57.3` as a separately recorded diagnostic environment; do not silently
change `pyproject.toml` or claim the original environment passed.

### CUDA out of memory

Verify that only the selected GPU/process is using memory and retain the error
artifacts. Do not respond by offloading to CPU, quantizing weights, changing
dtype, or splitting across GPUs: each would cease to be this Phase 0 baseline.
