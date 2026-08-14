#!/usr/bin/env python3
"""Single-GPU Qwen3-8B DFlash baseline smoke test."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import math
import shlex
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_environment import (
    DEFAULT_DRAFT_MODEL,
    DEFAULT_DRAFT_REVISION,
    DEFAULT_TARGET_MODEL,
    DEFAULT_TARGET_REVISION,
    collect_environment,
    write_json,
)

SCHEMA_VERSION = "1.0"
DEFAULT_PROMPT = "How many positive whole-number divisors does 196 have?"


class TeeStream:
    """Mirror Python stdout/stderr to the terminal and a UTF-8 log file."""

    def __init__(self, terminal: TextIO, log: TextIO) -> None:
        self.terminal = terminal
        self.log = log

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.terminal, name)

    @property
    def encoding(self) -> str:
        return self.terminal.encoding or "utf-8"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--target-revision", default=DEFAULT_TARGET_REVISION)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--draft-revision", default=DEFAULT_DRAFT_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase0-smoke"))
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than zero")
    if not args.prompt.strip():
        raise ValueError("--prompt must not be empty")
    return args


def run_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target_model": args.target_model,
        "target_revision": args.target_revision,
        "draft_model": args.draft_model,
        "draft_revision": args.draft_revision,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": 1,
        "temperature": 0.0,
        "max_new_tokens": args.max_new_tokens,
        "prompt": args.prompt,
        "trust_remote_code": True,
        "backend": "transformers",
        "enable_thinking": False,
        "output_dir": str(args.output_dir),
    }


def new_result_record(
    args: argparse.Namespace,
    environment: dict[str, Any],
    command: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "started",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository": environment["repository"],
        "command": command,
        "configuration": run_configuration(args),
        "environment": environment,
        "models": {
            "target": {
                "id": args.target_model,
                "requested_revision": args.target_revision,
                "resolved_revision": None,
                "class_name": None,
                "python_file": None,
            },
            "draft": {
                "id": args.draft_model,
                "requested_revision": args.draft_revision,
                "resolved_revision": None,
                "class_name": None,
                "python_file": None,
            },
        },
        "code_paths": {
            "spec_generate_definition_file": None,
        },
        "generation": {
            "prompt_tokens": None,
            "generated_tokens": None,
            "elapsed_seconds": None,
            "text": None,
        },
        "acceptance": {
            "available": False,
            "lengths": None,
            "mean_length": None,
            "source": None,
            "unavailable_reason": (
                "The pinned remote spec_generate API returns only output_ids; it does not expose "
                "its internal acceptance_lengths list."
            ),
        },
        "artifacts": {
            "result_json": None,
            "terminal_log": None,
            "environment_json": None,
        },
        "error": None,
    }


def validate_result_record(record: dict[str, Any]) -> None:
    required_top_level = {
        "schema_version",
        "status",
        "timestamp_utc",
        "repository",
        "command",
        "configuration",
        "environment",
        "models",
        "code_paths",
        "generation",
        "acceptance",
        "artifacts",
        "error",
    }
    missing = required_top_level - record.keys()
    if missing:
        raise ValueError(f"Result record is missing fields: {sorted(missing)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version: {record['schema_version']}")
    if record["status"] not in {"started", "ok", "error"}:
        raise ValueError(f"Invalid status: {record['status']}")
    if record["configuration"].get("batch_size") != 1:
        raise ValueError("Phase 0 smoke test schema requires batch_size=1")
    if record["configuration"].get("temperature") != 0.0:
        raise ValueError("Phase 0 smoke test schema requires temperature=0.0")
    for role in ("target", "draft"):
        model = record["models"].get(role, {})
        for field in (
            "id",
            "requested_revision",
            "resolved_revision",
            "class_name",
            "python_file",
        ):
            if field not in model:
                raise ValueError(f"models.{role}.{field} is required")
    for field in ("prompt_tokens", "generated_tokens", "elapsed_seconds", "text"):
        if field not in record["generation"]:
            raise ValueError(f"generation.{field} is required")
    if record["status"] == "ok":
        for role in ("target", "draft"):
            for field in ("resolved_revision", "class_name", "python_file"):
                if not record["models"][role][field]:
                    raise ValueError(
                        f"models.{role}.{field} must be populated on success"
                    )
        if not record["code_paths"]["spec_generate_definition_file"]:
            raise ValueError(
                "spec_generate_definition_file must be populated on success"
            )
        if any(record["generation"][field] is None for field in record["generation"]):
            raise ValueError("All generation fields must be populated on success")


def _require_cuda(args: argparse.Namespace) -> Any:
    if args.device != "cuda" and not args.device.startswith("cuda:"):
        raise RuntimeError(
            f"Phase 0 smoke test requires a CUDA device, got --device={args.device!r}; "
            "CPU and MPS fallback are intentionally disabled."
        )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install the Transformers backend before running the smoke test."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch. Run this smoke test on an NVIDIA GPU host with a "
            "CUDA-enabled PyTorch build; no CPU or MPS fallback will be attempted."
        )
    try:
        device_index = 0 if args.device == "cuda" else int(args.device.split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid CUDA device {args.device!r}; use 'cuda' or an indexed device such as 'cuda:0'."
        ) from exc
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {args.device}, but PyTorch reports {torch.cuda.device_count()} CUDA device(s)."
        )
    if importlib.util.find_spec("accelerate") is None:
        raise RuntimeError(
            "accelerate is not installed. It is required for the explicit single-GPU device_map; "
            "install the repository's Transformers extra."
        )
    return torch


def _torch_dtype(torch_module: Any, name: str) -> Any:
    if name == "auto":
        return "auto"
    return {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }[name]


def _python_file(obj: Any) -> str | None:
    candidate = obj if inspect.isclass(obj) or inspect.ismethod(obj) else obj.__class__
    try:
        return str(Path(inspect.getfile(candidate)).resolve())
    except (OSError, TypeError):
        return inspect.getsourcefile(candidate)


def _resolved_revision(model: Any) -> str | None:
    return getattr(getattr(model, "config", None), "_commit_hash", None)


def _verify_resolved_revision(role: str, requested: str, resolved: str | None) -> None:
    if not resolved:
        raise RuntimeError(
            f"Could not determine the resolved {role} model revision from the loaded config."
        )
    is_commit = 7 <= len(requested) <= 40 and all(
        character in "0123456789abcdefABCDEF" for character in requested
    )
    if is_commit and not resolved.lower().startswith(requested.lower()):
        raise RuntimeError(
            f"Resolved {role} revision {resolved} does not match requested commit {requested}."
        )


def _record_direct_acceptance(output: Any, record: dict[str, Any]) -> Any:
    """Use acceptance data only when the generation API returns it explicitly."""

    if not hasattr(output, "output_ids"):
        return output
    output_ids = output.output_ids
    lengths = getattr(output, "acceptance_lengths", None)
    if lengths is not None:
        numeric_lengths = [int(value) for value in lengths]
        record["acceptance"] = {
            "available": True,
            "lengths": numeric_lengths,
            "mean_length": mean(numeric_lengths) if numeric_lengths else None,
            "source": "spec_generate return object.acceptance_lengths",
            "unavailable_reason": None,
        }
    return output_ids


def run_smoke(args: argparse.Namespace, record: dict[str, Any]) -> None:
    torch = _require_cuda(args)
    try:
        import transformers
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is not installed. Install with: uv pip install --torch-backend=auto "
            "-e '.[transformers]'"
        ) from exc

    dtype = _torch_dtype(torch, args.dtype)
    print(f"Git commit: {record['repository']['commit_sha']}")
    print(f"Python: {record['environment']['python']['version']}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(args.device)}")
    print(f"Target: {args.target_model} @ {args.target_revision}")
    print(f"Draft: {args.draft_model} @ {args.draft_revision}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model,
        revision=args.target_revision,
        trust_remote_code=True,
    )
    print("Loading target model...")
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        revision=args.target_revision,
        trust_remote_code=True,
        dtype=dtype,
        device_map=args.device,
    ).eval()
    print("Loading DFlash draft model...")
    draft = AutoModel.from_pretrained(
        args.draft_model,
        revision=args.draft_revision,
        trust_remote_code=True,
        dtype=dtype,
        device_map=args.device,
    ).eval()

    if not hasattr(draft, "spec_generate"):
        raise RuntimeError(
            f"Loaded draft class {type(draft).__name__} has no spec_generate method. "
            "Check trust_remote_code and the draft revision."
        )

    target_file = _python_file(target)
    draft_file = _python_file(draft)
    spec_file = _python_file(draft.spec_generate)
    target_resolved_revision = _resolved_revision(target)
    draft_resolved_revision = _resolved_revision(draft)
    record["models"]["target"].update(
        {
            "resolved_revision": target_resolved_revision,
            "class_name": f"{target.__class__.__module__}.{target.__class__.__qualname__}",
            "python_file": target_file,
        }
    )
    record["models"]["draft"].update(
        {
            "resolved_revision": draft_resolved_revision,
            "class_name": f"{draft.__class__.__module__}.{draft.__class__.__qualname__}",
            "python_file": draft_file,
        }
    )
    record["code_paths"]["spec_generate_definition_file"] = spec_file
    _verify_resolved_revision("target", args.target_revision, target_resolved_revision)
    _verify_resolved_revision("draft", args.draft_revision, draft_resolved_revision)
    if not target_file or not draft_file or not spec_file:
        raise RuntimeError(
            "Could not resolve all required Python source paths for target, draft, and spec_generate."
        )
    print(f"Target Python file: {target_file}")
    print(f"Draft Python file: {draft_file}")
    print(f"spec_generate definition file: {spec_file}")

    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([chat_text], return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(args.device)
    if input_ids.shape[0] != 1:
        raise RuntimeError(
            f"Expected batch size 1, got input_ids shape {tuple(input_ids.shape)}"
        )
    prompt_tokens = int(input_ids.shape[1])
    print(f"Prompt tokens: {prompt_tokens}")

    torch.cuda.synchronize(args.device)
    started_at = time.perf_counter()
    output = draft.spec_generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        target=target,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter() - started_at
    output_ids = _record_direct_acceptance(output, record)
    if (
        not torch.is_tensor(output_ids)
        or output_ids.ndim != 2
        or output_ids.shape[0] != 1
    ):
        raise RuntimeError(
            "spec_generate returned an unsupported value; expected a [1, sequence] tensor or an "
            "object with output_ids."
        )
    generated_tokens = int(output_ids.shape[1]) - prompt_tokens
    if generated_tokens < 0:
        raise RuntimeError(
            f"spec_generate output is shorter than the prompt ({output_ids.shape[1]} < {prompt_tokens})."
        )
    generated_ids = output_ids[0, prompt_tokens:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    record["generation"].update(
        {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "elapsed_seconds": elapsed,
            "text": generated_text,
        }
    )
    record["status"] = "ok"
    print(f"Generated tokens: {generated_tokens}")
    print(f"Generation elapsed: {elapsed:.6f} s")
    if record["acceptance"]["available"]:
        print(f"Acceptance lengths: {record['acceptance']['lengths']}")
    else:
        print(
            f"Acceptance lengths: unavailable ({record['acceptance']['unavailable_reason']})"
        )
    print("Generated text:")
    print(generated_text)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    environment_path = output_dir / "environment.json"
    log_path = output_dir / "terminal.log"
    command = shlex.join([sys.executable, *sys.argv])
    environment = collect_environment(
        run_config=run_configuration(args),
        run_command=command,
    )
    record = new_result_record(args, environment, command)
    record["artifacts"] = {
        "result_json": str(result_path),
        "terminal_log": str(log_path),
        "environment_json": str(environment_path),
    }
    write_json(environment_path, environment)

    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        sys.stdout = TeeStream(original_stdout, log)
        sys.stderr = TeeStream(original_stderr, log)
        try:
            print(f"Phase 0 output directory: {output_dir}")
            run_smoke(args, record)
        except Exception as exc:  # noqa: BLE001 - every model/runtime failure must be persisted.
            record["status"] = "error"
            record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
        finally:
            try:
                validate_result_record(record)
                if record["generation"][
                    "elapsed_seconds"
                ] is not None and not math.isfinite(
                    record["generation"]["elapsed_seconds"]
                ):
                    raise ValueError("generation.elapsed_seconds must be finite")
                write_json(result_path, record)
                print(f"Result JSON: {result_path}")
                print(f"Environment JSON: {environment_path}")
                print(f"Terminal log: {log_path}")
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
    return 0 if record["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
