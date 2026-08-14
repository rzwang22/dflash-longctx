#!/usr/bin/env python3
"""Collect a reproducibility snapshot without requiring CUDA or model downloads."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_MODEL = "Qwen/Qwen3-8B"
DEFAULT_TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DEFAULT_DRAFT_MODEL = "z-lab/Qwen3-8B-DFlash-b16"
DEFAULT_DRAFT_REVISION = "9b41424b7109f9c5413454f481b09a82b85333f4"

OPTIONAL_DISTRIBUTIONS = {
    "accelerate": "accelerate",
    "datasets": "datasets",
    "flash_attn": "flash-attn",
    "huggingface_hub": "huggingface-hub",
    "loguru": "loguru",
    "numpy": "numpy",
    "requests": "requests",
    "rich": "rich",
    "tqdm": "tqdm",
    "typing_extensions": "typing-extensions",
}


def _distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _cuda_snapshot(torch_module: Any | None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "available": False,
        "runtime_version": None,
        "cudnn_version": None,
        "gpu_count": 0,
        "gpus": [],
        "probe_error": None,
    }
    if torch_module is None:
        return snapshot

    snapshot["runtime_version"] = getattr(
        getattr(torch_module, "version", None), "cuda", None
    )
    try:
        cudnn = getattr(getattr(torch_module, "backends", None), "cudnn", None)
        snapshot["cudnn_version"] = cudnn.version() if cudnn is not None else None
        snapshot["available"] = bool(torch_module.cuda.is_available())
        if not snapshot["available"]:
            return snapshot

        snapshot["gpu_count"] = int(torch_module.cuda.device_count())
        for index in range(snapshot["gpu_count"]):
            gpu: dict[str, Any] = {
                "index": index,
                "name": torch_module.cuda.get_device_name(index),
            }
            try:
                properties = torch_module.cuda.get_device_properties(index)
                gpu["total_memory_bytes"] = int(properties.total_memory)
                gpu["compute_capability"] = f"{properties.major}.{properties.minor}"
            except Exception as exc:  # noqa: BLE001 - a partial CUDA probe must not abort collection.
                gpu["properties_error"] = f"{type(exc).__name__}: {exc}"
            snapshot["gpus"].append(gpu)
    except Exception as exc:  # noqa: BLE001 - CPU-only and partial installs are valid inputs.
        snapshot["probe_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def collect_environment(
    *,
    repo_root: Path = REPO_ROOT,
    run_config: dict[str, Any] | None = None,
    run_command: str | None = None,
    torch_module: Any | None = None,
    import_torch: bool = True,
) -> dict[str, Any]:
    """Return a JSON-serializable environment snapshot.

    ``torch`` is imported only when this function is called. Importing this module is
    therefore safe for CLI/schema tests and cannot initiate a model download.
    """

    torch_import_error = None
    if torch_module is None and import_torch:
        try:
            torch_module = importlib.import_module("torch")
        except Exception as exc:  # noqa: BLE001 - record broken optional installs as environment data.
            torch_import_error = f"{type(exc).__name__}: {exc}"

    commit = _git(repo_root, "rev-parse", "HEAD")
    branch = _git(repo_root, "branch", "--show-current")
    status = _git(repo_root, "status", "--porcelain")
    torch_version = (
        str(getattr(torch_module, "__version__", "unknown"))
        if torch_module is not None
        else _distribution_version("torch")
    )
    transformers_version = _distribution_version("transformers")

    optional_dependencies = {}
    for import_name, distribution in OPTIONAL_DISTRIBUTIONS.items():
        version = _distribution_version(distribution)
        optional_dependencies[import_name] = {
            "installed": version is not None,
            "version": version,
        }

    return {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "root": str(repo_root.resolve()),
            "commit_sha": commit,
            "branch": branch,
            "dirty": bool(status) if status is not None else None,
        },
        "operating_system": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": {
            "torch": {
                "installed": torch_version is not None,
                "version": torch_version,
                "import_error": torch_import_error,
            },
            "transformers": {
                "installed": transformers_version is not None,
                "version": transformers_version,
            },
            "optional": optional_dependencies,
        },
        "cuda": _cuda_snapshot(torch_module),
        "run": {
            "command": run_command,
            "parameters": run_config or {},
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/phase0-environment.json")
    )
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--target-revision", default=DEFAULT_TARGET_REVISION)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--draft-revision", default=DEFAULT_DRAFT_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--prompt", default="How many positive whole-number divisors does 196 have?"
    )
    parser.add_argument(
        "--run-command",
        default=None,
        help="Exact model command this snapshot describes; defaults to this collector invocation.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_config = {
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
        "backend": "transformers",
        "trust_remote_code": True,
        "enable_thinking": False,
    }
    command = args.run_command or shlex.join([sys.executable, *sys.argv])
    snapshot = collect_environment(run_config=run_config, run_command=command)
    write_json(args.output, snapshot)
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    print(f"Environment snapshot written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
