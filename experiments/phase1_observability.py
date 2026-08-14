#!/usr/bin/env python3
"""Single-request Phase 1A DFlash observability and token-exact A/B runner."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import shlex
import statistics
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase1.backend import attention_backend_snapshot
from dflash.phase1.materialize import (
    DRAFT_MODEL_ID,
    DRAFT_REVISION,
    verify_materialization,
)
from dflash.phase1.source import locate_callable, sha256_file
from experiments.phase0_smoke import (
    DEFAULT_PROMPT,
    TeeStream,
    _pretrained_location_kwargs,
    _python_file,
    _require_cuda,
    _resolve_model_source,
    _torch_dtype,
)
from scripts.collect_environment import (
    DEFAULT_TARGET_MODEL,
    DEFAULT_TARGET_REVISION,
    collect_environment,
    write_json,
)

SCHEMA_VERSION = "1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--target-revision", default=DEFAULT_TARGET_REVISION)
    parser.add_argument("--target-model-path", type=Path, default=None)
    parser.add_argument("--draft-model", default=DRAFT_MODEL_ID)
    parser.add_argument("--draft-revision", default=DRAFT_REVISION)
    parser.add_argument(
        "--draft-model-path",
        type=Path,
        default=None,
        help="Original materialized draft path; required for --run-mode ab.",
    )
    parser.add_argument("--materialized-draft-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measurement-runs", type=int, default=1)
    parser.add_argument("--enable-diagnostics", action="store_true")
    parser.add_argument("--enable-profiling", action="store_true")
    parser.add_argument("--run-mode", choices=("observe", "ab"), default="observe")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/phase1-observability")
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than zero")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative")
    if args.measurement_runs <= 0:
        raise ValueError("--measurement-runs must be greater than zero")
    if not args.prompt.strip():
        raise ValueError("--prompt must not be empty")
    if args.draft_model != DRAFT_MODEL_ID or args.draft_revision != DRAFT_REVISION:
        raise ValueError(
            "Phase 1A is restricted to the pinned Qwen3-8B-DFlash-b16 model and revision"
        )
    if args.run_mode == "ab" and args.draft_model_path is None:
        raise ValueError("--run-mode ab requires --draft-model-path")
    return args


def run_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target_model": args.target_model,
        "target_revision": args.target_revision,
        "target_model_path": str(args.target_model_path)
        if args.target_model_path
        else None,
        "draft_model": args.draft_model,
        "draft_revision": args.draft_revision,
        "draft_model_path": str(args.draft_model_path)
        if args.draft_model_path
        else None,
        "materialized_draft_path": str(args.materialized_draft_path),
        "device": args.device,
        "dtype": args.dtype,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "batch_size": 1,
        "warmup_runs": args.warmup_runs,
        "measurement_runs": args.measurement_runs,
        "enable_diagnostics": args.enable_diagnostics,
        "enable_profiling": args.enable_profiling,
        "run_mode": args.run_mode,
        "trust_remote_code": True,
        "backend": "transformers",
        "output_dir": str(args.output_dir),
    }


def new_summary(
    args: argparse.Namespace, environment: dict[str, Any], command: str
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "started",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "repository": environment["repository"],
        "configuration": run_configuration(args),
        "generation": {
            "prompt_tokens": None,
            "generated_tokens": None,
            "output_ids": None,
            "text": None,
            "stop_reason": None,
        },
        "measurements": [],
        "instrumentation_overhead": None,
        "equivalence": None,
        "attention_backend": None,
        "artifacts": {},
        "error": None,
    }


def new_source_provenance() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "started",
        "repository_custom_code_commit": None,
        "base_model_id": DRAFT_MODEL_ID,
        "base_model_revision": DRAFT_REVISION,
        "materialized_model_path": None,
        "custom_code_sha256": None,
        "actual_loaded_module_path": None,
        "actual_loaded_module_sha256": None,
        "loaded_source_matches_repository": None,
        "spec_generate": None,
        "original": None,
        "error": None,
    }


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        for row in rows:
            temporary.write(json.dumps(row, sort_keys=True) + "\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _load_materialized_draft(
    AutoModel: Any,
    *,
    model_path: Path,
    implementation: str,
    dtype: Any,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    manifest = verify_materialization(
        model_path, expected_implementation=implementation
    )
    model = AutoModel.from_pretrained(
        str(model_path.expanduser().resolve()),
        trust_remote_code=True,
        local_files_only=True,
        dtype=dtype,
        device_map=device,
    ).eval()
    return model, manifest


def _loaded_source_record(draft: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    loaded_path_text = _python_file(draft)
    if not loaded_path_text:
        raise RuntimeError("Could not locate the loaded DFlash class source")
    loaded_path = Path(loaded_path_text)
    loaded_hash = sha256_file(loaded_path)
    expected_hash = manifest["effective_custom_code_sha256"]["dflash.py"]
    matches = loaded_hash == expected_hash
    if not matches:
        raise RuntimeError(
            f"Loaded DFlash source hash {loaded_hash} does not match repository-controlled "
            f"hash {expected_hash}"
        )
    unwrapped_spec_generate = inspect.unwrap(draft.spec_generate)
    dependency_symbols = {"utils.py": "sample"}
    if "phase1_observability.py" in manifest["effective_custom_code_sha256"]:
        dependency_symbols["phase1_observability.py"] = "build_round_accounting"
    loaded_dependencies = {}
    for filename, symbol_name in dependency_symbols.items():
        symbol = unwrapped_spec_generate.__globals__.get(symbol_name)
        source_file = inspect.getsourcefile(symbol) if symbol is not None else None
        if source_file is None:
            raise RuntimeError(f"Could not locate loaded dependency {symbol_name}")
        source_path = Path(source_file).resolve()
        source_hash = sha256_file(source_path)
        expected_dependency_hash = manifest["effective_custom_code_sha256"][filename]
        dependency_matches = source_hash == expected_dependency_hash
        if not dependency_matches:
            raise RuntimeError(
                f"Loaded {filename} hash {source_hash} does not match "
                f"repository-controlled hash {expected_dependency_hash}"
            )
        loaded_dependencies[filename] = {
            "symbol": symbol_name,
            "actual_loaded_path": str(source_path),
            "actual_loaded_sha256": source_hash,
            "repository_sha256": expected_dependency_hash,
            "matches_repository": dependency_matches,
        }
    generation_helpers = {}
    for name in ("_spec_generate_reference", "_spec_generate_observed"):
        helper = getattr(draft, name, None)
        if helper is not None:
            generation_helpers[name] = locate_callable(helper)
    return {
        "repository_custom_code_commit": manifest["repository"]["commit"],
        "base_model_id": manifest["model_id"],
        "base_model_revision": manifest["base_model_revision"],
        "implementation": manifest["implementation"],
        "materialized_model_path": manifest["materialized_model_path"],
        "custom_code_sha256": manifest["effective_custom_code_sha256"],
        "actual_loaded_module_path": str(loaded_path),
        "actual_loaded_module_sha256": loaded_hash,
        "loaded_source_matches_repository": matches,
        "actual_loaded_dependencies": loaded_dependencies,
        "controlled_but_not_imported_files": ["modeling_dflash.py"],
        "spec_generate": locate_callable(draft.spec_generate),
        "generation_helpers": generation_helpers,
        "materialization_id": manifest["materialization_id"],
        "weight_provenance": manifest["weight_provenance"],
    }


def _prepare_inputs(tokenizer: Any, prompt: str, device: str) -> tuple[Any, str]:
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([chat_text], return_tensors="pt")["input_ids"].to(device)
    if inputs.shape[0] != 1:
        raise RuntimeError(f"Phase 1A requires batch size 1, got {inputs.shape[0]}")
    return inputs, chat_text


def _call_generation(
    *,
    torch_module: Any,
    draft: Any,
    target: Any,
    input_ids: Any,
    max_new_tokens: int,
    stop_token_ids: list[int],
    return_diagnostics: bool,
    enable_profiling: bool,
) -> tuple[Any, dict[str, Any] | None, dict[str, Any], str]:
    started_at = time.perf_counter()
    start_event = end_event = None
    if not enable_profiling:
        start_event = torch_module.cuda.Event(enable_timing=True)
        end_event = torch_module.cuda.Event(enable_timing=True)
        start_event.record()
    output = draft.spec_generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        target=target,
        stop_token_ids=stop_token_ids,
        return_diagnostics=return_diagnostics,
        enable_profiling=enable_profiling,
    )
    output_type = f"{type(output).__module__}.{type(output).__qualname__}"
    diagnostics = None
    output_ids = output
    if return_diagnostics:
        if not hasattr(output, "output_ids") or not hasattr(output, "diagnostics"):
            raise RuntimeError("Instrumented spec_generate did not return diagnostics")
        output_ids = output.output_ids
        diagnostics = output.diagnostics
    if enable_profiling:
        cuda_elapsed_ms = diagnostics["profiling"]["total"]["cuda_elapsed_ms"]
        runner_synchronize_count = 0
    else:
        end_event.record()
        torch_module.cuda.synchronize(input_ids.device)
        cuda_elapsed_ms = start_event.elapsed_time(end_event)
        runner_synchronize_count = 1
    wall_elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    measurement = {
        "cpu_wall_elapsed_ms": wall_elapsed_ms,
        "cuda_elapsed_ms": cuda_elapsed_ms,
        "runner_explicit_cuda_synchronize_count": runner_synchronize_count,
        "internal_profiling_explicit_cuda_synchronize_count": (
            diagnostics["profiling"]["explicit_cuda_synchronize_count"]
            if diagnostics is not None
            else 0
        ),
    }
    if diagnostics is not None:
        measurement["profile_breakdown"] = diagnostics["profiling"]
        measurement["accepted_draft_token_lengths"] = diagnostics[
            "accepted_draft_token_lengths"
        ]
        measurement["cache_commit_lengths"] = diagnostics["cache_commit_lengths"]
        measurement["stop_reason"] = diagnostics["stop_reason"]
    return output_ids, diagnostics, measurement, output_type


def _token_equal(torch_module: Any, left: Any, right: Any) -> bool:
    return bool(torch_module.equal(left, right))


def _enrich_rounds(
    diagnostics: dict[str, Any], backend: dict[str, Any], measurement_run: int
) -> list[dict[str, Any]]:
    target_backend = backend["target"]
    rows = []
    for round_record in diagnostics["rounds"]:
        record = dict(round_record)
        record["measurement_run"] = measurement_run
        record["target_attention"] = {
            "configured_attention_implementation": target_backend[
                "configured_attention_implementation"
            ],
            "calls_pytorch_sdpa": target_backend["calls_pytorch_sdpa"],
            "dtype": target_backend["dtype"],
            "query_length": record["draft_block_size"],
            "effective_kv_length": record["context_length_before"]
            + record["draft_block_size"],
            "query_head_count": target_backend["query_head_count"],
            "key_value_head_count": target_backend["key_value_head_count"],
            "head_dimension": target_backend["head_dimension"],
            "shape_source": "target verification input plus DynamicCache semantics",
            "actual_cuda_kernel": None,
        }
        rows.append(record)
    return rows


def _median(records: list[dict[str, Any]], field: str) -> float:
    return float(statistics.median(record[field] for record in records))


def _overhead(
    control: list[dict[str, Any]], observed: list[dict[str, Any]]
) -> dict[str, Any]:
    result = {}
    for field in ("cpu_wall_elapsed_ms", "cuda_elapsed_ms"):
        control_median = _median(control, field)
        observed_median = _median(observed, field)
        result[field] = {
            "control_median": control_median,
            "instrumented_median": observed_median,
            "absolute_overhead": observed_median - control_median,
            "percent_overhead": (
                (observed_median - control_median) / control_median * 100.0
                if control_median
                else None
            ),
        }
    result["warning"] = (
        "This paired single-request estimate includes synchronization and control-flow "
        "effects and is not a formal performance baseline."
    )
    return result


def _decode(tokenizer: Any, output_ids: Any) -> str:
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def _run_observe(
    *,
    args: argparse.Namespace,
    torch_module: Any,
    target: Any,
    draft: Any,
    tokenizer: Any,
    input_ids: Any,
    summary: dict[str, Any],
    rounds: list[dict[str, Any]],
) -> None:
    for _ in range(args.warmup_runs):
        draft.spec_generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            target=target,
            stop_token_ids=[tokenizer.eos_token_id],
        )
    if args.warmup_runs:
        torch_module.cuda.synchronize(input_ids.device)

    instrumented = args.enable_diagnostics or args.enable_profiling
    control_measurements = []
    observed_measurements = []
    reference_output_ids = None
    reference_text = None
    for measurement_run in range(args.measurement_runs):
        if instrumented:
            control_ids, _, control_timing, _ = _call_generation(
                torch_module=torch_module,
                draft=draft,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                return_diagnostics=False,
                enable_profiling=False,
            )
            control_timing["measurement_run"] = measurement_run
            control_timing["path"] = "instrumented_default_control"
            control_measurements.append(control_timing)
        else:
            control_ids = None

        output_ids, diagnostics, timing, _ = _call_generation(
            torch_module=torch_module,
            draft=draft,
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            return_diagnostics=instrumented,
            enable_profiling=args.enable_profiling,
        )
        timing["measurement_run"] = measurement_run
        timing["path"] = "observed" if instrumented else "instrumented_default"
        observed_measurements.append(timing)
        if control_ids is not None and not _token_equal(
            torch_module, control_ids, output_ids
        ):
            raise RuntimeError("Diagnostics/profiling changed generated output IDs")
        if reference_output_ids is None:
            reference_output_ids = output_ids
            reference_text = _decode(tokenizer, output_ids)
        elif not _token_equal(torch_module, reference_output_ids, output_ids):
            raise RuntimeError("Greedy measurement runs produced different output IDs")
        if diagnostics is not None:
            rounds.extend(
                _enrich_rounds(
                    diagnostics, summary["attention_backend"], measurement_run
                )
            )
            summary["generation"]["stop_reason"] = diagnostics["stop_reason"]

    summary["measurements"] = control_measurements + observed_measurements
    summary["instrumentation_overhead"] = (
        _overhead(control_measurements, observed_measurements)
        if instrumented
        else {
            "status": "not_applicable",
            "reason": "Diagnostics and profiling were both disabled.",
        }
    )
    summary["equivalence"] = {
        "diagnostics_or_profiling_vs_default_output_ids_equal": True
        if instrumented
        else None,
        "measurement_run_output_ids_equal": True,
    }
    summary["generation"].update(
        {
            "prompt_tokens": int(input_ids.shape[1]),
            "generated_tokens": int(reference_output_ids.shape[1] - input_ids.shape[1]),
            "output_ids": reference_output_ids[0].tolist(),
            "text": reference_text,
        }
    )


def _run_ab(
    *,
    args: argparse.Namespace,
    torch_module: Any,
    AutoModel: Any,
    dtype: Any,
    target: Any,
    instrumented_draft: Any,
    tokenizer: Any,
    input_ids: Any,
    summary: dict[str, Any],
    source_provenance: dict[str, Any],
    rounds: list[dict[str, Any]],
) -> None:
    original_draft, original_manifest = _load_materialized_draft(
        AutoModel,
        model_path=args.draft_model_path,
        implementation="original",
        dtype=dtype,
        device=args.device,
    )
    source_provenance["original"] = _loaded_source_record(
        original_draft, original_manifest
    )
    original_output = original_draft.spec_generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        target=target,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    torch_module.cuda.synchronize(input_ids.device)
    original_type = (
        f"{type(original_output).__module__}.{type(original_output).__qualname__}"
    )
    original_text = _decode(tokenizer, original_output)
    del original_draft
    gc.collect()
    torch_module.cuda.empty_cache()

    default_output = instrumented_draft.spec_generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        target=target,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    default_type = (
        f"{type(default_output).__module__}.{type(default_output).__qualname__}"
    )
    diagnostic_result = instrumented_draft.spec_generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        target=target,
        stop_token_ids=[tokenizer.eos_token_id],
        return_diagnostics=True,
    )
    profiling_result = instrumented_draft.spec_generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        target=target,
        stop_token_ids=[tokenizer.eos_token_id],
        return_diagnostics=True,
        enable_profiling=True,
    )
    comparisons = {
        "original_vs_instrumented_default_ids": _token_equal(
            torch_module, original_output, default_output
        ),
        "original_vs_instrumented_default_text": original_text
        == _decode(tokenizer, default_output),
        "default_return_type_equal": original_type == default_type,
        "diagnostics_toggle_ids": _token_equal(
            torch_module, default_output, diagnostic_result.output_ids
        ),
        "profiling_toggle_ids": _token_equal(
            torch_module, default_output, profiling_result.output_ids
        ),
    }
    if not all(comparisons.values()):
        raise RuntimeError(
            f"Original/instrumented A/B equivalence failed: {comparisons}"
        )
    rounds.extend(
        _enrich_rounds(profiling_result.diagnostics, summary["attention_backend"], 0)
    )
    summary["equivalence"] = comparisons
    summary["instrumentation_overhead"] = {
        "status": "not_measured_in_ab_mode",
        "reason": "Use observe mode measurement runs for paired timing.",
    }
    summary["generation"].update(
        {
            "prompt_tokens": int(input_ids.shape[1]),
            "generated_tokens": int(default_output.shape[1] - input_ids.shape[1]),
            "output_ids": default_output[0].tolist(),
            "text": _decode(tokenizer, default_output),
            "stop_reason": profiling_result.diagnostics["stop_reason"],
        }
    )


def run_phase1(
    args: argparse.Namespace,
    summary: dict[str, Any],
    source_provenance: dict[str, Any],
    rounds: list[dict[str, Any]],
) -> None:
    torch_module = _require_cuda(args)
    try:
        import transformers
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Transformers backend is not installed") from exc

    target_source = _resolve_model_source(
        "target", args.target_model, args.target_revision, args.target_model_path
    )
    dtype = _torch_dtype(torch_module, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        target_source["load_source"],
        trust_remote_code=True,
        **_pretrained_location_kwargs(target_source),
    )
    target = AutoModelForCausalLM.from_pretrained(
        target_source["load_source"],
        trust_remote_code=True,
        dtype=dtype,
        device_map=args.device,
        **_pretrained_location_kwargs(target_source),
    ).eval()
    target_source["class_name"] = (
        f"{target.__class__.__module__}.{target.__class__.__qualname__}"
    )
    target_source["python_file"] = _python_file(target)
    source_provenance["target"] = target_source
    instrumented_draft, manifest = _load_materialized_draft(
        AutoModel,
        model_path=args.materialized_draft_path,
        implementation="instrumented",
        dtype=dtype,
        device=args.device,
    )
    source_provenance.update(_loaded_source_record(instrumented_draft, manifest))
    source_provenance["status"] = "ok"
    input_ids, _ = _prepare_inputs(tokenizer, args.prompt, args.device)
    summary["attention_backend"] = attention_backend_snapshot(
        torch_module, target=target, draft=instrumented_draft
    )
    summary["attention_backend"]["target"]["prefill_query_length"] = int(
        input_ids.shape[1]
    )
    summary["attention_backend"]["target"]["prefill_kv_length"] = int(
        input_ids.shape[1]
    )
    print(f"Git commit: {summary['repository']['commit_sha']}")
    print(f"Transformers: {transformers.__version__}")
    print(f"GPU: {torch_module.cuda.get_device_name(args.device)}")
    print(
        f"Instrumented DFlash module: {source_provenance['actual_loaded_module_path']}"
    )
    print(
        "Unwrapped spec_generate: "
        f"{source_provenance['spec_generate']['unwrapped']['definition_file']}:"
        f"{source_provenance['spec_generate']['unwrapped']['starting_line']}"
    )
    if args.run_mode == "ab":
        _run_ab(
            args=args,
            torch_module=torch_module,
            AutoModel=AutoModel,
            dtype=dtype,
            target=target,
            instrumented_draft=instrumented_draft,
            tokenizer=tokenizer,
            input_ids=input_ids,
            summary=summary,
            source_provenance=source_provenance,
            rounds=rounds,
        )
    else:
        _run_observe(
            args=args,
            torch_module=torch_module,
            target=target,
            draft=instrumented_draft,
            tokenizer=tokenizer,
            input_ids=input_ids,
            summary=summary,
            rounds=rounds,
        )
    summary["status"] = "ok"
    print(f"Generated tokens: {summary['generation']['generated_tokens']}")
    print("Generated text:")
    print(summary["generation"]["text"])


def validate_summary(summary: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "status",
        "timestamp_utc",
        "command",
        "repository",
        "configuration",
        "generation",
        "measurements",
        "instrumentation_overhead",
        "equivalence",
        "attention_backend",
        "artifacts",
        "error",
    }
    missing = required - summary.keys()
    if missing:
        raise ValueError(f"summary is missing fields: {sorted(missing)}")
    if summary["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported Phase 1A summary schema")
    if summary["status"] not in {"started", "ok", "error"}:
        raise ValueError(f"invalid Phase 1A status: {summary['status']}")
    if summary["status"] == "ok":
        if summary["generation"]["output_ids"] is None:
            raise ValueError("successful result has no output IDs")
        for measurement in summary["measurements"]:
            for field in ("cpu_wall_elapsed_ms", "cuda_elapsed_ms"):
                if not math.isfinite(measurement[field]):
                    raise ValueError(f"non-finite measurement {field}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    rounds_path = output_dir / "rounds.jsonl"
    environment_path = output_dir / "environment.json"
    provenance_path = output_dir / "source-provenance.json"
    terminal_log_path = output_dir / "terminal.log"
    command = shlex.join([sys.executable, *sys.argv])
    environment = collect_environment(
        run_config=run_configuration(args), run_command=command
    )
    summary = new_summary(args, environment, command)
    source_provenance = new_source_provenance()
    rounds: list[dict[str, Any]] = []
    summary["artifacts"] = {
        "summary_json": str(summary_path),
        "rounds_jsonl": str(rounds_path),
        "environment_json": str(environment_path),
        "source_provenance_json": str(provenance_path),
        "terminal_log": str(terminal_log_path),
    }
    write_json(environment_path, environment)

    original_stdout, original_stderr = sys.stdout, sys.stderr
    exit_code = 1
    with terminal_log_path.open("w", encoding="utf-8", buffering=1) as log:
        sys.stdout = TeeStream(original_stdout, log)
        sys.stderr = TeeStream(original_stderr, log)
        try:
            print(f"Phase 1A output directory: {output_dir}")
            run_phase1(args, summary, source_provenance, rounds)
            exit_code = 0
        except KeyboardInterrupt:
            summary["status"] = "error"
            summary["error"] = {
                "type": "KeyboardInterrupt",
                "message": "Interrupted by user",
            }
            source_provenance["status"] = "error"
            source_provenance["error"] = summary["error"]
            exit_code = 130
            print("ERROR: KeyboardInterrupt: interrupted by user", file=sys.stderr)
            traceback.print_exc()
        except Exception as exc:  # noqa: BLE001 - persist every runtime failure.
            summary["status"] = "error"
            summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
            source_provenance["status"] = "error"
            source_provenance["error"] = summary["error"]
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
        finally:
            try:
                validate_summary(summary)
                write_json(summary_path, summary)
                _atomic_write_jsonl(rounds_path, rounds)
                write_json(provenance_path, source_provenance)
                print(f"Summary JSON: {summary_path}")
                print(f"Rounds JSONL: {rounds_path}")
                print(f"Environment JSON: {environment_path}")
                print(f"Source provenance JSON: {provenance_path}")
                print(f"Terminal log: {terminal_log_path}")
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
