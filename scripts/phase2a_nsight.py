#!/usr/bin/env python3
"""Export compatible Nsight Systems reports and build a kernel breakdown."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase2a.nsight import (
    read_nvtx_kernel_summary,
    read_nvtx_projection_summary,
    write_nsight_outputs,
)

REQUIRED_REPORT = "nvtx_kern_sum"
OPTIONAL_REPORT = "nvtx_gpu_proj_sum"
REQUESTED_REPORTS = (REQUIRED_REPORT, OPTIONAL_REPORT)
MANIFEST_NAME = "nsight-export-manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser(
        "export", help="Run each supported official nsys stats report separately"
    )
    source = export.add_mutually_exclusive_group(required=True)
    source.add_argument("--nsys-rep", type=Path)
    source.add_argument("--sqlite", type=Path)
    source.add_argument("--nvtx-kernel-csv", type=Path)
    export.add_argument("--nvtx-projection-csv", type=Path, default=None)
    export.add_argument("--output-dir", type=Path, required=True)

    parse = subparsers.add_parser("parse", help="Parse existing report CSV files")
    parse.add_argument("--nvtx-kernel-csv", type=Path, required=True)
    parse.add_argument("--nvtx-projection-csv", type=Path, default=None)
    parse.add_argument("--output-dir", type=Path, required=True)
    return parser


def _run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )


def _probe_nsys() -> dict[str, Any]:
    version_command = ["nsys", "--version"]
    reports_command = ["nsys", "stats", "--help-reports"]
    try:
        version = _run_capture(version_command)
        reports = _run_capture(reports_command)
    except OSError as exc:
        return {
            "nsys_version": None,
            "available_reports": [],
            "probe_status": "failed",
            "probe_error": f"{type(exc).__name__}: {exc}",
            "version_command": version_command,
            "reports_command": reports_command,
        }

    help_text = f"{reports.stdout}\n{reports.stderr}"
    available = set()
    for line in help_text.splitlines():
        match = re.match(r"^\s*([a-z][a-z0-9_]+)(?:\s|:|-)", line)
        if match:
            available.add(match.group(1))
    for report_name in REQUESTED_REPORTS:
        if re.search(rf"\b{re.escape(report_name)}\b", help_text):
            available.add(report_name)
    version_text = f"{version.stdout}\n{version.stderr}".strip()
    return {
        "nsys_version": version_text or None,
        "available_reports": sorted(available),
        "probe_status": (
            "ok" if version.returncode == 0 and reports.returncode == 0 else "failed"
        ),
        "probe_error": (
            None
            if version.returncode == 0 and reports.returncode == 0
            else f"version_rc={version.returncode}, reports_rc={reports.returncode}"
        ),
        "version_command": version_command,
        "reports_command": reports_command,
        "version_returncode": version.returncode,
        "reports_returncode": reports.returncode,
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _candidate_paths(output_dir: Path, report_name: str) -> list[Path]:
    expected = output_dir / f"nsight_{report_name}.csv"
    candidates = [expected, *sorted(output_dir.glob(f"*{report_name}*.csv"))]
    unique = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _existing_report(
    output_dir: Path, report_name: str
) -> tuple[Path | None, list[Path]]:
    checked = _candidate_paths(output_dir, report_name)
    existing = [path for path in checked if path.is_file()]
    if not existing:
        return None, checked
    exact = output_dir.resolve() / f"nsight_{report_name}.csv"
    if exact in existing:
        return exact, checked
    if len(existing) == 1:
        return existing[0], checked
    raise RuntimeError(
        f"Multiple existing {report_name} CSVs found: "
        + ", ".join(str(path) for path in existing)
    )


def _run_report(
    source: Path, output_dir: Path, report_name: str
) -> tuple[Path | None, dict[str, Any], list[Path]]:
    prefix = output_dir / "nsight"
    command = ["nsys", "stats"]
    if source.suffix == ".nsys-rep":
        command.append("--force-export=true")
    command.extend(
        [
            "--force-overwrite=true",
            "--report",
            report_name,
            "--format",
            "csv",
            "--output",
            str(prefix),
            str(source),
        ]
    )
    print(shlex.join(command))
    try:
        completed = _run_capture(command)
    except OSError as exc:
        return (
            None,
            {
                "report": report_name,
                "command": command,
                "status": "failed",
                "returncode": None,
                "stdout": "",
                "stderr": f"{type(exc).__name__}: {exc}",
            },
            _candidate_paths(output_dir, report_name),
        )
    report_path, checked = _existing_report(output_dir, report_name)
    attempt = {
        "report": report_name,
        "command": command,
        "status": "succeeded"
        if completed.returncode == 0 and report_path
        else "failed",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_path": str(report_path) if report_path else None,
    }
    return report_path, attempt, checked


def _base_manifest(command: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "status": "started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "analysis_mode": None,
        "projection_report_status": None,
        "nsys_version": None,
        "available_reports": [],
        "requested_reports": list(REQUESTED_REPORTS),
        "required_reports": [REQUIRED_REPORT],
        "optional_reports": [OPTIONAL_REPORT],
        "succeeded_reports": [],
        "generated_reports": [],
        "reused_reports": [],
        "unavailable_reports": [],
        "failed_reports": [],
        "source_reports": [],
        "checked_paths": {},
        "report_attempts": [],
        "artifacts": {},
        "input": None,
        "error": None,
    }


def _required_report_error(manifest: dict[str, Any]) -> RuntimeError:
    checked = manifest["checked_paths"].get(REQUIRED_REPORT, [])
    return RuntimeError(
        f"Required Nsight report {REQUIRED_REPORT} is missing; "
        f"nsys_version={manifest['nsys_version']!r}; "
        f"available_reports={manifest['available_reports']}; "
        f"checked_paths={checked}"
    )


def _write_outputs(
    output_dir: Path,
    *,
    kernel_csv: Path,
    projection_csv: Path | None,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    kernel_rows = read_nvtx_kernel_summary(kernel_csv)
    nvtx_rows = read_nvtx_projection_summary(projection_csv) if projection_csv else None
    paths = write_nsight_outputs(
        output_dir,
        kernel_rows=kernel_rows,
        nvtx_rows=nvtx_rows,
        analysis_mode=manifest["analysis_mode"],
        projection_report_status=manifest["projection_report_status"],
        source_reports=manifest["source_reports"],
    )
    manifest_path = output_dir / MANIFEST_NAME
    paths["nsight_export_manifest_json"] = manifest_path
    manifest["artifacts"] = {name: str(path) for name, path in paths.items()}
    manifest["status"] = "ok"
    _write_manifest(manifest_path, manifest)
    return paths


def run_export(args: argparse.Namespace, *, command: str) -> dict[str, Path]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    manifest = _base_manifest(command)
    try:
        explicit_kernel = (
            args.nvtx_kernel_csv.expanduser().resolve()
            if args.nvtx_kernel_csv
            else None
        )
        explicit_projection = (
            args.nvtx_projection_csv.expanduser().resolve()
            if args.nvtx_projection_csv
            else None
        )
        source = args.nsys_rep or args.sqlite
        source = source.expanduser().resolve() if source else None
        input_path = explicit_kernel or source
        manifest["input"] = {
            "path": str(input_path),
            "kind": (
                "nvtx_kern_sum_csv"
                if explicit_kernel
                else "sqlite"
                if args.sqlite
                else "nsys_rep"
            ),
        }
        manifest.update(_probe_nsys())
        if input_path is None or not input_path.is_file():
            if explicit_kernel is not None:
                manifest["checked_paths"][REQUIRED_REPORT] = [str(explicit_kernel)]
                raise _required_report_error(manifest)
            raise RuntimeError(f"Nsight input does not exist: {input_path}")

        kernel_csv = explicit_kernel
        projection_csv = explicit_projection
        if kernel_csv:
            manifest["reused_reports"].append(REQUIRED_REPORT)
            manifest["checked_paths"][REQUIRED_REPORT] = [str(kernel_csv)]
        else:
            kernel_csv, checked = _existing_report(output_dir, REQUIRED_REPORT)
            manifest["checked_paths"][REQUIRED_REPORT] = [str(path) for path in checked]
            if kernel_csv:
                manifest["reused_reports"].append(REQUIRED_REPORT)
            else:
                kernel_csv, attempt, checked = _run_report(
                    source, output_dir, REQUIRED_REPORT
                )
                manifest["report_attempts"].append(attempt)
                manifest["checked_paths"][REQUIRED_REPORT] = [
                    str(path) for path in checked
                ]
                if kernel_csv and attempt["status"] == "succeeded":
                    manifest["generated_reports"].append(REQUIRED_REPORT)
                else:
                    manifest["failed_reports"].append(REQUIRED_REPORT)
                    kernel_csv = None
        if kernel_csv is None or not kernel_csv.is_file():
            raise _required_report_error(manifest)
        manifest["succeeded_reports"].append(REQUIRED_REPORT)

        if projection_csv:
            if not projection_csv.is_file():
                raise RuntimeError(
                    f"Optional projection CSV does not exist: {projection_csv}"
                )
            manifest["reused_reports"].append(OPTIONAL_REPORT)
            manifest["checked_paths"][OPTIONAL_REPORT] = [str(projection_csv)]
            manifest["succeeded_reports"].append(OPTIONAL_REPORT)
            projection_status = "succeeded"
        elif source:
            projection_csv, checked = _existing_report(output_dir, OPTIONAL_REPORT)
            manifest["checked_paths"][OPTIONAL_REPORT] = [str(path) for path in checked]
            if projection_csv:
                manifest["reused_reports"].append(OPTIONAL_REPORT)
                manifest["succeeded_reports"].append(OPTIONAL_REPORT)
                projection_status = "succeeded"
            elif (
                manifest.get("probe_status") == "ok"
                and OPTIONAL_REPORT not in manifest["available_reports"]
            ):
                manifest["unavailable_reports"].append(OPTIONAL_REPORT)
                projection_status = "unavailable"
            else:
                projection_csv, attempt, checked = _run_report(
                    source, output_dir, OPTIONAL_REPORT
                )
                manifest["report_attempts"].append(attempt)
                manifest["checked_paths"][OPTIONAL_REPORT] = [
                    str(path) for path in checked
                ]
                if projection_csv and attempt["status"] == "succeeded":
                    manifest["generated_reports"].append(OPTIONAL_REPORT)
                    manifest["succeeded_reports"].append(OPTIONAL_REPORT)
                    projection_status = "succeeded"
                else:
                    manifest["failed_reports"].append(OPTIONAL_REPORT)
                    projection_status = "failed"
        else:
            manifest["unavailable_reports"].append(OPTIONAL_REPORT)
            manifest["checked_paths"][OPTIONAL_REPORT] = []
            projection_status = "unavailable"

        manifest["projection_report_status"] = projection_status
        manifest["analysis_mode"] = (
            "full" if projection_status == "succeeded" else "kernel_only"
        )
        manifest["source_reports"] = [REQUIRED_REPORT]
        if projection_status == "succeeded":
            manifest["source_reports"].append(OPTIONAL_REPORT)
        return _write_outputs(
            output_dir,
            kernel_csv=kernel_csv,
            projection_csv=projection_csv,
            manifest=manifest,
        )
    except Exception as exc:
        manifest["status"] = "error"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_manifest(manifest_path, manifest)
        raise


def run_parse(args: argparse.Namespace, *, command: str) -> dict[str, Path]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    manifest = _base_manifest(command)
    manifest.update(_probe_nsys())
    kernel_csv = args.nvtx_kernel_csv.expanduser().resolve()
    projection_csv = (
        args.nvtx_projection_csv.expanduser().resolve()
        if args.nvtx_projection_csv
        else None
    )
    manifest["input"] = {"path": str(kernel_csv), "kind": "report_csvs"}
    manifest["checked_paths"] = {
        REQUIRED_REPORT: [str(kernel_csv)],
        OPTIONAL_REPORT: [str(projection_csv)] if projection_csv else [],
    }
    try:
        if not kernel_csv.is_file():
            raise _required_report_error(manifest)
        if projection_csv is not None and not projection_csv.is_file():
            raise RuntimeError(
                f"Optional projection CSV does not exist: {projection_csv}"
            )
        manifest["reused_reports"] = [REQUIRED_REPORT]
        manifest["succeeded_reports"] = [REQUIRED_REPORT]
        if projection_csv:
            manifest["reused_reports"].append(OPTIONAL_REPORT)
            manifest["succeeded_reports"].append(OPTIONAL_REPORT)
            manifest["source_reports"] = list(REQUESTED_REPORTS)
            manifest["analysis_mode"] = "full"
            manifest["projection_report_status"] = "succeeded"
        else:
            manifest["unavailable_reports"] = [OPTIONAL_REPORT]
            manifest["source_reports"] = [REQUIRED_REPORT]
            manifest["analysis_mode"] = "kernel_only"
            manifest["projection_report_status"] = "unavailable"
        return _write_outputs(
            output_dir,
            kernel_csv=kernel_csv,
            projection_csv=projection_csv,
            manifest=manifest,
        )
    except Exception as exc:
        manifest["status"] = "error"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_manifest(manifest_path, manifest)
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = shlex.join([sys.executable, *sys.argv])
    paths = (
        run_export(args, command=command)
        if args.command == "export"
        else run_parse(args, command=command)
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
