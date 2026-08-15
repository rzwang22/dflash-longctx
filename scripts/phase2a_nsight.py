"""Export Nsight Systems reports and build a transparent kernel breakdown."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dflash.phase2a.nsight import (
    read_nvtx_kernel_summary,
    read_nvtx_projection_summary,
    write_nsight_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export", help="Run official nsys stats reports")
    export.add_argument("--nsys-rep", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    parse = subparsers.add_parser("parse", help="Parse existing report CSV files")
    parse.add_argument("--nvtx-kernel-csv", type=Path, required=True)
    parse.add_argument("--nvtx-projection-csv", type=Path, required=True)
    parse.add_argument("--output-dir", type=Path, required=True)
    return parser


def _run_export(report: Path, output_dir: Path) -> tuple[Path, Path]:
    report = report.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not report.is_file():
        raise RuntimeError(f"Nsight report does not exist: {report}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "nsight"
    command = [
        "nsys",
        "stats",
        "--force-export=true",
        "--force-overwrite=true",
        "--report",
        "nvtx_kern_sum",
        "--report",
        "nvtx_gpu_proj_sum",
        "--format",
        "csv",
        "--output",
        str(prefix),
        str(report),
    ]
    print(shlex.join(command))
    subprocess.run(command, check=True)
    kernel_candidates = sorted(output_dir.glob("*nvtx_kern_sum*.csv"))
    projection_candidates = sorted(output_dir.glob("*nvtx_gpu_proj_sum*.csv"))
    if len(kernel_candidates) != 1 or len(projection_candidates) != 1:
        raise RuntimeError(
            "Could not uniquely locate nsys stats CSVs; run `nsys stats --help` "
            "and inspect the output directory"
        )
    return kernel_candidates[0], projection_candidates[0]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export":
        kernel_csv, projection_csv = _run_export(args.nsys_rep, args.output_dir)
    else:
        kernel_csv = args.nvtx_kernel_csv.expanduser().resolve()
        projection_csv = args.nvtx_projection_csv.expanduser().resolve()
    kernel_rows = read_nvtx_kernel_summary(kernel_csv)
    nvtx_rows = read_nvtx_projection_summary(projection_csv)
    paths = write_nsight_outputs(
        args.output_dir.expanduser().resolve(),
        kernel_rows=kernel_rows,
        nvtx_rows=nvtx_rows,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
