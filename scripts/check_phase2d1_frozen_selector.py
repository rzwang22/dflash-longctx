#!/usr/bin/env python3
"""Strict CPU-only checker for a Phase 2D.1B frozen selector directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsonschema import Draft202012Validator

from dflash.phase2d1.runtime import validate_frozen_selector


def check_frozen_selector(directory: Path) -> dict:
    root = directory.expanduser().resolve()
    config_path = root / "frozen-selector.json"
    manifest_path = root / "frozen-selector-manifest.json"
    if not config_path.is_file() or not manifest_path.is_file():
        raise ValueError("frozen selector directory lacks config or manifest")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_frozen_selector(config)
    schema = json.loads(
        (REPO_ROOT / "docs/schemas/phase2d1b-frozen-selector.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(config)
    if manifest.get("status") != "ok" or manifest.get("config_sha256") != config["config_sha256"]:
        raise ValueError("frozen selector manifest/config mismatch")
    if any(manifest.get("framework_modules_loaded", {}).values()):
        raise ValueError("frozen selector was not produced framework-free")
    if len(config["training_source_ids"]) != 14:
        raise ValueError("validated Phase 2D.1A freeze must contain all 14 sources")
    return {
        "status": "pass",
        "config_sha256": config["config_sha256"],
        "training_source_count": len(config["training_source_ids"]),
        "node_budgets": sorted(map(int, config["node_budget_rules"])),
        "target_blind": config["target_blind"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-selector-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(check_frozen_selector(args.frozen_selector_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
