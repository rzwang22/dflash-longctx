"""Build a local DFlash model from pinned Git-controlled code and external weights."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .source import sha256_file

DRAFT_MODEL_ID = "z-lab/Qwen3-8B-DFlash-b16"
DRAFT_REVISION = "9b41424b7109f9c5413454f481b09a82b85333f4"
REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = REPO_ROOT / "third_party" / "qwen3_8b_dflash_b16"
ORIGINAL_ROOT = VENDOR_ROOT / "original"
INSTRUMENTED_ROOT = VENDOR_ROOT / "instrumented"
BASE_FILES = ("config.json", "dflash.py", "modeling_dflash.py", "utils.py")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _metadata_lines(model_path: Path, filename: str) -> list[str]:
    metadata_root = model_path / ".cache" / "huggingface" / "download"
    candidates = sorted(metadata_root.rglob(f"{filename}.metadata"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one Hugging Face metadata file for {filename} under "
            f"{metadata_root}, found {len(candidates)}"
        )
    return candidates[0].read_text(encoding="utf-8").splitlines()


def _validate_repository_sources(provenance: dict[str, Any]) -> None:
    for filename, expected in provenance["original_files"].items():
        actual = sha256_file(ORIGINAL_ROOT / filename)
        if actual != expected["vendored_sha256"]:
            raise RuntimeError(
                f"Vendored {filename} hash {actual} does not match provenance "
                f"{expected['vendored_sha256']}"
            )


def _validate_weight_source(source_model_path: Path) -> dict[str, Any]:
    source_model_path = source_model_path.expanduser().resolve()
    if not source_model_path.is_dir():
        raise RuntimeError(f"Draft weight source does not exist: {source_model_path}")

    provenance = _read_json(VENDOR_ROOT / "provenance.json")
    _validate_repository_sources(provenance)
    if (
        provenance["model_id"] != DRAFT_MODEL_ID
        or provenance["revision"] != DRAFT_REVISION
    ):
        raise RuntimeError("Vendored provenance does not match the pinned draft model")

    for filename, expected in provenance["original_files"].items():
        source_file = source_model_path / filename
        if not source_file.is_file():
            raise RuntimeError(
                f"Draft source is missing {filename}: {source_model_path}"
            )
        actual = sha256_file(source_file)
        if actual != expected["upstream_sha256"]:
            raise RuntimeError(
                f"Draft source {filename} hash {actual} does not match pinned upstream "
                f"hash {expected['upstream_sha256']}"
            )

    metadata_root = source_model_path / ".cache" / "huggingface" / "download"
    revisions = set()
    for metadata_file in metadata_root.rglob("*.metadata"):
        lines = metadata_file.read_text(encoding="utf-8").splitlines()
        if lines and len(lines[0]) == 40:
            revisions.add(lines[0].lower())
    if revisions != {DRAFT_REVISION}:
        raise RuntimeError(
            f"Draft source metadata revisions {sorted(revisions)} do not equal {DRAFT_REVISION}"
        )

    expected_weights = provenance["weight_files_not_vendored"]
    actual_weight_names = {
        path.name
        for path in source_model_path.iterdir()
        if path.is_file() and path.suffix in {".safetensors", ".bin"}
    }
    if actual_weight_names != set(expected_weights):
        raise RuntimeError(
            f"Draft weight files {sorted(actual_weight_names)} do not match pinned files "
            f"{sorted(expected_weights)}"
        )

    weights = []
    for filename, expected in expected_weights.items():
        source_file = source_model_path / filename
        size = source_file.stat().st_size
        if size != expected["size_bytes"]:
            raise RuntimeError(
                f"Weight {filename} size {size} does not match pinned size "
                f"{expected['size_bytes']}"
            )
        with source_file.open("rb") as weight_stream:
            if weight_stream.read(64).startswith(b"version https://git-lfs"):
                raise RuntimeError(
                    f"Weight {filename} is an unresolved Git LFS pointer"
                )
        metadata = _metadata_lines(source_model_path, filename)
        if not metadata or metadata[0].lower() != DRAFT_REVISION:
            raise RuntimeError(
                f"Weight {filename} metadata is not pinned to {DRAFT_REVISION}"
            )
        etag = metadata[1].strip('"') if len(metadata) > 1 else None
        if etag != expected["git_lfs_oid_sha256"]:
            raise RuntimeError(
                f"Weight {filename} metadata etag {etag!r} does not match pinned LFS oid"
            )
        weights.append(
            {
                "filename": filename,
                "source_path": str(source_file),
                "size_bytes": size,
                "metadata_revision": metadata[0].lower(),
                "metadata_etag": etag,
                "expected_content_sha256_from_lfs_oid": expected["git_lfs_oid_sha256"],
                "full_weight_hash_recomputed": False,
            }
        )
    return {
        "source_model_path": str(source_model_path),
        "weights": weights,
        "upstream_source_hashes": {
            name: details["upstream_sha256"]
            for name, details in provenance["original_files"].items()
        },
    }


def _effective_sources(implementation: str) -> dict[str, Path]:
    if implementation not in {"original", "instrumented"}:
        raise ValueError("implementation must be 'original' or 'instrumented'")
    sources = {filename: ORIGINAL_ROOT / filename for filename in BASE_FILES}
    if implementation == "instrumented":
        sources["dflash.py"] = INSTRUMENTED_ROOT / "dflash.py"
        sources["phase1_observability.py"] = (
            INSTRUMENTED_ROOT / "phase1_observability.py"
        )
        sources["phase2c_online.py"] = REPO_ROOT / "dflash" / "phase2c" / "core.py"
    sources["LICENSE"] = VENDOR_ROOT / "LICENSE"
    return sources


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copyfile(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(serialized)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _materialization_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def materialize_draft(
    *,
    source_model_path: Path,
    output_dir: Path,
    implementation: str,
) -> dict[str, Any]:
    """Create or repair an idempotent local model directory without copying weights."""

    source_provenance = _validate_weight_source(source_model_path)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = _effective_sources(implementation)
    effective_hashes = {name: sha256_file(path) for name, path in sources.items()}
    repository_commit = _git("rev-parse", "HEAD")
    repository_branch = _git("branch", "--show-current")
    repository_status = _git("status", "--porcelain")
    if not repository_commit or repository_status is None:
        raise RuntimeError("Could not determine repository provenance")
    repository = {
        "commit": repository_commit,
        "branch": repository_branch,
        "dirty": bool(repository_status),
    }
    weight_provenance = [
        {
            **weight,
            "materialized_path": str(output_dir / weight["filename"]),
            "symlink_target": weight["source_path"],
        }
        for weight in source_provenance["weights"]
    ]
    identity = {
        "model_id": DRAFT_MODEL_ID,
        "base_model_revision": DRAFT_REVISION,
        "implementation": implementation,
        "repository": repository,
        "source_model_path": source_provenance["source_model_path"],
        "materialized_model_path": str(output_dir),
        "effective_custom_code_sha256": effective_hashes,
        "weight_provenance": weight_provenance,
    }
    materialization_id = _materialization_id(identity)
    manifest_path = output_dir / "materialization.json"
    existing = _read_json(manifest_path) if manifest_path.is_file() else None
    if existing and (
        existing.get("implementation") != implementation
        or existing.get("model_id") != DRAFT_MODEL_ID
        or existing.get("base_model_revision") != DRAFT_REVISION
        or existing.get("source_model_path") != source_provenance["source_model_path"]
    ):
        raise RuntimeError(
            "Existing materialization has a different implementation or source; "
            "choose a separate output directory"
        )
    created_at = (
        existing["created_at_utc"]
        if existing and existing.get("materialization_id") == materialization_id
        else datetime.now(timezone.utc).isoformat()
    )

    for filename, source in sources.items():
        _atomic_copy(source, output_dir / filename)
        if sha256_file(output_dir / filename) != effective_hashes[filename]:
            raise RuntimeError(f"Materialized source hash mismatch for {filename}")

    for weight in weight_provenance:
        source = Path(weight["source_path"])
        destination = output_dir / weight["filename"]
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise RuntimeError(
                    f"Existing weight symlink points elsewhere: {destination}"
                )
        elif destination.exists():
            raise RuntimeError(
                f"Refusing to replace non-symlink weight at {destination}"
            )
        else:
            destination.symlink_to(source.resolve())

    manifest = {
        "schema_version": "1.0",
        "materialization_id": materialization_id,
        "created_at_utc": created_at,
        **identity,
        "upstream_custom_code_sha256": source_provenance["upstream_source_hashes"],
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def load_materialization(path: Path) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve() / "materialization.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"No Phase 1A materialization manifest: {manifest_path}")
    return _read_json(manifest_path)


def verify_materialization(
    path: Path, *, expected_implementation: str | None = None
) -> dict[str, Any]:
    model_path = path.expanduser().resolve()
    manifest = load_materialization(model_path)
    if manifest.get("model_id") != DRAFT_MODEL_ID:
        raise RuntimeError(
            "Materialized draft model ID does not match the pinned model"
        )
    if manifest.get("base_model_revision") != DRAFT_REVISION:
        raise RuntimeError(
            "Materialized draft revision does not match the pinned revision"
        )
    if (
        expected_implementation
        and manifest.get("implementation") != expected_implementation
    ):
        raise RuntimeError(
            f"Expected {expected_implementation} materialization, found "
            f"{manifest.get('implementation')}"
        )
    identity_keys = (
        "model_id",
        "base_model_revision",
        "implementation",
        "repository",
        "source_model_path",
        "materialized_model_path",
        "effective_custom_code_sha256",
        "weight_provenance",
    )
    identity = {key: manifest[key] for key in identity_keys}
    if _materialization_id(identity) != manifest.get("materialization_id"):
        raise RuntimeError("Materialization manifest identity hash is invalid")
    current_repository_hashes = {
        name: sha256_file(source)
        for name, source in _effective_sources(manifest["implementation"]).items()
    }
    if current_repository_hashes != manifest["effective_custom_code_sha256"]:
        raise RuntimeError(
            "Materialized custom code does not match the current repository; rerun "
            "scripts/materialize_dflash.py"
        )
    for filename, expected_hash in manifest["effective_custom_code_sha256"].items():
        actual_hash = sha256_file(model_path / filename)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Materialized {filename} hash {actual_hash} does not match {expected_hash}"
            )
    for weight in manifest["weight_provenance"]:
        materialized = model_path / weight["filename"]
        if not materialized.is_symlink():
            raise RuntimeError(f"Materialized weight is not a symlink: {materialized}")
        if materialized.resolve() != Path(weight["source_path"]).resolve():
            raise RuntimeError(f"Materialized weight target changed: {materialized}")
        if materialized.stat().st_size != weight["size_bytes"]:
            raise RuntimeError(f"Materialized weight size changed: {materialized}")
    return manifest
