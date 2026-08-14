"""Source-location and hashing helpers that do not import model frameworks."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _callable_location(callable_object: Any) -> dict[str, Any]:
    source_file = inspect.getsourcefile(callable_object) or inspect.getfile(
        callable_object
    )
    source_path = Path(source_file).resolve()
    source_lines, starting_line = inspect.getsourcelines(callable_object)
    source_text = "".join(source_lines)
    return {
        "module": getattr(callable_object, "__module__", None),
        "qualified_name": getattr(
            callable_object,
            "__qualname__",
            getattr(callable_object, "__name__", type(callable_object).__qualname__),
        ),
        "definition_file": str(source_path),
        "starting_line": int(starting_line),
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "definition_file_sha256": sha256_file(source_path),
    }


def locate_callable(callable_object: Any) -> dict[str, Any]:
    """Locate both a decorated callable and its fully unwrapped definition."""

    unwrapped = inspect.unwrap(callable_object)
    return {
        "wrapped": _callable_location(callable_object),
        "unwrapped": _callable_location(unwrapped),
        "was_unwrapped": unwrapped is not callable_object,
    }
