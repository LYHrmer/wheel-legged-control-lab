"""Verify publication bytes with the standard library only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "d1-main-route-execution-13-package-v1"
DOCUMENT = "docs/main_route_execution_20260929_13.md"


def _safe_name(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name:
        raise RuntimeError("package manifest has an empty or non-string path")
    relative = PurePosixPath(name)
    if (relative.is_absolute() or name != relative.as_posix()
            or any(part in (".", "..") for part in relative.parts)):
        raise RuntimeError("package manifest has an unsafe path: " + name)
    return relative


def _check_file(path: Path, expected: dict[str, Any]) -> int:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("publication file is missing or not regular: " + str(path))
    if (not isinstance(expected, dict)
            or type(expected.get("bytes")) is not int or expected["bytes"] < 0
            or not isinstance(expected.get("sha256"), str)
            or len(expected["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in expected["sha256"])):
        raise RuntimeError("invalid byte identity for " + str(path))
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    if size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
        raise RuntimeError("publication byte identity differs: " + str(path))
    return size


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "package_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("package manifest is missing or not regular")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise RuntimeError("package manifest schema differs")
    expected = manifest.get("files")
    related = manifest.get("related_documents")
    if (not isinstance(expected, dict) or not expected
            or not isinstance(related, dict) or DOCUMENT not in related):
        raise RuntimeError("package manifest lacks its file or document closure")

    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("publication contains a symlink: " + str(path))
        if path.is_file() and path != manifest_path:
            actual.add(path.relative_to(root).as_posix())
    if actual != set(expected):
        raise RuntimeError("publication file set differs from package manifest")

    total = 0
    for name, identity in expected.items():
        relative = _safe_name(name)
        if relative.as_posix() == "package_manifest.json":
            raise RuntimeError("package manifest must exclude itself")
        total += _check_file(root / relative, identity)
    repository = root.parents[1]
    for name, identity in related.items():
        total += _check_file(repository / _safe_name(name), identity)
    return {"files_verified": len(expected),
            "related_documents_verified": len(related),
            "bytes_verified": total,
            "claim": "file identity only; no source import, model or physics"}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
