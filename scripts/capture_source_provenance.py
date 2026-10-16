"""Capture or verify the source used by a future experiment, without running it.

Scope: Git-tracked (including deleted) and nonignored untracked .py/.pyi/.xml/
.urdf/.toml files under src/, scripts/, examples/, plus root pyproject.toml.
Results, runtime bundles, binary meshes, dependencies, models and environment
versions need their own manifests. Symlinks record link text, never target bytes.
Use quiescent sources: inventory/stat checks detect ordinary concurrent changes,
but this is not an atomic snapshot or a proof of authenticity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

SCHEMA = "source-provenance-v1"
DIRECTORIES = ("src", "scripts", "examples")
EXTENSIONS = (".py", ".pyi", ".xml", ".urdf", ".toml")
SCOPE = {"directories": list(DIRECTORIES), "extensions": list(EXTENSIONS),
         "root_files": ["pyproject.toml"], "untracked": "nonignored only",
         "symlinks": "link text only; parent symlinks rejected"}


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args],
                          check=True, capture_output=True, timeout=30).stdout


def inventory(root: Path) -> list[str]:
    names = git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard",
                "--", *DIRECTORIES, "pyproject.toml").split(b"\0")
    return sorted({os.fsdecode(name) for name in names if name and (
        name == b"pyproject.toml" or Path(os.fsdecode(name)).suffix in EXTENSIONS)})


def signature(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def source_entry(root: Path, name: str) -> tuple[dict, tuple[int, ...] | None]:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe Git source path")
    for parent in relative.parents:
        if (root / parent).is_symlink():
            raise ValueError(f"source parent is a symlink: {name}")
    path = root / name
    try:
        before = signature(path.lstat())
    except FileNotFoundError:
        return {"path": name, "type": "missing"}, None
    if stat.S_ISLNK(before[2]):
        target = os.readlink(path)
        entry = {"path": name, "type": "symlink", "target": target,
                 "sha256": hashlib.sha256(os.fsencode(target)).hexdigest()}
    elif stat.S_ISREG(before[2]):
        digest = hashlib.sha256()
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        with os.fdopen(os.open(path, flags), "rb") as stream:
            if signature(os.fstat(stream.fileno())) != before:
                raise ValueError(f"source changed before reading: {name}")
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            if signature(os.fstat(stream.fileno())) != before:
                raise ValueError(f"source changed while reading: {name}")
        entry = {"path": name, "type": "file", "bytes": before[3], "sha256": digest.hexdigest()}
    else:
        raise ValueError(f"source is not a regular file or symlink: {name}")
    if signature(path.lstat()) != before:
        raise ValueError(f"source changed while reading: {name}")
    return entry, before


def capture(root: Path) -> dict:
    root = root.resolve(strict=True)
    actual_root = Path(os.fsdecode(git(root, "rev-parse", "--show-toplevel")).strip()).resolve()
    if actual_root != root:
        raise ValueError("--root must name the Git repository root")
    commit = git(root, "rev-parse", "HEAD").decode("ascii").strip()
    names = inventory(root)
    entries, signatures = [], {}
    for name in names:
        entry, before = source_entry(root, name)
        entries.append(entry)
        signatures[name] = before
    for name, before in signatures.items():
        try:
            after = signature((root / name).lstat())
        except FileNotFoundError:
            after = None
        if after != before:
            raise ValueError(f"source changed during capture: {name}")
    if inventory(root) != names or git(root, "rev-parse", "HEAD").decode("ascii").strip() != commit:
        raise ValueError("Git inventory or HEAD changed during capture")
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"schema": SCHEMA, "scope": SCOPE, "repository_root": str(root),
            "git_commit": commit, "git_dirty": bool(git(root, "status", "--porcelain=v1")),
            "source_sha256": hashlib.sha256(encoded).hexdigest(), "files": entries}


def verify(root: Path, manifest: Path) -> dict:
    expected = json.loads(manifest.read_text(encoding="utf-8"))
    if (not isinstance(expected, dict) or expected.get("schema") != SCHEMA
            or expected.get("scope") != SCOPE):
        raise ValueError("manifest schema or source scope differs")
    actual = capture(root)
    old = {entry["path"]: entry for entry in expected["files"]}
    new = {entry["path"]: entry for entry in actual["files"]}
    changed = sorted(name for name in old.keys() | new.keys() if old.get(name) != new.get(name))
    source_matches = expected["source_sha256"] == actual["source_sha256"] and not changed
    commit_matches = expected["git_commit"] == actual["git_commit"]
    return {"matches": source_matches and commit_matches, "source_matches": source_matches,
            "git_commit_matches": commit_matches, "changed_paths": changed,
            "expected_source_sha256": expected["source_sha256"],
            "actual_source_sha256": actual["source_sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path, help="write a new manifest outside source scope")
    action.add_argument("--verify", type=Path, help="compare sources and HEAD to a saved manifest")
    args = parser.parse_args(argv)
    try:
        if args.verify is not None:
            result = verify(args.root, args.verify)
            print(json.dumps(result, sort_keys=True))
            return 0 if result["matches"] else 1
        root = args.root.resolve(strict=True)
        output = args.output.absolute()
        if (output.resolve() == root / "pyproject.toml" or any(
                output.resolve().is_relative_to(root / name) for name in DIRECTORIES)):
            raise ValueError("--output must be outside the source directories and root pyproject.toml")
        if output.exists() or output.is_symlink():
            raise ValueError("--output must name a new file")
        record = capture(root)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(json.dumps({"manifest": str(output), "files": len(record["files"]),
                          "source_sha256": record["source_sha256"]}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"source provenance: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
