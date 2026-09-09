"""Package selected result files locally; never upload, delete, or overwrite inputs.

Paths are relative to this repository, not the caller's working directory. The
manifest is the completion marker: without it, an output is incomplete. A
failed output is retained for inspection; retry with another new directory.
Run only against quiescent results. Stat checks detect ordinary concurrent
changes, but do not provide a filesystem snapshot against a hostile writer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIRECTORIES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache"}
CACHE_SUFFIXES = {".pyc", ".pyo"}
CHUNK_BYTES = 1024 * 1024
ARCHIVE_NAME = "result_artifacts.tar.gz"
MANIFEST_NAME = "archive_manifest.json"


def _absolute(value: str | Path, root: Path) -> Path:
    path = Path(value)
    if ".." in path.parts or any(ord(char) < 32 or char == "\\" for char in str(path)):
        raise ValueError(f"unsafe path: {value!s}")
    return path if path.is_absolute() else root / path


def _no_symlinks(path: Path) -> None:
    for component in (*reversed(path.parents), path):
        if component.is_symlink():
            raise ValueError(f"symlink is not allowed: {component}")


def _signature(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def collect_inputs(root: Path, paths: list[str | Path]) -> dict:
    """Enumerate regular files, including failures and hidden metadata, without reading bytes."""
    results = root / "results"
    selected = []
    for value in paths:
        path = _absolute(value, root)
        if path == results or not path.is_relative_to(results):
            raise ValueError(f"select an explicit file/directory below results/: {value}")
        _no_symlinks(path)
        path.lstat()  # Missing targets must fail, including dangling symlinks.
        if any(path.is_relative_to(other) or other.is_relative_to(path) for other in selected):
            raise ValueError(f"duplicate or overlapping selection: {value}")
        selected.append(path)
    if not selected:
        raise ValueError("at least one results path is required")
    members, excluded = [], []

    def visit(path: Path) -> None:
        info = path.lstat()
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise ValueError(f"only regular files/directories are allowed: {relative}")
        if any(ord(char) < 32 or char == "\\" for char in relative):
            raise ValueError(f"unsafe archive member name: {relative!r}")
        if stat.S_ISDIR(info.st_mode):
            # Inspect cached subtrees too: a symlink or FIFO is never silently accepted.
            for child in sorted(path.iterdir()):
                visit(child)
        elif set(path.relative_to(results).parts[:-1]) & CACHE_DIRECTORIES or (
            path.suffix in CACHE_SUFFIXES
        ):
            excluded.append(relative)
        else:
            members.append({"path": relative, "bytes": info.st_size, "signature": _signature(info)})

    for path in sorted(selected):
        visit(path)
    if not members:
        raise ValueError("selection contains no non-cache regular files")
    return {
        "selected_paths": sorted(path.relative_to(root).as_posix() for path in selected),
        "members": sorted(members, key=lambda item: item["path"]),
        "excluded_cache_files": sorted(excluded),
    }


def _git_metadata(root: Path) -> dict:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("repository root does not match this tool's repository")
    return {
        "packaging_git_head": git("rev-parse", "--verify", "HEAD^{commit}"),
        "packaging_worktree_dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
        "commit_scope": "HEAD at packaging time, NOT the experiment-generating source commit",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


class _HashingReader:
    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int) -> bytes:
        block = self.handle.read(size)
        self.digest.update(block)
        self.bytes_read += len(block)
        return block


def _add_member(archive: tarfile.TarFile, root: Path, member: dict) -> dict:
    path = root / member["path"]
    _no_symlinks(path)
    # O_NOFOLLOW prevents a final-component symlink swap between validation and open.
    # Nonblocking open also prevents a concurrent FIFO replacement from hanging.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        if _signature(os.fstat(handle.fileno())) != member["signature"]:
            raise ValueError(f"input changed before packing: {member['path']}")
        entry = tarfile.TarInfo(member["path"])
        entry.size = member["bytes"]
        entry.mode = stat.S_IMODE(member["signature"][2]) & 0o777
        entry.mtime = member["signature"][4] // 1_000_000_000
        reader = _HashingReader(handle)
        archive.addfile(entry, reader)
        if (
            reader.bytes_read != member["bytes"]
            or handle.read(1)
            or _signature(os.fstat(handle.fileno())) != member["signature"]
        ):
            raise ValueError(f"input changed during packing: {member['path']}")
    return {"path": member["path"], "bytes": reader.bytes_read, "sha256": reader.digest.hexdigest()}


def package(paths: list[str | Path], output: str | Path, *, dry_run=False, root=REPO_ROOT) -> dict:
    root = Path(root).resolve()
    output = _absolute(output, root)
    _no_symlinks(output)
    if output.exists():
        raise ValueError(f"output must be a new directory: {output}")
    selection = collect_inputs(root, paths)
    if any(output.is_relative_to(root / selected) for selected in selection["selected_paths"]):
        raise ValueError("output must not be inside a selected input")
    plan = {
        "schema": "result-artifact-package-v1",
        "status": "dry-run" if dry_run else "complete",
        "packaging_started_at_utc": datetime.now(timezone.utc).isoformat(),
        **_git_metadata(root),
        "selected_paths": selection["selected_paths"],
        "member_count": len(selection["members"]),
        "total_input_bytes": sum(member["bytes"] for member in selection["members"]),
        "members": [
            {key: value for key, value in member.items() if key != "signature"}
            for member in selection["members"]
        ],
        "excluded_cache_files": selection["excluded_cache_files"],
        "cache_policy": {
            "directory_names": sorted(CACHE_DIRECTORIES),
            "file_suffixes": sorted(CACHE_SUFFIXES),
        },
        "audit_scope": {
            "includes": "all selected non-cache regular files, including failed experiments",
            "checks": "path/type checks; streamed member SHA256; size/stat checks before/after packing",
            "does_not_verify": "scientific metrics, experiment provenance, or authenticity",
            "snapshot_limit": "quiescent inputs required; ordinary changes detected, no atomic filesystem snapshot",
            "empty_directories": "not represented; tar members are regular files only",
        },
    }
    if dry_run:
        plan["audit_scope"]["checks"] = "path/type checks and stat sizes only; payload bytes not read"
        return plan

    output.mkdir(parents=True, exist_ok=False)
    archive_partial = output / (ARCHIVE_NAME + ".partial")
    with tarfile.open(archive_partial, "w:gz", compresslevel=6) as archive:
        plan["members"] = [_add_member(archive, root, member) for member in selection["members"]]
    if collect_inputs(root, paths) != selection:
        raise ValueError("input selection changed during packing; output is incomplete")
    plan.update(
        {
            "packaged_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive": {
                "path": ARCHIVE_NAME,
                "bytes": archive_partial.stat().st_size,
                "sha256": sha256_file(archive_partial),
            },
        }
    )
    manifest_partial = output / (MANIFEST_NAME + ".partial")
    with manifest_partial.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    sums_partial = output / "SHA256SUMS.partial"
    with sums_partial.open("x", encoding="ascii") as handle:
        handle.write(f"{plan['archive']['sha256']}  {ARCHIVE_NAME}\n")
        handle.write(f"{sha256_file(manifest_partial)}  {MANIFEST_NAME}\n")
    archive_partial.rename(output / ARCHIVE_NAME)
    sums_partial.rename(output / "SHA256SUMS")
    # Publish the completion marker last. Failure leaves no completed manifest.
    manifest_partial.rename(output / MANIFEST_NAME)
    return plan


def main(argv=None, *, root=REPO_ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths", nargs="+", required=True, help="files/directories below results/"
    )
    parser.add_argument("--output", type=Path, required=True, help="new local output directory")
    parser.add_argument(
        "--dry-run", action="store_true", help="list paths/sizes; do not create output"
    )
    args = parser.parse_args(argv)
    try:
        result = package(args.paths, args.output, dry_run=args.dry_run, root=root)
    except KeyboardInterrupt:
        print(
            "Packaging interrupted. No completed package; inspect any partial output.",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, tarfile.TarError, subprocess.SubprocessError) as exc:
        print(
            f"Packaging failed: {exc}. No completed package; inspect any partial output.",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
