"""Small, dependency-free helpers for recording experiment source provenance."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def capture_git_provenance(
    working_directory: Path,
) -> dict[str, str | bool | None]:
    """Capture HEAD plus a dirty-worktree fingerprint at experiment start."""

    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=working_directory,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "status", "--porcelain=v1"),
            cwd=working_directory,
            check=True,
            capture_output=True,
        ).stdout
        tracked_diff = subprocess.run(
            ("git", "diff", "--binary", "HEAD", "--"),
            cwd=working_directory,
            check=True,
            capture_output=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "git_commit": None,
            "git_dirty": None,
            "git_worktree_sha256": None,
        }
    dirty = bool(status)
    fingerprint = hashlib.sha256(status + b"\0" + tracked_diff).hexdigest() if dirty else None
    return {
        "git_commit": commit or None,
        "git_dirty": dirty,
        "git_worktree_sha256": fingerprint,
    }
