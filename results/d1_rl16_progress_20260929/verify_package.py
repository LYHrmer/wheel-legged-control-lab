"""Verify this publication's bytes without loading a policy or simulator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def verify(root: Path) -> dict:
    manifest = json.loads((root / "publication_manifest.json").read_text())
    expected = manifest["files"]
    actual_names = {str(p.relative_to(root)) for p in root.rglob("*")
                    if p.is_file() and p.name != "publication_manifest.json"}
    if actual_names != set(expected):
        raise RuntimeError("publication file set differs from manifest")
    total = 0
    for name, row in expected.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("expected regular publication file: " + name)
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise RuntimeError("publication byte identity differs: " + name)
        total += len(raw)
    related = manifest["related_documents"]
    for name, row in related.items():
        path = root.parents[1] / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("expected publication document: " + name)
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise RuntimeError("publication document identity differs: " + name)
    return {"files_verified": len(expected), "bytes_verified": total,
            "related_documents_verified": len(related),
            "engine_or_policy_loaded": False, "physics_replayed": False,
            "claim": "file integrity only; see evidence for experiment results"}

if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
