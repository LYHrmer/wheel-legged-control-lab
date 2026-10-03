"""Copy a byte-checked C32/C33 research subset; never run a model or reader."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat


MAX_FILE = 10 * 1024 * 1024
PUBLIC_SUFFIXES = {".py", ".json", ".md", ".log", ".txt"}
RAW_SUFFIXES = {".npz", ".npy", ".gz", ".jsonl", ".pt", ".pth",
                ".ckpt", ".safetensors", ".onnx", ".bin", ".h5",
                ".hdf5", ".parquet"}
TEMP_PARTS = {"tmp", "pytest_tmp"}


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def inventory(source_root: Path):
    files, nodes = [], []
    for section in ("continuation32", "continuation33"):
        root = source_root/section
        if not root.is_dir() or root.is_symlink():
            raise FileNotFoundError("missing or linked source section: "+str(root))
        for current, folders, names in os.walk(root, followlinks=False):
            current_path = Path(current)
            for folder in sorted(folders):
                path = current_path/folder
                if path.is_symlink():
                    nodes.append({"path": path.relative_to(source_root).as_posix(),
                                  "reason": "symbolic_link_not_followed", "kind": "directory"})
            folders[:] = sorted(folder for folder in folders
                                if not (current_path/folder).is_symlink())
            for name in sorted(names):
                path = current_path/name
                relative = path.relative_to(source_root).as_posix()
                mode = path.lstat().st_mode
                if stat.S_ISREG(mode):
                    files.append({"path": relative, **identity(path)})
                else:
                    nodes.append({"path": relative,
                                  "reason": "symbolic_link_not_followed" if stat.S_ISLNK(mode)
                                      else "non_regular_file", "kind": "file"})
    return sorted(files, key=lambda row: row["path"]), sorted(nodes, key=lambda row: row["path"])


def reason(path_text: str, size: int) -> str | None:
    path = Path(path_text)
    name = path.name.lower()
    parts = {part.lower() for part in path.parts}
    critical = "failure" in name or "receipt" in name
    if "__pycache__" in parts or name.endswith(".pyc"):
        return "python_bytecode_cache"
    if parts & TEMP_PARTS and not (critical or path.suffix.lower() == ".log"):
        return "temporary_fixture"
    if path.suffix.lower() in RAW_SUFFIXES or name.endswith(".jsonl.gz"):
        if critical:
            raise ValueError("failure/receipt is a raw replay payload: "+path_text)
        return "raw_replay_or_weight_payload"
    if path.suffix.lower() not in PUBLIC_SUFFIXES:
        if critical:
            raise ValueError("failure/receipt has unsupported type: "+path_text)
        return "not_public_evidence_type"
    if size > MAX_FILE:
        mandatory = (critical or path.suffix.lower() in {".py", ".log"}
                     or name in {"report.json", "coefficients.json", "predictions.json"}
                     or name.startswith("final_review"))
        if mandatory:
            raise ValueError("mandatory public evidence exceeds 10 MiB: "+path_text)
        return "per_file_10MiB_limit"
    return None


def copy_exact(source_root: Path, output: Path, record: dict) -> dict:
    relative = Path(record["path"])
    source = source_root/relative
    if not stat.S_ISREG(source.lstat().st_mode) or identity(source) != {
            "bytes": record["bytes"], "sha256": record["sha256"]}:
        raise RuntimeError("source changed before copy: "+record["path"])
    target = output/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1 << 20)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if identity(target) != {"bytes": record["bytes"], "sha256": record["sha256"]}:
        raise IOError("copied bytes differ: "+record["path"])
    return {"source": record["path"], "destination": record["path"],
            "bytes": record["bytes"], "sha256": record["sha256"]}


def write_new(path: Path, payload: bytes):
    if len(payload) > MAX_FILE:
        raise ValueError("generated publication file exceeds 10 MiB: "+path.name)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if identity(path) != {"bytes": len(payload),
                          "sha256": hashlib.sha256(payload).hexdigest()}:
        raise IOError("generated publication file differs: "+path.name)


def pack(source_root: Path, output: Path):
    source_root = source_root.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if any(output.resolve().is_relative_to(source_root/section)
           for section in ("continuation32", "continuation33")):
        raise ValueError("output cannot be inside the source sections")
    local, special = inventory(source_root)
    selected, skipped = [], list(special)
    for record in local:
        why = reason(record["path"], record["bytes"])
        if why is None:
            selected.append(record)
        else:
            skipped.append({**record, "reason": why})
    if not selected:
        raise RuntimeError("no public research files selected")
    selected_names = {record["path"] for record in selected}
    for section in ("continuation32", "continuation33"):
        if not any(name.startswith(section + "/") for name in selected_names):
            raise RuntimeError("one source section has no public evidence: "+section)
    receipt = {
        "schema": "d1-c32-c33-publication-subset-v1",
        "source_root": str(source_root), "output_root": str(output),
        "included_sections": ["continuation32", "continuation33"],
        "max_file_bytes": MAX_FILE,
        "local_archive_inventory": local,
        "skipped": sorted(skipped, key=lambda row: row["path"]),
        "copied_files": [{"source": row["path"], "destination": row["path"],
                          "bytes": row["bytes"], "sha256": row["sha256"]}
                         for row in selected],
        "copied_bytes": sum(row["bytes"] for row in selected),
        "new_policy_checkpoints_included": False,
        "policy_checkpoint_files_included": False,
        "offline_regression_coefficients_included": True,
        "portable_runtime_claim": False,
        "complete_reader_replay_from_public_subset": False,
        "full_raw_archive_remains_local": True,
        "scope_note": "Byte-exact source, reports, logs and receipts only. Full controls, native rows, states and other replay payloads remain in the local archive. This subset cannot independently run the complete saved reader.",
    }
    manifest = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    note = ("# C32/C33 research evidence subset\n\n"
            "This subset contains byte-exact sources, reports, logs and receipts. "
            "It adds no policy checkpoints and is not a portable runtime package. "
            "C32 offline regression coefficients are included as diagnostic evidence. "
            "The full controls, native rows, states and replay payloads remain "
            "in the local archive. The subset cannot independently reproduce "
            "the complete saved-data reader. See `publication_manifest32_33.json` "
            "for every local file hash and each exclusion reason.\n").encode()
    if len(manifest) > MAX_FILE:
        raise ValueError("publication manifest exceeds 10 MiB")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o755, exist_ok=False)
    for record in selected:
        copy_exact(source_root, output, record)
    write_new(output/"README_publication32_33.md", note)
    write_new(output/"publication_manifest32_33.json", manifest)
    for record in local:
        path = source_root/record["path"]
        if (not stat.S_ISREG(path.lstat().st_mode) or
                identity(path) != {"bytes": record["bytes"],
                                   "sha256": record["sha256"]}):
            raise RuntimeError("local archive changed during packaging: "+record["path"])
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = pack(args.source_root, args.output)
    print(json.dumps({"output": str(args.output),
                      "copied_files": len(receipt["copied_files"]),
                      "skipped": len(receipt["skipped"])}))


if __name__ == "__main__":
    main()
