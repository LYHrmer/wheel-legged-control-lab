#!/usr/bin/env python3
"""Deduplicate original files reversibly; preserve all per-file SHA256 values."""
import argparse
import hashlib
import json
import lzma
import shutil
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.input.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "blobs").mkdir()
    files, blobs, payload_to_blob = {}, {}, {}
    paths = [path for path in sorted(root.rglob("*")) if path.is_file()
             and "__pycache__" not in path.relative_to(root).parts]
    for index, path in enumerate(paths):
        raw = path.read_bytes()
        record = {"bytes": len(raw), "sha256": sha(raw)}
        payload = raw
        if path.name == "source.tar.gz":
            if raw[:3] != b"\x1f\x8b\x08":
                raise ValueError("source archive is not gzip")
            record["gzip_mtime_hex"] = raw[4:8].hex()
            payload = raw[:4]+b"\0"*4+raw[8:]
        payload_hash = sha(payload)
        if payload_hash not in payload_to_blob:
            stored, codec = payload, "raw"
            if len(payload) > 256 and path.suffix not in (".npz", ".gz"):
                candidate = lzma.compress(payload, preset=6)
                if len(candidate) < len(payload):
                    stored, codec = candidate, "xz"
            name = f"blobs/{payload_hash}.{codec}"
            with (output / name).open("xb") as stream:
                stream.write(stored)
            blobs[name] = {"codec": codec, "stored_bytes": len(stored),
                           "stored_sha256": sha(stored), "payload_sha256": payload_hash,
                           "payload_bytes": len(payload)}
            payload_to_blob[payload_hash] = name
        record["blob"] = payload_to_blob[payload_hash]
        files[str(path.relative_to(root))] = record
        if index % 100 == 0:
            print(json.dumps({"files_processed": index+1, "files_total": len(paths)}), flush=True)
    result = {"schema": "d1-lossless-evidence-bundle-v1", "files": files, "blobs": blobs,
              "recorded_runs": sum(path.name == "states.npz" for path in paths),
              "original_bytes": sum(row["bytes"] for row in files.values()),
              "blob_bytes": sum(row["stored_bytes"] for row in blobs.values()),
              "normalization": "Only source.tar.gz header mtime bytes4:8 are zeroed for deduplication. Exact original bytes are restored from gzip_mtime_hex; original SHA256 validates the result. All other payloads are exact original bytes, optionally XZ compressed.",
              "excluded": "Python __pycache__ only; all 49 raw run directories, including original manifests/models/source archives, are retained."}
    dump(output / "evidence_index.json", result)
    shutil.copyfile(root / "restore_evidence.py", output / "restore_evidence.py")
    print(json.dumps({key: result[key] for key in ("recorded_runs", "original_bytes", "blob_bytes")}), flush=True)


if __name__ == "__main__":
    main()
