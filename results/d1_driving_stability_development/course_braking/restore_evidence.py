#!/usr/bin/env python3
"""Verify or restore every original byte in the lossless evidence bundle."""
import argparse
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe bundle path")
    return Path(*path.parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify-only", action="store_true")
    action.add_argument("--output", type=Path, help="New directory; restoring duplicates needs about 4.1 GiB")
    args = parser.parse_args()
    index = json.loads((args.bundle / "evidence_index.json").read_text())
    if index["schema"] != "d1-lossless-evidence-bundle-v1":
        raise ValueError("unknown evidence format")
    if args.output is not None:
        if args.output.exists() or any(p.is_symlink() for p in (args.output, *args.output.parents)):
            raise FileExistsError("output must be new and must not use symlinks")
        args.output.mkdir(parents=True, exist_ok=False)
    cached, total = {}, 0
    for name, metadata in index["files"].items():
        blob_name = metadata["blob"]
        if blob_name not in cached:
            blob_metadata = index["blobs"][blob_name]
            packed = (args.bundle / safe_relative(blob_name)).read_bytes()
            if sha(packed) != blob_metadata["stored_sha256"]:
                raise ValueError(f"stored blob hash mismatch: {blob_name}")
            payload = lzma.decompress(packed) if blob_metadata["codec"] == "xz" else packed
            if sha(payload) != blob_metadata["payload_sha256"]:
                raise ValueError(f"decoded payload hash mismatch: {blob_name}")
            cached[blob_name] = payload
        raw = cached[blob_name]
        if "gzip_mtime_hex" in metadata:
            raw = raw[:4]+bytes.fromhex(metadata["gzip_mtime_hex"])+raw[8:]
        if len(raw) != metadata["bytes"] or sha(raw) != metadata["sha256"]:
            raise ValueError(f"original file hash mismatch: {name}")
        if args.output is not None:
            destination = args.output / safe_relative(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(raw)
        total += len(raw)
    print(json.dumps({"all_original_files_verified": True, "files": len(index["files"]),
                      "original_bytes": total, "recorded_runs": index["recorded_runs"],
                      "restored": args.output is not None}))


if __name__ == "__main__":
    main()
