"""Reassemble the original gzip bytes into a new path, validating every SHA256."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New, nonexistent output path")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent / "training/training_trace_parts"
    manifest = json.loads((folder / "parts.json").read_text())
    for item in manifest["parts"]:
        data = (folder / item["file"]).read_bytes()
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"Invalid archive part: {item['file']}")
    total = 0
    digest = hashlib.sha256()
    with args.output.open("xb") as target:
        for item in manifest["parts"]:
            with (folder / item["file"]).open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
    if total != manifest["original_bytes"] or digest.hexdigest() != manifest["original_sha256"]:
        raise ValueError("Reconstructed checksum mismatch; output retained for diagnosis")
    print(json.dumps({"output": str(args.output), "bytes": total, "sha256": digest.hexdigest()}))


if __name__ == "__main__":
    main()
