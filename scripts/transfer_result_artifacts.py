"""Split/join an existing result bundle for transport; never extract or upload.

Fixed 128 MiB pieces preserve the original archive and its two metadata files
byte for byte. Work only on quiescent local inputs. Failed new outputs retain
.partial files; retry in a different directory. After joining, run the existing
verify_result_artifacts.py: transport hashes do not verify tar contents or
authenticate the experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path

ARCHIVE = "result_artifacts.tar.gz"
METADATA = ("archive_manifest.json", "SHA256SUMS")
PARTS_MANIFEST = "parts_manifest.json"
PART_BYTES = 128 * 1024 * 1024
CHUNK = 1024 * 1024
SCHEMA = "result-artifact-parts-v1"
HEX = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(value):
    path = Path(value).absolute()
    require(".." not in path.parts and "\\" not in str(path), "unsafe path")
    require(not any(p.is_symlink() for p in (*path.parents, path)), "symlink is forbidden")
    return path


def signature(path):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), f"not a regular file: {path.name}")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def directories(source, output):
    source, output = safe_path(source), safe_path(output)
    require(source.is_dir(), "input directory is missing")
    require(not output.exists(), "output must be a new directory")
    require(
        not output.is_relative_to(source) and not source.is_relative_to(output),
        "input/output directories overlap",
    )
    for path in source.iterdir():
        require(not path.name.endswith(".partial"), "unfinished .partial input")
        signature(path)
    return source, output


def read_small(path, limit=64 * CHUNK):
    before = signature(path)
    require(before[2] <= limit, f"metadata too large: {path.name}")
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    require(len(value) <= limit and signature(path) == before, "metadata changed while reading")
    return value


def strict_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def number(value):
        parsed = float(value)
        require(float("-inf") < parsed < float("inf"), "nonfinite JSON value")
        return parsed

    return json.loads(data, object_pairs_hook=unique, parse_constant=number, parse_float=number)


def metadata(source):
    raw = {
        name: read_small(source / name, 512 if name == "SHA256SUMS" else 64 * CHUNK)
        for name in METADATA
    }
    sums = {}
    for line in raw["SHA256SUMS"].decode("ascii").splitlines():
        match = re.fullmatch(
            r"([0-9a-f]{64})  (result_artifacts\.tar\.gz|archive_manifest\.json)", line
        )
        require(match is not None, "invalid SHA256SUMS line")
        digest, name = match.groups()
        require(name not in sums, "duplicate SHA256SUMS filename")
        sums[name] = digest
    require(set(sums) == {ARCHIVE, METADATA[0]}, "SHA256SUMS must list the original two files")
    require(
        hashlib.sha256(raw[METADATA[0]]).hexdigest() == sums[METADATA[0]],
        "original manifest SHA256 mismatch",
    )
    manifest = strict_json(raw[METADATA[0]])
    require(
        type(manifest) is dict
        and manifest.get("schema") == "result-artifact-package-v1"
        and manifest.get("status") == "complete",
        "original bundle is not complete",
    )
    archive = manifest.get("archive")
    require(
        type(archive) is dict
        and archive.get("path") == ARCHIVE
        and type(archive.get("bytes")) is int
        and archive["bytes"] > 0
        and archive.get("sha256") == sums[ARCHIVE],
        "invalid original archive record",
    )
    return raw, archive


def part_name(index):
    return f"{ARCHIVE}.part{index:04d}"


def copy_bytes(stream, destination, limit, digest):
    """Copy at most limit bytes in bounded blocks; EOF is checked by the caller."""
    local, count = hashlib.sha256(), 0
    while count < limit:
        block = stream.read(min(CHUNK, limit - count))
        if not block:
            break
        if destination is not None:
            destination.write(block)
        local.update(block)
        digest.update(block)
        count += len(block)
    return count, local.hexdigest()


def open_regular(path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    stream = os.fdopen(fd, "rb")
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        stream.close()
        raise ValueError(f"not a regular file: {path.name}")
    return stream


def write_partial(output, name, data):
    with (output / (name + ".partial")).open("xb") as stream:
        stream.write(data)


def split(bundle, output, *, dry_run=False):
    source, output = directories(bundle, output)
    require(
        {p.name for p in source.iterdir()} == {ARCHIVE, *METADATA},
        "bundle must contain exactly its original three files",
    )
    raw, archive = metadata(source)
    before = {name: signature(source / name) for name in (ARCHIVE, *METADATA)}
    require(before[ARCHIVE][2] == archive["bytes"], "archive size mismatch")
    if not dry_run:
        output.mkdir(parents=True)
    parts, whole = [], hashlib.sha256()
    with open_regular(source / ARCHIVE) as stream:
        remaining = archive["bytes"]
        while remaining:
            size = min(PART_BYTES, remaining)
            name = part_name(len(parts) + 1)
            if dry_run:
                count, digest = copy_bytes(stream, None, size, whole)
            else:
                with (output / (name + ".partial")).open("xb") as destination:
                    count, digest = copy_bytes(stream, destination, size, whole)
            require(count == size, "archive ended early")
            parts.append({"path": name, "bytes": size, "sha256": digest})
            remaining -= size
        require(
            not stream.read(1) and whole.hexdigest() == archive["sha256"],
            "archive SHA256/size mismatch",
        )
    require(
        all(signature(source / name) == saved for name, saved in before.items()),
        "source changed during split",
    )
    plan = {
        "schema": SCHEMA,
        "status": "dry-run" if dry_run else "complete",
        "part_bytes": PART_BYTES,
        "archive": archive,
        "parts": parts,
        "metadata": [
            {"path": name, "bytes": len(raw[name]), "sha256": hashlib.sha256(raw[name]).hexdigest()}
            for name in METADATA
        ],
    }
    if not dry_run:
        write_partial(
            output, PARTS_MANIFEST, json.dumps(plan, indent=2, allow_nan=False).encode() + b"\n"
        )
        for name, data in raw.items():
            write_partial(output, name, data)
        for name in [*(part["path"] for part in parts), *METADATA, PARTS_MANIFEST]:
            (output / (name + ".partial")).rename(output / name)
    return plan


def validate_parts(source, plan, archive, raw):
    require(
        type(plan) is dict
        and set(plan) == {"schema", "status", "part_bytes", "archive", "parts", "metadata"},
        "invalid parts manifest fields",
    )
    require(
        plan["schema"] == SCHEMA
        and plan["status"] == "complete"
        and type(plan["part_bytes"]) is int
        and plan["part_bytes"] == PART_BYTES,
        "invalid parts schema/status/size",
    )
    require(plan["archive"] == archive, "parts and original archive records differ")
    expected_meta = [
        {"path": name, "bytes": len(raw[name]), "sha256": hashlib.sha256(raw[name]).hexdigest()}
        for name in METADATA
    ]
    require(plan["metadata"] == expected_meta, "original metadata bytes/SHA differ")
    parts = plan["parts"]
    count = (archive["bytes"] + PART_BYTES - 1) // PART_BYTES
    require(type(parts) is list and len(parts) == count, "wrong part count")
    for index, part in enumerate(parts, 1):
        size = min(PART_BYTES, archive["bytes"] - (index - 1) * PART_BYTES)
        require(
            type(part) is dict and set(part) == {"path", "bytes", "sha256"}, "invalid part record"
        )
        require(part["path"] == part_name(index), "unsafe, duplicate or out-of-order part filename")
        require(type(part["bytes"]) is int and part["bytes"] == size, "invalid part size")
        require(
            type(part["sha256"]) is str and HEX.fullmatch(part["sha256"]), "invalid part SHA256"
        )
        require(signature(source / part["path"])[2] == size, "part file size mismatch")
    require(
        {p.name for p in source.iterdir()}
        == {PARTS_MANIFEST, *METADATA, *(part["path"] for part in parts)},
        "missing or unexpected transport file",
    )
    return parts


def join(directory, output):
    source, output = directories(directory, output)
    raw, archive = metadata(source)
    plan = strict_json(read_small(source / PARTS_MANIFEST))
    parts = validate_parts(source, plan, archive, raw)
    before = {p.name: signature(p) for p in source.iterdir()}
    output.mkdir(parents=True)
    whole, total = hashlib.sha256(), 0
    with (output / (ARCHIVE + ".partial")).open("xb") as destination:
        for part in parts:
            with open_regular(source / part["path"]) as stream:
                size, digest = copy_bytes(stream, destination, part["bytes"], whole)
                require(
                    not stream.read(1) and size == part["bytes"] and digest == part["sha256"],
                    f"part SHA256/size mismatch: {part['path']}",
                )
                total += size
    require(
        total == archive["bytes"] and whole.hexdigest() == archive["sha256"],
        "joined archive SHA256/size mismatch",
    )
    require(
        all(signature(source / name) == saved for name, saved in before.items()),
        "source changed during join",
    )
    for name, data in raw.items():
        write_partial(output, name, data)
    for name in (ARCHIVE, *METADATA):
        (output / (name + ".partial")).rename(output / name)
    return {
        "status": "joined",
        "part_count": len(parts),
        "archive": archive,
        "scope": "transport bytes only; run verify_result_artifacts.py on the joined bundle",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    split_parser = commands.add_parser("split")
    split_parser.add_argument("--bundle", type=Path, required=True)
    split_parser.add_argument("--output", type=Path, required=True)
    split_parser.add_argument("--dry-run", action="store_true")
    join_parser = commands.add_parser("join")
    join_parser.add_argument("--directory", type=Path, required=True)
    join_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = (
            split(args.bundle, args.output, dry_run=args.dry_run)
            if args.mode == "split"
            else join(args.directory, args.output)
        )
    except KeyboardInterrupt:
        print(
            json.dumps({"status": "interrupted", "partial_output_retained": True}), file=sys.stderr
        )
        return 130
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(error),
                    "partial_output_retained": args.output.exists(),
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
