"""Read-only verification of a downloaded result-artifact-package-v1 bundle.

No extraction, execution, simulator import, Git lookup, or network access.
Checks byte integrity, not authenticity or scientific correctness. Extra
ordinary files at the bundle root are ignored and counted; partial outputs,
symlinks, directories, and special files are rejected.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import stat
import sys
import tarfile
from pathlib import Path, PurePosixPath

ARCHIVE = "result_artifacts.tar.gz"
MANIFEST = "archive_manifest.json"
SUMS = "SHA256SUMS"
CHUNK = 1024 * 1024
MAX_MANIFEST_BYTES = 64 * CHUNK
HEX256 = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, label):
    _require(type(value) is int and value >= 0, f"{label} must be a nonnegative integer")
    return value


def _digest(stream):
    digest = hashlib.sha256()
    count = 0
    while block := stream.read(CHUNK):
        digest.update(block)
        count += len(block)
    return count, digest.hexdigest()


def _read_small(path, limit):
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    _require(len(value) <= limit, f"metadata exceeds size limit: {path.name}")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _finite_float(value):
    number = float(value)
    _require(float("-inf") < number < float("inf"), "nonfinite JSON number")
    return number


def _safe_member(name):
    _require(isinstance(name, str) and name, "member path must be a nonempty string")
    parts = name.split("/")
    _require(
        len(parts) >= 2
        and parts[0] == "results"
        and all(part not in ("", ".", "..") for part in parts)
        and not any(ord(char) < 32 or char == "\\" for char in name),
        f"unsafe results member path: {name!r}",
    )
    return PurePosixPath(name)


def _manifest_entries(manifest):
    _require(isinstance(manifest, dict), "manifest must be a JSON object")
    _require(manifest.get("schema") == "result-artifact-package-v1", "unknown manifest schema")
    _require(manifest.get("status") == "complete", "manifest does not mark a completed bundle")
    selected = manifest.get("selected_paths")
    _require(isinstance(selected, list) and selected, "selected_paths must be a nonempty list")
    selection = []
    for name in selected:
        path = _safe_member(name)
        _require(
            not any(
                path.is_relative_to(other) or other.is_relative_to(path) for other in selection
            ),
            "duplicate/overlapping selected_paths",
        )
        selection.append(path)
    members = manifest.get("members")
    _require(isinstance(members, list) and members, "members must be a nonempty list")
    expected = {}
    for member in members:
        _require(isinstance(member, dict), "each manifest member must be an object")
        path = _safe_member(member.get("path"))
        _require(
            any(path.is_relative_to(parent) for parent in selection),
            "member outside selected_paths",
        )
        name = str(path)
        _require(name not in expected, f"duplicate manifest member: {name}")
        _integer(member.get("bytes"), f"{name} bytes")
        _require(
            isinstance(member.get("sha256"), str) and HEX256.fullmatch(member["sha256"]),
            f"invalid member SHA256: {name}",
        )
        expected[name] = member
    _require(
        _integer(manifest.get("member_count"), "member_count") == len(expected),
        "member_count mismatch",
    )
    _require(
        _integer(manifest.get("total_input_bytes"), "total_input_bytes")
        == sum(member["bytes"] for member in expected.values()),
        "total_input_bytes mismatch",
    )
    return expected


class _BoundedTarInfo(tarfile.TarInfo):
    def _proc_member(self, archive):
        # TarInfo documents this as the member-processing subclass hook. Some
        # Python security backports bypass frombuf(), so validate here instead.
        metadata = (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME)
        _require(
            self.type in (tarfile.REGTYPE, tarfile.AREGTYPE, *metadata),
            "non-regular tar header (links, directories, devices and sparse types forbidden)",
        )
        if self.type in metadata:
            _require(0 <= self.size <= CHUNK, "oversized tar metadata header")
        return super()._proc_member(archive)


class _BoundedReader:
    def __init__(self, stream):
        self.stream = stream
        self.last_header = b""

    def read(self, size):
        _require(0 <= size <= CHUNK, "oversized tar read")
        data = self.stream.read(size)
        if size == tarfile.BLOCKSIZE:
            self.last_header = data
        return data

    def seek(self, *args):
        return self.stream.seek(*args)

    def tell(self):
        return self.stream.tell()


def _inspect_archive(path, expected):
    seen, total = set(), 0
    # Sequential iteration over gzip: only small headers and one payload chunk
    # are held, not an extracted tree or the whole uncompressed archive.
    with gzip.open(path, "rb") as compressed:
        reader = _BoundedReader(compressed)
        with tarfile.open(fileobj=reader, mode="r:", tarinfo=_BoundedTarInfo) as archive:
            for member in archive:
                name = str(_safe_member(member.name))
                _require(
                    member.isfile() and member.sparse is None,
                    "only ordinary regular tar files allowed",
                )
                _require(name not in seen, f"duplicate tar member: {name}")
                _require(name in expected, f"unexpected tar member: {name}")
                info = expected[name]
                _require(member.size == info["bytes"], f"tar header size mismatch: {name}")
                with archive.extractfile(member) as stream:
                    size, digest = _digest(stream)
                _require(size == info["bytes"], f"member byte count mismatch: {name}")
                _require(digest == info["sha256"], f"member SHA256 mismatch: {name}")
                seen.add(name)
                total += size
            # tarfile stops at its first end block. Read the remainder to force
            # gzip CRC/truncation checks and reject hidden appended tar content.
            _require(reader.last_header == bytes(tarfile.BLOCKSIZE), "missing tar end marker")
            padding = compressed.read(tarfile.RECORDSIZE + 1)
            _require(
                tarfile.BLOCKSIZE <= len(padding) <= tarfile.RECORDSIZE and not any(padding),
                "unexpected content after tar end marker",
            )
    _require(seen == set(expected), f"missing tar members: {sorted(set(expected) - seen)}")
    return len(seen), total


def verify(directory) -> dict:
    directory = Path(directory).absolute()
    _require(
        not any(path.is_symlink() for path in (*directory.parents, directory)),
        "bundle directory must not use symlinks",
    )
    _require(directory.is_dir(), "bundle directory does not exist")
    files = {}
    for path in directory.iterdir():
        _require(not path.name.endswith(".partial"), "unfinished .partial output is present")
        info = path.lstat()
        _require(
            stat.S_ISREG(info.st_mode),
            "bundle root entries must be regular files, not links/directories",
        )
        files[path.name] = (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
    _require(
        {ARCHIVE, MANIFEST, SUMS} <= files.keys(),
        "bundle is missing archive, manifest or SHA256SUMS",
    )
    sums = {}
    for line in _read_small(directory / SUMS, 512).decode("ascii").splitlines():
        match = re.fullmatch(
            r"([0-9a-f]{64})  (result_artifacts\.tar\.gz|archive_manifest\.json)", line
        )
        _require(match is not None, "invalid SHA256SUMS line or filename")
        digest, name = match.groups()
        _require(name not in sums, "duplicate SHA256SUMS filename")
        sums[name] = digest
    _require(
        set(sums) == {ARCHIVE, MANIFEST}, "SHA256SUMS must contain exactly two fixed filenames"
    )
    for name, expected_digest in sums.items():
        with (directory / name).open("rb") as stream:
            _, actual = _digest(stream)
        _require(actual == expected_digest, f"external SHA256 mismatch: {name}")
    manifest = json.loads(
        _read_small(directory / MANIFEST, MAX_MANIFEST_BYTES),
        object_pairs_hook=_unique_object,
        parse_constant=_finite_constant,
        parse_float=_finite_float,
    )
    expected = _manifest_entries(manifest)
    archived = manifest.get("archive")
    _require(
        isinstance(archived, dict) and archived.get("path") == ARCHIVE, "invalid archive filename"
    )
    _require(
        _integer(archived.get("bytes"), "archive bytes") == files[ARCHIVE][2],
        "archive size mismatch",
    )
    _require(archived.get("sha256") == sums[ARCHIVE], "archive SHA256 differs between manifests")
    count, total = _inspect_archive(directory / ARCHIVE, expected)
    _require(
        count == manifest["member_count"] and total == manifest["total_input_bytes"],
        "tar totals differ from manifest",
    )
    for name, fingerprint in files.items():
        info = (directory / name).lstat()
        _require(
            stat.S_ISREG(info.st_mode)
            and (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            == fingerprint,
            f"bundle file changed during verification: {name}",
        )
    return {
        "status": "verified",
        "member_count": count,
        "total_input_bytes": total,
        "archive_bytes": files[ARCHIVE][2],
        "ignored_extra_file_count": len(files) - 3,
        "scope": "byte integrity only; not authenticity or scientific validity; extra regular files ignored",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = verify(args.directory)
    except (OSError, ValueError, EOFError, tarfile.TarError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
