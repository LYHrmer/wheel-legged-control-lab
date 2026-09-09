"""Independent handwritten tar/JSON fixtures; no Git, packager or simulator."""

import gzip
import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_result_artifacts.py"


@pytest.fixture(scope="module")
def verifier():
    spec = importlib.util.spec_from_file_location("result_verify_test_subject", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _save_manifest(folder, manifest):
    (folder / "archive_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _sums(folder)


def _sums(folder):
    (folder / "SHA256SUMS").write_text(
        "".join(
            f"{_sha((folder / name).read_bytes())}  {name}\n"
            for name in ("result_artifacts.tar.gz", "archive_manifest.json")
        )
    )


def _bundle(folder, *, entries=None):
    folder.mkdir()
    if entries is None:
        entries = [
            ("results/run/failed.csv", b"fall,-2\n", tarfile.REGTYPE),
            ("results/run/数据.bin", bytes(range(256)) * 9000, tarfile.REGTYPE),
            ("results/run/empty", b"", tarfile.REGTYPE),
        ]
    with tarfile.open(folder / "result_artifacts.tar.gz", "w:gz") as archive:
        for name, data, kind in entries:
            info = tarfile.TarInfo(name)
            info.type, info.size = kind, len(data)
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                info.linkname = "../../outside"
            archive.addfile(info, io.BytesIO(data))
    members = [
        {"path": name, "bytes": len(data), "sha256": _sha(data)} for name, data, _ in entries
    ]
    manifest = {
        "schema": "result-artifact-package-v1",
        "status": "complete",
        "selected_paths": ["results/run"],
        "members": members,
        "member_count": len(members),
        "total_input_bytes": sum(item["bytes"] for item in members),
        "archive": {
            "path": "result_artifacts.tar.gz",
            "bytes": (folder / "result_artifacts.tar.gz").stat().st_size,
            "sha256": _sha((folder / "result_artifacts.tar.gz").read_bytes()),
        },
    }
    _save_manifest(folder, manifest)
    return manifest


def test_verified_streaming_unicode_empty_and_extra_regular_file_without_writes(verifier, tmp_path):
    folder = tmp_path / "bundle"
    manifest = _bundle(folder)
    (folder / "download-notes.txt").write_bytes(b"ignored extra")
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    result = verifier.verify(folder)
    assert result["status"] == "verified"
    assert result["member_count"] == manifest["member_count"] == 3
    assert result["total_input_bytes"] == manifest["total_input_bytes"]
    assert result["ignored_extra_file_count"] == 1
    assert "not authenticity" in result["scope"]
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("target", ["result_artifacts.tar.gz", "archive_manifest.json"])
def test_external_corruption_rejected(verifier, tmp_path, target):
    folder = tmp_path / "bundle"
    _bundle(folder)
    with (folder / target).open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="external SHA256"):
        verifier.verify(folder)


@pytest.mark.parametrize(
    "text",
    [
        "0" * 64 + "  ../result_artifacts.tar.gz\n",
        "0" * 64 + "  /archive_manifest.json\n",
        "0" * 64 + "  result_artifacts.tar.gz\n" + "0" * 64 + "  result_artifacts.tar.gz\n",
        "0" * 64 + " *result_artifacts.tar.gz\n",
        "0" * 64 + "  archive_manifest.json\n",
        "",
        "not-a-digest  result_artifacts.tar.gz\n",
    ],
)
def test_strict_checksum_file_names_counts_and_spacing(verifier, tmp_path, text):
    folder = tmp_path / "bundle"
    _bundle(folder)
    (folder / "SHA256SUMS").write_text(text)
    with pytest.raises(ValueError, match="SHA256SUMS"):
        verifier.verify(folder)


@pytest.mark.parametrize(
    "extra", ["result_artifacts.tar.gz.partial", "unexpected-link", "subdirectory"]
)
def test_incomplete_or_nonregular_bundle_entries_rejected(verifier, tmp_path, extra):
    folder = tmp_path / "bundle"
    _bundle(folder)
    if extra.endswith("partial"):
        (folder / extra).write_bytes(b"")
    elif extra == "unexpected-link":
        (folder / extra).symlink_to(tmp_path / "missing")
    else:
        (folder / extra).mkdir()
    with pytest.raises(ValueError, match="partial|regular"):
        verifier.verify(folder)


@pytest.mark.parametrize(
    "missing", ["archive_manifest.json", "SHA256SUMS", "result_artifacts.tar.gz"]
)
def test_missing_required_file_rejected(verifier, tmp_path, missing):
    folder = tmp_path / "bundle"
    _bundle(folder)
    (folder / missing).unlink()
    with pytest.raises(ValueError, match="missing"):
        verifier.verify(folder)


@pytest.mark.parametrize(
    "replacement",
    [
        '{"status":"complete","status":"complete"}',
        '{"nested":{"a":1,"a":2}}',
        '{"bad":NaN}',
        '{"bad":Infinity}',
        '{"bad":-Infinity}',
        '{"bad":1e999}',
    ],
)
def test_duplicate_and_nonfinite_json_even_when_external_hash_is_updated(
    verifier, tmp_path, replacement
):
    folder = tmp_path / "bundle"
    _bundle(folder)
    (folder / "archive_manifest.json").write_text(replacement)
    _sums(folder)
    with pytest.raises(ValueError, match="duplicate|nonfinite"):
        verifier.verify(folder)


@pytest.mark.parametrize(
    "field,value",
    [
        ("member_count", 9),
        ("member_count", True),
        ("total_input_bytes", -1),
        ("total_input_bytes", 0),
        ("status", "dry-run"),
        ("schema", "future"),
        ("members", []),
        ("selected_paths", ["results/other"]),
    ],
)
def test_manifest_status_schema_stat_totals_and_selection(verifier, tmp_path, field, value):
    folder = tmp_path / "bundle"
    manifest = _bundle(folder)
    manifest[field] = value
    _save_manifest(folder, manifest)
    with pytest.raises(ValueError):
        verifier.verify(folder)


@pytest.mark.parametrize(
    "field,value", [("path", "other.tar.gz"), ("bytes", 1), ("sha256", "0" * 64)]
)
def test_archive_manifest_filename_size_hash(verifier, tmp_path, field, value):
    folder = tmp_path / "bundle"
    manifest = _bundle(folder)
    manifest["archive"][field] = value
    _save_manifest(folder, manifest)
    with pytest.raises(ValueError, match="archive"):
        verifier.verify(folder)


@pytest.mark.parametrize(
    "name",
    [
        "../results/run/data",
        "/results/run/data",
        "results/run/../data",
        "results//run/data",
        "results/run/./data",
        "results/run/evil\\name",
        "results/run/evil\nname",
        "outside/data",
    ],
)
def test_unsafe_paths_rejected_even_when_manifest_and_archive_agree(verifier, tmp_path, name):
    folder = tmp_path / "bundle"
    _bundle(folder, entries=[(name, b"bytes", tarfile.REGTYPE)])
    with pytest.raises(ValueError, match="unsafe"):
        verifier.verify(folder)


@pytest.mark.parametrize(
    "kind",
    [
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
        tarfile.DIRTYPE,
        tarfile.FIFOTYPE,
        tarfile.CHRTYPE,
        tarfile.GNUTYPE_SPARSE,
    ],
)
def test_nonregular_tar_headers_rejected(verifier, tmp_path, kind):
    folder = tmp_path / "bundle"
    _bundle(folder, entries=[("results/run/file", b"", kind)])
    with pytest.raises(ValueError, match="non-regular|ordinary"):
        verifier.verify(folder)


def test_duplicate_tar_member_is_rejected_independently_of_manifest_duplicates(verifier, tmp_path):
    folder = tmp_path / "bundle"
    entry = ("results/run/file", b"bytes", tarfile.REGTYPE)
    manifest = _bundle(folder, entries=[entry, entry])
    with pytest.raises(ValueError, match="duplicate manifest"):
        verifier.verify(folder)
    manifest["members"] = manifest["members"][:1]
    manifest["member_count"], manifest["total_input_bytes"] = 1, len(entry[1])
    _save_manifest(folder, manifest)
    with pytest.raises(ValueError, match="duplicate tar"):
        verifier.verify(folder)


@pytest.mark.parametrize("change", ["member-hash", "header-size", "missing", "unexpected"])
def test_tar_content_must_match_manifest_exactly(verifier, tmp_path, change):
    folder = tmp_path / "bundle"
    manifest = _bundle(folder)
    if change == "member-hash":
        manifest["members"][0]["sha256"] = "0" * 64
    elif change == "header-size":
        manifest["members"][0]["bytes"] += 1
        manifest["total_input_bytes"] += 1
    elif change == "missing":
        manifest["members"].append({"path": "results/run/absent", "bytes": 0, "sha256": _sha(b"")})
        manifest["member_count"] += 1
    else:
        removed = manifest["members"].pop()
        manifest["member_count"] -= 1
        manifest["total_input_bytes"] -= removed["bytes"]
    _save_manifest(folder, manifest)
    with pytest.raises(ValueError, match="SHA256|size mismatch|missing tar|unexpected tar"):
        verifier.verify(folder)


@pytest.mark.parametrize("corruption", ["gzip-truncated", "tar-appended", "gzip-crc", "no-tar-end"])
def test_checksums_cannot_hide_truncated_compression_or_content_after_tar_end(
    verifier, tmp_path, corruption
):
    folder = tmp_path / "bundle"
    manifest = _bundle(folder)
    archive = folder / "result_artifacts.tar.gz"
    data = archive.read_bytes()
    if corruption == "gzip-truncated":
        data = data[:-6]
    elif corruption == "gzip-crc":
        data = data[:-8] + bytes([data[-8] ^ 1]) + data[-7:]
    elif corruption == "no-tar-end":
        raw = gzip.decompress(data)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as contents:
            last = contents.getmembers()[-1]
            end = last.offset_data + (last.size + 511) // 512 * 512
        data = gzip.compress(raw[:end])
    else:
        data = gzip.compress(gzip.decompress(data) + b"hidden second archive")
    archive.write_bytes(data)
    manifest["archive"].update(bytes=len(data), sha256=_sha(data))
    _save_manifest(folder, manifest)
    with pytest.raises((OSError, EOFError, ValueError)):
        verifier.verify(folder)


def test_oversized_extended_header_rejected_before_payload_allocation(verifier, tmp_path):
    folder = tmp_path / "bundle"
    manifest = _bundle(folder)
    header = tarfile.TarInfo("././@PaxHeader")
    header.type, header.size = tarfile.XHDTYPE, 2**31
    data = gzip.compress(header.tobuf())
    (folder / "result_artifacts.tar.gz").write_bytes(data)
    manifest["archive"].update(bytes=len(data), sha256=_sha(data))
    _save_manifest(folder, manifest)
    with pytest.raises(ValueError, match="oversized tar metadata"):
        verifier.verify(folder)


def test_library_and_cli_failure_status_and_short_json(verifier, tmp_path, capsys):
    folder = tmp_path / "bundle"
    _bundle(folder)
    assert verifier.main(["--directory", str(folder)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    (folder / "archive_manifest.json").unlink()
    assert verifier.main(["--directory", str(folder)]) == 1
    capture = capsys.readouterr()
    assert not capture.out and json.loads(capture.err)["status"] == "failed"


def test_bundle_directory_symlink_is_not_followed(verifier, tmp_path):
    folder = tmp_path / "bundle"
    _bundle(folder)
    alias = tmp_path / "alias"
    alias.symlink_to(folder, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        verifier.verify(alias)
