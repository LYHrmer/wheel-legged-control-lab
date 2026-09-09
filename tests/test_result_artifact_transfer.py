"""Tiny transport fixtures; no Git, network, simulator or real release data."""

import hashlib
import importlib.util
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def transfer(monkeypatch):
    module = load("transfer_result_artifacts")
    assert module.PART_BYTES == 128 * 1024 * 1024
    monkeypatch.setattr(module, "PART_BYTES", 64)
    monkeypatch.setattr(module, "CHUNK", 19)
    return module


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    folder = tmp_path / "bundle"
    folder.mkdir()
    body = bytes(range(256)) * 4
    name = "results/run/failure.bin"
    archive = folder / "result_artifacts.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    manifest = {
        "schema": "result-artifact-package-v1",
        "status": "complete",
        "selected_paths": ["results/run"],
        "member_count": 1,
        "total_input_bytes": len(body),
        "members": [{"path": name, "bytes": len(body), "sha256": sha(body)}],
        "archive": {
            "path": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": sha(archive.read_bytes()),
        },
    }
    (folder / "archive_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    sums = "".join(
        f"{sha((folder / name).read_bytes())}  {name}\n"
        for name in (archive.name, "archive_manifest.json")
    )
    (folder / "SHA256SUMS").write_text(sums)
    return folder


def contents(folder):
    return {path.name: path.read_bytes() for path in folder.iterdir()}


def test_split_join_exact_bytes_existing_verifier_and_no_input_changes(transfer, bundle, tmp_path):
    original = contents(bundle)
    parts_dir, joined = tmp_path / "parts", tmp_path / "joined"
    plan = transfer.split(bundle, parts_dir)
    assert len(plan["parts"]) > 1
    assert all(part["bytes"] == 64 for part in plan["parts"][:-1])
    for index, part in enumerate(plan["parts"], 1):
        assert part["path"] == f"result_artifacts.tar.gz.part{index:04d}"
        data = (parts_dir / part["path"]).read_bytes()
        assert part["bytes"] == len(data) and part["sha256"] == sha(data)
    assert (
        b"".join((parts_dir / part["path"]).read_bytes() for part in plan["parts"])
        == original[transfer.ARCHIVE]
    )
    before_join = contents(parts_dir)
    result = transfer.join(parts_dir, joined)
    assert result["status"] == "joined" and result["part_count"] == len(plan["parts"])
    assert contents(bundle) == contents(joined) == original
    assert contents(parts_dir) == before_join
    assert load("verify_result_artifacts").verify(joined)["member_count"] == 1
    assert not list(joined.glob("*.partial")) and not list(parts_dir.glob("*.partial"))


def test_dry_run_scans_real_bytes_but_creates_nothing(transfer, bundle, tmp_path):
    before = contents(bundle)
    output = tmp_path / "missing" / "parts"
    plan = transfer.split(bundle, output, dry_run=True)
    assert plan["status"] == "dry-run" and not output.parent.exists()
    assert contents(bundle) == before
    assert sum(part["bytes"] for part in plan["parts"]) == len(before[transfer.ARCHIVE])


@pytest.mark.parametrize("mode", ("split", "join"))
@pytest.mark.parametrize("bad", ("existing", "overlap", "symlink", "parent_symlink"))
def test_output_boundary_is_rejected_before_writes(transfer, bundle, tmp_path, mode, bad):
    source = bundle
    if mode == "join":
        source = tmp_path / "parts"
        transfer.split(bundle, source)
    output = tmp_path / "new"
    if bad == "existing":
        output.mkdir()
    elif bad == "overlap":
        output = source / "new"
    elif bad == "symlink":
        output.symlink_to(tmp_path / "missing")
    else:
        (tmp_path / "linked").symlink_to(tmp_path, target_is_directory=True)
        output = tmp_path / "linked" / "new"
    before = contents(source)
    with pytest.raises(ValueError):
        getattr(transfer, mode)(source, output)
    assert contents(source) == before


@pytest.mark.parametrize("bad", ("missing", "link", "fifo", "extra", "partial"))
def test_download_directory_rejects_missing_or_unsafe_entries(transfer, bundle, tmp_path, bad):
    directory = tmp_path / "parts"
    plan = transfer.split(bundle, directory)
    part = directory / plan["parts"][0]["path"]
    if bad == "missing":
        part.unlink()
    elif bad == "link":
        part.unlink()
        part.symlink_to(bundle / transfer.ARCHIVE)
    elif bad == "fifo":
        part.unlink()
        os.mkfifo(part)
    elif bad == "extra":
        (directory / "other.bin").write_bytes(b"unexpected")
    else:
        (directory / "interrupted.partial").write_bytes(b"incomplete")
    output = tmp_path / "joined"
    with pytest.raises((OSError, ValueError)):
        transfer.join(directory, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "bad",
    (
        "path",
        "backslash",
        "absolute",
        "duplicate",
        "order",
        "bytes",
        "bool_bytes",
        "hash",
        "count",
        "chunk_size",
        "archive",
        "metadata",
        "status",
        "unknown_field",
    ),
)
def test_parts_manifest_is_strict_and_validated_before_join(transfer, bundle, tmp_path, bad):
    directory = tmp_path / "parts"
    plan = transfer.split(bundle, directory)
    if bad in ("path", "backslash", "absolute"):
        plan["parts"][0]["path"] = {
            "path": "../outside",
            "backslash": "x\\a",
            "absolute": "/outside",
        }[bad]
    elif bad == "duplicate":
        plan["parts"][1] = plan["parts"][0].copy()
    elif bad == "order":
        plan["parts"].reverse()
    elif bad in ("bytes", "bool_bytes", "hash"):
        key, value = {
            "bytes": ("bytes", 63),
            "bool_bytes": ("bytes", True),
            "hash": ("sha256", "invalid"),
        }[bad]
        plan["parts"][0][key] = value
    elif bad == "count":
        plan["parts"].pop()
    elif bad == "chunk_size":
        plan["part_bytes"] *= 2
    elif bad == "archive":
        plan["archive"]["sha256"] = "0" * 64
    elif bad == "metadata":
        plan["metadata"][0]["sha256"] = "0" * 64
    elif bad == "status":
        plan["status"] = "dry-run"
    else:
        plan["unexpected"] = 1
    (directory / transfer.PARTS_MANIFEST).write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        transfer.join(directory, tmp_path / "joined")
    assert not (tmp_path / "joined").exists()


@pytest.mark.parametrize("payload", (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'))
def test_duplicate_and_nonfinite_json_rejected(transfer, payload):
    with pytest.raises(ValueError):
        transfer.strict_json(payload)


@pytest.mark.parametrize("mode", ("split", "join"))
def test_corrupt_bytes_fail_with_partial_not_completed_output(transfer, bundle, tmp_path, mode):
    if mode == "split":
        source, target = bundle, bundle / transfer.ARCHIVE
    else:
        source = tmp_path / "parts"
        plan = transfer.split(bundle, source)
        target = source / plan["parts"][0]["path"]
    data = target.read_bytes()
    target.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    before = contents(source)
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="SHA256"):
        getattr(transfer, mode)(source, output)
    assert list(output.glob("*.partial"))
    assert not (output / transfer.PARTS_MANIFEST).exists()
    assert not (output / transfer.ARCHIVE).exists()
    assert contents(source) == before


def test_modified_part_and_its_digest_still_fail_whole_archive_hash(transfer, bundle, tmp_path):
    directory = tmp_path / "parts"
    plan = transfer.split(bundle, directory)
    target = directory / plan["parts"][0]["path"]
    data = bytearray(target.read_bytes())
    data[0] ^= 1
    target.write_bytes(data)
    plan["parts"][0]["sha256"] = sha(data)
    (directory / transfer.PARTS_MANIFEST).write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="joined archive SHA256"):
        transfer.join(directory, tmp_path / "joined")
    assert (tmp_path / "joined" / (transfer.ARCHIVE + ".partial")).exists()


def test_copy_reads_only_bounded_blocks(transfer):
    class BoundedInput(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= transfer.CHUNK
            return super().read(size)

    source = BoundedInput(bytes(range(256)) * 7)
    output, digest = io.BytesIO(), hashlib.sha256()
    count, actual = transfer.copy_bytes(source, output, 1300, digest)
    assert count == 1300 and len(output.getvalue()) == count
    assert actual == digest.hexdigest() == sha(output.getvalue())


def test_cli_failure_nonzero_and_dry_run_reports_status(transfer, bundle, tmp_path, capsys):
    assert (
        transfer.main(
            ["split", "--bundle", str(bundle), "--output", str(tmp_path / "new"), "--dry-run"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "dry-run"
    assert (
        transfer.main(
            ["join", "--directory", str(tmp_path / "absent"), "--output", str(tmp_path / "new")]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().err)["status"] == "failed"
    assert not (tmp_path / "new").exists()
