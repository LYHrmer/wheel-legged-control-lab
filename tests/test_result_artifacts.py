"""Small local fixtures only: packaging correctness and fail-closed boundaries."""

import hashlib
import importlib.util
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/package_result_artifacts.py"


@pytest.fixture(scope="module")
def packager():
    spec = importlib.util.spec_from_file_location("result_artifact_test_subject", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for command in (
        ["init", "--quiet"],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(root), *command], check=True, capture_output=True)
    folder = root / "results/run"
    folder.mkdir(parents=True)
    (folder / "failed.csv").write_bytes(b"terminal_reason,reward\nfall,-2\n")
    (folder / ".metadata.json").write_bytes(b'{"status":"failed"}\n')
    (folder / "empty.bin").write_bytes(b"")
    (folder / "数据.npz").write_bytes(bytes(range(256)) * 4097)
    cache = folder / "__pycache__"
    cache.mkdir()
    (cache / "old.pyc").write_bytes(b"cache")
    return root


def _files(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / "results").rglob("*")
        if path.is_file()
    }


def test_round_trip_all_member_hashes_inputs_unchanged_and_external_checksums(packager, repo):
    before = _files(repo)
    output = repo / "bundle"
    manifest = packager.package(["results/run"], output, root=repo)
    assert _files(repo) == before
    assert {path.name for path in output.iterdir()} == {
        "result_artifacts.tar.gz",
        "archive_manifest.json",
        "SHA256SUMS",
    }
    assert json.loads((output / "archive_manifest.json").read_text()) == manifest
    assert manifest["status"] == "complete"
    assert manifest["packaging_worktree_dirty"]
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert manifest["packaging_git_head"] == head
    assert "NOT the experiment" in manifest["commit_scope"]
    assert manifest["excluded_cache_files"] == ["results/run/__pycache__/old.pyc"]
    members = {item["path"]: item for item in manifest["members"]}
    assert set(members) == set(before) - set(manifest["excluded_cache_files"])
    assert manifest["member_count"] == len(members) == 4
    assert manifest["total_input_bytes"] == sum(len(before[name]) for name in members)
    with tarfile.open(output / "result_artifacts.tar.gz", "r:gz") as archive:
        assert archive.getnames() == sorted(members)
        for item in archive:
            assert item.isfile() and not item.issym() and not item.islnk()
            content = archive.extractfile(item).read()
            assert content == before[item.name]
            assert len(content) == members[item.name]["bytes"]
            assert hashlib.sha256(content).hexdigest() == members[item.name]["sha256"]
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def test_dry_run_checks_real_selection_but_creates_nothing(packager, repo, capsys):
    before = _files(repo)
    output = repo / "does-not-exist/bundle"
    assert (
        packager.main(["--paths", "results/run", "--output", str(output), "--dry-run"], root=repo)
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "dry-run" and plan["member_count"] == 4
    assert all("sha256" not in item for item in plan["members"])
    assert not output.parent.exists()
    assert _files(repo) == before


@pytest.mark.parametrize(
    "paths",
    [
        ["results"],
        ["."],
        ["results/no-such-run"],
        ["results/../results/run"],
        ["results/run", "results/run"],
        ["results/run", "results/run/failed.csv"],
        ["results/run/failed.csv", "results/run"],
        ["results/run/empty.bin/nope"],
    ],
)
def test_reject_bad_paths_before_creating_output(packager, repo, paths):
    with pytest.raises((ValueError, OSError)):
        packager.package(paths, repo / "bundle", root=repo)
    assert not (repo / "bundle").exists()


def test_absolute_in_scope_files_and_disjoint_directories_allowed(packager, repo):
    other = repo / "results/other"
    other.mkdir()
    (other / "log.txt").write_text("keep logs")
    plan = packager.package(
        [repo / "results/run/failed.csv", "results/other"], repo / "bundle", root=repo, dry_run=True
    )
    assert [item["path"] for item in plan["members"]] == [
        "results/other/log.txt",
        "results/run/failed.csv",
    ]
    with pytest.raises(ValueError, match="below results"):
        packager.package([repo.parent], repo / "bad", root=repo, dry_run=True)


@pytest.mark.parametrize("kind", ["file", "directory", "dangling", "inside-cache"])
def test_symlinks_rejected_including_cached_subtrees(packager, repo, kind):
    link = repo / "results/run/link"
    target = repo / "results/run/failed.csv"
    if kind == "directory":
        target = repo / "results/run/__pycache__"
    elif kind == "dangling":
        target = repo / "missing"
    elif kind == "inside-cache":
        link = repo / "results/run/__pycache__/link"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="regular|symlink"):
        packager.package(["results/run"], repo / "bundle", root=repo)
    assert not (repo / "bundle").exists()


def test_symlink_ancestor_and_symlink_results_root_rejected(packager, repo):
    (repo / "results/alias").symlink_to(repo / "results/run", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        packager.package(["results/alias/failed.csv"], repo / "bundle", root=repo)
    (repo / "results").rename(repo / "renamed-results")
    (repo / "results").symlink_to(repo / "renamed-results", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        packager.package(["results/run"], repo / "bundle", root=repo)


@pytest.mark.parametrize("dry_run", [False, True])
def test_existing_output_and_self_inclusion_rejected(packager, repo, dry_run):
    existing = repo / "existing"
    existing.mkdir()
    (existing / "sentinel").write_text("untouched")
    with pytest.raises(ValueError, match="new directory"):
        packager.package(["results/run"], existing, root=repo, dry_run=dry_run)
    assert (existing / "sentinel").read_text() == "untouched"
    with pytest.raises(ValueError, match="inside a selected"):
        packager.package(["results/run"], repo / "results/run/bundle", root=repo, dry_run=dry_run)
    assert not (repo / "results/run/bundle").exists()


def test_output_symlink_ancestor_rejected(packager, repo):
    (repo / "out-link").symlink_to(repo.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        packager.package(["results/run"], repo / "out-link/bundle", root=repo)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX special-file test")
def test_fifo_rejected_before_any_read_can_block(packager, repo):
    os.mkfifo(repo / "results/run/fifo")
    with pytest.raises(ValueError, match="regular"):
        packager.package(["results/run"], repo / "bundle", root=repo)
    assert not (repo / "bundle").exists()


def test_cache_only_selection_is_not_a_successful_empty_package(packager, repo):
    with pytest.raises(ValueError, match="no non-cache"):
        packager.package(["results/run/__pycache__"], repo / "bundle", root=repo)


@pytest.mark.parametrize("change", ["append", "same-size", "replace", "add", "delete"])
def test_concurrent_changes_leave_no_completed_manifest(packager, repo, monkeypatch, change):
    add = packager._add_member
    changed = False

    def mutate(archive, root, member):
        nonlocal changed
        result = add(archive, root, member)
        if not changed:
            changed = True
            path = root / member["path"]
            if change == "append":
                with path.open("ab") as handle:
                    handle.write(b"changed")
            elif change == "same-size":
                path.write_bytes(b"x" * path.stat().st_size)
            elif change == "replace":
                replacement = path.with_name("replacement")
                replacement.write_bytes(path.read_bytes())
                replacement.replace(path)
            elif change == "add":
                (root / "results/run/new-file").write_bytes(b"new")
            else:
                path.unlink()
        return result

    monkeypatch.setattr(packager, "_add_member", mutate)
    output = repo / "bundle"
    with pytest.raises(ValueError, match="changed"):
        packager.package(["results/run"], output, root=repo)
    assert not (output / "archive_manifest.json").exists()
    assert not (output / "SHA256SUMS").exists()


def test_compression_failure_has_nonzero_cli_status_and_no_success_manifest(
    packager,
    repo,
    monkeypatch,
    capsys,
):
    before = _files(repo)

    def fail(*args, **kwargs):
        raise OSError("injected disk full")

    monkeypatch.setattr(packager, "_add_member", fail)
    output = repo / "bundle"
    assert packager.main(["--paths", "results/run", "--output", str(output)], root=repo) == 1
    capture = capsys.readouterr()
    assert "Packaging failed" in capture.err and "disk full" in capture.err
    assert not capture.out
    assert not (output / "archive_manifest.json").exists()
    assert (output / "result_artifacts.tar.gz.partial").exists()
    assert _files(repo) == before


def test_unborn_or_missing_git_head_fails_before_output(packager, repo, monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git", stderr="unknown HEAD")

    monkeypatch.setattr(packager.subprocess, "run", fail)
    with pytest.raises(subprocess.SubprocessError):
        packager.package(["results/run"], repo / "bundle", root=repo)
    assert not (repo / "bundle").exists()


@pytest.mark.parametrize("kind", ["symlink", "directory", "deleted", "fifo"])
def test_swap_after_enumeration_fails_without_reading_an_unapproved_target(
    packager,
    repo,
    monkeypatch,
    kind,
):
    metadata = packager._git_metadata
    sentinel = repo / "private-sentinel"
    sentinel.write_bytes(b"must not be archived")

    def swap(root):
        value = metadata(root)
        target = root / "results/run/.metadata.json"
        target.unlink()
        if kind == "symlink":
            target.symlink_to(sentinel)
        elif kind == "directory":
            target.mkdir()
        elif kind == "fifo":
            os.mkfifo(target)
        return value

    monkeypatch.setattr(packager, "_git_metadata", swap)
    with pytest.raises((OSError, ValueError)):
        packager.package(["results/run"], repo / "bundle", root=repo)
    assert not (repo / "bundle/archive_manifest.json").exists()
    assert sentinel.read_bytes() == b"must not be archived"


@pytest.mark.parametrize("kind", ["grow", "truncate", "read-error"])
def test_changes_during_streaming_do_not_publish_success(packager, repo, monkeypatch, kind):
    read = packager._HashingReader.read
    changed = False
    target = repo / "results/run/数据.npz"

    def change_on_read(reader, size):
        nonlocal changed
        assert 0 < size <= packager.CHUNK_BYTES
        block = read(reader, size)
        if not changed:
            changed = True
            if kind == "read-error":
                raise OSError("injected read failure")
            with target.open("r+b") as handle:
                if kind == "grow":
                    handle.seek(0, 2)
                    handle.write(b"more")
                else:
                    handle.truncate(1)
        return block

    monkeypatch.setattr(packager._HashingReader, "read", change_on_read)
    with pytest.raises((OSError, ValueError)):
        packager.package(["results/run/数据.npz"], repo / "bundle", root=repo)
    assert not (repo / "bundle/archive_manifest.json").exists()
    assert not (repo / "bundle/SHA256SUMS").exists()


def test_no_partial_success_when_final_checksum_publication_fails(packager, repo, monkeypatch):
    rename = Path.rename

    def fail(path, target):
        if path.name == "SHA256SUMS.partial":
            raise OSError("injected publication failure")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(OSError, match="publication failure"):
        packager.package(["results/run"], repo / "bundle", root=repo)
    assert not (repo / "bundle/archive_manifest.json").exists()


def test_hardlinks_are_regular_members_and_cache_named_experiments_are_not_excluded(
    packager,
    repo,
):
    folder = repo / "results/cache_ablation"
    folder.mkdir()
    os.link(repo / "results/run/failed.csv", folder / "error log.txt")
    (folder / "archive_manifest.json").write_bytes(b"experiment's own manifest")
    (folder / "SHA256SUMS").write_bytes(b"experiment's own checksums")
    plan = packager.package(["results/run", "results/cache_ablation"], repo / "bundle", root=repo)
    assert plan["member_count"] == 7
    with tarfile.open(repo / "bundle/result_artifacts.tar.gz") as archive:
        assert all(item.isfile() and not item.islnk() for item in archive)
        assert archive.extractfile("results/cache_ablation/archive_manifest.json").read() == (
            b"experiment's own manifest"
        )


def test_interrupt_exits_nonzero_and_keeps_inputs_and_completion_marker_absent(
    packager,
    repo,
    monkeypatch,
    capsys,
):
    before = _files(repo)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(packager, "_add_member", interrupt)
    assert packager.main(["--paths", "results/run", "--output", "bundle"], root=repo) == 130
    assert "interrupted" in capsys.readouterr().err
    assert _files(repo) == before
    assert not (repo / "bundle/archive_manifest.json").exists()
