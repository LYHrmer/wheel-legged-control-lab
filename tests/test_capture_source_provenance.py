"""The opt-in CLI binds newly added code without changing the frozen helper."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/capture_source_provenance.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = load(SCRIPT, "capture_source_provenance")
legacy = load(ROOT / "src/wheel_legged_control/provenance.py", "legacy_source_provenance")


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    (root / "src").mkdir()
    (root / "src/base.py").write_text("GAIN = 1\n")
    (root / ".gitignore").write_text("ignored.py\nresults/\n")
    git(root, "add", ".")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "initial source")
    return root


def cli(root, *args):
    return subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), *map(str, args)],
                          capture_output=True, text=True, check=False)


def test_legacy_collision_and_new_cli_detects_untracked_source_changes(repository, tmp_path):
    added = repository / "src/controller.py"
    added.write_text("GAIN = 1\n")
    before = legacy.capture_git_provenance(repository)
    manifest = tmp_path / "before.json"
    assert cli(repository, "--output", manifest).returncode == 0
    assert cli(repository, "--verify", manifest).returncode == 0
    added.write_text("GAIN = 999\n")
    # Reproduce the known frozen-helper defect before asserting the new behavior.
    assert legacy.capture_git_provenance(repository) == before
    checked = cli(repository, "--verify", manifest)
    assert checked.returncode == 1
    result = json.loads(checked.stdout)
    assert not result["source_matches"] and result["git_commit_matches"]
    assert result["changed_paths"] == ["src/controller.py"]


@pytest.mark.parametrize("change", ("modify", "delete", "rename", "stage_rename"))
def test_verify_detects_tracked_source_changes(repository, tmp_path, change):
    manifest = tmp_path / "before.json"
    assert cli(repository, "--output", manifest).returncode == 0
    source = repository / "src/base.py"
    if change == "modify":
        source.write_text("GAIN = 99\n")
    elif change == "delete":
        source.unlink()
    else:
        source.rename(repository / "src/renamed.py")
        if change == "stage_rename":
            git(repository, "add", "-A")
    result = json.loads(cli(repository, "--verify", manifest).stdout)
    assert not result["matches"]
    if change == "delete":
        assert {entry["path"]: entry["type"] for entry in tool.capture(repository)["files"]} == {
            "src/base.py": "missing"}


@pytest.mark.parametrize("rename", (False, True))
def test_verify_detects_untracked_source_removal_or_rename(repository, tmp_path, rename):
    source = repository / "src/new controller.py"
    source.write_text("GAIN = 1\n")
    manifest = tmp_path / "before.json"
    assert cli(repository, "--output", manifest).returncode == 0
    if rename:
        source.rename(repository / "src/renamed controller.py")
    else:
        source.unlink()
    checked = cli(repository, "--verify", manifest)
    assert checked.returncode == 1
    assert "src/new controller.py" in json.loads(checked.stdout)["changed_paths"]


def test_scope_excludes_ignored_code_results_dependencies_and_binary_assets(repository):
    before = tool.capture(repository)["source_sha256"]
    for name in ("src/ignored.py", "results/run.py", ".local-deps/module.py", "src/mesh.STL"):
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"excluded\n")
    assert tool.capture(repository)["source_sha256"] == before
    # Tracked files stay in scope even if their names later match an ignore rule.
    git(repository, "add", "-f", "src/ignored.py")
    assert tool.capture(repository)["source_sha256"] != before


def test_scope_includes_declared_directories_extensions_and_root_config(repository):
    expected = {"src/base.py", "pyproject.toml"}
    for directory in tool.DIRECTORIES:
        for suffix in tool.EXTENSIONS:
            name = f"{directory}/new{suffix}"
            path = repository / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"source")
            expected.add(name)
    (repository / "pyproject.toml").write_text("[project]\n")
    assert {entry["path"] for entry in tool.capture(repository)["files"]} == expected


def test_symlinks_bind_link_text_and_never_read_the_target(repository, tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"outside content")
    link = repository / "src/link.py"
    link.symlink_to(target)
    before = tool.capture(repository)
    target.write_bytes(b"different outside content")
    assert tool.capture(repository)["source_sha256"] == before["source_sha256"]
    link.unlink()
    link.symlink_to(tmp_path / "missing")
    assert tool.capture(repository)["source_sha256"] != before["source_sha256"]


def test_parent_symlink_and_special_file_are_rejected(repository, tmp_path):
    base = repository / "src/base.py"
    base.unlink()
    os.mkfifo(base)
    with pytest.raises(ValueError, match="regular file or symlink"):
        tool.capture(repository)
    base.unlink()
    (repository / "src").rmdir()
    (repository / "src").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="parent is a symlink"):
        tool.capture(repository)


def test_cli_refuses_overwrite_and_output_inside_source_scope(repository, tmp_path):
    output = tmp_path / "before.json"
    assert cli(repository, "--output", output).returncode == 0
    saved = output.read_bytes()
    assert cli(repository, "--output", output).returncode == 2
    assert output.read_bytes() == saved
    assert cli(repository, "--output", repository / "src/manifest.json").returncode == 2
    assert not (repository / "src/manifest.json").exists()


def test_verify_checks_head_and_rejects_different_scope(repository, tmp_path):
    manifest = tmp_path / "before.json"
    assert cli(repository, "--output", manifest).returncode == 0
    git(repository, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "--allow-empty", "-qm", "new head")
    checked = json.loads(cli(repository, "--verify", manifest).stdout)
    assert checked["source_matches"] and not checked["git_commit_matches"]
    record = json.loads(manifest.read_text())
    record["scope"] = {}
    manifest.write_text(json.dumps(record))
    assert cli(repository, "--verify", manifest).returncode == 2


def test_capture_rejects_ordinary_concurrent_source_changes(repository, monkeypatch):
    original = tool.source_entry

    def change_after_read(root, name):
        result = original(root, name)
        (root / name).write_text("concurrently edited source\n")
        return result

    monkeypatch.setattr(tool, "source_entry", change_after_read)
    with pytest.raises(ValueError, match="changed during capture"):
        tool.capture(repository)
