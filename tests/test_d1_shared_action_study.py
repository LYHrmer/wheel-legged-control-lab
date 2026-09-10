"""Command-plan contracts and fake-process failure tests; these do not train PPO."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_d1_shared_action_study as study


def flags(command):
    return {command[i]: command[i + 1] for i in range(3, len(command), 2)}


def read_json(path):
    return json.loads(path.read_text())


def test_fixed_budget_and_shared_controller_without_dr(tmp_path):
    # Opus supplied the three plan-test outlines; field names were checked here.
    output = tmp_path / "formal"
    for item in study.build_runs(output):
        command = item["command"]
        assert command[:3] == [sys.executable, str(study.ROOT / study.CHILD_SCRIPT), item["mode"]]
        option = flags(command)
        assert option["--baseline"] == "wheel_leg"
        assert option["--source"] == "imu_encoder_fusion"
        assert option["--terrain-suite"] == "action_compare_v1"
        assert option["--action-mode"] == item["action_mode"]
        assert int(option["--seed"]) == item["seed"]
        assert int(option["--history"]) == 1
        assert int(option["--measurement-delay"]) == 0
        assert int(option["--workers"]) == 4
        assert int(option["--steps"]) == 32768
        assert float(option["--duration"]) == 60.0
        for key, value in {
            "--wheel-kp": 0.55,
            "--wheel-ki": 1.5,
            "--yaw-feedback-gain": 4,
            "--leg-feedback-scale": 1,
            "--attitude-feedback-scale": 0.25,
        }.items():
            assert float(option[key]) == value
        assert "--delay-randomization" not in command
        if item["mode"] == "train":
            assert int(option["--steps"]) == 32768
            assert int(option["--steps"]) % (4 * 128) == 0
            assert "--split" not in option
        else:
            assert option["--split"] == item["split"]
    assert not output.exists()


def test_complete_mode_major_inventory_and_holdout_order(tmp_path):
    runs = study.build_runs(tmp_path / "study")
    names = [item["name"] for item in runs]
    assert len(names) == len(set(names)) == 20
    assert names[0] == "zero_development"
    assert names[1:13] == [
        f"{mode}_seed{seed}{suffix}"
        for mode in ("shared2", "independent8")
        for seed in (31000, 32000, 33000)
        for suffix in ("", "_development")
    ]
    assert names[13] == "zero_holdout"
    assert names[14:] == [
        f"{mode}_seed{seed}_holdout"
        for mode in ("shared2", "independent8")
        for seed in (31000, 32000, 33000)
    ]
    assert max(i for i, r in enumerate(runs) if r["split"] == "development") < min(
        i for i, r in enumerate(runs) if r["split"] == "holdout"
    )
    smoke = study.build_runs(tmp_path / "smoke", smoke=True)
    assert len(smoke) == 8
    assert {r["seed"] for r in smoke if not r["name"].startswith("zero_")} == {31001}
    for item in smoke:
        option = flags(item["command"])
        assert float(option["--duration"]) == 0.2
        if item["mode"] == "train":
            assert int(option["--steps"]) == 512


def test_checkpoints_belong_to_matching_mode_seed_and_need_not_exist(tmp_path):
    output = tmp_path / "not_created"
    evaluations = 0
    for item in study.build_runs(output):
        option = flags(item["command"])
        assert Path(option["--output"]) == output / item["name"]
        if item["name"].startswith("zero_") or item["mode"] == "train":
            assert "--policy" not in option and "--metadata" not in option
            if item["name"].startswith("zero_"):
                assert item["action_mode"] == "independent8" and item["seed"] == 31000
        else:
            evaluations += 1
            model = output / f"{item['action_mode']}_seed{item['seed']}"
            assert Path(option["--policy"]) == model / "checkpoint.zip"
            assert Path(option["--metadata"]) == model / "checkpoint.json"
    assert evaluations == 12 and not output.exists()


@pytest.fixture
def repository(tmp_path, monkeypatch):
    """An isolated source tree, not a replacement trained model or scientific result."""
    root = tmp_path / "fake_repository"
    files = {
        study.SCRIPT: "# fixture orchestrator\n",
        study.CHILD_SCRIPT: "# fixture child runner\n",
        "pyproject.toml": "[project]\nname = 'fake'\n",
        "src/wheel_legged_control/model.py": "# fixture dynamics\n",
        "src/wheel_legged_control/d1/assets/model.xml": "<mujoco/>\n",
        "src/wheel_legged_control/d1/assets/texture.png": "fake texture bytes\n",
    }
    for name, contents in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(study, "ROOT", root)
    return root


def fake_process(monkeypatch, callback=None):
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write("FAKE process for orchestration tests only\n")
        if callback:
            callback(command, kwargs, len(calls))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(study.subprocess, "run", execute)
    return calls


def test_success_records_all_commands_not_quality_pass(repository, tmp_path, monkeypatch):
    output = tmp_path / "fake_output"

    def check(command, kwargs, index):
        protocol = read_json(output / "protocol.json")
        assert len(protocol["runs"]) == 20  # Frozen before the very first child.
        assert command == protocol["runs"][index - 1]["command"]
        assert kwargs["cwd"] == repository
        assert kwargs["stderr"] is subprocess.STDOUT and kwargs["check"] is False
        assert all(kwargs["env"][key] == "1" for key in study.THREAD_ENV)
        assert kwargs["env"]["PYTHONPATH"].split(":")[0] == str(repository / "src")
        assert not (output / "summary.json").exists()

    calls = fake_process(monkeypatch, check)
    result = study.run(output)
    assert len(calls) == 20
    assert result["status"] == "commands_completed"
    assert result["commands_completed"] == result["planned_commands"] == 20
    assert "quality_pass" not in result
    assert result["quality_claim"].startswith("none;")
    records = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
    assert len(records) == 20
    for record in records:
        assert record["returncode"] == 0 and record["elapsed_s"] >= 0
        assert record["started_utc"] <= record["ended_utc"]
        assert (output / record["log"]).read_text().startswith("FAKE")
    assert read_json(output / "source_consistency.json")["unchanged"] is True
    assert not (output / "failure.json").exists()


@pytest.mark.parametrize("error", ["exit", "launch", "interrupt"])
def test_child_failures_stop_and_retain_partial_output(repository, tmp_path, monkeypatch, error):
    output, calls = tmp_path / "fake_failed", []

    def execute(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write("fake child diagnostic\n")
        if len(calls) == 2:
            if error == "launch":
                raise OSError("fake launch failed")
            if error == "interrupt":
                raise KeyboardInterrupt("fake interruption")
            return subprocess.CompletedProcess(command, 7)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(study.subprocess, "run", execute)
    with pytest.raises((RuntimeError, OSError, KeyboardInterrupt)):
        study.run(output, smoke=True)
    assert len(calls) == 2 and output.is_dir()
    assert not (output / "summary.json").exists()
    failure = read_json(output / "failure.json")
    assert failure["commands_completed"] == 1 and failure["status"] == "failed"
    ledger = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
    assert len(ledger) == 2
    assert ledger[-1]["returncode"] == (7 if error == "exit" else None)
    assert read_json(output / "source_consistency.json")["unchanged"] is True
    with pytest.raises(ValueError, match="exists"):
        study.run(output, smoke=True)
    assert len(calls) == 2


@pytest.mark.parametrize("change", ["edit", "add", "delete", "asset"])
def test_source_changes_stop_before_any_next_child(repository, tmp_path, monkeypatch, change):
    output = tmp_path / "fake_changed"

    def mutate(command, kwargs, count):
        path = repository / "src/wheel_legged_control/model.py"
        if change == "delete":
            path.unlink()
        elif change == "add":
            (path.parent / "new_source.py").write_text("# unexpected new module\n")
        elif change == "asset":
            (repository / "src/wheel_legged_control/d1/assets/texture.png").write_text("changed")
        else:
            path.write_text("# mutated during a fake command\n")

    calls = fake_process(monkeypatch, mutate)
    with pytest.raises(RuntimeError, match="source changed"):
        study.run(output, smoke=True)
    assert len(calls) == 1 and not (output / "summary.json").exists()
    consistency = read_json(output / "source_consistency.json")
    assert consistency["unchanged"] is False and len(consistency["changed"]) == 1


def test_source_inventory_is_actual_bytes_and_does_not_include_results(repository):
    hashes = study.source_hashes()
    assert len(hashes) == 6
    assert all(
        digest == hashlib.sha256((repository / name).read_bytes()).hexdigest()
        for name, digest in hashes.items()
    )
    output = repository / "results/unrelated.py"
    output.parent.mkdir()
    output.write_text("# unrelated result\n")
    assert study.source_hashes() == hashes


@pytest.mark.parametrize("kind", ["exists", "symlink", "parent_symlink", "source"])
def test_unsafe_output_is_rejected_before_any_child(repository, tmp_path, monkeypatch, kind):
    output = tmp_path / "bad_output"
    if kind == "exists":
        output.mkdir()
    elif kind == "symlink":
        output.symlink_to(tmp_path / "absent_target", target_is_directory=True)
    elif kind == "parent_symlink":
        output.symlink_to(repository, target_is_directory=True)
        output = output / "new_folder"
    else:
        output = repository / "src/new_results"
    calls = fake_process(monkeypatch)
    with pytest.raises(ValueError):
        study.run(output)
    assert calls == []


def test_missing_source_preflight_creates_no_output(repository, tmp_path, monkeypatch):
    (repository / study.CHILD_SCRIPT).unlink()
    output = tmp_path / "no_partial"
    calls = fake_process(monkeypatch)
    with pytest.raises(ValueError, match="source"):
        study.run(output)
    assert not output.exists() and calls == []


def test_symlink_directory_cannot_hide_untracked_source(repository, tmp_path, monkeypatch):
    outside = tmp_path / "external_source"
    outside.mkdir()
    (outside / "module.py").write_text("# outside source inventory\n")
    (repository / "src/alias").symlink_to(outside, target_is_directory=True)
    output = tmp_path / "no_partial_alias"
    calls = fake_process(monkeypatch)
    with pytest.raises(ValueError, match="symlink"):
        study.run(output)
    assert not output.exists() and calls == []


def test_smoke_protocol_is_separate_and_cli_defaults_cannot_reuse_formal(repository, monkeypatch):
    seen = []

    def fake_run(output, *, smoke=False):
        seen.append((output, smoke))
        return {"status": "commands_completed"}

    monkeypatch.setattr(study, "run", fake_run)
    study.main([])
    study.main(["--smoke"])
    assert seen == [
        (repository / "results/d1_shared_action_study", False),
        (repository / "results/d1_shared_action_smoke", True),
    ]


def test_real_smoke_protocol_construction_with_only_fake_children(
    repository, tmp_path, monkeypatch
):
    fake_process(monkeypatch)
    output = tmp_path / "fake_smoke"
    summary = study.run(output, smoke=True)
    protocol = read_json(output / "protocol.json")
    assert protocol["schema"] == "d1-shared-action-study-v1"
    assert protocol["kind"] == summary["kind"] == "smoke"
    assert protocol["training_seeds"] == [31001]
    assert protocol["timesteps_per_model"] == 512 and protocol["duration_s"] == 0.2
    assert protocol["interpretation"] == "interface/reload check only"
    assert summary["commands_completed"] == 8


def test_orchestrator_imports_only_standard_library():
    tree = ast.parse(Path(study.__file__).read_text())
    imports = {
        name.split(".")[0]
        for node in ast.walk(tree)
        for name in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module]
            if isinstance(node, ast.ImportFrom)
            else []
        )
    }
    assert imports <= sys.stdlib_module_names | {"__future__"}
