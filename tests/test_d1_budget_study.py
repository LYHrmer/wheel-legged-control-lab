"""Offline command and process contracts, not shortened performance experiments."""

from __future__ import annotations

import ast
import csv
import ctypes
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import run_d1_budget_study as study


def option(command, flag):
    return command[command.index(flag) + 1]


def read_json(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize(
    "smoke,count,cases,models,budgets,total,calls",
    [
        (False, 56, 200, 6, [32768, 65536, 131072, 262144], 1572864, 3072),
        (True, 12, 40, 2, [512, 1024], 2048, 4),
    ],
)
def test_complete_protocol_and_continuous_training_counts(
    tmp_path, smoke, count, cases, models, budgets, total, calls
):
    output = tmp_path / "uncreated"
    protocol = study.build_protocol(output, smoke=smoke)
    assert protocol["schema"] == "d1-budget-study-v1"
    assert protocol["training_mode"] == "single_continuous_learn"
    assert protocol["checkpoint_budgets"] == budgets
    assert protocol["total_timesteps_per_model"] == budgets[-1]
    assert protocol["continuous_training_trajectories"] == len(protocol["models"]) == models
    assert protocol["checkpoint_count"] == len(protocol["checkpoints"]) == models * len(budgets)
    assert protocol["planned_commands"] == len(protocol["runs"]) == count
    assert protocol["planned_evaluation_cases"] == cases
    assert protocol["total_training_transitions"] == total
    assert protocol["expected_train_calls"] == calls
    assert protocol["n_steps"] == 128 and protocol["workers"] == 4
    assert protocol["evaluation_seeds"] == {"development": [1017, 1029], "holdout": [4617, 4629]}
    assert "elapsed_s" not in protocol and "created_utc" not in protocol
    assert not output.exists()


@pytest.mark.parametrize("smoke", [False, True])
def test_all_training_then_all_development_then_holdout(tmp_path, smoke):
    runs = study.build_runs(tmp_path / "new", smoke=smoke)
    names = [item["name"] for item in runs]
    seeds = [31001] if smoke else [31000, 32000, 33000]
    budgets = [512, 1024] if smoke else [32768, 65536, 131072, 262144]
    models = [f"{mode}_seed{seed}" for mode in ("shared2", "independent8") for seed in seeds]
    assert names == [
        "zero_development",
        *models,
        *[f"{name}_budget{budget}_development" for name in models for budget in budgets],
        "zero_holdout",
        *[f"{name}_budget{budget}_holdout" for name in models for budget in budgets],
    ]
    assert len(names) == len(set(names))
    last_train = max(i for i, item in enumerate(runs) if item["mode"] == "train")
    first_trained_dev = next(i for i, item in enumerate(runs) if item["checkpoint_id"])
    assert last_train < first_trained_dev
    assert max(i for i, item in enumerate(runs) if item["split"] == "development") < names.index(
        "zero_holdout"
    )


@pytest.mark.parametrize("smoke", [False, True])
def test_argv_and_checkpoint_associations(tmp_path, smoke):
    output = tmp_path / "new"
    protocol = study.build_protocol(output, smoke=smoke)
    model_ids = {model["model_id"] for model in protocol["models"]}
    checkpoints = {item["checkpoint_id"]: item for item in protocol["checkpoints"]}
    for item in protocol["runs"]:
        command = item["command"]
        assert command[:3] == [sys.executable, str(study.ROOT / study.CHILD_SCRIPT), item["mode"]]
        assert Path(option(command, "--output")) == output / item["name"]
        assert option(command, "--baseline") == "wheel_leg"
        assert option(command, "--source") == "imu_encoder_fusion"
        assert option(command, "--terrain-suite") == "budget_compare_v1"
        assert option(command, "--action-mode") == item["action_mode"]
        assert int(option(command, "--seed")) == item["seed"]
        assert option(command, "--history") == "1"
        assert option(command, "--measurement-delay") == "0"
        assert option(command, "--workers") == "4"
        assert float(option(command, "--duration")) == (0.2 if smoke else 60)
        assert "--delay-randomization" not in command
        for name, value in protocol["controller_parameters"].items():
            assert float(option(command, "--" + name.replace("_", "-"))) == value
        if item["mode"] == "train":
            assert item["model_name"] == item["name"] in model_ids
            assert item["checkpoint_id"] is None and item["phase"] == "train"
            assert item["budget"] == protocol["total_timesteps_per_model"]
            assert command[command.index("--checkpoint-budgets") + 1 :] == list(
                map(str, protocol["checkpoint_budgets"])
            )
            assert "--policy" not in command and "--split" not in command
        elif item["checkpoint_id"]:
            checkpoint = checkpoints[item["checkpoint_id"]]
            assert item["budget"] == checkpoint["budget"] == int(option(command, "--steps"))
            assert item["model_name"] == checkpoint["model_id"] in model_ids
            assert checkpoint["expected_update_index"] == item["budget"] // 512 - 1
            assert Path(option(command, "--policy")) == output / checkpoint["relative_zip"]
            assert Path(option(command, "--metadata")) == output / checkpoint["relative_metadata"]
            assert item["split"] == item["phase"] == option(command, "--split")
            assert "--checkpoint-budgets" not in command
        else:
            assert item["name"] in ("zero_development", "zero_holdout")
            assert item["budget"] is None and item["model_name"] is None
            assert item["action_mode"] == "independent8" and item["seed"] == 31000
            assert "--policy" not in command and "--metadata" not in command
    assert not output.exists()


def test_preregistered_ppo_settings_match_actual_child():
    from scripts.run_d1_locomotion_experiment import PPO_SETTINGS

    assert study.PPO_SETTINGS == PPO_SETTINGS


def test_preregistered_terrains_match_actual_suite(tmp_path):
    from wheel_legged_control.d1.locomotion_terrain import locomotion_terrain_configs

    protocol = study.build_protocol(tmp_path / "new")
    for split in ("train", "development", "holdout"):
        actual = [
            json.loads(config.to_json())
            for config in locomotion_terrain_configs(split, suite="budget_compare_v1")
        ]
        assert protocol["terrains"][split] == actual


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / "fake_repository"
    files = {
        study.SCRIPT: "# fixture orchestrator\n",
        study.CHILD_SCRIPT: "# fixture child\n",
        "pyproject.toml": "[project]\nname='fixture'\n",
        "src/wheel_legged_control/model.py": "# fixture dynamics\n",
        "src/wheel_legged_control/assets/cart.xml": "<fixture/>\n",
        "src/wheel_legged_control/d1/assets/model.xml": "<fixture/>\n",
        "src/wheel_legged_control/d1/assets/LICENSE.txt": "fixture license\n",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    monkeypatch.setattr(study, "ROOT", root)
    return root


def _fake_artifacts(command):
    output = Path(option(command, "--output"))
    output.mkdir()
    archive = b"fixture archive bytes; not a real experiment"
    (output / "source.tar.gz").write_bytes(archive)
    (output / "source.json").write_text(
        json.dumps({"sha256": {}, "archive_sha256": hashlib.sha256(archive).hexdigest()})
    )
    (output / "protocol.json").write_text('{"fixture": true}')
    result = "training.json" if command[2] == "train" else "evaluation.json"
    (output / result).write_text('{"fixture": true}')


def fake_children(monkeypatch, callback=None):
    calls, kills = [], []

    class Child:
        def __init__(self, command, **kwargs):
            self.command, self.kwargs = command, kwargs
            calls.append(command)
            self.index, self.pid, self.returncode = len(calls), 90000 + len(calls), None
            kwargs["stdout"].write("FAKE child: orchestration tests only\n")

        def wait(self, timeout=None):
            if timeout is not None:
                self.returncode = -15
                return self.returncode
            result = callback(self.command, self.kwargs, self.index) if callback else None
            _fake_artifacts(self.command)
            self.returncode = 0 if result is None else result
            return self.returncode

    monkeypatch.setattr(study.subprocess, "Popen", Child)

    def kill_group(pid, sig):
        if sig == 0:
            raise ProcessLookupError
        kills.append((pid, sig))

    monkeypatch.setattr(study.os, "killpg", kill_group)
    return calls, kills


def test_success_records_paired_json_csv_and_child_artifact_hashes(
    repository, tmp_path, monkeypatch
):
    output = tmp_path / "fake_run"

    def inspect(command, kwargs, index):
        protocol = read_json(output / "protocol.json")
        assert command == protocol["runs"][index - 1]["command"]
        assert kwargs["cwd"] == repository and kwargs["start_new_session"] is True
        assert kwargs["stderr"] is subprocess.STDOUT
        assert all(kwargs["env"][key] == "1" for key in study.THREAD_ENV)
        assert kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(repository / "src")
        assert not (output / "summary.json").exists()

    calls, kills = fake_children(monkeypatch, inspect)
    summary = study.run(output, smoke=True)
    assert len(calls) == summary["commands_completed"] == summary["planned_commands"] == 12
    assert summary["status"] == "commands_completed" and summary["quality_claim"].startswith(
        "none;"
    )
    assert not kills and not (output / "failure.json").exists()
    records = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
    with (output / "runs.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(records) == len(rows) == 12
    for sequence, (record, row) in enumerate(zip(records, rows), 1):
        assert record["sequence"] == int(row["sequence"]) == sequence
        assert record["command"] == json.loads(row["command_json"])
        assert record["child_artifact_sha256"] == json.loads(row["child_artifact_sha256_json"])
        assert record["returncode"] == 0 and record["elapsed_s"] >= 0
        assert record["started_utc"] <= record["ended_utc"]
        if sequence > 1:
            assert records[sequence - 2]["ended_utc"] <= record["started_utc"]
        for name, expected in record["child_artifact_sha256"].items():
            assert (
                hashlib.sha256((output / record["name"] / name).read_bytes()).hexdigest()
                == expected
            )
    source = read_json(output / "source.json")
    assert source["sha256"] == read_json(output / "protocol.json")["source_sha256"]
    assert source["archived_extras"][study.SCRIPT] == "orchestrator_source.py"
    assert len(source["archived_extras"]) == 3
    for name, local in source["archived_extras"].items():
        assert (output / local).read_bytes() == (repository / name).read_bytes()
    assert read_json(output / "source_consistency.json")["unchanged"] is True


@pytest.mark.parametrize("failure", ["exit", "launch", "interrupt", "system_exit", "sigterm"])
def test_failure_stops_child_group_records_error_and_preserves_outputs(
    repository, tmp_path, monkeypatch, failure
):
    output = tmp_path / "failed"
    previous = signal.getsignal(signal.SIGTERM)

    def fail(command, kwargs, index):
        if index == 2:
            if failure == "exit":
                return 7
            if failure == "launch":
                raise OSError("fixture execution error")
            if failure == "interrupt":
                raise KeyboardInterrupt("fixture interruption")
            if failure == "system_exit":
                raise SystemExit(3)
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    calls, kills = fake_children(monkeypatch, fail)
    with pytest.raises((RuntimeError, OSError, KeyboardInterrupt, SystemExit)):
        study.run(output, smoke=True)
    assert len(calls) == 2 and kills == [(90002, signal.SIGTERM)]
    assert signal.getsignal(signal.SIGTERM) == previous
    assert not (output / "summary.json").exists()
    error = read_json(output / "failure.json")
    assert error["commands_completed"] == 1 and error["status"] == "failed"
    records = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
    assert len(records) == 2 and records[-1]["returncode"] == (7 if failure == "exit" else None)
    assert read_json(output / "source_consistency.json")["unchanged"] is True
    with pytest.raises(ValueError, match="exists"):
        study.run(output, smoke=True)
    assert len(calls) == 2


def test_launch_error_records_without_nonexistent_child_cleanup(repository, tmp_path, monkeypatch):
    def launch(*args, **kwargs):
        raise OSError("fixture Popen launch failed")

    monkeypatch.setattr(study.subprocess, "Popen", launch)
    output = tmp_path / "launch_failure"
    with pytest.raises(OSError):
        study.run(output, smoke=True)
    assert read_json(output / "failure.json")["commands_completed"] == 0
    record = json.loads((output / "runs.jsonl").read_text())
    assert record["returncode"] is None and record["error_type"] == "OSError"


def test_cleanup_escalates_to_sigkill_and_restores_handlers(monkeypatch):
    signals, waits = [], []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    class Child:
        pid = 90000

        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                raise subprocess.TimeoutExpired("fixture", timeout)
            return -9

    monkeypatch.setattr(
        study.os, "killpg", lambda pid, sig: signals.append((pid, sig)) if sig else None
    )
    study._stop_child(Child())
    assert signals == [(90000, signal.SIGTERM), (90000, signal.SIGKILL)]
    assert waits == [5, 5]
    assert previous == {sig: signal.getsignal(sig) for sig in previous}


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group and child-subreaper test")
def test_real_cleanup_kills_worker_after_group_leader_has_already_exited():
    # Adopt the fixture grandchild so this test also reaps it, instead of leaving
    # an orphan zombie for the container's init process. No simulation is started.
    libc = ctypes.CDLL(None)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    worker_pid, process = None, None
    child_code = (
        "import os,signal\n"
        "if os.fork(): os._exit(0)\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        "print(os.getpid(),flush=True)\n"
        "os.close(1)\nos.close(2)\n"
        "while True: signal.pause()\n"
    )
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", child_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stdout, _ = process.communicate(timeout=3)
        worker_pid = int(stdout.strip())
        assert process.returncode == 0  # The group leader is already reaped.
        assert os.getpgid(worker_pid) == process.pid
        study._stop_child(process)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            reaped, status = os.waitpid(worker_pid, os.WNOHANG)
            if reaped:
                assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL
                worker_pid = None
                break
            time.sleep(0.01)
        else:
            pytest.fail("TERM-ignoring worker survived group cleanup")
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
        if worker_pid is not None:
            os.waitpid(worker_pid, 0)
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


@pytest.mark.parametrize("change", ["edit", "add", "delete", "asset", "non_python"])
def test_source_inventory_changes_stop_after_one_child(repository, tmp_path, monkeypatch, change):
    def mutate(command, kwargs, index):
        path = repository / "src/wheel_legged_control/model.py"
        if change == "delete":
            path.unlink()
        elif change == "add":
            (path.parent / "new.py").write_text("# fixture change\n")
        elif change == "asset":
            (repository / "src/wheel_legged_control/d1/assets/model.xml").write_text("changed")
        elif change == "non_python":
            (path.parent / "parameters.json").write_text("{}")
        else:
            path.write_text("# fixture change\n")

    calls, _ = fake_children(monkeypatch, mutate)
    output = tmp_path / "source_changed"
    with pytest.raises(RuntimeError, match="source changed"):
        study.run(output, smoke=True)
    assert len(calls) == 1 and not (output / "summary.json").exists()
    consistency = read_json(output / "source_consistency.json")
    assert consistency["unchanged"] is False and len(consistency["changed"]) == 1


def test_successful_child_requires_complete_hash_bound_artifacts(repository, tmp_path, monkeypatch):
    fake_children(monkeypatch)
    original = study._child_artifacts

    def corrupt(directory, mode):
        (directory / "source.tar.gz").write_bytes(b"fixture corruption")
        return original(directory, mode)

    monkeypatch.setattr(study, "_child_artifacts", corrupt)
    output = tmp_path / "corrupt_child"
    with pytest.raises(ValueError, match="archive SHA256"):
        study.run(output, smoke=True)
    record = json.loads((output / "runs.jsonl").read_text())
    assert record["returncode"] == 0 and record["error_type"] == "ValueError"
    assert read_json(output / "failure.json")["commands_completed"] == 0


def test_inventory_ignores_results_and_python_caches_but_hashes_all_source(repository):
    before = study.source_hashes()
    assert len(before) == 7
    for name, expected in before.items():
        assert hashlib.sha256((repository / name).read_bytes()).hexdigest() == expected
    cache = repository / "src/wheel_legged_control/__pycache__/generated.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"runtime bytecode")
    results = repository / "results/unrelated.py"
    results.parent.mkdir()
    results.write_text("unrelated")
    assert study.source_hashes() == before


@pytest.mark.parametrize("kind", ["traversal", "absolute", "source_alias", "output_alias"])
def test_archived_extras_reject_escape_and_aliases(repository, tmp_path, kind):
    output = tmp_path / "extras"
    output.mkdir()
    hashes = {study.SCRIPT: hashlib.sha256((repository / study.SCRIPT).read_bytes()).hexdigest()}
    if kind == "traversal":
        hashes = {"../outside.txt": "0" * 64}
    elif kind == "absolute":
        hashes = {str(tmp_path / "outside.txt"): "0" * 64}
    elif kind == "source_alias":
        original = repository / study.SCRIPT
        target = tmp_path / "fixture_source.py"
        target.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(target)
    else:
        (output / "orchestrator_source.py").symlink_to(tmp_path / "never_created")
    with pytest.raises(ValueError):
        study._archive_extras(output, hashes)
    assert not (tmp_path / "never_created").exists()


@pytest.mark.parametrize(
    "kind",
    [
        "existing_dir",
        "existing_file",
        "symlink",
        "parent_alias",
        "traversal",
        "source",
        "scripts",
        "tests",
    ],
)
def test_unsafe_or_reused_outputs_rejected_even_in_dry_run(repository, tmp_path, monkeypatch, kind):
    output = tmp_path / "unsafe"
    if kind == "existing_dir":
        output.mkdir()
    elif kind == "existing_file":
        output.write_text("user data")
    elif kind == "symlink":
        output.symlink_to(tmp_path / "nonexistent")
    elif kind == "parent_alias":
        output.symlink_to(repository, target_is_directory=True)
        output /= "fresh"
    elif kind == "traversal":
        output = tmp_path / ".." / "fresh"
    else:
        output = repository / ({"source": "src"}.get(kind, kind)) / "fresh"
    calls, _ = fake_children(monkeypatch)
    with pytest.raises(ValueError):
        study.run(output, dry_run=True)
    assert not calls


@pytest.mark.parametrize("kind", ["missing", "symlink"])
def test_invalid_source_preflight_leaves_no_partial_output(repository, tmp_path, monkeypatch, kind):
    if kind == "missing":
        (repository / study.CHILD_SCRIPT).unlink()
    else:
        (repository / "src/alias").symlink_to(tmp_path, target_is_directory=True)
    calls, _ = fake_children(monkeypatch)
    output = tmp_path / "no_partial"
    with pytest.raises(ValueError, match="source"):
        study.run(output, dry_run=True)
    assert not calls and not output.exists()


def test_dry_run_cli_prints_entire_protocol_without_writes_or_processes(
    repository, tmp_path, monkeypatch, capsys
):
    calls, _ = fake_children(monkeypatch)
    output = tmp_path / "dry_run"
    before = {
        p.relative_to(repository): p.read_bytes() for p in repository.rglob("*") if p.is_file()
    }
    protocol = study.main(["--output", str(output), "--smoke", "--dry-run"])
    assert json.loads(capsys.readouterr().out) == protocol
    assert len(protocol["runs"]) == 12 and protocol["source_sha256"]
    assert not output.exists() and not calls
    assert before == {
        p.relative_to(repository): p.read_bytes() for p in repository.rglob("*") if p.is_file()
    }


def test_cli_defaults_keep_smoke_separate(repository, monkeypatch, capsys):
    calls = []

    def fake_run(output, **kwargs):
        calls.append((output, kwargs))
        return {"fixture": True}

    monkeypatch.setattr(study, "run", fake_run)
    study.main([])
    study.main(["--smoke", "--dry-run"])
    assert calls == [
        (repository / "results/d1_budget_study", {"smoke": False, "dry_run": False}),
        (repository / "results/d1_budget_smoke", {"smoke": True, "dry_run": True}),
    ]


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
