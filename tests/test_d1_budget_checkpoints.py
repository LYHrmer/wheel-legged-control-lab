"""Budget checkpoint contracts; tiny real PPO runs are not performance evidence."""

import hashlib
import importlib.util
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wheel_legged_control.d1 import budget_checkpoints as checkpoints
from wheel_legged_control.d1.budget_checkpoints import (
    BudgetCheckpointWriter,
    validate_checkpoint_budgets,
)
from wheel_legged_control.d1.ppo_update_audit import parameter_sha256

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_d1_locomotion_experiment.py"


@pytest.fixture(scope="module")
def experiment():
    spec = importlib.util.spec_from_file_location("budget_experiment_test_subject", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(path.read_text())


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fake_schedule(tmp_path, monkeypatch):
    """Explicit arithmetic adapter only; real saves/sidecars are tested below."""
    audit = tmp_path / "updates"
    audit.mkdir()
    events = []
    reference = object()
    learner = SimpleNamespace(
        num_timesteps=0, _n_updates=0, _audit_index=0, policy=torch.nn.Linear(1, 1, bias=False)
    )

    def save(path):
        events.append(("save", learner.num_timesteps, learner._n_updates))
        Path(path).write_bytes(b"explicit fake unit-test checkpoint")

    def metadata(model, env, output, *, extra):
        assert env is reference
        events.append(("metadata", learner.num_timesteps, learner._n_updates))
        with Path(output).open("x") as stream:
            json.dump({"extra": extra, "model_sha256": file_sha(model)}, stream)

    learner.save = save
    monkeypatch.setattr(checkpoints, "write_checkpoint_metadata", metadata)
    writer = BudgetCheckpointWriter(
        tmp_path / "checkpoints",
        (128, 256),
        workers=1,
        n_steps=128,
        training_seed=17,
        reference_env=reference,
        audit_directory=audit,
    )
    writer.begin_learning()

    def update():
        events.append(("train", learner.num_timesteps, learner._n_updates))
        with torch.no_grad():
            learner.policy.weight.add_(1)
        learner._n_updates += 4
        row = {
            "audit_index": learner._audit_index,
            "num_timesteps": learner.num_timesteps,
            "n_updates": learner._n_updates,
            "param_sha256_after": parameter_sha256(learner.policy),
        }
        with (audit / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        learner._audit_index += 1
        return "complete-train-result"

    return SimpleNamespace(
        writer=writer, learner=learner, update=update, events=events, root=tmp_path, audit=audit
    )


def test_save_runs_after_whole_train_and_manifest_binds_real_saved_bytes(fake_schedule):
    state = fake_schedule
    for budget in (128, 256):
        state.learner.num_timesteps = budget
        assert state.writer.execute_update(state.learner, state.update) == "complete-train-result"
    assert [event[0] for event in state.events] == ["train", "save", "metadata"] * 2
    manifest = read_json(state.root / "checkpoints/manifest.json")
    assert manifest["schema"] == "d1-budget-checkpoints-v1"
    assert manifest["complete"] is True and manifest["failed"] is False
    assert manifest["pending_budgets"] == []
    assert [row["budget"] for row in manifest["saved"]] == [128, 256]
    for index, row in enumerate(manifest["saved"]):
        assert row["num_timesteps"] == row["budget"]
        assert row["after_update_index"] == index
        assert row["n_updates"] == 4 * (index + 1)
        assert row["zip_sha256"] == file_sha(state.root / "checkpoints" / row["model_path"])
        assert row["metadata_sha256"] == file_sha(state.root / "checkpoints" / row["metadata_path"])
        assert row["elapsed_s"] >= 0 and row["saved_at_utc"]
    phases = [
        json.loads(line)
        for line in (state.root / "checkpoints/phases.jsonl").read_text().splitlines()
    ]
    assert phases


@pytest.mark.parametrize("exception", [RuntimeError("optimizer failed"), KeyboardInterrupt()])
def test_failed_or_interrupted_update_never_claims_completed(fake_schedule, exception):
    state = fake_schedule
    state.learner.num_timesteps = 128

    def fail():
        raise exception

    with pytest.raises(type(exception)):
        state.writer.execute_update(state.learner, fail)
    manifest = read_json(state.root / "checkpoints/manifest.json")
    assert manifest["complete"] is False and manifest["failed"] is True
    assert manifest["saved"] == []
    assert not list((state.root / "checkpoints").glob("budget*/checkpoint.zip"))


def test_metadata_failure_keeps_partial_checkpoint_without_claiming_success(
    fake_schedule, monkeypatch
):
    state = fake_schedule
    state.learner.num_timesteps = 128

    def fail_metadata(*args, **kwargs):
        raise OSError("explicit test storage failure")

    monkeypatch.setattr(checkpoints, "write_checkpoint_metadata", fail_metadata)
    with pytest.raises(OSError, match="storage failure"):
        state.writer.execute_update(state.learner, state.update)
    manifest = read_json(state.root / "checkpoints/manifest.json")
    assert manifest["complete"] is False and manifest["failed"] is True
    assert manifest["saved"] == []
    assert (state.root / "checkpoints/budget128/checkpoint.zip").is_file()
    assert not (state.root / "checkpoints/budget128/checkpoint.json").exists()


@pytest.mark.parametrize("mutation", ["wrong_step", "wrong_hash", "wrong_index", "wrong_updates"])
def test_audit_mismatch_fails_closed_before_saving(fake_schedule, mutation):
    state = fake_schedule
    state.learner.num_timesteps = 128

    def wrong_update():
        state.update()
        path = state.audit / "updates.jsonl"
        row = json.loads(path.read_text())
        key, value = {
            "wrong_step": ("num_timesteps", 256),
            "wrong_hash": ("param_sha256_after", "0" * 64),
            "wrong_index": ("audit_index", 1),
            "wrong_updates": ("n_updates", 5),
        }[mutation]
        row[key] = value
        path.write_text(json.dumps(row) + "\n")

    with pytest.raises((ValueError, RuntimeError)):
        state.writer.execute_update(state.learner, wrong_update)
    assert [event[0] for event in state.events] == ["train"]
    assert read_json(state.root / "checkpoints/manifest.json")["complete"] is False


@pytest.mark.parametrize(
    "field,value",
    [("num_timesteps", 128.0), ("num_timesteps", True), ("num_timesteps", 256), ("_n_updates", 1)],
)
def test_invalid_update_counters_do_not_execute_optimizer(fake_schedule, field, value):
    state = fake_schedule
    state.learner.num_timesteps = 128
    setattr(state.learner, field, value)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        state.writer.execute_update(state.learner, state.update)
    assert state.events == []


def test_completed_writer_cannot_execute_an_extra_update(fake_schedule):
    state = fake_schedule
    for budget in (128, 256):
        state.learner.num_timesteps = budget
        state.writer.execute_update(state.learner, state.update)
    previous_events = list(state.events)
    state.learner.num_timesteps = 384
    with pytest.raises((ValueError, RuntimeError)):
        state.writer.execute_update(state.learner, state.update)
    assert state.events == previous_events


def test_failed_writer_cannot_resume_on_the_same_directory(fake_schedule):
    state = fake_schedule
    state.learner.num_timesteps = 128

    def fail():
        raise RuntimeError("explicit failed optimizer")

    with pytest.raises(RuntimeError):
        state.writer.execute_update(state.learner, fail)
    with pytest.raises((ValueError, RuntimeError)):
        state.writer.execute_update(state.learner, state.update)
    assert state.events == []


def test_altered_learner_identity_is_rejected_before_update(fake_schedule):
    state = fake_schedule
    state.learner.num_timesteps = 128
    state.writer.execute_update(state.learner, state.update)
    other = SimpleNamespace(**vars(state.learner))
    other.num_timesteps = 256
    previous_events = list(state.events)
    with pytest.raises((ValueError, RuntimeError)):
        state.writer.execute_update(other, state.update)
    assert state.events == previous_events


def test_preexisting_budget_directory_is_never_overwritten(fake_schedule):
    state = fake_schedule
    target = state.root / "checkpoints/budget128"
    target.mkdir()
    marker = target / "checkpoint.zip"
    marker.write_bytes(b"previous unrelated checkpoint")
    state.learner.num_timesteps = 128
    with pytest.raises((FileExistsError, RuntimeError)):
        state.writer.execute_update(state.learner, state.update)
    assert marker.read_bytes() == b"previous unrelated checkpoint"
    assert not (target / "checkpoint.json").exists()


def test_writer_rejects_existing_and_symlink_outputs_without_changes(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "old.txt"
    marker.write_text("keep")
    symlink = tmp_path / "link"
    symlink.symlink_to(existing, target_is_directory=True)
    for target in (existing, symlink, symlink / "new"):
        with pytest.raises((ValueError, FileExistsError, RuntimeError)):
            BudgetCheckpointWriter(
                target,
                [128],
                workers=1,
                n_steps=128,
                training_seed=17,
                reference_env=object(),
                audit_directory=tmp_path / "updates",
            )
    assert marker.read_text() == "keep"


def test_none_preserves_legacy_and_valid_budgets_preserve_order():
    assert validate_checkpoint_budgets(None, total_steps=256, workers=1) is None
    assert validate_checkpoint_budgets([128, 256], total_steps=256, workers=1) == (128, 256)
    assert validate_checkpoint_budgets([512, 1024], total_steps=1024, workers=4) == (512, 1024)


@pytest.mark.parametrize(
    "budgets",
    [
        [],
        [128, 128],
        [256, 128],
        [0, 256],
        [-128, 256],
        [True, 256],
        [128.0, 256],
        [127, 256],
        [128],
        "128",
    ],
)
def test_invalid_budgets_are_not_coerced(budgets):
    with pytest.raises((TypeError, ValueError)):
        validate_checkpoint_budgets(budgets, total_steps=256, workers=1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("total_steps", True),
        ("total_steps", 256.0),
        ("workers", True),
        ("workers", 0),
        ("n_steps", 0),
        ("n_steps", 128.0),
    ],
)
def test_budget_dimensions_are_strict_integers(field, value):
    arguments = {"total_steps": 256, "workers": 1, "n_steps": 128}
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        validate_checkpoint_budgets([128, 256], **arguments)


@pytest.mark.parametrize(
    "mode,budgets",
    [
        ("evaluate", ["128", "256"]),
        ("benchmark", ["128", "256"]),
        ("train", ["128", "128"]),
        ("train", ["128"]),
        ("train", ["127", "256"]),
    ],
)
def test_cli_rejects_invalid_schedule_before_creating_output(
    experiment, monkeypatch, tmp_path, mode, budgets
):
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            mode,
            "--output",
            str(output),
            "--steps",
            "256",
            "--workers",
            "1",
            "--checkpoint-budgets",
            *budgets,
        ],
    )
    with pytest.raises(SystemExit) as caught:
        experiment.main()
    assert caught.value.code == 2
    assert not output.exists()


def test_cli_declares_budget_protocol_before_dispatch(experiment, monkeypatch, tmp_path):
    output = tmp_path / "protocol-only"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "train",
            "--output",
            str(output),
            "--steps",
            "256",
            "--workers",
            "1",
            "--checkpoint-budgets",
            "128",
            "256",
        ],
    )
    monkeypatch.setattr(experiment, "snapshot", lambda *_: {})
    monkeypatch.setattr(experiment, "verify_source", lambda *_: None)
    calls = []

    def dispatched(args, directory):
        protocol = read_json(directory / "protocol.json")
        assert protocol["arguments"]["checkpoint_budgets"] == [128, 256]
        assert protocol["checkpoint_budgets"] == [128, 256]
        assert protocol["checkpoint_save_timing"] == "after_complete_ppo_train_call"
        assert (
            protocol["selection_rule"]
            == "keep all predeclared budgets; no best or holdout checkpoint selection"
        )
        calls.append(args.steps)

    monkeypatch.setattr(experiment, "train", dispatched)
    experiment.main()
    assert calls == [256]


def _args(experiment, output, mode, enabled):
    flags = [
        "train",
        "--output",
        str(output),
        "--steps",
        "256",
        "--workers",
        "1",
        "--duration",
        "0.2",
        "--seed",
        "17",
        "--action-mode",
        mode,
        "--source",
        "imu_encoder_fusion",
        "--wheel-kp",
        "0.55",
        "--wheel-ki",
        "1.5",
        "--yaw-feedback-gain",
        "4",
        "--leg-feedback-scale",
        "1",
        "--attitude-feedback-scale",
        "0.25",
    ]
    if enabled:
        flags += ["--checkpoint-budgets", "128", "256"]
    return experiment.parser().parse_args(flags)


@pytest.fixture(scope="module", params=["shared2", "independent8"])
def tiny_pair(experiment, tmp_path_factory, request):
    from wheel_legged_control.d1.ppo_update_audit import AuditedPPO

    experiment.torch.set_num_threads(1)
    root = tmp_path_factory.mktemp(f"budget-{request.param}")
    runs = {}
    original_learn = AuditedPPO.learn
    learn_calls = []

    def counted_learn(learner, *args, **kwargs):
        learn_calls.append((id(learner), kwargs.get("total_timesteps"), learner.num_timesteps))
        return original_learn(learner, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(AuditedPPO, "learn", counted_learn)
        for enabled in (False, True):
            directory = root / ("budget" if enabled else "legacy")
            directory.mkdir()
            args = _args(experiment, directory, request.param, enabled)
            result = experiment.train(args, directory)
            runs[enabled] = (directory, args, result)
    assert len(learn_calls) == 2
    assert [(steps, start) for _, steps, start in learn_calls] == [(256, 0), (256, 0)]
    return request.param, runs


def test_real_two_rollout_training_is_unchanged_by_checkpoint_saves(tiny_pair):
    _, runs = tiny_pair
    old, new = runs[False][0], runs[True][0]
    old_updates = [
        json.loads(line) for line in (old / "updates/updates.jsonl").read_text().splitlines()
    ]
    new_updates = [
        json.loads(line) for line in (new / "updates/updates.jsonl").read_text().splitlines()
    ]
    assert len(old_updates) == len(new_updates) == 2
    for left, right in zip(old_updates, new_updates, strict=True):
        assert left["num_timesteps"] == right["num_timesteps"]
        assert left["param_sha256_before"] == right["param_sha256_before"]
        assert left["param_sha256_after"] == right["param_sha256_after"]
        with (
            np.load(old / "updates" / left["npz"]) as a,
            np.load(new / "updates" / right["npz"]) as b,
        ):
            assert set(a.files) == set(b.files)
            for field in a.files:
                np.testing.assert_array_equal(a[field], b[field], err_msg=field)
    assert not (old / "checkpoints").exists()
    assert set(read_json(old / "checkpoint.json")["extra"]) == {"training_seed", "num_timesteps"}
    assert runs[True][2]["num_timesteps"] == 256
    assert runs[True][2]["learn_calls"] == 1
    assert runs[True][2]["training_mode"] == "single_continuous_learn"
    assert runs[True][2]["checkpoint_budgets"] == [128, 256]


def test_real_each_budget_is_loadable_and_matches_its_completed_audited_update(
    experiment, tiny_pair
):
    from wheel_legged_control.d1.locomotion_checkpoint import load_locomotion_policy

    mode, runs = tiny_pair
    directory, args, _ = runs[True]
    updates = [
        json.loads(line) for line in (directory / "updates/updates.jsonl").read_text().splitlines()
    ]
    env = experiment.make_env(
        args.baseline,
        0,
        args.duration,
        args.source,
        args.history,
        args.delay_randomization,
        wheel_control=experiment.wheel_control_config(args),
        action_mode=args.action_mode,
        terrain_suite=args.terrain_suite,
    )
    try:
        env.reset(seed=args.seed)
        for budget, row in zip((128, 256), updates, strict=True):
            folder = directory / "checkpoints" / f"budget{budget}"
            model, sidecar = folder / "checkpoint.zip", folder / "checkpoint.json"
            metadata = read_json(sidecar)
            assert metadata["extra"]["training_seed"] == 17
            assert metadata["extra"]["num_timesteps"] == budget
            assert metadata["extra"]["checkpoint_budget"] == budget
            assert metadata["extra"]["after_update_index"] == budget // 128 - 1
            assert metadata["extra"]["param_sha256_after"] == row["param_sha256_after"]
            policy = load_locomotion_policy(model, sidecar, env)
            assert policy.action_space.shape == ((2,) if mode == "shared2" else (8,))
            assert parameter_sha256(policy.policy) == row["param_sha256_after"]
            with zipfile.ZipFile(model) as archive:
                data = json.loads(archive.read("data"))
            assert "_budget_writer" not in data
            assert "_audit_directory" not in data
        final = load_locomotion_policy(
            directory / "checkpoint.zip", directory / "checkpoint.json", env
        )
        assert parameter_sha256(final.policy) == updates[-1]["param_sha256_after"]
    finally:
        env.close()


def test_real_reload_witness_and_save_manifest_have_exact_budget_matrix(tiny_pair):
    _, runs = tiny_pair
    directory = runs[True][0]
    manifest = read_json(directory / "checkpoints/manifest.json")
    witness = read_json(directory / "checkpoint_reload.json")
    training = read_json(directory / "training.json")
    assert manifest["complete"] is True and manifest["failed"] is False
    assert witness["schema"] == "d1-budget-checkpoint-reload-v1"
    assert witness["phase"] == "after_training"
    assert witness["training_final_num_timesteps"] == 256
    assert witness["training_finished_at_utc"] == training["training_finished_at_utc"]
    started = datetime.fromisoformat(training["training_started_at_utc"])
    ended = datetime.fromisoformat(training["training_finished_at_utc"])
    assert started < ended <= datetime.fromisoformat(witness["started_at_utc"])
    assert witness["elapsed_s"] > 0
    assert [row["budget"] for row in manifest["saved"]] == [128, 256]
    assert [row["budget"] for row in witness["records"]] == [128, 256]
    assert (
        0
        < manifest["saved"][0]["elapsed_s"]
        < manifest["saved"][1]["elapsed_s"]
        < training["wall_s"]
    )
    for saved, reloaded in zip(manifest["saved"], witness["records"], strict=True):
        assert saved["after_update_index"] == reloaded["after_update_index"]
        assert saved["param_sha256_after"] == reloaded["reloaded_policy_param_sha256"]
        assert saved["zip_sha256"] == reloaded["zip_sha256"]
        assert saved["metadata_sha256"] == reloaded["metadata_sha256"]
        assert started <= datetime.fromisoformat(saved["saved_at_utc"]) <= ended
        assert 0 < saved["save_duration_s"] <= saved["elapsed_s"]
        assert reloaded["zip_sha256"] == file_sha(directory / "checkpoints" / saved["model_path"])
        assert reloaded["metadata_sha256"] == file_sha(
            directory / "checkpoints" / saved["metadata_path"]
        )
    final = witness["final_checkpoint"]
    assert final["num_timesteps"] == 256 and final["after_update_index"] == 1
    assert final["model_path"] == "checkpoint.zip" and final["metadata_path"] == "checkpoint.json"
    assert final["reloaded_policy_param_sha256"] == manifest["saved"][-1]["param_sha256_after"]
    assert final["zip_sha256"] == file_sha(directory / "checkpoint.zip")
    assert final["metadata_sha256"] == file_sha(directory / "checkpoint.json")
