"""Same-controller policy mapping and experiment accounting, without PPO training."""

import importlib.util
import json
import pickle
import sys
from dataclasses import astuple
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest
import torch

from wheel_legged_control.d1.locomotion_terrain import locomotion_terrain_configs

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_d1_locomotion_experiment.py"
OLD_ROWS = {
    "train": (
        ("straight", 1.0, 0.5, 0.005, 0.6, 0.9, 0.0, 0.2, 0.005),
        ("straight", -1.5, -0.5, 0.01, 0.9, 1.2, 0.6, 0.8, 0.01),
        ("left_offset", 1.5, -1.0, 0.0075, 0.6, 1.2, 0.6, 0.2, 0.0075),
        ("left_offset", -1.0, 1.0, 0.005, 0.9, 0.9, 0.0, 0.8, 0.005),
    ),
    "development": (
        ("right_offset", 1.25, -0.75, 0.006, 0.75, 1.05, 0.3, 0.5, 0.006),
        ("right_offset", -1.25, 0.75, 0.008, 0.85, 1.15, 0.9, 1.1, 0.008),
    ),
    "holdout": (
        ("s_bend", 1.4, 0.65, 0.009, 0.7, 1.0, 1.4, 1.6, 0.009),
        ("diagonal", -1.4, -0.65, 0.0065, 0.8, 1.1, 2.0, 2.2, 0.0065),
    ),
}
NEW_ROWS = {
    "development": (
        ("right_offset", 1.30, -0.70, 0.0065, 0.77, 1.07, 0.45, 0.65, 0.0065),
        ("right_offset", -1.30, 0.70, 0.0070, 0.87, 1.17, 1.05, 1.25, 0.0070),
    ),
    "holdout": (
        ("s_bend", 1.35, 0.60, 0.0085, 0.72, 1.02, 1.55, 1.75, 0.0085),
        ("diagonal", -1.35, -0.60, 0.0070, 0.82, 1.12, 2.15, 2.35, 0.0070),
    ),
}


@pytest.fixture(scope="module")
def experiment():
    spec = importlib.util.spec_from_file_location("d1_action_comparison_subject", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # Top-level partials must survive spawn pickling.
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("split", OLD_ROWS)
def test_original_terrain_values_and_default_are_unchanged(split):
    old = locomotion_terrain_configs(split)
    assert old == locomotion_terrain_configs(split, suite="v1")
    assert tuple(astuple(c)[:-1] for c in old) == OLD_ROWS[split]


@pytest.mark.parametrize("split", ("train", "development", "holdout"))
def test_comparison_suite_has_frozen_values_and_only_new_parameter_instances(split):
    new = locomotion_terrain_configs(split, suite="action_compare_v1")
    expected = OLD_ROWS["train"] if split == "train" else NEW_ROWS[split]
    assert tuple(astuple(c)[:-1] for c in new) == expected
    if split != "train":
        all_old = {c.to_json() for name in OLD_ROWS for c in locomotion_terrain_configs(name)}
        assert all(c.to_json() not in all_old for c in new)


@pytest.mark.parametrize("split,suite", (("bad", "v1"), ("train", "bad"), ("bad", "bad")))
def test_unknown_terrain_split_or_suite_is_rejected(split, suite):
    with pytest.raises(ValueError):
        locomotion_terrain_configs(split, suite=suite)


def _args(experiment, output, *flags):
    return experiment.parser().parse_args(["evaluate", "--output", str(output), *flags])


def test_cli_defaults_and_explicit_contracts(experiment, tmp_path):
    args = _args(experiment, tmp_path)
    assert args.action_mode is None and args.terrain_suite == "v1"
    shared = experiment.declared_action_contract("wheel_leg", "shared2")
    assert shared == {
        "action_mode": "shared2",
        "action_schema": "d1-shared-wheel-leg-extension-speed-v1",
        "policy_action_size": 2,
        "physical_action_size": 8,
        "physical_action_schema": "d1-wheel-leg-extension-speed-v1",
        "policy_to_physical_indices": [0, 0, 0, 0, 1, 1, 1, 1],
    }
    assert experiment.declared_action_contract("wheel_leg") == (
        experiment.declared_action_contract("wheel_leg", "independent8")
    )
    force = experiment.declared_action_contract("lqr")
    assert force["action_mode"] == "legacy_force2"
    assert force["action_schema"] == "d1-v3-body-force-residual-v1"
    assert force["policy_action_size"] == force["physical_action_size"] == 2


@pytest.mark.parametrize("baseline", ("lqr", "mpc"))
@pytest.mark.parametrize("mode", ("shared2", "independent8"))
def test_explicit_nonwheel_mode_fails_before_creating_output(
    experiment, monkeypatch, tmp_path, baseline, mode
):
    output = tmp_path / "must_not_exist"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "train",
            "--output",
            str(output),
            "--baseline",
            baseline,
            "--action-mode",
            mode,
        ],
    )
    monkeypatch.setattr(experiment, "snapshot", lambda *_: pytest.fail("started work"))
    with pytest.raises(SystemExit) as stopped:
        experiment.main()
    assert stopped.value.code == 2
    assert not output.exists()


@pytest.fixture
def fake_env(experiment, monkeypatch):
    class FakeEnv(gym.Env):
        instances: ClassVar[list] = []
        observation_schema = "fake82"
        source_schema = "fake-source"

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            contract = experiment.declared_action_contract(
                kwargs["baseline"], kwargs.get("action_mode")
            )
            for name, value in contract.items():
                setattr(self, name, tuple(value) if name == "policy_to_physical_indices" else value)
            self.action_space = gym.spaces.Box(-1, 1, (self.policy_action_size,), np.float32)
            self.observation_space = gym.spaces.Box(-5, 5, (82,), np.float32)
            self.plant = SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(3)))
            self.closed = False
            self.steps = 0
            self.reset_seeds = []
            self.episode_metadata = contract
            self.instances.append(self)

        def reset(self, *, seed=None, options=None):
            self.reset_seeds.append(seed)
            self.steps = 0
            return np.zeros(82, np.float32), {"episode_metadata": self.episode_metadata.copy()}

        def step(self, action):
            self.steps += 1
            self.plant.data.qpos += 1
            # Deliberately not broadcast(policy): recorder must use actual receipt.
            applied = np.arange(self.physical_action_size, dtype=float) / 10
            info = {
                "applied_action": applied,
                "policy_action": np.asarray(action).copy(),
                "metrics": {
                    "time_s": self.steps * 0.01,
                    "velocity_error_mps": 0.0,
                    "yaw_rate_error_rps": 0.0,
                    "height_error_m": 0.0,
                    "roll_error_rad": 0.0,
                    "pitch_error_rad": 0.0,
                    "mechanical_power_w": 1.0,
                },
                "command": {"forward_velocity_mps": 0.0},
                "reward_terms": {"tracking": 1.0},
                "terrain_exposure": {"nonflat_now": False, "nonflat_fraction": 0.0},
                "terminal_reason": "time_limit" if self.steps == 2 else None,
            }
            return np.zeros(82, np.float32), 1.0, False, self.steps == 2, info

        def close(self):
            self.closed = True

    monkeypatch.setattr(experiment, "D1LocomotionEnv", FakeEnv)
    return FakeEnv


@pytest.mark.parametrize("mode", (None, "shared2", "independent8"))
def test_make_env_forwards_mode_and_actual_selected_road(experiment, fake_env, mode):
    env = experiment.make_env(
        "wheel_leg",
        5,
        0.02,
        "oracle",
        1,
        action_mode=mode,
        terrain_suite="action_compare_v1",
    )
    try:
        base = env.unwrapped
        assert base.kwargs["action_mode"] == mode
        assert base.kwargs["terrain"] == locomotion_terrain_configs("train", "action_compare_v1")[1]
    finally:
        env.close()
    assert fake_env.instances[-1].closed


@pytest.mark.parametrize("legacy_namespace", (False, True))
def test_worker_factories_forward_contract_and_remain_pickleable(
    experiment, monkeypatch, tmp_path, legacy_namespace
):
    args = _args(
        experiment, tmp_path, "--action-mode", "shared2", "--terrain-suite", "action_compare_v1"
    )
    if legacy_namespace:
        del args.action_mode
        del args.terrain_suite
    captured = {}

    def vector(factories, **kwargs):
        captured.update(factories=factories, kwargs=kwargs)
        return captured

    monkeypatch.setattr(experiment, "SubprocVecEnv", vector)
    experiment.make_vec(args, 4)
    assert captured["kwargs"] == {"start_method": "spawn"}
    for worker, factory in enumerate(captured["factories"]):
        assert factory.args[1] == worker
        assert factory.keywords["action_mode"] == (None if legacy_namespace else "shared2")
        assert factory.keywords["terrain_suite"] == (
            "v1" if legacy_namespace else "action_compare_v1"
        )
        restored = pickle.loads(pickle.dumps(factory))
        assert restored.func is experiment.make_env
        assert restored.args == factory.args and restored.keywords == factory.keywords


@pytest.mark.parametrize("suite", ("v1", "action_compare_v1"))
def test_protocol_records_selected_seeds_roads_and_policy_mapping(
    experiment, monkeypatch, tmp_path, suite
):
    output = tmp_path / "new"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "benchmark",
            "--output",
            str(output),
            "--action-mode",
            "shared2",
            "--terrain-suite",
            suite,
        ],
    )
    monkeypatch.setattr(experiment, "version", lambda *_: "fixture")
    monkeypatch.setattr(experiment, "snapshot", lambda *_: {})
    monkeypatch.setattr(experiment, "verify_source", lambda *_: None)
    monkeypatch.setattr(experiment, "benchmark", lambda *_: None)
    experiment.main()
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["terrain_suite"] == suite
    assert protocol["action_contract"] == experiment.declared_action_contract(
        "wheel_leg", "shared2"
    )
    expected_seeds = (
        {"development": [1017, 1029], "holdout": [1617, 1629]}
        if suite != "v1"
        else {"development": [17, 29], "holdout": [617, 629]}
    )
    assert protocol["evaluation_seeds"] == expected_seeds
    for split in OLD_ROWS:
        assert protocol["terrain_splits"][split] == [
            json.loads(c.to_json()) for c in locomotion_terrain_configs(split, suite)
        ]
    assert protocol["ppo_settings"] == json.loads(json.dumps(experiment.PPO_SETTINGS))
    assert protocol["quality_thresholds"] == experiment.QUALITY


def test_rollout_distinguishes_raw_and_clipped_policy_from_physical_receipt(experiment, tmp_path):
    callback = experiment.RolloutAudit(tmp_path)
    raw = np.array([[1.5, -0.25]])
    clipped = np.array([[1.0, -0.25]])
    physical = np.arange(8) / 10
    callback.locals = {
        "actions": raw,
        "clipped_actions": clipped,
        "rewards": [1.0],
        "infos": [
            {
                "applied_action": physical,
                "terrain_exposure": {"nonflat_now": False},
                "terminal_reason": None,
                "metrics": {},
                "reward_terms": {},
            }
        ],
    }
    assert callback._on_step()
    row = callback.rows[0]
    for axis in range(2):
        assert row[f"raw_action_{axis}"] == row[f"policy_raw_action_{axis}"] == raw[0, axis]
        assert (
            row[f"applied_action_{axis}"]
            == row[f"policy_clipped_action_{axis}"]
            == clipped[0, axis]
        )
    assert "raw_action_2" not in row and "applied_action_2" not in row
    for axis in range(8):
        assert row[f"physical_applied_action_{axis}"] == physical[axis]
    assert row["raw_action_clipped_fraction"] == 0.5
    for name, values in (
        ("policy_raw_action", raw),
        ("policy_clipped_action", clipped),
        ("physical_applied_action", physical),
    ):
        assert row[f"{name}_rms"] == pytest.approx(np.sqrt(np.mean(values**2)))


@pytest.mark.parametrize(
    "suite,split,seeds",
    (
        ("v1", "development", [17, 29]),
        ("v1", "holdout", [617, 629]),
        ("action_compare_v1", "development", [1017, 1029]),
        ("action_compare_v1", "holdout", [1617, 1629]),
        ("action_compare_v1", "flat", [1017, 1029]),
    ),
)
def test_evaluation_records_actual_physical_actions_and_new_seed_suite(
    experiment, fake_env, monkeypatch, tmp_path, suite, split, seeds
):
    from wheel_legged_control.d1 import locomotion_checkpoint

    action = np.array([0.25, -0.5], np.float32)
    monkeypatch.setattr(
        locomotion_checkpoint,
        "load_locomotion_policy",
        lambda *_: SimpleNamespace(predict=lambda *a, **kw: (action.copy(), None)),
    )
    args = _args(
        experiment, tmp_path, "--action-mode", "shared2", "--terrain-suite", suite, "--split", split
    )
    args.policy, args.metadata = Path("fake.zip"), Path("fake.json")
    result = experiment.evaluate(args, tmp_path)
    assert len(result) == (2 if split == "flat" else 4)
    assert [r["seed"] for r in result] == seeds * (1 if split == "flat" else 2)
    for r, env in zip(result, fake_env.instances, strict=True):
        assert env.kwargs["action_mode"] == "shared2" and env.closed
        assert env.reset_seeds == [r["seed"]]
        if split != "flat":
            index = int(r["case"].split("_")[0][4:])
            assert env.kwargs["terrain"] == locomotion_terrain_configs(split, suite)[index]
        with np.load(tmp_path / f"{r['case']}.npz", allow_pickle=False) as recording:
            assert recording["actions"].shape == recording["policy_actions"].shape == (2, 2)
            assert recording["physical_actions"].shape == (2, 8)
            assert recording["qpos"].shape == (3, 3)
            np.testing.assert_array_equal(recording["actions"], recording["policy_actions"])
            np.testing.assert_array_equal(recording["actions"], np.tile(action, (2, 1)))
            np.testing.assert_array_equal(
                recording["physical_actions"], np.tile(np.arange(8) / 10, (2, 1))
            )
            assert json.loads(recording["action_contract_json"].item())["policy_action_size"] == 2
        assert r["action_rms"] == pytest.approx(np.sqrt(np.mean((np.arange(8) / 10) ** 2)))
        metadata = json.loads((tmp_path / f"{r['case']}_episode.json").read_text())
        assert metadata["terrain_suite"] == suite
        assert metadata["policy_to_physical_indices"] == [0, 0, 0, 0, 1, 1, 1, 1]
        assert "action_contract" not in r  # Existing evaluation summary shape is preserved.


def test_flat_evaluation_rejects_unknown_suite_before_building_env(experiment, fake_env, tmp_path):
    args = _args(experiment, tmp_path, "--split", "flat")
    args.terrain_suite = "unknown"
    with pytest.raises(ValueError):
        experiment.evaluate(args, tmp_path)
    assert not fake_env.instances


def test_training_reference_matches_workers_and_counts_actual_network_parameters(
    experiment, fake_env, monkeypatch, tmp_path
):
    from wheel_legged_control.d1 import locomotion_checkpoint, ppo_update_audit

    args = _args(
        experiment, tmp_path, "--action-mode", "shared2", "--terrain-suite", "action_compare_v1"
    )
    args.mode, args.steps = "train", 128
    vector = SimpleNamespace(action_space=gym.spaces.Box(-1, 1, (2,), np.float32), closed=False)
    vector.close = lambda: setattr(vector, "closed", True)
    captured = {}
    monkeypatch.setattr(experiment, "make_vec", lambda *a: vector)

    class FakeLearner:
        def __init__(self, policy, env, **kwargs):
            self.action_space = env.action_space
            self.policy = torch.nn.Linear(3, 2)  # Exactly eight trainable parameters.
            self._n_updates = 1

        def learn(self, total_timesteps, callback):
            self.num_timesteps = total_timesteps

        def save(self, path):
            Path(path).write_bytes(b"fake checkpoint, not a trained policy")

    def metadata(path, env, *a, **kwargs):
        captured["reference"] = env.unwrapped

    monkeypatch.setattr(ppo_update_audit, "AuditedPPO", FakeLearner)
    monkeypatch.setattr(locomotion_checkpoint, "write_checkpoint_metadata", metadata)
    result = experiment.train(args, tmp_path)
    ref = captured["reference"]
    assert ref.kwargs["action_mode"] == "shared2"
    assert ref.kwargs["terrain"] == locomotion_terrain_configs("train", "action_compare_v1")[0]
    assert ref.closed and vector.closed
    assert result["policy_action_size"] == 2 and result["physical_action_size"] == 8
    assert result["policy_parameter_count"] == 8
    assert json.loads((tmp_path / "training.json").read_text()) == result


def test_real_shared_and_independent_envs_keep_zero_action_trajectory_and_actual_contract(
    experiment,
):
    envs = [
        experiment.make_env(
            "wheel_leg", 0, 0.02, "oracle", 1, action_mode=mode, terrain_suite="action_compare_v1"
        )
        for mode in ("shared2", "independent8")
    ]
    try:
        resets = [env.reset(seed=1017) for env in envs]
        np.testing.assert_array_equal(resets[0][0], resets[1][0])
        for _ in range(2):
            steps = [env.step(np.zeros(env.action_space.shape, np.float32)) for env in envs]
            np.testing.assert_array_equal(steps[0][0], steps[1][0])
            assert steps[0][1:4] == steps[1][1:4]
            np.testing.assert_array_equal(
                steps[0][4]["applied_action"], steps[1][4]["applied_action"]
            )
            np.testing.assert_array_equal(
                envs[0].unwrapped.plant.data.qpos, envs[1].unwrapped.plant.data.qpos
            )
        for env, expected in zip(envs, (2, 8), strict=True):
            actual = experiment.environment_action_contract(env)
            assert actual["policy_action_size"] == expected and actual["physical_action_size"] == 8
    finally:
        for env in envs:
            env.close()
