import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "continuous_training_under_test", ROOT / "scripts/train_d1_continuous_policy.py"
)
training = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = training
SPEC.loader.exec_module(training)


class TinyTask(gym.Env):
    observation_space = spaces.Box(-5, 5, (45,), dtype=np.float32)
    action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.used_seed = seed
        return np.zeros(45, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(45, dtype=np.float32), 2.0, True, False, {}


def test_protocol_freezes_disjoint_cases_and_sensor_task():
    protocol = training.build_protocol()
    assert protocol["training_seeds"] == [19000, 20000, 21000]
    assert protocol["checkpoint_budgets"] == [32768, 65536, 131072]
    assert protocol["n_envs"] == 4 and protocol["sensor_delay_steps"] == 0
    cases = protocol["evaluation_cases"]
    assert {row["seed"] for row in cases["development"]} == {17, 29, 43}
    assert {(row["seed"], row["duration_s"]) for row in cases["holdout"]} == {
        (seed, duration) for seed in (617, 629, 643) for duration in (45.0, 50.0)
    }
    assert not (
        {row["seed"] for row in cases["development"]} & {row["seed"] for row in cases["holdout"]}
    )
    assert protocol["training_config"]["ground_reference_mode"] == "estimated"
    assert protocol["training_config"]["observation_layout"] == "command45"
    assert protocol["sensor_noise"]["gyro_std_rad_s"] == 0.002
    assert protocol["raw_reward"].endswith("-5*terminated")
    assert protocol["policy_gating"] is False


def test_smoke_does_not_claim_full_budget_or_evaluate_holdout():
    protocol = training.build_protocol(smoke=True)
    assert protocol["smoke_only"] is True
    assert protocol["training_seeds"] == [19000]
    assert protocol["checkpoint_budgets"] == [1024]
    assert protocol["smoke_evaluation_steps"] == 16


def test_automatic_episode_seeds_are_reproducible_and_independent(tmp_path):
    a = training.EpisodeSeedStream(TinyTask(), 19000, 0, tmp_path / "a.jsonl")
    b = training.EpisodeSeedStream(TinyTask(), 19000, 0, tmp_path / "b.jsonl")
    c = training.EpisodeSeedStream(TinyTask(), 19000, 1, tmp_path / "c.jsonl")
    seeds = [[], [], []]
    for index, env in enumerate((a, b, c)):
        for _ in range(3):
            env.reset(seed=None)
            seeds[index].append(env.env.used_seed)
        records = [json.loads(line) for line in env.output.read_text().splitlines()]
        assert [record["sensor_seed"] for record in records] == seeds[index]
        assert [record["episode"] for record in records] == [0, 1, 2]
    assert seeds[0] == seeds[1] != seeds[2]
    assert len(set(seeds[0])) == 3
    assert min(seeds[0]) >= 1_000_000


def test_monitor_records_raw_reward_while_ppo_receives_scaled_reward():
    env = training.TrainingRewardScale(Monitor(TinyTask()))
    env.reset()
    _, reward, _, _, info = env.step(np.zeros(2))
    assert reward == pytest.approx(0.02)
    assert info["episode"]["r"] == 2
    assert info["raw_task_reward"] == 2


def test_actual_task_has_sensor_actor_and_measured_ground(monkeypatch):
    task = training.make_task()
    training.assert_sensor_task(task)
    assert task.source.delay_steps == 0
    assert task.source.measurements.noise == training.NOISE
    monkeypatch.setattr(
        task.plant,
        "training_ground_reference",
        lambda *_: pytest.fail("terrain oracle leaked to actor input"),
    )
    assert task.observation().shape == (45,)
    assert np.isfinite(task._command().pitch_rad)
    task.source = object()
    with pytest.raises(TypeError, match="state must"):
        training.assert_sensor_task(task)


def test_training_audit_records_raw_gaussian_and_actual_clipped_action(tmp_path):
    callback = training.TrainingAudit(tmp_path)
    callback.locals = {
        "actions": np.asarray([[2.0, -0.2], [-1.5, 0.4]]),
        "clipped_actions": np.asarray([[1.0, -0.2], [-1.0, 0.4]]),
        "rewards": np.array([0.01, 0.02]),
        "infos": [
            {"raw_task_reward": 1.0, "stage": "settle", "terrain_section": "flat"},
            {"raw_task_reward": 2.0, "stage": "cruise_bumps", "terrain_section": "bumps"},
        ],
    }
    assert callback._on_step()
    summary = callback.save()
    assert summary["clipped_fraction_per_action"] == [1.0, 0.0]
    assert summary["physical_samples"] == 2
    with np.load(tmp_path / "training_samples.npz") as sample:
        np.testing.assert_array_equal(
            sample["clipped_action"], np.clip(sample["raw_gaussian"], -1, 1)
        )
        np.testing.assert_allclose(sample["scaled_reward"], sample["raw_reward"] * 0.01)
    callback.locals["clipped_actions"] = np.zeros((2, 2))
    with pytest.raises(AssertionError):
        callback._on_step()


def test_frozen_source_guard_detects_edits(tmp_path, monkeypatch):
    path = tmp_path / "dependency.py"
    path.write_text("first")
    monkeypatch.setattr(training, "ROOT", tmp_path)
    snapshot = {path.name: training.ROLLOUT.sha256(path)}
    training.verify_source(snapshot)
    path.write_text("second")
    with pytest.raises(RuntimeError, match="frozen source changed"):
        training.verify_source(snapshot)


def test_oracle_and_delayed_tasks_are_not_accepted():
    with pytest.raises(TypeError):
        training.assert_sensor_task(SimpleNamespace(source=object()))
    task = training.make_task()
    task.source.delay_steps = 2
    with pytest.raises(ValueError, match="zero measurement delay"):
        training.assert_sensor_task(task)


def test_real_checkpoint_load_restores_all_training_rngs(tmp_path):
    torch.set_num_threads(1)
    model = PPO(
        "MlpPolicy",
        TinyTask(),
        seed=918,
        n_steps=8,
        batch_size=8,
        policy_kwargs={"net_arch": [8]},
        device="cpu",
    )
    path = tmp_path / "model.zip"
    model.save(path)
    random.seed(991)
    np.random.seed(992)
    torch.manual_seed(993)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    with training.preserve_training_rng():
        random.random()
        np.random.rand()
        torch.rand(3)
        PPO.load(path, device="cpu")
    assert random.getstate() == python_state
    assert np.random.get_state()[0] == numpy_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    with pytest.raises(RuntimeError, match="intentional"), training.preserve_training_rng():
        torch.rand(3)
        raise RuntimeError("intentional dev failure")
    assert torch.equal(torch.random.get_rng_state(), torch_state)
