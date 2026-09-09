"""Physical paired checks for training-only reward units; no holdout cases."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from wheel_legged_control.d1.terrain_tracking_env import D1TerrainTrackingEnv


@pytest.fixture(scope="module")
def runner():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_d1_terrain_curriculum.py"
    specification = importlib.util.spec_from_file_location("terrain_reward_units_cli", script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_training_scale_preserves_real_d1_physics_reward_terms_and_raw_monitor_return(runner):
    monitor_module = pytest.importorskip("stable_baselines3.common.monitor")
    raw = D1TerrainTrackingEnv(training_mode="flat", episode_seconds=4.0, randomize=False)
    monitor = monitor_module.Monitor(
        D1TerrainTrackingEnv(training_mode="flat", episode_seconds=4.0, randomize=False)
    )
    wrapped = runner.TrainingRewardScale(monitor, scale=0.01)
    try:
        options = {
            "terrain": {"kind": "ramp", "slope_deg": 4.0},
            "velocity_mps": 0.35,
            "height_m": 0.455,
        }
        raw_observation, _ = raw.reset(seed=31, options=options)
        scaled_observation, _ = wrapped.reset(seed=31, options=options)
        np.testing.assert_array_equal(raw_observation, scaled_observation)
        raw_rewards = []
        scaled_rewards = []
        for action in np.random.default_rng(913).uniform(-0.15, 0.15, (400, 2)):
            raw_transition = raw.step(action)
            scaled_transition = wrapped.step(action)
            np.testing.assert_array_equal(raw_transition[0], scaled_transition[0])
            assert raw_transition[2:4] == scaled_transition[2:4]
            assert not raw_transition[2]
            np.testing.assert_array_equal(raw.plant.data.qpos, wrapped.unwrapped.plant.data.qpos)
            np.testing.assert_array_equal(raw.plant.data.qvel, wrapped.unwrapped.plant.data.qvel)
            raw_reward, scaled_reward = raw_transition[1], scaled_transition[1]
            assert scaled_reward == 0.01 * raw_reward
            raw_terms = raw_transition[4]["reward_terms"]
            scaled_terms = scaled_transition[4]["reward_terms"]
            assert raw_terms == scaled_terms
            assert scaled_terms["total"] == raw_reward
            # "total" is already the sum, not another reward component.
            assert (
                sum(value for name, value in scaled_terms.items() if name != "total") == raw_reward
            )
            raw_rewards.append(raw_reward)
            scaled_rewards.append(scaled_reward)

        assert raw_transition[3]
        assert scaled_transition[4]["episode"]["l"] == 400
        assert monitor.get_episode_rewards() == [sum(raw_rewards)]
        # Monitor writes episode.r rounded to six decimals; it is not a scaled sum.
        assert scaled_transition[4]["episode"]["r"] == pytest.approx(
            sum(raw_rewards), rel=0.0, abs=5.1e-7
        )
        assert sum(scaled_rewards) == pytest.approx(0.01 * sum(raw_rewards), rel=1e-14)
    finally:
        raw.close()
        wrapped.close()


def test_evaluate_case_reports_unscaled_real_d1_transition_rewards(runner):
    # These are existing training parameters, not fresh holdout geometry.
    case = {
        "case_id": "reward_units_development_bumps",
        "environment_seed": 31,
        "terrain": {"kind": "bumps", "amplitude_m": 0.005, "wavelength_m": 0.8, "phase_rad": 0.0},
        "velocity_mps": 0.35,
        "height_m": 0.455,
    }
    seconds = 1.0
    rows, metrics = runner.evaluate_case(D1TerrainTrackingEnv, case, seconds)
    raw = D1TerrainTrackingEnv(training_mode="flat", episode_seconds=seconds, randomize=False)
    try:
        raw.reset(
            seed=case["environment_seed"],
            options={name: case[name] for name in ("terrain", "velocity_mps", "height_m")},
        )
        raw_rewards = []
        for row in rows:
            _, reward, terminated, truncated, info = raw.step(np.zeros(2, dtype=np.float32))
            assert row["reward"] == reward == info["reward_terms"]["total"]
            assert row["position_x_m"] == info["position_x_m"]
            assert row["terminated"] == int(terminated)
            assert row["truncated"] == int(truncated)
            raw_rewards.append(reward)
        assert len(rows) == 100
        assert metrics["episode_return"] == sum(raw_rewards)
        assert metrics["episode_return"] > 100.0
    finally:
        raw.close()


def test_true_termination_does_not_add_value_bootstrap_to_scaled_reward(runner):
    torch = pytest.importorskip("torch")
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    class TerminalFixture(gym.Env):
        """One genuine terminal transition, not a simulated D1 success case."""

        def __init__(self):
            self.observation_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return np.ones(1, dtype=np.float32), 7.5, True, False, {"raw_reward": 7.5}

    class CaptureTerminations(BaseCallback):
        def __init__(self):
            super().__init__()
            self.terminal_infos = []

        def _on_step(self):
            self.terminal_infos.append(self.locals["infos"][0])
            return True

    vector_env = DummyVecEnv(
        [lambda: runner.TrainingRewardScale(Monitor(TerminalFixture()), scale=0.01)]
    )
    try:
        model = sb3.PPO(
            "MlpPolicy", vector_env, n_steps=2, batch_size=2, gamma=0.9, seed=31, device="cpu"
        )
        with torch.no_grad():
            model.policy.value_net.weight.zero_()
            model.policy.value_net.bias.fill_(0.4)
        capture = CaptureTerminations()
        _, callback = model._setup_learn(2, callback=capture)
        assert model.collect_rollouts(vector_env, callback, model.rollout_buffer, n_rollout_steps=2)
        assert len(capture.terminal_infos) == 2
        for info in capture.terminal_infos:
            assert info["TimeLimit.truncated"] is False
            np.testing.assert_array_equal(info["terminal_observation"], np.ones(1, np.float32))
            assert info["episode"]["r"] == info["raw_reward"] == 7.5
        np.testing.assert_allclose(model.rollout_buffer.values, 0.4, rtol=0.0, atol=1e-7)
        # Unlike timeout transitions, genuine terminal transitions omit gamma*V.
        np.testing.assert_allclose(model.rollout_buffer.rewards, 0.075, rtol=0.0, atol=1e-8)
        # GAE forms (r - V) + V in float32, so equality is within one float32 epsilon.
        np.testing.assert_allclose(
            model.rollout_buffer.returns,
            model.rollout_buffer.rewards,
            rtol=0.0,
            atol=np.finfo(np.float32).eps,
        )
    finally:
        vector_env.close()
