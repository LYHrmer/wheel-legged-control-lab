"""A recording PPO must train identically to real SB3 PPO, including RNG state."""

import json
import random

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")
PPO = pytest.importorskip("stable_baselines3").PPO

from wheel_legged_control.d1.ppo_update_audit import AuditedPPO


class DeterministicAuditEnv(gym.Env):
    """Narrow action bounds make raw Gaussian samples visibly differ from commands."""

    def __init__(self, *, offset=0.0, horizon=5, truncate=False):
        self.observation_space = gym.spaces.Box(-100, 100, (3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-0.3, 0.3, (2,), dtype=np.float32)
        self.commands = []
        self.rewards = []
        self.offset, self.horizon, self.truncate = offset, horizon, truncate

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_index = 0
        return np.asarray((self.offset, 0, 0), np.float32), {}

    def step(self, action):
        self.step_index += 1
        self.commands.append(np.asarray(action).copy())
        observation = np.asarray(
            (self.offset + self.step_index / self.horizon, *action), np.float32
        )
        reward = float(1 - np.square(action).sum() - 0.01 * self.step_index)
        self.rewards.append(reward)
        done = self.step_index == self.horizon
        return observation, reward, done and not self.truncate, done and self.truncate, {}


@pytest.fixture(autouse=True)
def one_torch_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def make_model(cls=PPO, **kwargs):
    env = DeterministicAuditEnv()
    model = cls(
        "MlpPolicy",
        env,
        n_steps=8,
        batch_size=4,
        n_epochs=2,
        gamma=0.91,
        gae_lambda=0.83,
        learning_rate=5e-4,
        policy_kwargs={"net_arch": [8, 8], "log_std_init": 1.0},
        seed=19,
        device="cpu",
        **kwargs,
    )
    return model, env


def weights(model):
    return {key: value.detach().cpu().clone() for key, value in model.policy.state_dict().items()}


def rng_state():
    state = np.random.get_state()
    return (
        torch.random.get_rng_state().clone(),
        (state[0], state[1].copy(), *state[2:]),
        random.getstate(),
    )


def assert_same_rng(left, right):
    assert torch.equal(left[0], right[0])
    assert left[1][0] == right[1][0] and left[1][2:] == right[1][2:]
    np.testing.assert_array_equal(left[1][1], right[1][1])
    assert left[2] == right[2]


@pytest.mark.parametrize("enabled", (False, True))
def test_recording_does_not_change_real_ppo_weights_commands_or_rng_across_two_learn_calls(
    tmp_path, enabled
):
    reference, reference_env = make_model()
    before = weights(reference)
    reference.learn(32)
    first_weights, first_rng = weights(reference), rng_state()
    reference.learn(32, reset_num_timesteps=False)
    second_weights, second_rng = weights(reference), rng_state()
    expected_commands = np.asarray(reference_env.commands)
    reference_env.close()

    directory = tmp_path / "audit" if enabled else None
    observed, observed_env = make_model(AuditedPPO, audit_directory=directory)
    try:
        observed.learn(32)
        for name, value in weights(observed).items():
            assert torch.equal(value, first_weights[name]), name
        assert_same_rng(rng_state(), first_rng)
        observed.learn(32, reset_num_timesteps=False)
        for name, value in weights(observed).items():
            assert torch.equal(value, second_weights[name]), name
        assert_same_rng(rng_state(), second_rng)
        assert any(not torch.equal(before[name], second_weights[name]) for name in before)
        np.testing.assert_array_equal(observed_env.commands, expected_commands)
        assert observed.num_timesteps == 64 and observed._n_updates == 16
        if enabled:
            rows = [
                json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()
            ]
            assert [row["audit_index"] for row in rows] == list(range(8))
            assert [row["num_timesteps"] for row in rows] == list(range(8, 65, 8))
            assert [row["n_updates"] for row in rows] == list(range(2, 17, 2))
            assert {path.name for path in directory.iterdir()} == {
                "updates.jsonl",
                *(f"sample_{index:06d}.npz" for index in range(8)),
            }
        else:
            assert list(tmp_path.iterdir()) == []
    finally:
        observed_env.close()


def test_saved_then_loaded_policy_does_not_reopen_the_old_audit_directory(tmp_path):
    directory = tmp_path / "audit"
    model, env = make_model(AuditedPPO, audit_directory=directory)
    try:
        model.learn(32)
        prediction = model.predict(np.asarray((0.5, 0.1, -0.1), np.float32), deterministic=True)[0]
        model.save(tmp_path / "model.zip")
        before = {path.name: path.read_bytes() for path in directory.iterdir()}
    finally:
        env.close()
    new_env = DeterministicAuditEnv()
    loaded = AuditedPPO.load(tmp_path / "model.zip", env=new_env, device="cpu")
    try:
        np.testing.assert_array_equal(
            loaded.predict(np.asarray((0.5, 0.1, -0.1), np.float32), deterministic=True)[0],
            prediction,
        )
        loaded.learn(16, reset_num_timesteps=False)
        assert loaded.num_timesteps == 48
        assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    finally:
        new_env.close()


def expanded_gae(rewards, values, starts, last_values, last_dones, gamma, lam):
    """Independent expanded discounted TD-error sums, not SB3's backward recursion."""
    count, n_envs = rewards.shape
    result = np.zeros((count, n_envs))
    for env_index in range(n_envs):
        for first in range(count):
            weight = 1.0
            for step in range(first, count):
                done = last_dones[env_index] if step == count - 1 else starts[step + 1, env_index]
                successor = (
                    last_values[env_index] if step == count - 1 else values[step + 1, env_index]
                )
                delta = rewards[step, env_index] - values[step, env_index]
                if not done:
                    delta += gamma * successor
                result[first, env_index] += weight * delta
                if done:
                    break
                weight *= gamma * lam
    return result


def test_full_rollout_allows_independent_gae_across_episode_boundaries_and_the_128_sample_cut(
    tmp_path,
):
    from stable_baselines3.common.vec_env import DummyVecEnv

    # Env 0 has timeout boundaries; env 1 has no boundary in this rollout.
    # 160 collected samples exceed the 128-sample likelihood fragment.
    left = DeterministicAuditEnv(horizon=5, truncate=True)
    right = DeterministicAuditEnv(offset=2, horizon=101)
    env = DummyVecEnv([lambda: left, lambda: right])
    directory = tmp_path / "audit"
    model = AuditedPPO(
        "MlpPolicy",
        env,
        audit_directory=directory,
        n_steps=80,
        batch_size=32,
        n_epochs=1,
        gamma=0.91,
        gae_lambda=0.83,
        policy_kwargs={"net_arch": [8, 8], "log_std_init": 1},
        seed=19,
        device="cpu",
    )
    try:
        model.learn(160)
        with np.load(directory / "sample_000000.npz", allow_pickle=False) as archive:
            data = {name: archive[name].copy() for name in archive.files}
        row = json.loads((directory / "updates.jsonl").read_text())
        assert row["n_audit_samples"] == 128 and row["num_timesteps"] == 160
        assert data["observations"].shape == (128, 3)
        assert data["actions_raw"].shape == (128, 2)
        assert np.all(data["observations"][:80, 0] < 1)
        assert np.all(data["observations"][80:, 0] >= 2)
        assert data["gae_rewards"].shape == data["gae_values"].shape == (80, 2)
        np.testing.assert_array_equal(data["gae_last_dones"], (True, False))
        assert data["gae_gamma"] == 0.91 and data["gae_lambda"] == 0.83
        assert data["gae_episode_starts"][5, 0] == 1
        assert np.count_nonzero(data["gae_episode_starts"][:, 1]) == 1
        # The SB3 timeout bootstrap is recorded, not confused with raw env reward.
        np.testing.assert_allclose(data["gae_rewards"][:, 1], right.rewards, atol=1e-7)
        assert abs(data["gae_rewards"][4, 0] - left.rewards[4]) > 1e-5
        calculated = expanded_gae(
            data["gae_rewards"].astype(float),
            data["gae_values"].astype(float),
            data["gae_episode_starts"],
            data["gae_last_values"],
            data["gae_last_dones"],
            float(data["gae_gamma"]),
            float(data["gae_lambda"]),
        )
        expected_advantages = np.concatenate((calculated[:, 0], calculated[:, 1]))[:128]
        expected_values = np.concatenate((data["gae_values"][:, 0], data["gae_values"][:, 1]))[:128]
        np.testing.assert_allclose(data["advantages"], expected_advantages, atol=3e-6, rtol=1e-6)
        np.testing.assert_array_equal(data["old_values"], expected_values)
        np.testing.assert_allclose(
            data["returns"], expected_advantages + expected_values, atol=3e-6, rtol=1e-6
        )
        # Truncating the full rollout to the saved env-1 fragment changes its GAE.
        shortened = expanded_gae(
            data["gae_rewards"][:48, 1:2].astype(float),
            data["gae_values"][:48, 1:2].astype(float),
            data["gae_episode_starts"][:48, 1:2],
            data["gae_last_values"][1:],
            data["gae_last_dones"][1:],
            0.91,
            0.83,
        )
        assert abs(calculated[47, 1] - shortened[-1, 0]) > 0.01

        actual_commands = np.concatenate((left.commands, right.commands))[:128]
        np.testing.assert_array_equal(np.clip(data["actions_raw"], -0.3, 0.3), actual_commands)
        assert np.any(np.abs(data["actions_raw"]) > 0.3)
        np.testing.assert_allclose(data["ratio_before"], 1, atol=2e-6, rtol=0)
        for label in ("before", "after"):
            mean = data[f"action_mean_{label}"].astype(float)
            std = data[f"action_std_{label}"].astype(float)
            raw = data["actions_raw"].astype(float)
            log_prob = np.sum(
                -0.5 * ((raw - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi), axis=1
            )
            np.testing.assert_allclose(data[f"log_prob_{label}"], log_prob, atol=2e-6, rtol=1e-6)
            logratio = data[f"log_prob_{label}"] - data["old_log_prob"]
            np.testing.assert_allclose(
                data[f"ratio_{label}"], np.exp(logratio), atol=1e-12, rtol=1e-12
            )
            assert row[f"approx_reverse_kl_{label}"] == pytest.approx(
                float(np.mean(np.exp(logratio) - 1 - logratio)), abs=1e-12
            )
            assert row[f"ratio_clip_fraction_{label}"] == pytest.approx(
                float(np.mean(np.abs(data[f"ratio_{label}"] - 1) > row["clip_range"]))
            )
        assert row["param_sha256_before"] != row["param_sha256_after"]
        assert row["parameters_changed"] is True
    finally:
        env.close()
