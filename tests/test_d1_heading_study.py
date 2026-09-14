"""Check completed-update checkpoint timing with actual PPO on a small fixture."""

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from scripts.run_d1_heading_study import AfterUpdatePPO, main
from wheel_legged_control.d1.ppo_update_audit import parameter_sha256


class SmallHeadingFixture(gym.Env):
    observation_space = gym.spaces.Box(-5, 5, (85,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (8,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(85, np.float32), {}

    def step(self, action):
        self.steps += 1
        observation = np.full(85, self.steps / 10, np.float32)
        reward = float(1.0 - np.square(action).mean())
        return observation, reward, False, self.steps == 4, {}


class ObserveRolloutEnd(BaseCallback):
    def __init__(self):
        super().__init__()
        self.events = []

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        self.events.append((self.model.num_timesteps, self.model._n_updates,
                            parameter_sha256(self.model.policy)))


def test_saved_checkpoints_follow_full_updates_and_reload_as_stock_ppo(tmp_path):
    after_updates = []

    def hook(model):
        record = (model.num_timesteps, model._n_updates, parameter_sha256(model.policy))
        model.save(tmp_path / f"step{model.num_timesteps}.zip")
        after_updates.append(record)

    env = SmallHeadingFixture()
    model = AfterUpdatePPO("MlpPolicy", env, after_update=hook, n_steps=8,
                           batch_size=8, n_epochs=2, seed=3,
                           policy_kwargs={"net_arch": [8, 8]}, device="cpu")
    callback = ObserveRolloutEnd()
    initial = parameter_sha256(model.policy)
    try:
        model.learn(16, callback=callback)
        assert [(steps, epochs) for steps, epochs, _ in callback.events] == [(8, 0), (16, 2)]
        assert [(steps, epochs) for steps, epochs, _ in after_updates] == [(8, 2), (16, 4)]
        assert model.completed_train_calls == 2
        assert callback.events[0][2] == initial
        assert after_updates[0][2] != initial
        assert callback.events[1][2] == after_updates[0][2]
        for index, (steps, epochs, expected_hash) in enumerate(after_updates, start=1):
            loaded = PPO.load(tmp_path / f"step{steps}.zip", device="cpu")
            assert loaded.num_timesteps == steps and loaded._n_updates == epochs
            assert loaded.completed_train_calls == index
            assert parameter_sha256(loaded.policy) == expected_hash
            assert not hasattr(loaded, "_after_update")
    finally:
        env.close()


def test_failed_update_does_not_claim_or_save_checkpoint(monkeypatch):
    def fail(_):
        raise RuntimeError("injected update failure")

    events = []
    model = AfterUpdatePPO("MlpPolicy", SmallHeadingFixture(), after_update=events.append,
                           n_steps=8, batch_size=8, n_epochs=1, seed=4)
    monkeypatch.setattr(PPO, "train", fail)
    try:
        with pytest.raises(RuntimeError, match="injected update failure"):
            model.learn(8)
        assert events == [] and model.completed_train_calls == 0
    finally:
        model.get_env().close()


@pytest.mark.parametrize("args", [
    ["--mode", "train", "--steps", "16384"],
    ["--mode", "train", "--seeds", "48001"],
    ["--mode", "smoke", "--steps", "33"],
])
def test_invalid_budget_or_formal_seed_rejected_before_output_creation(tmp_path, args):
    output = tmp_path / "not_created"
    with pytest.raises(ValueError):
        main([*args, "--output", str(output)])
    assert not output.exists()
