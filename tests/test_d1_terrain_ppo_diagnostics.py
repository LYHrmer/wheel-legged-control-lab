"""Numerical checks for read-only, reproducible PPO diagnostic probes."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def diagnostic():
    spec = importlib.util.spec_from_file_location(
        "terrain_ppo_diagnostics", ROOT / "scripts" / "diagnose_d1_terrain_ppo.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_numpy_targets_bootstrap_and_episode_boundary(diagnostic):
    returns = diagnostic.lambda_returns(
        rewards=np.array([[1.0], [2.0], [3.0]]),
        values=np.array([[0.5], [0.6], [0.7]]),
        episode_starts=np.array([[1.0], [0.0], [1.0]]),
        last_values=np.array([0.8]),
        dones=np.array([False]),
        gamma=0.9,
        gae_lambda=0.8,
    )
    # Step 1 ends an episode. Step 2 cannot leak its advantage backwards.
    np.testing.assert_allclose(returns[:, 0], [2.548, 2.0, 3.72])


def test_explained_variance_is_not_a_measure_of_constant_value_bias(diagnostic):
    target = np.array([1.0, 2.0, 3.0])
    assert diagnostic.explained_variance(target - 100, target) == 1.0
    assert diagnostic.value_statistics(target - 100, target)["mse"] == 10000
    assert diagnostic.explained_variance(target, np.ones(3)) is None


@pytest.mark.parametrize("field", ["rewards", "episode_starts", "last_values", "dones"])
def test_numpy_targets_reject_shape_mismatch(diagnostic, field):
    arguments = {
        "rewards": np.ones((3, 2)),
        "values": np.zeros((3, 2)),
        "episode_starts": np.zeros((3, 2)),
        "last_values": np.zeros(2),
        "dones": np.zeros(2),
        "gamma": 0.99,
        "gae_lambda": 0.95,
    }
    arguments[field] = np.zeros(4)
    with pytest.raises(ValueError):
        diagnostic.lambda_returns(**arguments)


class ObservationFixture(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-5.0, 5.0, (44,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(44, np.float32), {}


@pytest.fixture(scope="module")
def model():
    torch = pytest.importorskip("torch")
    sb3 = pytest.importorskip("stable_baselines3")
    torch.set_num_threads(1)
    return sb3.PPO(
        "MlpPolicy",
        ObservationFixture(),
        n_steps=8,
        batch_size=4,
        seed=71,
        device="cpu",
        policy_kwargs={"net_arch": [64, 64]},
    )


@pytest.fixture()
def frozen_data(model):
    import torch

    observations = np.random.default_rng(81).normal(0, 0.3, (4, 2, 44)).astype(np.float32)
    with torch.no_grad():
        actions, values, log_probs = model.policy(torch.from_numpy(observations.reshape(8, 44)))
    values = values.numpy().reshape(4, 2)
    advantages = np.arange(8, dtype=np.float32).reshape(4, 2) + 20
    return {
        "observations": observations,
        "actions": actions.numpy().reshape(4, 2, 2),
        "values": values,
        "returns": values + advantages,
        "advantages": advantages,
        "log_probs": log_probs.numpy().reshape(4, 2),
        "probe_indices": np.array([5, 1, 7, 0]),
    }


def test_scaled_value_target_does_not_change_normalized_actor_gradient(
    diagnostic, model, frozen_data
):
    before = {name: tensor.clone() for name, tensor in model.policy.state_dict().items()}
    raw = diagnostic.gradient_probe(model, frozen_data, 1.0)
    scaled = diagnostic.gradient_probe(model, frozen_data, 0.01)
    assert scaled["actor_gradient_norm"] == pytest.approx(raw["actor_gradient_norm"], rel=2e-6)
    assert scaled["weighted_critic_gradient_norm"] < raw["weighted_critic_gradient_norm"]
    assert raw["shared_parameter_tensors"] == 0
    assert 0 < raw["global_clip_factor"] <= 1
    for name, tensor in model.policy.state_dict().items():
        assert tensor.equal(before[name])


@pytest.mark.parametrize("mode", ["global", "separate", "none"])
def test_optimizer_probe_updates_copy_only(diagnostic, model, frozen_data, mode):
    before = {name: tensor.clone() for name, tensor in model.policy.state_dict().items()}
    result = diagnostic.optimizer_probe(model, frozen_data, mode)
    assert result["actor_parameter_step_norm"] > 0
    assert result["mean_absolute_action_mean_change"] > 0
    for name, tensor in model.policy.state_dict().items():
        assert tensor.equal(before[name])


def test_fixed_batch_critic_fit_is_deterministic_and_does_not_update_original(
    diagnostic, model, frozen_data
):
    before = {name: tensor.clone() for name, tensor in model.policy.state_dict().items()}
    first = diagnostic.fit_critic(model, frozen_data, scale=0.01, fresh=True, steps=3, seed=17)
    second = diagnostic.fit_critic(model, frozen_data, scale=0.01, fresh=True, steps=3, seed=17)
    assert first == second
    for name, tensor in model.policy.state_dict().items():
        assert tensor.equal(before[name])


def test_actual_d1_rollout_repeats_and_diagnostic_keeps_checkpoint(diagnostic, model, tmp_path):
    checkpoint = tmp_path / "model.zip"
    model.save(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "observation_schema": "d1-terrain-oracle-v1",
                "model_sha256": digest,
            }
        )
    )
    _, first = diagnostic.collect_rollout(checkpoint, seed=811, steps=8, n_envs=1)
    _, second = diagnostic.collect_rollout(checkpoint, seed=811, steps=8, n_envs=1)
    for field in first:
        np.testing.assert_array_equal(first[field], second[field])
    args = SimpleNamespace(
        checkpoint=checkpoint,
        output=tmp_path / "probe",
        seed=811,
        rollout_steps=8,
        envs=1,
        fit_steps=2,
        batch_size=4,
        require_ev=None,
    )
    summary = diagnostic.run(args)
    assert summary["checkpoint_unchanged"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    assert summary["gradient_optimizer_minibatch_size"] == 4
    assert summary["numpy_sb3_return_max_abs_difference"] < 1e-4
    with np.load(args.output / "rollout.npz") as data:
        assert data["observations"].shape == (8, 1, 44)
        assert data["probe_indices"].shape == (4,)
    manifest = json.loads((args.output / "manifest.json").read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((args.output / name).read_bytes()).hexdigest() == expected
    with pytest.raises(FileExistsError):
        diagnostic.run(args)
