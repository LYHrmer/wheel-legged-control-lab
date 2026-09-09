"""Likelihood-preserving mean bounds and the physical one-channel residual seam."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")
sb3 = pytest.importorskip("stable_baselines3")
from stable_baselines3.common.policies import ActorCriticPolicy

from wheel_legged_control.d1.ppo_action_policies import (
    BoundedMeanActorCriticPolicy,
)
from wheel_legged_control.d1.residual_action_env import (
    D1LongitudinalResidualEnv,
)
from wheel_legged_control.d1.terrain_tracking_env import D1TerrainTrackingEnv


@pytest.fixture(autouse=True)
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_policy(policy_class=BoundedMeanActorCriticPolicy, *, seed=17, **kwargs):
    torch.manual_seed(seed)
    return policy_class(
        gym.spaces.Box(-5, 5, (4,), np.float32),
        kwargs.pop("action_space", gym.spaces.Box(-1, 1, (2,), np.float32)),
        lambda _: 3e-4,
        net_arch={"pi": [8], "vf": [8]},
        **kwargs,
    )


def set_constant_head(policy, bias=(0.7, -1.2), sigma=(0.8, 0.6)):
    with torch.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.bias.copy_(torch.tensor(bias))
        policy.log_std.copy_(torch.log(torch.tensor(sigma)))


def test_bounded_mean_has_exact_normal_likelihood_without_a_jacobian():
    policy = make_policy()
    set_constant_head(policy)
    observations = torch.zeros((2, 4))
    # Out-of-box raw actions are intentional: the PPO buffer retains them.
    actions = torch.tensor([[1.7, -1.5], [-2.0, 0.3]])
    distribution = policy.get_distribution(observations)
    expected_mu = np.tanh([0.7, -1.2])
    expected_log_prob = [
        sum(
            -0.5 * ((float(a) - mu) / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))
            for a, mu, sigma in zip(row, expected_mu, (0.8, 0.6), strict=True)
        )
        for row in actions
    ]
    torch.testing.assert_close(
        distribution.log_prob(actions),
        torch.tensor(expected_log_prob),
        rtol=1e-6,
        atol=1e-6,
        check_dtype=False,
    )
    _, log_prob, entropy = policy.evaluate_actions(observations, actions)
    torch.testing.assert_close(log_prob, distribution.log_prob(actions))
    expected_entropy = sum(
        math.log(sigma * math.sqrt(2 * math.pi * math.e)) for sigma in (0.8, 0.6)
    )
    torch.testing.assert_close(entropy, torch.full((2,), expected_entropy), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        distribution.distribution.mean.detach().numpy(),
        np.tile(expected_mu, (2, 1)),
        rtol=0,
        atol=1e-7,
    )
    # Negative control: the original Gaussian must fail the bounded-mean contract.
    baseline = make_policy(ActorCriticPolicy)
    set_constant_head(baseline)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            baseline.get_distribution(observations).distribution.mean.detach().numpy(),
            np.tile(expected_mu, (2, 1)),
            rtol=0,
            atol=1e-7,
        )


def test_mean_parameter_gradient_includes_tanh_derivative():
    policy = make_policy()
    bias, sigma = (0.7, -1.2), (0.8, 0.6)
    set_constant_head(policy, bias, sigma)
    observation, action = torch.zeros((1, 4)), torch.tensor([[1.7, -1.5]])
    policy.zero_grad(set_to_none=True)
    policy.evaluate_actions(observation, action)[1].sum().backward()
    gradient = policy.action_net.bias.grad.detach().numpy().copy()
    analytic = np.array(
        [
            (a - math.tanh(b)) / s**2 * (1 - math.tanh(b) ** 2)
            for a, b, s in zip((1.7, -1.5), bias, sigma, strict=True)
        ]
    )
    np.testing.assert_allclose(gradient, analytic, rtol=2e-6, atol=2e-6)
    finite_difference = []
    epsilon = 1e-3
    for index, value in enumerate(bias):
        probes = []
        for direction in (1, -1):
            with torch.no_grad():
                policy.action_net.bias[index] = value + direction * epsilon
                probes.append(float(policy.evaluate_actions(observation, action)[1].sum()))
        with torch.no_grad():
            policy.action_net.bias[index] = value
        finite_difference.append((probes[0] - probes[1]) / (2 * epsilon))
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize("policy_class", [ActorCriticPolicy, BoundedMeanActorCriticPolicy])
def test_forward_and_evaluate_actions_preserve_raw_action_ratio(policy_class):
    policy = make_policy(policy_class)
    set_constant_head(policy, (2.0, -2.0), (0.9, 0.9))
    observations = torch.zeros((128, 4))
    torch.manual_seed(123)
    with torch.no_grad():
        raw_actions, _, old_log_prob = policy(observations, deterministic=False)
        _, new_log_prob, _ = policy.evaluate_actions(observations, raw_actions)
        _, clipped_log_prob, _ = policy.evaluate_actions(observations, raw_actions.clamp(-1, 1))
    torch.testing.assert_close(
        (new_log_prob - old_log_prob).exp(), torch.ones(128), rtol=0, atol=1e-7
    )
    outside = torch.any(raw_actions.abs() > 1, dim=1)
    assert outside.any()  # Bounding only the mean must not squash Gaussian samples.
    assert torch.max(torch.abs(clipped_log_prob[outside] - old_log_prob[outside])) > 0.01
    with torch.no_grad():
        deterministic, _, _ = policy(observations, deterministic=True)
        mean = policy.get_distribution(observations).distribution.mean
    torch.testing.assert_close(deterministic, mean)


def test_bounded_mean_preserves_the_complete_initial_state_dict():
    baseline = make_policy(ActorCriticPolicy, seed=204)
    bounded = make_policy(BoundedMeanActorCriticPolicy, seed=204)
    left, right = baseline.state_dict(), bounded.state_dict()
    assert left.keys() == right.keys()
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
    assert baseline.optimizer.state_dict() == bounded.optimizer.state_dict()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_sde": True},
        {"action_space": gym.spaces.Box(-2, 2, (2,), np.float32)},
        {
            "action_space": gym.spaces.Box(
                np.array([-1.0, -0.5], dtype=np.float32),
                np.ones(2, dtype=np.float32),
                dtype=np.float32,
            )
        },
        {"action_space": gym.spaces.Discrete(2)},
    ],
)
def test_bounded_mean_rejects_other_distribution_or_action_contracts(kwargs):
    with pytest.raises(ValueError):
        make_policy(**kwargs)


@pytest.mark.parametrize(
    "policy_class,environment",
    [
        (ActorCriticPolicy, D1TerrainTrackingEnv),
        (BoundedMeanActorCriticPolicy, D1TerrainTrackingEnv),
        (ActorCriticPolicy, D1LongitudinalResidualEnv),
    ],
)
def test_real_ppo_updates_keep_raw_buffer_and_survive_save_load(
    policy_class, environment, tmp_path
):
    class RecordExecutedAction(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.executed = []

        def step(self, action):
            self.executed.append(np.asarray(action).copy())
            return self.env.step(action)

    env = RecordExecutedAction(
        environment(episode_seconds=0.16, training_mode="flat", randomize=False)
    )
    try:
        model = sb3.PPO(
            policy_class,
            env,
            n_steps=16,
            batch_size=8,
            n_epochs=2,
            policy_kwargs={"net_arch": [8, 8]},
            seed=91,
            device="cpu",
        )
        with torch.no_grad():
            model.policy.action_net.bias.fill_(1.4)
            model.policy.log_std.fill_(math.log(2.0))
        before = model.policy.action_net.bias.detach().clone()
        model.learn(total_timesteps=32)
        assert model.num_timesteps == 32 and model._n_updates == 4
        assert not torch.equal(before, model.policy.action_net.bias)
        raw = model.rollout_buffer.actions.reshape(-1, env.action_space.shape[0])
        assert len(raw) == 16 and np.any(np.abs(raw) > 1)
        np.testing.assert_array_equal(np.clip(raw, -1, 1), np.asarray(env.executed[-16:]))
        assert np.isfinite(model.rollout_buffer.log_probs).all()
        assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
        observation, _ = env.reset(seed=300)
        expected_action = model.predict(observation, deterministic=True)[0]
        path = tmp_path / "policy.zip"
        model.save(path)
        loaded = sb3.PPO.load(path, device="cpu")
        assert isinstance(loaded.policy, policy_class)
        assert loaded.num_timesteps == 32 and loaded._n_updates == 4
        np.testing.assert_array_equal(
            loaded.predict(observation, deterministic=True)[0], expected_action
        )
        for name, value in model.policy.state_dict().items():
            torch.testing.assert_close(loaded.policy.state_dict()[name], value, rtol=0, atol=0)
        original_optimizer, loaded_optimizer = (
            model.policy.optimizer.state_dict(),
            loaded.policy.optimizer.state_dict(),
        )
        assert loaded_optimizer["param_groups"] == original_optimizer["param_groups"]
        assert loaded_optimizer["state"].keys() == original_optimizer["state"].keys()
        for parameter, values in original_optimizer["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(
                    loaded_optimizer["state"][parameter][name], value, rtol=0, atol=0
                )
    finally:
        env.close()


@pytest.mark.parametrize(
    "terrain",
    [
        {"kind": "flat"},
        {"kind": "bumps", "amplitude_m": 0.01, "wavelength_m": 0.8, "phase_rad": 0.3},
        {"kind": "ramp", "slope_deg": 4.0},
    ],
)
def test_longitudinal_environment_matches_full_physics_rewards_and_observations(terrain):
    one = D1LongitudinalResidualEnv(episode_seconds=0.2, training_mode="flat", randomize=False)
    two = D1TerrainTrackingEnv(episode_seconds=0.2, training_mode="flat", randomize=False)
    try:
        options = {"terrain": terrain, "velocity_mps": 0.35, "height_m": 0.455}
        first, first_info = one.reset(seed=12, options=options)
        second, second_info = two.reset(seed=12, options=options)
        np.testing.assert_array_equal(first, second)
        assert one.action_space.shape == (1,) and two.action_space.shape == (2,)
        assert one.observation_space.shape == two.observation_space.shape == (44,)
        assert one.observation_schema == two.observation_schema
        assert one.reward_schema == two.reward_schema and one.control_schema == two.control_schema
        assert one.action_schema == "d1-terrain-residual-longitudinal-only-v1"
        assert one.action_schema != two.action_schema
        assert first_info["initial_height_lift_m"] == second_info["initial_height_lift_m"]
        for step in range(20):
            action = 2.0 if step % 7 == 0 else 0.6 * math.sin(0.2 * step)
            a = one.step(np.array([action]))
            b = two.step(np.array([action, 0.0]))
            np.testing.assert_array_equal(a[0], b[0])
            assert a[1:4] == b[1:4]
            assert a[4]["reward_terms"] == b[4]["reward_terms"]
            np.testing.assert_array_equal(a[4]["residual_action"], [np.clip(action, -1, 1), 0])
            np.testing.assert_array_equal(a[4]["torque_nm"], b[4]["torque_nm"])
            for left, right in zip(
                one.plant.simulation_state(), two.plant.simulation_state(), strict=True
            ):
                np.testing.assert_array_equal(left, right)
            assert a[0][41] == 0.0  # Previous vertical action in the unchanged 44-value layout.
        assert a[3] and not a[2]
    finally:
        one.close()
        two.close()


@pytest.mark.parametrize("action", [0.0, [], [0.0, 0.0], [np.nan], [np.inf]])
def test_longitudinal_environment_rejects_wrong_action_contract(action):
    env = D1LongitudinalResidualEnv(episode_seconds=0.01)
    try:
        env.reset(seed=0)
        with pytest.raises(ValueError, match="one finite"):
            env.step(np.asarray(action))
    finally:
        env.close()
