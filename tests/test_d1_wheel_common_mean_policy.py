"""Distribution, gradient and serialization checks for the wheel mean prior."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from scripts.d1_wheel_common_mean_policy import (
    D1WheelCommonMeanPolicy,
    bounded_wheel_mean,
)


def spaces():
    return gym.spaces.Box(-5, 5, (82,), np.float32), gym.spaces.Box(-1, 1, (8,), np.float32)


def policy(limit=0.05, **kwargs):
    return D1WheelCommonMeanPolicy(
        *spaces(), lambda _: 3e-4, net_arch=[16, 16],
        wheel_common_mean_limit=limit, **kwargs,
    )


def test_transform_preserves_legs_and_all_pairwise_wheel_differences():
    raw = torch.randn(3, 7, 8, dtype=torch.float64)
    original = raw.clone()
    actual = bounded_wheel_mean(raw, 0.05)
    assert actual.shape == raw.shape and actual.dtype == raw.dtype
    assert actual.device == raw.device
    assert torch.equal(raw, original)
    assert torch.equal(actual[..., :4], raw[..., :4])
    assert torch.all(actual[..., 4:].mean(-1).abs() <= 0.05 + 1e-14)
    for a in range(4, 8):
        for b in range(4, 8):
            torch.testing.assert_close(actual[..., a] - actual[..., b], raw[..., a] - raw[..., b])


def test_zero_common_mean_unchanged_and_large_common_input_saturates():
    raw = torch.tensor([[1., 2., 3., 4., -0.3, 0.3, -0.2, 0.2]], dtype=torch.float64)
    torch.testing.assert_close(bounded_wheel_mean(raw, 0.05), raw)
    negative = torch.full((8,), -100., dtype=torch.float64)
    assert torch.equal(
        bounded_wheel_mean(negative, 0.05)[4:], torch.full((4,), -0.05, dtype=torch.float64)
    )


def test_none_is_exact_identity_and_no_new_weights_or_rng_draws():
    raw = torch.randn(2, 8)
    assert torch.equal(bounded_wheel_mean(raw, None), raw)
    torch.manual_seed(871)
    reference = ActorCriticPolicy(*spaces(), lambda _: 3e-4, net_arch=[16, 16])
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(871)
    candidate = policy(None)
    assert torch.equal(torch.get_rng_state(), reference_rng)
    assert candidate.state_dict().keys() == reference.state_dict().keys()
    for name, value in candidate.state_dict().items():
        assert torch.equal(value, reference.state_dict()[name])
    obs = torch.randn(5, 82)
    for deterministic in (False, True):
        torch.manual_seed(991)
        expected = reference(obs, deterministic=deterministic)
        torch.manual_seed(991)
        actual = candidate(obs, deterministic=deterministic)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_transform_has_correct_finite_gradients_in_common_and_differential_directions():
    raw = (torch.randn(2, 8, dtype=torch.float64) * 0.025).requires_grad_()
    assert torch.autograd.gradcheck(lambda x: bounded_wheel_mean(x, 0.05), (raw,))
    value = bounded_wheel_mean(raw, 0.05)
    common_grad = torch.autograd.grad(value[..., 4:].sum(), raw, retain_graph=True)[0]
    assert torch.all(common_grad[..., 4:] > 0)
    assert torch.equal(common_grad[..., :4], torch.zeros_like(common_grad[..., :4]))
    diff_grad = torch.autograd.grad((value[..., 4] - value[..., 5]).sum(), raw)[0]
    torch.testing.assert_close(diff_grad[..., 4], torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(diff_grad[..., 5], -torch.ones(2, dtype=torch.float64))


def test_bounded_and_unbounded_policies_start_with_identical_weights_and_rng():
    torch.manual_seed(48001)
    reference = policy(None)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(48001)
    candidate = policy(0.05)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value, reference.state_dict()[name])
               for name, value in candidate.state_dict().items())


def test_sampling_log_probability_and_training_use_the_same_normal_distribution():
    candidate = policy()
    with torch.no_grad():
        candidate.action_net.weight.zero_()
        candidate.action_net.bias.copy_(torch.tensor([0., .1, -.1, 0., -1., -1., -1., -1.]))
    obs = torch.zeros(1024, 82)
    actions, values, rollout_log_prob = candidate(obs)
    distribution = candidate.get_distribution(obs).distribution
    expected = torch.distributions.Normal(distribution.mean, candidate.log_std.exp())
    torch.testing.assert_close(rollout_log_prob, expected.log_prob(actions).sum(-1))
    _, update_log_prob, entropy = candidate.evaluate_actions(obs, actions.detach())
    torch.testing.assert_close(update_log_prob, rollout_log_prob)
    torch.testing.assert_close(entropy, expected.entropy().sum(-1))
    # The constraint applies to the mean. Exploration samples remain full-rank
    # independent normals; projecting samples would invalidate PPO's log_prob.
    assert torch.all(distribution.mean[..., 4:].mean(-1).abs() <= 0.050001)
    assert torch.any(actions[..., 4:].mean(-1).abs() > 0.1)
    assert torch.any(actions.abs() > 1)
    candidate.optimizer.zero_grad()
    loss = -update_log_prob.mean() + values.square().mean()
    loss.backward()
    assert candidate.action_net.bias.grad is not None
    assert torch.isfinite(candidate.action_net.bias.grad).all()


@pytest.mark.parametrize("limit", [None, 0.05, 0.125])
def test_policy_and_ppo_roundtrip_preserve_constraint_and_predictions(tmp_path, limit):
    candidate = policy(limit)
    obs = np.random.default_rng(888).normal(size=82).astype(np.float32)
    expected = candidate.predict(obs, deterministic=True)[0]
    path = tmp_path / "policy.pt"
    candidate.save(path)
    loaded = D1WheelCommonMeanPolicy.load(path)
    assert loaded.wheel_common_mean_limit == limit
    np.testing.assert_array_equal(loaded.predict(obs, deterministic=True)[0], expected)

    class SpaceOnlyEnv(gym.Env):
        observation_space, action_space = spaces()

    model = PPO(
        D1WheelCommonMeanPolicy, SpaceOnlyEnv(), n_steps=8, batch_size=8,
        policy_kwargs={"wheel_common_mean_limit": limit}, seed=88,
    )
    expected = model.predict(obs, deterministic=True)[0]
    model.save(tmp_path / "ppo.zip")
    restored = PPO.load(tmp_path / "ppo.zip", device="cpu")
    assert isinstance(restored.policy, D1WheelCommonMeanPolicy)
    assert restored.policy.wheel_common_mean_limit == limit
    np.testing.assert_array_equal(restored.predict(obs, deterministic=True)[0], expected)


@pytest.mark.parametrize("bad", [True, np.bool_(True), 0, -1, float("nan"), float("inf"), "0.05"])
def test_invalid_mean_limit_is_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        policy(bad)


def test_unsupported_distribution_and_task_spaces_are_rejected():
    with pytest.raises(ValueError):
        policy(use_sde=True)
    obs, act = spaces()
    for bad_obs, bad_act in (
        (obs, gym.spaces.Box(-1, 1, (4,), np.float32)),
        (obs, gym.spaces.Box(-2, 2, (8,), np.float32)),
        (gym.spaces.Box(-5, 5, (81,), np.float32), act),
        (obs, gym.spaces.Discrete(8)),
    ):
        with pytest.raises((ValueError, TypeError)):
            D1WheelCommonMeanPolicy(bad_obs, bad_act, lambda _: 3e-4)


def test_wrong_tensor_shape_is_not_silently_reshaped():
    with pytest.raises((ValueError, TypeError)):
        bounded_wheel_mean(torch.zeros(2, 4), 0.05)


def test_native_clipping_can_break_even_the_deterministic_common_mean_bound():
    raw = torch.tensor([0., 0., 0., 0., 3., -1., -1., -1.])
    mean = bounded_wheel_mean(raw, 0.05)
    assert mean[4:].mean().item() == 0
    assert torch.clamp(mean, -1, 1)[4:].mean().item() == -0.5


@pytest.mark.parametrize("limit", [1e40, 1e-50])
def test_limit_must_be_representable_in_the_actual_tensor_dtype(limit):
    with pytest.raises(ValueError, match="torch.float32"):
        bounded_wheel_mean(torch.full((8,), .1), limit)
    with pytest.raises(ValueError, match="torch.float32"):
        policy(limit)
    raw = torch.full((8,), .1, dtype=torch.float64, requires_grad=True)
    result = bounded_wheel_mean(raw, limit)
    result.sum().backward()
    assert torch.isfinite(result).all()
    assert torch.isfinite(raw.grad).all()
