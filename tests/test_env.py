import numpy as np
from gymnasium.utils.env_checker import check_env

from wheel_legged_control.env import OBSERVATION_SIZE, WheelLeggedResidualEnv


def test_environment_follows_gymnasium_contract() -> None:
    env = WheelLeggedResidualEnv(randomize=False, episode_seconds=0.2)
    check_env(env, skip_render_check=True)
    env.close()


def test_seeded_reset_is_reproducible() -> None:
    first = WheelLeggedResidualEnv(randomize=True, episode_seconds=0.2)
    second = WheelLeggedResidualEnv(randomize=True, episode_seconds=0.2)
    observation_a, _ = first.reset(seed=12)
    observation_b, _ = second.reset(seed=12)
    np.testing.assert_allclose(observation_a, observation_b)
    first.close()
    second.close()


def test_zero_residual_rollout_is_finite() -> None:
    env = WheelLeggedResidualEnv(randomize=False, episode_seconds=1.0)
    observation, _ = env.reset(seed=3, options={"scenario": "nominal"})
    assert observation.shape == (OBSERVATION_SIZE,)
    for _ in range(50):
        observation, reward, terminated, truncated, info = env.step(np.zeros(2))
        assert np.all(np.isfinite(observation))
        assert np.isfinite(reward)
        assert reward == info["reward_terms"]["total"]
        assert not terminated
    assert truncated
    env.close()
