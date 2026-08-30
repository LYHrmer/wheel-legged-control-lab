import numpy as np
from gymnasium.utils.env_checker import check_env

from wheel_legged_control.d1.env import D1_OBSERVATION_SIZE, D1ResidualEnv
from wheel_legged_control.d1.model import NOMINAL_JOINT_POSITION
from wheel_legged_control.d1.rewards import calculate_d1_reward


def test_d1_environment_follows_gymnasium_contract() -> None:
    env = D1ResidualEnv(randomize=False, episode_seconds=0.12)
    check_env(env, skip_render_check=True)
    env.close()


def test_d1_seeded_reset_is_reproducible() -> None:
    first = D1ResidualEnv(randomize=True, episode_seconds=0.12)
    second = D1ResidualEnv(randomize=True, episode_seconds=0.12)
    observation_a, info_a = first.reset(seed=19)
    observation_b, info_b = second.reset(seed=19)
    np.testing.assert_allclose(observation_a, observation_b)
    assert info_a["domain"] == info_b["domain"]
    first.close()
    second.close()


def test_d1_zero_residual_rollout_is_finite() -> None:
    env = D1ResidualEnv(randomize=False, episode_seconds=0.2)
    observation, _ = env.reset(seed=4, options={"scenario": "nominal"})
    assert observation.shape == (D1_OBSERVATION_SIZE,)
    for _ in range(20):
        observation, reward, terminated, truncated, info = env.step(np.zeros(2))
        assert np.isfinite(observation).all()
        assert reward == info["reward_terms"]["total"]
        assert not terminated
    assert truncated
    env.close()


def test_d1_reward_penalizes_vertical_residual_twice_as_much() -> None:
    common = {
        "forward_velocity_mps": 0.0,
        "roll_rad": 0.0,
        "pitch_rad": 0.0,
        "base_height_m": 0.455,
        "joint_position": NOMINAL_JOINT_POSITION,
        "command_velocity_mps": 0.0,
        "command_height_m": 0.455,
        "previous_normalized_action": np.zeros(2),
        "undesired_contacts": 0,
        "terminated": False,
    }
    longitudinal = calculate_d1_reward(
        normalized_action=np.asarray((1.0, 0.0)),
        **common,
    )
    vertical = calculate_d1_reward(
        normalized_action=np.asarray((0.0, 1.0)),
        **common,
    )
    assert longitudinal.height_tracking == 1.0
    assert vertical.residual_effort == 2.0 * longitudinal.residual_effort
