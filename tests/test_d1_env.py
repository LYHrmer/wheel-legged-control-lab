import numpy as np
import pytest
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


def test_estimated_state_mode_delays_the_whole_control_stack() -> None:
    env = D1ResidualEnv(
        state_mode="estimated",
        randomize=False,
        episode_seconds=0.5,
    )
    observation, info = env.reset(
        seed=9,
        options={
            "scenario": "nominal",
            "randomize": False,
            "action_delay_steps": 1,
            "state_delay_steps": 2,
            "sensor_noise": 1.0,
        },
    )
    assert observation.shape == (D1_OBSERVATION_SIZE,)
    assert info["state_estimation_mode"] == "estimated"
    assert info["action_delay_steps"] == 1
    assert info["state_delay_steps"] == 2

    ages = []
    for _ in range(50):
        observation, _, terminated, _, info = env.step(np.zeros(2))
        ages.append(info["state_age_ms"])
        assert np.isfinite(observation).all()
        assert not terminated
    assert max(ages) == pytest.approx(20.0)
    env.close()


def test_oracle_reports_only_applied_delay_and_legacy_delay_is_action_only() -> None:
    oracle = D1ResidualEnv(state_mode="oracle", randomize=False, episode_seconds=0.1)
    _, oracle_info = oracle.reset(
        seed=3,
        options={"delay_steps": 2, "state_delay_steps": 3},
    )
    assert oracle_info["action_delay_steps"] == 2
    assert oracle_info["state_delay_steps"] == 0
    assert oracle_info["state_age_ms"] == 0.0
    oracle.close()

    estimated = D1ResidualEnv(state_mode="estimated", randomize=False, episode_seconds=0.1)
    _, estimated_info = estimated.reset(seed=3, options={"delay_steps": 2})
    assert estimated_info["action_delay_steps"] == 2
    assert estimated_info["state_delay_steps"] == 0
    estimated.close()


def test_estimated_env_reports_latency_compensation_without_hiding_state_age() -> None:
    env = D1ResidualEnv(
        state_mode="estimated",
        latency_compensation="constant_velocity",
        randomize=False,
        episode_seconds=0.1,
    )
    env.reset(seed=2, options={"state_delay_steps": 2, "sensor_noise": 0.0})

    info = {}
    for _ in range(4):
        _, _, _, _, info = env.step(np.zeros(2))

    assert info["state_age_ms"] == pytest.approx(20.0)
    assert info["latency_compensation"] == "constant_velocity"
    assert info["latency_compensation_status"] == "applied"
    assert info["latency_compensation_horizon_ms"] == pytest.approx(20.0)
    np.testing.assert_allclose(info["estimated_state"], info["control_state"])
    assert info["raw_estimated_state"].shape == info["control_state"].shape
    assert not np.shares_memory(info["raw_estimated_state"], info["control_state"])
    env.close()


def test_environment_exposes_constrained_contact_allocation_diagnostics() -> None:
    env = D1ResidualEnv(
        contact_allocation="constrained",
        randomize=False,
        episode_seconds=0.2,
        measure_contact_wrench=True,
        profile_allocation_timing=True,
    )
    _, info = env.reset(seed=5, options={"scenario": "nominal"})
    assert info["contact_allocation"] == "constrained"
    assert info["allocation_status"] == "not_run"
    assert info["allocation_wrench_tracking_status"] == "not_run"

    for _ in range(20):
        _, _, terminated, _, info = env.step(np.zeros(2))
        assert not terminated

    assert info["allocation_status"] in {"converged", "feasible_nonconverged"}
    assert info["allocation_status_reason"].startswith("slsqp_")
    assert info["allocation_wrench_tracking_status"] in {"tracked", "limited"}
    assert info["allocation_timing_measured"] is True
    assert info["allocation_solve_ms"] > 0.0
    assert info["allocation_constraint_violation"] <= 1e-7
    assert info["allocation_desired_wrench_world"].shape == (6,)
    assert info["allocation_achieved_wrench_world"].shape == (6,)
    assert info["measured_contact_wrench_world"].shape == (6,)
    assert info["measured_wheel_contact_force_world_n"].shape == (4, 3)
    assert info["contact_wrench_reference_position_world_m"].shape == (3,)
    assert info["measured_contact_wrench_world"][2] > 0.0
    assert info["contact_wrench_physics_samples"] == env.plant.physics_steps
    assert info["wheel_contact_active_sample_fraction"].shape == (4,)
    for metric in (
        "allocation_force_error_norm_n",
        "allocation_moment_error_norm_nm",
        "contact_force_model_error_norm_n",
        "contact_moment_model_error_norm_nm",
        "contact_force_tracking_error_norm_n",
        "contact_moment_tracking_error_norm_nm",
    ):
        assert info[metric] >= 0.0
    env.close()


def test_environment_rejects_an_unknown_contact_allocation_mode() -> None:
    with pytest.raises(ValueError, match="contact_allocation"):
        D1ResidualEnv(contact_allocation="magic")
