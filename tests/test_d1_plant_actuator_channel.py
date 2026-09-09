"""The channel must affect every joint at physics rate, not only policy residuals."""

import numpy as np
import pytest

from wheel_legged_control.d1.actuator_channel import ActuatorChannel, ActuatorChannelConfig
from wheel_legged_control.d1.model import JOINT_TORQUE_LIMIT, D1Plant


def channel(**kwargs):
    return ActuatorChannel(
        ActuatorChannelConfig(torque_limit_nm=tuple(JOINT_TORQUE_LIMIT), **kwargs)
    )


@pytest.mark.parametrize("mode", ["legacy_mixed", "synchronized"])
def test_ideal_channel_preserves_full_body_trajectory(mode):
    direct = D1Plant(sampling_mode=mode)
    sampled = D1Plant(sampling_mode=mode, actuator_channel=channel())
    rng = np.random.default_rng(14)
    for _ in range(20):
        torque = rng.uniform(-14, 14, 16)
        for plant in (direct, sampled):
            plant.step(torque)
        np.testing.assert_array_equal(direct.data.qpos, sampled.data.qpos)
        np.testing.assert_array_equal(direct.data.qvel, sampled.data.qvel)
        assert len(sampled.last_control_interval_actuator_traces) == 5
        for trace in sampled.last_control_interval_actuator_traces:
            np.testing.assert_array_equal(trace.requested_nm, torque)
            np.testing.assert_array_equal(
                trace.applied_nm, np.clip(torque, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
            )
        np.testing.assert_array_equal(
            sampled.data.ctrl[sampled.actuator_ids],
            sampled.last_control_interval_actuator_traces[-1].applied_nm,
        )


def test_delay_is_in_physics_steps_and_applies_to_all_sixteen_axes():
    plant = D1Plant(sampling_mode="synchronized", actuator_channel=channel(delay_steps=2))
    torque = np.linspace(1, 2, 16)
    plant.step(torque)
    traces = plant.last_control_interval_actuator_traces
    for index, trace in enumerate(traces):
        np.testing.assert_array_equal(trace.applied_nm, np.zeros(16) if index < 2 else torque)
    # The next control interval still starts with the last interval's request.
    plant.step(-torque)
    for index, trace in enumerate(plant.last_control_interval_actuator_traces):
        np.testing.assert_array_equal(trace.applied_nm, torque if index < 2 else -torque)


def test_delay_changes_zero_policy_baseline_dynamics():
    direct = D1Plant(actuator_channel=channel())
    delayed = D1Plant(actuator_channel=channel(delay_steps=5))
    # This is a complete controller torque; there is no RL residual here.
    request = np.tile((1.0, 2.0, -3.0, 1.0), 4)
    direct.step(request)
    delayed.step(request)
    assert np.linalg.norm(direct.data.qvel - delayed.data.qvel) > 1e-3
    np.testing.assert_array_equal(delayed.data.ctrl, 0)


def test_response_is_updated_five_times_with_fresh_joint_speeds():
    plant = D1Plant(actuator_channel=channel(time_constant_s=0.01, gain=0.8))
    request = np.ones(16)
    plant.step(request)
    traces = plant.last_control_interval_actuator_traces
    for index, trace in enumerate(traces):
        expected = 0.8 * (1 - np.exp(-((index + 1) * 0.002) / 0.01))
        np.testing.assert_allclose(trace.applied_nm, expected, atol=1e-15)
    np.testing.assert_array_equal(traces[0].joint_velocity_rps, 0)
    assert np.linalg.norm(traces[-1].joint_velocity_rps) > 0.01


def test_reset_clears_motor_history_and_receipts():
    plant = D1Plant(actuator_channel=channel(delay_steps=2, time_constant_s=0.01))
    plant.step(np.ones(16))
    plant.reset()
    assert plant.last_control_interval_actuator_traces == ()
    plant.step(np.zeros(16))
    for trace in plant.last_control_interval_actuator_traces:
        np.testing.assert_array_equal(trace.applied_nm, 0)


def test_mismatched_limits_fail_before_motor_history_or_physics_advances():
    plant = D1Plant(actuator_channel=channel(delay_steps=1))
    plant.set_domain(actuator_strength_scale=0.8)
    with pytest.raises(ValueError, match="torque limits"):
        plant.step(np.ones(16))
    assert plant.data.time == 0
    plant.set_domain()
    plant.step(np.ones(16))
    np.testing.assert_array_equal(plant.last_control_interval_actuator_traces[0].applied_nm, 0)


@pytest.mark.parametrize(
    "config", [ActuatorChannelConfig(n_axes=1), ActuatorChannelConfig(physics_dt_s=0.01)]
)
def test_channel_dimensions_and_clock_are_not_silently_converted(config):
    with pytest.raises(ValueError, match="axes.*timestep"):
        D1Plant(actuator_channel=ActuatorChannel(config))
