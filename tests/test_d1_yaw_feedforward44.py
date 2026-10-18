"""Yaw command feedforward: identity at ff=1 and at zero yaw command, exact law otherwise."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("mujoco")
from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource
from wheel_legged_control.d1.wheel_leg_controller import D1WheelLegController
from wheel_legged_control.d1.yaw_feedforward44 import (
    D1WheelLegYawFeedforwardController,
    adopt_yaw_feedforward,
)


@pytest.fixture
def state():
    plant = D1Plant(sampling_mode="synchronized")
    return D1MujocoTruthStateSource(plant).reset()


def same(a, b):
    assert np.array_equal(a.nominal_joint_target_rad, b.nominal_joint_target_rad)
    assert np.array_equal(a.nominal_wheel_speed_rad_s, b.nominal_wheel_speed_rad_s)
    assert np.array_equal(a.leg_extension_target_m, b.leg_extension_target_m)
    assert a.effective_yaw_request_rps == b.effective_yaw_request_rps


@pytest.mark.parametrize("feedforward", [1.0, 3.9])
def test_identity_at_unit_feedforward_and_at_zero_yaw_command(state, feedforward):
    base = D1WheelLegController(control_dt=0.01)
    candidate = D1WheelLegYawFeedforwardController(
        control_dt=0.01, yaw_command_feedforward=feedforward
    )
    turning = D1Command(forward_velocity_mps=0.6, yaw_rate_rps=0.3)
    straight = D1Command(forward_velocity_mps=0.6, yaw_rate_rps=0.0)
    same(candidate.nominal_targets(straight, state), base.nominal_targets(straight, state))
    if feedforward == 1.0:
        same(candidate.nominal_targets(turning, state), base.nominal_targets(turning, state))


def test_law_is_feedforward_plus_frozen_feedback_with_same_clip_and_kinematics(state):
    ff = 3.9
    candidate = D1WheelLegYawFeedforwardController(control_dt=0.01, yaw_command_feedforward=ff)
    candidate.yaw_request_limit_rps = 1.2
    base = D1WheelLegController(control_dt=0.01)
    base.yaw_request_limit_rps = 1.2
    command = D1Command(forward_velocity_mps=0.6, yaw_rate_rps=0.1)  # unsaturated: 0.5 and 3.9+0.4
    got, ref = candidate.nominal_targets(command, state), base.nominal_targets(command, state)
    measured = float(state.base_angular_velocity_body[2])
    expected = float(np.clip(ff * 0.1 + 4.0 * (0.1 - measured), -1.2, 1.2))
    assert got.effective_yaw_request_rps == pytest.approx(expected, abs=1e-12)
    assert got.effective_yaw_request_rps > ref.effective_yaw_request_rps
    # legs and extension are untouched; only the differential wheel speed changes
    assert np.array_equal(got.nominal_joint_target_rad, ref.nominal_joint_target_rad)
    assert np.array_equal(got.leg_extension_target_m, ref.leg_extension_target_m)
    assert not np.array_equal(got.nominal_wheel_speed_rad_s, ref.nominal_wheel_speed_rad_s)
    # common mode is essentially unchanged (foot offsets are only nearly symmetric); the
    # left/right difference carries the change
    common = got.nominal_wheel_speed_rad_s.mean() - ref.nominal_wheel_speed_rad_s.mean()
    assert abs(common) < 1e-3
    delta = got.nominal_wheel_speed_rad_s - ref.nominal_wheel_speed_rad_s
    assert np.abs(delta - delta.mean()).max() > 0.1


def test_request_limit_still_applies(state):
    candidate = D1WheelLegYawFeedforwardController(control_dt=0.01, yaw_command_feedforward=8.0)
    candidate.yaw_request_limit_rps = 1.2
    command = D1Command(yaw_rate_rps=0.3)
    assert candidate.nominal_targets(command, state).effective_yaw_request_rps == 1.2


def test_adopt_converts_only_the_exact_frozen_class_and_keeps_state(state):
    base = D1WheelLegController(control_dt=0.01)
    base._wheel_integral_nm[:] = 0.25
    adopted = adopt_yaw_feedforward(base, 3.9)
    assert adopted is base and type(adopted) is D1WheelLegYawFeedforwardController
    assert adopted.yaw_command_feedforward == 3.9 and np.all(adopted._wheel_integral_nm == 0.25)
    assert adopted.control_schema.endswith("yaw-feedforward-v5")
    with pytest.raises(TypeError):
        adopt_yaw_feedforward(adopted, 2.0)


@pytest.mark.parametrize("bad", [-0.1, 9.0, float("nan"), float("inf")])
def test_invalid_feedforward_is_rejected(bad):
    with pytest.raises(ValueError):
        D1WheelLegYawFeedforwardController(control_dt=0.01, yaw_command_feedforward=bad)
    with pytest.raises(ValueError):
        adopt_yaw_feedforward(D1WheelLegController(control_dt=0.01), bad)


def test_compute_uses_the_feedforward_law(state):
    candidate = D1WheelLegYawFeedforwardController(control_dt=0.01, yaw_command_feedforward=3.9)
    base = D1WheelLegController(control_dt=0.01)
    candidate.yaw_request_limit_rps = base.yaw_request_limit_rps = 1.2
    command = D1Command(forward_velocity_mps=0.3, yaw_rate_rps=0.1)
    candidate.compute(command, state, np.zeros(8))
    base.compute(command, state, np.zeros(8))
    assert candidate.last_result.wheel_speed_target_rad_s is not None
    assert not np.array_equal(
        candidate.last_result.wheel_speed_target_rad_s, base.last_result.wheel_speed_target_rad_s
    )


def test_zero_yaw_command_is_bitwise_frozen_even_with_nonzero_measured_yaw_rate(state):
    body = np.array([0.0, 0.0, 0.37])
    spinning = replace(
        state,
        base_angular_velocity_body=body,
        base_angular_velocity_world=state.base_rotation @ body,
    )
    base = D1WheelLegController(control_dt=0.01)
    candidate = D1WheelLegYawFeedforwardController(control_dt=0.01, yaw_command_feedforward=3.9)
    for limit in (0.6, 1.2):
        base.yaw_request_limit_rps = candidate.yaw_request_limit_rps = limit
        command = D1Command(forward_velocity_mps=0.9, yaw_rate_rps=0.0)
        same(candidate.nominal_targets(command, spinning), base.nominal_targets(command, spinning))
