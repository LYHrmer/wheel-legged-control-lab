"""Eight bounded pure-arithmetic checks for the fixed drive damping stage.

Authored by GPT-6-sol as the fallback after the requested Opus test call failed.
No engine, model, controller, or physical case is imported or executed here.
"""

import numpy as np
import pytest
from drive_damping_math_04 import drive_increment, protect_requested

B = 126.4374005337902
WHEELS = (3, 7, 11, 15)
ATOL = 1e-10
RTOL = 1e-12


def _input():
    rotation = np.eye(3)
    jacobian = np.zeros((4, 3, 4))
    jacobian[:, 0, 0] = 1.0
    velocity = np.zeros(16)
    return rotation, jacobian, velocity


def _neutral_position():
    return np.tile([0.0, 0.8, -1.5, 0.0], 4)


def test_signed_zero_commands_leave_identity_despite_leg_motion():
    rotation, jacobian, velocity = _input()
    velocity[[0, 4, 8, 12]] = [1.0, -2.0, 3.0, -4.0]
    for zero in (+0.0, -0.0):
        result = drive_increment(rotation, jacobian, velocity, raw_forward_mps=zero)
        assert result["active"] is False
        np.testing.assert_array_equal(result["relative_forward_mps"], [1, -2, 3, -4])
        np.testing.assert_array_equal(result["delta_torque_nm"], np.zeros(16))
        assert result["raw_joint_power_w"] == 0.0


def test_forward_and_reverse_use_same_nonzero_raw_gate_and_gain():
    rotation, jacobian, velocity = _input()
    velocity[0] = 0.5
    forward = drive_increment(rotation, jacobian, velocity, raw_forward_mps=0.2)
    reverse = drive_increment(rotation, jacobian, velocity, raw_forward_mps=-0.3)
    assert forward["active"] is reverse["active"] is True
    np.testing.assert_array_equal(forward["delta_torque_nm"], reverse["delta_torque_nm"])
    np.testing.assert_allclose(forward["delta_torque_nm"][0], -0.5 * B,
                               rtol=RTOL, atol=ATOL)


def test_nonidentity_rotation_uses_first_column_and_ignores_fourth_jacobian_column():
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    jacobian = np.zeros((4, 3, 4))
    jacobian[0, 0, 0] = 9.0
    jacobian[0, 1, 0] = 2.0
    jacobian[0, 1, 3] = 1000.0
    velocity = np.zeros(16)
    velocity[0] = 3.0
    velocity[3] = 100.0
    result = drive_increment(rotation, jacobian, velocity, raw_forward_mps=0.2)
    np.testing.assert_array_equal(result["projected_jx"][0], [2.0, 0.0, 0.0])
    assert result["relative_forward_mps"][0] == 6.0
    np.testing.assert_allclose(result["delta_torque_nm"][0], -12.0 * B,
                               rtol=RTOL, atol=ATOL)
    assert result["delta_torque_nm"][3] == 0.0


def test_four_leg_delta_and_raw_sample_power_match_independent_numbers():
    rotation, jacobian, velocity = _input()
    jacobian[:, 0, 0] = [1.0, 2.0, 3.0, 4.0]
    velocity[[0, 4, 8, 12]] = [0.1, -0.2, 0.3, -0.4]
    velocity[list(WHEELS)] = [100.0, -80.0, 60.0, -40.0]
    result = drive_increment(rotation, jacobian, velocity, raw_forward_mps=0.225)
    relative = np.array([0.1, -0.4, 0.9, -1.6])
    expected = np.zeros(16)
    expected[[0, 4, 8, 12]] = -B * np.array([0.1, -0.8, 2.7, -6.4])
    np.testing.assert_allclose(result["relative_forward_mps"], relative,
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result["delta_torque_nm"], expected,
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(result["delta_torque_nm"][list(WHEELS)], np.zeros(4))
    expected_power = -B * (0.01 + 0.16 + 0.81 + 2.56)
    np.testing.assert_allclose(result["raw_joint_power_w"], expected_power,
                               rtol=RTOL, atol=ATOL)
    assert result["raw_joint_power_w"] < 0.0
    velocity[list(WHEELS)] = [0.0, 0.0, 0.0, 0.0]
    without_wheel_motion = drive_increment(rotation, jacobian, velocity,
                                           raw_forward_mps=0.225)
    np.testing.assert_array_equal(without_wheel_motion["delta_torque_nm"],
                                  result["delta_torque_nm"])


def test_stationary_legs_with_rotating_wheels_have_zero_damping():
    rotation, jacobian, velocity = _input()
    velocity[list(WHEELS)] = [30.0, -30.0, 50.0, -50.0]
    result = drive_increment(rotation, jacobian, velocity, raw_forward_mps=0.2)
    assert result["active"] is True
    np.testing.assert_array_equal(result["relative_forward_mps"], np.zeros(4))
    np.testing.assert_array_equal(result["delta_torque_nm"], np.zeros(16))
    assert result["raw_joint_power_w"] == 0.0


def test_rated_clip_precedes_exact_position_boundary_suppression():
    position = _neutral_position()
    velocity = np.zeros(16)
    requested = np.zeros(16)
    position[[0, 4]] = 0.785398
    position[[1, 5]] = -1.8326
    requested[[0, 1, 4, 5]] = [100.0, -100.0, -100.0, 100.0]
    requested[3] = 20.0
    safe, limited = protect_requested(requested, position, velocity)
    np.testing.assert_array_equal(safe[[0, 1, 4, 5, 3]], [0.0, 0.0, -80.0, 80.0, 12.0])
    assert all(limited[[0, 1, 4, 5, 3]])
    assert np.count_nonzero(safe) == 3


def test_speed_limit_boundary_suppresses_outward_but_allows_inward_on_both_signs():
    position = _neutral_position()
    velocity = np.zeros(16)
    requested = np.zeros(16)
    velocity[[0, 1, 2, 3, 7]] = [20.0, -20.0, 20.0, -30.0, 30.0]
    requested[[0, 1, 2, 3, 7]] = [5.0, -5.0, -5.0, 5.0, 5.0]
    safe, limited = protect_requested(requested, position, velocity)
    np.testing.assert_array_equal(safe[[0, 1, 2, 3, 7]], [0.0, 0.0, -5.0, 5.0, 0.0])
    np.testing.assert_array_equal(limited[[0, 1, 2, 3, 7]],
                                  [True, True, False, False, True])


def test_malformed_shape_type_and_nonfinite_inputs_are_rejected():
    rotation, jacobian, velocity = _input()
    cases = [
        (rotation, jacobian, velocity, True),
        (rotation, jacobian, velocity, 0.2 + 0j),
        (rotation, jacobian, velocity, float("nan")),
        (np.ones((2, 2)), jacobian, velocity, 0.2),
        (rotation, np.zeros((4, 3, 3)), velocity, 0.2),
        (rotation, jacobian, np.zeros(15), 0.2),
        (np.full((3, 3), float("inf")), jacobian, velocity, 0.2),
        (rotation, np.full((4, 3, 4), float("nan")), velocity, 0.2),
        (rotation, jacobian, np.full(16, float("inf")), 0.2),
    ]
    for matrix, feet, rates, raw in cases:
        with pytest.raises((TypeError, ValueError)):
            drive_increment(matrix, feet, rates, raw_forward_mps=raw)
    neutral = _neutral_position()
    for request, position, rates in [
        (np.zeros(15), neutral, velocity),
        (np.zeros(16, dtype=bool), neutral, velocity),
        (np.full(16, float("nan")), neutral, velocity),
        (np.zeros(16), np.full(16, float("inf")), velocity),
        (np.zeros(16), neutral, np.full(16, float("nan"))),
    ]:
        with pytest.raises((TypeError, ValueError)):
            protect_requested(request, position, rates)
