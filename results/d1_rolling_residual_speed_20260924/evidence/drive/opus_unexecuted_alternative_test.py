"""Pure arithmetic tests for the fixed drive damping candidate.

Standard library + NumPy + pytest only. No engine, model, controller or
network import. Expected values are hand-computed from the contract formula
`delta[leg] = -b * jx * (jx @ qdot_leg)`, never by calling the tested function.
"""

from __future__ import annotations

import numpy as np
import pytest

from drive_damping_math_04 import (
    GAIN_NSPM,
    POSITION_HIGH,
    POSITION_LOW,
    TORQUE_LIMIT,
    VELOCITY_LIMIT,
    WHEEL_INDICES,
    drive_increment,
    protect_requested,
)

B = 126.4374005337902  # hand-written copy of the frozen stopping gain
TOL = dict(rtol=1e-12, atol=1e-10)  # algebraic recomputation roundoff only

IDENTITY = np.eye(3)
# Cyclic permutation matrix: orthogonal, det = +1, so a proper rotation.
# column 0 = (0, 1, 0)  while  row 0 = (0, 0, 1)  -> column/row confusion visible.
CYCLIC = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

# Interior, finite joint positions for every leg (joint 2 range is [-2.775, -0.855]).
SAFE_POSITION = np.tile((0.0, 0.5, -1.5, 0.25), 4)


def _jacobian(leg_blocks: object, wheel_column: float = 7.5) -> np.ndarray:
    """Build (4, 3, 4) feet Jacobians; the wheel column must be ignored."""
    feet = np.zeros((4, 3, 4))
    feet[:, :, :3] = np.asarray(leg_blocks, dtype=np.float64)
    feet[:, :, 3] = wheel_column
    return feet


# Row 0 of each block is the only row an identity forward axis can select.
FOUR_LEG_BLOCKS = [
    [[1.0, 2.0, 0.0], [9.0, -4.0, 6.0], [-3.0, 5.0, 2.0]],
    [[0.0, 1.0, -1.0], [2.0, 8.0, -7.0], [4.0, -6.0, 1.0]],
    [[2.0, 0.0, 1.0], [-5.0, 3.0, 9.0], [7.0, 2.0, -8.0]],
    [[0.0, 0.0, 3.0], [6.0, 1.0, -2.0], [-9.0, 4.0, 5.0]],
]
FOUR_LEG_JX = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0], [2.0, 0.0, 1.0], [0.0, 0.0, 3.0]])
FOUR_LEG_QDOT = np.array([[1.0, 0.0, 3.0], [5.0, 3.0, 1.0], [1.0, 7.0, 2.0], [4.0, 4.0, -1.0]])
FOUR_LEG_U = np.array([1.0, 2.0, 4.0, -3.0])  # jx @ qdot_leg, by hand


def _four_leg_velocity(wheel_velocity: tuple[float, float, float, float]) -> np.ndarray:
    velocity = np.zeros(16)
    for leg in range(4):
        velocity[4 * leg : 4 * leg + 3] = FOUR_LEG_QDOT[leg]
    velocity[list(WHEEL_INDICES)] = wheel_velocity
    return velocity


def test_signed_zero_command_is_inactive_identity_despite_motion() -> None:
    jacobian = _jacobian(FOUR_LEG_BLOCKS)
    velocity = _four_leg_velocity((25.0, -30.0, 0.0, 12.5))
    for command in (0.0, -0.0):
        result = drive_increment(IDENTITY, jacobian, velocity, raw_forward_mps=command)
        assert result["active"] is False
        # The gate suppresses torque only; relative motion is still reported.
        np.testing.assert_allclose(result["relative_forward_mps"], FOUR_LEG_U, **TOL)
        assert np.count_nonzero(result["relative_forward_mps"]) == 4
        assert np.array_equal(result["delta_torque_nm"], np.zeros(16))
        assert result["raw_joint_power_w"] == 0.0


def test_gate_is_direction_and_magnitude_independent() -> None:
    assert GAIN_NSPM == B  # the module keeps the frozen stopping gain verbatim
    jacobian = _jacobian(FOUR_LEG_BLOCKS)
    velocity = _four_leg_velocity((25.0, -30.0, 0.0, 12.5))
    expected = np.zeros(16)
    for leg in range(4):
        expected[4 * leg : 4 * leg + 3] = -B * FOUR_LEG_JX[leg] * FOUR_LEG_U[leg]
    reference = None
    for command in (0.4, -0.4, 2.5, -1e-9):
        result = drive_increment(IDENTITY, jacobian, velocity, raw_forward_mps=command)
        assert result["active"] is True
        np.testing.assert_allclose(result["delta_torque_nm"], expected, **TOL)
        if reference is None:
            reference = result["delta_torque_nm"]
        else:  # identical damping for forward, reverse and any magnitude
            assert np.array_equal(result["delta_torque_nm"], reference)


def test_rotation_projects_first_column_not_first_row() -> None:
    block = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]]
    jacobian = _jacobian([block] * 4)
    velocity = np.tile((0.0, 3.0, 0.0, 11.0), 4)
    column_jx = np.array([0.0, 2.0, 0.0])  # CYCLIC[:, 0] @ block -> block row 1
    row_jx = np.array([0.0, 0.0, 4.0])  # CYCLIC[0, :] @ block -> block row 2 (wrong)
    result = drive_increment(CYCLIC, jacobian, velocity, raw_forward_mps=0.2)
    for leg in range(4):
        np.testing.assert_allclose(result["projected_jx"][leg], column_jx, **TOL)
        assert not np.allclose(result["projected_jx"][leg], row_jx)
    np.testing.assert_allclose(result["relative_forward_mps"], np.full(4, 6.0), **TOL)
    expected_leg = -B * column_jx * 6.0
    for leg in range(4):
        np.testing.assert_allclose(result["delta_torque_nm"][4 * leg : 4 * leg + 3], expected_leg, **TOL)


def test_four_leg_delta_power_identity_and_wheel_independence() -> None:
    jacobian = _jacobian(FOUR_LEG_BLOCKS)
    expected = np.zeros(16)
    for leg in range(4):
        expected[4 * leg : 4 * leg + 3] = -B * FOUR_LEG_JX[leg] * FOUR_LEG_U[leg]
    # Sample raw-increment identity only: -b * sum(u^2). Not global passivity.
    expected_power = -B * float(FOUR_LEG_U @ FOUR_LEG_U)
    first = None
    for wheels in ((25.0, -30.0, 0.0, 12.5), (-18.0, 4.0, 30.0, -7.25)):
        velocity = _four_leg_velocity(wheels)
        result = drive_increment(IDENTITY, jacobian, velocity, raw_forward_mps=0.25)
        np.testing.assert_allclose(result["relative_forward_mps"], FOUR_LEG_U, **TOL)
        np.testing.assert_allclose(result["delta_torque_nm"], expected, **TOL)
        np.testing.assert_allclose(result["raw_joint_power_w"], expected_power, **TOL)
        assert result["raw_joint_power_w"] < 0.0
        assert np.array_equal(result["delta_torque_nm"][list(WHEEL_INDICES)], np.zeros(4))
        if first is None:
            first = (result["delta_torque_nm"], result["raw_joint_power_w"])
        else:  # arbitrary wheel speeds cannot perturb leg damping or raw power
            assert np.array_equal(result["delta_torque_nm"], first[0])
            assert result["raw_joint_power_w"] == first[1]


def test_stationary_legs_with_spinning_wheels_give_zero_damping() -> None:
    jacobian = _jacobian(FOUR_LEG_BLOCKS)
    velocity = np.zeros(16)
    velocity[list(WHEEL_INDICES)] = (30.0, -30.0, 17.5, -22.0)
    result = drive_increment(IDENTITY, jacobian, velocity, raw_forward_mps=0.2)
    assert result["active"] is True
    assert np.array_equal(result["relative_forward_mps"], np.zeros(4))
    assert np.array_equal(result["delta_torque_nm"], np.zeros(16))
    assert result["raw_joint_power_w"] == 0.0


def test_rated_clip_then_outward_position_suppression() -> None:
    assert np.array_equal(TORQUE_LIMIT, np.tile((80.0, 80.0, 80.0, 12.0), 4))
    assert np.array_equal(POSITION_LOW, np.tile((-0.785398, -1.8326, -2.775, -np.inf), 4))
    assert np.array_equal(POSITION_HIGH, np.tile((0.785398, 3.40339, -0.855, np.inf), 4))
    assert np.isfinite(SAFE_POSITION).all()
    assert (SAFE_POSITION > POSITION_LOW).all() and (SAFE_POSITION < POSITION_HIGH).all()

    safe, limited = protect_requested(
        np.tile((100.0, -95.0, 80.0, 20.0), 4), SAFE_POSITION, np.zeros(16))
    assert np.array_equal(safe, np.tile((80.0, -80.0, 80.0, 12.0), 4))
    assert np.array_equal(limited, np.tile((True, True, False, True), 4))

    position = SAFE_POSITION.copy()
    position[0] = 0.785398  # exactly at the upper bound
    position[5] = -1.8326  # exactly at the lower bound
    position[8] = 0.785398
    position[13] = -1.8326
    requested = np.zeros(16)
    requested[0] = 3.0  # outward at upper bound -> suppressed
    requested[5] = -3.0  # outward at lower bound -> suppressed
    requested[8] = -3.0  # inward at upper bound -> allowed
    requested[13] = 3.0  # inward at lower bound -> allowed
    requested[3] = 6.0  # wheel: infinite position bounds
    safe, limited = protect_requested(requested, position, np.zeros(16))
    expected = np.zeros(16)
    expected[8], expected[13], expected[3] = -3.0, 3.0, 6.0
    assert np.array_equal(safe, expected)
    assert np.array_equal(np.flatnonzero(limited), np.array([0, 5]))


def test_exact_speed_boundary_suppresses_only_outward_torque() -> None:
    assert np.array_equal(VELOCITY_LIMIT, np.tile((20.0, 20.0, 20.0, 30.0), 4))
    velocity = np.zeros(16)
    requested = np.zeros(16)
    velocity[0], requested[0] = 20.0, 5.0  # outward at +limit
    velocity[1], requested[1] = -20.0, -5.0  # outward at -limit
    velocity[2], requested[2] = 20.0, -5.0  # inward at +limit -> allowed
    velocity[4], requested[4] = 19.999, 5.0  # below limit -> allowed
    velocity[5], requested[5] = 20.0, 0.0  # zero torque is not outward
    velocity[6], requested[6] = -25.0, -3.0  # beyond -limit, outward
    velocity[3], requested[3] = 30.0, 4.0  # wheel outward at +limit
    velocity[7], requested[7] = -30.0, -4.0  # wheel outward at -limit
    velocity[11], requested[11] = -30.0, 4.0  # wheel inward -> allowed
    velocity[15], requested[15] = 30.0, -4.0  # wheel inward -> allowed
    safe, limited = protect_requested(requested, SAFE_POSITION, velocity)
    expected = np.zeros(16)
    expected[2], expected[4], expected[11], expected[15] = -5.0, 5.0, 4.0, -4.0
    assert np.array_equal(safe, expected)
    assert np.array_equal(np.flatnonzero(limited), np.array([0, 1, 3, 6, 7]))


def test_malformed_inputs_are_rejected() -> None:
    jacobian = _jacobian(FOUR_LEG_BLOCKS)
    velocity = _four_leg_velocity((1.0, 2.0, 3.0, 4.0))
    nan_rotation = IDENTITY.copy()
    nan_rotation[1, 1] = np.nan
    inf_velocity = velocity.copy()
    inf_velocity[6] = np.inf
    nan_position = SAFE_POSITION.copy()
    nan_position[9] = np.nan
    cases = [
        (TypeError, lambda: drive_increment(np.eye(4), jacobian, velocity, raw_forward_mps=0.2)),
        (TypeError, lambda: drive_increment(IDENTITY, np.zeros((4, 3, 3)), velocity, raw_forward_mps=0.2)),
        (TypeError, lambda: drive_increment(IDENTITY, jacobian, np.zeros(15), raw_forward_mps=0.2)),
        (TypeError, lambda: drive_increment(IDENTITY, jacobian, velocity.astype(complex), raw_forward_mps=0.2)),
        (TypeError, lambda: drive_increment(IDENTITY, jacobian, velocity, raw_forward_mps=True)),
        (TypeError, lambda: drive_increment(IDENTITY, jacobian, velocity, raw_forward_mps=np.bool_(True))),
        (TypeError, lambda: drive_increment(IDENTITY, jacobian, velocity, raw_forward_mps="0.2")),
        (ValueError, lambda: drive_increment(IDENTITY, jacobian, velocity, raw_forward_mps=float("nan"))),
        (ValueError, lambda: drive_increment(IDENTITY, jacobian, velocity, raw_forward_mps=float("inf"))),
        (ValueError, lambda: drive_increment(nan_rotation, jacobian, velocity, raw_forward_mps=0.2)),
        (ValueError, lambda: drive_increment(IDENTITY, jacobian, inf_velocity, raw_forward_mps=0.2)),
        (TypeError, lambda: protect_requested(np.zeros((16, 1)), SAFE_POSITION, velocity)),
        (TypeError, lambda: protect_requested(np.zeros(16, dtype=bool), SAFE_POSITION, velocity)),
        (ValueError, lambda: protect_requested(np.zeros(16), nan_position, velocity)),
        (ValueError, lambda: protect_requested(np.zeros(16), SAFE_POSITION, inf_velocity)),
    ]
    assert len(cases) == 15
    for exception, call in cases:
        with pytest.raises(exception):
            call()
