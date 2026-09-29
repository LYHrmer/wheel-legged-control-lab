"""Hand-derived pure tests for residual16_math_08.

Eight nonparameterized cases. Expected values are derived independently on
paper from the contract constants (dt .01, leg PD 80/3, wheel PI 2.2/3,
I +/-4 Nm, leg offsets (.12,.25,.25) rad, wheel offset 4 rad/s, damping gain
126.4374005337902, wheel radius .087 m, torque 80/80/80/12, speed 20/20/20/30,
leg position limits +/-.785398 / -1.8326..3.40339 / -2.775..-.855) and are
written as literals, not recomputed with the implementation's expressions.

No engine import happens here: residual16_math_08 depends on NumPy only.
"""

from __future__ import annotations

import numpy as np
import pytest
from residual16_math_08 import compute_residual16, nominal_support

NOMINAL_Q = np.tile((0.0, 0.8, -1.5, 0.0), 4)
TOLERANCE = 1e-9


def _base_inputs() -> dict:
    """Neutral standing state: nominal pose, no motion, no gates but the action."""
    return {
        "joint_position_rad": NOMINAL_Q.copy(),
        "joint_velocity_rad_s": np.zeros(16),
        "nominal_joint_target_rad": NOMINAL_Q.copy(),
        "nominal_wheel_speed_rad_s": np.zeros(4),
        "support_torque_nm": np.zeros(16),
        "base_rotation_world_from_body": np.eye(3),
        "foot_jacobian_world": np.zeros((4, 3, 4)),
        "body_com_forward_mps": 0.0,
        "action_enabled": True,
        "leg_longitudinal_damping_active": False,
        "body_common_p_active": False,
    }


def test_zero_action_reproduces_hand_derived_support_pd_and_wheel_pi():
    """Symmetric support allocation plus the frozen PD/PI arithmetic at zero action."""
    offsets = np.asarray(
        ((0.2, 0.15, -0.4), (0.2, -0.15, -0.4), (-0.2, 0.15, -0.4), (-0.2, -0.15, -0.4))
    )
    jacobian = np.zeros((4, 3, 4))
    jacobian[:, 2, :3] = (0.0, -0.1, 0.05)
    support = nominal_support(
        {
            "base_rpy_rad": np.zeros(3),
            "base_angular_velocity_world": np.zeros(3),
            "base_rotation_world_from_body": np.eye(3),
            "foot_offset_world": offsets,
            "foot_jacobian_world": jacobian,
            "nominal_mass_kg": 40.0,
            "nominal_base_com_offset_body_m": np.zeros(3),
            "commanded_roll_rad": 0.0,
            "commanded_pitch_rad": 0.0,
        }
    )
    # 40 kg * 9.81 shared equally by a rectangular, zero-moment stance.
    assert support["upward_force_n"] == pytest.approx(392.4, abs=TOLERANCE)
    np.testing.assert_allclose(support["support_force_n"], [98.1] * 4, atol=TOLERANCE)
    assert not support["force_clip_mask"].any()
    np.testing.assert_allclose(
        support["support_torque_nm"], np.tile((0.0, 9.81, -4.905, 0.0), 4), atol=TOLERANCE
    )

    inputs = _base_inputs()
    inputs["joint_position_rad"] = np.concatenate(
        ((0.1, 0.9, -1.4, 0.5), NOMINAL_Q[4:])
    )
    inputs["joint_velocity_rad_s"] = np.concatenate(((0.0, 0.2, -0.1, 2.0), np.zeros(12)))
    inputs["nominal_wheel_speed_rad_s"] = np.asarray((3.0, 0.0, 0.0, 0.0))
    inputs["support_torque_nm"] = np.asarray(support["support_torque_nm"])
    inputs["foot_jacobian_world"] = jacobian

    result = compute_residual16(inputs, np.zeros(16), np.zeros(4))

    # Leg targets are reachable in one dt. Before the wheel-position override,
    # FL wheel .5 -> nominal 0 exceeds its .3 rad/tick geometric rate bound.
    # Its recorded pre-override mask is true; final wheel target remains .5.
    np.testing.assert_array_equal(
        result["rate_limit_mask"], [False, False, False, True] + [False] * 12
    )
    assert not result["position_clip_mask"].any()
    np.testing.assert_allclose(
        result["leg_pd_nm"][:4], (-8.0, -8.6, -7.7, 0.0), atol=TOLERANCE
    )
    # Wheel FL: error 3 - 2 = 1 rad/s, I 0 -> .03, request 2.2 + .03.
    np.testing.assert_allclose(result["wheel_speed_error_rad_s"], (1.0, 0.0, 0.0, 0.0), atol=TOLERANCE)
    np.testing.assert_allclose(result["wheel_integral_after_nm"], (0.03, 0.0, 0.0, 0.0), atol=TOLERANCE)
    expected = np.tile((0.0, 9.81, -4.905, 0.0), 4)
    expected[:4] = (-8.0, 1.21, -12.605, 2.23)
    np.testing.assert_allclose(result["safe_torque_nm"], expected, atol=TOLERANCE)
    assert not result["protection_changed_mask"].any()
    np.testing.assert_allclose(result["wheel_target_position_rad"], (0.5, 0.0, 0.0, 0.0), atol=TOLERANCE)


def test_single_hip_action_moves_only_that_joint():
    """Action slot 0 is FL hip: .12 rad scale, 80 Nm/rad, no cross-leg leakage."""
    action = np.zeros(16)
    action[0] = 0.5

    result = compute_residual16(_base_inputs(), action, np.zeros(4))

    np.testing.assert_allclose(result["leg_action_offset_rad"][0], 0.06, atol=TOLERANCE)
    assert np.count_nonzero(result["leg_action_offset_rad"]) == 1
    assert result["geometric_joint_target_rad"][0] == pytest.approx(0.06, abs=TOLERANCE)
    np.testing.assert_allclose(result["geometric_joint_target_rad"][1:], NOMINAL_Q[1:], atol=TOLERANCE)
    # 80 * .06 rad with zero velocity, zero support and zero wheel error.
    assert result["safe_torque_nm"][0] == pytest.approx(4.8, abs=TOLERANCE)
    assert np.count_nonzero(result["safe_torque_nm"][1:]) == 0
    assert not result["wheel_integral_after_nm"].any()


def test_wheel_pi_updates_once_and_holds_integral_on_saturation():
    """Anti-windup freezes I when the PI request exceeds the 12 Nm wheel limit."""
    inputs = _base_inputs()
    inputs["nominal_wheel_speed_rad_s"] = np.asarray((30.0, 1.0, 0.0, 0.0))
    integral_before = np.asarray((4.0, 0.5, 0.0, 0.0))

    result = compute_residual16(inputs, np.zeros(16), integral_before)

    np.testing.assert_allclose(result["wheel_speed_error_rad_s"], (30.0, 1.0, 0.0, 0.0), atol=TOLERANCE)
    # Wheel 0: 4 + 3*30*.01 = 4.9 clipped to 4; request 66 + 4 = 70 > 12 and not
    # unwinding, so the integral stays at 4. Wheel 1: 0.5 + .03 committed.
    np.testing.assert_allclose(result["wheel_integral_candidate_nm"], (4.0, 0.53, 0.0, 0.0), atol=TOLERANCE)
    np.testing.assert_allclose(result["wheel_candidate_request_nm"], (70.0, 2.73, 0.0, 0.0), atol=TOLERANCE)
    np.testing.assert_array_equal(result["wheel_integral_commit_mask"], (False, True, True, True))
    np.testing.assert_allclose(result["wheel_integral_after_nm"], (4.0, 0.53, 0.0, 0.0), atol=TOLERANCE)
    np.testing.assert_allclose(
        result["base_wheel_torque_nm"][[3, 7, 11, 15]], (70.0, 2.73, 0.0, 0.0), atol=TOLERANCE
    )
    np.testing.assert_allclose(
        result["safe_torque_nm"][[3, 7, 11, 15]], (12.0, 2.73, 0.0, 0.0), atol=TOLERANCE
    )
    assert result["torque_clip_mask"][3]
    assert not result["torque_clip_mask"][7]

    # One integration per call: the same arguments must give the same integral.
    repeat = compute_residual16(inputs, np.zeros(16), integral_before)
    np.testing.assert_array_equal(result["wheel_integral_after_nm"], repeat["wheel_integral_after_nm"])


def test_common_wheel_p_keeps_differential_and_integral():
    """The common increment is one shared Nm offset; differential and I stay put."""
    inputs = _base_inputs()
    inputs["joint_velocity_rad_s"] = np.zeros(16)
    inputs["joint_velocity_rad_s"][[3, 7, 11, 15]] = (10.0, 6.0, 8.0, 8.0)
    inputs["nominal_wheel_speed_rad_s"] = np.asarray((12.0, 6.0, 8.0, 8.0))
    inputs["body_com_forward_mps"] = 0.348
    inputs["body_common_p_active"] = True

    result = compute_residual16(inputs, np.zeros(16), np.zeros(4))

    # mean omega 8 rad/s vs .348/.087 = 4 rad/s body equivalent; 2.2 * 4 = 8.8 Nm.
    assert result["common_wheel_error_rad_s"] == pytest.approx(4.0, abs=TOLERANCE)
    assert result["common_wheel_scalar_delta_nm"] == pytest.approx(8.8, abs=TOLERANCE)
    assert np.count_nonzero(result["common_wheel_delta_torque_nm"]) == 4
    # Only wheel 0 has speed error 2 rad/s: PI 4.4 + .06 = 4.46 Nm differential.
    np.testing.assert_allclose(result["wheel_integral_after_nm"], (0.06, 0.0, 0.0, 0.0), atol=TOLERANCE)
    np.testing.assert_allclose(
        result["final_request_torque_nm"][[3, 7, 11, 15]], (13.26, 8.8, 8.8, 8.8), atol=TOLERANCE
    )
    np.testing.assert_allclose(
        result["safe_torque_nm"][[3, 7, 11, 15]], (12.0, 8.8, 8.8, 8.8), atol=TOLERANCE
    )
    assert result["final_request_torque_nm"][3] - result["final_request_torque_nm"][7] == pytest.approx(
        4.46, abs=TOLERANCE
    )

    inputs["body_common_p_active"] = False
    without = compute_residual16(inputs, np.zeros(16), np.zeros(4))
    np.testing.assert_array_equal(result["wheel_integral_after_nm"], without["wheel_integral_after_nm"])
    np.testing.assert_array_equal(result["base_wheel_torque_nm"], without["base_wheel_torque_nm"])


def test_longitudinal_leg_damping_is_dissipative_and_skips_wheels():
    """-126.4374005337902 * Jx * (Jx . qdot) removes joint power, wheels untouched."""
    inputs = _base_inputs()
    jacobian = np.zeros((4, 3, 4))
    jacobian[0, 0, :3] = (1.0, 0.0, 0.0)
    jacobian[1, 0, :3] = (0.0, 0.5, 0.0)
    inputs["foot_jacobian_world"] = jacobian
    velocity = np.zeros(16)
    velocity[0] = 2.0
    velocity[5] = -4.0
    inputs["joint_velocity_rad_s"] = velocity
    inputs["leg_longitudinal_damping_active"] = True

    result = compute_residual16(inputs, np.zeros(16), np.zeros(4))

    np.testing.assert_allclose(result["projected_jx"][0], (1.0, 0.0, 0.0), atol=TOLERANCE)
    np.testing.assert_allclose(result["relative_forward_mps"], (2.0, -2.0, 0.0, 0.0), atol=TOLERANCE)
    delta = result["leg_damping_delta_torque_nm"]
    assert delta[0] == pytest.approx(-252.8748010675804, abs=1e-9)
    assert delta[5] == pytest.approx(126.4374005337902, abs=1e-9)
    assert np.count_nonzero(delta) == 2
    np.testing.assert_array_equal(delta[[3, 7, 11, 15]], np.zeros(4))
    # 2 * (-252.8748010675804) rad/s*Nm on leg 0 plus (-4) * 126.4374005337902.
    assert result["leg_damping_raw_joint_power_w"] < 0.0
    assert result["leg_damping_raw_joint_power_w"] == pytest.approx(-1011.4992042703216, abs=1e-8)
    # PD gives -6 Nm and +12 Nm, so both damped joints saturate at the 80 Nm box.
    assert result["final_request_torque_nm"][0] == pytest.approx(-258.8748010675804, abs=1e-8)
    assert result["safe_torque_nm"][0] == pytest.approx(-80.0, abs=TOLERANCE)
    assert result["safe_torque_nm"][5] == pytest.approx(80.0, abs=TOLERANCE)
    assert result["torque_clip_mask"][0] and result["torque_clip_mask"][5]
    np.testing.assert_array_equal(result["safe_torque_nm"][[3, 7, 11, 15]], np.zeros(4))


def test_disabled_action_keeps_raw_but_applies_zero():
    """Raw/clipped stay auditable, applied is exactly zero, PI still advances once."""
    inputs = _base_inputs()
    inputs["action_enabled"] = False
    inputs["nominal_wheel_speed_rad_s"] = np.ones(4)
    raw = np.asarray(
        (
            2.0, -3.0, 0.5, -0.25, 0.75, 1.0, -1.0, 0.1,
            -0.1, 0.9, -0.9, 0.3, 5.0, -5.0, 0.5, -0.5,
        )
    )

    result = compute_residual16(inputs, raw, np.zeros(4))

    np.testing.assert_array_equal(result["raw_action"], raw)
    expected_clipped = np.asarray(
        (
            1.0, -1.0, 0.5, -0.25, 0.75, 1.0, -1.0, 0.1,
            -0.1, 0.9, -0.9, 0.3, 1.0, -1.0, 0.5, -0.5,
        )
    )
    np.testing.assert_array_equal(result["clipped_action"], expected_clipped)
    expected_mask = np.zeros(16, dtype=bool)
    expected_mask[[0, 1, 12, 13]] = True
    np.testing.assert_array_equal(result["action_clip_mask"], expected_mask)
    np.testing.assert_array_equal(result["applied_action"], np.zeros(16))
    np.testing.assert_array_equal(result["leg_action_offset_rad"], np.zeros(12))
    np.testing.assert_array_equal(result["wheel_action_offset_rad_s"], np.zeros(4))
    np.testing.assert_allclose(result["geometric_joint_target_rad"], NOMINAL_Q, atol=TOLERANCE)
    np.testing.assert_allclose(result["wheel_speed_target_rad_s"], np.ones(4), atol=TOLERANCE)
    # The single PI update is independent of the action gate: 3 * 1 * .01.
    np.testing.assert_allclose(result["wheel_integral_after_nm"], np.full(4, 0.03), atol=TOLERANCE)
    np.testing.assert_allclose(
        result["safe_torque_nm"][[3, 7, 11, 15]], np.full(4, 2.23), atol=TOLERANCE
    )
    np.testing.assert_array_equal(result["safe_torque_nm"][[0, 1, 2]], np.zeros(3))


def test_rate_then_position_then_outward_protection_order():
    """Rate clip precedes position clip, and outward suppression follows the box clip."""
    inputs = _base_inputs()
    inputs["joint_position_rad"] = np.asarray(
        (
            0.785398, 0.8, -1.5, 0.0,
            -0.785398, 0.8, -1.5, 0.0,
            0.0, 0.8, -1.5, 0.0,
            0.0, 0.8, -0.9, 0.0,
        )
    )
    inputs["nominal_joint_target_rad"] = np.asarray(
        (
            0.0, 0.8, -1.5, 0.0,
            0.0, 0.8, -1.5, 0.0,
            0.0, 0.8, -1.5, 0.0,
            0.0, 0.8, -0.9, 0.0,
        )
    )
    velocity = np.zeros(16)
    velocity[8] = 20.0
    inputs["joint_velocity_rad_s"] = velocity
    support = np.zeros(16)
    support[0] = 25.0
    support[4] = -25.0
    support[5] = 200.0
    support[8] = 100.0
    inputs["support_torque_nm"] = support
    action = np.zeros(16)
    action[10] = 1.0
    action[11] = 1.0

    result = compute_residual16(inputs, action, np.zeros(4))

    # FL/FR hips are 0.2 rad away, RR thigh wants +.25 rad, RR calf wants +.25 rad.
    expected_rate = np.zeros(16, dtype=bool)
    expected_rate[[0, 4, 13, 14]] = True
    np.testing.assert_array_equal(result["rate_limit_mask"], expected_rate)
    np.testing.assert_allclose(
        result["rate_limited_joint_target_rad"][[0, 4, 13, 14]],
        (0.585398, -0.585398, 1.0, -0.7),
        atol=TOLERANCE,
    )
    # Only the RR calf rate-limited target leaves the -.855 rad upper bound.
    expected_position = np.zeros(16, dtype=bool)
    expected_position[14] = True
    np.testing.assert_array_equal(result["position_clip_mask"], expected_position)
    assert result["joint_target_rad"][14] == pytest.approx(-0.855, abs=TOLERANCE)

    assert result["final_request_torque_nm"][8] == pytest.approx(40.0, abs=TOLERANCE)
    expected_upper = np.zeros(16, dtype=bool)
    expected_upper[0] = True
    expected_lower = np.zeros(16, dtype=bool)
    expected_lower[4] = True
    expected_speed = np.zeros(16, dtype=bool)
    expected_speed[8] = True
    np.testing.assert_array_equal(result["upper_outward_mask"], expected_upper)
    np.testing.assert_array_equal(result["lower_outward_mask"], expected_lower)
    np.testing.assert_array_equal(result["speed_outward_mask"], expected_speed)
    expected_clip = np.zeros(16, dtype=bool)
    expected_clip[5] = True
    np.testing.assert_array_equal(result["torque_clip_mask"], expected_clip)

    expected_torque = np.zeros(16)
    expected_torque[5] = 80.0
    expected_torque[13] = 16.0
    expected_torque[14] = 3.6
    np.testing.assert_allclose(result["safe_torque_nm"], expected_torque, atol=TOLERANCE)


def test_invalid_inputs_are_rejected_without_mutating_arguments():
    """Shape/dtype/finiteness/rotation/integral domains fail loudly; inputs are read-only."""
    with pytest.raises(KeyError):
        missing = _base_inputs()
        del missing["support_torque_nm"]
        compute_residual16(missing, np.zeros(16), np.zeros(4))

    with pytest.raises(TypeError):
        extra = _base_inputs()
        extra["wheel_kp"] = 2.2
        compute_residual16(extra, np.zeros(16), np.zeros(4))

    with pytest.raises(TypeError):
        wrong_shape = _base_inputs()
        wrong_shape["joint_position_rad"] = np.zeros(15)
        compute_residual16(wrong_shape, np.zeros(16), np.zeros(4))

    with pytest.raises(TypeError):
        compute_residual16(_base_inputs(), np.ones(16, dtype=bool), np.zeros(4))

    with pytest.raises(TypeError):
        gate = _base_inputs()
        gate["action_enabled"] = 1
        compute_residual16(gate, np.zeros(16), np.zeros(4))

    nonfinite = np.zeros(16)
    nonfinite[7] = np.nan
    with pytest.raises(ValueError):
        compute_residual16(_base_inputs(), nonfinite, np.zeros(4))

    with pytest.raises(ValueError):
        scaled = _base_inputs()
        scaled["base_rotation_world_from_body"] = 1.5 * np.eye(3)
        compute_residual16(scaled, np.zeros(16), np.zeros(4))

    with pytest.raises(ValueError):
        reflected = _base_inputs()
        reflected["base_rotation_world_from_body"] = np.diag((1.0, 1.0, -1.0))
        compute_residual16(reflected, np.zeros(16), np.zeros(4))

    with pytest.raises(ValueError):
        compute_residual16(_base_inputs(), np.zeros(16), np.asarray((5.0, 0.0, 0.0, 0.0)))

    with pytest.raises(ValueError):
        wheel_support = _base_inputs()
        wheel_support["support_torque_nm"][3] = 1.0
        compute_residual16(wheel_support, np.zeros(16), np.zeros(4))

    # The final protected torque may remain finite while diagnostics from an
    # inactive branch overflow. Such a record must be rejected before return.
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError):
            inactive_common = _base_inputs()
            inactive_common["body_com_forward_mps"] = 1e308
            compute_residual16(inactive_common, np.zeros(16), np.zeros(4))

        with pytest.raises(ValueError):
            projected = _base_inputs()
            projected["joint_velocity_rad_s"][0] = 1e155
            projected["foot_jacobian_world"][0, 0, 0] = 1e155
            compute_residual16(projected, np.zeros(16), np.zeros(4))

        with pytest.raises(ValueError):
            support_overflow = {
                "base_rpy_rad": np.zeros(3),
                "base_angular_velocity_world": np.zeros(3),
                "base_rotation_world_from_body": np.eye(3),
                "foot_offset_world": np.zeros((4, 3)),
                "foot_jacobian_world": np.zeros((4, 3, 4)),
                "nominal_mass_kg": 1e308,
                "nominal_base_com_offset_body_m": np.zeros(3),
                "commanded_roll_rad": 0.0,
                "commanded_pitch_rad": 0.0,
            }
            nominal_support(support_overflow)

    inputs = _base_inputs()
    inputs["joint_position_rad"] = np.concatenate(((0.1, 0.9, -1.4, 0.5), NOMINAL_Q[4:]))
    inputs["nominal_wheel_speed_rad_s"] = np.asarray((3.0, 0.0, 0.0, 0.0))
    action = np.full(16, 0.5)
    integral = np.asarray((1.0, -1.0, 0.0, 2.0))
    snapshot = {key: np.array(value, copy=True) for key, value in inputs.items()
                if isinstance(value, np.ndarray)}
    action_snapshot = action.copy()
    integral_snapshot = integral.copy()

    result = compute_residual16(inputs, action, integral)

    for key, value in snapshot.items():
        np.testing.assert_array_equal(inputs[key], value)
    np.testing.assert_array_equal(action, action_snapshot)
    np.testing.assert_array_equal(integral, integral_snapshot)
    assert result["safe_torque_nm"].flags.writeable is False
    assert result["wheel_integral_after_nm"].flags.writeable is False
