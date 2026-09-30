"""Pure C17 controller arithmetic and independent-checker regression fixtures."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
W = HERE.parent
sys.path.insert(0, str(W))
sys.path.insert(0, str(W / "course_impl08"))
sys.path.insert(0, str(HERE))

from residual16_math_08 import compute_residual16
from residual17 import compute_residual17, nominal_support
from verify_control17 import check_controller


WHEELS = (3, 7, 11, 15)
NOMINAL_Q = np.tile((0.0, .8, -1.5, 0.0), 4)


def inputs(*, variant="ramp", servo=.45, body=.5, wheel_omega=4.5,
           wheel_target=None, common=True):
    qdot = np.zeros(16)
    qdot[list(WHEELS)] = wheel_omega
    return {
        "joint_position_rad": NOMINAL_Q.copy(),
        "joint_velocity_rad_s": qdot,
        "nominal_joint_target_rad": NOMINAL_Q.copy(),
        "nominal_wheel_speed_rad_s": np.full(4, servo/.087)
        if wheel_target is None else np.asarray(wheel_target, dtype=float),
        "support_torque_nm": np.zeros(16),
        "base_rotation_world_from_body": np.eye(3),
        "foot_jacobian_world": np.zeros((4, 3, 4)),
        "body_com_forward_mps": body,
        "servo_forward_mps": servo,
        "controller_variant": variant,
        "action_enabled": True,
        "leg_longitudinal_damping_active": common,
        "body_common_p_active": common,
    }


def test_body_overspeed_with_wheel_underspeed_changes_only_integral_direction():
    state = inputs()
    ramp = compute_residual17(state, np.zeros(16), np.zeros(4))
    baseline = compute_residual17({**state, "controller_variant": "baseline"},
                                  np.zeros(16), np.zeros(4))
    assert np.all(ramp["wheel_speed_error_rad_s"] > 0)
    assert np.all(ramp["wheel_integral_error_rad_s"] < 0)
    assert np.all(ramp["wheel_integral_after_nm"] < 0)
    assert np.all(baseline["wheel_integral_after_nm"] > 0)
    np.testing.assert_array_equal(ramp["wheel_speed_target_rad_s"],
                                  baseline["wheel_speed_target_rad_s"])
    np.testing.assert_array_equal(ramp["common_wheel_delta_torque_nm"],
                                  baseline["common_wheel_delta_torque_nm"])


def test_new_integral_preserves_differential_error_and_old_baseline_arithmetic():
    state = inputs(wheel_omega=4.0, wheel_target=(4.0, 5.0, 6.0, 7.0))
    ramp = compute_residual17(state, np.zeros(16), np.zeros(4))
    expected = np.asarray((-1.5, -.5, .5, 1.5)) + (-.05/.087)
    np.testing.assert_allclose(ramp["wheel_integral_error_rad_s"], expected, atol=1e-12)
    np.testing.assert_allclose(
        ramp["wheel_integral_error_rad_s"] - np.mean(ramp["wheel_integral_error_rad_s"]),
        ramp["wheel_speed_error_rad_s"] - np.mean(ramp["wheel_speed_error_rad_s"]),
        atol=1e-12,
    )
    baseline_input = {**state, "controller_variant": "baseline"}
    baseline = compute_residual17(baseline_input, np.zeros(16), np.zeros(4))
    old_input = {key: value for key, value in baseline_input.items()
                 if key not in ("servo_forward_mps", "controller_variant")}
    old = compute_residual16(old_input, np.zeros(16), np.zeros(4))
    for key, value in old.items():
        if key in ("schema", "wheel_integral_antiwindup_scope"):
            continue
        actual = baseline[key]
        if isinstance(value, np.ndarray):
            if value.dtype.kind == "b":
                np.testing.assert_array_equal(actual, value)
            else:
                np.testing.assert_allclose(actual, value, atol=1e-12)
        else:
            assert actual == value


def test_stop_gate_reverts_integral_error_and_keeps_speed_protection():
    state = inputs(servo=0.0, body=0.0, wheel_omega=30.0,
                   wheel_target=(30.0, 30.0, 30.0, 30.0), common=False)
    state["action_enabled"] = False
    result = compute_residual17(state, np.ones(16), np.full(4, 4.0))
    assert result["wheel_integral_mode"] == "differential_wheel_feedback"
    np.testing.assert_array_equal(result["wheel_integral_error_rad_s"],
                                  result["wheel_speed_error_rad_s"])
    assert not np.any(result["applied_action"])
    assert np.all(result["speed_outward_mask"][list(WHEELS)])
    assert not np.any(result["safe_torque_nm"][list(WHEELS)])


def test_saturated_candidate_unwinds_with_new_integral_error_sign():
    state = inputs(servo=.45, body=1.0, wheel_omega=0.0,
                   wheel_target=(8.0, 8.0, 8.0, 8.0))
    before = np.full(4, -3.0)
    ramp = compute_residual17(state, np.zeros(16), before)
    old = compute_residual17({**state, "controller_variant": "baseline"},
                             np.zeros(16), before)
    assert np.all(ramp["wheel_candidate_request_nm"] > 12.0)
    assert np.all(ramp["wheel_integral_error_rad_s"] < 0.0)
    assert np.all(ramp["wheel_integral_commit_mask"])
    assert np.all(ramp["wheel_integral_after_nm"] < before)
    assert not np.any(old["wheel_integral_commit_mask"])
    np.testing.assert_array_equal(old["wheel_integral_after_nm"], before)


def fixture_record(*, servo_yaw=0.0, body_yaw_rate=0.0, actor_action=None):
    offsets = np.asarray(((.2, .15, -.4), (.2, -.15, -.4),
                          (-.2, .15, -.4), (-.2, -.15, -.4)))
    support_inputs = {
        "base_rpy_rad": np.zeros(3),
        "base_angular_velocity_world": np.asarray((0.0, 0.0, body_yaw_rate)),
        "base_rotation_world_from_body": np.eye(3),
        "foot_offset_world": offsets,
        "foot_jacobian_world": np.zeros((4, 3, 4)),
        "nominal_mass_kg": 40.0,
        "nominal_base_com_offset_body_m": np.zeros(3),
        "commanded_roll_rad": 0.0,
        "commanded_pitch_rad": 0.0,
    }
    support = nominal_support(support_inputs)
    state = inputs(variant="combined", servo=.45, body=0.0, wheel_omega=0.0)
    state["support_torque_nm"] = support["support_torque_nm"]
    unlimited = servo_yaw + 4.0*(servo_yaw-body_yaw_rate)
    effective = float(np.clip(unlimited, -1.2, 1.2))
    state["nominal_wheel_speed_rad_s"] = (.45-effective*offsets[:, 1])/.087
    action = np.zeros(16) if actor_action is None else np.asarray(actor_action, dtype=float)
    calc = compute_residual17(state, action, np.zeros(4))
    raw = {"forward_velocity_mps": .45, "lateral_velocity_mps": 0.0,
           "yaw_rate_rps": servo_yaw, "jump_requested": False}
    adapter = {
        "schema": "d1-course17-yawlimit-commonintegral-adapter-v1",
        "controller_variant": "combined",
        "test_actor_probe": False,
        "action_gate_enabled": True,
        "pre_body_yaw_rad": 0.0,
        "pre_body_yaw_rate_rps": body_yaw_rate,
        "servo_forward_mps": .45,
        "servo_yaw_rps": servo_yaw,
        "nominal_yaw_feedback_gain": 4.0,
        "nominal_yaw_limit_rps": 1.2,
        "nominal_unclipped_yaw_rps": unlimited,
        "nominal_unlimited_yaw_rps": unlimited,
        "nominal_foot_lateral_m": offsets[:, 1],
        "nominal_effective_yaw_rps": effective,
        "stop_latched_after": False,
        "nominal_support": support,
        "calculation": calc,
    }
    info = {"controller_record": adapter,
            "consumed_command": {"forward_velocity_mps": .45},
            "raw_operator_command": raw,
            "policy_input_action": action,
            "policy_clipped_action": action,
            "applied_action": action}
    qpos = np.zeros(23)
    qpos[3] = 1.0
    qvel = np.zeros(22)
    qvel[5] = body_yaw_rate
    return {"info": info, "policy_input_action": action}, qpos, qvel


@pytest.mark.parametrize("servo_yaw,body_yaw_rate,expected", (
    (.25, .1, .85),
    (.5, -.1, 1.2),
    (-.5, .1, -1.2),
))
def test_yaw_limit_and_nominal_wheel_kinematics_both_signs(
        servo_yaw, body_yaw_rate, expected):
    action = np.full(16, .2)
    record, qpos, qvel = fixture_record(
        servo_yaw=servo_yaw, body_yaw_rate=body_yaw_rate, actor_action=action)
    checked = check_controller(record, 0, pre_qpos=qpos, pre_qvel=qvel,
                               pre_body_forward_mps=0.0, servo_yaw_rps=servo_yaw)
    assert record["info"]["controller_record"]["nominal_effective_yaw_rps"] == pytest.approx(expected)
    np.testing.assert_allclose(checked["computed"]["applied_action"], action)
    np.testing.assert_allclose(checked["calculation"]["wheel_action_offset_rad_s"],
                               np.full(4, .8))
    # Wrong old yaw limit must fail the independent pre-state/servo binding.
    wrong = copy.deepcopy(record)
    wrong["info"]["controller_record"]["nominal_yaw_limit_rps"] = .6
    with pytest.raises(ValueError):
        check_controller(wrong, 0, pre_qpos=qpos, pre_qvel=qvel,
                         pre_body_forward_mps=0.0, servo_yaw_rps=servo_yaw)


@pytest.mark.parametrize("field", (
    "wheel_integral_error_rad_s", "consumed_servo_forward_mps",
    "wheel_integral_candidate_nm", "safe_torque_nm",
))
def test_independent_checker_rejects_tampered_new_arithmetic(field):
    record, qpos, qvel = fixture_record()
    check_controller(record, 0, pre_qpos=qpos, pre_qvel=qvel,
                     pre_body_forward_mps=0.0, servo_yaw_rps=0.0)
    tampered = copy.deepcopy(record)
    value = tampered["info"]["controller_record"]["calculation"][field]
    if isinstance(value, np.ndarray):
        edited = value.copy()
        edited.flat[0] += .1
        tampered["info"]["controller_record"]["calculation"][field] = edited
    else:
        tampered["info"]["controller_record"]["calculation"][field] = value + .1
    with pytest.raises(ValueError):
        check_controller(tampered, 0, pre_qpos=qpos, pre_qvel=qvel,
                         pre_body_forward_mps=0.0, servo_yaw_rps=0.0)
