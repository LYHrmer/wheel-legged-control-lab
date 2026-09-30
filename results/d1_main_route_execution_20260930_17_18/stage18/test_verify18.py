"""Pure C18 saved-controller checker fixture; run in its own pytest process."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(HERE))

from residual18 import compute_residual18, nominal_support
from verify_control18 import check_controller


def fixture_record():
    offsets = np.asarray(((.2, .15, -.4), (.2, -.15, -.4),
                          (-.2, .15, -.4), (-.2, -.15, -.4)))
    support = nominal_support({
        "base_rpy_rad": np.zeros(3),
        "base_angular_velocity_world": np.zeros(3),
        "base_rotation_world_from_body": np.eye(3),
        "foot_offset_world": offsets,
        "foot_jacobian_world": np.zeros((4, 3, 4)),
        "nominal_mass_kg": 40.0,
        "nominal_base_com_offset_body_m": np.zeros(3),
        "commanded_roll_rad": 0.0,
        "commanded_pitch_rad": 0.0,
    })
    q = np.tile((0.0, .8, -1.5, 0.0), 4)
    dq = np.zeros(16)
    action = np.zeros(16)
    servo, body = .45, .1
    calc = compute_residual18({
        "joint_position_rad": q,
        "joint_velocity_rad_s": dq,
        "nominal_joint_target_rad": q.copy(),
        "nominal_wheel_speed_rad_s": np.full(4, servo/.087),
        "support_torque_nm": support["support_torque_nm"],
        "base_rotation_world_from_body": np.eye(3),
        "foot_jacobian_world": np.zeros((4, 3, 4)),
        "body_com_forward_mps": body,
        "servo_forward_mps": servo,
        "controller_variant": "combined",
        "action_enabled": True,
        "leg_longitudinal_damping_active": True,
        "body_common_p_active": True,
    }, action, np.zeros(4), 0.0)
    adapter = {
        "schema": "d1-course18-yawlimit-filteredcommonintegral-adapter-v1",
        "controller_variant": "combined",
        "test_actor_probe": False,
        "action_gate_enabled": True,
        "pre_body_yaw_rad": 0.0,
        "pre_body_yaw_rate_rps": 0.0,
        "servo_forward_mps": servo,
        "servo_yaw_rps": 0.0,
        "nominal_yaw_feedback_gain": 4.0,
        "nominal_yaw_limit_rps": 1.2,
        "nominal_unclipped_yaw_rps": 0.0,
        "nominal_unlimited_yaw_rps": 0.0,
        "nominal_foot_lateral_m": offsets[:, 1],
        "nominal_effective_yaw_rps": 0.0,
        "wheel_common_reference_z_before_rad_s": 0.0,
        "wheel_common_reference_z_after_rad_s": calc["wheel_common_reference_z_after_rad_s"],
        "stop_latched_after": False,
        "nominal_support": support,
        "calculation": calc,
    }
    raw = {"forward_velocity_mps": servo, "lateral_velocity_mps": 0.0,
           "yaw_rate_rps": 0.0, "jump_requested": False}
    info = {"controller_record": adapter,
            "consumed_command": {"forward_velocity_mps": servo},
            "raw_operator_command": raw,
            "policy_input_action": action,
            "policy_clipped_action": action,
            "applied_action": action}
    qpos = np.zeros(23)
    qpos[3] = 1.0
    qvel = np.zeros(22)
    qvel[0] = body
    return {"info": info, "policy_input_action": action}, qpos, qvel, body


def test_independent_c18_checker_accepts_complete_saved_record_and_rejects_tamper():
    record, qpos, qvel, body = fixture_record()
    checked = check_controller(record, 0, pre_qpos=qpos, pre_qvel=qvel,
                               pre_body_forward_mps=body, servo_yaw_rps=0.0)
    assert checked["computed"]["wheel_common_reference_z_after_rad_s"] > 0.0
    for field in ("wheel_common_reference_z_after_rad_s", "wheel_common_reference_alpha"):
        wrong = copy.deepcopy(record)
        calc = wrong["info"]["controller_record"]["calculation"]
        calc[field] += .1
        if field == "wheel_common_reference_z_after_rad_s":
            wrong["info"]["controller_record"][field] += .1
        with pytest.raises(ValueError):
            check_controller(wrong, 0, pre_qpos=qpos, pre_qvel=qvel,
                             pre_body_forward_mps=body, servo_yaw_rps=0.0)
