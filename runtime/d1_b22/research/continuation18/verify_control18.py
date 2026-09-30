"""Independent, saved-data-only recomputation of the C18 control decision.

Do not import controller18 or residual18 here. The numerical path below uses
frozen constants and saved pre-state, servo, support, and action records.
"""

from __future__ import annotations

import math

import numpy as np
from verify_course_e_08_03 import (
    LEG_SCALE, LEGS, POSITION_HIGH, POSITION_LOW, TORQUE_LIMIT,
    VELOCITY_LIMIT, WHEELS, check_nominal_support, close, equal, require,
)


VARIANTS = ("baseline", "yaw", "ramp", "combined")
FILTERED_INTEGRAL = ("ramp", "combined")
YAW_LIMIT = {"baseline": .6, "yaw": 1.2, "ramp": .6, "combined": 1.2}
ADAPTER_SCHEMA = "d1-course18-yawlimit-filteredcommonintegral-adapter-v1"
CONTROL_SCHEMA = "d1-course18-residual16-filtered-common-integral-v1"
ACTION_SCHEMA = "d1-course-residual16-leg-angle-wheel-speed-v1"


def _pre_yaw(pre_qpos) -> float:
    q = np.asarray(pre_qpos, dtype=np.float64)
    require(q.ndim == 1 and len(q) >= 7 and np.isfinite(q).all(),
            "C18 pre qpos is invalid")
    w, x, y, z = q[3:7]
    require(abs(float(w*w + x*x + y*y + z*z) - 1.0) <= 1e-6,
            "C18 pre quaternion is invalid")
    return math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))


def recompute_controller18(calc: dict, action) -> dict:
    """Independently derive every arithmetic stage from saved consumed inputs."""
    raw_action = np.asarray(action, dtype=np.float64)
    require(raw_action.shape == (16,) and np.isfinite(raw_action).all(),
            "C18 physical action must be finite 16D")
    clipped_action = np.clip(raw_action, -1.0, 1.0)
    applied_action = clipped_action if calc["action_enabled"] else np.zeros(16)
    q = np.asarray(calc["consumed_joint_position_rad"], dtype=np.float64)
    dq = np.asarray(calc["consumed_joint_velocity_rad_s"], dtype=np.float64)
    nominal = np.asarray(calc["consumed_nominal_joint_target_rad"], dtype=np.float64)
    nominal_wheel = np.asarray(calc["consumed_nominal_wheel_speed_rad_s"], dtype=np.float64)
    support = np.asarray(calc["consumed_support_torque_nm"], dtype=np.float64)
    integral_before = np.asarray(calc["wheel_integral_before_nm"], dtype=np.float64)
    rotation = np.asarray(calc["consumed_base_rotation_world_from_body"], dtype=np.float64)
    jacobian = np.asarray(calc["consumed_foot_jacobian_world"], dtype=np.float64)
    require(q.shape == dq.shape == nominal.shape == support.shape == (16,)
            and nominal_wheel.shape == integral_before.shape == (4,)
            and rotation.shape == (3, 3) and jacobian.shape == (4, 3, 4)
            and all(np.isfinite(a).all() for a in
                    (q, dq, nominal, nominal_wheel, support, integral_before,
                     rotation, jacobian)),
            "C18 consumed arrays have invalid shape or value")
    body = float(calc["consumed_body_com_forward_mps"])
    servo = float(calc["consumed_servo_forward_mps"])
    require(math.isfinite(body) and math.isfinite(servo),
            "C18 consumed body or servo forward speed is nonfinite")
    variant = calc["controller_variant"]
    require(type(variant) is str and variant in VARIANTS, "C18 variant is invalid")
    leg_offset = LEG_SCALE * applied_action[:12]
    wheel_offset = 4.0 * applied_action[12:]
    geometric = nominal.copy()
    geometric[list(LEGS)] += leg_offset
    rate = np.clip(geometric, q - VELOCITY_LIMIT*.01, q + VELOCITY_LIMIT*.01)
    position = np.clip(rate, POSITION_LOW, POSITION_HIGH)
    target = position.copy()
    target[list(WHEELS)] = q[list(WHEELS)]
    leg_pd = 80.0*(target-q) - 3.0*dq
    leg_pd[list(WHEELS)] = 0.0
    wheel_request = nominal_wheel + wheel_offset
    wheel_target = np.clip(wheel_request, -30.0, 30.0)
    wheel_omega = dq[list(WHEELS)]
    wheel_error = wheel_target - wheel_omega
    z_before = float(calc["wheel_common_reference_z_before_rad_s"])
    require(math.isfinite(z_before), "C18 common reference state is nonfinite")
    body_error = (servo - body)/.087
    reference_difference = float(np.mean(wheel_error) - body_error)
    active = variant in FILTERED_INTEGRAL and calc["body_common_p_active"]
    z_after = z_before + (reference_difference - z_before)/21.0 if active else 0.0
    integral_error = wheel_error - z_after if active else wheel_error.copy()
    candidate = np.clip(integral_before + 3.0*.01*integral_error, -4.0, 4.0)
    candidate_request = 2.2*wheel_error + candidate
    commit = ((np.abs(candidate_request) <= TORQUE_LIMIT[list(WHEELS)])
              | ((candidate_request*integral_error) < 0))
    integral_after = np.where(commit, candidate, integral_before)
    wheel_pi = np.zeros(16)
    wheel_pi[list(WHEELS)] = 2.2*wheel_error + integral_after
    base = leg_pd + support + wheel_pi
    projected = np.zeros((4, 3))
    relative = np.zeros(4)
    damping = np.zeros(16)
    for leg in range(4):
        jx = rotation[:, 0] @ jacobian[leg, :, :3]
        projected[leg] = jx
        relative[leg] = jx @ dq[4*leg:4*leg+3]
        if calc["leg_longitudinal_damping_active"]:
            damping[4*leg:4*leg+3] = -126.4374005337902*jx*relative[leg]
    common_error = float(np.mean(wheel_omega) - body/.087)
    common_scalar = 2.2*common_error if calc["body_common_p_active"] else 0.0
    common = np.zeros(16)
    common[list(WHEELS)] = common_scalar
    final = base + damping + common
    safe = np.clip(final, -TORQUE_LIMIT, TORQUE_LIMIT)
    torque_clip = safe != final
    upper = (q >= POSITION_HIGH) & (safe > 0)
    lower = (q <= POSITION_LOW) & (safe < 0)
    speed = (np.abs(dq) >= VELOCITY_LIMIT) & (safe*dq > 0)
    safe[upper | lower | speed] = 0.0
    return {
        "raw_action": raw_action,
        "clipped_action": clipped_action,
        "action_clip_mask": clipped_action != raw_action,
        "applied_action": applied_action,
        "leg_action_offset_rad": leg_offset,
        "wheel_action_offset_rad_s": wheel_offset,
        "geometric_joint_target_rad": geometric,
        "rate_limited_joint_target_rad": rate,
        "rate_limit_mask": rate != geometric,
        "position_limited_joint_target_rad": position,
        "position_clip_mask": position != rate,
        "joint_target_rad": target,
        "wheel_target_position_rad": q[list(WHEELS)],
        "leg_pd_nm": leg_pd,
        "wheel_speed_request_rad_s": wheel_request,
        "wheel_speed_target_rad_s": wheel_target,
        "wheel_speed_target_clip_mask": wheel_target != wheel_request,
        "wheel_speed_error_rad_s": wheel_error,
        "wheel_common_reference_difference_rad_s": reference_difference,
        "wheel_common_reference_z_after_rad_s": z_after,
        "wheel_integral_error_rad_s": integral_error,
        "wheel_integral_candidate_nm": candidate,
        "wheel_candidate_request_nm": candidate_request,
        "wheel_integral_commit_mask": commit,
        "wheel_integral_after_nm": integral_after,
        "base_wheel_torque_nm": wheel_pi,
        "base_request_torque_nm": base,
        "projected_jx": projected,
        "relative_forward_mps": relative,
        "leg_damping_delta_torque_nm": damping,
        "leg_damping_raw_joint_power_w": np.sum(damping*dq),
        "common_wheel_error_rad_s": common_error,
        "common_wheel_scalar_delta_nm": common_scalar,
        "common_wheel_delta_torque_nm": common,
        "final_request_torque_nm": final,
        "torque_clip_mask": torque_clip,
        "upper_outward_mask": upper,
        "lower_outward_mask": lower,
        "speed_outward_mask": speed,
        "protection_changed_mask": safe != final,
        "safe_torque_nm": safe,
    }


def check_controller(record: dict, index: int, *, pre_qpos, pre_qvel,
                     pre_body_forward_mps: float, servo_yaw_rps: float) -> dict:
    """Bind policy, current pre-state, nominal yaw, support and C18 arithmetic."""
    info = record["info"]
    adapter = info["controller_record"]
    calc = adapter["calculation"]
    variant = adapter["controller_variant"]
    require(variant in VARIANTS and calc["controller_variant"] == variant
            and adapter["schema"] == ADAPTER_SCHEMA
            and calc["schema"] == CONTROL_SCHEMA
            and calc["action_schema"] == ACTION_SCHEMA,
            f"C18 {index} controller schema or variant differs")
    qvel = np.asarray(pre_qvel, dtype=np.float64)
    require(qvel.ndim == 1 and len(qvel) >= 6 and np.isfinite(qvel).all(),
            f"C18 {index} pre qvel invalid")
    yaw = _pre_yaw(pre_qpos)
    body_yaw_rate = float(qvel[5])
    servo_forward = float(info["consumed_command"]["forward_velocity_mps"])
    require(close(adapter["pre_body_yaw_rad"], yaw, atol=1e-8)
            and close(adapter["pre_body_yaw_rate_rps"], body_yaw_rate, atol=1e-8)
            and close(adapter["servo_yaw_rps"], servo_yaw_rps)
            and close(adapter["servo_forward_mps"], servo_forward)
            and close(calc["consumed_servo_forward_mps"], servo_forward)
            and close(calc["consumed_body_com_forward_mps"], pre_body_forward_mps, atol=1e-6)
            and close(adapter["nominal_yaw_feedback_gain"], 4.0)
            and close(adapter["nominal_yaw_limit_rps"], YAW_LIMIT[variant]),
            f"C18 {index} pre-state, servo or nominal parameters differ")
    unclipped_yaw = servo_yaw_rps + 4.0*(servo_yaw_rps-body_yaw_rate)
    effective_yaw = float(np.clip(
        unclipped_yaw, -YAW_LIMIT[variant], YAW_LIMIT[variant],
    ))
    offsets = np.asarray(adapter["nominal_support"]["consumed_foot_offset_world"],
                         dtype=np.float64)
    require(offsets.shape == (4, 3) and np.isfinite(offsets).all(),
            f"C18 {index} foot offsets invalid")
    lateral = offsets @ np.asarray((-math.sin(yaw), math.cos(yaw), 0.0))
    nominal_wheel = np.clip((servo_forward-effective_yaw*lateral)/.087, -30.0, 30.0)
    require(close(adapter["nominal_unclipped_yaw_rps"], unclipped_yaw)
            and close(adapter["nominal_unlimited_yaw_rps"], unclipped_yaw)
            and close(adapter["nominal_foot_lateral_m"], lateral)
            and close(adapter["nominal_effective_yaw_rps"], effective_yaw)
            and close(calc["consumed_nominal_wheel_speed_rad_s"], nominal_wheel),
            f"C18 {index} nominal yaw/wheel kinematics differ")
    actor = np.asarray(record["policy_input_action"], dtype=np.float64)
    require(actor.shape == (16,) and np.isfinite(actor).all()
            and equal(actor, info["policy_input_action"]),
            f"C18 {index} saved actor differs from environment input")
    clipped = np.clip(actor, -1.0, 1.0)
    raw = info["raw_operator_command"]
    enabled = bool(raw["forward_velocity_mps"] != 0.0
                   or raw["lateral_velocity_mps"] != 0.0
                   or raw["yaw_rate_rps"] != 0.0
                   or raw["jump_requested"]
                   or adapter["test_actor_probe"])
    physical = clipped if enabled else np.zeros(16)
    require(equal(info["policy_clipped_action"], clipped)
            and adapter["action_gate_enabled"] is enabled
            and calc["action_enabled"] is enabled
            and equal(calc["raw_action"], physical)
            and equal(info["applied_action"], physical),
            f"C18 {index} policy clip or raw-command permission chain differs")
    support = check_nominal_support(adapter["nominal_support"])
    computed = recompute_controller18(calc, physical)
    for key, expected in computed.items():
        require(close(calc[key], expected),
                f"C18 {index} controller arithmetic differs: {key}")
    require(equal(info["applied_action"], computed["applied_action"])
            and close(adapter["wheel_common_reference_z_before_rad_s"],
                      calc["wheel_common_reference_z_before_rad_s"])
            and close(adapter["wheel_common_reference_z_after_rad_s"],
                      computed["wheel_common_reference_z_after_rad_s"])
            and close(support, calc["consumed_support_torque_nm"])
            and close(calc["control_dt_s"], .01)
            and close(calc["wheel_kp_nm_per_rad_s"], 2.2)
            and close(calc["wheel_ki_nm_per_rad"], 3.0)
            and close(calc["wheel_integral_limit_nm"], 4.0)
            and close(calc["wheel_radius_m"], .087)
            and close(calc["wheel_common_reference_alpha"], 1.0/21.0)
            and close(calc["wheel_common_reference_tau_s"], .20)
            and calc["wheel_integral_mode"] == (
                "filtered_wheel_body_reference" if variant in FILTERED_INTEGRAL and servo_forward != 0.0
                else "differential_wheel_feedback")
            and calc["body_common_p_active"] == (servo_forward != 0.0)
            and calc["leg_longitudinal_damping_active"] == (
                servo_forward != 0.0 or adapter["stop_latched_after"]),
            f"C18 {index} consumed command/support/parameter chain differs")
    return {"computed": computed, "calculation": calc}


__all__ = ("check_controller", "recompute_controller18")
