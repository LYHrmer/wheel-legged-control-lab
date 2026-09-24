"""Pure validation of the actual authority -> drive -> body-P -> stop chain."""

from __future__ import annotations

from typing import Any

import numpy as np
from body_speed_math_06 import (
    WHEEL_KP,
    WHEEL_RADIUS_M,
    WHEELS,
    body_common_increment,
    finite_array,
)
from drive_damping_math_04 import GAIN_NSPM, drive_increment, protect_requested
from drive_damping_validator_04 import validate_precontrol_archive

ATOL = 1e-10
RTOL = 1e-12
DRIVE_SCHEMA = "d1-drive-jx-leg-damping-record-v1"
BODY_SCHEMA = "d1-drive-body-common-p-record-v1"


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    return finite_array(value, shape, label)


def _exact(got: Any, expected: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    left = _array(got, shape, label)
    right = _array(expected, shape, label + " expected")
    if not np.array_equal(left, right):
        raise ValueError(f"{label} differs from actual source")
    return left


def _close(got: Any, expected: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    left = _array(got, shape, label)
    right = _array(expected, shape, label + " expected")
    if not np.allclose(left, right, atol=ATOL, rtol=RTOL):
        raise ValueError(f"{label} differs from exact arithmetic")
    return left


def _scalar(actual: Any, expected: float, label: str) -> float:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise TypeError(f"{label} must be a real scalar")
    got = float(actual)
    if not np.isfinite(got) or not np.isclose(got, expected, atol=ATOL, rtol=RTOL):
        raise ValueError(f"{label} differs from exact arithmetic")
    return got


def validate_body_trace(
    trace_rows: list[dict[str, Any]], *, expected_raw_schedule: list[float] | None = None,
    pre_control_states: list[dict[str, Any]] | None = None,
    native_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(trace_rows, list) or not trace_rows:
        raise ValueError("body-P trace must contain completed controls")
    if expected_raw_schedule is not None and len(expected_raw_schedule) != len(trace_rows):
        raise ValueError("raw schedule length differs")
    if pre_control_states is not None and len(pre_control_states) != len(trace_rows):
        raise ValueError("pre-control archive length differs")
    if native_rows is not None and len(native_rows) != 5 * len(trace_rows):
        raise ValueError("native return count differs from five per completed control")
    drive_active = body_active = stop_active = 0
    previous_pi_after: np.ndarray | None = None
    max_raw_body_delta = max_actual_body_delta = 0.0
    for tick, row in enumerate(trace_rows):
        if type(row.get("tick")) is not int or row["tick"] != tick:
            raise ValueError(f"trace tick differs: {tick}")
        raw = row["raw_command"]["forward_velocity_mps"]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not np.isfinite(raw):
            raise TypeError(f"raw forward command malformed: {tick}")
        forward = float(raw)
        if expected_raw_schedule is not None and forward != expected_raw_schedule[tick]:
            raise ValueError(f"raw forward schedule differs: {tick}")
        result = row["controller_result"]
        drive = result["drive_damping"]
        body = result["body_common_p"]
        authority = row["authority_record"]
        stop = row["stop_record"]
        if drive["schema"] != DRIVE_SCHEMA or body["schema"] != BODY_SCHEMA:
            raise ValueError(f"intermediate stage schema differs: {tick}")
        active = forward != 0.0
        if (type(drive["active"]) is not bool or drive["active"] != active
                or type(body["active"]) is not bool or body["active"] != active):
            raise ValueError(f"raw-forward stage gates differ: {tick}")
        if (drive["raw_forward_mps"] != forward
                or drive["servo_forward_mps"] != forward
                or body["raw_forward_mps"] != forward
                or body["servo_forward_mps"] != forward
                or drive["damping_nspm"] != GAIN_NSPM
                or body["wheel_kp"] != WHEEL_KP
                or body["wheel_radius_m"] != WHEEL_RADIUS_M):
            raise ValueError(f"raw binding or original constants differ: {tick}")
        for record in (drive, body):
            if (type(record["provider_state_sequence"]) is not int
                    or record["provider_state_sequence"] != tick
                    or record["provider_state_age_s"] != 0.0
                    or not np.isfinite(record["control_time_s"])
                    or abs(record["control_time_s"] - tick * .01) > 1e-10):
                raise ValueError(f"synchronized provider timing differs: {tick}")
        rotation = _array(drive["base_rotation"], (3, 3), "drive rotation")
        jacobian = _array(drive["foot_jacobian"], (4, 3, 4), "drive jacobian")
        position = _exact(body["joint_position"], drive["joint_position"],
                          (16,), "body/drive joint position")
        velocity = _exact(body["joint_velocity"], drive["joint_velocity"],
                          (16,), "body/drive joint velocity")
        _exact(body["base_rotation"], rotation, (3, 3), "body/drive rotation")
        dformula = drive_increment(rotation, jacobian, velocity,
                                   raw_forward_mps=forward)
        _close(drive["projected_jx"], dformula["projected_jx"], (4, 3), "drive Jx")
        _close(drive["relative_forward_mps"], dformula["relative_forward_mps"],
               (4,), "drive relative speed")
        ddelta = _close(drive["delta_torque_nm"], dformula["delta_torque_nm"],
                        (16,), "drive delta")
        if np.any(ddelta[list(WHEELS)] != 0.0):
            raise ValueError(f"drive delta changes a wheel: {tick}")
        _scalar(drive["raw_joint_power_w"], dformula["raw_joint_power_w"], "drive power")
        original_req = _array(drive["base_requested_torque_nm"], (16,), "original request")
        original_safe, _ = protect_requested(original_req, position, velocity)
        _exact(drive["base_protected_torque_nm"], original_safe,
               (16,), "original protection")
        _exact(authority["wheel_request_nm"], original_req[list(WHEELS)],
               (4,), "authority/original wheel request")
        _exact(authority["wheel_torque_nm"], original_safe[list(WHEELS)],
               (4,), "authority/original wheel protected")
        drive_req = _array(drive["total_requested_torque_nm"], (16,), "drive request")
        drive_safe = _array(drive["total_protected_torque_nm"], (16,), "drive protected")
        if active:
            _close(drive_req, original_req + ddelta, (16,), "drive request sum")
            expected_drive_safe, _ = protect_requested(drive_req, position, velocity)
            _exact(drive_safe, expected_drive_safe, (16,), "drive protection")
        else:
            _exact(drive_req, original_req, (16,), "inactive drive request")
            _exact(drive_safe, original_safe, (16,), "inactive drive protection")
        _close(drive["actual_protected_increment_nm"], drive_safe - original_safe,
               (16,), "actual drive protected increment")
        _exact(body["parent_requested_torque_nm"], drive_req,
               (16,), "drive-to-body request")
        _exact(body["parent_protected_torque_nm"], drive_safe,
               (16,), "drive-to-body protection")
        parent_wheel = _array(body["parent_wheel_nm"], (16,), "parent wheel feedback")
        if (np.any(parent_wheel[[i for i in range(16) if i not in WHEELS]] != 0.0)
                or not np.array_equal(parent_wheel[list(WHEELS)], drive_req[list(WHEELS)])):
            raise ValueError(f"parent wheel term differs from original PI: {tick}")
        body_velocity = _array(body["base_linear_velocity_body"], (3,), "body COM velocity")
        omega = _exact(body["wheel_omega_rad_s"], velocity[list(WHEELS)],
                       (4,), "measured wheel omega")
        bformula = body_common_increment(omega, body_velocity, raw_forward_mps=forward)
        for key in ("mean_omega_rad_s", "body_equivalent_omega_rad_s",
                    "common_error_rad_s", "scalar_delta_nm"):
            _scalar(body[key], bformula[key], key)
        bdelta = _close(body["delta_torque_nm"], bformula["delta_torque_nm"],
                        (16,), "body common delta")
        if np.any(bdelta[[i for i in range(16) if i not in WHEELS]] != 0.0):
            raise ValueError(f"body common delta changes a leg: {tick}")
        body_req = _array(body["total_requested_torque_nm"], (16,), "body request")
        body_safe = _array(body["total_protected_torque_nm"], (16,), "body protected")
        body_wheel = _array(body["total_wheel_nm"], (16,), "body wheel feedback")
        if active:
            _close(body_req, drive_req + bdelta, (16,), "body request sum")
            _close(body_wheel, parent_wheel + bdelta, (16,), "body wheel sum")
            expected_body_safe, expected_limited = protect_requested(
                body_req, position, velocity
            )
            _exact(body_safe, expected_body_safe, (16,), "body protection")
            if not np.array_equal(np.asarray(body["total_torque_limited"]), expected_limited):
                raise ValueError(f"body rated/outward limit flags differ: {tick}")
        else:
            _exact(body_req, drive_req, (16,), "inactive body request")
            _exact(body_safe, drive_safe, (16,), "inactive body protected")
            _exact(body_wheel, parent_wheel, (16,), "inactive body wheel")
            if not np.array_equal(np.asarray(body["total_torque_limited"]),
                                  np.asarray(drive_safe != drive_req)):
                raise ValueError(f"inactive body protection flags differ: {tick}")
        _close(body["actual_protected_increment_nm"], body_safe - drive_safe,
               (16,), "actual body protected increment")
        _exact(body["original_pi_integral_before_nm"],
               authority["wheel_integral_before_nm"], (4,), "original PI before")
        integral = _exact(body["original_pi_integral_after_nm"],
                          authority["wheel_integral_after_nm"], (4,), "original PI after")
        pi_before = _exact(body["original_pi_integral_before_nm"],
                           result["memory_before"]["wheel_integral_nm"],
                           (4,), "actual parent PI memory before")
        _exact(integral, result["memory_after"]["wheel_integral_nm"],
               (4,), "actual parent PI memory after")
        if tick == 0 and not np.array_equal(pi_before, np.zeros(4)):
            raise ValueError("original PI memory was not reset before first control")
        if previous_pi_after is not None:
            _exact(pi_before, previous_pi_after, (4,), "PI memory continuity")
        previous_pi_after = integral.copy()
        _exact(body["wheel_speed_target_rad_s"], authority["wheel_target_rad_s"],
               (4,), "unchanged final wheel target")
        _exact(body["wheel_speed_target_rad_s"], result["wheel_speed_target_rad_s"],
               (4,), "actual result wheel target")
        _exact(body["nominal_wheel_speed_rad_s"],
               result["nominal_wheel_speed_rad_s"], (4,), "unchanged nominal wheel target")
        # The actual, committed antiwindup integral is used; the candidate may
        # have been rejected by the original PI, and is not a valid substitute.
        wheel_error = _array(body["wheel_speed_target_rad_s"], (4,), "target") - omega
        candidate = np.clip(pi_before + 3.0 * wheel_error * .01, -4.0, 4.0)
        candidate_request = WHEEL_KP * wheel_error + candidate
        committed = np.where((np.abs(candidate_request) <= 12.0)
                             | (candidate_request * wheel_error < 0.0), candidate, pi_before)
        _exact(integral, committed, (4,), "original single PI antiwindup update")
        expected_parent_wheel = WHEEL_KP * wheel_error + integral
        _close(parent_wheel[list(WHEELS)], expected_parent_wheel, (4,),
               "original committed PI identity")
        _exact(stop["base_requested_torque_nm"], body_req, (16,), "body-to-stop request")
        _close(stop["base_leg_pd_nm"],
               _array(drive["base_leg_pd_nm"], (16,), "original leg PD") + ddelta,
               (16,), "body-to-stop leg PD")
        sdelta = _array(stop["delta_torque_nm"], (16,), "stop delta")
        if np.any(sdelta[list(WHEELS)] != 0.0):
            raise ValueError(f"stop stage changes wheel channels: {tick}")
        stop_req = _array(stop["total_requested_torque_nm"], (16,), "stop request")
        _close(stop_req, body_req + sdelta, (16,), "stop request sum")
        if type(stop["active"]) is not bool or stop["active"] != row["stop_active"]:
            raise ValueError(f"stop gate differs: {tick}")
        if stop["active"] and active:
            raise ValueError(f"drive and stop damping overlap: {tick}")
        if stop["active"]:
            expected_final, _ = protect_requested(stop_req, position, velocity)
            _exact(stop["safe_torque_nm"], expected_final, (16,), "stop protection")
        else:
            _exact(sdelta, np.zeros(16), (16,), "inactive stop delta")
            _exact(stop_req, body_req, (16,), "inactive stop request")
            _exact(stop["safe_torque_nm"], body_safe, (16,), "inactive stop identity")
        final = _exact(result["torque_nm"], row["torque_nm"], (16,), "executed torque")
        _exact(stop["safe_torque_nm"], final, (16,), "stop/executed torque")
        _exact(result["requested_torque_nm"], stop_req, (16,), "final request")
        _exact(result["wheel_nm"], body_wheel, (16,), "final wheel term")
        _exact(result["leg_pd_nm"],
               _array(drive["base_leg_pd_nm"], (16,), "original leg PD") + ddelta + sdelta,
               (16,), "final leg term")
        _exact(result["clipped_action"], row["applied_action"], (8,), "applied action")
        _exact(row["action"], row["applied_action"], (8,), "physical action")
        if native_rows is not None:
            for substep in range(5):
                index = 5 * tick + substep
                native = native_rows[index]
                if (native["index"] != index or native["returned"] is not True
                        or native["error"] is not None):
                    raise ValueError(f"native entry/return differs: {tick}/{substep}")
                _exact(native["ctrl_nm"], final, (16,), "actual native control torque")
        if pre_control_states is not None:
            before = pre_control_states[tick]
            if type(before.get("tick")) is not int or before["tick"] != tick:
                raise ValueError(f"pre-control tick differs: {tick}")
            for key, source, shape in (
                ("base_rotation", "provider_base_rotation", (3, 3)),
                ("foot_jacobian", "provider_foot_jacobian", (4, 3, 4)),
                ("joint_position", "provider_joint_position", (16,)),
                ("joint_velocity", "provider_joint_velocity", (16,)),
            ):
                _exact(drive[key], before[source], shape, f"actual provider {key}")
            _exact(body_velocity, before["provider_base_linear_velocity_body"],
                   (3,), "actual provider base COM velocity")
            if (body["provider_state_sequence"] != before["provider_state_sequence"]
                    or body["provider_state_age_s"] != before["provider_state_age_s"]
                    or body["control_time_s"] != before["provider_control_time_s"]):
                raise ValueError(f"actual provider timing differs: {tick}")
        drive_active += int(drive["active"])
        body_active += int(body["active"])
        stop_active += int(stop["active"])
        max_raw_body_delta = max(max_raw_body_delta, abs(float(body["scalar_delta_nm"])))
        max_actual_body_delta = max(max_actual_body_delta,
                                    float(np.max(np.abs(body_safe - drive_safe))))
    return {
        "schema": "d1-drive-body-common-p-stage-chain-validation-v1",
        "record_valid": True,
        "validated_controls": len(trace_rows),
        "drive_active_controls": drive_active,
        "body_common_p_active_controls": body_active,
        "stop_active_controls": stop_active,
        "max_abs_raw_common_delta_nm": max_raw_body_delta,
        "max_abs_actual_protected_common_delta_nm": max_actual_body_delta,
        "native_ctrl_returns_linked": None if native_rows is None else len(native_rows),
        "algebraic_tolerance": {"atol": ATOL, "rtol": RTOL},
        "no_engine_or_model_calls": True,
    }


def validate_body_precontrol_archive(
    pre_control_states: list[dict[str, Any]], *, saved_qpos: Any, saved_qvel: Any,
    joint_qpos_addresses: Any, joint_dof_addresses: Any,
    base_body_ipos_local_m: Any, endpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind synchronized provider COM velocity to saved raw state and endpoints."""
    base = validate_precontrol_archive(
        pre_control_states, saved_qpos=saved_qpos, saved_qvel=saved_qvel,
        joint_qpos_addresses=joint_qpos_addresses,
        joint_dof_addresses=joint_dof_addresses,
    )
    qvel = np.asarray(saved_qvel)
    ipos = _array(base_body_ipos_local_m, (3,), "compiled base inertial offset")
    if len(endpoints) != len(pre_control_states) + 1 or qvel.shape[1] < 6:
        raise ValueError("endpoint or saved freejoint velocity length differs")
    for tick, before in enumerate(pre_control_states):
        rotation = _array(before["provider_base_rotation"], (3, 3), "provider rotation")
        raw = _array(before["raw_precontrol_qvel"], (qvel.shape[1],), "raw qvel")
        expected = rotation.T @ raw[:3] + np.cross(raw[3:6], ipos)
        _close(before["provider_base_linear_velocity_body"], expected,
               (3,), "provider/raw COM velocity")
        endpoint = endpoints[tick]
        if endpoint["tick"] != tick:
            raise ValueError(f"endpoint/precontrol tick differs: {tick}")
        _scalar(endpoint["body_vx_mps"], float(expected[0]), "endpoint/provider body vx")
    return {**base, "linked_body_com_velocity_snapshots": len(pre_control_states),
            "compiled_base_body_ipos_local_m": ipos.tolist(),
            "full_body_com_velocity_linked": True,
            "whole_system_com_substituted": False}
