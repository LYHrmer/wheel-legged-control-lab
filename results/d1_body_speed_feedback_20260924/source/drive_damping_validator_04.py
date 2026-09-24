"""Pure validation of the actual drive-stage and frozen stop-stage trace chain.

Only NumPy and saved JSON/NPZ values are used. No model, controller or engine is
imported. Numerical tolerance applies to algebraic recomputation, never to a
physical task gate or an exact source/action/torque identity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from drive_damping_math_04 import GAIN_NSPM, drive_increment, protect_requested

RECORD_SCHEMA = "d1-drive-jx-leg-damping-record-v1"
ATOL = 1e-10
RTOL = 1e-12
WHEELS = (3, 7, 11, 15)


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in ("f", "i", "u") or array.shape != shape:
        raise TypeError(f"{label} must have real numeric shape {shape}")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} is nonfinite")
    return result


def _close(actual: Any, expected: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    got = _array(actual, shape, label)
    want = _array(expected, shape, label + " expected")
    if not np.allclose(got, want, atol=ATOL, rtol=RTOL):
        raise ValueError(f"{label} differs from fixed arithmetic")
    return got


def _exact(actual: Any, expected: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    got = _array(actual, shape, label)
    want = _array(expected, shape, label + " expected")
    if not np.array_equal(got, want):
        raise ValueError(f"{label} differs from executed stage/torque")
    return got


def validate_drive_trace(
    trace_rows: list[dict[str, Any]], *, expected_raw_schedule: list[float] | None = None,
    pre_control_states: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reject malformed drive formula, protection, source or stop-stage links."""
    if not isinstance(trace_rows, list) or not trace_rows:
        raise ValueError("drive trace must contain at least one completed control")
    if expected_raw_schedule is not None and len(expected_raw_schedule) != len(trace_rows):
        raise ValueError("expected raw schedule length differs")
    if pre_control_states is not None and len(pre_control_states) != len(trace_rows):
        raise ValueError("pre-control provider/raw state count differs")
    active_count = stop_count = 0
    peak_abs_raw_delta = peak_abs_protected_delta = 0.0
    for tick, row in enumerate(trace_rows):
        if not isinstance(row, dict) or type(row.get("tick")) is not int or row["tick"] != tick:
            raise ValueError(f"drive trace tick differs: {tick}")
        raw = row["raw_command"]
        if not isinstance(raw, dict) or type(raw.get("forward_velocity_mps")) not in (int, float):
            raise TypeError(f"raw forward command missing or malformed at tick {tick}")
        forward = float(raw["forward_velocity_mps"])
        if not np.isfinite(forward):
            raise ValueError(f"raw forward command nonfinite at tick {tick}")
        if expected_raw_schedule is not None and forward != expected_raw_schedule[tick]:
            raise ValueError(f"raw forward schedule differs at tick {tick}")
        controller, stop = row["controller_result"], row["stop_record"]
        if not isinstance(controller, dict) or not isinstance(stop, dict):
            raise TypeError(f"controller/stop record missing at tick {tick}")
        record = controller.get("drive_damping")
        if not isinstance(record, dict) or record.get("schema") != RECORD_SCHEMA:
            raise ValueError(f"drive record absent or wrong schema at tick {tick}")
        if type(record.get("active")) is not bool or record["active"] != (forward != 0.0):
            raise ValueError(f"drive gate differs from raw forward at tick {tick}")
        if record["raw_forward_mps"] != forward or record["servo_forward_mps"] != forward:
            raise ValueError(f"drive raw/servo forward binding differs at tick {tick}")
        if record["damping_nspm"] != GAIN_NSPM:
            raise ValueError(f"drive gain differs at tick {tick}")
        if (type(record.get("provider_state_sequence")) is not int
                or record["provider_state_sequence"] != tick
                or not np.isfinite(float(record["provider_state_age_s"]))
                or record["provider_state_age_s"] != 0.0
                or not np.isfinite(float(record["control_time_s"]))
                or abs(float(record["control_time_s"]) - tick * .01) > 1e-10):
            raise ValueError(f"consumed provider state timing malformed at tick {tick}")
        rotation = _array(record["base_rotation"], (3, 3), "base_rotation")
        jacobian = _array(record["foot_jacobian"], (4, 3, 4), "foot_jacobian")
        position = _array(record["joint_position"], (16,), "joint_position")
        velocity = _array(record["joint_velocity"], (16,), "joint_velocity")
        formula = drive_increment(rotation, jacobian, velocity,
                                  raw_forward_mps=forward)
        _close(record["projected_jx"], formula["projected_jx"], (4, 3),
               "projected_jx")
        _close(record["relative_forward_mps"], formula["relative_forward_mps"],
               (4,), "relative_forward_mps")
        delta = _close(record["delta_torque_nm"], formula["delta_torque_nm"],
                       (16,), "delta_torque_nm")
        if not np.array_equal(delta[list(WHEELS)], np.zeros(4)):
            raise ValueError(f"wheel damping increment is nonzero at tick {tick}")
        if not record["active"] and not np.array_equal(delta, np.zeros(16)):
            raise ValueError(f"inactive drive stage is not identity at tick {tick}")
        power = float(record["raw_joint_power_w"])
        if (not np.isfinite(power)
                or not np.isclose(power, formula["raw_joint_power_w"], atol=ATOL, rtol=RTOL)
                or not np.isclose(power, -GAIN_NSPM * float(np.sum(
                    formula["relative_forward_mps"] ** 2
                )) if record["active"] else 0.0, atol=ATOL, rtol=RTOL)):
            raise ValueError(f"raw sampled power identity differs at tick {tick}")
        base_req = _array(record["base_requested_torque_nm"], (16,), "base_requested")
        base_safe = _array(record["base_protected_torque_nm"], (16,), "base_protected")
        base_leg = _array(record["base_leg_pd_nm"], (16,), "base_leg_pd")
        total_req = _array(record["total_requested_torque_nm"], (16,), "drive_requested")
        total_safe = _array(record["total_protected_torque_nm"], (16,), "drive_protected")
        expected_base_safe, _ = protect_requested(base_req, position, velocity)
        _exact(base_safe, expected_base_safe, (16,), "inherited base protection")
        if record["active"]:
            _close(total_req, base_req + delta, (16,), "drive requested sum")
            expected_safe, _ = protect_requested(total_req, position, velocity)
            _exact(total_safe, expected_safe, (16,), "drive protection order")
        else:
            _exact(total_req, base_req, (16,), "inactive drive requested")
            _exact(total_safe, base_safe, (16,), "inactive drive protected")
        _close(record["actual_protected_increment_nm"], total_safe - base_safe,
               (16,), "actual protected increment")
        _close(stop["base_requested_torque_nm"], total_req, (16,),
               "drive output to stop input request")
        _close(stop["base_leg_pd_nm"], base_leg + delta, (16,),
               "drive output to stop input leg feedback")
        stop_delta = _array(stop["delta_torque_nm"], (16,), "stop delta")
        if np.any(stop_delta[list(WHEELS)] != 0.0):
            raise ValueError(f"stop wheel damping increment is nonzero at tick {tick}")
        _close(stop["total_requested_torque_nm"], total_req + stop_delta,
               (16,), "stop requested sum")
        _close(controller["leg_pd_nm"], base_leg + delta + stop_delta,
               (16,), "final leg feedback sum")
        stop_active = stop.get("active")
        if type(stop_active) is not bool or stop_active != row["stop_active"]:
            raise ValueError(f"stop record gate differs at tick {tick}")
        if stop_active and record["active"]:
            raise ValueError(f"drive and stop damping overlap at tick {tick}")
        if not stop_active:
            _exact(stop["total_requested_torque_nm"], total_req, (16,),
                   "inactive stop requested")
            _exact(stop["safe_torque_nm"], total_safe, (16,),
                   "inactive stop protected")
        else:
            expected_stop_safe, _ = protect_requested(
                total_req + stop_delta, position, velocity
            )
            _exact(stop["safe_torque_nm"], expected_stop_safe, (16,),
                   "active stop protection order")
        final = _exact(row["torque_nm"], controller["torque_nm"], (16,),
                       "executed/controller torque")
        _exact(stop["safe_torque_nm"], final, (16,), "stop/executed torque")
        _exact(controller["requested_torque_nm"], stop["total_requested_torque_nm"],
               (16,), "final requested torque")
        _exact(controller["clipped_action"], row["applied_action"], (8,),
               "actual applied action")
        _exact(row["action"], row["applied_action"], (8,), "physical action")
        authority = row["authority_record"]
        if not isinstance(authority, dict):
            raise TypeError(f"authority record missing at tick {tick}")
        _exact(authority["wheel_torque_nm"], final[list(WHEELS)], (4,),
               "inherited wheel torque")
        if pre_control_states is not None:
            before = pre_control_states[tick]
            if type(before.get("tick")) is not int or before["tick"] != tick:
                raise ValueError(f"pre-control state tick differs: {tick}")
            for field, shape, key in (
                ("base_rotation", (3, 3), "provider_base_rotation"),
                ("foot_jacobian", (4, 3, 4), "provider_foot_jacobian"),
                ("joint_position", (16,), "provider_joint_position"),
                ("joint_velocity", (16,), "provider_joint_velocity"),
            ):
                _exact(record[field], before[key], shape, f"provider {field}")
            if (record["provider_state_sequence"] != before["provider_state_sequence"]
                    or record["provider_state_age_s"] != before["provider_state_age_s"]
                    or record["control_time_s"] != before["provider_control_time_s"]):
                raise ValueError(f"actual consumed provider state link differs: {tick}")
        active_count += int(record["active"])
        stop_count += int(stop_active)
        peak_abs_raw_delta = max(peak_abs_raw_delta, float(np.max(np.abs(delta))))
        protected_delta = _array(record["actual_protected_increment_nm"], (16,),
                                 "actual_protected_increment_nm")
        peak_abs_protected_delta = max(peak_abs_protected_delta,
                                       float(np.max(np.abs(protected_delta))))
    return {
        "schema": "d1-drive-jx-stage-chain-validation-v1",
        "record_valid": True,
        "validated_controls": len(trace_rows),
        "drive_active_controls": active_count,
        "stop_active_controls": stop_count,
        "max_abs_raw_drive_increment_nm": peak_abs_raw_delta,
        "max_abs_actual_protected_increment_nm": peak_abs_protected_delta,
        "algebraic_tolerance": {"atol": ATOL, "rtol": RTOL},
        "no_engine_or_model_calls": True,
    }


def validate_precontrol_archive(
    pre_control_states: list[dict[str, Any]], *, saved_qpos: Any, saved_qvel: Any,
    joint_qpos_addresses: Any, joint_dof_addresses: Any,
) -> dict[str, Any]:
    """Link pre-control raw snapshots to NPZ using the compiled joint address map.

    Provider estimates and raw qpos/qvel have distinct semantics. This check
    never silently equates filtered provider values with simulator truth.
    """
    qpos = np.asarray(saved_qpos)
    qvel = np.asarray(saved_qvel)
    qadr = np.asarray(joint_qpos_addresses)
    vadr = np.asarray(joint_dof_addresses)
    if (qpos.ndim != 2 or qvel.ndim != 2 or qpos.shape[0] != len(pre_control_states) + 1
            or qvel.shape[0] != len(pre_control_states) + 1
            or qadr.shape != (16,) or vadr.shape != (16,)
            or qadr.dtype.kind not in ("i", "u") or vadr.dtype.kind not in ("i", "u")
            or len(set(qadr.tolist())) != 16 or len(set(vadr.tolist())) != 16
            or np.any(qadr < 0) or np.any(vadr < 0)
            or np.any(qadr >= qpos.shape[1]) or np.any(vadr >= qvel.shape[1])):
        raise ValueError("saved state or actual compiled joint address map is malformed")
    for tick, before in enumerate(pre_control_states):
        if type(before.get("tick")) is not int or before["tick"] != tick:
            raise ValueError(f"pre-control archive tick differs: {tick}")
        raw_qpos = _array(before["raw_precontrol_qpos"], (qpos.shape[1],), "raw qpos")
        raw_qvel = _array(before["raw_precontrol_qvel"], (qvel.shape[1],), "raw qvel")
        _exact(raw_qpos, qpos[tick], (qpos.shape[1],), "saved pre-control qpos")
        _exact(raw_qvel, qvel[tick], (qvel.shape[1],), "saved pre-control qvel")
        _exact(before["raw_joint_position"], raw_qpos[qadr], (16,),
               "compiled-address raw joint position")
        _exact(before["raw_joint_velocity"], raw_qvel[vadr], (16,),
               "compiled-address raw joint velocity")
        _exact(before["provider_joint_position"], raw_qpos[qadr], (16,),
               "synchronized provider joint position")
        _exact(before["provider_joint_velocity"], raw_qvel[vadr], (16,),
               "synchronized provider joint velocity")
        quaternion = raw_qpos[3:7]
        if quaternion.shape != (4,) or not np.isclose(np.linalg.norm(quaternion), 1.0,
                                                       atol=1e-10, rtol=0):
            raise ValueError(f"raw base quaternion malformed at tick {tick}")
        w, x, y, z = quaternion
        rotation = np.array([
            [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
            [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
            [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
        ])
        _close(before["provider_base_rotation"], rotation, (3, 3),
               "synchronized provider base rotation")
    return {"record_valid": True, "linked_precontrol_snapshots": len(pre_control_states),
            "fixed_synchronized_provider_linked_to_raw_truth": True}
