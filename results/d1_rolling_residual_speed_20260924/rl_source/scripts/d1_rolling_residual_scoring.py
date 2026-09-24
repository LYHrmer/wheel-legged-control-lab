"""Pure faithful-record and physical gates for rolling residual episodes.

Endpoint/native parsing and cross-links come from the frozen zero-task scorer;
the action and variable-speed command are checked here against their real rows.
No recorded input is rewritten to fit the old zero-action/0.2 m/s scorer.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from scripts import d1_single_step_scoring as old
from scripts.d1_rolling_residual_task import (
    ORIGINAL_LEG_EXTENSION_SCALE_M,
    PHYSICAL_ACTION_SIZE,
    RAW_WORLD_HEIGHT_M,
    RollingEpisodeSpec,
    expand_policy_action,
    raw_command_at_tick,
)


def _number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _real_vector(value: Any, length: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in ("f", "i", "u") or array.shape != (length,):
        raise ValueError(f"{name} must be a real vector of length {length}")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def validate_rolling_action_record(row: dict, raw_forward_velocity_mps: float) -> None:
    """Reject any raw/clipped/gate/physical/applied action disagreement."""
    if not isinstance(row, dict):
        raise TypeError("rolling trace row must be a dict")
    policy = _real_vector(row["policy_action"], 1, "policy_action")
    expected = expand_policy_action(policy, raw_forward_velocity_mps)
    clipped = _number(row["clipped_policy_scalar"], "clipped_policy_scalar")
    if clipped != expected.clipped_scalar:
        raise ValueError("clipped policy scalar disagrees with the raw input")
    active = row["forward_gate_active"]
    if type(active) is not bool or active != expected.forward_gate_active:
        raise ValueError("forward residual gate disagrees with raw command")
    physical = _real_vector(row["action"], PHYSICAL_ACTION_SIZE, "action")
    applied = _real_vector(row["applied_action"], PHYSICAL_ACTION_SIZE, "applied_action")
    if not np.array_equal(physical, np.asarray(expected.physical_action)):
        raise ValueError("physical action differs from shared-leg expansion")
    if not np.array_equal(applied, physical):
        raise ValueError("actual applied action differs from recorded physical action")


def validate_rolling_command_record(
    command: dict, tick: int, spec: RollingEpisodeSpec
) -> None:
    """Check actual raw command/height against this episode's frozen schedule."""
    if not isinstance(command, dict):
        raise TypeError("raw command must be a dict")
    expected = raw_command_at_tick(spec, tick)
    actual_tick = command["tick"]
    if isinstance(actual_tick, bool) or type(actual_tick) is not int or actual_tick != tick:
        raise ValueError("raw command tick mismatch")
    if _number(command["forward_velocity_mps"], "forward_velocity_mps") != expected["forward_velocity_mps"]:
        raise ValueError("raw forward command differs from the scenario")
    if _number(command["yaw_rate_rps"], "yaw_rate_rps") != 0.0:
        raise ValueError("raw yaw command must be zero")
    for key in ("raw_world_height_m", "prepared_world_height_m"):
        if _number(command[key], key) != RAW_WORLD_HEIGHT_M:
            raise ValueError(f"{key} differs from fixed world z")
    ground = _number(command["ground_height_m"], "ground_height_m")
    if ground not in ((0.0, 0.015) if spec.obstacle_enabled else (0.0,)):
        raise ValueError("ground reference differs from the fixed plane/box")
    if _number(command["motion_clearance_m"], "motion_clearance_m") != RAW_WORLD_HEIGHT_M - ground:
        raise ValueError("clearance adaptation disagrees with world z minus ground")


def _validate_trace(row: dict, tick: int, spec: RollingEpisodeSpec) -> tuple[float, float]:
    if not isinstance(row, dict) or type(row.get("tick")) is not int or row["tick"] != tick:
        raise ValueError("trace tick mismatch")
    if type(row.get("endpoint_tick")) is not int or row["endpoint_tick"] != tick + 1:
        raise ValueError("trace endpoint tick mismatch")
    command = row["raw_command"]
    validate_rolling_command_record(command, tick, spec)
    forward = float(command["forward_velocity_mps"])
    validate_rolling_action_record(row, forward)
    if type(row["stop_active"]) is not bool or row["stop_active"] != (tick >= spec.forward_stop_tick):
        raise ValueError("stop latch differs from scheduled stop")
    if type(row["turn_active"]) is not bool or row["turn_active"]:
        raise ValueError("raw turn authority unexpectedly active")
    torque = _real_vector(row["torque_nm"], old.N_JOINTS, "torque_nm")
    if np.any(np.abs(torque) > old.TORQUE_LIMITS_NM):
        raise ValueError("protected torque exceeds original per-channel limit")
    controller = row["controller_result"]
    stop = row["stop_record"]
    if not isinstance(controller, dict) or not isinstance(stop, dict):
        raise TypeError("actual controller and stop records must be present")
    if not np.array_equal(
        _real_vector(controller["clipped_action"], PHYSICAL_ACTION_SIZE,
                     "controller_result.clipped_action"),
        _real_vector(row["applied_action"], PHYSICAL_ACTION_SIZE, "applied_action"),
    ):
        raise ValueError("controller clipped action differs from applied physical action")
    for name, actual in (("controller_result.torque_nm", controller["torque_nm"]),
                         ("stop_record.safe_torque_nm", stop["safe_torque_nm"])):
        if not np.array_equal(_real_vector(actual, old.N_JOINTS, name), torque):
            raise ValueError(f"{name} differs from executed protected torque")
    count = row["callback_count_after_prepare"]
    if type(count) is not int or count != tick + 2:
        raise ValueError("command callback count mismatch")
    return abs(float(row["policy_action"][0])), max(abs(float(v)) for v in row["action"][:4])


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def score_rolling_episode(payload: dict, spec: RollingEpisodeSpec) -> dict[str, Any]:
    """Score actual full records, keeping record validity separate from task pass."""
    result: dict[str, Any] = {
        "record_valid": False,
        "complete": False,
        "all_time_envelope_passed": False,
        "no_nonwheel_contact": False,
        "wheel_box_contact_qualified": None,
        "mechanical_crossing": None,
        "final_stable": False,
        "speed_gate_passed": False,
        "task_passed": False,
        "failure_reasons": [],
        "metrics": {},
    }
    try:
        if not isinstance(spec, RollingEpisodeSpec) or not isinstance(payload, dict):
            raise TypeError("spec/payload type invalid")
        if type(payload.get("obstacle_enabled")) is not bool or payload["obstacle_enabled"] != spec.obstacle_enabled:
            raise ValueError("payload obstacle identity differs from episode spec")
        if payload.get("spec") != spec.as_dict() or payload.get("source_identity_valid") is not True:
            raise ValueError("episode spec or source identity invalid")
        if payload.get("error") is not None:
            raise ValueError("episode has an execution error")
        for flag in ("terminated", "truncated"):
            if type(payload.get(flag)) is not bool:
                raise ValueError(f"{flag} must be bool")
        count = payload.get("completed_control_intervals")
        if type(count) is not int or not 0 < count <= 1200:
            raise ValueError("completed control count invalid")
        raw_endpoints, raw_trace, raw_native = (
            payload[name] for name in ("endpoints", "trace", "native")
        )
        if (not all(isinstance(value, list) for value in (raw_endpoints, raw_trace, raw_native))
                or len(raw_endpoints) != count + 1 or len(raw_trace) != count
                or len(raw_native) != 5 * count):
            raise ValueError("endpoint/trace/native lengths disagree with controls")
        old.validate_record_links(payload)
        endpoints = [old._parse_endpoint(row, tick, f"endpoint[{tick}]")
                     for tick, row in enumerate(raw_endpoints)]
        raw_max = applied_max = 0.0
        for tick, row in enumerate(raw_trace):
            policy_abs, leg_abs = _validate_trace(row, tick, spec)
            raw_max = max(raw_max, policy_abs)
            applied_max = max(applied_max, leg_abs)
        native = []
        previous = None
        for index, row in enumerate(raw_native):
            parsed = old._parse_native(row, index, f"native[{index}]")
            if previous is not None and (
                not np.array_equal(previous["qpos_returned"], parsed["qpos_before"])
                or not np.array_equal(previous["qvel_returned"], parsed["qvel_before"])
            ):
                raise ValueError("native qpos/qvel chain break")
            native.append(parsed)
            previous = parsed
        result["record_valid"] = True
    except (KeyError, TypeError, ValueError, IndexError, old._Invalid) as exc:
        result["failure_reasons"].append(f"record_invalid:{type(exc).__name__}:{exc}")
        return result

    metrics = result["metrics"]
    reasons = result["failure_reasons"]
    complete = count == 1200 and len(native) == 6000
    result["complete"] = complete
    metrics.update({
        "completed_control_intervals": count,
        "endpoint_count": len(endpoints),
        "native_count": len(native),
        "max_abs_raw_policy_scalar": raw_max,
        "max_abs_applied_leg_normalized": applied_max,
        "max_abs_leg_extension_residual_m": applied_max * ORIGINAL_LEG_EXTENSION_SCALE_M,
    })
    x0, y0, yaw0 = float(endpoints[0]["pos"][0]), float(endpoints[0]["pos"][1]), float(endpoints[0]["rpy"][2])
    max_roll = max_pitch = max_yaw_dev = max_lateral = 0.0
    for row in endpoints:
        roll, pitch, yaw = row["rpy"]
        max_roll = max(max_roll, abs(math.degrees(float(roll))))
        max_pitch = max(max_pitch, abs(math.degrees(float(pitch))))
        max_yaw_dev = max(max_yaw_dev, abs(math.degrees(_wrap_pi(float(yaw) - yaw0))))
        max_lateral = max(max_lateral, abs(float(row["pos"][1]) - y0))
    for row in native:
        for roll, pitch, yaw, ypos in row["poses"]:
            max_roll = max(max_roll, abs(math.degrees(roll)))
            max_pitch = max(max_pitch, abs(math.degrees(pitch)))
            max_yaw_dev = max(max_yaw_dev, abs(math.degrees(_wrap_pi(yaw - yaw0))))
            max_lateral = max(max_lateral, abs(float(ypos) - y0))
    metrics.update({
        "max_abs_roll_deg": max_roll,
        "max_abs_pitch_deg": max_pitch,
        "max_abs_yaw_deviation_deg": max_yaw_dev,
        "max_abs_lateral_deviation_m": max_lateral,
    })
    envelope = (max_roll <= old.ROLL_PITCH_LIMIT_DEG and max_pitch <= old.ROLL_PITCH_LIMIT_DEG
                and max_yaw_dev <= old.YAW_DEV_LIMIT_DEG and max_lateral <= old.LATERAL_LIMIT_M)
    result["all_time_envelope_passed"] = bool(envelope)

    nonwheel_endpoint = sum(row["nonwheel"] for row in endpoints)
    nonwheel_native = sum(row["nonwheel"] for row in native)
    no_nonwheel = nonwheel_endpoint == 0 and nonwheel_native == 0
    metrics["nonwheel_contact_endpoint_total"] = nonwheel_endpoint
    metrics["nonwheel_contact_native_total"] = nonwheel_native
    result["no_nonwheel_contact"] = no_nonwheel

    first_geom_box = first_pos_box = None
    box_positive_steps = 0
    box_wheels: set[int] = set()
    feature_counts: dict[str, int] = {}
    load_without_detail = False
    for index, row in enumerate(native):
        if row["geometric_box"] and first_geom_box is None:
            first_geom_box = index
        detail = [contact for contact in row["contacts"]
                  if contact["is_box"] and contact["wheel"] is not None
                  and contact["active"] and contact["load"] > 0.0]
        array_positive = bool(np.any(row["box_loads"] > 0.0))
        if detail:
            box_positive_steps += 1
            for contact in detail:
                box_wheels.add(int(contact["wheel"]))
                feature = contact["feature"] if contact["feature"] is not None else "unlabelled"
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        if array_positive and detail and first_pos_box is None:
            first_pos_box = index
        if array_positive and not detail:
            load_without_detail = True
    if load_without_detail:
        reasons.append("positive_wheel_box_load_without_contact_detail")
    metrics.update({
        "first_geometric_box_native_index": first_geom_box if spec.obstacle_enabled else None,
        "first_positive_wheel_box_native_index": first_pos_box if spec.obstacle_enabled else None,
        "positive_wheel_box_native_steps": box_positive_steps if spec.obstacle_enabled else None,
        "positive_wheel_box_wheel_indices": sorted(box_wheels) if spec.obstacle_enabled else None,
        "positive_wheel_box_feature_counts": feature_counts if spec.obstacle_enabled else None,
    })
    qualified = bool(first_pos_box is not None) if spec.obstacle_enabled else None
    result["wheel_box_contact_qualified"] = qualified

    final_endpoint_window = len(endpoints) > old.FINAL_TICK_LAST
    final_native_window = len(native) > old.NATIVE_WINDOW_LAST
    metrics["final_endpoint_window_ticks"] = [old.FINAL_TICK_FIRST, old.FINAL_TICK_LAST]
    metrics["final_native_window_indices"] = [old.NATIVE_WINDOW_FIRST, old.NATIVE_WINDOW_LAST]
    vx_max = vz_max = z_rmse = fractions = None
    if final_endpoint_window:
        window = endpoints[old.FINAL_TICK_FIRST:old.FINAL_TICK_LAST + 1]
        vx_max = max(abs(row["vx"]) for row in window)
        vz_max = max(abs(row["vz"]) for row in window)
        heights = np.asarray([float(row["pos"][2]) - old.COMMAND_HEIGHT_M for row in window])
        z_rmse = float(math.sqrt(float(np.mean(heights * heights))))
    if final_native_window:
        loads = np.asarray([row["wheel_loads"] for row in
                            native[old.NATIVE_WINDOW_FIRST:old.NATIVE_WINDOW_LAST + 1]])
        fractions = [float(np.mean(loads[:, wheel] > 0.0)) for wheel in range(old.N_WHEELS)]
    metrics.update({
        "final_window_max_abs_body_vx_mps": vx_max,
        "final_window_max_abs_com_vz_mps": vz_max,
        "final_window_base_z_rmse_m": z_rmse,
        "final_window_wheel_positive_load_fractions": fractions,
    })
    stable = bool(final_endpoint_window and final_native_window
                  and vx_max is not None and vx_max <= old.FINAL_VX_LIMIT
                  and vz_max is not None and vz_max <= old.FINAL_VZ_LIMIT
                  and z_rmse is not None and z_rmse <= old.FINAL_Z_RMSE_LIMIT
                  and fractions is not None
                  and all(fraction >= old.WHEEL_LOAD_FRACTION_MIN for fraction in fractions))
    result["final_stable"] = stable

    crossing = None
    if spec.obstacle_enabled:
        far_edge = x0 + old.FAR_EDGE_OFFSET_M
        final_bounds = endpoints[-1]["bounds"]
        min_x = min(float(bound["min"][0]) for bound in final_bounds)
        clearance = min(float(bound["min"][0]) - far_edge - float(bound["margin"])
                        for bound in final_bounds)
        wheels = sorted({int(bound["wheel"]) for bound in final_bounds if bound["wheel"] is not None})
        wheels_ok = wheels == [0, 1, 2, 3]
        geoms_ok = all(float(bound["min"][0]) > far_edge + float(bound["margin"])
                       for bound in final_bounds)
        crossing = bool(complete and qualified and no_nonwheel and wheels_ok and geoms_ok)
        metrics.update({
            "far_edge_m": far_edge,
            "final_min_robot_x_m": min_x,
            "minimum_clearance_beyond_far_edge_m": clearance,
            "final_wheel_indices_present": wheels,
            "final_collision_geom_count": len(final_bounds),
        })
        if not wheels_ok:
            reasons.append("final_bounds_missing_four_wheel_geoms")
    else:
        metrics.update(dict.fromkeys((
            "far_edge_m", "final_min_robot_x_m", "minimum_clearance_beyond_far_edge_m",
            "final_wheel_indices_present", "final_collision_geom_count",
        )))
    result["mechanical_crossing"] = crossing

    speed_start = spec.forward_start_tick + 100
    speed_stop = spec.forward_stop_tick - 100
    metrics["speed_window_endpoint_ticks"] = [speed_start, speed_stop]
    speed_mean = speed_rms = None
    if len(endpoints) > speed_stop:
        velocities = np.asarray([row["vx"] for row in endpoints[speed_start:speed_stop + 1]])
        speed_mean = float(np.mean(velocities))
        speed_rms = float(np.sqrt(np.mean((velocities - spec.forward_velocity_mps) ** 2)))
    speed_ok = bool(speed_mean is not None and speed_mean >= 0.90 * spec.forward_velocity_mps
                    and speed_rms is not None and speed_rms <= 0.05)
    metrics["speed_window_mean_body_vx_mps"] = speed_mean
    metrics["speed_window_rms_command_error_mps"] = speed_rms
    metrics["commanded_forward_velocity_mps"] = spec.forward_velocity_mps
    result["speed_gate_passed"] = speed_ok

    if not complete:
        reasons.append("trial_incomplete")
    if payload["terminated"]:
        reasons.append("physical_termination")
    if not envelope:
        reasons.append("all_time_envelope_violated")
    if not no_nonwheel:
        reasons.append("nonwheel_terrain_contact_present")
    if not stable:
        reasons.append("final_stability_window_failed")
    if spec.obstacle_enabled and not qualified:
        reasons.append("no_positive_wheel_box_normal_load")
    if spec.obstacle_enabled and not crossing:
        reasons.append("mechanical_crossing_failed")
    if not speed_ok:
        reasons.append("speed_window_failed")
    result["task_passed"] = bool(complete and not payload["terminated"] and envelope
                                 and no_nonwheel and stable and speed_ok
                                 and (not spec.obstacle_enabled or (qualified and crossing)))
    return result
