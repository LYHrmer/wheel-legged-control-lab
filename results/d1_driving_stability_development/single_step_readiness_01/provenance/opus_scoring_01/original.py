"""Deterministic single-step (fixed 15 mm box vs. plane) paired readiness scorer.

Pure Python/numpy.  No MuJoCo import, no instantiation, no physical execution:
this module only audits an already-recorded archive payload and reports
record faithfulness and task gates as independent outputs.

Public API
----------
score_single_step(payload: dict) -> dict
    Keys: record_valid, complete, wheel_box_contact_qualified,
    mechanical_crossing, final_stable, all_time_envelope_passed,
    no_nonwheel_contact, task_passed, failure_reasons, metrics.

Design rules honoured here:
  * Record validity (faithfulness of the archive) and task gates (physics
    outcome) are computed independently.  A robot that falls over, fails to
    cross the box, or ends with residual velocity still yields
    record_valid=True; only corruption / identity failure / malformed schema
    invalidates the record.
  * Fail closed: any malformed, missing, non-finite or inconsistent field
    produces record_valid=False plus an explicit reason.  Unknown exceptions
    are caught and reported as invalid records.
  * Never NaN success: every gate is a strict boolean; unavailable metrics are
    reported as None (never NaN / infinity), and non-finite inputs invalidate.
  * No invented tolerances.  Time arithmetic uses <= 1e-9; the raw command
    schedule and zero-arrays use exact float / bitwise equality; quaternion
    unit norm uses the specified 1e-6.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------- constants --
CONTROL_DT = 0.01
NATIVE_DT = 0.002
NATIVE_PER_INTERVAL = 5
TARGET_INTERVALS = 1200
TARGET_NATIVE = TARGET_INTERVALS * NATIVE_PER_INTERVAL      # 6000
TIME_TOL = 1e-9
QUAT_TOL = 1e-6

COMMAND_HEIGHT_M = 0.455
FORWARD_CMD_MPS = 0.2
FORWARD_START_TICK = 200
FORWARD_STOP_TICK = 1000
ALLOWED_GROUND_HEIGHTS_BOX = (0.0, 0.015)
ALLOWED_GROUND_HEIGHTS_PLANE = (0.0,)

TORQUE_LIMITS_NM = np.array([80.0, 80.0, 80.0, 12.0] * 4, dtype=float)
N_JOINTS = 16
N_QPOS = 23
N_QVEL = 22
XFRC_SHAPE = (18, 6)

ROLL_PITCH_LIMIT_DEG = 10.0
YAW_DEV_LIMIT_DEG = 5.0
LATERAL_LIMIT_M = 0.1

FINAL_TICK_FIRST = 1101
FINAL_TICK_LAST = 1200
FINAL_VX_LIMIT = 0.03
FINAL_VZ_LIMIT = 0.03
FINAL_Z_RMSE_LIMIT = 0.015
NATIVE_WINDOW_FIRST = 5500
NATIVE_WINDOW_LAST = 5999
WHEEL_LOAD_FRACTION_MIN = 0.95

FAR_EDGE_OFFSET_M = 0.88
N_WHEELS = 4


class _Invalid(Exception):
    """Raised for any record-faithfulness violation (fail closed)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = str(reason)


# ------------------------------------------------------------ type coercion --
def _req(container, key, ctx):
    if not isinstance(container, dict) or key not in container:
        raise _Invalid("missing_field:%s.%s" % (ctx, key))
    return container[key]


def _b(value, ctx):
    if not isinstance(value, (bool, np.bool_)):
        raise _Invalid("not_bool:%s" % ctx)
    return bool(value)


def _i(value, ctx):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise _Invalid("not_int:%s" % ctx)
    return int(value)


def _f(value, ctx):
    if isinstance(value, (bool, np.bool_)):
        raise _Invalid("not_float:%s" % ctx)
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise _Invalid("not_float:%s" % ctx)
    out = float(value)
    if not math.isfinite(out):
        raise _Invalid("not_finite:%s" % ctx)
    return out


def _seq(value, ctx):
    if isinstance(value, (str, bytes, dict)) or not isinstance(value, (list, tuple, np.ndarray)):
        raise _Invalid("not_list:%s" % ctx)
    return list(value)


def _vec(value, n, ctx):
    items = _seq(value, ctx)
    if len(items) != n:
        raise _Invalid("bad_length:%s" % ctx)
    return np.array([_f(x, ctx) for x in items], dtype=float)


def _mat(value, shape, ctx):
    if isinstance(value, (str, bytes, dict)) or not isinstance(value, (list, tuple, np.ndarray)):
        raise _Invalid("not_array:%s" % ctx)
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        raise _Invalid("not_numeric_array:%s" % ctx)
    if arr.shape != shape:
        raise _Invalid("bad_shape:%s" % ctx)
    if not np.all(np.isfinite(arr)):
        raise _Invalid("not_finite:%s" % ctx)
    return arr


def _opt_int(value, ctx, lo=None, hi=None):
    if value is None:
        return None
    out = _i(value, ctx)
    if lo is not None and out < lo:
        raise _Invalid("out_of_range:%s" % ctx)
    if hi is not None and out > hi:
        raise _Invalid("out_of_range:%s" % ctx)
    return out


# --------------------------------------------------------------- kinematics --
def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _quat_to_rpy(quat: np.ndarray, ctx: str):
    """MuJoCo (w, x, y, z) -> (roll, pitch, yaw), pure math/numpy."""
    w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or abs(norm - 1.0) > QUAT_TOL:
        raise _Invalid("quaternion_not_unit:%s" % ctx)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sp = 2.0 * (w * y - z * x)
    sp = 1.0 if sp > 1.0 else (-1.0 if sp < -1.0 else sp)
    pitch = math.asin(sp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


# ------------------------------------------------------- element validation --
def _parse_bounds(raw, ctx):
    items = _seq(raw, ctx)
    if not items:
        raise _Invalid("empty_collision_bounds:%s" % ctx)
    out = []
    for j, entry in enumerate(items):
        ectx = "%s[%d]" % (ctx, j)
        if not isinstance(entry, dict):
            raise _Invalid("not_dict:%s" % ectx)
        identity = _req(entry, "identity", ectx)
        if not isinstance(identity, str) or not identity:
            raise _Invalid("bad_identity:%s" % ectx)
        wheel = _opt_int(_req(entry, "wheel_index", ectx), ectx + ".wheel_index", 0, N_WHEELS - 1)
        lo = _vec(_req(entry, "minimum_world_m", ectx), 3, ectx + ".minimum_world_m")
        hi = _vec(_req(entry, "maximum_world_m", ectx), 3, ectx + ".maximum_world_m")
        if np.any(hi < lo):
            raise _Invalid("inverted_bounds:%s" % ectx)
        margin = _f(_req(entry, "margin_m", ectx), ectx + ".margin_m")
        if margin < 0.0:
            raise _Invalid("negative_margin:%s" % ectx)
        out.append({"identity": identity, "wheel": wheel, "min": lo, "max": hi, "margin": margin})
    return out


def _parse_endpoint(raw, k, ctx):
    if not isinstance(raw, dict):
        raise _Invalid("not_dict:%s" % ctx)
    if _i(_req(raw, "tick", ctx), ctx + ".tick") != k:
        raise _Invalid("tick_mismatch:%s" % ctx)
    t = _f(_req(raw, "time_s", ctx), ctx + ".time_s")
    if abs(t - CONTROL_DT * k) > TIME_TOL:
        raise _Invalid("time_mismatch:%s" % ctx)
    nonwheel = _i(_req(raw, "nonwheel_terrain_contacts", ctx), ctx + ".nonwheel_terrain_contacts")
    if nonwheel < 0:
        raise _Invalid("negative_contact_count:%s" % ctx)
    return {
        "pos": _vec(_req(raw, "base_position_m", ctx), 3, ctx + ".base_position_m"),
        "rpy": _vec(_req(raw, "rpy_rad", ctx), 3, ctx + ".rpy_rad"),
        "vx": _f(_req(raw, "body_vx_mps", ctx), ctx + ".body_vx_mps"),
        "vz": _f(_req(raw, "com_vz_mps", ctx), ctx + ".com_vz_mps"),
        "nonwheel": nonwheel,
        "bounds": _parse_bounds(_req(raw, "collision_bounds", ctx), ctx + ".collision_bounds"),
    }


def _parse_trace(raw, k, obstacle, ctx):
    if not isinstance(raw, dict):
        raise _Invalid("not_dict:%s" % ctx)
    if _i(_req(raw, "tick", ctx), ctx + ".tick") != k:
        raise _Invalid("tick_mismatch:%s" % ctx)
    action = _vec(_req(raw, "action", ctx), 8, ctx + ".action")
    if not np.array_equal(action, np.zeros(8, dtype=float)):
        raise _Invalid("action_not_zero:%s" % ctx)
    cmd = _req(raw, "raw_command", ctx)
    if not isinstance(cmd, dict):
        raise _Invalid("not_dict:%s.raw_command" % ctx)
    cctx = ctx + ".raw_command"
    if _i(_req(cmd, "tick", cctx), cctx + ".tick") != k:
        raise _Invalid("tick_mismatch:%s" % cctx)
    fwd = _f(_req(cmd, "forward_velocity_mps", cctx), cctx + ".forward_velocity_mps")
    expected_fwd = FORWARD_CMD_MPS if (FORWARD_START_TICK <= k < FORWARD_STOP_TICK) else 0.0
    if fwd != expected_fwd:
        raise _Invalid("forward_schedule_mismatch:%s" % cctx)
    if _f(_req(cmd, "yaw_rate_rps", cctx), cctx + ".yaw_rate_rps") != 0.0:
        raise _Invalid("yaw_command_nonzero:%s" % cctx)
    if _f(_req(cmd, "raw_world_height_m", cctx), cctx + ".raw_world_height_m") != COMMAND_HEIGHT_M:
        raise _Invalid("raw_height_mismatch:%s" % cctx)
    if _f(_req(cmd, "prepared_world_height_m", cctx), cctx + ".prepared_world_height_m") != COMMAND_HEIGHT_M:
        raise _Invalid("prepared_height_mismatch:%s" % cctx)
    ground = _f(_req(cmd, "ground_height_m", cctx), cctx + ".ground_height_m")
    allowed = ALLOWED_GROUND_HEIGHTS_BOX if obstacle else ALLOWED_GROUND_HEIGHTS_PLANE
    if ground not in allowed:
        raise _Invalid("ground_height_unexpected:%s" % cctx)
    clearance = _f(_req(cmd, "motion_clearance_m", cctx), cctx + ".motion_clearance_m")
    if clearance != (COMMAND_HEIGHT_M - ground):
        raise _Invalid("motion_clearance_mismatch:%s" % cctx)
    if _b(_req(raw, "stop_active", ctx), ctx + ".stop_active") != (k >= FORWARD_STOP_TICK):
        raise _Invalid("stop_flag_mismatch:%s" % ctx)
    if _b(_req(raw, "turn_active", ctx), ctx + ".turn_active"):
        raise _Invalid("turn_active_unexpected:%s" % ctx)
    torque = _vec(_req(raw, "torque_nm", ctx), N_JOINTS, ctx + ".torque_nm")
    if np.any(np.abs(torque) > TORQUE_LIMITS_NM):
        raise _Invalid("torque_channel_limit_exceeded:%s" % ctx)
    if _i(_req(raw, "callback_count_after_prepare", ctx), ctx + ".callback_count_after_prepare") != k + 2:
        raise _Invalid("callback_count_mismatch:%s" % ctx)
    return {"forward": fwd, "ground": ground}


def _parse_contact(raw, ctx):
    if not isinstance(raw, dict):
        raise _Invalid("not_dict:%s" % ctx)
    for key in ("geom1", "geom2", "body1", "body2"):
        _i(_req(raw, key, ctx), "%s.%s" % (ctx, key))
    _mat(_req(raw, "frame_geom1_to_geom2", ctx), (3, 3), ctx + ".frame")
    _vec(_req(raw, "position_world_m", ctx), 3, ctx + ".position_world_m")
    _vec(_req(raw, "local_force_torque", ctx), 6, ctx + ".local_force_torque")
    load = _f(_req(raw, "normal_load_n", ctx), ctx + ".normal_load_n")
    if load < 0.0:
        raise _Invalid("negative_normal_load:%s" % ctx)
    active = _b(_req(raw, "active", ctx), ctx + ".active")
    if not _b(_req(raw, "frame_orthonormal", ctx), ctx + ".frame_orthonormal"):
        raise _Invalid("frame_not_orthonormal:%s" % ctx)
    if not _b(_req(raw, "plane_normal_valid", ctx), ctx + ".plane_normal_valid"):
        raise _Invalid("plane_normal_invalid:%s" % ctx)
    box_feature = _req(raw, "box_feature", ctx)
    feature = None
    if box_feature is not None:
        if not isinstance(box_feature, dict):
            raise _Invalid("not_dict:%s.box_feature" % ctx)
        if not _b(_req(box_feature, "geometric_support_valid", ctx + ".box_feature"),
                  ctx + ".box_feature.geometric_support_valid"):
            raise _Invalid("box_feature_support_invalid:%s" % ctx)
        feature = _req(box_feature, "feature", ctx + ".box_feature")
        if not isinstance(feature, str) or not feature:
            raise _Invalid("box_feature_label_invalid:%s" % ctx)
    _opt_int(_req(raw, "terrain_geom_id", ctx), ctx + ".terrain_geom_id")
    _opt_int(_req(raw, "robot_geom_id", ctx), ctx + ".robot_geom_id")
    wheel = _opt_int(_req(raw, "wheel_index", ctx), ctx + ".wheel_index", 0, N_WHEELS - 1)
    return {"load": load, "active": active, "feature": feature, "wheel": wheel,
            "is_box": box_feature is not None}


def _parse_native(raw, idx, ctx):
    if not isinstance(raw, dict):
        raise _Invalid("not_dict:%s" % ctx)
    if _i(_req(raw, "index", ctx), ctx + ".index") != idx:
        raise _Invalid("index_mismatch:%s" % ctx)
    if not _b(_req(raw, "returned", ctx), ctx + ".returned"):
        raise _Invalid("native_step_not_returned:%s" % ctx)
    start = _f(_req(raw, "start_time_s", ctx), ctx + ".start_time_s")
    end = _f(_req(raw, "end_time_s", ctx), ctx + ".end_time_s")
    dt = _f(_req(raw, "actual_dt_s", ctx), ctx + ".actual_dt_s")
    if abs(dt - NATIVE_DT) > TIME_TOL:
        raise _Invalid("native_dt_mismatch:%s" % ctx)
    if abs(end - (start + dt)) > TIME_TOL:
        raise _Invalid("native_interval_mismatch:%s" % ctx)
    if abs(start - idx * NATIVE_DT) > TIME_TOL:
        raise _Invalid("native_start_time_mismatch:%s" % ctx)
    qpos_b = _vec(_req(raw, "qpos_before", ctx), N_QPOS, ctx + ".qpos_before")
    qpos_r = _vec(_req(raw, "qpos_returned", ctx), N_QPOS, ctx + ".qpos_returned")
    qvel_b = _vec(_req(raw, "qvel_before", ctx), N_QVEL, ctx + ".qvel_before")
    qvel_r = _vec(_req(raw, "qvel_returned", ctx), N_QVEL, ctx + ".qvel_returned")
    ctrl = _vec(_req(raw, "ctrl_nm", ctx), N_JOINTS, ctx + ".ctrl_nm")
    if np.any(np.abs(ctrl) > TORQUE_LIMITS_NM):
        raise _Invalid("native_torque_channel_limit_exceeded:%s" % ctx)
    xfrc = _mat(_req(raw, "xfrc_applied", ctx), XFRC_SHAPE, ctx + ".xfrc_applied")
    if not np.array_equal(xfrc, np.zeros(XFRC_SHAPE, dtype=float)):
        raise _Invalid("foreign_wrench_applied:%s" % ctx)
    qfrc = _vec(_req(raw, "qfrc_applied", ctx), N_QVEL, ctx + ".qfrc_applied")
    if not np.array_equal(qfrc, np.zeros(N_QVEL, dtype=float)):
        raise _Invalid("foreign_generalized_force:%s" % ctx)

    cdata = _req(raw, "contacts", ctx)
    if not isinstance(cdata, dict):
        raise _Invalid("not_dict:%s.contacts" % ctx)
    cctx = ctx + ".contacts"
    wheel_loads = _vec(_req(cdata, "wheel_positive_normal_load_n", cctx), N_WHEELS,
                       cctx + ".wheel_positive_normal_load_n")
    box_loads = _vec(_req(cdata, "wheel_box_positive_normal_load_n", cctx), N_WHEELS,
                     cctx + ".wheel_box_positive_normal_load_n")
    if np.any(wheel_loads < 0.0) or np.any(box_loads < 0.0):
        raise _Invalid("negative_wheel_load:%s" % cctx)
    nonwheel = _i(_req(cdata, "nonwheel_terrain_contacts", cctx), cctx + ".nonwheel_terrain_contacts")
    if nonwheel < 0:
        raise _Invalid("negative_contact_count:%s" % cctx)
    geometric_box = _b(_req(cdata, "geometric_box_contact", cctx), cctx + ".geometric_box_contact")
    contacts = [_parse_contact(c, "%s.contacts[%d]" % (cctx, j))
                for j, c in enumerate(_seq(_req(cdata, "contacts", cctx), cctx + ".contacts"))]

    roll_b, pitch_b, yaw_b = _quat_to_rpy(qpos_b[3:7], ctx + ".qpos_before")
    roll_r, pitch_r, yaw_r = _quat_to_rpy(qpos_r[3:7], ctx + ".qpos_returned")
    return {
        "qpos_returned": qpos_r, "qvel_returned": qvel_r,
        "qpos_before": qpos_b, "qvel_before": qvel_b,
        "poses": ((roll_b, pitch_b, yaw_b, qpos_b[1]), (roll_r, pitch_r, yaw_r, qpos_r[1])),
        "wheel_loads": wheel_loads, "box_loads": box_loads,
        "nonwheel": nonwheel, "geometric_box": geometric_box, "contacts": contacts,
    }


# ------------------------------------------------------------------ scoring --
def _score(payload, result):
    reasons = result["failure_reasons"]
    metrics = result["metrics"]
    if not isinstance(payload, dict):
        raise _Invalid("payload_not_dict")

    obstacle = _b(_req(payload, "obstacle_enabled", "payload"), "obstacle_enabled")
    if not _b(_req(payload, "source_identity_valid", "payload"), "source_identity_valid"):
        raise _Invalid("source_identity_invalid")
    error = _req(payload, "error", "payload")
    if error is not None:
        if not isinstance(error, dict):
            raise _Invalid("error_field_malformed")
        raise _Invalid("execution_error:%s" % str(error.get("type", "unknown"))[:64])
    terminated = _b(_req(payload, "terminated", "payload"), "terminated")
    truncated = _b(_req(payload, "truncated", "payload"), "truncated")

    T = _i(_req(payload, "completed_control_intervals", "payload"), "completed_control_intervals")
    if T < 0 or T > TARGET_INTERVALS:
        raise _Invalid("interval_count_out_of_range")
    endpoints_raw = _seq(_req(payload, "endpoints", "payload"), "endpoints")
    trace_raw = _seq(_req(payload, "trace", "payload"), "trace")
    native_raw = _seq(_req(payload, "native", "payload"), "native")
    if T == 0:
        raise _Invalid("no_control_intervals")
    if len(endpoints_raw) != T + 1:
        raise _Invalid("endpoint_count_mismatch")
    if len(trace_raw) != T:
        raise _Invalid("trace_count_mismatch")
    if len(native_raw) == 0:
        raise _Invalid("native_samples_missing")

    endpoints = [_parse_endpoint(e, k, "endpoints[%d]" % k) for k, e in enumerate(endpoints_raw)]
    for k, tr in enumerate(trace_raw):
        _parse_trace(tr, k, obstacle, "trace[%d]" % k)

    native = []
    prev = None
    for idx, nb in enumerate(native_raw):
        cur = _parse_native(nb, idx, "native[%d]" % idx)
        if prev is not None:
            if not np.array_equal(prev["qpos_returned"], cur["qpos_before"]):
                raise _Invalid("qpos_chain_break:native[%d]" % idx)
            if not np.array_equal(prev["qvel_returned"], cur["qvel_before"]):
                raise _Invalid("qvel_chain_break:native[%d]" % idx)
        native.append(cur)
        prev = cur

    result["record_valid"] = True
    metrics["completed_control_intervals"] = T
    metrics["endpoint_count"] = len(endpoints)
    metrics["native_count"] = len(native)
    metrics["terminated"] = terminated
    metrics["truncated"] = truncated

    complete = (T == TARGET_INTERVALS) and (len(native) == TARGET_NATIVE)
    result["complete"] = bool(complete)
    if len(native) != NATIVE_PER_INTERVAL * T:
        reasons.append("native_count_inconsistent_with_intervals")

    # ---- reference frame -------------------------------------------------
    x0 = float(endpoints[0]["pos"][0])
    y0 = float(endpoints[0]["pos"][1])
    yaw0 = float(endpoints[0]["rpy"][2])
    metrics["initial_x_m"] = x0
    metrics["initial_y_m"] = y0
    metrics["initial_yaw_rad"] = yaw0

    # ---- all-time envelope (endpoints + native) --------------------------
    max_roll = max_pitch = max_yaw_dev = max_lat = 0.0
    for ep in endpoints:
        max_roll = max(max_roll, abs(math.degrees(float(ep["rpy"][0]))))
        max_pitch = max(max_pitch, abs(math.degrees(float(ep["rpy"][1]))))
        max_yaw_dev = max(max_yaw_dev, abs(math.degrees(_wrap_pi(float(ep["rpy"][2]) - yaw0))))
        max_lat = max(max_lat, abs(float(ep["pos"][1]) - y0))
    for st in native:
        for roll, pitch, yaw, ypos in st["poses"]:
            max_roll = max(max_roll, abs(math.degrees(roll)))
            max_pitch = max(max_pitch, abs(math.degrees(pitch)))
            max_yaw_dev = max(max_yaw_dev, abs(math.degrees(_wrap_pi(yaw - yaw0))))
            max_lat = max(max_lat, abs(float(ypos) - y0))
    metrics["max_abs_roll_deg"] = max_roll
    metrics["max_abs_pitch_deg"] = max_pitch
    metrics["max_abs_yaw_deviation_deg"] = max_yaw_dev
    metrics["max_abs_lateral_deviation_m"] = max_lat
    envelope = (max_roll <= ROLL_PITCH_LIMIT_DEG and max_pitch <= ROLL_PITCH_LIMIT_DEG
                and max_yaw_dev <= YAW_DEV_LIMIT_DEG and max_lat <= LATERAL_LIMIT_M)
    result["all_time_envelope_passed"] = bool(envelope)

    # ---- non-wheel terrain contacts (geometric labels, not load) ---------
    ep_nonwheel = int(sum(ep["nonwheel"] for ep in endpoints))
    nat_nonwheel = int(sum(st["nonwheel"] for st in native))
    metrics["nonwheel_contact_endpoint_total"] = ep_nonwheel
    metrics["nonwheel_contact_native_total"] = nat_nonwheel
    no_nonwheel = (ep_nonwheel == 0 and nat_nonwheel == 0)
    result["no_nonwheel_contact"] = bool(no_nonwheel)

    # ---- wheel/box contact qualification (box only) ----------------------
    first_geom_box = None
    first_pos_box = None
    feature_counts = {}
    pos_steps = 0
    pos_wheels = set()
    load_without_detail = False
    for idx, st in enumerate(native):
        if st["geometric_box"] and first_geom_box is None:
            first_geom_box = idx
        detail = [c for c in st["contacts"]
                  if c["is_box"] and c["wheel"] is not None and c["active"] and c["load"] > 0.0]
        array_positive = bool(np.any(st["box_loads"] > 0.0))
        if detail:
            pos_steps += 1
            for c in detail:
                pos_wheels.add(int(c["wheel"]))
                key = c["feature"] if c["feature"] is not None else "unlabelled"
                feature_counts[key] = feature_counts.get(key, 0) + 1
        if array_positive and detail and first_pos_box is None:
            first_pos_box = idx
        if array_positive and not detail:
            load_without_detail = True
    if load_without_detail:
        reasons.append("positive_wheel_box_load_without_contact_detail")
    metrics["first_geometric_box_native_index"] = first_geom_box if obstacle else None
    metrics["first_positive_wheel_box_native_index"] = first_pos_box if obstacle else None
    metrics["positive_wheel_box_feature_counts"] = dict(feature_counts) if obstacle else None
    metrics["positive_wheel_box_native_steps"] = pos_steps if obstacle else None
    metrics["positive_wheel_box_wheel_indices"] = sorted(pos_wheels) if obstacle else None
    qualified = None
    if obstacle:
        qualified = bool(first_pos_box is not None)
        result["wheel_box_contact_qualified"] = qualified
    else:
        result["wheel_box_contact_qualified"] = None

    # ---- final stability (exact windows only) ----------------------------
    ep_window_ok = (len(endpoints) > FINAL_TICK_LAST)
    nat_window_ok = (len(native) > NATIVE_WINDOW_LAST)
    metrics["final_endpoint_window_ticks"] = [FINAL_TICK_FIRST, FINAL_TICK_LAST]
    metrics["final_native_window_indices"] = [NATIVE_WINDOW_FIRST, NATIVE_WINDOW_LAST]
    vx_max = vz_max = z_rmse = None
    fractions = None
    if ep_window_ok:
        win = endpoints[FINAL_TICK_FIRST:FINAL_TICK_LAST + 1]
        vx_max = max(abs(ep["vx"]) for ep in win)
        vz_max = max(abs(ep["vz"]) for ep in win)
        dz = np.array([float(ep["pos"][2]) - COMMAND_HEIGHT_M for ep in win], dtype=float)
        z_rmse = float(math.sqrt(float(np.mean(dz * dz))))
    if nat_window_ok:
        loads = np.array([st["wheel_loads"] for st in
                          native[NATIVE_WINDOW_FIRST:NATIVE_WINDOW_LAST + 1]], dtype=float)
        fractions = [float(np.mean(loads[:, w] > 0.0)) for w in range(N_WHEELS)]
    metrics["final_window_max_abs_body_vx_mps"] = vx_max
    metrics["final_window_max_abs_com_vz_mps"] = vz_max
    metrics["final_window_base_z_rmse_m"] = z_rmse
    metrics["final_window_wheel_positive_load_fractions"] = fractions
    final_stable = bool(
        ep_window_ok and nat_window_ok
        and vx_max is not None and vx_max <= FINAL_VX_LIMIT
        and vz_max is not None and vz_max <= FINAL_VZ_LIMIT
        and z_rmse is not None and z_rmse <= FINAL_Z_RMSE_LIMIT
        and fractions is not None and all(fr >= WHEEL_LOAD_FRACTION_MIN for fr in fractions))
    result["final_stable"] = final_stable

    # ---- mechanical crossing (box only, all collision geoms) -------------
    if obstacle:
        far_edge = x0 + FAR_EDGE_OFFSET_M
        final_bounds = endpoints[-1]["bounds"]
        min_x = min(float(b["min"][0]) for b in final_bounds)
        clearance = min(float(b["min"][0]) - far_edge - float(b["margin"]) for b in final_bounds)
        wheels_present = sorted({int(b["wheel"]) for b in final_bounds if b["wheel"] is not None})
        wheels_ok = (wheels_present == [0, 1, 2, 3])
        geoms_ok = all(float(b["min"][0]) > far_edge + float(b["margin"]) for b in final_bounds)
        metrics["far_edge_m"] = far_edge
        metrics["final_min_robot_x_m"] = min_x
        metrics["minimum_clearance_beyond_far_edge_m"] = clearance
        metrics["final_wheel_indices_present"] = wheels_present
        metrics["final_collision_geom_count"] = len(final_bounds)
        if not wheels_ok:
            reasons.append("final_bounds_missing_four_wheel_geoms")
        crossing = bool(complete and qualified and no_nonwheel and wheels_ok and geoms_ok)
        result["mechanical_crossing"] = crossing
    else:
        for key in ("far_edge_m", "final_min_robot_x_m", "minimum_clearance_beyond_far_edge_m",
                    "final_wheel_indices_present", "final_collision_geom_count"):
            metrics[key] = None
        crossing = None
        result["mechanical_crossing"] = None

    # ---- task gates ------------------------------------------------------
    if not complete:
        reasons.append("trial_incomplete")
        if not terminated and not truncated:
            reasons.append("incomplete_without_termination_flag")
    if terminated:
        reasons.append("physical_termination")
    if not envelope:
        reasons.append("all_time_envelope_violated")
    if not no_nonwheel:
        reasons.append("nonwheel_terrain_contact_present")
    if not final_stable:
        reasons.append("final_stability_window_failed")
    if obstacle:
        if not qualified:
            reasons.append("no_positive_wheel_box_normal_load")
        if not crossing:
            reasons.append("mechanical_crossing_failed")

    task = bool(complete and (not terminated) and envelope and no_nonwheel and final_stable)
    if obstacle:
        task = bool(task and qualified and crossing)
    result["task_passed"] = task


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    return obj


def score_single_step(payload: dict) -> dict:
    """Score one recorded fixed-step archive (box or plane) with fail-closed audit."""
    result = {
        "record_valid": False,
        "complete": False,
        "wheel_box_contact_qualified": None,
        "mechanical_crossing": None,
        "final_stable": False,
        "all_time_envelope_passed": False,
        "no_nonwheel_contact": False,
        "task_passed": False,
        "failure_reasons": [],
        "metrics": {},
    }
    obstacle_hint = None
    if isinstance(payload, dict) and isinstance(payload.get("obstacle_enabled"), (bool, np.bool_)):
        obstacle_hint = bool(payload["obstacle_enabled"])
    if obstacle_hint:
        result["wheel_box_contact_qualified"] = False
        result["mechanical_crossing"] = False

    try:
        _score(payload, result)
    except _Invalid as exc:
        result["failure_reasons"].append("record_invalid:%s" % exc.reason)
        result["record_valid"] = False
    except Exception as exc:  # unknown failure -> fail closed
        result["failure_reasons"].append(
            "record_invalid:unhandled_exception:%s" % type(exc).__name__)
        result["record_valid"] = False

    if not result["record_valid"]:
        result["complete"] = False
        result["final_stable"] = False
        result["all_time_envelope_passed"] = False
        result["no_nonwheel_contact"] = False
        result["task_passed"] = False
        if obstacle_hint:
            result["wheel_box_contact_qualified"] = False
            result["mechanical_crossing"] = False
        else:
            result["wheel_box_contact_qualified"] = None
            result["mechanical_crossing"] = None

    result["metrics"] = _sanitize(result["metrics"])
    result["failure_reasons"] = [str(r) for r in result["failure_reasons"]]
    result["task_passed"] = bool(result["task_passed"] and result["record_valid"])
    return result
