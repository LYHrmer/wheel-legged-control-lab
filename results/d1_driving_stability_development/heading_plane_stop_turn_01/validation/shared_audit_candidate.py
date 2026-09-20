"""Independent saved-state audit; no integration, collision solve, or controller compute.

The original G1 scorer is used only as a pure reduction AFTER its physical/raw
inputs have been checked against saved qpos/qvel and the frozen command profile.
Native solved-force caches and synchronized endpoint velocities remain separate.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import traceback
from contextlib import ExitStack
from dataclasses import asdict
from unittest.mock import patch

import mujoco
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def same(a, b):
    """Exact JSON values including the sign bit of floating-point zero."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return struct.pack("!d", a) == struct.pack("!d", b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


class Audit:
    def __init__(self):
        self.checks = 0
        self.context = "initialization"
        self.maximum_errors = {}
        self.input_sha256 = {}
        self.cases = []
        self.forbidden_calls = []

    def need(self, condition, label):
        self.checks += 1
        if not bool(condition):
            raise AssertionError(f"{self.context}: {label}")

    def close(self, actual, expected, label, atol=2e-10):
        a, b = np.asarray(actual), np.asarray(expected)
        self.need(a.shape == b.shape, label + " shape")
        self.need(np.isfinite(a).all() and np.isfinite(b).all(), label + " finite")
        error = float(np.max(np.abs(a-b))) if a.size else 0.
        self.maximum_errors[label] = max(self.maximum_errors.get(label, 0.), error)
        self.need(error <= atol, f"{label}: max error {error} > {atol}")

    def bits(self, actual, expected, label):
        a, b = np.asarray(actual), np.asarray(expected)
        self.need(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), label)

    def exact(self, actual, expected, label):
        self.need(same(actual, expected), label)

    def remember(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None:
            self.need(digest == expected, "input hash " + str(path))
        self.input_sha256[str(path)] = digest
        return digest

    def manifest(self, folder, filename, require_complete=False):
        manifest_path = folder / filename
        self.remember(manifest_path)
        manifest = read_json(manifest_path)
        for relative, item in manifest.items():
            path = (folder / relative).resolve()
            self.need(path.is_relative_to(folder.resolve()), "manifest path containment")
            self.need(path.stat().st_size == item["bytes"], "manifest size " + relative)
            self.remember(path, item["sha256"])
        if require_complete:
            actual = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and p != manifest_path}
            self.need(set(manifest) == actual, "complete manifest covers every file")

    def forbid(self, name):
        def blocked(*_args, **_kwargs):
            self.forbidden_calls.append(name)
            raise RuntimeError("offline audit forbids " + name)
        return blocked


def validate_native(a, native, diagnostics, trace, case, model, wheel_bodies, g1):
    count = len(trace)
    a.need(len(native) == 5*count, "five native entries per control interval")
    last_end, summed_dt = 0., 0.
    maximum_utilization, unsupported, max_wheel_torque = 0., 0, 0.
    net_moments = []
    for n, entry in enumerate(native):
        k, sub = divmod(n, 5)
        a.context = f"{case['name']} native {n}"
        a.need(entry["returned"] is True, "native returned")
        a.close(entry["start_time_s"], last_end, "native continuous time", 1e-12)
        a.close(entry["actual_dt_s"], .002, "native dt", 1e-12)
        a.close(entry["end_time_s"]-entry["start_time_s"], entry["actual_dt_s"], "native clock duration", 1e-12)
        last_end = entry["end_time_s"]
        summed_dt += entry["actual_dt_s"]
        a.bits(np.array(entry["ctrl_nm"]), np.array(diagnostics[k]["applied_torque_nm"][sub]), "native ctrl vs applied torque bits")
        max_wheel_torque = max(max_wheel_torque, max(abs(entry["ctrl_nm"][i]) for i in (3, 7, 11, 15)))
        a.bits(np.array(entry["wrench_world"]), g1.prescribed_wrench(case, k), "native prescribed raw wrench bits")
        sample = trace[k]["physics_wrench_samples"][sub]
        for key in ("start_time_s", "actual_dt_s", "wrench_world"):
            a.exact(entry[key], sample[key], "trace native sample " + key)
        c = entry["contacts"]
        a.need(c["sampling"] == "native_step_solved_cache_not_synchronized_endpoint", "native cache phase")
        a.close(c["max_horizontal_normal"], 0., "native horizontal normal", 1e-12)
        a.close(c["max_vertical_normal_error"], 0., "native vertical normal", 1e-12)
        a.need(c["active_nonwheel_contacts"] == 0, "native no nonwheel load")
        counts = [0]*4
        force, moment, normal, tangent = (np.zeros(3) for _ in range(4))
        for contact in c["contacts"]:
            wheel = contact["wheel"]
            a.need(wheel in range(4) and model.geom_bodyid[contact["robot_geom_id"]] == wheel_bodies[wheel], "native wheel body identity")
            counts[wheel] += 1
            normal_axis = np.array(contact["normal_terrain_to_robot_world"])
            a.close(normal_axis, [0., 0., 1.], "native terrain-to-robot normal", 1e-12)
            f, fn, ft = (np.array(contact[v]) for v in ("force_world_n", "normal_force_world_n", "tangent_force_world_n"))
            a.close(f, fn+ft, "native force decomposition", 1e-12)
            a.close(fn, float(f @ normal_axis)*normal_axis, "native normal projection", 1e-12)
            a.need(float(fn[2]) >= -1e-10, "native nonnegative normal force")
            # MuJoCo elliptic/pyramidal contact conventions are not re-solved.
            # This archived-force ratio is a diagnostic, not a hard Coulomb pass gate.
            if fn[2] > 1e-9:
                maximum_utilization = max(maximum_utilization, float(np.linalg.norm(ft)/(.9*fn[2])))
            force += f
            normal += fn
            tangent += ft
            moment += np.cross(np.array(contact["pos_world_m"])-c["reference_world_m"], f)+contact["torque_world_nm"]
        a.exact(c["active_contacts_by_wheel"], counts, "native wheel contact counts")
        a.need(c["geometric_wheel_contact_count"] >= sum(counts), "native geometric vs active contacts")
        unsupported += not any(counts)
        a.close(c["summed_normal_force_world_n"], normal, "native total normal", 1e-10)
        a.close(c["summed_tangent_force_world_n"], tangent, "native total tangent", 1e-10)
        a.close(c["total_wrench_world_6"], np.r_[force, moment], "native force and moment summation", 1e-10)
        net_moments.append(float(moment[2]))
    return {"native_calls": len(native), "actual_time_s": last_end, "summed_actual_dt_s": summed_dt,
        "native_substeps_without_active_wheel_contact": unsupported,
        "maximum_saved_force_tangent_over_mu_normal": maximum_utilization,
        "peak_abs_actual_wheel_torque_nm": max_wheel_torque,
        "native_pulse_yaw_moment_mean_nm": float(np.mean(net_moments[1000:1250])) if case["command"]["yaw_pulse"] else None}


def validate_endpoint(a, c, k, model, data, base, wheel_bodies, leg_dofs, wheel_dofs, jp):
    a.need(c["sampling"] == "synchronized_endpoint_not_control_average", "endpoint cache phase")
    a.close(c["measurement_time_s"], .01*k, "endpoint measurement time", 2e-11)
    a.close(c["reference_world_m"], data.xpos[base], "endpoint reference position", 1e-12)
    counts = [0]*4
    forces = np.zeros((4, 3))
    force, moment = np.zeros(3), np.zeros(3)
    leg_velocity, wheel_velocity, base_velocity = (np.zeros(model.nv) for _ in range(3))
    leg_velocity[leg_dofs] = data.qvel[leg_dofs]
    wheel_velocity[wheel_dofs] = data.qvel[wheel_dofs]
    base_velocity[:6] = data.qvel[:6]
    for contact in c["contacts"]:
        wheel = contact["wheel_index"]
        a.need(wheel in range(4) and contact["robot_body_id"] == wheel_bodies[wheel], "endpoint wheel body")
        a.need(model.geom_bodyid[contact["robot_geom_id"]] == wheel_bodies[wheel] and contact["terrain_body_id"] == 0 and contact["terrain_geom_id"] == 0, "endpoint geom identities")
        counts[wheel] += 1
        n = np.array(contact["normal_world"])
        a.close(n, [0., 0., 1.], "endpoint normal", 1e-12)
        mujoco.mj_jac(model, data, jp, None, np.array(contact["pos_world_m"]), wheel_bodies[wheel])
        velocity = jp @ data.qvel
        for field, value in (("robot_point_velocity_world_mps", velocity), ("relative_velocity_world_mps", velocity),
            ("leg_dof_velocity_world_mps", jp @ leg_velocity), ("wheel_dof_velocity_world_mps", jp @ wheel_velocity),
            ("free_base_velocity_world_mps", jp @ base_velocity), ("terrain_point_velocity_world_mps", np.zeros(3))):
            a.close(contact[field], value, "endpoint " + field, 2e-12)
        tangent = velocity-(velocity @ n)*n
        a.close(contact["tangential_relative_velocity_world_mps"], tangent, "endpoint tangent velocity", 2e-12)
        a.close(contact["tangential_relative_speed_mps"], np.linalg.norm(tangent), "endpoint tangent speed", 2e-12)
        a.need(contact["normal_force_n"] >= -1e-10, "endpoint nonnegative normal load")
        f = np.array(contact["force_world_n"])
        a.close(float(f @ n), contact["normal_force_n"], "endpoint normal load projection", 1e-10)
        a.close(np.linalg.norm(f-(f @ n)*n), contact["tangential_force_magnitude_n"], "endpoint tangent force", 1e-10)
        force += f
        forces[wheel] += f
        moment += np.cross(np.array(contact["pos_world_m"])-c["reference_world_m"], f)+contact["torque_world_nm"]
    a.exact(c["contact_count_by_wheel"], counts, "endpoint counts")
    a.close(c["wheel_force_world_n"], forces, "endpoint wheel forces", 1e-10)
    a.close(c["total_wrench_world_6"], np.r_[force, moment], "endpoint total wrench", 1e-10)


def audit_case(a, folder, baseline, case, gates, model, constants, g1):
    names, prefixes, limits, lows, highs, speeds = constants
    a.context = case["name"]
    a.manifest(folder, "complete_manifest.json", require_complete=True)
    a.manifest(baseline, "complete_manifest.json", require_complete=True)
    z, bz = arrays(folder/"states.npz"), arrays(baseline/"states.npz")
    trace, diagnostics, compensation, native = [rows(folder/f) for f in
        ("trace.jsonl.gz", "plane_diagnostics.jsonl.gz", "turn_compensation.jsonl.gz", "native_physics_entries.jsonl.gz")]
    bt, bd, bn = [rows(baseline/f) for f in ("trace.jsonl.gz", "plane_diagnostics.jsonl.gz", "native_physics_entries.jsonl.gz")]
    summary, original = read_json(folder/"summary.json"), read_json(baseline/"summary.json")
    candidate_summary = read_json(folder/"candidate_summary.json")
    a.need(case["max_transitions"] == len(trace) == len(diagnostics) == len(compensation) == 800, "fixed complete intervals")
    for key, shape in {"qpos": (801, 23), "qvel": (801, 22), "observations": (801, 85), "truth_positions_world_m": (801, 3), "requested_actions": (800, 8), "applied_actions": (800, 8)}.items():
        a.need(z[key].shape == shape and np.isfinite(z[key]).all(), "state shape/finite " + key)
    a.need(z["observations"].dtype == np.float32, "observation dtype")
    a.need(not np.any(z["requested_actions"]) and not np.any(z["applied_actions"]), "zero residual all intervals")
    memory = read_json(folder/"initial_controller_memory.json")
    a.exact(memory, read_json(baseline/"initial_controller_memory.json"), "reset memory paired")
    a.exact(memory["wheel_integral_nm"], [0., 0., 0., 0.], "reset zero PI")
    meta = read_json(folder/"episode_metadata.json")["episode_metadata"]
    bmeta = read_json(baseline/"episode_metadata.json")["episode_metadata"]
    for key in ("collision_terrain", "terrain", "requested_terrain_config", "controller_parameters", "actuator", "domain", "provider", "spawn_position_m", "source_schema", "command_seed", "measurement_seed", "control_dt_s", "physics_dt_s"):
        a.exact(meta[key], bmeta[key], "paired metadata " + key)
    a.need(meta["controller_schema"] == "d1-turn-wheel-center-velocity-compensation-v1", "candidate controller identity")
    a.need(meta["task_schema"] == meta["heading_task_config"]["task_schema"] == "d1-heading-plane-turn-center-zero-task-v1", "candidate task identity")
    config = meta["heading_task_config"]["turn_center_compensation"]
    for key, expected in {"coefficient": 1., "wheel_radius_m": .087, "original_inner_yaw_cap_rps": .6, "final_wheel_speed_clip_rad_s": 30., "stop_leg_damping": False, "zero_only": True, "external_raw_callback_required": True}.items():
        a.exact(config[key], expected, "fixed candidate config " + key)
    a.need(meta["collision_terrain"]["mujoco_version"] == mujoco.__version__, "MuJoCo version identity")
    a.exact(meta["controller_parameters"], {"attitude_feedback_scale": 1., "leg_feedback_scale": 1., "wheel_ki": 3., "wheel_kp": 2.2, "yaw_feedback_gain": 4.}, "original controller parameters")
    a.need(meta["provider"]["kind"] == "oracle" and meta["actuator"]["delay_steps"] == 0 and meta["actuator"]["gain"] == 1. and meta["actuator"]["time_constant_s"] == 0., "oracle ideal actuator")
    base = model.body("base_link").id
    wheel_bodies = [model.body(p+"_foot").id for p in prefixes]
    dofs = np.array([model.jnt_dofadr[model.joint(n).id] for n in names]).reshape(4, 4)
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    leg_dofs, wheel_dofs = dofs[:, :3].ravel(), dofs[:, 3]
    local_wheels = np.array([3, 7, 11, 15])
    data, jp, velocity = mujoco.MjData(model), np.zeros((3, model.nv)), np.zeros(6)
    geometries = []
    for k in range(801):
        a.context = f"{case['name']} saved state {k}"
        data.qpos[:] = z["qpos"][k]
        data.qvel[:] = z["qvel"][k]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        a.need(data.time == 0., "kinematic audit time remains zero")
        a.bits(data.qpos, z["qpos"][k], "kinematics does not mutate qpos")
        a.bits(data.qvel, z["qvel"][k], "kinematics does not mutate qvel")
        rotation = data.xmat[base].reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1., 1.)))
        horizontal = np.array([np.cos(yaw), np.sin(yaw), 0.])
        lateral = (data.xpos[wheel_bodies]-data.xpos[base]) @ np.array([-np.sin(yaw), np.cos(yaw), 0.])
        u, u_body = np.zeros(4), np.zeros(4)
        for leg, body in enumerate(wheel_bodies):
            mujoco.mj_jacBody(model, data, jp, None, body)
            relative = jp[:, dofs[leg, :3]] @ data.qvel[dofs[leg, :3]]
            u[leg], u_body[leg] = horizontal @ relative, rotation[:, 0] @ relative
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base, velocity, 0)
        body_vx, body_yaw = float(rotation[:, 0] @ velocity[3:]), float(rotation[:, 2] @ velocity[:3])
        a.close(z["truth_positions_world_m"][k], data.xpos[base], "saved visible base origin", 1e-12)
        geometries.append({"yaw": yaw, "body_vx": body_vx, "body_yaw": body_yaw, "u": u, "u_body": u_body, "lateral": lateral})
        if k:
            row = trace[k-1]
            a.close(row["body_forward_mps"], body_vx, "endpoint body COM forward", 2e-12)
            a.close(row["heading_task"]["truth_heading_after"], yaw, "endpoint actual yaw", 2e-12)
            a.close(row["heading_task"]["truth_yaw_rate_after_rps"], body_yaw, "endpoint actual body yaw rate", 2e-12)
            a.close(row["actual_roll_pitch_rad"], [roll, pitch], "endpoint roll pitch", 2e-12)
            a.close(row["metrics"]["height_error_m"], data.xpos[base, 2]-case["command"]["height_m"], "raw endpoint height error", 2e-12)
            validate_endpoint(a, diagnostics[k-1]["endpoint_contacts"], k, model, data, base, wheel_bodies, leg_dofs, wheel_dofs, jp)
    reference = float(read_json(folder/"episode_metadata.json")["heading_reference_initial"]["heading_rad"])
    a.close(reference, geometries[0]["yaw"], "initial raw reference")
    previous_integral = np.zeros(4)
    active_ticks, boundaries, independently_scored_rows = [], {}, copy.deepcopy(trace)
    for k, (row, d, c) in enumerate(zip(trace, diagnostics, compensation)):
        a.context = f"{case['name']} execution {k}"
        raw = asdict(g1.command_at_tick(case["command"], k))
        pre, end, record = geometries[k], geometries[k+1], c["record"]
        a.need(row["tick"] == d["tick"] == c["tick"] == k and row["endpoint_tick"] == d["endpoint_tick"] == c["endpoint_tick"] == k+1, "T to T+1 indices")
        a.exact(row["heading_task"]["user_command_before"], raw, "frozen raw command")
        a.exact(d["raw_user_command"], raw, "diagnostic raw command")
        a.exact(d["servo_command"], row["heading_task"]["servo_command_before"], "servo phase identity")
        for field, key in (("raw_forward_mps", "forward_velocity_mps"), ("raw_yaw_rate_rps", "yaw_rate_rps")):
            a.exact(record[field], raw[key], "execution record " + field)
        a.exact(record["servo_forward_mps"], raw["forward_velocity_mps"], "raw forward preserved")
        a.exact(d["servo_command"]["forward_velocity_mps"], raw["forward_velocity_mps"], "servo forward no release")
        a.need(c["raw_callback_count_after_prepare"] == k+2, "one callback per prepared decision plus initial")
        a.close(record["control_time_s"], .01*k, "execution record time", 2e-11)
        active = raw["forward_velocity_mps"] == 0. and raw["yaw_rate_rps"] != 0.
        a.need(record["active"] is active and record["enabled"] is True, "execution gate snapshot")
        if active:
            active_ticks.append(k)
        heading_error = wrap(reference-pre["yaw"])
        servo_unclipped = raw["yaw_rate_rps"]+2.*heading_error-.4*(pre["body_yaw"]-raw["yaw_rate_rps"])
        servo = float(np.clip(servo_unclipped, -1., 1.))
        h = row["heading_task"]
        a.close(h["reference_heading_before"], reference, "raw integrated reference before", 2e-11)
        a.close(h["heading_error_before"], heading_error, "original outer heading error", 2e-11)
        a.close(h["servo_unclipped_yaw_rate_before_rps"], servo_unclipped, "original outer yaw law", 2e-10)
        a.close([d["servo_command"]["yaw_rate_rps"], record["servo_yaw_rate_rps"]], [servo, servo], "original outer yaw clip", 2e-10)
        effective = float(np.clip(servo+4.*(servo-pre["body_yaw"]), -.6, .6))
        a.close([d["effective_yaw_request_rps"], record["effective_yaw_request_rps"]], [effective, effective], "original inner yaw law", 2e-10)
        a.need(d["inner_yaw_limit_occupied"] == (abs(effective) >= .6-1e-12), "inner cap diagnostic")
        base_unclipped = (raw["forward_velocity_mps"]-effective*pre["lateral"])/.087
        correction = pre["u"]/.087 if active else np.zeros(4)
        corrected_unclipped = base_unclipped+correction
        target = np.clip(corrected_unclipped, -30., 30.)
        for field, expected in (("leg_center_forward_mps", pre["u"]), ("correction_unclipped_rad_s", correction),
            ("base_unclipped_rad_s", base_unclipped), ("base_target_rad_s", np.clip(base_unclipped, -30., 30.)),
            ("corrected_unclipped_rad_s", corrected_unclipped), ("corrected_target_rad_s", target)):
            a.close(record[field], expected, "independent " + field, 2e-10)
        if not active:
            a.bits(np.array(record["correction_unclipped_rad_s"]), np.zeros(4), "inactive exact positive-zero correction")
        a.exact(record["target_clipped"], (target != corrected_unclipped).tolist(), "target final clip mask")
        a.close(d["wheel_target_rad_s"], target, "executed wheel target", 2e-10)
        centered = pre["lateral"]-np.mean(pre["lateral"])
        denominator = float(centered @ centered)
        a.need(denominator > 1e-4, "nondegenerate yaw-fit geometry")
        equivalent = float(-(centered @ (.087*target))/denominator)
        a.close(record["target_equivalent_yaw_rps"], equivalent, "target equivalent wheel yaw", 2e-10)
        a.close(d["leg_forward_velocity_before_mps"], pre["u_body"], "original diagnostic body-x leg velocity", 2e-12)
        a.close(d["body_forward_before_mps"], pre["body_vx"], "pre body COM forward", 2e-12)
        a.close(d["body_yaw_rate_before_rps"], pre["body_yaw"], "pre body yaw", 2e-12)
        qd = z["qvel"][k, dofs.ravel()]
        a.close(d["wheel_rolling_before_mps"], .087*np.mean(qd[local_wheels]), "pre wheel rolling mean", 2e-12)
        error = target-qd[local_wheels]
        before = np.array(d["wheel_integral_before_nm"])
        a.bits(before, previous_integral, "PI memory continuity bits")
        trial = np.clip(before+3.*error*.01, -4., 4.)
        trial_request = 2.2*error+trial
        accept = (np.abs(trial_request) <= limits[local_wheels]) | (trial_request*error < 0)
        after = np.where(accept, trial, before)
        a.close(d["wheel_error_before_rad_s"], error, "wheel PI error", 2e-10)
        a.close(d["wheel_integral_after_nm"], after, "wheel PI antiwindup", 2e-10)
        unlimited = np.array(d["unlimited_torque_nm"])
        a.close(unlimited[local_wheels], 2.2*error+after, "wheel unlimited PI request", 5e-10)
        previous_integral = np.array(d["wheel_integral_after_nm"])
        safe = np.clip(unlimited, -limits, limits)
        qj = z["qpos"][k, qadr]
        outward = ((qj >= highs) & (safe > 0)) | ((qj <= lows) & (safe < 0)) | ((np.abs(qd) >= speeds) & (safe*qd > 0))
        safe[outward] = 0.
        a.bits(np.array(d["requested_torque_nm"]), safe, "all-axis original safety bits")
        a.exact(d["torque_protected"], (safe != unlimited).tolist(), "safety mask")
        a.bits(np.array(d["applied_torque_nm"]), np.tile(safe, (5, 1)), "five held ideal actuator torques bits")
        a.need(len(row["physics_wrench_samples"]) == 5, "five trace native samples")
        actual_dt = sum(sample["actual_dt_s"] for sample in row["physics_wrench_samples"])
        a.close(h["actual_dt_s"], actual_dt, "raw interval actual dt", 1e-12)
        reference = wrap(reference+actual_dt*raw["yaw_rate_rps"])
        endpoint_error = wrap(end["yaw"]-reference)
        a.close(h["reference_heading_after"], reference, "raw integrated reference after", 2e-11)
        a.close(row["heading_error_rad"], endpoint_error, "raw endpoint heading score", 2e-11)
        a.close(row["user_forward_error_mps"], end["body_vx"]-raw["forward_velocity_mps"], "raw endpoint speed score", 2e-12)
        a.close(h["user_yaw_rate_error_after"], end["body_yaw"]-raw["yaw_rate_rps"], "raw endpoint yaw-rate score", 2e-12)
        a.need(row["phase"] == g1.phase_name(case, k), "original raw phase")
        a.need(not row["terminated"] and row["truncated"] == (k == 799), "only final time-limit truncation")
        for field in ("requested_action", "applied_action"):
            a.bits(np.array(row[field]), z[field+"s"][k], "trace action/state archive bits")
        # Re-score with independently reconstructed body/raw values. Other
        # original score inputs (contact counters, reward terms) stay archived.
        independent = independently_scored_rows[k]
        independent["body_forward_mps"] = end["body_vx"]
        independent["user_forward_error_mps"] = end["body_vx"]-raw["forward_velocity_mps"]
        independent["heading_error_rad"] = endpoint_error
        independent["heading_task"]["user_yaw_rate_error_after"] = end["body_yaw"]-raw["yaw_rate_rps"]
        if k in (199, 200, 249, 250):
            boundaries[str(k)] = {"active": active, "raw_yaw_rps": raw["yaw_rate_rps"], "callback_count_after_prepare": c["raw_callback_count_after_prepare"], "peak_abs_correction_rad_s": float(np.max(np.abs(correction)))}
    a.need(active_ticks == (list(range(200, 250)) if case["command"]["yaw_pulse"] else []), "exact active tick set")
    metric, gate_result = g1.gates_and_metrics(case, trace, z["truth_positions_world_m"], gates)
    for key, value in metric.items():
        a.exact(summary[key], value, "original raw metric " + key)
    a.exact(summary["gates"], gate_result, "original gates recomputed")
    independent_metric, independent_gates = g1.gates_and_metrics(case, independently_scored_rows, z["truth_positions_world_m"], gates)
    for key in ("heading_peak_rad", "heading_rmse_rad", "velocity_rmse_mps", "user_yaw_rate_rmse_rps"):
        a.close(independent_metric[key], metric[key], "saved-state independent score " + key, 2e-10)
    a.exact(independent_gates, gate_result, "saved-state independent gates")
    initial_hash = hashlib.sha256(z["qpos"][0].tobytes()+z["qvel"][0].tobytes()+z["observations"][0].tobytes()).hexdigest()
    a.need(summary["initial_state_observation_sha256"] == initial_hash, "initial state/observation SHA")
    native_result = validate_native(a, native, diagnostics, trace, case, model, wheel_bodies, g1)
    receipt = read_json(folder/"plane_execution_receipt.json")
    native_receipt = read_json(folder/"native_entry_receipt.json")
    for key, expected in {"actual_physics_substeps_from_clock": 4000, "actual_completed_control_intervals": 800, "partial_interval_physics_substeps": 0, "recorded_diagnostic_intervals": 800, "error": None}.items():
        a.exact(receipt[key], expected, "execution receipt " + key)
    for key, expected in {"observed_native_calls": 4000, "returned_native_calls": 4000, "failure": None, "archival_error": None, "nonfinite_value_count": 0, "no_retry_or_padding": True}.items():
        a.exact(native_receipt[key], expected, "native receipt " + key)
    a.close(native_receipt["summed_actual_dt_s"], native_result["summed_actual_dt_s"], "native receipt time", 1e-12)
    a.close(summary["actual_seconds"], native_result["actual_time_s"], "summary native time", 1e-12)
    a.need(summary["actual_transitions"] == 800 and summary["actual_observed_physics_substeps"] == 4000 and summary["wrench_zero_after_episode"] is True, "original summary physical ledger")
    execution = arrays(folder/"execution_states.npz")
    a.need(execution["tag"].tolist() == ["reset"]+["completed_step"]*800+["close"], "execution archive no partial or retry tags")
    for key in ("qpos", "qvel"):
        a.bits(execution[key][:801], z[key], "execution T+1 archive " + key)
        a.bits(execution[key][-1], z[key][-1], "close final archive " + key)
    expected_times = np.r_[0., [native[5*k+4]["end_time_s"] for k in range(800)], native[-1]["end_time_s"]]
    a.bits(execution["time_s"], expected_times, "execution/native time bits")
    a.need(not (folder/"partial_state.npz").exists(), "complete run no partial state")
    prefix = 200 if case["command"]["yaw_pulse"] else 800
    for key in z:
        states = key in ("qpos", "qvel", "truth_positions_world_m")
        count = prefix+int(states or (key == "observations" and prefix == 800))
        a.bits(z[key][:count], bz[key][:count], "paired state/action bytes " + key)
    # The trace at execution 199 contains NEXT observation metadata (tick 200),
    # which is legitimately candidate-specific. Compare executed physics and
    # state fields, omit only this explicitly named next-observation metadata.
    for k in range(prefix):
        a.exact(diagnostics[k], bd[k], "paired complete controller/contact diagnostic")
        left, right = copy.deepcopy(trace[k]), copy.deepcopy(bt[k])
        for value in (left, right):
            for key in list(value["heading_task"]):
                if key.startswith("appended_observation_"):
                    del value["heading_task"][key]
        a.exact(left, right, "paired executed trace")
    for n in range(5*prefix):
        a.exact(native[n], bn[n], "paired complete native record")
    if prefix == 800:
        for key, value in original.items():
            if key != "model":
                a.exact(summary[key], value, "noop original summary field " + key)
        a.exact(trace, bt, "noop full trace including next observations")
    for key, value in summary.items():
        a.exact(candidate_summary[key], value, "candidate summary preserves original " + key)
    a.need(candidate_summary["plant_semantics_valid"] is True, "candidate plane semantics")
    expected_diagnostics = {"active_intervals": len(active_ticks),
        "peak_abs_correction_rad_s": max(abs(v) for c in compensation for v in c["record"]["correction_unclipped_rad_s"]),
        "peak_abs_target_rad_s": max(abs(v) for c in compensation for v in c["record"]["corrected_target_rad_s"]),
        "peak_abs_target_equivalent_yaw_rps": max(abs(c["record"]["target_equivalent_yaw_rps"]) for c in compensation),
        "target_clipped_wheel_intervals": sum(sum(c["record"]["target_clipped"]) for c in compensation),
        "torque_protected_joint_intervals": sum(sum(d["torque_protected"]) for d in diagnostics), "raw_callback_count": 801}
    a.exact(candidate_summary["turn_compensation_diagnostics"], expected_diagnostics, "candidate diagnostics independently aggregated")
    peak = max(trace, key=lambda r: abs(r["heading_error_rad"]))
    return {"case": case["name"], "audit_passed": True, "original_gates": gate_result,
        "actual_transitions": 800, "native": native_result, "boundaries": boundaries,
        "paired_transition_prefix": prefix, "paired_state_rows": prefix+1,
        "paired_observation_rows": prefix+int(prefix == 800), "diagnostics": expected_diagnostics,
        "baseline_heading_peak_rad": original["heading_peak_rad"], "heading_peak_rad": metric["heading_peak_rad"],
        "heading_peak_endpoint_tick": peak["endpoint_tick"],
        "raw_reference_at_peak_rad": peak["heading_task"]["reference_heading_after"],
        "actual_heading_at_peak_rad": peak["heading_task"]["truth_heading_after"],
        "all_noop_original_summary_fields_except_model_preserved": prefix == 800}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="NEW JSON report path; never overwritten")
    parser.add_argument("--repo", type=Path, default=Path("/home/lyh/wheel-legged-control-lab"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    a = Audit()
    report = {"schema": "d1-plane-turn-center-independent-offline-audit-v1", "passed": False,
        "new_physics_steps": 0, "new_control_transitions": 0, "controller_compute_calls": 0,
        "fresh_contact_solves": 0, "mujoco_version": mujoco.__version__}
    try:
        repo, candidate, baseline = args.repo.resolve(), args.candidate.resolve(), args.baseline.resolve()
        a.need(not args.output.resolve().is_relative_to(candidate) and not args.output.resolve().is_relative_to(baseline), "output outside immutable inputs")
        for path in (repo, repo/"src", repo/".local-deps"):
            sys.path.insert(0, str(path))
        with ExitStack() as guards:
            for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse", "mj_collision"):
                guards.enter_context(patch.object(mujoco, name, side_effect=a.forbid(name)))
            from wheel_legged_control.d1.model import (D1_JOINT_NAMES, LEG_PREFIXES, JOINT_TORQUE_LIMIT,
                JOINT_POSITION_LOW, JOINT_POSITION_HIGH, JOINT_VELOCITY_LIMIT, build_d1_model)
            from scripts import probe_d1_heading_g1 as g1
            a.remember(__file__)
            a.manifest(candidate, "manifest.json", require_complete=True)
            protocol, summary = read_json(candidate/"protocol.json"), read_json(candidate/"summary.json")
            original_path = repo/"results/d1_driving_stability_development/heading_g1_01/evaluation_protocol.json"
            a.remember(original_path, "cf5dbd51042bfafa163116f4a7fb2a990726d47c5a6000558f28aa6866a664fc")
            original = read_json(original_path)
            fixed_cases = [original["cases"][i] for i in (2, 3, 0, 1)]
            a.exact(protocol["cases"], fixed_cases, "exact original four raw cases")
            a.exact(protocol["original_gates"], original["proposed_gates"], "original gates unchanged")
            a.need(Path(protocol["baseline_source"]).resolve() == baseline, "baseline path identity")
            for key, expected in {"coefficient": 1., "original_inner_yaw_cap_rps": .6, "wheel_speed_clip_rad_s": 30.,
                "maximum_new_control_transitions": 3200, "maximum_new_physics_substeps": 16000,
                "new_training_steps": 0, "stop_damping": False, "release_shaping": False, "default_changed": False}.items():
                a.exact(protocol[key], expected, "protocol fixed scope " + key)
            for path, digest in protocol["input_sha256"].items():
                a.remember(path, digest)
            frozen_path = repo/"results/d1_budget_study/protocol.json"
            a.remember(frozen_path)
            frozen = read_json(frozen_path)["source_sha256"]
            a.need(len(frozen) == 77, "77 frozen inputs")
            for path, digest in frozen.items():
                a.remember(repo/path, digest)
            model = build_d1_model()
            a.need(model.nhfield == 0 and model.nq == 23 and model.nv == 22 and model.nu == 16, "native plane model dimensions")
            a.need(model.geom_type[0] == mujoco.mjtGeom.mjGEOM_PLANE and model.geom_bodyid[0] == 0, "native floor type")
            a.close(model.opt.timestep, .002, "model timestep", 0.)
            a.close(model.geom_friction[0], [.9, .005, .0001], "model floor friction", 0.)
            constants = (D1_JOINT_NAMES, LEG_PREFIXES, JOINT_TORQUE_LIMIT, JOINT_POSITION_LOW, JOINT_POSITION_HIGH, JOINT_VELOCITY_LIMIT)
            for case in fixed_cases:
                a.cases.append(audit_case(a, candidate/case["name"], baseline/case["name"], case, original["proposed_gates"], model, constants, g1))
            for key, expected in {"actual_new_control_transitions": 3200, "actual_new_physics_substeps": 16000,
                "comparison_valid": True, "no_op_stop_failures_preserved": True, "frozen77_unchanged": True,
                "default_changed": False, "full_driving_goal_complete": False}.items():
                a.exact(summary[key], expected, "root summary " + key)
            turn_pass = all(c["original_gates"]["passed"] for c in a.cases[:2])
            a.exact(summary["candidate_passed_both_turn_cases"], turn_pass, "root turn gate conclusion")
            a.need(len(summary["episodes"]) == len(summary["pairs"]) == 4, "root complete episode/pair set")
            for case, episode in zip(fixed_cases, summary["episodes"]):
                a.exact(episode, read_json(candidate/case["name"]/"candidate_summary.json"), "root episode summary identity")
            for path, digest in a.input_sha256.items():
                a.need(sha(path) == digest, "input unchanged after audit " + path)
            a.need(not a.forbidden_calls, "zero prohibited API calls")
            report.update(passed=True, inputs_unchanged=True, frozen77_unchanged=True,
                all_audit_data_times_zero=True, observed_candidate_control_transitions=3200,
                observed_candidate_native_substeps=16000, observed_reused_baseline_control_transitions=3200,
                candidate_passed_both_turn_cases=turn_pass)
    except BaseException as error:
        report.update(error={"type": type(error).__name__, "message": str(error), "context": a.context,
            "traceback": traceback.format_exc()})
    report.update(check_count=a.checks, maximum_absolute_errors=a.maximum_errors,
        prohibited_api_calls=a.forbidden_calls, input_sha256=a.input_sha256, cases=a.cases,
        limitations=["Audit passed means provenance, formulas, scoring and paired execution are consistent; it does not mean original task gates passed.",
            "Wheel PI is independently reconstructed. All 16 safety outputs are reconstructed from archived unlimited requests; unchanged leg PD/support requests are not independently recomputed.",
            "Native forces are actual archived solved-cache records; this audit checks their identities, signs and sums, and never re-solves historical forces.",
            "Endpoint contact point velocities are independently reconstructed at the exact saved endpoint. They are never multiplied by native solved-cache forces as if synchronous.",
            "The imported original scorer is a pure function; independently reconstructed raw errors and original score windows are checked before accepting its gates.",
            "No gain selection, additional candidate, integration, controller compute or discrete-time stability certificate is produced."])
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], "report": str(args.output), "new_physics_steps": 0, "checks": a.checks, "completed_cases": len(a.cases)}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
