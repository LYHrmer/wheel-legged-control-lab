"""Offline readback of one closed 08-E authority and baseline ladder attempt.

This uses saved numeric/contact/force records only. It imports no simulator,
policy, training stack or course worker, and never re-simulates an interval.
Early failure remains an observed failure with a checked completed prefix.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.abc
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

FORBIDDEN = {"mujoco", "glfw", "torch", "stable_baselines3", "gym", "gymnasium"}


class NoPhysics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in FORBIDDEN:
            raise RuntimeError("E readback forbids physics/policy import: " + fullname)


if any(name.partition(".")[0] in FORBIDDEN for name in sys.modules):
    raise RuntimeError("physics/policy already imported into E verifier")
sys.meta_path.insert(0, NoPhysics())
sys.dont_write_bytecode = True
import numpy as np

W = Path(__file__).resolve().parent
CONTRACT_SHA = "475273fd4517263606af7abd07008cdd6811f9992e6affacf6eb758a9ccbddd7"
SCHEMA = "d1-course-e-rl16-authority-speed-08-v1"
FAMILIES = {"floor": 1, "rough": 63, "ramp": 3, "stair": 9, "bump": 13, "jump": 3}
STATE = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation", "time")
NATIVE = ("qpos", "qvel", "ctrl", "qacc_warmstart")
WHEELS = (3, 7, 11, 15)
LEGS = tuple(i for i in range(16) if i not in WHEELS)
TORQUE_LIMIT = np.tile((80.0, 80.0, 80.0, 12.0), 4)
VELOCITY_LIMIT = np.tile((20.0, 20.0, 20.0, 30.0), 4)
POSITION_LOW = np.tile((-.785398, -1.8326, -2.775, -np.inf), 4)
POSITION_HIGH = np.tile((.785398, 3.40339, -.855, np.inf), 4)
LEG_SCALE = np.tile((.12, .25, .25), 4)
HOLD_WINDOWS = ((.4, 255, 355), (.6, 395, 495), (.9, 555, 655),
                (1.2, 715, 815), (1.6, 895, 1095))


def require(condition: bool, label: str) -> None:
    if not condition:
        raise ValueError(label)


def read(path: Path):
    require(path.is_file() and not path.is_symlink(), f"missing regular saved file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def identity(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), f"missing regular source: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def equal(a, b) -> bool:
    left, right = np.asarray(a), np.asarray(b)
    if left.shape != right.shape:
        return False
    if left.dtype.kind in "biuf" and right.dtype.kind in "biuf":
        if left.dtype != right.dtype:
            if isinstance(a, np.ndarray):
                right = right.astype(left.dtype)
            elif isinstance(b, np.ndarray):
                left = left.astype(right.dtype)
            else:
                left, right = left.astype(np.float64), right.astype(np.float64)
        return left.tobytes() == right.tobytes()
    return bool(np.array_equal(left, right))


def close(a, b, *, atol: float = 1e-10) -> bool:
    aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return aa.shape == bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all() and bool(
        np.allclose(aa, bb, rtol=0.0, atol=atol)
    )


def check_source(run: Path) -> dict:
    session = read(run / "session.json")
    require(session["schema"] == SCHEMA and session["phase"] == "E"
            and session["output_directory"] == str(run)
            and session["segments"] == [800, 1600]
            and session["control_limit"] == 2400
            and session["normal_native_limit"] == 12000
            and session["compiler_native_limit"] == 2
            and session["wallclock_limit_s"] == 600
            and session["seed"] == 88401
            and session["contract_sha256"] == CONTRACT_SHA
            and session["retry_permitted"] is False,
            "E source session/budget differs from frozen contract")
    frozen = session["source_hashes"]
    require(isinstance(frozen, dict) and len(frozen) >= 100,
            "E source closure is incomplete")
    for name, expected in frozen.items():
        source = Path(name)
        require(source.is_absolute() and identity(source) == expected,
                f"E source drift: {name}")
    for source in (W / "run_course_e_08.py", W / "launch_course_e_08.py",
                   W / "course_impl08/course_native_guard_08.py",
                   W / "next_rl_pilot_contract_08.md"):
        require(str(source) in frozen, f"critical E source not frozen: {source}")
    require(identity(Path(session["contract_path"]))["sha256"] == CONTRACT_SHA,
            "E contract hash differs")
    go_candidates = [Path(name) for name, expected in frozen.items()
                     if expected["sha256"] == session["go_sha256"]
                     and Path(name).name.startswith("astra_course_e_go_08")
                     and Path(name).suffix == ".json"]
    require(len(go_candidates) == 1,
            "actual E GO cannot be identified uniquely in frozen execution inputs")
    go_path = go_candidates[0]
    require(identity(go_path)["sha256"] == session["go_sha256"],
            "actual E GO hash differs")
    go = read(go_path)
    require(go["decision"] == "GO" and go["contract_sha256"] == CONTRACT_SHA,
            "actual E GO decision differs")
    for name, expected in go["inputs"].items():
        require(identity(Path(name)) == expected, f"E GO bound input drift: {name}")
    return {"source_count": len(frozen), "go_input_count": len(go["inputs"]),
            "actual_go_path": str(go_path),
            "source_hashes_verified": True}


def family(name: str) -> str:
    if name == "floor":
        return "floor"
    for key in ("rough", "ramp", "stair", "bump", "jump"):
        if name.startswith(f"terrain_{key}_"):
            return key
    raise ValueError("unknown compiled course terrain name: " + name)


def check_binding(saved: dict) -> dict:
    geom_bodyid = np.asarray(saved["geom_bodyid"], dtype=np.int64)
    geom_type = np.asarray(saved["geom_type"], dtype=np.int64)
    geom_size = np.asarray(saved["geom_size"], dtype=np.float64)
    geom_contype = np.asarray(saved["geom_contype"], dtype=np.int64)
    geom_conaffinity = np.asarray(saved["geom_conaffinity"], dtype=np.int64)
    n = len(geom_bodyid)
    require(n > 92 and geom_type.shape == (n,)
            and geom_size.shape == (n, 3)
            and geom_contype.shape == geom_conaffinity.shape == (n,),
            "E actual compiled robot/terrain geom arrays differ")
    wheel_map = {int(key): value for key, value in saved["wheel_index_by_body_id"].items()}
    require(len(wheel_map) == 4 and set(wheel_map.values()) == set(range(4))
            and all(body > 0 for body in wheel_map),
            "E actual wheel body binding is not four distinct compiled bodies")
    qpos = np.asarray(saved["joint_qpos_addresses"], dtype=np.int64)
    dof = np.asarray(saved["joint_dof_addresses"], dtype=np.int64)
    joint_ids = np.asarray(saved["joint_ids"], dtype=np.int64)
    actuator_ids = np.asarray(saved["actuator_ids"], dtype=np.int64)
    trnid = np.asarray(saved["actuator_trnid"], dtype=np.int64)
    gear = np.asarray(saved["actuator_gear"], dtype=np.float64)
    require(qpos.shape == dof.shape == joint_ids.shape == actuator_ids.shape == (16,)
            and len(set(qpos)) == len(set(dof)) == 16
            and np.all(qpos >= 7) and np.all(dof >= 6)
            and np.array_equal(trnid[actuator_ids, 0], joint_ids)
            and np.all(gear[actuator_ids, 0] == 1.0)
            and np.all(gear[actuator_ids, 1:] == 0.0),
            "E actual joint/actuator direct-torque binding differs")
    ipos = np.asarray(saved["base_body_ipos"], dtype=np.float64)
    mass = np.asarray(saved["body_mass"], dtype=np.float64)
    require(ipos.shape == (3,) and np.isfinite(ipos).all()
            and 0 < int(saved["base_body_id"]) < len(mass)
            and np.isfinite(mass).all() and close(float(np.sum(mass)),
                                                   saved["nominal_total_mass_kg"], atol=1e-8),
            "E compiled base COM offset/body mass differs")
    return {
        "geom_bodyid": geom_bodyid, "geom_type": geom_type,
        "geom_size": geom_size, "geom_contype": geom_contype,
        "geom_conaffinity": geom_conaffinity, "wheel_map": wheel_map,
        "qpos_addresses": qpos, "dof_addresses": dof,
        "actuator_ids": actuator_ids,
        "base_body_ipos": ipos, "base_body_id": int(saved["base_body_id"]),
    }


def geom_map(folder: Path, binding: dict) -> dict[int, dict]:
    manifest = read(folder / "geometry_manifest.json")
    rows = manifest["world_collision_geoms"]
    require(len(rows) == 92 and manifest["control_dt_s"] == .01
            and manifest["native_dt_s"] == .002,
            "actual compiled course layout/timing differs")
    require(Counter(family(row["name"]) for row in rows) == FAMILIES,
            "compiled course terrain family counts differ")
    mapping = {}
    kinds: dict[str, set[int]] = {"plane": set(), "box": set()}
    for row in rows:
        gid = row["geom_id"]
        require(type(gid) is int and gid not in mapping
                and 0 <= gid < len(binding["geom_bodyid"])
                and row["body_id"] == 0 and row["collision"] is True
                and binding["geom_bodyid"][gid] == 0
                and (binding["geom_contype"][gid] != 0
                     or binding["geom_conaffinity"][gid] != 0)
                and close(binding["geom_size"][gid], row["size_m"])
                and row["type"] == ("plane" if row["name"] == "floor" else "box"),
                "compiled course terrain identity invalid")
        quat = np.asarray(row["quaternion_wxyz"], dtype=np.float64)
        require(close(quat @ quat, 1.0, atol=1e-12)
                and np.isfinite(row["position_m"]).all()
                and np.isfinite(row["size_m"]).all(),
                "compiled course primitive has invalid transform")
        kinds[row["type"]].add(int(binding["geom_type"][gid]))
        mapping[gid] = row
    require(len(kinds["plane"]) == len(kinds["box"]) == 1
            and kinds["plane"] != kinds["box"],
            "actual compiled terrain plane/box geom types are not distinct")
    return mapping


def _rotation(quat) -> np.ndarray:
    w, x, y, z = map(float, quat)
    return np.asarray(((1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)),
                       (2*(x*y+w*z), 1 - 2*(x*x+z*z), 2*(y*z-w*x)),
                       (2*(x*z-w*y), 2*(y*z+w*x), 1 - 2*(x*x+y*y))))


def _roll_pitch_deg(qpos) -> tuple[float, float]:
    q = np.asarray(qpos, dtype=np.float64)
    require(q.ndim == 1 and q.size >= 7 and np.isfinite(q).all(),
            "native qpos lacks a finite free-base quaternion")
    w, x, y, z = q[3:7]
    require(abs(float(w*w+x*x+y*y+z*z)-1.0) <= 1e-6,
            "native free-base quaternion norm differs")
    return (
        math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))),
        math.degrees(math.asin(float(np.clip(2*(w*y-z*x), -1, 1)))),
    )


def _body_com_velocity(qpos, qvel, binding: dict) -> np.ndarray:
    """Freejoint origin velocity plus rotating compiled inertial COM offset."""
    q, v = np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)
    rotation = _rotation(q[3:7])
    world_omega = rotation @ v[3:6]
    world_com = v[:3] + np.cross(world_omega, rotation @ binding["base_body_ipos"])
    return rotation.T @ world_com


def check_contact(contact: dict, geoms: dict[int, dict], binding: dict) -> tuple[bool, str | None]:
    """Independent local rotated-box cone check; never calls engine geometry."""
    g1, g2 = contact["geom1"], contact["geom2"]
    require(0 <= g1 < len(binding["geom_bodyid"])
            and 0 <= g2 < len(binding["geom_bodyid"]),
            "native contact geom outside actual compiled model")
    t1, t2 = g1 in geoms, g2 in geoms
    require(not (t1 and t2), "native terrain-terrain contact")
    frame = np.asarray(contact["frame_world"], dtype=np.float64)
    require(frame.shape == (3, 3) and np.isfinite(frame).all()
            and close(frame @ frame.T, np.eye(3))
            and abs(float(np.linalg.det(frame))-1) <= 1e-10,
            "native contact frame invalid")
    for key in ("position_world_m", "friction", "solref", "solreffriction", "solimp",
                "geom1_xpos", "geom2_xpos", "geom1_xmat", "geom2_xmat"):
        require(np.isfinite(np.asarray(contact[key], dtype=np.float64)).all(),
                f"native contact cache nonfinite: {key}")
    if contact["efc_address"] >= 0:
        wrench = np.asarray(contact["contact_force_local"], dtype=np.float64)
        require(wrench.shape == (6,) and np.isfinite(wrench).all()
                and contact["force_source"] == "mj_contactForce_actual_native_cache",
                "actual native contact force record absent")
    if not (t1 or t2):
        return False, None
    terrain = g1 if t1 else g2
    robot = g2 if t1 else g1
    row = geoms[terrain]
    actual_body = int(binding["geom_bodyid"][robot])
    require(contact["robot_geom_id"] == robot
            and contact["robot_body_id"] == actual_body
            and contact["robot_wheel_index"] == binding["wheel_map"].get(actual_body),
            "native contact robot/wheel binding differs from compiled model")
    outward = frame[0] if t1 else -frame[0]
    pos = np.asarray(contact["position_world_m"], dtype=np.float64)
    tol = (abs(float(contact["distance_m"]))
           + max(float(contact["geom1_compiled_margin"]),
                 float(contact["geom2_compiled_margin"]),
                 float(contact["includemargin_m"])) + 1e-7)
    if row["name"] == "floor":
        require(close(outward, (0.0, 0.0, 1.0)) and abs(float(pos[2])) <= tol,
                "native floor contact normal/point invalid")
    else:
        rotation = _rotation(row["quaternion_wxyz"])
        # Native world cache is checked against the actual compiled static geom.
        require(close(contact[f"geom{1 if t1 else 2}_xpos"], row["position_m"])
                and close(contact[f"geom{1 if t1 else 2}_xmat"], rotation),
                "native rotated terrain cache differs from compiled primitive")
        local = rotation.T @ (pos - np.asarray(row["position_m"], dtype=np.float64))
        normal = rotation.T @ outward
        half = np.asarray(row["size_m"], dtype=np.float64)
        require(np.isfinite(local).all() and np.isfinite(normal).all()
                and np.all(local >= -half-tol) and np.all(local <= half+tol),
                "native rotated box contact is outside actual extents")
        support = np.zeros(3)
        for axis in range(3):
            if abs(float(normal[axis])) > 1e-6:
                face = math.copysign(float(half[axis]), float(normal[axis]))
                if abs(float(local[axis])-face) <= tol:
                    support[axis] = normal[axis]
        require(np.any(support) and np.linalg.norm(normal-support) <= .0021,
                "native rotated box normal cone invalid")
    require(contact["terrain_geom_id"] == terrain
            and contact["terrain_geom_name"] == row["name"]
            and contact["terrain_family"] == family(row["name"]),
            "native contact terrain identity differs")
    load = False
    if contact["efc_address"] >= 0:
        force = frame.T @ np.asarray(contact["contact_force_local"][:3], dtype=np.float64)
        if not t1:
            force = -force
        normal_load = float(force @ outward)
        load = contact["robot_wheel_index"] in range(4) and normal_load > 0.0
        require(close(contact["force_on_robot_world_n"], force)
                and close(contact["normal_load_on_robot_n"], normal_load),
                "native contact force transform differs from recorded solver force")
    require(contact["positive_wheel_load"] is load,
            "positive wheel load flag differs from actual bound contact force")
    return load, family(row["name"])


def recompute_controller(calc: dict, action) -> dict[str, np.ndarray]:
    """Independent numerical recomputation from saved consumed inputs."""
    raw_action = np.asarray(action, dtype=np.float64)
    clipped_action = np.clip(raw_action, -1.0, 1.0)
    a = clipped_action.copy()
    if not calc["action_enabled"]:
        a = np.zeros(16)
    q = np.asarray(calc["consumed_joint_position_rad"], dtype=np.float64)
    dq = np.asarray(calc["consumed_joint_velocity_rad_s"], dtype=np.float64)
    nominal = np.asarray(calc["consumed_nominal_joint_target_rad"], dtype=np.float64)
    nominal_wheel = np.asarray(calc["consumed_nominal_wheel_speed_rad_s"], dtype=np.float64)
    support = np.asarray(calc["consumed_support_torque_nm"], dtype=np.float64)
    integral = np.asarray(calc["wheel_integral_before_nm"], dtype=np.float64)
    leg_offset = LEG_SCALE * a[:12]
    wheel_offset = 4.0 * a[12:]
    geometric = nominal.copy()
    geometric[list(LEGS)] += leg_offset
    rate = np.clip(geometric, q - VELOCITY_LIMIT*.01, q + VELOCITY_LIMIT*.01)
    position = np.clip(rate, POSITION_LOW, POSITION_HIGH)
    target = position.copy()
    target[list(WHEELS)] = q[list(WHEELS)]
    leg_pd = 80.0*(target-q)-3.0*dq
    leg_pd[list(WHEELS)] = 0.0
    wheel_request = nominal_wheel + wheel_offset
    wheel_target = np.clip(wheel_request, -30.0, 30.0)
    error = wheel_target-dq[list(WHEELS)]
    candidate = np.clip(integral+3.0*error*.01, -4.0, 4.0)
    candidate_request = 2.2*error+candidate
    commit = ((np.abs(candidate_request) <= TORQUE_LIMIT[list(WHEELS)])
              | ((candidate_request*error) < 0))
    integral_after = np.where(commit, candidate, integral)
    wheel_pi = np.zeros(16)
    wheel_pi[list(WHEELS)] = 2.2*error+integral_after
    base = leg_pd+support+wheel_pi
    rotation = np.asarray(calc["consumed_base_rotation_world_from_body"], dtype=np.float64)
    jac = np.asarray(calc["consumed_foot_jacobian_world"], dtype=np.float64)
    projected = np.zeros((4, 3))
    relative = np.zeros(4)
    damping = np.zeros(16)
    for leg in range(4):
        jx = rotation[:, 0] @ jac[leg, :, :3]
        projected[leg] = jx
        relative[leg] = jx @ dq[4*leg:4*leg+3]
        if calc["leg_longitudinal_damping_active"]:
            damping[4*leg:4*leg+3] = -126.4374005337902*jx*relative[leg]
    common_error = float(np.mean(dq[list(WHEELS)])
                         - float(calc["consumed_body_com_forward_mps"])/.087)
    common_scalar = 2.2*common_error if calc["body_common_p_active"] else 0.0
    common = np.zeros(16)
    common[list(WHEELS)] = common_scalar
    final = base+damping+common
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
        "applied_action": a, "leg_action_offset_rad": leg_offset,
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
        "wheel_speed_error_rad_s": error,
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


def check_nominal_support(saved: dict) -> np.ndarray:
    rpy = np.asarray(saved["consumed_base_rpy_rad"], dtype=np.float64)
    angular = np.asarray(saved["consumed_base_angular_velocity_world"], dtype=np.float64)
    rotation = np.asarray(saved["consumed_base_rotation_world_from_body"], dtype=np.float64)
    foot_offset = np.asarray(saved["consumed_foot_offset_world"], dtype=np.float64)
    jacobian = np.asarray(saved["consumed_foot_jacobian_world"], dtype=np.float64)
    com_offset = np.asarray(saved["consumed_nominal_base_com_offset_body_m"], dtype=np.float64)
    mass = float(saved["consumed_nominal_mass_kg"])
    yaw = float(rpy[2])
    heading = np.asarray(((math.cos(yaw), -math.sin(yaw), 0.0),
                          (math.sin(yaw), math.cos(yaw), 0.0),
                          (0.0, 0.0, 1.0)))
    angular_yaw = heading.T @ angular
    requested = np.asarray((saved["consumed_commanded_roll_rad"],
                            saved["consumed_commanded_pitch_rad"]), dtype=np.float64)
    feedback = 180.0*(requested-rpy[:2])-24.0*angular_yaw[:2]
    upward = mass*9.81
    moment = (np.cross(rotation @ com_offset, np.asarray((0.0, 0.0, upward)))
              + heading @ np.r_[feedback, 0.0])
    allocation = np.vstack((np.ones(4), foot_offset[:, 1], -foot_offset[:, 0]))
    forces = np.linalg.lstsq(allocation, np.r_[upward, moment[:2]], rcond=None)[0]
    clipped = np.clip(forces, 0.0, .65*upward)
    support = np.zeros(16)
    for leg in range(4):
        support[4*leg:4*leg+3] = -jacobian[leg, 2, :3]*clipped[leg]
    for key, expected in (
        ("heading_rotation_world_from_yaw", heading),
        ("base_angular_velocity_yaw_frame_rad_s", angular_yaw),
        ("attitude_feedback_nm", feedback),
        ("upward_force_n", upward),
        ("support_moment_world_nm", moment),
        ("allocation_matrix", allocation),
        ("least_squares_force_n", forces),
        ("support_force_n", clipped),
        ("support_torque_nm", support),
    ):
        require(close(saved[key], expected), f"E nominal support arithmetic differs: {key}")
    return support


def check_controller(record: dict, index: int) -> dict:
    info = record["info"]
    adapter = info["controller_record"]
    calc = adapter["calculation"]
    support = check_nominal_support(adapter["nominal_support"])
    action = np.asarray(record["policy_input_action"], dtype=np.float64)
    require(action.shape == (16,) and np.isfinite(action).all()
            and equal(action, info["policy_input_action"])
            and equal(action, calc["raw_action"]),
            f"E{index} actual requested action differs from controller input")
    values = recompute_controller(calc, action)
    for key, expected in values.items():
        require(close(calc[key], expected),
                f"E{index} residual16 saved arithmetic stage differs: {key}")
    require(equal(info["applied_action"], values["applied_action"])
            and adapter["action_gate_enabled"] == calc["action_enabled"]
            and close(support, adapter["nominal_support"]["support_torque_nm"])
            and equal(adapter["nominal_support"]["support_torque_nm"],
                      calc["consumed_support_torque_nm"])
            and adapter["servo_forward_mps"] == info["consumed_command"]["forward_velocity_mps"]
            and calc["body_common_p_active"] == (adapter["servo_forward_mps"] != 0.0)
            and calc["leg_longitudinal_damping_active"] == (
                adapter["servo_forward_mps"] != 0.0 or adapter["stop_latched_after"]),
            f"E{index} consumed command/gate/nominal support differs")
    return {"computed": values, "calculation": calc}


def _native_rows(folder: Path, receipt: dict) -> list[dict]:
    files = receipt["native_files"]
    require(files == [f"native_block_{i:04d}.jsonl.gz" for i in range(len(files))],
            "E native recorder blocks are missing or reordered")
    rows = []
    for name in files:
        path = folder / name
        require(path.is_file() and not path.is_symlink(), "E native block missing")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream)
    return rows


def _expected_action(tick: int) -> np.ndarray:
    result = np.zeros(16)
    phase = tick - 200
    if 0 <= phase < 320 and phase % 20 < 10:
        result[phase // 20] = .15
    return result


def _expected_raw(tick: int) -> float:
    boundaries = ((175, 0.0), (355, .4), (495, .6), (655, .9),
                  (815, 1.2), (1095, 1.6), (1600, 0.0))
    for end, speed in boundaries:
        if tick < end:
            return speed
    raise ValueError("E1 tick outside 1600-control ladder")


def _speed_stats(records: list[dict]) -> dict:
    stages = []
    for target, start, end in HOLD_WINDOWS:
        selected = [row for row in records if start <= row["tick"] < end]
        if len(selected) != end-start:
            stages.append({"target_mps": target, "hold_ticks": end-start,
                           "observed_ticks": len(selected), "complete": False})
            continue
        speed = np.asarray([row["info"]["metrics"]["body_com_vx_mps"]
                            for row in selected], dtype=np.float64)
        wheel = np.asarray([row["info"]["controller_record"]["calculation"]
                            ["consumed_wheel_omega_rad_s"] for row in selected])
        clip = np.asarray([row["info"]["controller_record"]["calculation"]
                           ["torque_clip_mask"] for row in selected], dtype=bool)
        protect = np.asarray([row["info"]["controller_record"]["calculation"]
                              ["protection_changed_mask"] for row in selected], dtype=bool)
        integral = np.asarray([row["info"]["controller_record"]["calculation"]
                               ["wheel_integral_after_nm"] for row in selected])
        stages.append({
            "target_mps": target, "hold_ticks": end-start, "observed_ticks": len(selected),
            "complete": True, "actual_body_com_mean_mps": float(np.mean(speed)),
            "actual_body_com_rmse_mps": float(np.sqrt(np.mean((speed-target)**2))),
            "actual_body_com_min_mps": float(np.min(speed)),
            "actual_body_com_max_mps": float(np.max(speed)),
            "actual_wheel_mean_rad_s": np.mean(wheel, axis=0).tolist(),
            "wheel_integral_mean_nm": np.mean(integral, axis=0).tolist(),
            "torque_clip_fraction": float(np.mean(clip)),
            "protection_fraction": float(np.mean(protect)),
        })
    stop = [row for row in records if 1415 <= row["tick"] < 1600]
    stop_speed = [abs(row["info"]["metrics"]["body_com_vx_mps"]) for row in stop]
    final_100 = [abs(row["info"]["metrics"]["body_com_vx_mps"])
                 for row in records if 1500 <= row["tick"] < 1600]
    x0 = records[0]["info"]["metrics"]["x_m"] if records else None
    x1 = records[-1]["info"]["metrics"]["x_m"] if records else None
    stop_origin = records[1094]["info"]["metrics"]["x_m"] if len(records) > 1094 else None
    return {
        "stages": stages,
        "stopping_ticks_observed": len(stop),
        "stopping_mean_abs_body_com_vx_mps": (
            float(np.mean(stop_speed)) if stop_speed else None
        ),
        "final_100_mean_abs_body_com_vx_mps": (
            float(np.mean(final_100)) if len(final_100) == 100 else None
        ),
        "stopping_x_displacement_m": (
            float(x1-stop_origin) if stop_origin is not None else None
        ),
        "actual_x_displacement_m": None if x0 is None else float(x1-x0),
        "formal_400_tick_1p6_qualification_claimed": False,
        "speed_threshold_applied": False,
    }


def check_segment(run: Path, index: int, guard_row: dict, binding: dict) -> dict:
    label = ("e0_action_authority", "e1_zero_residual_speed")[index]
    folder = run / label
    receipt = read(folder / "segment_receipt.json")
    before = read(folder / "boundary_before.json")
    after = read(folder / "segment_receipt.json")["boundary_after"]
    reset = read(folder / "reset_receipt.json")
    records = read(folder / "control_records.json")
    geoms = geom_map(folder, binding)
    native = _native_rows(folder, guard_row)
    limit = (800, 1600)[index]
    completed = len(records)
    require(receipt["name"] == guard_row["name"] == label
            and receipt["limit"] == guard_row["control_limit"] == limit
            and receipt["completed"] == completed <= limit
            and receipt["stop_reason"] in ("control_limit", "exception",
                                           "E_native_safety_qualification_failed",
                                           "nonwheel_ground_contact", "course_map_boundary",
                                           "fall_or_low_clearance")
            and reset["seed"] == 88401
            and type(reset["model_address"]) is int and reset["model_address"] > 0
            and type(reset["data_address"]) is int and reset["data_address"] > 0
            and reset["model_address"] != reset["data_address"],
            f"E{index} segment/guard/reset identity differs")
    require(all(reset["before"]["C_state"][key] == reset["after"]["C_state"][key]
                for key in ("construction_attempts", "construction_returns",
                            "control_attempts", "control_returns"))
            and all(reset["before"]["python"][key] == reset["after"]["python"][key]
                    for key in ("control_attempted", "control_completed",
                                "native_attempted", "native_returned")),
            f"E{index} reset refunded C/Python budget")
    require(len(native) >= 5*completed
            and guard_row["native_returned"] >= len(native)
            and guard_row["native_attempted"] <= 5*limit,
            f"E{index} native recorder/physical budget differs")
    if receipt["failure"] is None:
        require(len(native) == guard_row["native_returned"] == 5*completed
                and guard_row["native_attempted"] == 5*completed,
                f"E{index} successful controls lack exactly five native returns")
    else:
        require(guard_row["task_contact_failure_files"],
                f"E{index} partial physical failure lacks raw contact snapshot")
    with np.load(folder / "states.npz", allow_pickle=False) as archive:
        require(set(STATE) <= set(archive.files), f"E{index} endpoint archive incomplete")
        states = {name: archive[name] for name in STATE}
    require(all(states[key].shape[0] == completed+1 and np.isfinite(states[key]).all()
                for key in STATE)
            and states["observation"].shape[1:] == (99,),
            f"E{index} endpoint archive shape/finite differs")
    require(close(states["time"], np.arange(completed+1)*.01, atol=1e-10),
            f"E{index} endpoint clock differs")
    contacts_checked = 0
    rotated_box_contacts = 0
    nonwheel = 0
    previous_servo = 0.0
    previous_integral = np.zeros(4)
    for tick, record in enumerate(records):
        require(record["tick"] == tick and record["wall_ns"] >= 0,
                f"E{index} control tick order/wall differs")
        requested = _expected_action(tick) if index == 0 else np.zeros(16)
        require(equal(record["policy_input_action"], requested),
                f"E{index} requested 16D action schedule differs at {tick}")
        info = record["info"]
        ctrl = check_controller(record, index)
        require(equal(ctrl["calculation"]["wheel_integral_before_nm"], previous_integral),
                f"E{index} wheel PI memory skipped/refunded at {tick}")
        previous_integral = np.asarray(ctrl["calculation"]["wheel_integral_after_nm"])
        require(equal(ctrl["calculation"]["consumed_joint_position_rad"],
                      states["qpos"][tick][binding["qpos_addresses"]])
                and equal(ctrl["calculation"]["consumed_joint_velocity_rad_s"],
                          states["qvel"][tick][binding["dof_addresses"]]),
                f"E{index} controller consumed joint state differs from real compiled addresses")
        require(close(record["reward"], sum(info["reward_terms"].values()))
                and info["completed_control_intervals"] == tick+1,
                f"E{index} saved reward/episode count differs at {tick}")
        if index == 0:
            require(info["raw_operator_command"]["forward_velocity_mps"] == 0.0
                    and info["consumed_command"]["forward_velocity_mps"] == 0.0,
                    f"E0 stationary raw/servo command differs at {tick}")
        else:
            raw = _expected_raw(tick)
            servo = previous_servo + max(-.005, min(.005, raw-previous_servo))
            require(info["raw_operator_command"]["forward_velocity_mps"] == raw
                    and info["servo_receipt"]["raw_target"]["forward_velocity_mps"] == raw
                    and info["servo_receipt"]["tick"] == tick
                    and close(info["servo_receipt"]["applied"]["forward_velocity_mps"], servo)
                    and close(info["servo_receipt"]["forward_increment_mps"],
                              servo-previous_servo)
                    and close(info["consumed_command"]["forward_velocity_mps"], servo),
                    f"E1 exact raw/consumed servo ladder differs at {tick}")
            previous_servo = servo
        interval = info["native_interval_summary"]
        five = native[5*tick:5*tick+5]
        require(len(five) == 5
                and interval["native_returns"] == 5
                and close(interval["start_time_s"], tick*.01)
                and close(interval["end_time_s"], (tick+1)*.01),
                f"E{index} five-native summary/clock differs at {tick}")
        requested_torque = ctrl["computed"]["safe_torque_nm"]
        traces = record["native_actuator_traces"]
        require(len(traces) == 5, f"E{index} actuator physics traces missing at {tick}")
        interval_contacts = 0
        interval_loads: Counter[str] = Counter()
        for substep, (row, trace) in enumerate(zip(five, traces)):
            require(close(row["start_time_s"], tick*.01+substep*.002)
                    and close(row["end_time_s"], tick*.01+(substep+1)*.002)
                    and row["native_index"] == before["python"]["native_attempted"]
                    + 5*tick + substep
                    and row["contact_count"] == len(row["contacts"]),
                    f"E{index} native substep clock/contact count differs")
            for key in NATIVE:
                require(np.isfinite(np.asarray(row["before"][key], dtype=np.float64)).all()
                        and np.isfinite(np.asarray(row["after"][key], dtype=np.float64)).all(),
                        f"E{index} native integrator value nonfinite: {key}")
            roll, pitch = _roll_pitch_deg(row["after"]["qpos"])
            require(close(row["roll_deg"], roll)
                    and close(row["pitch_deg"], pitch)
                    and max(abs(roll), abs(pitch)) <= 10,
                    f"E{index} native posture differs from safety bound")
            for key in ("qpos", "qvel", "qacc_warmstart"):
                if substep == 0:
                    require(equal(row["before"][key], states[key][tick]),
                            f"E{index} first native entry differs from endpoint {key}")
                if substep == 4:
                    require(equal(row["after"][key], states[key][tick+1]),
                            f"E{index} native exit differs from endpoint {key}")
                if substep:
                    require(equal(row["before"][key], five[substep-1]["after"][key]),
                            f"E{index} native state discontinuity: {key}")
            if substep == 4:
                require(equal(row["after"]["ctrl"], states["ctrl"][tick+1]),
                        f"E{index} native applied ctrl differs from endpoint")
            for key in ("requested_nm", "limited_nm", "delayed_nm", "applied_nm"):
                require(close(trace[key], requested_torque),
                        f"E{index} zero-delay/gain-one actuator chain differs: {key}")
            require(equal(row["before"]["ctrl"], trace["applied_nm"])
                    and equal(row["after"]["ctrl"], trace["applied_nm"])
                    and close(row["after"]["actuator_force"], trace["applied_nm"]),
                    f"E{index} saved native ctrl/actual actuator force differs")
            candidates = 0
            positive: Counter[str] = Counter()
            for contact in row["contacts"]:
                load, fam = check_contact(contact, geoms, binding)
                if fam is not None:
                    rotated_box_contacts += fam != "floor"
                    candidate = contact["robot_wheel_index"] is None
                    require(contact["nonwheel_ground_candidate"] is candidate
                            and contact["nonwheel_ground_contact"] is candidate
                            and contact["nonwheel_active_solver_contact"] is (
                                candidate and contact["efc_address"] >= 0),
                            f"E{index} nonwheel candidate/active classification differs")
                candidates += bool(contact["nonwheel_ground_contact"])
                if load:
                    positive[fam] += 1
                contacts_checked += 1
            require(candidates == row["nonwheel_contact_count"]
                    and dict(positive) == row["terrain_family_positive_wheel_load"],
                    f"E{index} native contact summary differs")
            interval_contacts += candidates
            interval_loads.update(positive)
        require(interval_contacts == interval["nonwheel_contact_count"]
                and dict(interval_loads) == interval["terrain_family_positive_wheel_load"],
                f"E{index} interval contact summary differs")
        nonwheel += interval_contacts
        metrics = info["metrics"]
        com_body = _body_com_velocity(states["qpos"][tick+1],
                                      states["qvel"][tick+1], binding)
        require(np.isfinite(tuple(metrics.values())).all()
                and close(com_body[0], metrics["body_com_vx_mps"], atol=1e-6)
                and close(com_body[1], metrics["body_com_vy_mps"], atol=1e-6)
                and close(states["qpos"][tick+1][:2],
                          (metrics["x_m"], metrics["y_m"]), atol=1e-7)
                and abs(metrics["x_m"]) <= 11 and abs(metrics["y_m"]) <= 6,
                f"E{index} actual COM/map metric invalid")
    c_before, c_after = before["C_state"], after["C_state"]
    p_before, p_after = before["python"], after["python"]
    if receipt["failure"] is None:
        require(c_after["control_attempts"]-c_before["control_attempts"] == 5*completed
                and c_after["control_returns"]-c_before["control_returns"] == 5*completed
                and p_after["control_attempted"]-p_before["control_attempted"] == completed
                and p_after["control_completed"]-p_before["control_completed"] == completed
                and p_after["native_attempted"]-p_before["native_attempted"] == 5*completed
                and p_after["native_returned"]-p_before["native_returned"] == 5*completed,
                f"E{index} C/Python segment delta differs from exact 5T")
    return {
        "label": label, "completed_controls": completed,
        "recorded_native": len(native), "actual_native_returned": guard_row["native_returned"],
        "compiled_terrain_geoms": len(geoms), "contacts_checked": contacts_checked,
        "rotated_box_contacts_checked": rotated_box_contacts,
        "nonwheel_candidates": nonwheel,
        "full_segment_completed": receipt["qualification_passed"] is True,
        "stop_reason": receipt["stop_reason"], "failure": receipt["failure"],
        "C_before": before["C_state"], "C_after": after["C_state"],
        "P_before": before["python"], "P_after": after["python"],
        "speed": _speed_stats(records) if index == 1 else None,
        "records": records,
    }


def _authority(records: list[dict]) -> dict:
    dimensions = []
    for dim in range(16):
        pulses = [row for row in records if 200+20*dim <= row["tick"] < 210+20*dim]
        target_changed = torque_changed = 0
        for row in pulses:
            calc = row["info"]["controller_record"]["calculation"]
            actual = recompute_controller(calc, row["policy_input_action"])
            zero = recompute_controller(calc, np.zeros(16))
            if dim < 12:
                joint = LEGS[dim]
                target_changed += bool(actual["geometric_joint_target_rad"][joint]
                                       != zero["geometric_joint_target_rad"][joint])
            else:
                wheel = dim-12
                target_changed += bool(actual["wheel_speed_request_rad_s"][wheel]
                                       != zero["wheel_speed_request_rad_s"][wheel])
            torque_changed += bool(not equal(actual["safe_torque_nm"],
                                             zero["safe_torque_nm"]))
        dimensions.append({
            "dimension": dim, "pulse_controls_observed": len(pulses),
            "target_stage_changed_controls": target_changed,
            "protected_torque_changed_controls": torque_changed,
            "authority_shown": (len(pulses) == 10 and target_changed > 0
                                and torque_changed > 0),
        })
    return {"dimensions": dimensions,
            "all_16_actual_target_and_motor_authority_shown": all(
                item["authority_shown"] for item in dimensions
            ),
            "counterfactual_zero_was_pure_arithmetic_only": True}


def _manifest(run: Path) -> dict:
    files = {}
    total = 0
    for path in sorted(run.rglob("*")):
        require(not path.is_symlink(), f"E closed output contains a symlink: {path}")
        if path.is_file():
            item = identity(path)
            files[path.relative_to(run).as_posix()] = item
            total += item["bytes"]
    require("worker_receipt.json" in files and "session.json" in files,
            "E closed file manifest lacks worker/session")
    return {"schema": "d1-course-e-closed-file-manifest-v1",
            "run": str(run), "file_count": len(files),
            "total_bytes": total, "files": files}


def verify(run: Path) -> tuple[dict, dict]:
    require(run.is_absolute() and run.is_dir(), "--run must be a saved absolute E run")
    prior_source = W / "verify_course_e_08.py"
    prior_output = W / "course_e_readback_08.json"
    require(identity(prior_source)["sha256"]
            == "14d5089cc06db84d42ca896cc5da8f0c0a8982eeb579685483aa834404445304",
            "first E reader source changed after its saved failed attempt")
    prior = read(prior_output)
    require(prior["run"] == str(run)
            and prior["record_integrity_errors"]
            == [{"message": "'model_address'", "segment": 0, "type": "KeyError"},
                {"message": "'model_address'", "segment": 1, "type": "KeyError"}],
            "first E readback failure provenance differs")
    source = check_source(run)
    session = read(run / "session.json")
    origins = read(run / "module_origins.json")
    require(isinstance(origins, dict) and origins
            and all(Path(path).parent == W / "course_impl08"
                    and path in session["source_hashes"] for path in origins.values())
            and "course_native_guard_08" in origins
            and "full_drive_env_08" in origins,
            "E actual imported control module origins differ from frozen sources")
    worker, launcher = read(run / "worker_receipt.json"), read(run / "launcher_receipt.json")
    initial, construction = read(run / "runtime_initial.json"), read(run / "construction_receipt.json")
    scan = read(run / "ground_scan_benchmark.json")
    require(scan["pure_cpu_only"] is True and scan["calls"] == 512
            and scan["terrain_geoms"] == 92 and scan["native_counter_delta"] == 0
            and scan["before"] == scan["after"]
            and len(scan["durations_ns"]) == 512
            and all(type(value) is int and value >= 0 for value in scan["durations_ns"]),
            "E pure compiled-ground scan benchmark altered native state or lacks 512 calls")
    proof, c0 = initial["proof"], initial["C_state"]
    require(proof["passed"] is True and len(proof["jump_slots"]) == 4
            and all(slot["passed"] is True for slot in proof["jump_slots"])
            and all(c0[key] == 0 for key in ("construction_attempts", "construction_returns",
                                            "control_attempts", "control_returns", "violations"))
            and construction["C_state"]["construction_attempts"] == 2
            and construction["C_state"]["construction_returns"] == 2
            and construction["nominal_cache"]["misses"] == 1
            and construction["nominal_cache"]["currsize"] == 1,
            "E actual GOT/2 compiler/cold nominal construction evidence differs")
    require(worker["schema"] == launcher["schema"] == SCHEMA
            and worker["training_steps"] == 0
            and worker["retry_permitted"] is False
            and launcher["retry_permitted"] is False
            and launcher["source_hash_mismatches"] == []
            and launcher["worker_exited"] is True,
            "E worker/launcher provenance differs")
    cfinal, ledger, guard = worker["C_final"], worker["python"], worker["native_guard"]
    require(cfinal["construction_attempts"] == cfinal["construction_returns"] == 2
            and cfinal["control_attempts"] <= 12000
            and cfinal["control_returns"] <= cfinal["control_attempts"]
            and cfinal["phase"] == cfinal["target_model"] == cfinal["target_data"] == 0
            and ledger["control_attempted"] <= 2400
            and ledger["control_completed"] <= ledger["control_attempted"]
            and ledger["native_attempted"] <= 12000
            and ledger["native_returned"] <= ledger["native_attempted"]
            and len(guard["segments"]) <= 2
            and guard["force_sampling_performed"] is True
            and guard["full_contact_qualification_claimed"] is True,
            "E C/Python/native guard actual counts or caps differ")
    if cfinal["ccd_attempts"]:
        require(cfinal["ccd_attempts"] == cfinal["ccd_returns"]
                and cfinal["native_ccd_caller_verified"] is True,
                "E native CCD caller provenance differs")
    else:
        require(cfinal["first_ccd_caller"] == 0,
                "E reported a nonexistent native CCD caller")
    require(cfinal["native_construction_caller_verified"] is True,
            "E compiler caller provenance differs")
    binding = check_binding(construction["actual_geometry_binding"])
    segments = []
    errors = []
    for index, row in enumerate(guard["segments"]):
        try:
            segments.append(check_segment(run, index, row, binding))
        except (ValueError, KeyError, TypeError, IndexError) as error:
            errors.append({"segment": index, "type": type(error).__name__,
                           "message": str(error)})
    if len(segments) == 2:
        first, second = segments
        first_reset = read(run / "e0_action_authority/reset_receipt.json")
        second_reset = read(run / "e1_zero_residual_speed/reset_receipt.json")
        require(first_reset["model_address"] == second_reset["model_address"]
                and first_reset["data_address"] == second_reset["data_address"],
                "E0/E1 reset did not preserve actual model/data identity")
        require(first["C_after"]["control_attempts"] == second["C_before"]["control_attempts"]
                and first["C_after"]["control_returns"] == second["C_before"]["control_returns"]
                and first["P_after"]["control_attempted"] == second["P_before"]["control_attempted"]
                and first["P_after"]["native_returned"] == second["P_before"]["native_returned"],
                "E reset/segment transition refunded physical budget")
        first_states = np.load(run / "e0_action_authority/states.npz", allow_pickle=False)
        second_states = np.load(run / "e1_zero_residual_speed/states.npz", allow_pickle=False)
        try:
            require(all(equal(first_states[key][0], second_states[key][0])
                        for key in STATE),
                    "E0/E1 same-seed reset initial pair differs")
        finally:
            first_states.close()
            second_states.close()
    authority = _authority(segments[0]["records"]) if segments else None
    authority_passed = bool(authority and authority["all_16_actual_target_and_motor_authority_shown"])
    completed = sum(row["completed_controls"] for row in segments)
    count_coherent = (guard["native_attempted"] == ledger["native_attempted"]
                      and guard["native_returned"] == ledger["native_returned"]
                      and cfinal["control_attempts"] == ledger["native_attempted"]
                      and cfinal["control_returns"] == ledger["native_returned"])
    physical_exact = (not errors and count_coherent
                      and cfinal["violations"] == 0
                      and ledger["native_failed"] == ledger["forbidden_entries"] == 0
                      and cfinal["control_attempts"] == cfinal["control_returns"]
                      == 5*completed and ledger["native_attempted"]
                      == ledger["native_returned"] == 5*completed
                      and ledger["control_attempted"] == ledger["control_completed"]
                      == completed and guard["native_checked"] == 5*completed)
    geometry_accounting_passed = physical_exact and all(
        row["recorded_native"] == 5*row["completed_controls"]
        and row["compiled_terrain_geoms"] == 92 for row in segments
    )
    safety = (geometry_accounting_passed and len(segments) == 2
              and worker["warnings"] == [] and worker["failure"] is None
              and all(row["full_segment_completed"] and row["nonwheel_candidates"] == 0
                      for row in segments)
              and guard["max_abs_roll_deg"] <= 10
              and guard["max_abs_pitch_deg"] <= 10
              and guard["failure"] is None)
    execution_complete = (safety and authority_passed and completed == 2400
                          and worker["execution_complete"] is True
                          and launcher["exit_code"] == 0
                          and launcher["failure"] is None)
    speed = segments[1]["speed"] if len(segments) == 2 else None
    report_segments = [{key: value for key, value in row.items()
                        if key not in ("records", "C_before", "C_after", "P_before", "P_after")}
                       for row in segments]
    result = {
        "schema": "d1-course-e-independent-saved-readback-v1",
        "run": str(run), "source_closure": source,
        "previous_failed_readback": {"source": str(prior_source),
                                    "source_identity": identity(prior_source),
                                    "output": str(prior_output),
                                    "output_identity": identity(prior_output),
                                    "reason": "verifier-only missing guard model_address field"},
        "actual_control_module_origins": origins,
        "pure_ground_scan_median_ns": scan["median_ns"],
        "record_integrity_errors": errors,
        "C_Python_guard_counts_coherent": count_coherent,
        "C_violations": cfinal["violations"],
        "python_native_failed": ledger["native_failed"],
        "python_forbidden_entries": ledger["forbidden_entries"],
        "geometry_accounting_passed": geometry_accounting_passed,
        "baseline_safety_passed": safety,
        "E0_16d_actual_authority": authority,
        "E0_authority_passed": authority_passed,
        "speed_ladder": speed,
        "speed_ladder_completed": bool(speed and len(segments) == 2
                                      and segments[1]["completed_controls"] == 1600
                                      and all(stage["complete"] for stage in speed["stages"])
                                      and speed["stopping_ticks_observed"] == 185),
        "execution_complete": execution_complete,
        "controls_completed": completed,
        "normal_native_returned": ledger["native_returned"],
        "compiler_native_returned": cfinal["construction_returns"],
        "segments": report_segments,
        "E1_not_formal_400_tick_1p6_qualification": True,
        "engine_or_policy_imported": False,
        "physics_rerun_or_rescore_performed": False,
    }
    return result, _manifest(run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    require(args.run.is_absolute(), "--run must be absolute")
    result, manifest = verify(args.run.resolve(strict=True))
    for path in (args.output, args.manifest_output):
        if path is not None:
            require(path.is_absolute() and not path.exists(),
                    "output must be an exclusive absolute new JSON file")
    if args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    if args.manifest_output is not None:
        with args.manifest_output.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
