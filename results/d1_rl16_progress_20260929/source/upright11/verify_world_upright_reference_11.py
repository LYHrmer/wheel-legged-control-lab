"""Independent saved-record readback of the one 11-R world-upright run.

This imports only pure arithmetic/record validators. It never imports a model,
simulator, policy or learner and never advances a physical state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
from verify_course_e_08_03 import (
    _body_com_velocity,
    _native_rows,
    _roll_pitch_deg,
    _rotation,
    check_binding,
    check_contact,
    check_controller,
    close,
    equal,
    geom_map,
    identity,
    read,
    require,
)
from verify_rl16_training_08 import (
    _ground_hit,
    _terrain_reference,
    _yaw_from_qpos,
)

SCHEMA = "d1-world-upright-reference-11-v1"
CONTRACT_SHA = "2f891044eac2c881f7240dc0da22a79ced22458c75f1df7ac6b103b7a01ae375"
STATE = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation", "time")
RAMP = ("terrain_ramp_up", "terrain_ramp_deck", "terrain_ramp_down")


def _json(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), "saved record absent: " + str(path))
    value = read(path)
    require(isinstance(value, dict), "saved record must be an object: " + str(path))
    return value


def _source(run: Path) -> tuple[dict, dict]:
    session = _json(run / "session.json")
    launcher = _json(run / "launcher_receipt.json")
    worker = _json(run / "worker_receipt.json")
    require(session["schema"] == launcher["schema"] == worker["schema"] == SCHEMA
            and session["phase"] == "reference_baseline"
            and session["output_directory"] == str(run)
            and session["contract_sha256"] == CONTRACT_SHA
            and session["control_limit"] == 1800
            and session["normal_native_limit"] == 9000
            and session["compiler_native_limit"] == 2
            and session["wallclock_limit_s"] == 180
            and session["segments"] == [1800]
            and session["seed"] == 88611
            and session["retry_permitted"] is False,
            "11-R session source/budget identity differs")
    frozen = session["source_hashes"]
    require(isinstance(frozen, dict) and len(frozen) >= 100,
            "11-R source closure absent")
    for name, expected in frozen.items():
        require(Path(name).is_absolute() and identity(Path(name)) == expected,
                "11-R frozen source drifted: " + name)
    for path in (HERE / "run_world_upright_reference_11.py",
                 HERE / "world_upright_course_11.py",
                 HERE / "next_reference_contract_11.md",
                 HERE / "astra_reference_go_11.json"):
        require(str(path) in frozen, "11-R critical source absent: " + str(path))
    go = _json(HERE / "astra_reference_go_11.json")
    require(identity(HERE / "astra_reference_go_11.json")["sha256"]
            == session["go_sha256"] and go["decision"] == "GO"
            and go["contract_sha256"] == CONTRACT_SHA,
            "11-R actual SOURCE GO differs")
    for name, expected in go["inputs"].items():
        require(frozen.get(name) == expected and identity(Path(name)) == expected,
                "11-R GO input changed: " + name)
    require(launcher["exit_code"] == 0 and launcher["failure"] is None
            and launcher["source_hash_mismatches"] == []
            and launcher["worker_exited"] is True
            and launcher["owned_worker_cleanup"]["no_orphans"] is True
            and launcher["fully_reserved_budget_closed"] is True
            and launcher["retry_permitted"] is False
            and worker["failure"] is None and worker["warnings"] == []
            and worker["source_hash_mismatches"] == []
            and worker["record_valid"] is True
            and worker["policy_loaded"] is False
            and worker["policy_predictions"] == 0
            and worker["training_or_learning_performed"] is False
            and worker["retry_permitted"] is False,
            "11-R host/worker is not a closed zero-policy valid record")
    return session, {"frozen_source_count": len(frozen),
                     "go_sha256": session["go_sha256"]}


def _schedule(run: Path) -> tuple[dict, list[float]]:
    saved = _json(run / "schedule.json")
    raw, servo, nominal = (saved[key] for key in
                           ("raw_commands", "pure_servo_commands", "pure_nominal_xy_m"))
    require(saved["schema"] == "d1-world-upright-reference-schedule-11-v1"
            and saved["seed"] == 88611
            and saved["spawn_position_m"] == [2.75, 0.0, 0.455]
            and saved["pure_nominal_only_not_actual_progress"] is True
            and len(raw) == len(servo) == 1800 and len(nominal) == 1801
            and close(nominal[0], [2.75, 0.0]),
            "11-R preregistered pure schedule absent")
    previous = 0.0
    x = 2.75
    speeds = []
    for tick, (request, applied) in enumerate(zip(raw, servo)):
        target = .4 if 175 <= tick < 1355 else 0.0
        previous += max(-.005, min(.005, target - previous))
        x += previous * .01
        require(request["forward_velocity_mps"] == target
                and applied["forward_velocity_mps"] == previous
                and all(request[key] == applied[key] == 0.0 for key in
                        ("lateral_velocity_mps", "yaw_rate_rps"))
                and request["clearance_m"] == applied["clearance_m"] == .455
                and request["jump_requested"] is applied["jump_requested"] is False
                and close(nominal[tick + 1], [x, 0.0], atol=1e-10),
                "11-R exact 175/1355 raw and .005 servo schedule differs")
        speeds.append(previous)
    require(7.0 < x < 8.0, "11-R nominal path outside predeclared range")
    return saved, speeds


def _states(folder: Path, completed: int) -> dict[str, np.ndarray]:
    path = folder / "states.npz"
    require(path.is_file() and not path.is_symlink(), "11-R endpoint NPZ absent")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == set(STATE), "11-R state array set differs")
        arrays = {key: archive[key] for key in STATE}
    require(all(value.shape[0] == completed + 1 and np.isfinite(value).all()
                for value in arrays.values())
            and arrays["observation"].shape == (completed + 1, 99)
            and close(arrays["time"], np.arange(completed + 1) * .01, atol=1e-10),
            "11-R finite state/99D/time links differ")
    return arrays


def _counter_delta(before: dict, after: dict, completed: int) -> None:
    cb, ca = before["C_state"], after["C_state"]
    pb, pa = before["python"], after["python"]
    require(ca["control_attempts"] - cb["control_attempts"] == 5 * completed
            and ca["control_returns"] - cb["control_returns"] == 5 * completed
            and ca["construction_attempts"] == cb["construction_attempts"]
            and ca["construction_returns"] == cb["construction_returns"]
            and pa["control_attempted"] - pb["control_attempted"] == completed
            and pa["control_completed"] - pb["control_completed"] == completed
            and pa["native_attempted"] - pb["native_attempted"] == 5 * completed
            and pa["native_returned"] - pb["native_returned"] == 5 * completed
            and pa["clock_advanced_substeps"] - pb["clock_advanced_substeps"]
            == 5 * completed,
            "11-R C/Python did not record exact 5T controls")


def _projection(run: Path, binding: dict, geoms: dict[int, dict],
                states: dict[str, np.ndarray]) -> dict:
    saved = _json(run / "final_compiled_x_projection.json")
    require(saved["cache_source"] == "plant.measurement_data_post_refresh_measurements"
            and saved["measurement_and_integration_qpos_qvel_time_equal"] is True
            and equal(saved["measurement_qpos"], states["qpos"][-1])
            and equal(saved["measurement_qvel"], states["qvel"][-1])
            and close(saved["measurement_time_s"], states["time"][-1], atol=1e-12),
            "11-R final geom cache is not linked to the true post-state")
    ramp, wheel = saved["ramp"], saved["wheel"]
    require(set(ramp) == set(RAMP) and set(wheel) == set(map(str, range(4))),
            "11-R final ramp/wheel projection identity differs")
    for name, row in ramp.items():
        gid = row["geom_id"]
        compiled = geoms[gid]
        rotation = _rotation(compiled["quaternion_wxyz"])
        half = np.asarray(compiled["size_m"], dtype=np.float64)
        center = np.asarray(compiled["position_m"], dtype=np.float64)
        require(compiled["name"] == name
                and close(row["rotation_world_from_local"], rotation)
                and close(row["compiled_half_size_m"], half)
                and close(row["center_world_m"], center)
                and close(row["world_x_max_m"], center[0] + np.abs(rotation[0]) @ half),
                "11-R ramp exact rotated box world-X support differs")
    for index, row in wheel.items():
        gid = row["geom_id"]
        rotation = np.asarray(row["rotation_world_from_local"], dtype=np.float64)
        center = np.asarray(row["center_world_m"], dtype=np.float64)
        radius, width = row["compiled_radius_m"], row["compiled_half_width_m"]
        body = int(binding["geom_bodyid"][gid])
        extent = radius * math.hypot(*rotation[0, :2]) + width * abs(rotation[0, 2])
        require(binding["wheel_map"].get(body) == int(index)
                and int(binding["geom_type"][gid]) == 5
                and (binding["geom_contype"][gid]
                     or binding["geom_conaffinity"][gid])
                and close(binding["geom_size"][gid, :2], [radius, width])
                and close(rotation @ rotation.T, np.eye(3), atol=1e-10)
                and close(np.linalg.det(rotation), 1.0, atol=1e-10)
                and close(row["world_x_min_m"], center[0] - extent),
                "11-R actual wheel cylinder world-X support arithmetic differs")
    edge = max(row["world_x_max_m"] for row in ramp.values())
    past = all(row["world_x_min_m"] > edge for row in wheel.values())
    require(saved["all_wheel_collision_shapes_past_ramp"] is past,
            "11-R final four-wheel projection claim differs")
    return {"ramp_max_world_x_m": edge,
            "wheel_min_world_x_m": {k: v["world_x_min_m"] for k, v in wheel.items()},
            "all_four_wheel_collision_cylinders_past_all_ramps": past}


def _manifest(run: Path) -> dict:
    files = {}
    for path in sorted(run.rglob("*")):
        require(not path.is_symlink(), "11-R archived run contains symlink")
        if path.is_file():
            files[path.relative_to(run).as_posix()] = identity(path)
    return {"schema": "d1-world-upright-reference-closed-manifest-v1",
            "run": str(run), "file_count": len(files),
            "total_bytes": sum(row["bytes"] for row in files.values()),
            "files": files}


def verify(run: Path) -> tuple[dict, dict]:
    require(run.is_absolute() and run.is_dir(), "11-R --run must be absolute")
    session, source = _source(run)
    schedule, speeds = _schedule(run)
    initial = _json(run / "runtime_initial.json")
    construction = _json(run / "construction_receipt.json")
    worker = _json(run / "worker_receipt.json")
    c, p = (worker["C_Python_final"][key] for key in ("C_state", "python"))
    guard = worker["native_guard"]
    require(initial["proof"]["passed"] is True
            and initial["proof"]["dso_path"] == session["library"]
            and initial["proof"]["dso_sha256"]
            == session["source_hashes"][session["library"]]["sha256"]
            and len(initial["proof"]["jump_slots"]) == 4
            and all(slot["passed"] is True for slot in initial["proof"]["jump_slots"])
            and construction["C_state"]["construction_attempts"]
            == construction["C_state"]["construction_returns"] == 2
            and construction["nominal_cache"]["misses"] == 1,
            "11-R actual GOT/DSO/cold2 proof differs")
    binding = check_binding(construction["actual_geometry_binding"])
    folder = run / "ramp_zero_reference"
    geoms = geom_map(folder, binding)
    require(construction["compiled_geometry"] == _json(folder / "geometry_manifest.json"),
            "11-R construction/segment compiled92 source differs")
    receipt = _json(folder / "segment_receipt.json")
    before = _json(folder / "boundary_before.json")
    after = receipt["boundary_after"]
    reset = _json(folder / "reset_receipt.json")
    records = read(folder / "control_records.json")
    completed = len(records)
    native_segment = receipt["native_segment"]
    native = _native_rows(folder, native_segment)
    require(receipt["name"] == native_segment["name"] == "ramp_zero_reference"
            and receipt["limit"] == native_segment["control_limit"] == 1800
            and receipt["completed_controls"] == completed <= 1800
            and receipt["failure"] is None
            and native_segment["mode"] == "heldout"
            and native_segment["force_sampling_performed"] is True
            and native_segment["full_contact_qualification_recorded"] is True
            and native_segment["failure"] is None
            and native_segment["partial_native_interval"] is False
            and native_segment["native_attempted"]
            == native_segment["native_returned"] == len(native) == completed * 5
            and reset["seed"] == 88611
            and reset["episode_metadata"]["task_schema"]
            == "d1-course-world-upright-rl16-task-v1"
            and reset["episode_metadata"]["reference_schema"]
            == "d1-course-world-upright-roll-pitch-zero-v1"
            and reset["episode_metadata"]["max_steps"] == 1800
            and reset["model_address"] == construction["model_address"]
            and reset["data_address"] == construction["data_address"],
            "11-R actual segment/seed/world task/native closure differs")
    _counter_delta(before, after, completed)
    _counter_delta(reset["before"], reset["after"], 0)
    require(before == reset["before"] and after == worker["C_Python_final"],
            "11-R reset/final segment ledger boundary differs")
    require(c["construction_attempts"] == c["construction_returns"] == 2
            and c["control_attempts"] == c["control_returns"] == 5 * completed
            and c["control_limit"] == 9000
            and c["phase"] == c["target_model"] == c["target_data"] == 0
            and c["violations"] == 0
            and c["native_construction_caller_verified"] is True
            and c["ccd_attempts"] == c["ccd_returns"]
            and (c["ccd_attempts"] == 0 or c["native_ccd_caller_verified"] is True)
            and c["first_control_step_caller_dladdr"]["offset"] == 777003
            and p["control_attempted"] == p["control_completed"] == completed
            and p["native_attempted"] == p["native_returned"] == 5 * completed
            and p["clock_advanced_substeps"] == 5 * completed
            and p["native_failed"] == p["forbidden_entries"] == 0
            and p["fatal_latched"] is False
            and guard["native_attempted"] == guard["native_returned"]
            == guard["native_checked"] == 5 * completed
            and guard["failure"] is None
            and guard["full_contact_qualification_claimed"] is True,
            "11-R final actual C/Python/guard budget or DSO caller differs")
    states = _states(folder, completed)
    initial_roll, initial_pitch = _roll_pitch_deg(states["qpos"][0])
    heading = _yaw_from_qpos(states["qpos"][0])
    xy = states["qpos"][:, :2] - np.asarray([2.75, 0.0])
    lateral = -math.sin(heading) * xy[:, 0] + math.cos(heading) * xy[:, 1]
    initial_hit = _ground_hit(geoms, *states["qpos"][0, :2])
    min_clearance = float(states["qpos"][0, 2] - initial_hit[0])
    max_posture = max(abs(initial_roll), abs(initial_pitch))
    max_lateral = float(np.max(np.abs(lateral)))
    max_abs_x = float(np.max(np.abs(states["qpos"][:, 0])))
    max_abs_y = float(np.max(np.abs(states["qpos"][:, 1])))
    load_by_name: Counter[str] = Counter()
    load_by_family: Counter[str] = Counter()
    total_contacts = 0
    nonwheel_candidates = 0
    nonwheel_active = 0
    rotated_contacts = 0
    previous_servo = 0.0
    previous_integral = np.zeros(4)
    for tick, record in enumerate(records):
        info = record["info"]
        calc = check_controller(record, 11)["calculation"]
        require(record["tick"] == tick and equal(record["policy_input_action"], np.zeros(16))
                and equal(calc["wheel_integral_before_nm"], previous_integral)
                and equal(calc["consumed_joint_position_rad"],
                          states["qpos"][tick][binding["qpos_addresses"]])
                and equal(calc["consumed_joint_velocity_rad_s"],
                          states["qvel"][tick][binding["dof_addresses"]])
                and close(record["reward"], sum(info["reward_terms"].values()))
                and info["completed_control_intervals"] == tick + 1,
                "11-R zero action/reward/controller state chain differs")
        previous_integral = np.asarray(calc["wheel_integral_after_nm"])
        raw = schedule["raw_commands"][tick]
        previous_servo = speeds[tick]
        consumed = info["consumed_command"]
        world = record["consumed_world_command"]
        ground = record["consumed_actual_ground_reference"]
        pre_height, pre_normal, pre_id = _ground_hit(
            geoms, *states["qpos"][tick, :2],
        )
        pre_pitch = math.atan2(float(pre_normal[0]), float(pre_normal[2]))
        pre_roll = math.atan2(-float(pre_normal[1]), float(pre_normal[2]))
        require(info["task_schema"] == "d1-course-world-upright-rl16-task-v1"
                and info["reward_schema"]
                == "d1-course-world-upright-bodycom-yaw-terrain-action-torque-v1"
                and info["reference_schema"]
                == "d1-course-world-upright-roll-pitch-zero-v1"
                and info["raw_operator_command"] == raw
                and info["servo_receipt"]["tick"] == tick
                and info["servo_receipt"]["raw_target"] == raw
                and close(consumed["forward_velocity_mps"], previous_servo)
                and close(world["forward_velocity_mps"], previous_servo)
                and world["roll_rad"] == world["pitch_rad"] == 0.0
                and world["raw_forward_velocity_mps"] == raw["forward_velocity_mps"]
                and close(ground["height_m"], pre_height, atol=1e-7)
                and close(ground["pitch_rad"], pre_pitch, atol=1e-7)
                and close(ground["roll_rad"], pre_roll, atol=1e-7)
                and info["scan_center_geom_id"] == pre_id
                and close(world["base_height_m"], pre_height + .455, atol=1e-7)
                and calc["consumed_nominal_joint_target_rad"] is not None
                and info["controller_record"]["nominal_support"]
                ["consumed_commanded_roll_rad"] == 0.0
                and info["controller_record"]["nominal_support"]
                ["consumed_commanded_pitch_rad"] == 0.0,
                "11-R true pre-ground/servo/world-upright target differs")
        five = native[5*tick:5*tick + 5]
        traces = record["native_actuator_traces"]
        interval = info["native_interval_summary"]
        require(len(five) == len(traces) == 5
                and interval["native_returns"] == 5
                and close(interval["start_time_s"], tick * .01)
                and close(interval["end_time_s"], (tick + 1) * .01),
                "11-R real five-native tick missing")
        interval_nonwheel = 0
        interval_loads: Counter[str] = Counter()
        for substep, (row, trace) in enumerate(zip(five, traces)):
            require(row["native_index"] == 5*tick + substep
                    and close(row["start_time_s"], tick*.01 + substep*.002)
                    and close(row["end_time_s"], tick*.01 + (substep+1)*.002)
                    and row["contact_count"] == len(row["contacts"]),
                    "11-R native row order/clock/contact count differs")
            roll, pitch = _roll_pitch_deg(row["after"]["qpos"])
            max_posture = max(max_posture, abs(roll), abs(pitch))
            native_qpos = np.asarray(row["after"]["qpos"], dtype=np.float64)
            native_height, _, _ = _ground_hit(geoms, *native_qpos[:2])
            min_clearance = min(min_clearance, float(native_qpos[2] - native_height))
            max_abs_x = max(max_abs_x, abs(float(native_qpos[0])))
            max_abs_y = max(max_abs_y, abs(float(native_qpos[1])))
            native_xy = native_qpos[:2] - np.asarray([2.75, 0.0])
            max_lateral = max(max_lateral, abs(float(
                -math.sin(heading)*native_xy[0] + math.cos(heading)*native_xy[1],
            )))
            require(close(row["roll_deg"], roll) and close(row["pitch_deg"], pitch)
                    and max(abs(roll), abs(pitch)) <= 45,
                    "11-R native quaternion/posture record differs")
            for key in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
                require(np.isfinite(np.asarray(row["before"][key])).all(),
                        "11-R native pre-integrator array nonfinite: " + key)
            for key in ("qpos", "qvel", "ctrl", "qacc_warmstart", "qacc",
                        "actuator_force", "qfrc_actuator"):
                require(np.isfinite(np.asarray(row["after"][key])).all(),
                        "11-R native post-integrator/force array nonfinite: " + key)
            for key in ("qpos", "qvel", "qacc_warmstart"):
                if substep == 0:
                    require(equal(row["before"][key], states[key][tick]),
                            "11-R native first/pre state differs: " + key)
                if substep == 4:
                    require(equal(row["after"][key], states[key][tick+1]),
                            "11-R native last/post state differs: " + key)
                if substep:
                    require(equal(row["before"][key], five[substep-1]["after"][key]),
                            "11-R native substep state discontinuity: " + key)
            if substep == 4:
                require(equal(row["after"]["ctrl"], states["ctrl"][tick+1]),
                        "11-R final native ctrl differs from 100Hz state")
            torque = calc["safe_torque_nm"]
            require(all(close(trace[key], torque) for key in
                        ("requested_nm", "limited_nm", "delayed_nm", "applied_nm"))
                    and equal(row["before"]["ctrl"], trace["applied_nm"])
                    and equal(row["after"]["ctrl"], trace["applied_nm"])
                    and close(row["after"]["actuator_force"], trace["applied_nm"]),
                    "11-R requested/limited/delayed/applied/native torque differs")
            count = 0
            names: Counter[str] = Counter()
            families: Counter[str] = Counter()
            for contact in row["contacts"]:
                load, family = check_contact(contact, geoms, binding)
                if family is not None:
                    candidate = contact["robot_wheel_index"] is None
                    require(contact["nonwheel_ground_candidate"] is candidate
                            and contact["nonwheel_ground_contact"] is candidate
                            and contact["nonwheel_active_solver_contact"] is (
                                candidate and contact["efc_address"] >= 0),
                            "11-R nonwheel candidate/solver status differs")
                    rotated_contacts += family != "floor"
                count += bool(contact["nonwheel_ground_contact"])
                nonwheel_active += bool(contact["nonwheel_active_solver_contact"])
                if load:
                    names[contact["terrain_geom_name"]] += 1
                    families[family] += 1
                total_contacts += 1
            require(count == row["nonwheel_contact_count"]
                    and dict(families) == row["terrain_family_positive_wheel_load"],
                    "11-R native force/contact summary differs")
            interval_nonwheel += count
            interval_loads.update(families)
            load_by_family.update(families)
            load_by_name.update(names)
        require(interval_nonwheel == interval["nonwheel_contact_count"]
                and dict(interval_loads) == interval["terrain_family_positive_wheel_load"],
                "11-R interval contact/positive wheel load differs")
        nonwheel_candidates += interval_nonwheel
        metrics = info["metrics"]
        qpos, qvel = states["qpos"][tick+1], states["qvel"][tick+1]
        body = _body_com_velocity(qpos, qvel, binding)
        roll, pitch = _roll_pitch_deg(qpos)
        height, normal, geom_id = _ground_hit(geoms, *qpos[:2])
        r_ref, p_ref = _terrain_reference(normal, _yaw_from_qpos(qpos))
        clearance = float(qpos[2] - height)
        min_clearance = min(min_clearance, clearance)
        require(close(body[0], metrics["body_com_vx_mps"], atol=1e-6)
                and close(body[1], metrics["body_com_vy_mps"], atol=1e-6)
                and close(qvel[5], metrics["body_yaw_rate_rps"], atol=1e-7)
                and close(qpos[:2], [metrics["x_m"], metrics["y_m"]], atol=1e-7)
                and geom_id == metrics["ground_geom_id"]
                and close(height, metrics["ground_height_m"], atol=1e-7)
                and close(clearance, metrics["clearance_m"], atol=1e-7)
                and close(math.radians(roll), metrics["task_roll_error_rad"], atol=1e-7)
                and close(math.radians(pitch), metrics["task_pitch_error_rad"], atol=1e-7)
                and metrics["reference_roll_rad"] == metrics["reference_pitch_rad"] == 0.0
                and close(math.radians(roll) - r_ref,
                          metrics["relative_roll_rad"], atol=1e-7)
                and close(math.radians(pitch) - p_ref,
                          metrics["relative_pitch_rad"], atol=1e-7)
                and close(info["reward_terms"]["attitude"],
                          -.2 * ((math.radians(roll)/.2)**2
                                 + (math.radians(pitch)/.2)**2), atol=1e-7),
                "11-R actual COM/ground/world task/old geometry diagnostic differs")
        if tick < completed-1:
            require(record["terminated"] is False and record["truncated"] is False,
                    "11-R terminal occurred before archived final control")
    if completed:
        last = records[-1]
        if completed < 1800:
            require(last["terminated"] is True and last["truncated"] is False
                    and last["info"]["terminal_reason"] in (
                        "fall_or_low_clearance", "course_map_boundary",
                        "nonwheel_ground_contact")
                    and receipt["stop_reason"] == last["info"]["terminal_reason"]
                    and worker["execution_complete"] is False,
                    "11-R partial run was not a valid task terminal")
        else:
            require(last["terminated"] is False and last["truncated"] is True
                    and receipt["stop_reason"] == "time_limit_1800"
                    and worker["execution_complete"] is True,
                    "11-R full 1800 controls did not end by true TimeLimit")
    require(guard["contacts_checked"] == total_contacts
            and guard["nonwheel_ground_contacts"] == nonwheel_candidates
            and guard["positive_wheel_load_by_family"] == dict(load_by_family),
            "11-R final guard contact totals differ from all native rows")
    projection = _projection(run, binding, geoms, states)
    speeds_actual = np.asarray([row["info"]["metrics"]["body_com_vx_mps"]
                                for row in records], dtype=np.float64)
    fixed = speeds_actual[600:min(1000, completed)]
    drive = speeds_actual[175:min(1355, completed)]
    final = records[-100:] if completed == 1800 else []
    final_com = np.asarray([row["info"]["metrics"]["body_com_vx_mps"]
                            for row in final], dtype=np.float64)
    final_yaw = np.asarray([row["info"]["metrics"]["body_yaw_rate_rps"]
                            for row in final], dtype=np.float64)
    final_height = np.asarray([row["info"]["metrics"]["clearance_m"]
                               for row in final], dtype=np.float64)
    heading_displacement = (math.cos(heading) * xy[:, 0]
                            + math.sin(heading) * xy[:, 1])
    stop_distance = (float(heading_displacement[-1] - heading_displacement[1355])
                     if completed >= 1355 else None)
    geometry_safety = (max_posture <= 10 and min_clearance >= .28
                       and nonwheel_active == 0 and max_lateral <= .25
                       and max_abs_x <= 10.5 and max_abs_y <= 5.8)
    ramp_loads = all(load_by_name[name] > 0 for name in RAMP)
    settled = (completed == 1800 and float(np.mean(np.abs(final_com))) <= .05
               and float(np.mean(np.abs(final_yaw))) <= .05
               and float(np.std(final_height)) <= .02)
    qualified = (completed == 1800 and geometry_safety and ramp_loads
                 and projection["all_four_wheel_collision_cylinders_past_all_ramps"]
                 and settled)

    def stats(values: np.ndarray, target: float, expected: int) -> dict:
        return {"sample_count": len(values), "expected_samples": expected,
                "complete": len(values) == expected,
                "actual_COM_mean_mps": float(np.mean(values)) if len(values) else None,
                "actual_COM_RMS_error_mps": float(np.sqrt(np.mean((values-target)**2)))
                if len(values) else None}

    result = {
        "schema": "d1-world-upright-reference-independent-readback-v1",
        "run": str(run), "source_closure": source,
        "offline_reader_source": {"path": str(Path(__file__).resolve()),
                                  **identity(Path(__file__).resolve())},
        "execution_record_valid": True,
        "completed_controls": completed,
        "normal_native_returns": 5*completed,
        "compiler_native_returns": 2,
        "full_1800_controls": completed == 1800,
        "task_stop_reason": receipt["stop_reason"],
        "geometry_and_safety_passed": bool(geometry_safety),
        "actual_terrain_wheel_positive_load_counts": dict(load_by_name),
        "all_three_ramp_geom_positive_wheel_load": bool(ramp_loads),
        "contact_rows_checked": total_contacts,
        "rotated_box_contact_rows_checked": rotated_contacts,
        "nonwheel_ground_candidates": nonwheel_candidates,
        "nonwheel_active_solver_contacts": nonwheel_active,
        "max_abs_initial_or_native_world_roll_pitch_deg": float(max_posture),
        "min_initial_or_native_clearance_m": min_clearance,
        "max_spawn_heading_lateral_offset_m": max_lateral,
        "max_abs_world_x_m": max_abs_x,
        "max_abs_world_y_m": max_abs_y,
        "final_collision_shape_projection": projection,
        "last_100_mean_abs_actual_COM_vx_mps": float(np.mean(np.abs(final_com)))
        if len(final_com) else None,
        "last_100_mean_abs_actual_body_yaw_rate_rps": float(np.mean(np.abs(final_yaw)))
        if len(final_yaw) else None,
        "last_100_clearance_population_std_m": float(np.std(final_height))
        if len(final_height) else None,
        "drive_interval_175_1355_descriptive": stats(drive, .4, 1180),
        "fixed_600_1000_descriptive": stats(fixed, .4, 400),
        "release_1355_to_end_spawn_heading_distance_m": stop_distance,
        "reference_baseline_passed": bool(qualified),
        "RL_contribution_proven": False,
        "high_speed_or_full_course_qualified": False,
        "physics_rescored_or_engine_imported": False,
    }
    return result, _manifest(run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    require(args.run.is_absolute(), "11-R --run must be absolute")
    result, manifest = verify(args.run.resolve(strict=True))
    for target in (args.output, args.manifest_output):
        if target is not None:
            require(target.is_absolute() and not target.exists(),
                    "11-R offline output must be new and exclusive")
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
