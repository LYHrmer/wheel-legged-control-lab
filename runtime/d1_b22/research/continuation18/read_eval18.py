"""Independent C18 saved-evaluation reader; no model or physics import.

The C16 physical/native chain is retained; C18 session seeds and controller
math are verified explicitly. This reader never imports the live controller.

The full controller/native/contact/force loop below is adapted from frozen
rl11/verify_short_rl16_training_11_04.py. Only saved identity, actor labels,
seed table, and archive transaction checks change.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
W = HERE.parent
for directory in (W, W / "course_impl08", W / "rl11", W / "continuation13",
                  W / "continuation15"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from archive13.atomic_archive_13 import verify_manifest
import score18
from recipes18 import ORDER, TASK_BY_ID, floor_schedule, make_schedule
from heldout15 import PAIR_FIELDS, RECORD_SCHEMA
from rl11.verify_short_rl16_training_11_04 import boundary_delta
from verify_control18 import check_controller
from upright11.verify_world_upright_reference_11 import _projection
from verify_course_e_08_03 import (
    _body_com_velocity, _native_rows, _roll_pitch_deg, _rotation,
    check_binding, check_contact, close, equal, geom_map, identity, require,
)
from verify_rl16_training_08 import _ground_hit, _terrain_reference, _yaw_from_qpos

READ_SCHEMA = "d1-world-upright-stage18-independent-eval-readback-v1"


def _committed(path: Path) -> None:
    require(path.is_file() and not path.is_symlink(), f"missing saved payload: {path}")
    manifest = path.with_name(path.name + ".manifest.json")
    require(manifest.is_file() and not manifest.is_symlink()
            and verify_manifest(manifest), f"archive transaction invalid: {path}")


def document(path: Path) -> dict:
    _committed(path)
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"saved JSON is not an object: {path}")
    return value


def plain_document(path: Path) -> dict:
    """Read a host-owned receipt whose identity is checked by the session."""
    require(path.is_file() and not path.is_symlink(), f"host receipt missing: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"host receipt is not an object: {path}")
    return value


def arrays(path: Path) -> dict[str, np.ndarray]:
    _committed(path)
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def control_rows(folder: Path, blocks: list[dict]) -> list[dict]:
    rows = []
    for index, block in enumerate(blocks):
        name = f"control_records_{index:04d}.jsonl.gz"
        require(block["file"] == name and 0 < block["rows"] <= 200
                and block["archive_manifest"] == name + ".manifest.json",
                "heldout control archive block ordering differs")
        path = folder / name
        _committed(path)
        require(identity(path) == {key: block[key] for key in ("sha256", "bytes")},
                "heldout control archive digest differs")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            part = [json.loads(line) for line in stream]
        require(len(part) == block["rows"], "heldout control block count differs")
        rows.extend(part)
    return rows


def native_rows(folder: Path, guard: dict) -> list[dict]:
    files = guard["native_files"]
    require(guard["archive_block_manifests"]
            == [name + ".manifest.json" for name in files],
            "heldout native archive manifest order differs")
    for name in files:
        _committed(folder / name)
    return _native_rows(folder, guard)


def heldout_schedule(folder: Path, spec: tuple, actor: str, *, mirror: bool = False) -> dict:
    case_id, terrain, speed, seed, cap, hold_start, release = spec
    schedule = (floor_schedule(seed) if case_id == "floor_0p4_600"
                else make_schedule(case_id, seed, mirror=mirror))
    require((schedule.terrain, schedule.seed, schedule.control_cap)
            == (terrain, seed, cap), "new heldout recipe identity differs")
    saved = document(folder / "schedule.json")
    require(saved["schema"] == schedule.schema
            and saved["record_schema"] == RECORD_SCHEMA
            and (saved["case_id"], saved["actor"], saved["seed"],
                 saved["terrain"]) == (case_id, actor, seed, terrain)
            and saved["scoring_actor"] == ("zero" if actor == "zero" else "final_policy")
            and saved["spawn_position_m"] == list(schedule.spawn_position_m)
            and saved["control_cap"] == cap
            and saved["raw_command_sha256"] == schedule.command_sha256
            and saved["raw_commands"] == [asdict(row) for row in schedule.raw_commands],
            "new heldout saved raw schedule differs")
    return saved


def heldout_case(run: Path, spec: tuple, actor: str, binding: dict,
                 expected_geometry: dict, *,
                 expected_variant: str,
                 mirror: bool = False,
                 folder_override: Path | None = None) -> tuple[dict, dict]:
    case_id, terrain, speed, seed, cap, _hold, _release = spec
    folder = (run / "heldout" / f"{case_id}_{actor}"
              if folder_override is None else folder_override)
    receipt = document(folder / "case_receipt.json")
    schedule = heldout_schedule(folder, spec, actor, mirror=mirror)
    reset = document(folder / "reset_receipt.json")
    before = document(folder / "boundary_before.json")
    rows = control_rows(folder, receipt["controller_record_blocks"])
    guard = receipt["native_segment"]
    native = native_rows(folder, guard)
    states = arrays(folder / "states.npz")
    initial = arrays(folder / "initial_state.npz")
    require(document(folder / "geometry_manifest.json") == expected_geometry,
            "heldout actual compiled 92 geom manifest changed between cases")
    geometry = geom_map(folder, binding)
    n = len(rows)
    require(receipt["schema"] == RECORD_SCHEMA
            and (receipt["case_id"], receipt["actor"], receipt["seed"],
                 receipt["terrain"]) == (case_id, actor, seed, terrain)
            and receipt["scoring_actor"]
            == ("zero" if actor == "zero" else "final_policy")
            and receipt["checkpoint_sha256"] == schedule["checkpoint_sha256"]
            and receipt["raw_command_sha256"] == schedule["raw_command_sha256"]
            and 1 <= receipt["completed_controls"] == n <= cap
            and receipt["control_cap"] == cap
            and receipt["failure"] is None and receipt["record_valid"] is True
            and receipt["termination_recorded"] is True
            and receipt["policy_prediction_attempts"]
            == receipt["policy_predictions"] == (n if actor != "zero" else 0)
            and guard["mode"] == "heldout"
            and guard["native_attempted"] == guard["native_returned"] == len(native) == 5*n
            and guard["failure"] is None and guard["archive_failure"] is None
            and guard["partial_native_interval"] is False
            and guard["force_sampling_performed"] is True
            and guard["full_contact_qualification_recorded"] is True
            and reset["seed"] == seed
            and reset["model_address"] > 0 and reset["data_address"] > 0,
            "heldout source/record/native boundary invalid")
    require(all(equal(states[field][0], initial[field]) for field in PAIR_FIELDS)
            and all(states[field].shape[0] == n + 1 and np.isfinite(states[field]).all()
                    for field in (*PAIR_FIELDS, "time"))
            and states["observation"].shape == (n+1, 99)
            and close(states["time"], np.arange(n+1)*.01),
            "heldout initial/endpoint state linkage invalid")
    for key in ("control_attempts", "control_returns", "construction_attempts",
                "construction_returns"):
        require(reset["before"]["C_state"][key] == reset["after"]["C_state"][key],
                "heldout reset refunded C budget")
    for key in ("control_attempted", "control_completed", "native_attempted",
                "native_returned"):
        require(reset["before"]["python"][key] == reset["after"]["python"][key],
                "heldout reset refunded Python budget")
    boundary_delta(before, receipt["boundary_after"], n)
    require(reset["before"] == before and receipt["policy_predictions"]
            == (n if actor != "zero" else 0),
            "heldout reset entry/policy predict count differs")
    initial_qpos = states["qpos"][0]
    initial_yaw = _yaw_from_qpos(initial_qpos)
    forward_axis = np.asarray((math.cos(initial_yaw), math.sin(initial_yaw)))
    lateral_axis = np.asarray((-math.sin(initial_yaw), math.cos(initial_yaw)))
    initial_xy = initial_qpos[:2]
    initial_height, initial_normal, _ = _ground_hit(
        geometry, initial_xy[0], initial_xy[1],
    )
    initial_roll, initial_pitch = _roll_pitch_deg(initial_qpos)
    initial_tilt = math.degrees(math.acos(float(np.clip(
        _rotation(initial_qpos[3:7])[:, 2] @ initial_normal, -1.0, 1.0,
    ))))
    # Arrays are canonical raw numeric inputs for the future pure scorer.
    canonical: dict[str, object] = {
        "case_id": case_id,
        "actor": "zero" if actor == "zero" else "final_policy",
        "experiment_actor": actor,
        "scoring_actor": "zero" if actor == "zero" else "final_policy",
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "terrain": terrain, "seed": seed, "mirror": mirror,
        "target_speed_mps": speed, "control_cap": cap,
        "task_schema": "d1-course-world-upright-rl16-task-v1",
        "reference_schema": "d1-course-world-upright-roll-pitch-zero-v1",
        "reward_schema": "d1-course-world-upright-bodycom-yaw-terrain-action-torque-v1",
        "initial_base_x_m": float(initial_xy[0]),
        "initial_base_y_m": float(initial_xy[1]),
        "completed_controls": n, "terminated": receipt["terminated"],
        "truncated": receipt["truncated"], "stop_reason": receipt["stop_reason"],
        "spawn_position_m": schedule["spawn_position_m"],
        "tick_index": np.arange(n, dtype=np.int64),
        "native_index": np.arange(5*n, dtype=np.int64),
        "initial_roll_deg": initial_roll, "initial_pitch_deg": initial_pitch,
        "initial_terrain_relative_tilt_deg": initial_tilt,
        "initial_clearance_m": float(initial_qpos[2]-initial_height),
        "applied_servo_vx_mps": [], "applied_servo_yaw_rps": [],
        "com_vx_mps": [], "body_yaw_rate_rps": [],
        "heading_rad": [], "clearance_m": [], "lateral_offset_m": [],
        "forward_projection_m": [], "motor_torque_nm": [],
        "native_roll_deg": [], "native_pitch_deg": [],
        "native_terrain_relative_tilt_deg": [], "native_clearance_m": [],
        "native_forward_projection_m": [],
        "native_base_x_m": [], "native_base_y_m": [],
        "native_nonwheel_ground_candidate_count": [],
        "positive_wheel_load_native_counts_by_family": {},
        "ramp_positive_wheel_load_native_counts_by_geom": {},
        "warning_count": 0, "geometry_invalid_count": 0,
        "map_escape_count": 0, "fall_count": 0,
    }
    family_loads: Counter[str] = Counter()
    ramp_geom_loads: Counter[str] = Counter()
    nonwheel = contacts_checked = rotated_box_contacts = 0
    servo_vx = servo_yaw = 0.0
    previous_servo_forward: float | None = None
    stop_latched = False
    previous_integral = np.zeros(4, dtype=np.float64)
    previous_common_reference_z = 0.0  # New controller reset state for every episode.
    for tick, record in enumerate(rows):
        require(record["tick"] == tick and record["actor"] == actor
                and record["scoring_actor"] == canonical["scoring_actor"]
                and record["checkpoint_sha256"] == receipt["checkpoint_sha256"]
                and record["policy_predict_called"] is (actor != "zero")
                and equal(record["input_observation99"], states["observation"][tick])
                and np.isfinite(np.asarray(record["policy_input_action"], dtype=float)).all(),
                "heldout actor/observation/prediction link differs")
        if actor == "zero":
            require(equal(record["policy_input_action"], np.zeros(16)),
                    "heldout zero actor has nonzero residual")
        info = record["info"]
        raw = schedule["raw_commands"][tick]
        servo_vx += max(-.005, min(.005, raw["forward_velocity_mps"] - servo_vx))
        servo_yaw += max(-.006, min(.006, raw["yaw_rate_rps"] - servo_yaw))
        require(info["raw_operator_command"] == raw
                and close(info["consumed_command"]["forward_velocity_mps"], servo_vx)
                and close(info["consumed_command"]["yaw_rate_rps"], servo_yaw)
                and info["completed_control_intervals"] == tick + 1
                and close(record["reward"], sum(info["reward_terms"].values()))
                and info["reference_roll_rad"] == 0.0
                and info["reference_pitch_rad"] == 0.0
                and info["controller_record"]["nominal_support"]
                ["consumed_commanded_roll_rad"] == 0.0
                and info["controller_record"]["nominal_support"]
                ["consumed_commanded_pitch_rad"] == 0.0,
                "heldout raw/consumed servo/reward timeline differs")
        adapter = info["controller_record"]
        if servo_vx != 0.0:
            stop_after = False
        elif previous_servo_forward is not None and previous_servo_forward != 0.0:
            stop_after = True
        else:
            stop_after = stop_latched
        require(adapter["controller_variant"] == expected_variant
                and adapter["mode"] == "eval"
                and adapter["test_actor_probe"] is False
                and adapter["previous_servo_forward_mps"] == previous_servo_forward
                and adapter["stop_latched_before"] is stop_latched
                and adapter["stop_latched_after"] is stop_after
                and adapter["calculation"]["controller_variant"] == expected_variant,
                "C18 adapter variant/eval/probe/stop memory differs from session and servo")
        previous_servo_forward, stop_latched = servo_vx, stop_after
        pre_qpos, pre_qvel = states["qpos"][tick], states["qvel"][tick]
        pre_body_forward = float(_body_com_velocity(pre_qpos, pre_qvel, binding)[0])
        calc = check_controller(record, tick, pre_qpos=pre_qpos, pre_qvel=pre_qvel,
                                servo_yaw_rps=servo_yaw,
                                pre_body_forward_mps=pre_body_forward)
        require(equal(calc["calculation"]["wheel_integral_before_nm"], previous_integral),
                "heldout wheel PI integral before does not continue previous after/reset zero")
        previous_integral = np.asarray(
            calc["calculation"]["wheel_integral_after_nm"], dtype=np.float64)
        require(close(calc["calculation"]["wheel_common_reference_z_before_rad_s"],
                      previous_common_reference_z)
                and close(adapter["wheel_common_reference_z_before_rad_s"],
                          previous_common_reference_z),
                "heldout common reference z before does not continue previous after/reset zero")
        previous_common_reference_z = float(
            calc["calculation"]["wheel_common_reference_z_after_rad_s"])
        require(close(adapter["wheel_common_reference_z_after_rad_s"],
                      previous_common_reference_z),
                "heldout adapter common reference z after differs from independent controller")
        require(equal(calc["calculation"]["consumed_joint_position_rad"],
                      states["qpos"][tick][binding["qpos_addresses"]])
                and equal(calc["calculation"]["consumed_joint_velocity_rad_s"],
                          states["qvel"][tick][binding["dof_addresses"]]),
                "heldout controller used another actual compiled joint state")
        five = native[5*tick:5*tick+5]
        traces = record["native_actuator_traces"]
        interval = info["native_interval_summary"]
        require(len(five) == len(traces) == interval["native_returns"] == 5
                and close(interval["start_time_s"], tick*.01)
                and close(interval["end_time_s"], (tick+1)*.01),
                "heldout control lacks five real native/actuator returns")
        interval_nonwheel = 0
        interval_loads: Counter[str] = Counter()
        torques = []
        for substep, (native_row, trace) in enumerate(zip(five, traces)):
            require(native_row["native_index"] == before["python"]["native_attempted"]
                    + 5*tick + substep
                    and close(native_row["start_time_s"], tick*.01+substep*.002)
                    and close(native_row["end_time_s"], tick*.01+(substep+1)*.002)
                    and native_row["contact_count"] == len(native_row["contacts"]),
                    "heldout actual native index/clock/contact count differs")
            for field in ("qpos", "qvel", "qacc_warmstart"):
                if substep == 0:
                    require(equal(native_row["before"][field], states[field][tick]),
                            "heldout native entry differs from saved endpoint")
                else:
                    require(equal(native_row["before"][field], five[substep-1]["after"][field]),
                            "heldout native integrator chain has discontinuity")
                if substep == 4:
                    require(equal(native_row["after"][field], states[field][tick+1]),
                            "heldout native exit differs from saved endpoint")
            require(equal(native_row["before"]["ctrl"], trace["applied_nm"])
                    and equal(native_row["after"]["ctrl"], trace["applied_nm"])
                    and close(native_row["after"]["actuator_force"], trace["applied_nm"])
                    and all(close(trace[key], calc["computed"]["safe_torque_nm"])
                            for key in ("requested_nm", "limited_nm", "delayed_nm", "applied_nm")),
                    "heldout actual 16D requested/delayed/applied motor chain differs")
            roll, pitch = _roll_pitch_deg(native_row["after"]["qpos"])
            require(close((roll, pitch),
                          (native_row["roll_deg"], native_row["pitch_deg"])),
                    "heldout raw native posture arithmetic differs")
            interval_candidates = 0
            native_positive: Counter[str] = Counter()
            native_ramp_positive: set[str] = set()
            for contact in native_row["contacts"]:
                positive, family = check_contact(contact, geometry, binding)
                if family is not None:
                    candidate = contact["robot_wheel_index"] is None
                    require(contact["nonwheel_ground_candidate"] is candidate
                            and contact["nonwheel_ground_contact"] is candidate
                            and contact["nonwheel_active_solver_contact"] is (
                                candidate and contact["efc_address"] >= 0
                            ), "T nonwheel candidate/active solver classification differs")
                interval_candidates += bool(contact["nonwheel_ground_contact"])
                if positive:
                    native_positive[family] += 1
                    if contact["terrain_geom_name"] in (
                        "terrain_ramp_up", "terrain_ramp_deck", "terrain_ramp_down"
                    ):
                        native_ramp_positive.add(contact["terrain_geom_name"])
                contacts_checked += 1
                rotated_box_contacts += family not in (None, "floor")
            require(interval_candidates == native_row["nonwheel_contact_count"]
                    and dict(native_positive)
                    == native_row["terrain_family_positive_wheel_load"],
                    "heldout native contact/force/family summary differs")
            interval_nonwheel += interval_candidates
            interval_loads.update(native_positive)
            nonwheel += interval_candidates
            family_loads.update({family: 1 for family in native_positive})
            ramp_geom_loads.update({name: 1 for name in native_ramp_positive})
            native_qpos = np.asarray(native_row["after"]["qpos"], dtype=np.float64)
            nheight, nnormal, _ = _ground_hit(
                geometry, native_qpos[0], native_qpos[1],
            )
            up = _rotation(native_qpos[3:7])[:, 2]
            tilt = math.degrees(math.acos(float(np.clip(up @ nnormal, -1.0, 1.0))))
            canonical["native_roll_deg"].append(roll)
            canonical["native_pitch_deg"].append(pitch)
            canonical["native_terrain_relative_tilt_deg"].append(tilt)
            canonical["native_clearance_m"].append(float(native_qpos[2]-nheight))
            canonical["native_forward_projection_m"].append(float(
                forward_axis @ (native_qpos[:2]-initial_xy)
            ))
            canonical["native_base_x_m"].append(float(native_qpos[0]))
            canonical["native_base_y_m"].append(float(native_qpos[1]))
            canonical["native_nonwheel_ground_candidate_count"].append(
                interval_candidates
            )
            canonical["map_escape_count"] += int(
                abs(float(native_qpos[0])) > 10.5 or abs(float(native_qpos[1])) > 5.8
            )
            torques.append(np.asarray(trace["applied_nm"], dtype=np.float64))
        require(interval["nonwheel_contact_count"] == interval_nonwheel
                and interval["terrain_family_positive_wheel_load"] == dict(interval_loads),
                "heldout interval contact/force summary differs")
        metrics = info["metrics"]
        post_qpos, post_qvel = states["qpos"][tick+1], states["qvel"][tick+1]
        vx, vy = _body_com_velocity(post_qpos, post_qvel, binding)[:2]
        abs_roll, abs_pitch = _roll_pitch_deg(post_qpos)
        height, normal, gid = _ground_hit(geometry, post_qpos[0], post_qpos[1])
        yaw = _yaw_from_qpos(post_qpos)
        relative_roll, relative_pitch = _terrain_reference(normal, yaw)
        require(close(vx, metrics["body_com_vx_mps"], atol=1e-6)
                and close(vy, metrics["body_com_vy_mps"], atol=1e-6)
                and close(post_qvel[5], metrics["body_yaw_rate_rps"], atol=1e-6)
                and close(post_qpos[:2], (metrics["x_m"], metrics["y_m"]))
                and close(post_qpos[2]-height, metrics["clearance_m"], atol=1e-6)
                and close(metrics["ground_height_m"], height, atol=1e-6)
                and metrics["ground_geom_id"] == gid
                and close(math.radians(abs_roll)-relative_roll,
                          metrics["relative_roll_rad"], atol=1e-6)
                and close(math.radians(abs_pitch)-relative_pitch,
                          metrics["relative_pitch_rad"], atol=1e-6)
                and close(math.radians(abs_roll),
                          metrics["task_roll_error_rad"], atol=1e-6)
                and close(math.radians(abs_pitch),
                          metrics["task_pitch_error_rad"], atol=1e-6)
                and metrics["reference_roll_rad"] == 0.0
                and metrics["reference_pitch_rad"] == 0.0
                and info["task_schema"] == canonical["task_schema"]
                and info["reference_schema"] == canonical["reference_schema"]
                and info["reward_schema"] == canonical["reward_schema"],
                "heldout actual post COM/yaw/course ground metric differs")
        canonical["com_vx_mps"].append(float(vx))
        canonical["body_yaw_rate_rps"].append(float(post_qvel[5]))
        canonical["clearance_m"].append(float(post_qpos[2]-height))
        canonical["heading_rad"].append(float(yaw-initial_yaw))
        canonical["lateral_offset_m"].append(float(
            lateral_axis @ (post_qpos[:2]-initial_xy)
        ))
        canonical["forward_projection_m"].append(float(
            forward_axis @ (post_qpos[:2]-initial_xy)
        ))
        canonical["applied_servo_vx_mps"].append(servo_vx)
        canonical["applied_servo_yaw_rps"].append(servo_yaw)
        canonical["motor_torque_nm"].append(np.stack(torques).T)
    canonical["positive_wheel_load_native_counts_by_family"] = dict(family_loads)
    canonical["ramp_positive_wheel_load_native_counts_by_geom"] = dict(ramp_geom_loads)
    if terrain == "ramp":
        projection = _projection(folder, binding, geometry, states)
        saved_projection = document(folder / "final_compiled_x_projection.json")
        canonical["ramp_geom_world_x_max_m_by_name"] = {
            name: row["world_x_max_m"] for name, row in saved_projection["ramp"].items()
        }
        canonical["wheel_collision_world_x_min_m_by_index"] = {
            name: row["world_x_min_m"] for name, row in saved_projection["wheel"].items()
        }
        require(projection["all_four_wheel_collision_cylinders_past_all_ramps"]
                is saved_projection["all_wheel_collision_shapes_past_ramp"],
                "11-S full ramp shape clearance claim differs")
    canonical["fall_count"] = int(receipt["stop_reason"] == "fall_or_low_clearance")
    require(receipt["terminated"] == rows[-1]["terminated"]
            and receipt["truncated"] == rows[-1]["truncated"]
            and receipt["stop_reason"] == rows[-1]["info"]["terminal_reason"],
            "heldout genuine terminal state differs from final recorded control")
    summary = {
        "case_id": case_id, "actor": actor, "seed": seed, "terrain": terrain,
        "completed_controls": n, "native_returned": len(native),
        "predict_calls": receipt["policy_predictions"],
        "full_horizon_controls": n == cap,
        "legitimate_task_failure": bool(receipt["terminated"] and n < cap),
        "contacts_checked": contacts_checked,
        "rotated_box_contacts_checked": rotated_box_contacts,
        "nonwheel_candidates": nonwheel,
        "positive_wheel_load_by_family": dict(family_loads),
        "actual_model_address": reset["model_address"],
        "actual_data_address": reset["data_address"],
        "physical_record_integrity_passed": True,
        "task_qualification_pending_pure_scorer": True,
    }
    return summary, canonical


def read_floor(folder: Path, *, expected_seed: int | None = None) -> dict[str, Any]:
    """Read one committed floor folder; legitimate early termination fails gate."""
    if not isinstance(folder, Path) or not folder.is_dir() or folder.is_symlink():
        raise ValueError("floor folder is absent or is a symlink")
    eval_run = folder.parent.parent
    session = plain_document(eval_run / "session.json")
    frozen_seed = session.get("floor_seed")
    require(type(frozen_seed) is int and 0 <= frozen_seed < 2**32
            and (expected_seed is None or expected_seed == frozen_seed),
            "floor seed missing from C18 session or caller differs")
    expected_seed = frozen_seed
    receipt = document(folder / "case_receipt.json")
    saved_schedule = document(folder / "schedule.json")
    geometry = document(folder / "geometry_manifest.json")
    reset = document(folder / "reset_receipt.json")
    before = document(folder / "boundary_before.json")
    initial = arrays(folder / "initial_state.npz")
    states = arrays(folder / "states.npz")
    guard = receipt["native_segment"]
    require(receipt["schema"] == RECORD_SCHEMA
            and receipt["case_id"] == floor_schedule(expected_seed).case_id
            and receipt["actor"] == "grouped_continue"
            and receipt["failure"] is None
            and receipt["record_valid"] is True
            and receipt["termination_recorded"] is True
            and isinstance(guard, dict)
            and guard["record_valid"] is True
            and guard["archive_failure"] is None,
            "floor software/archive record is invalid")
    controls = control_rows(folder, receipt["controller_record_blocks"])
    native = native_rows(folder, guard)
    n = receipt["completed_controls"]
    require(type(n) is int and 1 <= n <= 600
            and len(controls) == n and len(native) == 5*n
            and all(states[key].shape[0] == n+1
                    for key in (*PAIR_FIELDS, "time"))
            and all(np.array_equal(states[key][0], initial[key])
                    for key in PAIR_FIELDS)
            and reset["seed"] == receipt["seed"]
            and reset["before"] == before
            and reset["model_address"] > 0 and reset["data_address"] > 0,
            "floor saved endpoint/reset chain differs")
    boundary_delta(before, receipt["boundary_after"], n)
    identity_doc = document(eval_run / "eval_identity.json")
    binding = check_binding(identity_doc["actual_geometry_binding"])
    full_summary, full_canonical = heldout_case(
        eval_run, ("floor_0p4_600", "flat", .4, expected_seed, 600, 325, 425),
        receipt["actor"], binding, identity_doc["compiled_geometry"],
        expected_variant=session["controller_variant"],
        folder_override=folder,
    )
    require(full_summary["physical_record_integrity_passed"] is True
            and full_canonical["completed_controls"] == n,
            "floor controller/native/contact/force chain differs")
    require(saved_schedule["seed"] == expected_seed
            and saved_schedule["raw_command_sha256"] == floor_schedule(expected_seed).command_sha256,
            "floor saved schedule differs from session seed")
    result = score18.score_floor_canonical(full_canonical, expected_seed=expected_seed)
    native_clearance = min([full_canonical["initial_clearance_m"],
                            *full_canonical["native_clearance_m"],
                            *full_canonical["clearance_m"]])
    result["terrain_aware_min_initial_native_clearance_m"] = native_clearance
    if native_clearance < .28:
        result["numeric_gate_passed"] = False
        result["reasons"].append("compiled-terrain clearance below 0.28 m")
    result["archive_manifests_verified"] = True
    result["five_initial_state_arrays_matched"] = True
    result["C_Python_5T_boundary_verified"] = True
    result["controller_native_contact_force_chain_verified"] = True
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_spec(row: dict) -> tuple[tuple, bool, tuple[str, ...]]:
    """Bind one session row to the unchanged task pattern and explicit seed."""
    require(type(row) is dict and set(row) == {"case_id", "seed", "mirror", "actors", "horizon"},
            "C18 case specification fields differ")
    case_id, seed, mirror = row["case_id"], row["seed"], row["mirror"]
    require(case_id in TASK_BY_ID and type(seed) is int and 0 <= seed < 2**32
            and type(mirror) is bool and (not mirror or case_id == "flat_1p2_yaw")
            and row["horizon"] == TASK_BY_ID[case_id].horizon,
            "C18 task/seed/mirror/horizon differs")
    actors = row["actors"]
    require(actors in (["grouped_continue"], ["zero", "grouped_continue"]),
            "C18 actor order must be grouped or paired zero then grouped")
    task = TASK_BY_ID[case_id]
    spec = (case_id, task.terrain, task.speed_mps, seed,
            task.horizon, task.hold_start, task.release_tick)
    return spec, mirror, tuple(actors)


def read_run(run: Path) -> dict[str, Any]:
    """Independently verify one frozen C18 worker and score its saved cases."""
    if not isinstance(run, Path) or not run.is_dir() or run.is_symlink():
        raise ValueError("C18 run is absent or symlinked")
    session = plain_document(run / "session.json")
    worker = plain_document(run / "worker_receipt.json")
    host = plain_document(run / "host_receipt.json")
    arm = session.get("arm")
    variant = session.get("controller_variant")
    specs = session.get("case_specs")
    require(type(arm) is str and arm
            and session.get("schema") == "d1-control-repair-session-18-v1"
            and variant in ("baseline", "yaw", "ramp", "combined")
            and type(session.get("floor_seed")) is int
            and 0 <= session["floor_seed"] < 2**32
            and type(specs) is list and specs
            and session.get("retry_permitted") is False
            and worker.get("schema") == "d1-control-repair-worker-18-v1"
            and worker.get("arm") == arm
            and worker.get("session_identity") == identity(run / "session.json")
            and worker.get("execution_complete") is True
            and worker.get("failure") is None
            and worker.get("cleanup_errors") == []
            and worker.get("warnings") == []
            and worker.get("archive_failed") is False
            and worker.get("control_step_caller_verified") is True
            and host.get("arm") == arm and host.get("exit_code") == 0
            and host.get("failure") is None and host.get("cleanup_errors") == []
            and host.get("changed_sources") == []
            and host.get("postcheck_complete") is True
            and host.get("no_live_owned_processes") is True
            and host.get("fully_reserved_budget_closed") is True
            and host.get("retry_permitted") is False
            and host.get("elapsed_s", float("inf")) <= session["hard_s"],
            "C18 worker/host/session closure failed")
    parsed = [_case_spec(row) for row in specs]
    require(len({spec[0][0] for spec in parsed}) == len(parsed),
            "duplicate case ID within one C18 worker")
    reserved = 600 + sum(spec[0][4] * len(spec[2]) for spec in parsed)
    require(session["control_limit"] == reserved
            and session["normal_native_limit"] == 5*reserved
            and session["compiler_native_limit"] == 2
            and session["segments"] == [600] + [spec[0][4] for spec in parsed
                                                  for _actor in spec[2]],
            "C18 frozen control/native/segment reservation differs")
    plan_path = Path(session["plan_path"])
    plan = plain_document(plan_path)
    require(identity(plan_path) == session["plan_identity"]
            and plan["schema"] == "d1-control-repair-plan-18-v1"
            and plan["status"] == "GO"
            and plan["retry_permitted"] is False
            and arm in plan["arms"]
            and plan["arms"][arm]["case_specs"] == specs
            and plan["arms"][arm]["floor_seed"] == session["floor_seed"]
            and plan["arms"][arm]["controller_variant"] == variant
            and plan["arms"][arm]["controls"] == reserved,
            "C18 source plan/case budget differs")
    identity_doc = document(run / "eval_identity.json")
    source_hashes = identity_doc["source_hashes"]
    require(identity_doc["execution_stage"] == 18
            and identity_doc["controller_variant"] == variant
            and source_hashes == session["source_hashes"]
            and isinstance(source_hashes, dict) and source_hashes,
            "C18 evaluation source/variant identity differs")
    for name, expected in source_hashes.items():
        path = Path(name)
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and identity(path) == {key: expected[key] for key in ("sha256", "bytes")},
                "frozen source differs: " + name)
    for critical in ("read_eval18.py", "score18.py", "recipes18.py", "verify_control18.py",
                     "controller18.py", "residual18.py",
                     "floor_bridge18.py", "offline_floor18.py", "host18.py", "worker18.py",
                     "eval18.py"):
        require(str(HERE / critical) in source_hashes,
                "C18 critical source missing from closure: " + critical)
    expected_sha = identity_doc["checkpoint_sha256_by_actor"]
    paths = identity_doc["checkpoint_paths_by_actor"]
    require(set(expected_sha) == set(paths) == {"grouped_continue"},
            "C18 checkpoint table must contain the unchanged grouped actor only")
    checkpoint = Path(paths["grouped_continue"])
    manifest = session["eval_manifests"]["grouped"]
    require(checkpoint.is_absolute() and checkpoint.is_file() and not checkpoint.is_symlink()
            and _file_sha256(checkpoint) == expected_sha["grouped_continue"]
            and expected_sha["grouped_continue"] == manifest["files"]["final_model.zip"]["sha256"]
            and checkpoint == Path(manifest["folder"]) / "final_model.zip",
            "C18 grouped policy checkpoint changed")
    require(identity_doc["warnings"] == worker["warnings"] == [],
            "C18 worker recorded a warning")
    binding = check_binding(identity_doc["actual_geometry_binding"])
    geometry = identity_doc["compiled_geometry"]
    folder = run / "floor" / "grouped_continue"
    floor = read_floor(folder, expected_seed=session["floor_seed"])
    require(floor["experiment_actor"] == "grouped_continue"
            and floor["checkpoint_sha256"] == expected_sha["grouped_continue"]
            and document(folder / "geometry_manifest.json") == geometry
            and document(run / "grouped_floor_gate.json") == floor
            and worker["result"]["floors"] == {"grouped": floor},
            "C18 floor independent gate/worker/geometry differs")
    enabled = floor["numeric_gate_passed"] is True
    expected_members = [(spec, mirror, actor) for spec, mirror, actors in parsed
                        for actor in actors] if enabled else []
    summaries, canonical, saved = [], [], []
    for spec, mirror, actor in expected_members:
        summary, numeric = heldout_case(run, spec, actor, binding, geometry,
                                        expected_variant=variant, mirror=mirror)
        require(numeric["checkpoint_sha256"] == expected_sha.get(actor)
                and summary["actual_model_address"] > 0
                and summary["actual_data_address"] > 0,
                "C18 task actor/checkpoint identity differs")
        summaries.append(summary)
        canonical.append(numeric)
        receipt = document(run / "heldout" / f"{spec[0]}_{actor}" / "case_receipt.json")
        saved.append({key: receipt[key] for key in (
            "case_id", "actor", "seed", "completed_controls", "record_valid",
            "terminated", "truncated", "stop_reason", "policy_predictions",
            "checkpoint_sha256")})
    for spec, _mirror, actors in (parsed if enabled else []):
        if actors != ("zero", "grouped_continue"):
            continue
        zero_folder = run / "heldout" / f"{spec[0]}_zero"
        grouped_folder = run / "heldout" / f"{spec[0]}_grouped_continue"
        first = arrays(zero_folder / "initial_state.npz")
        second = arrays(grouped_folder / "initial_state.npz")
        zero_reset = document(zero_folder / "reset_receipt.json")
        grouped_reset = document(grouped_folder / "reset_receipt.json")
        require(zero_reset["exact_initial_pair"] is False
                and grouped_reset["exact_initial_pair"] is True
                and grouped_reset["paired_with"] == "zero"
                and all(zero_reset[key] == grouped_reset[key]
                        for key in ("model_address", "data_address"))
                and all(first[key].dtype == second[key].dtype
                        and first[key].shape == second[key].shape
                        and first[key].tobytes() == second[key].tobytes()
                        for key in PAIR_FIELDS),
                "C18 paired zero/grouped initial state differs bitwise")
    scored = [dict(score18.score_case(case, expected_seed=case["seed"],
                                      mirror=case["mirror"]),
                   experiment_actor=case["experiment_actor"],
                   checkpoint_sha256=case["checkpoint_sha256"])
              for case in canonical]
    pair_score = None
    if (enabled and len(parsed) == 6
            and all(actors == ("zero", "grouped_continue")
                    for _spec, _mirror, actors in parsed)
            and all(not mirror for _spec, mirror, _actors in parsed)
            and {spec[0][0] for spec in parsed} == set(ORDER)):
        pair_score = score18.score_pairs(scored)
    observed = floor["completed_controls"] + sum(row["completed_controls"] for row in summaries)
    c_final, python, native_guard = (worker[key] for key in
                                     ("C_final", "python", "native_guard"))
    require(observed <= reserved
            and worker["result"]["completed_controls"] == observed
            and worker["result"]["enabled_arms"] == (["grouped"] if enabled else [])
            and worker["result"]["cases"] == saved
            and worker["result"]["full_comparison_executed"] is (len(summaries) ==
                sum(len(actors) for _spec, _mirror, actors in parsed))
            and c_final["construction_attempts"] == c_final["construction_returns"] == 2
            and c_final["control_attempts"] == c_final["control_returns"] == 5*observed
            and c_final["phase"] == c_final["target_model"] == c_final["target_data"] == 0
            and c_final["violations"] == 0
            and python["control_attempted"] == python["control_completed"] == observed
            and python["native_attempted"] == python["native_returned"]
            == python["clock_advanced_substeps"] == 5*observed
            and python["forbidden_entries"] == 0
            and native_guard["native_attempted"] == native_guard["native_returned"]
            == native_guard["native_checked"] == 5*observed
            and native_guard["failure"] is None,
            "C18 actual C/Python/native/compiler counts differ")
    calls = worker["model_calls"]["counts"]
    predicted = floor["completed_controls"] + sum(row["predict_calls"] for row in summaries)
    expected_calls = {"load": 1, "torch_load": 3, "save": 0, "learn": 0,
                      "train": 0, "forward": 0, "evaluate_actions": 0,
                      "predict_values": 0, "backward": 0, "predict": 1 + predicted}
    require(all(calls.get(key, {}).get("attempted", 0)
                == calls.get(key, {}).get("returned", 0) == value
                for key, value in expected_calls.items()),
            "C18 actual model API counts differ from one probe and grouped controls")
    return {"schema": READ_SCHEMA, "run": str(run.resolve()),
            "controller_variant": variant, "source_closure_verified": True,
            "checkpoint_ZIP_verified": True, "floor": floor,
            "heldout_summaries": summaries, "numeric_scores": scored,
            "six_task_zero_pair": pair_score,
            "actual_completed_controls": observed,
            "actual_normal_native_returns": 5*observed,
            "actual_compiler_native_returns": 2,
            "physical_calls_performed_by_reader": 0,
            "model_loaded_by_reader": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = read_run(args.run.resolve())
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
