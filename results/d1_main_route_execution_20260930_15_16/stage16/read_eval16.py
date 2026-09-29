"""Independent stage-16 saved-evaluation reader; no model or physics import.

Only stage identity and the location of the frozen stage15 readers differ.
All numeric thresholds and controller/native verification remain unchanged.

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

HERE = Path(__file__).resolve().parent.parent / "continuation15"
W = HERE.parent
for directory in (W, W / "course_impl08", W / "rl11", W / "continuation13"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from archive13.atomic_archive_13 import verify_manifest
import score15
from recipes15 import ORDER, TASKS, floor_schedule, heldout_schedule as make_schedule
from readback15 import score_canonical_cases, score_floor_saved
from heldout15 import PAIR_FIELDS, RECORD_SCHEMA
from rl11.verify_short_rl16_training_11_04 import boundary_delta, check_controller
from upright11.verify_world_upright_reference_11 import _projection
from verify_course_e_08_03 import (
    _body_com_velocity, _native_rows, _roll_pitch_deg, _rotation,
    check_binding, check_contact, close, equal, geom_map, identity, require,
)
from verify_rl16_training_08 import _ground_hit, _terrain_reference, _yaw_from_qpos

READ_SCHEMA = "d1-world-upright-stage16-independent-eval-readback-v1"


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


def heldout_schedule(folder: Path, spec: tuple, actor: str) -> dict:
    case_id, terrain, speed, seed, cap, hold_start, release = spec
    schedule = floor_schedule() if case_id == "floor_0p4_600" else make_schedule(case_id)
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
                 folder_override: Path | None = None) -> tuple[dict, dict]:
    case_id, terrain, speed, seed, cap, _hold, _release = spec
    folder = (run / "heldout" / f"{case_id}_{actor}"
              if folder_override is None else folder_override)
    receipt = document(folder / "case_receipt.json")
    schedule = heldout_schedule(folder, spec, actor)
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
        "terrain": terrain, "seed": seed,
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
        calc = check_controller(record, tick)
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


def read_floor(folder: Path) -> dict[str, Any]:
    """Read one committed floor folder; legitimate early termination fails gate."""
    if not isinstance(folder, Path) or not folder.is_dir() or folder.is_symlink():
        raise ValueError("floor folder is absent or is a symlink")
    receipt = document(folder / "case_receipt.json")
    saved_schedule = document(folder / "schedule.json")
    geometry = document(folder / "geometry_manifest.json")
    reset = document(folder / "reset_receipt.json")
    before = document(folder / "boundary_before.json")
    initial = arrays(folder / "initial_state.npz")
    states = arrays(folder / "states.npz")
    guard = receipt["native_segment"]
    require(receipt["schema"] == RECORD_SCHEMA
            and receipt["case_id"] == floor_schedule().case_id
            and receipt["actor"] in ("global_continue", "grouped_continue")
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
    eval_run = folder.parent.parent
    identity_doc = document(eval_run / "eval_identity.json")
    binding = check_binding(identity_doc["actual_geometry_binding"])
    full_summary, full_canonical = heldout_case(
        eval_run, ("floor_0p4_600", "flat", .4, 151090, 600, 325, 425),
        receipt["actor"], binding, identity_doc["compiled_geometry"],
        folder_override=folder,
    )
    require(full_summary["physical_record_integrity_passed"] is True
            and full_canonical["completed_controls"] == n,
            "floor controller/native/contact/force chain differs")
    result = score_floor_saved(
        receipt, saved_schedule, states, controls, native, geometry,
        warning_count=0, geometry_invalid_count=0,
    )
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


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator / denominator)


def _no_degradation(numerator: float | None, denominator: float | None,
                    threshold: float) -> bool:
    if numerator is None or denominator is None:
        return False
    if denominator == 0:
        return numerator == 0
    return numerator / denominator <= threshold


def _hold_sse(canonical: dict, case_id: str) -> float | None:
    begin, end = score15.case_windows(case_id)["hold"]
    if canonical["completed_controls"] < end:
        return None
    vx = np.asarray(canonical["com_vx_mps"], dtype=np.float64)[begin:end]
    servo = np.asarray(canonical["applied_servo_vx_mps"], dtype=np.float64)[begin:end]
    yaw = np.asarray(canonical["body_yaw_rate_rps"], dtype=np.float64)[begin:end]
    servo_yaw = np.asarray(canonical["applied_servo_yaw_rps"], dtype=np.float64)[begin:end]
    return float(np.sum(np.square((vx-servo)/.25) + np.square((yaw-servo_yaw)/.4)))


def grouped_global_promotion(scored: dict, canonical: list[dict],
                             floors: dict[str, dict]) -> dict[str, Any]:
    """Apply the preregistered grouped/global physical promotion arithmetic."""
    by_score = {(row["case_id"], row["experiment_actor"]): row
                for row in scored["scores"]}
    by_case = {(row["case_id"], row["experiment_actor"]): row
               for row in canonical}
    reasons = []
    if not scored["all_three_actors_present"]:
        reasons.append("both continuation actors did not complete six-task evaluation")
    if (floors["global_continue"]["numeric_gate_passed"] is not True
            or floors["grouped_continue"]["numeric_gate_passed"] is not True):
        reasons.append("both new finals did not pass the floor gate")
    if reasons:
        return {"passed": False, "reasons": reasons,
                "common_task_count": 0, "pooled_drive_sse_ratio": None,
                "pooled_torque_cost_ratio": None, "hold_diagnostics": {}}
    common = [case_id for case_id in ORDER
              if by_score[(case_id, "global_continue")]["task_passed"]
              and by_score[(case_id, "grouped_continue")]["task_passed"]]
    grouped_all = all(by_score[(case_id, "grouped_continue")]["task_passed"]
                      for case_id in ORDER)
    if not grouped_all:
        reasons.append("grouped did not pass all six tasks")
    if len(common) < 4:
        reasons.append("fewer than four jointly passed tasks")
    drive = {}
    individual_good = 0
    for case_id in common:
        global_sse = by_score[(case_id, "global_continue")]["rl_terms"]["sse_total"]
        grouped_sse = by_score[(case_id, "grouped_continue")]["rl_terms"]["sse_total"]
        ratio = _ratio(grouped_sse, global_sse)
        good = _no_degradation(grouped_sse, global_sse, 1.02)
        individual_good += int(good)
        drive[case_id] = {"global_sse": global_sse, "grouped_sse": grouped_sse,
                          "ratio": ratio, "at_most_1p02": good}
    global_sum = sum(drive[key]["global_sse"] for key in common)
    grouped_sum = sum(drive[key]["grouped_sse"] for key in common)
    pooled_sse = _ratio(grouped_sum, global_sum)
    if pooled_sse is None or pooled_sse > .85:
        reasons.append("pooled common-task drive SSE ratio exceeds 0.85 or is undefined")
    if individual_good < 4:
        reasons.append("fewer than four individual drive SSE ratios are at most 1.02")
    diagnostic = {}
    for case_id in ORDER[:4]:
        global_hold = _hold_sse(by_case[(case_id, "global_continue")], case_id)
        grouped_hold = _hold_sse(by_case[(case_id, "grouped_continue")], case_id)
        good = _no_degradation(grouped_hold, global_hold, 1.02)
        diagnostic[case_id] = {"global_hold_sse": global_hold,
                               "grouped_hold_sse": grouped_hold,
                               "ratio": _ratio(grouped_hold, global_hold),
                               "at_most_1p02": good}
        if not good:
            reasons.append(case_id + " hold SSE regression or incomplete hold")
    high_global = by_score[("flat_1p6", "global_continue")]["speed"][
        "rms_com_vx_error_vs_target_mps"]
    high_grouped = by_score[("flat_1p6", "grouped_continue")]["speed"][
        "rms_com_vx_error_vs_target_mps"]
    high_ratio = _ratio(high_grouped, high_global)
    high_good = high_ratio is not None and high_ratio <= .90
    if not high_good:
        reasons.append("flat_1p6 hold vx RMS ratio exceeds 0.90 or is undefined")
    cost_complete = all(by_score[(key, actor)]["rl_terms"]["drive_complete"]
                        for key in ORDER for actor in ("global_continue", "grouped_continue"))
    cost_by_task = {}
    for key in ORDER:
        global_cost = by_score[(key, "global_continue")]["rl_terms"]["torque_cost_sum"]
        grouped_cost = by_score[(key, "grouped_continue")]["rl_terms"]["torque_cost_sum"]
        cost_by_task[key] = {"global": global_cost, "grouped": grouped_cost,
                             "ratio": _ratio(grouped_cost, global_cost),
                             "both_drive_complete": all(
                                 by_score[(key, actor)]["rl_terms"]["drive_complete"]
                                 for actor in ("global_continue", "grouped_continue"))}
    cost_global = sum(row["global"] for row in cost_by_task.values())
    cost_grouped = sum(row["grouped"] for row in cost_by_task.values())
    pooled_cost = _ratio(cost_grouped, cost_global)
    if not cost_complete:
        reasons.append("a six-task paired drive interval is incomplete")
    if pooled_cost is None or pooled_cost > 1.20:
        reasons.append("pooled six-task torque cost ratio exceeds 1.20 or is undefined")
    return {
        "passed": not reasons, "reasons": reasons,
        "common_task_count": len(common), "common_tasks": common,
        "pooled_drive_sse_ratio": pooled_sse,
        "individual_drive_sse_ratios_at_most_1p02": individual_good,
        "drive_sse_by_task": drive,
        "hold_diagnostics": diagnostic,
        "flat_1p6_vx_rms_ratio": high_ratio,
        "flat_1p6_vx_rms_gate": high_good,
        "pooled_torque_cost_ratio": pooled_cost,
        "six_task_drive_cost_complete": cost_complete,
        "torque_cost_by_task": cost_by_task,
        "torque_cost_is_normalized_squared_proxy_not_energy": True,
    }


def verify_eval(run: Path) -> dict[str, Any]:
    """Verify saved evaluation archives and score all permitted task cases."""
    if not run.is_dir() or run.is_symlink():
        raise ValueError("stage-15 eval run is absent or symlinked")
    session = plain_document(run / "session.json")
    worker = plain_document(run / "worker_receipt.json")
    host = plain_document(run / "host_receipt.json")
    require(session.get("schema") == "d1-groupclip-eval-session-16-v1"
            and session.get("arm") == "eval"
            and session.get("control_limit") == 30600
            and session.get("normal_native_limit") == 153000
            and session.get("compiler_native_limit") == 2
            and session.get("retry_permitted") is False
            and worker.get("schema") == "d1-groupclip-eval-worker-16-v1"
            and worker.get("arm") == "eval"
            and worker.get("session_identity") == identity(run / "session.json")
            and worker.get("execution_complete") is True
            and worker.get("failure") is None
            and worker.get("cleanup_errors") == []
            and worker.get("warnings") == []
            and worker.get("archive_failed") is False
            and worker.get("control_step_caller_verified") is True
            and host.get("arm") == "eval"
            and host.get("exit_code") == 0
            and host.get("failure") is None
            and host.get("cleanup_errors") == []
            and host.get("changed_sources") == []
            and host.get("postcheck_complete") is True
            and host.get("no_live_owned_processes") is True
            and host.get("fully_reserved_budget_closed") is True
            and host.get("retry_permitted") is False
            and host.get("elapsed_s", float("inf")) <= session["hard_s"],
            "eval worker/host source, cleanup or physical owner did not close")
    identity_doc = document(run / "eval_identity.json")
    source_hashes = identity_doc["source_hashes"]
    require(source_hashes == session["source_hashes"],
            "eval identity source closure differs from reserved session")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("eval source closure is absent")
    for name, expected in source_hashes.items():
        path = Path(name)
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and identity(path) == {key: expected[key] for key in ("sha256", "bytes")},
                "eval frozen source differs: " + name)
    for path in (Path(__file__).resolve(), HERE / "read_eval15.py", HERE / "readback15.py",
                 HERE / "heldout15.py", HERE / "recipes15.py", HERE / "score15.py"):
        require(str(path) in source_hashes, "eval critical reader/recipe source is unfrozen")
    expected_sha = identity_doc["checkpoint_sha256_by_actor"]
    checkpoint_paths = identity_doc["checkpoint_paths_by_actor"]
    require(set(expected_sha) == set(checkpoint_paths)
            == {"global_continue", "grouped_continue"},
            "eval two checkpoint identity table differs")
    for actor, path_text in checkpoint_paths.items():
        path = Path(path_text)
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and _file_sha256(path) == expected_sha[actor],
                "eval continuation checkpoint ZIP differs: " + actor)
    binding = check_binding(identity_doc["actual_geometry_binding"])
    geometry = identity_doc["compiled_geometry"]
    require(identity_doc["warnings"] == worker["warnings"] == [],
            "eval actual worker recorded a warning")
    floors = {}
    for actor in ("global_continue", "grouped_continue"):
        folder = run / "floor" / actor
        floor = read_floor(folder)
        require(floor["experiment_actor"] == actor
                and floor["checkpoint_sha256"] == expected_sha[actor]
                and document(folder / "geometry_manifest.json") == geometry,
                "eval floor actor/checkpoint/geometry differs")
        floors[actor] = floor
    floor_global = arrays(run / "floor" / "global_continue" / "initial_state.npz")
    floor_grouped = arrays(run / "floor" / "grouped_continue" / "initial_state.npz")
    floor_reset_global = document(run / "floor" / "global_continue" / "reset_receipt.json")
    floor_reset_grouped = document(run / "floor" / "grouped_continue" / "reset_receipt.json")
    require(floor_reset_global["exact_initial_pair"] is False
            and floor_reset_grouped["exact_initial_pair"] is True
            and floor_reset_grouped["paired_with"] == "global_continue"
            and all(floor_reset_global[key] == floor_reset_grouped[key]
                    for key in ("model_address", "data_address"))
            and all(floor_global[key].dtype == floor_grouped[key].dtype
                    and floor_global[key].shape == floor_grouped[key].shape
                    and floor_global[key].tobytes() == floor_grouped[key].tobytes()
                    for key in PAIR_FIELDS),
            "eval two final floor resets differ bitwise")
    enabled = tuple(actor for actor in ("global_continue", "grouped_continue")
                    if floors[actor]["numeric_gate_passed"])
    for actor in ("global_continue", "grouped_continue"):
        arm = actor.removesuffix("_continue")
        require(document(run / f"{arm}_floor_gate.json") == floors[actor]
                and worker["result"]["floors"][arm] == floors[actor],
                "eval worker floor gate differs from independent reader")
    summaries, canonical, saved_case_summaries = [], [], []
    for task in (TASKS if enabled else ()):
        spec = (task.case_id, task.terrain, task.speed_mps, task.seed,
                task.horizon, task.hold_start, task.release_tick)
        for actor in ("zero", *enabled):
            summary, numeric = heldout_case(run, spec, actor, binding, geometry)
            require(numeric["checkpoint_sha256"] == expected_sha.get(actor)
                    and summary["actual_model_address"] > 0
                    and summary["actual_data_address"] > 0,
                    "eval task actor/checkpoint identity differs")
            summaries.append(summary)
            canonical.append(numeric)
            saved_receipt = document(run / "heldout" /
                                     f"{task.case_id}_{actor}" / "case_receipt.json")
            saved_case_summaries.append({key: saved_receipt[key] for key in (
                "case_id", "actor", "seed", "completed_controls", "record_valid",
                "terminated", "truncated", "stop_reason", "policy_predictions",
                "checkpoint_sha256",
            )})
        if enabled:
            zero_folder = run / "heldout" / f"{task.case_id}_zero"
            first = arrays(zero_folder / "initial_state.npz")
            zero_reset = document(zero_folder / "reset_receipt.json")
            require(zero_reset["exact_initial_pair"] is False,
                    "eval zero initial reset was not the pair anchor")
            for actor in enabled:
                folder = run / "heldout" / f"{task.case_id}_{actor}"
                next_initial = arrays(folder / "initial_state.npz")
                reset = document(folder / "reset_receipt.json")
                require(reset["exact_initial_pair"] is True
                        and reset["paired_with"] == "zero"
                        and all(reset[key] == zero_reset[key]
                                for key in ("model_address", "data_address"))
                        and all(first[key].dtype == next_initial[key].dtype
                                and first[key].shape == next_initial[key].shape
                                and first[key].tobytes() == next_initial[key].tobytes()
                                for key in PAIR_FIELDS),
                        "eval three-actor initial state pair differs bitwise")
    scored = (score_canonical_cases(
        canonical, checkpoint_sha256_by_actor={actor: expected_sha[actor]
                                               for actor in enabled},
    ) if enabled else None)
    promotion = (grouped_global_promotion(scored, canonical, floors)
                 if scored is not None else {
                     "passed": False, "reasons": ["neither final passed floor"]})
    observed_controls = sum(floor["completed_controls"] for floor in floors.values())
    observed_controls += sum(row["completed_controls"] for row in summaries)
    c_final, python, native_guard = (worker[key] for key in
                                     ("C_final", "python", "native_guard"))
    require(observed_controls <= 30600
            and worker["result"]["completed_controls"] == observed_controls
            and worker["result"]["enabled_arms"]
            == [actor.removesuffix("_continue") for actor in enabled]
            and worker["result"]["cases"] == saved_case_summaries
            and worker["result"]["full_comparison_executed"] is (len(summaries) == 18)
            and c_final["construction_attempts"] == c_final["construction_returns"] == 2
            and c_final["control_attempts"] == c_final["control_returns"]
            == 5*observed_controls
            and c_final["phase"] == c_final["target_model"] == c_final["target_data"] == 0
            and c_final["violations"] == 0
            and python["control_attempted"] == python["control_completed"]
            == observed_controls
            and python["native_attempted"] == python["native_returned"]
            == python["clock_advanced_substeps"] == 5*observed_controls
            and python["forbidden_entries"] == 0
            and native_guard["native_attempted"] == native_guard["native_returned"]
            == native_guard["native_checked"] == 5*observed_controls
            and native_guard["failure"] is None,
            "eval actual C/Python/native/compiler counts do not match saved cases")
    calls = worker["model_calls"]["counts"]
    predicted = sum(floor["completed_controls"] for floor in floors.values())
    predicted += sum(row["predict_calls"] for row in summaries)
    expected_calls = {"load": 2, "torch_load": 6, "save": 0,
                      "learn": 0, "train": 0, "forward": 0,
                      "evaluate_actions": 0, "predict_values": 0,
                      "backward": 0, "predict": 2 + predicted}
    require(all(calls.get(key, {}).get("attempted", 0)
                == calls.get(key, {}).get("returned", 0) == value
                for key, value in expected_calls.items()),
            "eval actual model API counts differ from two probes and saved cases")
    return {
        "schema": READ_SCHEMA, "run": str(run.resolve()),
        "source_closure_verified": True, "checkpoint_ZIPs_verified": True,
        "floor": floors, "heldout_summaries": summaries,
        "actual_completed_controls": observed_controls,
        "actual_normal_native_returns": 5*observed_controls,
        "actual_compiler_native_returns": 2,
        "numeric_scores": scored, "grouped_vs_global_promotion": promotion,
        "zero_contribution_by_actor": (
            {} if scored is None else {
                actor: scored["zero_pairs"][actor]["RL_contribution_passed"]
                for actor in enabled
            }
        ),
        "physical_calls_performed_by_reader": 0,
        "model_loaded_by_reader": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_eval(args.run.resolve())
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
