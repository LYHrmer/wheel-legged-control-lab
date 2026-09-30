"""Independent C20 saved-evaluation reader; no model or physics import.

The C18 physical/native chain is retained; C20 session seeds and C18 controller
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
                  W / "continuation15", W / "continuation18"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from archive13.atomic_archive_13 import verify_manifest
import score20
from recipes20 import SPEC, CASE_SPECS, ORDER, TASK_BY_ID, floor_schedule, make_schedule
from heldout20 import PAIR_FIELDS, RECORD_SCHEMA
from rl11.verify_short_rl16_training_11_04 import boundary_delta
from verify_control18 import check_controller
from upright11.verify_world_upright_reference_11 import _projection
from verify_course_e_08_03 import (
    _body_com_velocity, _native_rows, _roll_pitch_deg, _rotation,
    check_binding, check_contact, close, equal, geom_map, identity, require,
)
from verify_rl16_training_08 import _ground_hit, _terrain_reference, _yaw_from_qpos

READ_SCHEMA = "d1-world-upright-stage20-independent-eval-readback-v1"


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
    if case_id != "floor_0p4_600":
        require(saved["raw_commands"] == score20.expected_raw_commands(case_id),
                "recipe and independent scorer fixed command values differ")
    return saved


def _reset_state(reset: dict) -> dict:
    state = reset.get("control_reset_state")
    required = {"variant", "wheel_integral_nm", "wheel_common_reference_z_rad_s",
        "previous_servo_forward_mps", "stop_latched", "servo_last_tick", "servo_applied",
        "servo_receipt", "provider_sequence", "provider_control_time_s", "previous_action", "context"}
    require(type(state) is dict and set(state) == required,
            "C20 complete controller/provider reset snapshot is missing")
    zero = {"forward_velocity_mps": 0., "lateral_velocity_mps": 0., "yaw_rate_rps": 0.,
            "clearance_m": .455, "jump_requested": False}
    require(state["variant"] == "combined" and equal(state["wheel_integral_nm"], np.zeros(4))
            and state["wheel_common_reference_z_rad_s"] == 0.
            and state["previous_servo_forward_mps"] is None and state["stop_latched"] is False
            and state["servo_last_tick"] == 0 and state["servo_applied"] == zero
            and type(state["servo_receipt"]) is dict
            and state["servo_receipt"].get("tick") == 0
            and state["servo_receipt"].get("applied") == zero
            and state["servo_receipt"].get("raw_target") == zero and state["provider_sequence"] == 0
            and state["provider_control_time_s"] == 0.
            and np.asarray(state["previous_action"]).ndim == 1
            and np.asarray(state["previous_action"]).size == 16
            and np.all(np.asarray(state["previous_action"]) == 0)
            and type(state["context"]) is dict,
            "C20 controller/servo/provider reset state differs from clean reset")
    return state


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
    _reset_state(reset)
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
    require(equal(initial["observation"][59:63], np.zeros(4))
            and equal(initial["observation"][63:79], np.zeros(16))
            and equal(initial["observation"][63:79], reset["control_reset_state"]["previous_action"]),
            "initial observation PI/previous-action reset slots differ")
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
        if tick == 0:
            require(reset["control_reset_state"]["servo_receipt"] == info["servo_receipt"],
                    "reset servo receipt differs from the actually consumed first tick")
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
            and receipt["actor"] in ("A", "B")
            and folder.name == receipt["actor"]
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
    result = score20.score_floor_canonical(full_canonical, expected_seed=expected_seed)
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
    require(type(row) is dict and row.get("case_id") in CASE_SPECS,
            "unknown C20 preregistered case")
    case_id = row["case_id"]
    frozen = CASE_SPECS[case_id]
    actors = ["A", "B"] if "source_case_id" in frozen else ["zero", "old", "A", "B"]
    require(row == dict(frozen, actors=actors),
            "C20 case values/seed/command/windows/actors differ from the scientific spec")
    task = TASK_BY_ID[case_id]
    spec = (case_id, task.terrain, task.speed_mps, task.seed,
            task.horizon, task.hold_start, task.release_tick)
    return spec, frozen.get("mirror", False), tuple(actors)


def _same_arrays(first: dict, second: dict) -> bool:
    return all(first[key].dtype == second[key].dtype
               and first[key].shape == second[key].shape
               and first[key].tobytes() == second[key].tobytes() for key in PAIR_FIELDS)


def _new_pairs(run: Path, parsed, enabled) -> list[dict]:
    pairs = []
    for spec, _mirror, actors in parsed:
        eligible = [actor for actor in actors if actor not in ("A", "B") or actor in enabled]
        if not eligible:
            continue
        folders = [run / "heldout" / f"{spec[0]}_{actor}" for actor in eligible]
        resets = [document(folder / "reset_receipt.json") for folder in folders]
        initial = [arrays(folder / "initial_state.npz") for folder in folders]
        first_state = _reset_state(resets[0])
        require(resets[0]["exact_initial_pair"] is False
                and resets[0]["paired_with"] is None, "first C20 actor unexpectedly paired")
        for index in range(1, len(folders)):
            require(resets[index]["exact_initial_pair"] is True
                    and resets[index]["paired_with"] == eligible[0]
                    and all(resets[index][key] == resets[0][key]
                            for key in ("model_address", "data_address"))
                    and _same_arrays(initial[0], initial[index])
                    and _reset_state(resets[index]) == first_state,
                    "C20 actor initial arrays or complete control/provider state differ")
        pairs.append({"case_id": spec[0], "actors": eligible,
                      "five_initial_arrays_bitwise_equal": True,
                      "complete_control_reset_state_equal": True,
                      "reset_state_sha256": hashlib.sha256(json.dumps(first_state,
                          sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()})
    return pairs


def _frozen_document(path: Path, sources: dict) -> dict:
    require(str(path) in sources and identity(path) == sources[str(path)],
            "reused evidence is not bound to the C20 source GO: " + str(path))
    return plain_document(path)


def _first_record(folder: Path, receipt: dict, sources: dict) -> dict:
    block = receipt["controller_record_blocks"][0]
    path = folder/block["file"]
    require(str(path) in sources and identity(path) == sources[str(path)]
            and identity(path) == {key:block[key] for key in ("sha256", "bytes")},
            "old first controller block is not frozen")
    _committed(path)
    with gzip.open(path,"rt",encoding="utf-8") as stream:
        row = json.loads(next(stream))
    require(row["tick"] == 0, "old controller first tick is not zero")
    return row


def _old_reset_equivalence(old_folder: Path, new_folder: Path, sources: dict) -> dict:
    initial_old, initial_new = arrays(old_folder/"initial_state.npz"), arrays(new_folder/"initial_state.npz")
    require(_same_arrays(initial_old,initial_new), "C18/C20 initial five arrays differ bitwise")
    old_reset, new_reset = document(old_folder/"reset_receipt.json"), document(new_folder/"reset_receipt.json")
    state = _reset_state(new_reset)
    receipt = document(old_folder/"case_receipt.json")
    first = _first_record(old_folder,receipt,sources)
    adapter = first["info"]["controller_record"]
    calc = adapter["calculation"]
    require(old_reset["seed"] == new_reset["seed"]
            and old_reset["episode_metadata"]["controller_schema"]
                == new_reset["episode_metadata"]["controller_schema"]
                == "d1-course18-yawlimit-filteredcommonintegral-adapter-v1"
            and old_reset["episode_metadata"]["mode"] == new_reset["episode_metadata"]["mode"] == "eval"
            and old_reset["episode_metadata"]["pure_test_components"] is False
            and new_reset["episode_metadata"]["pure_test_components"] is False
            and equal(calc["wheel_integral_before_nm"],state["wheel_integral_nm"])
            and calc["wheel_common_reference_z_before_rad_s"] == state["wheel_common_reference_z_rad_s"]
            and adapter["previous_servo_forward_mps"] == state["previous_servo_forward_mps"]
            and adapter["stop_latched_before"] == state["stop_latched"]
            and adapter["provider_sequence"] == state["provider_sequence"] == 0
            and adapter["control_time_s"] == state["provider_control_time_s"] == 0.
            and first["info"]["servo_receipt"] == state["servo_receipt"]
            and state["servo_last_tick"] == first["info"]["servo_receipt"]["tick"]
            and state["servo_applied"] == first["info"]["servo_receipt"]["applied"]
            and equal(initial_old["observation"][59:63],np.zeros(4))
            and equal(initial_old["observation"][63:79],state["previous_action"])
            and equal(first["input_observation99"],initial_old["observation"]),
            "C18/C20 reset controller/servo/provider evidence differs")
    return {"five_initial_arrays_bitwise_equal":True,
            "proof_kind":"source_derived_reset_equivalence",
            "first_tick_PI_z_stop_servo_provider_equal":True,
            "observation_reset_PI_slots":[59,63], "observation_previous_action_slots":[63,79],
            "old_full_context_was_saved":False,
            "context_equivalence_basis":"same frozen deterministic oracle and actuator reset/prepare source, initial arrays and compiled geometry; oracle reset discards seed; old full context was not archived"}


def _regression_reuse(run, session, sources, identity_doc, parsed, enabled, scored) -> dict:
    """Reuse frozen C18 full readers, with fresh source and initial-state bridges."""
    try:
        if not session["include_floors"]:
            path = Path(session["development_readback_path"])
            previous = _frozen_document(path,sources)
            require(previous["schema"] == READ_SCHEMA and previous["evaluation_split"] == "development"
                    and previous["stage"] == session["stage"] == 1
                    and previous["source_closure_verified"] is True
                    and previous["checkpoint_ZIP_verified"] is True
                    and previous["regression_reuse"]["complete"] is True,
                    "final-without-extension lacks valid frozen development regression evidence")
            old_session = _frozen_document(Path(previous["run"])/"session.json",sources)
            require(old_session["eval_manifests"] == session["eval_manifests"],
                    "final-without-extension changed policies since development")
            reused = [row for row in previous["numeric_scores"]
                      if row["case_id"] in {r["case_id"] for r in SPEC["evaluation"]["regression"]}]
            reused += previous["regression_reuse"]["scores"]
            require(len(reused) == 28, "development reuse lacks all four actors for seven regressions")
            return {"complete":True,"scores":reused,"pairs":previous["regression_reuse"]["pairs"],
                    "proof_kind":"frozen_same_checkpoint_development_regression_reuse",
                    "development_readback_identity":identity(path),
                    "six_task_zero_pairs":previous["six_task_zero_pairs"],"reasons":[]}
        config = session["regression_reuse"]
        require(set(config) == {"primary_run","mirror_run","primary_readback","mirror_readback"},
                "C18 regression reuse locations incomplete")
        bridge_names = [W/"continuation18"/name for name in ("controller18.py","residual18.py")]
        bridge_names += [W/"course_impl08"/name for name in (
            "full_drive_env_08.py","full_drive_controller_08.py","full_drive_loop_08.py",
            "full_drive_servo_08.py","full_drive_observation_08.py","course_plant_08.py")]
        bridge_names += [W/"upright11/world_upright_course_11.py"]
        repository = Path("/home/lyh/wheel-legged-control-lab")
        bridge_names += [repository/"src/wheel_legged_control/d1"/name for name in (
            "control_loop.py","control_context.py","state_provider.py","state_estimation.py",
            "wheel_leg_controller.py","model.py","actuator_channel.py")]
        old = {}
        for name in ("primary","mirror"):
            old_run = Path(config[name+"_run"])
            readback = _frozen_document(Path(config[name+"_readback"]),sources)
            require(readback["schema"] == "d1-world-upright-stage18-independent-eval-readback-v1"
                    and readback["run"] == str(old_run.resolve())
                    and readback["controller_variant"] == "combined"
                    and readback["source_closure_verified"] is True
                    and readback["checkpoint_ZIP_verified"] is True
                    and readback["physical_calls_performed_by_reader"] == 0
                    and readback["model_loaded_by_reader"] is False,
                    "frozen C18 readback identity or full evidence chain differs")
            prior_identity = _frozen_document(old_run/"eval_identity.json",sources)
            _committed(old_run/"eval_identity.json")
            require(prior_identity["actual_geometry_binding"] == identity_doc["actual_geometry_binding"]
                    and prior_identity["compiled_geometry"] == identity_doc["compiled_geometry"]
                    and prior_identity["checkpoint_sha256_by_actor"]["grouped_continue"]
                        == identity_doc["checkpoint_sha256_by_actor"]["old"],
                    "C18/C20 actual geometry or unchanged old checkpoint differs")
            for source in bridge_names:
                require(str(source) in sources and str(source) in prior_identity["source_hashes"]
                        and sources[str(source)] == prior_identity["source_hashes"][str(source)],
                        "C18/C20 deterministic reset source bridge missing or changed: " + str(source))
            for source in (W/"continuation15/heldout15.py",W/"continuation18/eval18.py",W/"continuation18/worker18.py"):
                require(str(source) in prior_identity["source_hashes"]
                        and prior_identity["source_hashes"][str(source)] == sources.get(str(source)),
                        "old recorder/runtime source provenance differs")
            old[name] = (old_run,readback)
        reused, pairs = [], []
        regression_ids = {row["case_id"] for row in SPEC["evaluation"]["regression"]}
        for spec, _mirror, actors in parsed:
            case_id = spec[0]
            if case_id not in regression_ids:
                continue
            row = CASE_SPECS[case_id]
            old_run, readback = old["mirror" if row["mirror"] else "primary"]
            source_case = row["source_case_id"]
            new_actors = [actor for actor in actors if actor in enabled]
            require(new_actors, "regression pair absent after failed floors")
            new_folder = run/"heldout"/f"{case_id}_{new_actors[0]}"
            for actor, old_actor in (("zero","zero"),("old","grouped_continue")):
                folder = old_run/"heldout"/f"{source_case}_{old_actor}"
                require(all(str(path) in sources for path in folder.rglob("*") if path.is_file()),
                        "C18 case archive is not fully source-frozen")
                schedule = document(folder/"schedule.json")
                require(schedule["seed"] == row["seed"] and schedule["control_cap"] == row["horizon"]
                        and schedule["raw_commands"] == score20.expected_raw_commands(case_id),
                        "C18 reuse seed, horizon or full raw command array differs")
                matching = [score for score in readback["numeric_scores"]
                            if score["case_id"] == source_case and score["experiment_actor"] == old_actor]
                require(len(matching) == 1, "C18 reused score missing or duplicated")
                score = matching[0]
                windows = score20.case_windows(case_id)
                require(score["seed"] == row["seed"] and score["mirror"] is row["mirror"]
                        and score["terrain"] == row["terrain"]
                        and score["target_speed_mps"] == row["speed_mps"]
                        and all(score["windows"][key] == value for key,value in windows.items()),
                        "C18 reused scoring window or command identity differs")
                proof = _old_reset_equivalence(folder,new_folder,sources)
                pairs.append(dict(proof,case_id=case_id,old_actor=actor,
                                  new_actors=new_actors,source_case_id=source_case))
                reused.append(dict(score,case_id=case_id,experiment_actor=actor,
                                   reuse_source_run=str(old_run),reuse_source_case_id=source_case))
        require(len(reused) == 14 and len(pairs) == 14,
                "original seven regressions lack complete zero/old source pairs")
        return {"complete":True,"scores":reused,"pairs":pairs,"reasons":[],
                "proof_kind":"source_derived_reset_equivalence",
                "old_full_context_was_saved":False,
                "reset_source_bridge_sha256":{str(p):sources[str(p)]["sha256"] for p in bridge_names}}
    except (ValueError,KeyError,FileNotFoundError,TypeError,StopIteration) as error:
        return {"complete":False,"scores":[],"pairs":[],
                "reasons":[str(error)],"status":"inconclusive_pairing_evidence",
                "additional_physics_authorized":False}


def read_run(run: Path) -> dict[str, Any]:
    """Independently verify one frozen C20 saved worker; no model or simulation."""
    require(isinstance(run, Path) and run.is_dir() and not run.is_symlink(),
            "C20 run absent or symlinked")
    session = plain_document(run / "session.json")
    worker = plain_document(run / "worker_receipt.json")
    host = plain_document(run / "host_receipt.json")
    arm, variant = session.get("arm"), session.get("controller_variant")
    specs, split, stage = session.get("case_specs"), session.get("evaluation_split"), session.get("stage")
    include_floors = session.get("include_floors")
    require(session.get("schema") == "d1-coverage-session-20-v1"
            and session.get("mode") == "eval" and variant == "combined"
            and (split, stage) in (("development", 1), ("final", 1), ("final", 2))
            and type(include_floors) is bool
            and include_floors is (split == "development" or stage == 2)
            and type(specs) is list and specs and session.get("retry_permitted") is False
            and worker.get("schema") == "d1-coverage-worker-20-v1"
            and worker.get("arm") == arm
            and worker.get("session_identity") == identity(run / "session.json")
            and worker.get("execution_complete") is True
            and worker.get("failure") is None and worker.get("cleanup_errors") == []
            and worker.get("warnings") == [] and worker.get("archive_failed") is False
            and worker.get("control_step_caller_verified") is True
            and host.get("arm") == arm and host.get("exit_code") == 0
            and host.get("failure") is None and host.get("cleanup_errors") == []
            and host.get("changed_sources") == [] and host.get("postcheck_complete") is True
            and host.get("no_live_owned_processes") is True
            and host.get("fully_reserved_budget_closed") is True
            and host.get("retry_permitted") is False
            and host.get("elapsed_s", float("inf")) <= session["hard_s"],
            "C20 worker/host/session closure failed")
    parsed = [_case_spec(row) for row in specs]
    expected_ids = {row["case_id"] for row in SPEC["evaluation"][
        "development" if split == "development" else "final_sealed"]}
    if include_floors:
        expected_ids |= {row["case_id"] for row in SPEC["evaluation"]["regression"]}
    require(len({row[0][0] for row in parsed}) == len(parsed)
            and {row[0][0] for row in parsed} == expected_ids,
            "C20 split omitted, duplicated, or added a preregistered case")
    require(session["floor_seed"] == SPEC["evaluation"]["floor"][f"seed_stage{stage}"],
            "C20 floor seed differs from preregistration")
    reserved = (1200 if include_floors else 0) + sum(spec[0][4]*len(spec[2]) for spec in parsed)
    require(session["control_limit"] == reserved
            and session["normal_native_limit"] == 5*reserved
            and session["compiler_native_limit"] == 2,
            "C20 control/native/compiler reservation differs")
    plan_path = Path(session["plan_path"])
    plan = plain_document(plan_path)
    key = session["reservation_key"]
    require(identity(plan_path) == session["plan_identity"]
            and plan.get("schema") == "d1-coverage-source-go-20-v1"
            and plan.get("status") == "GO" and plan.get("retry_permitted") is False
            and key in plan["arms"]
            and all(plan["arms"][key][field] == session[field] for field in
                ("case_specs", "include_floors", "stage", "evaluation_split", "arm", "hard_s"))
            and plan["arms"][key]["controls"] == reserved,
            "C20 source GO/session reservation differs")
    identity_doc = document(run / "eval_identity.json")
    sources = identity_doc["source_hashes"]
    require(identity_doc["execution_stage"] == 20
            and identity_doc["controller_variant"] == variant
            and identity_doc["evaluation_split"] == split
            and sources == session["source_hashes"]
            and sources == {**plan["inputs"], str(plan_path): identity(plan_path)},
            "C20 evaluation source/controller/split identity differs")
    for name, expected in sources.items():
        path = Path(name)
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and identity(path) == {k: expected[k] for k in ("sha256", "bytes")},
                "frozen source differs: " + name)
    for directory, files in ((HERE, ("read_eval20.py", "score20.py", "recipes20.py", "heldout20.py",
             "state20.py", "spec20.json", "floor_bridge20.py", "offline_floor20.py", "host20.py",
             "worker20.py", "eval20.py", "runtime_support20.py")),
             (W / "continuation18", ("controller18.py", "residual18.py", "verify_control18.py", "score18.py"))):
        for name in files:
            require(str(directory/name) in sources, "critical source absent: " + name)
    expected_sha, paths = (identity_doc[key] for key in ("checkpoint_sha256_by_actor", "checkpoint_paths_by_actor"))
    require(set(expected_sha) == set(paths) == set(session["eval_manifests"]) == {"old", "A", "B"},
            "C20 checkpoint table must contain old/A/B exactly")
    for actor in ("old", "A", "B"):
        path = Path(paths[actor])
        manifest = session["eval_manifests"][actor]
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and path == Path(manifest["folder"])/"final_model.zip"
                and _file_sha256(path) == expected_sha[actor] == manifest["files"]["final_model.zip"]["sha256"]
                and str(path) in sources and sources[str(path)]["sha256"] == expected_sha[actor],
                "C20 checkpoint identity differs: " + actor)
    require(expected_sha["old"] == SPEC["parent"]["model_sha256"],
            "old actor is not the preregistered C15 parent")
    require(identity_doc["warnings"] == worker["warnings"] == [], "C20 recorded a warning")
    binding, geometry = check_binding(identity_doc["actual_geometry_binding"]), identity_doc["compiled_geometry"]
    floors, enabled = {}, []
    if include_floors:
        for actor in ("A", "B"):
            folder = run/"floor"/actor
            gate = read_floor(folder, expected_seed=session["floor_seed"])
            require(gate["experiment_actor"] == actor and gate["checkpoint_sha256"] == expected_sha[actor]
                    and document(folder/"geometry_manifest.json") == geometry
                    and document(run/(actor+"_floor_gate.json")) == gate,
                    "C20 independent floor gate/checkpoint/geometry differs")
            floors[actor] = gate
            if gate["numeric_gate_passed"]:
                enabled.append(actor)
    else:
        enabled = ["A", "B"]
    require(worker["result"]["floors"] == floors, "C20 worker floor results differ")
    members = [(spec, mirror, actor) for spec, mirror, actors in parsed for actor in actors
               if actor not in ("A", "B") or actor in enabled]
    summaries, canonical, saved = [], [], []
    for spec, mirror, actor in members:
        summary, numeric = heldout_case(run, spec, actor, binding, geometry,
                                        expected_variant=variant, mirror=mirror)
        require(numeric["checkpoint_sha256"] == expected_sha.get(actor),
                "C20 case actor checkpoint differs")
        summaries.append(summary)
        canonical.append(numeric)
        receipt = document(run/"heldout"/f"{spec[0]}_{actor}"/"case_receipt.json")
        saved.append({k: receipt[k] for k in ("case_id", "actor", "seed", "completed_controls",
            "record_valid", "terminated", "truncated", "stop_reason", "policy_predictions", "checkpoint_sha256")})
    require({p.name for p in (run/"heldout").iterdir() if p.is_dir()}
            == {f"{spec[0]}_{actor}" for spec, _mirror, actor in members},
            "C20 heldout archive contains omitted or unplanned actor folders")
    pairs = _new_pairs(run, parsed, enabled)
    scored = [dict(score20.score_case(row, expected_seed=row["seed"], mirror=row["mirror"]),
                   experiment_actor=row["experiment_actor"], checkpoint_sha256=row["checkpoint_sha256"])
              for row in canonical]
    observed = sum(gate["completed_controls"] for gate in floors.values()) + sum(row["completed_controls"] for row in summaries)
    c, py, ng = (worker[k] for k in ("C_final", "python", "native_guard"))
    require(observed <= reserved and worker["result"]["completed_controls"] == observed
            and worker["result"]["enabled_arms"] == enabled and worker["result"]["cases"] == saved
            and worker["result"]["full_comparison_executed"] is (len(summaries) == sum(len(x[2]) for x in parsed))
            and c["construction_attempts"] == c["construction_returns"] == 2
            and c["control_attempts"] == c["control_returns"] == 5*observed
            and c["phase"] == c["target_model"] == c["target_data"] == c["violations"] == 0
            and py["control_attempted"] == py["control_completed"] == observed
            and py["native_attempted"] == py["native_returned"] == py["clock_advanced_substeps"] == 5*observed
            and py["forbidden_entries"] == 0
            and ng["native_attempted"] == ng["native_returned"] == ng["native_checked"] == 5*observed
            and ng["failure"] is None,
            "C20 actual C/Python/native/compiler closure differs")
    predicted = sum(gate["completed_controls"] for gate in floors.values()) + sum(row["predict_calls"] for row in summaries)
    calls = worker["model_calls"]["counts"]
    expected_calls = {"load":3, "torch_load":9, "save":0, "learn":0, "train":0, "forward":0,
                      "evaluate_actions":0, "predict_values":0, "backward":0, "predict":3+predicted}
    require(all(calls.get(k, {}).get("attempted",0) == calls.get(k, {}).get("returned",0) == value
                for k, value in expected_calls.items()), "C20 evaluation model call counts differ")
    require(worker["model_calls"]["actor_rows"] == predicted+96
            and worker["model_calls"]["critic_rows"] == 0,
            "C20 actor probe rows or critic rows differ")
    reuse = _regression_reuse(run, session, sources, identity_doc, parsed, enabled, scored)
    all_scores = scored + reuse["scores"]
    contribution = reuse.get("six_task_zero_pairs", {})
    if reuse["complete"] and include_floors:
        for actor in ("A", "B"):
            if actor in enabled:
                contribution[actor] = score20.score_pairs(all_scores, policy_actor=actor)
    return {"schema":READ_SCHEMA, "run":str(run.resolve()), "evaluation_split":split, "stage":stage,
            "controller_variant":variant, "source_closure_verified":True,
            "checkpoint_ZIP_verified":True, "floors":floors, "enabled_arms":enabled,
            "heldout_summaries":summaries, "numeric_scores":scored,
            "new_actor_initial_pairs":pairs, "regression_reuse":reuse,
            "equal_case_pool":score20.equal_case_pool(scored,split=split),
            "six_task_zero_pairs":contribution,
            "original_six_pool":score20.original_six_pool(all_scores),
            "actual_completed_controls":observed, "actual_normal_native_returns":5*observed,
            "actual_compiler_native_returns":2, "physical_calls_performed_by_reader":0,
            "model_loaded_by_reader":False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = read_run(args.run.resolve())
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
