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
                  W / "continuation15", W / "continuation18"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from archive13.atomic_archive_13 import verify_manifest
from geometry24 import (PAIR_FIELDS, RECORD_SCHEMA, CONTRACT_ID, CASE_ID, SEED,
                        CONTROLS, make_schedule, FAR, FRONT, SPAWN)
from verify24 import check_binding, check_contact, geom_map
from kinematics24 import reconstruct24, collision_bounds24, robot_binding_bridge24
from score24 import score_case24
from rl11.verify_short_rl16_training_11_04 import boundary_delta
from verify_control18 import check_controller
from verify_course_e_08_03 import (
    _body_com_velocity, _native_rows, _roll_pitch_deg, _rotation,
    close, equal, identity, require,
)
from verify_rl16_training_08 import _ground_hit, _terrain_reference, _yaw_from_qpos

READ_SCHEMA = "d1-true15mm-independent-readback-24-v1"


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
    schedule = make_schedule()
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
            "heldout actual compiled two-primitive manifest changed between cases")
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
        "native_heading_rad": [], "native_lateral_offset_m": [],
        "native_positive_load_by_wheel": [], "box_feature_positive_counts": {},
        "positive_wheel_load_native_counts_by_family": {},
        "ramp_positive_wheel_load_native_counts_by_geom": {},
        "warning_count": 0, "geometry_invalid_count": 0,
        "map_escape_count": 0, "fall_count": 0,
    }
    features = Counter()
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
            wheel_loads = set()
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
                    wheel_loads.add(contact["robot_wheel_index"])
                    if family == "step":
                        features[contact["geometric_feature"]] += 1
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
            canonical["native_positive_load_by_wheel"].append([int(i in wheel_loads) for i in range(4)])
            canonical["native_heading_rad"].append(float(_yaw_from_qpos(native_qpos)-initial_yaw))
            canonical["native_lateral_offset_m"].append(float(lateral_axis@(native_qpos[:2]-initial_xy)))
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
                abs(float(native_qpos[0])) > 6.0 or abs(float(native_qpos[1])) > 3.0
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
    canonical["box_feature_positive_counts"] = dict(features)
    endpoint_path = folder / 'endpoint_geometry24.jsonl.gz'
    _committed(endpoint_path)
    with gzip.open(endpoint_path,'rt') as stream:
        endpoints = [json.loads(line) for line in stream]
    require(len(endpoints)==n+1, 'C24 independent endpoint geometry count differs')
    kbinding = binding['kinematics24']
    whole_vz, final_bounds = [], None
    query_count = 0
    for tick,endpoint in enumerate(endpoints):
        require(endpoint['tick']==tick and close(endpoint['time_s'],tick*.01)
                and endpoint['cache_source']=='plant.measurement_data_post_refresh_measurements'
                and equal(endpoint['qpos'],states['qpos'][tick])
                and equal(endpoint['qvel'],states['qvel'][tick]),
                'C24 endpoint geometry not bound to actual integrated state')
        recomputed = reconstruct24(kbinding,states['qpos'][tick],states['qvel'][tick])
        bounds = collision_bounds24(kbinding,recomputed,binding['wheel_map'])
        require(close(endpoint['whole_com_position_m'],recomputed['whole_com_position'],atol=2e-10)
                and close(endpoint['whole_com_velocity_mps'],recomputed['whole_com_velocity'],atol=2e-9)
                and len(bounds)==len(endpoint['collision_bounds']),
                'C24 whole-system COM/complete collision set differs')
        for saved_bound,recomputed_bound in zip(endpoint['collision_bounds'],bounds):
            for key in ('geom_id','body_id','geom_type','wheel_index','margin_m'):
                require(saved_bound[key]==recomputed_bound[key], 'C24 collision binding differs')
            for key in ('minimum_world_m','maximum_world_m'):
                require(close(saved_bound[key],recomputed_bound[key],atol=2e-10),
                        'C24 primitive support differs from actual qpos/compiled geometry')
        if tick==0:
            require(all(b['maximum_world_m'][0]+b['margin_m'] < FRONT for b in bounds),
                    'C24 initial robot collision geometry already touches/straddles box')
        require(endpoint['readonly_mj_objectVelocity_calls']==len(kbinding['body_mass'])-1,
                'C24 read-only velocity query count differs')
        query_count += endpoint['readonly_mj_objectVelocity_calls']
        whole_vz.append(float(recomputed['whole_com_velocity'][2]))
        final_bounds = bounds
    require(receipt['readonly_mj_objectVelocity_calls']==query_count,
            'C24 case read-only query total differs')
    from geometry24 import reset_geometry_gate24
    gate = document(folder/'reset_geometry_gate24.json')
    samples=gate['admission']['ground_samples']
    for sample in samples:
        h,_normal,gid=_ground_hit(geometry,sample['x_m'],sample['y_m'])
        require(close(h,sample['height_m']) and geometry[gid]['name']==sample['geom_name'],
                'C24 saved reset oracle proof differs from compiled actual plane/box')
    require(gate['before']==gate['after']==reset['after']
            and gate['admission']==reset_geometry_gate24(endpoints[0],expected_geometry,samples),
            'C24 reset admission proof or zero-count boundary differs')
    canonical['whole_system_com_vz_mps'] = whole_vz[1:]
    canonical['base_world_z_m'] = states['qpos'][1:,2].tolist()
    canonical['final_collision_bounds'] = final_bounds
    canonical['readonly_mj_objectVelocity_calls'] = query_count
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
        "readonly_mj_objectVelocity_calls": query_count,
    }
    return summary, canonical

def read_run(run: Path) -> dict:
    require(run.is_absolute() and run.is_dir() and not run.is_symlink(), 'C24 run absent')
    session=plain_document(run/'session.json')
    worker=plain_document(run/'worker_receipt.json')
    host=plain_document(run/'host_receipt.json')
    supervisor=plain_document(run/'supervisor_receipt.json')
    require(session.get('execution_contract_id')==CONTRACT_ID
            and session.get('mode')=='eval' and session.get('arm')=='step_pair'
            and session.get('controller_variant')=='combined'
            and session.get('controls')==session.get('control_limit')==2400
            and session.get('retry_permitted') is False
            and worker.get('schema')=='d1-true15mm-worker-24-v1'
            and worker.get('execution_contract_id')==CONTRACT_ID
            and worker.get('arm')=='step_pair'
            and worker.get('session_identity')==identity(run/'session.json')
            and worker.get('execution_complete') is True and worker.get('failure') is None
            and worker.get('cleanup_errors')==[] and worker.get('warnings')==[]
            and worker.get('archive_failed') is False
            and worker.get('control_step_caller_verified') is True
            and host.get('arm')=='step_pair' and host.get('exit_code')==0
            and host.get('failure') is None and host.get('source_mismatches')==[]
            and host.get('owned_no_orphans') is True and host.get('reservation_closed') is True
            and host.get('reserved_controls')==2400 and host.get('retry_permitted') is False
            and supervisor.get('failure') is None and supervisor.get('exit_code')==0
            and supervisor.get('cleanup',{}).get('remaining')=={}
            and supervisor.get('outer_limit_s')==900
            and supervisor.get('elapsed_s',float('inf'))<=900,
            'C24 actual host/worker/supervisor closure differs')
    go_path=Path(session['source_go_path'])
    go=plain_document(go_path)
    require(identity(go_path)==session['source_go_identity'] and go['decision']=='GO'
            and go['arms']['step_pair']['control_limit']==2400
            and all(session.get(k)==v for k,v in go['arms']['step_pair'].items()),
            'C24 source GO/spec does not bind actual session')
    sources=session['source_hashes']
    require(all(sources.get(k)==v for k,v in go['inputs'].items()),'C24 GO closure missing')
    for name,expected in sources.items():
        require(identity(Path(name))==expected,'C24 frozen source changed: '+name)
    for name in ('worker24.py','eval24.py','heldout24.py','env24.py','plant24.py',
                 'native_guard24.py','geometry24.py','kinematics24.py','verify24.py',
                 'read24.py','score24.py'):
        require(str(HERE/name) in sources,'C24 critical module missing from source closure: '+name)
    construction=document(run/'construction_receipt.json')
    ident=document(run/'eval_identity.json')
    require(ident['source_hashes']==sources and ident['execution_contract_id']==CONTRACT_ID
            and ident['controller_variant']=='combined' and ident['warnings']==[]
            and construction['geometry_binding']==ident['actual_geometry_binding']
            and construction['compiled_geometry']==ident['compiled_geometry'],
            'C24 actual geometry/evaluation identity differs')
    proof=construction['proof']
    require(proof['passed'] is True and proof['dso_path']==session['library']
            and proof['dso_sha256']==sources[session['library']]['sha256']
            and len(proof['jump_slots'])==4 and all(s['passed'] is True for s in proof['jump_slots'])
            and construction['nominal_cache']['misses']==1
            and construction['C_state']['construction_attempts']==construction['C_state']['construction_returns']==2,
            'C24 actual engine binding/cold compiler proof differs')
    refpath=Path(session['reference_construction_path'])
    require(str(refpath) in sources,'C24 frozen reference compiled geometry absent')
    reference=plain_document(refpath)['geometry_binding']
    require(document(run/'robot_binding_bridge24.json')==robot_binding_bridge24(
        construction['geometry_binding'],reference),'C24 old/new robot source bridge differs')
    checkpoint=Path(ident['checkpoint_path'])
    manifest=session['checkpoint_manifest']
    relocation=session['checkpoint_relocation']
    original_path=Path(relocation['original_manifest_path'])
    original=plain_document(original_path)
    require(identity(original_path)==relocation['original_manifest_identity']
            ==sources.get(str(original_path))
            and relocation['changed_fields']==['folder']
            and original['folder']==relocation['original_folder']
            and relocation['current_folder']==session['checkpoint_folder']
            and manifest=={**original,'folder':session['checkpoint_folder']},
            'C24 checkpoint relocation differs beyond exact manifest folder field')
    require(checkpoint==Path(session['checkpoint_folder'])/'final_model.zip'
            and manifest['folder']==session['checkpoint_folder']
            and identity(checkpoint)==manifest['files']['final_model.zip']
            and identity(checkpoint)['sha256']==ident['checkpoint_sha256']==session['checkpoint_sha256'],
            'C24 B22 exact ZIP identity differs')
    load=document(run/'B_load.json')
    require(load['probe_rows']==32 and load['probe_actions_byte_exact'] is True
            and load['matches_training_final_policy_state'] is True
            and load['verified_without_engine_or_reset'] is True
            and load['observation_space']['shape']==[99] and load['action_space']['shape']==[16],
            'C24 strict final-checkpoint/probe record invalid')
    binding=check_binding(ident['actual_geometry_binding'])
    geometry=ident['compiled_geometry']
    spec=(CASE_ID,'single_step',.2,SEED,CONTROLS,275,1000)
    summaries,canonicals,saved=[],[],[]
    for actor in ('zero','B'):
        summary,canonical=heldout_case(run,spec,actor,binding,geometry,expected_variant='combined')
        require(canonical['checkpoint_sha256']==(None if actor=='zero' else session['checkpoint_sha256']),
                'C24 actor checkpoint differs')
        summaries.append(summary)
        canonicals.append(canonical)
        receipt=document(run/'heldout'/(CASE_ID+'_'+actor)/'case_receipt.json')
        saved.append({k:receipt[k] for k in ('case_id','actor','seed','completed_controls',
            'record_valid','terminated','truncated','stop_reason','policy_predictions',
            'checkpoint_sha256','readonly_mj_objectVelocity_calls')})
    folders=[run/'heldout'/(CASE_ID+'_'+actor) for actor in ('zero','B')]
    initial=[arrays(f/'initial_state.npz') for f in folders]
    reset=[document(f/'reset_receipt.json') for f in folders]
    require(reset[0]['exact_initial_pair'] is False and reset[1]['exact_initial_pair'] is True
            and reset[1]['paired_with']=='zero'
            and all(reset[0][k]==reset[1][k] for k in ('model_address','data_address','control_reset_state'))
            and all(initial[0][k].dtype==initial[1][k].dtype
                    and initial[0][k].shape==initial[1][k].shape
                    and initial[0][k].tobytes()==initial[1][k].tobytes() for k in PAIR_FIELDS),
            'C24 zero/B five-array and complete control reset pairing differs')
    require(np.array_equal(initial[0]['observation'][59:79],np.zeros(20,dtype=np.float32)),
            'C24 actual initial PI/previous action observation not reset')
    controls=sum(s['completed_controls'] for s in summaries)
    query_count=sum(s['readonly_mj_objectVelocity_calls'] for s in summaries)
    c,p,g=worker['C_final'],worker['python'],worker['native_guard']
    require(controls<=2400 and worker['result']['completed_controls']==controls
            and worker['result']['cases']==saved and worker['result']['full_comparison_executed'] is True
            and worker['result']['readonly_mj_objectVelocity_calls']==query_count
            and c['construction_attempts']==c['construction_returns']==2
            and c['control_attempts']==c['control_returns']==5*controls
            and c['phase']==c['target_model']==c['target_data']==0 and c['violations']==0
            and c['native_construction_caller_verified'] is True
            and c['ccd_attempts']==c['ccd_returns']
            and (c['ccd_attempts']==0 or c['native_ccd_caller_verified'] is True)
            and p['control_attempted']==p['control_completed']==controls
            and p['native_attempted']==p['native_returned']==p['clock_advanced_substeps']==5*controls
            and p['forbidden_entries']==0 and p['fatal_latched'] is False
            and g['native_attempted']==g['native_returned']==g['native_checked']==5*controls
            and g['failure'] is None,'C24 C/Python/native/compiler ledger differs')
    calls=worker['model_calls']['counts']
    predicted=summaries[1]['predict_calls']
    expected={'load':1,'torch_load':3,'save':0,'learn':0,'train':0,'forward':0,
              'evaluate_actions':0,'predict_values':0,'backward':0,'predict':1+predicted}
    require(all(calls.get(k,{}).get('attempted',0)==calls.get(k,{}).get('returned',0)==v
                for k,v in expected.items()),'C24 actual model API counts differ')
    require(calls['predict']['rows_attempted']==calls['predict']['rows_returned']==32+predicted,
            'C24 strict probe/actor row count differs')
    scored=[score_case24(canonical) for canonical in canonicals]
    return dict(schema=READ_SCHEMA,run=str(run),execution_contract_id=CONTRACT_ID,
        source_closure_verified=True,checkpoint_ZIP_verified=True,
        exact_five_array_and_control_reset_pair=True,heldout_summaries=summaries,
        numeric_scores=scored,B22_true_15mm_qualified=scored[1]['task_qualified'],
        zero_true_15mm_qualified=scored[0]['task_qualified'],
        actual_completed_controls=controls,actual_normal_native_returns=5*controls,
        actual_compiler_native_returns=2,actual_model_calls=worker['model_calls'],
        readonly_mj_objectVelocity_calls=query_count,
        physical_calls_performed_by_reader=0,model_loaded_by_reader=False,
        claim_scope='one preregistered matched real-box scenario; no training or statistical generalization claim')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=read_run(args.run.resolve(strict=True))
    with args.output.open('x') as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


if __name__=='__main__':
    main()
