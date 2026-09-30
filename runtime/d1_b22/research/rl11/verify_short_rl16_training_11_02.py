"""Independent saved-data readback of the one world-upright 11-S execution.

Only stdlib, NumPy and frozen pure validators are imported. Training's compact
native blocks establish accounting and finite numeric provenance; only the
heldout full native/contact records support physical qualification.
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

import numpy as np

HERE = Path(__file__).resolve().parent
W = HERE.parent
sys.path.insert(0, str(W))
sys.path.insert(0, str(W / "course_impl08"))
sys.path.insert(0, str(HERE))
from rl16_heldout_score_11 import score_case, score_pairs
from short_plan_recipe_11 import select_episode

from upright11.verify_world_upright_reference_11 import _projection
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

SCHEMA = "d1-world-upright-short-rl16-training-heldout-11-v1"
CONTRACT_SHA = "e6e9d9d61d43d5eb3184c6c0cb31493f9f95e76cf4ab5632de02ce28135cd929"
SPAWNS = {"flat": [-8.0, -4.7, .455], "bumps": [.35, -2.2, .455],
          "rough": [.35, 0.0, .455], "ramp": [2.75, 0.0, .455]}
CASES = (
    ("flat_0p6", "flat", .6, 88701, 1600, 295, 695),
    ("flat_1p6", "flat", 1.6, 88702, 1600, 495, 895),
    ("flat_1p2_yaw", "flat", 1.2, 88703, 1600, 415, 815),
    ("bumps_0p4", "bumps", .4, 88704, 1600, 255, 655),
    ("rough_0p35", "rough", .35, 88705, 1600, 245, 645),
    ("ramp_0p45_complete", "ramp", .45, 88706, 1800, 600, 1355),
)
PAIR_FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation")
TRAIN_FIELDS = (
    "control_index", "episode_index", "episode_tick", "input_observation99",
    "policy_env_input16", "policy_clipped16", "effective_action16",
    "raw_command_vx_vy_yaw_clearance", "servo_command_vx_vy_yaw_clearance",
    "reward_terms9", "reward", "terminated", "truncated",
    "body_com_vx_mps", "body_com_vy_mps", "body_yaw_rate_rps",
    "base_x_m", "base_y_m", "clearance_m",
    "geometry_relative_roll_rad", "geometry_relative_pitch_rad",
    "task_roll_error_rad", "task_pitch_error_rad", "native_nonwheel_contacts",
    "actuator_delayed5x16", "actuator_applied5x16", "stage_safe_torque_nm",
)


def document(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), f"record missing: {path}")
    row = read(path)
    require(isinstance(row, dict), f"record is not an object: {path}")
    return row


def arrays(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file() and not path.is_symlink(), f"array record missing: {path}")
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def match_file(path: Path, record: dict) -> None:
    require(identity(path) == {key: record[key] for key in ("sha256", "bytes")},
            f"file identity differs: {path}")


def source(run: Path) -> tuple[dict, dict, dict]:
    session = document(run / "session.json")
    launcher = document(run / "launcher_receipt.json")
    worker = document(run / "worker_receipt.json")
    require(session["schema"] == launcher["schema"] == worker["schema"] == SCHEMA
            and session["phase"] == "T11" and session["output_directory"] == str(run)
            and session["contract_sha256"] == CONTRACT_SHA
            and session["control_limit"] == 85136
            and session["normal_native_limit"] == 425680
            and session["compiler_native_limit"] == 2
            and session["wallclock_limit_s"] == 1800
            and session["training_control_limit"] == 65536
            and session["heldout_control_limit"] == 19600
            and session["segments"] == [65536] + [1600] * 10 + [1800] * 2
            and session["ppo_seed"] == 88621
            and session["command_seed"] == 88622
            and session["measurement_seed"] == 88623
            and session["retry_permitted"] is False,
            "11-S frozen session identity/budget differs")
    frozen = session["source_hashes"]
    require(isinstance(frozen, dict) and len(frozen) >= 100,
            "11-S full source closure missing")
    for name, expected in frozen.items():
        require(Path(name).is_absolute(), "nonabsolute frozen input")
        match_file(Path(name), expected)
    for name in ("run_short_rl16_training_11.py", "short_curriculum_11.py",
                 "short_heldout_11.py", "rl16_heldout_score_11.py",
                 "next_short_training_contract_11.md"):
        require(str(HERE / name) in frozen, f"critical 11-S source not frozen: {name}")
    require(identity(Path(session["contract_path"]))["sha256"] == CONTRACT_SHA,
            "11-S signed contract changed")
    go_path = HERE / "astra_short_rl_go_11.json"
    go = document(go_path)
    require(identity(go_path)["sha256"] == session["go_sha256"]
            and go["decision"] == "GO" and go["contract_sha256"] == CONTRACT_SHA,
            "11-S actual GO is not the frozen decision")
    for name, expected in go["inputs"].items():
        require(frozen.get(name) == expected, f"GO input absent from frozen session: {name}")
        match_file(Path(name), expected)
    require(launcher["exit_code"] == 0 and launcher["failure"] is None
            and launcher["source_hash_mismatches"] == []
            and launcher["worker_exited"] is True
            and launcher["owned_worker_cleanup"]["no_orphans"] is True
            and launcher["fully_reserved_budget_closed"] is True
            and launcher["retry_permitted"] is False,
            "11-S external host closure or budget failed")
    return session, worker, {"frozen_inputs": len(frozen),
                             "go_inputs": len(go["inputs"]),
                             "go_sha256": session["go_sha256"]}


def boundary_delta(before: dict, after: dict, controls: int) -> None:
    cb, ca = before["C_state"], after["C_state"]
    pb, pa = before["python"], after["python"]
    require(ca["construction_returns"] == cb["construction_returns"]
            and ca["construction_attempts"] == cb["construction_attempts"]
            and ca["control_attempts"] - cb["control_attempts"] == 5 * controls
            and ca["control_returns"] - cb["control_returns"] == 5 * controls
            and pa["control_attempted"] - pb["control_attempted"] == controls
            and pa["control_completed"] - pb["control_completed"] == controls
            and pa["native_attempted"] - pb["native_attempted"] == 5 * controls
            and pa["native_returned"] - pb["native_returned"] == 5 * controls
            and pa["clock_advanced_substeps"] - pb["clock_advanced_substeps"]
            == 5 * controls,
            "11-S C/Python segment did not advance exact 5T")


def training(run: Path) -> dict:
    folder = run / "training"
    manifest = document(folder / "training_blocks_manifest.json")
    segment = document(folder / "training_segment_receipt.json")
    before = document(folder / "boundary_before.json")
    n = 65536
    require(manifest["schema"] == "d1-course-world-upright-short1024-numeric-block-v1"
            and manifest["complete"] is True
            and manifest["completed_controls"] == manifest["gaussian_records"] == n
            and manifest["pending_gaussian_control_index"] is None
            and segment["failure"] is None
            and segment["completed_controls"] == n
            and segment["curriculum_completed_controls"] == n
            and segment["gaussian_records"] == n,
            "11-S training compact record is not complete")
    boundary_delta(before, segment["boundary_after"], n)
    guard = segment["native_segment"]
    require(guard["mode"] == "train"
            and guard["native_attempted"] == guard["native_returned"] == 5 * n
            and guard["failure"] is None and guard["partial_native_interval"] is False
            and guard["full_contact_qualification_recorded"] is False,
            "11-S compact training native scope/count differs")
    number, gaussian = manifest["numeric_blocks"], manifest["gaussian_blocks"]
    require(len(number) == len(gaussian) == 64, "training 1024 block count differs")
    seen = nonzero = clipped = nonwheel = 0
    episodes: dict[int, dict] = {}
    current_episode = -1
    for index, (left, right) in enumerate(zip(number, gaussian)):
        require(left["file"] == f"controls_{index:04d}.npz"
                and right["file"] == f"gaussian_{index:04d}.npz"
                and left["rows"] == right["rows"] == 1024
                and left["first_control_index"] == right["first_control_index"] == seen
                and left["last_control_index"] == right["last_control_index"] == seen + 1023,
                "training block index/size differs")
        match_file(folder / "training_numeric_blocks" / left["file"], left)
        match_file(folder / "training_gaussian_blocks" / right["file"], right)
        a = arrays(folder / "training_numeric_blocks" / left["file"])
        g = arrays(folder / "training_gaussian_blocks" / right["file"])
        require(set(TRAIN_FIELDS) <= set(a)
                and all(v.shape[0] == 1024 and np.isfinite(v).all() for v in a.values())
                and all(v.shape[0] == 1024 and np.isfinite(v).all() for v in g.values())
                and np.array_equal(a["control_index"], np.arange(seen, seen + 1024))
                and np.array_equal(g["control_index"], a["control_index"])
                and np.array_equal(g["episode_index"], a["episode_index"])
                and np.array_equal(g["episode_tick"], a["episode_tick"])
                and np.array_equal(g["clipped_action16"], a["policy_env_input16"])
                and np.array_equal(a["policy_clipped16"], np.clip(a["policy_env_input16"], -1, 1))
                and np.array_equal(g["clipped_action16"],
                                   np.clip(g["raw_gaussian_action16"], -1, 1))
                and np.array_equal(g["effective_action16"], a["effective_action16"])
                and a["input_observation99"].shape == (1024, 99)
                and a["effective_action16"].shape == (1024, 16)
                and a["actuator_applied5x16"].shape == (1024, 5, 16)
                and np.array_equal(a["actuator_applied5x16"], a["actuator_delayed5x16"])
                and np.array_equal(a["actuator_applied5x16"], np.broadcast_to(
                    a["stage_safe_torque_nm"][:, None, :], (1024, 5, 16)))
                and np.allclose(a["reward_terms9"].sum(axis=1), a["reward"],
                                rtol=0, atol=1e-9),
                "11-S actual Gaussian/reward/torque numeric chain differs")
        nonzero += int(np.count_nonzero(np.any(a["effective_action16"] != 0, axis=1)))
        clipped += int(np.count_nonzero(np.any(
            g["raw_gaussian_action16"] != g["clipped_action16"], axis=1)))
        nonwheel += int(np.sum(a["native_nonwheel_contacts"]))
        for offset in range(1024):
            episode_id, tick = int(a["episode_index"][offset]), int(a["episode_tick"][offset])
            if episode_id != current_episode:
                require(episode_id == current_episode + 1 and tick == 0,
                        "11-S training episode was selected out of order")
                if current_episode >= 0:
                    previous = episodes[current_episode]
                    require(previous["terminated"] or previous["truncated"],
                            "11-S next training episode began without a real terminal")
                current_episode = episode_id
                expected = select_episode(episode_id)
                saved = document(folder / f"training_episode_{episode_id:06d}_schedule.json")
                reset = document(folder / f"training_episode_{episode_id:06d}_reset.json")
                outer_seed = int(np.random.SeedSequence([88623, episode_id]).generate_state(
                    1, dtype=np.uint32)[0])
                provider_seed = int(np.random.default_rng(outer_seed).integers(0, 2**31))
                require(saved["episode_index"] == episode_id
                        and saved["plan_schema"] == "d1-world-upright-short1000-preregistered-plan-v1"
                        and saved["selection_seed"] == 88622
                        and saved["choice"] == json.loads(expected.choice_json)
                        and saved["raw_command_sha256"] == expected.command_sha256
                        and saved["raw_commands"] == [
                            asdict(command) for command in expected.raw_commands]
                        and hashlib.sha256(json.dumps(saved["raw_commands"],
                            sort_keys=True, separators=(",", ":"),
                            allow_nan=False).encode()).hexdigest()
                        == expected.command_sha256
                        and saved["episode_reset_seed"] == reset["episode_reset_seed"]
                        == outer_seed
                        and reset["episode_index"] == episode_id
                        and type(reset["provider_measurement_seed"]) is int
                        and reset["actual_episode_metadata"]["reference_schema"]
                        == "d1-course-world-upright-roll-pitch-zero-v1"
                        and reset["actual_episode_metadata"]["measurement_seed"]
                        == reset["provider_measurement_seed"] == provider_seed
                        and saved["start_completed_global_controls"] == seen + offset,
                        "11-S independent episode choice/reset seed/source differs")
                if saved["terrain"] == "flat":
                    stem = f"training_episode_{episode_id:06d}_flat_corridor"
                    footprint = document(folder / f"{stem}_precheck.json")
                    require(footprint["passed"] is True
                            and footprint["counters_unchanged"] is True
                            and footprint["nominal_only_not_actual_trajectory"] is True
                            and footprint["geometry_sha256"]
                            == identity(folder / footprint["geometry_file"])["sha256"]
                            and footprint["centerline_sha256"]
                            == identity(folder / footprint["centerline_file"])["sha256"]
                            and len(arrays(folder / footprint["centerline_file"])
                                    ["xy_m"]) == 1001,
                            "11-S flat actual reset footprint provenance differs")
                episodes[episode_id] = {"schedule": saved, "count": 0,
                                        "servo_vx": 0.0, "servo_yaw": 0.0,
                                        "terminated": False, "truncated": False}
            episode = episodes[episode_id]
            raw = episode["schedule"]["raw_commands"][tick]
            require(tick == episode["count"] and tick < 1000
                    and not episode["terminated"] and not episode["truncated"],
                    "11-S training episode tick skipped or continued after terminal")
            episode["servo_vx"] += max(-.005, min(.005,
                raw["forward_velocity_mps"] - episode["servo_vx"]))
            episode["servo_yaw"] += max(-.006, min(.006,
                raw["yaw_rate_rps"] - episode["servo_yaw"]))
            require(close(a["raw_command_vx_vy_yaw_clearance"][offset], [
                raw[k] for k in ("forward_velocity_mps", "lateral_velocity_mps",
                                 "yaw_rate_rps", "clearance_m")])
                and close(a["servo_command_vx_vy_yaw_clearance"][offset], [
                    episode["servo_vx"], 0.0, episode["servo_yaw"], .455])
                and not (a["terminated"][offset] and a["truncated"][offset]),
                "11-S raw/servo/terminal compact row differs")
            if tick < 175:
                require(not np.any(a["effective_action16"][offset]),
                        "11-S residual physically applied during raw-zero settle")
            else:
                require(np.array_equal(a["effective_action16"][offset],
                                       a["policy_clipped16"][offset]),
                        "11-S forward residual differs from clipped action")
            episode["count"] += 1
            episode["terminated"] = bool(a["terminated"][offset])
            episode["truncated"] = bool(a["truncated"][offset])
            require(not episode["truncated"] or tick == 999,
                    "11-S training time-limit truncation did not occur at 1000")
        seen += 1024
    require(seen == n and nonzero > 0 and nonwheel >= 0,
            "11-S training control prefix/effective action absent")
    partial = manifest["partial_episode"]
    require(partial is None or (partial["episode_index"] == current_episode
            and partial["completed_controls"] == episodes[current_episode]["count"]),
            "11-S final incomplete episode marker differs")
    closed = sum(int(row["terminated"] or row["truncated"]) for row in episodes.values())
    require(closed == manifest["closed_episodes"],
            "11-S closed episode count differs from saved compact terminal rows")
    with (folder / "curriculum_events.ndjson").open(encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream]
    resets = [row for row in events if row["event"] == "episode_reset"]
    closings = [row for row in events if row["event"] == "episode_closed"]
    postbudget = [row for row in events if row["event"] == "zero_control_postbudget_reset"]
    require(events[0]["event"] == "curriculum_initialized"
            and events[0]["measurement_seed_stream_seed"] == 88623
            and events[-1]["event"] == "training_sealed"
            and events[-1]["completed_controls"] == 65536
            and len(resets) == len(episodes) and len(closings) == closed
            and len(postbudget) == manifest["zero_control_postbudget_resets"] <= 1
            and all(row["completed_global_controls"] == 65536 for row in postbudget)
            and all(row["episode_index"] == index
                    and row["raw_command_sha256"]
                    == episodes[index]["schedule"]["raw_command_sha256"]
                    for index, row in enumerate(resets))
            and all(row["episode_index"] == index
                    and row["completed_controls"] == episodes[index]["count"]
                    and row["terminated"] == episodes[index]["terminated"]
                    and row["truncated"] == episodes[index]["truncated"]
                    for index, row in enumerate(closings)),
            "11-S durable fixed curriculum events differ from numeric stream")
    learning = document(folder / "learning_receipt.json")
    actual = learning["actual"]
    require(learning["status"] == "complete"
            and learning["qualified_for_final_checkpoint"] is True
            and learning["violations"] == learning["failures"] == []
            and [actual[k] for k in ("num_timesteps", "transitions", "train_calls",
                                     "epochs", "optimizer_steps", "rollouts")]
            == [65536, 65536, 64, 256, 1024, 64]
            and len(learning["updates"]) == len(learning["rollouts"]) == 64
            and learning["initial_hashes"]["policy_state"]
            != learning["final_hashes"]["policy_state"]
            and learning["initial_hashes"]["optimizer_state"]
            != learning["final_hashes"]["optimizer_state"]
            and learning["policy_state_changed"] is True
            and learning["optimizer_state_changed"] is True,
            "11-S PPO actual updates or parameter hashes differ")
    for index, update in enumerate(learning["updates"]):
        require(update["update_index"] == index
                and update["num_timesteps"] == (index + 1) * 1024
                # PPO.train performs updates after rollout collection; no
                # environment control is sampled inside the train call.
                and update["timesteps_delta"] == 0
                and update["n_updates_delta"] == 4
                and update["optimizer_steps"] == 16
                and update["nonfinite_grad_steps"] == 0
                and update["missing_grad_params"] == 0,
                "11-S PPO per-rollout update count/gradient failure differs")
    for index in range(64):
        progress = document(folder / f"progress_rollout_{index:04d}.json")
        require(progress["completed_controls"] == progress["gaussian_records"]
                == (index + 1) * 1024
                and progress["native_attempted"] == progress["native_returned"]
                == progress["native_checked"] == (index + 1) * 5120
                and progress["audit_rollouts"] == index + 1
                and progress["audit_transitions"] == (index + 1) * 1024
                and progress["train_calls_completed_before_pending_update"] == index
                and progress["optimizer_steps_before_pending_update"] == 16 * index,
                "11-S durable rollout/native progress differs")
    native = training_native(folder, guard, n)
    return {"controls": n, "native": native, "episodes_started": len(episodes),
            "closed_episodes": manifest["closed_episodes"],
            "unfinished_episode": partial is not None,
            "nonzero_effective_controls": nonzero,
            "Gaussian_clipped_controls": clipped,
            "compact_nonwheel_candidates": nonwheel,
            "learning_actual": actual,
            "full_contact_force_qualification_available": False}


def training_native(folder: Path, guard: dict, controls: int) -> dict:
    names, array_names = guard["native_files"], guard["train_array_files"]
    require(len(names) == len(array_names) == math.ceil(controls / 1024),
            "11-S compact native block count differs")
    checked = nonwheel = 0
    for index, (name, array_name) in enumerate(zip(names, array_names)):
        require(name == f"native_block_{index:04d}.jsonl.gz"
                and array_name == f"native_arrays_{index:04d}.npz",
                "11-S compact native block ordering differs")
        with gzip.open(folder / name, "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        a = arrays(folder / array_name)
        controls = arrays(folder / "training_numeric_blocks"
                          / f"controls_{index:04d}.npz")
        require(len(rows) == 5120 and all(v.shape[0] == 5120
                and np.isfinite(v).all() for v in a.values()),
                "11-S compact native array row/finite differs")
        require(a["ctrl_after"].shape == (5120, 16)
                and controls["actuator_applied5x16"].shape == (1024, 5, 16)
                and np.array_equal(a["ctrl_after"].reshape(1024, 5, 16),
                                   controls["actuator_applied5x16"])
                and np.array_equal(a["ctrl_before"].reshape(1024, 5, 16),
                                   controls["actuator_applied5x16"])
                and np.array_equal(controls["actuator_applied5x16"],
                                   np.broadcast_to(controls["stage_safe_torque_nm"][:, None],
                                                   (1024, 5, 16))),
                "11-S every native ctrl differs from its numeric safe torque")
        for field in ("qpos", "qvel", "qacc_warmstart"):
            before = a[f"{field}_before"].reshape(1024, 5, -1)
            after = a[f"{field}_after"].reshape(1024, 5, -1)
            require(np.array_equal(before[:, 1:], after[:, :-1]),
                    f"11-S compact native {field} chain breaks within a control")
        for offset, row in enumerate(rows):
            index_native = checked + offset
            require(row["native_index"] == index_native
                    and close(row["start_time_s"], a["start_time_s"][offset], atol=1e-12)
                    and close(row["end_time_s"], a["end_time_s"][offset], atol=1e-12)
                    and close(row["end_time_s"] - row["start_time_s"], .002, atol=1e-12)
                    and row["nonwheel_contact_count"] >= 0,
                    "11-S compact native index/clock/summary differs")
            nonwheel += row["nonwheel_contact_count"]
        require(np.allclose(a["end_time_s"] - a["start_time_s"], .002,
                            rtol=0, atol=1e-12)
                and np.array_equal(a["ctrl_before"], a["ctrl_after"])
                and np.allclose(a["actuator_force_after"], a["ctrl_after"],
                                rtol=0, atol=1e-8),
                "11-S compact native actuator/time chain differs")
        checked += len(rows)
    require(checked == 5 * controls, "11-S compact native count differs")
    return {"native_rows_checked": checked,
            "nonwheel_candidates_recorded": nonwheel,
            "full_contact_force_qualification_available": False}


def heldout_schedule(folder: Path, spec: tuple, actor: str) -> dict:
    case_id, terrain, speed, seed, cap, hold_start, release = spec
    saved = document(folder / "schedule.json")
    require(saved["schema"] == "d1-world-upright-short-rl16-heldout-v1"
            and (saved["case_id"], saved["actor"], saved["seed"],
                 saved["terrain"]) == (case_id, actor, seed, terrain)
            and saved["spawn_position_m"] == SPAWNS[terrain]
            and saved["control_cap"] == cap
            and len(saved["raw_commands"]) == cap,
            "11-S preregistered heldout schedule identity differs")
    for tick, command in enumerate(saved["raw_commands"]):
        vx = speed if 175 <= tick < release else 0.0
        yaw = 0.0
        if case_id == "flat_1p2_yaw" and hold_start <= tick < hold_start + 400:
            local = tick - hold_start
            yaw = .3 if local < 100 or local >= 300 else -.3
        require(command["forward_velocity_mps"] == vx
                and command["lateral_velocity_mps"] == 0.0
                and command["yaw_rate_rps"] == yaw
                and command["clearance_m"] == .455
                and command["jump_requested"] is False,
                f"11-S heldout raw schedule differs at {case_id}:{tick}")
    digest = hashlib.sha256(json.dumps(saved["raw_commands"], sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(saved["raw_command_sha256"] == digest,
            "11-S heldout saved raw command digest differs")
    return saved


def control_rows(folder: Path, blocks: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for index, block in enumerate(blocks):
        require(block["file"] == f"control_records_{index:04d}.jsonl.gz"
                and 0 < block["rows"] <= 200,
                "11-S heldout control block order differs")
        path = folder / block["file"]
        match_file(path, block)
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            part = [json.loads(line) for line in stream]
        require(len(part) == block["rows"], "11-S heldout block count differs")
        rows.extend(part)
    return rows


def heldout_case(run: Path, spec: tuple, actor: str, binding: dict,
                 expected_geometry: dict) -> tuple[dict, dict]:
    case_id, terrain, speed, seed, cap, _hold, _release = spec
    folder = run / "heldout" / f"{case_id}_{actor}"
    receipt = document(folder / "case_receipt.json")
    schedule = heldout_schedule(folder, spec, actor)
    reset = document(folder / "reset_receipt.json")
    before = document(folder / "boundary_before.json")
    rows = control_rows(folder, receipt["controller_record_blocks"])
    guard = receipt["native_segment"]
    native = _native_rows(folder, guard)
    states = arrays(folder / "states.npz")
    initial = arrays(folder / "initial_state.npz")
    require(document(folder / "geometry_manifest.json") == expected_geometry,
            "heldout actual compiled 92 geom manifest changed between cases")
    geometry = geom_map(folder, binding)
    n = len(rows)
    require(receipt["schema"] == "d1-world-upright-short-rl16-heldout-v1"
            and (receipt["case_id"], receipt["actor"], receipt["seed"],
                 receipt["terrain"]) == (case_id, actor, seed, terrain)
            and receipt["raw_command_sha256"] == schedule["raw_command_sha256"]
            and 1 <= receipt["completed_controls"] == n <= cap
            and receipt["control_cap"] == cap
            and receipt["failure"] is None and receipt["record_valid"] is True
            and receipt["termination_recorded"] is True
            and receipt["policy_prediction_attempts"]
            == receipt["policy_predictions"] == (n if actor == "final_policy" else 0)
            and guard["mode"] == "heldout"
            and guard["native_attempted"] == guard["native_returned"] == len(native) == 5*n
            and guard["failure"] is None
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
            == (n if actor == "final_policy" else 0),
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
        "case_id": case_id, "actor": actor, "terrain": terrain, "seed": seed,
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
                and record["policy_predict_called"] is (actor == "final_policy")
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


def exact_pairs(run: Path, summaries: list[dict]) -> None:
    require(len(summaries) == 12, "11-S heldout pair count differs")
    for index, spec in enumerate(CASES):
        zero, policy = summaries[2*index:2*index+2]
        require((zero["case_id"], policy["case_id"]) == (spec[0], spec[0])
                and (zero["actor"], policy["actor"]) == ("zero", "final_policy")
                and zero["seed"] == policy["seed"] == spec[3]
                and zero["actual_model_address"] == policy["actual_model_address"]
                and zero["actual_data_address"] == policy["actual_data_address"],
                "11-S preregistered matched pair identity differs")
        first = run / "heldout" / f"{spec[0]}_zero"
        second = run / "heldout" / f"{spec[0]}_final_policy"
        a, b = arrays(first / "initial_state.npz"), arrays(second / "initial_state.npz")
        reset_a, reset_b = document(first / "reset_receipt.json"), document(
            second / "reset_receipt.json")
        require(reset_a["exact_initial_pair"] is False
                and reset_b["exact_initial_pair"] is True
                and reset_b["paired_with"] == "zero"
                and all(a[field].dtype == b[field].dtype
                        and a[field].shape == b[field].shape
                        and a[field].tobytes() == b[field].tobytes()
                        for field in PAIR_FIELDS),
                "11-S five-array initial zero/policy states differ bitwise")


def checkpoint(run: Path, learning: dict, session: dict) -> dict:
    manifest = document(run / "final_checkpoint_manifest.json")
    folder = run / "final_checkpoint"
    trusted = manifest["trusted_fields"]
    require(manifest["folder"] == str(folder)
            and trusted["run_id"] == session["run_id"]
            and trusted["observation_size"] == 99
            and trusted["action_size"] == 16
            and trusted["task_schema"] == "d1-course-world-upright-rl16-task-v1"
            and trusted["reference_schema"]
            == "d1-course-world-upright-roll-pitch-zero-v1"
            and trusted["reward_schema"]
            == "d1-course-world-upright-bodycom-yaw-terrain-action-torque-v1"
            and trusted["ppo_seed"] == 88621,
            "11-S final checkpoint identity/task differs")
    for name, expected in manifest["files"].items():
        match_file(folder / name, expected)
    metadata = folder / "final_metadata.json"
    require(identity(metadata)["sha256"] == manifest["metadata_sha256"],
            "11-S final metadata hash differs")
    doc = document(metadata)
    require(doc["files"] == manifest["files"]
            and doc["training_receipt"] == learning
            and doc["observation_size"] == 99 and doc["action_size"] == 16
            and doc["task_schema"] == trusted["task_schema"]
            and doc["reference_schema"] == trusted["reference_schema"],
            "11-S saved checkpoint metadata/actual learning identity differs")
    saved_reload = document(run / "final_checkpoint_reload.json")
    first_reload = manifest["reload_verification"]
    require(first_reload["reference_policy_byte_exact"] is True
            and saved_reload["reference_policy_byte_exact"] is None
            and all(saved_reload[key] == value for key, value in first_reload.items()
                    if key != "reference_policy_byte_exact")
            and saved_reload["matches_training_final_policy_state"] is True,
            "11-S two strict reload/probe receipts differ")
    origin = document(run / "training/final_probe_origin.json")
    source_file = run / "training/training_numeric_blocks/controls_0000.npz"
    require(origin["source_file"] == str(source_file)
            and identity(source_file)["sha256"] == origin["source_file_sha256"]
            and origin["rows"] == [0, 32]
            and origin["additional_physical_probe_calls"] == 0,
            "11-S final probe did not use first32 actual training observations")
    first = arrays(source_file)["input_observation99"][:32]
    require((folder / "final_probe_observations_f32.bin").read_bytes()
            == np.ascontiguousarray(first, dtype=np.float32).tobytes(),
            "11-S saved final probe bytes differ from training states")
    return {"files": manifest["files"],
            "metadata_sha256": manifest["metadata_sha256"],
            "strict_reload_saved_evidence": True,
            "model_imported_or_loaded_by_reader": False}


def manifest(run: Path) -> dict:
    files = {}
    for path in sorted(run.rglob("*")):
        require(not path.is_symlink(), "11-S saved run contains symlink")
        if path.is_file():
            files[path.relative_to(run).as_posix()] = identity(path)
    return {"schema": "d1-world-upright-short-rl16-closed-manifest-11-v1",
            "run": str(run), "file_count": len(files),
            "total_bytes": sum(row["bytes"] for row in files.values()),
            "files": files}


def verify(run: Path) -> tuple[dict, dict]:
    require(run.is_absolute() and run.is_dir(), "11-S --run must be absolute saved directory")
    session, worker, origins = source(run)
    initial = document(run / "runtime_initial.json")
    construction = document(run / "construction_receipt.json")
    require(worker["failure"] is None and worker["warnings"] == []
            and worker["execution_complete"] is True
            and worker["retry_permitted"] is False
            and initial["proof"]["passed"] is True
            and initial["proof"]["dso_path"] == session["library"]
            and initial["proof"]["dso_sha256"]
            == session["source_hashes"][session["library"]]["sha256"]
            and len(initial["proof"]["jump_slots"]) == 4
            and all(row["passed"] is True for row in initial["proof"]["jump_slots"])
            and initial["C_state"]["control_attempts"]
            == initial["C_state"]["control_returns"] == 0
            and construction["C_state"]["construction_attempts"]
            == construction["C_state"]["construction_returns"] == 2
            and construction["nominal_cache"]["misses"] == 1,
            "11-S actual worker/GOT/cold2 provenance differs")
    actual_origins = document(run / "module_origins.json")
    require(actual_origins["engine_binding"] == session["binding_module"]
            and all(path in session["source_hashes"] for path in actual_origins.values()),
            "11-S actual imported module origin not in frozen source closure")
    for name in ("training/learning_dependency_origins_before_physics.json",
                 "learning_dependency_origins_before_heldout.json"):
        dependency = document(run / name)
        require(all(path in session["source_hashes"] for path in
                    dependency["module_origins"].values())
                and set(dependency["synthetic_module_aliases"])
                == {"torch.ops", "torch.classes"}
                and all(alias["synthetic_module_alias"] is True
                        and alias["identity_verified"] is True
                        and alias["defining_source"] in session["source_hashes"]
                        for alias in dependency["synthetic_module_aliases"].values()),
                "11-S actual learning dependency origin not frozen")
    binding = check_binding(construction["actual_geometry_binding"])
    training_result = training(run)
    require(worker["training_guard_segment"] == document(
        run / "training/training_segment_receipt.json")["native_segment"],
        "11-S worker training native boundary differs from saved segment")
    saved_learning = document(run / "training/learning_receipt.json")
    final_checkpoint = checkpoint(run, saved_learning, session)
    mode = document(run / "heldout_mode_transition.json")
    pair = [construction["course_plant_model_address"],
            construction["course_plant_data_address"]]
    require(mode["from"] == "train" and mode["to"] == "eval"
            and mode["same_compiled_model_data_pair"] == pair
            and mode["numerical_controller_unchanged"] is True
            and mode["training_partial_episode_preserved"] is True,
            "11-S heldout changed compiled plant/controller")
    summaries, canonical = [], []
    for spec in CASES:
        for actor in ("zero", "final_policy"):
            row, numeric = heldout_case(run, spec, actor, binding,
                                        construction["compiled_geometry"])
            require([row["actual_model_address"], row["actual_data_address"]] == pair,
                    "11-S heldout changed actual model/data address")
            summaries.append(row)
            canonical.append(numeric)
    exact_pairs(run, summaries)
    predicted = sum(row["predict_calls"] for row in summaries)
    require(predicted <= 9800 and worker["heldout_cases"] == [
        {key: document(run / "heldout" / f"{row['case_id']}_{row['actor']}"
                       / "case_receipt.json")[key]
         for key in ("case_id", "actor", "seed", "terrain", "control_cap",
                     "completed_controls", "policy_prediction_attempts",
                     "policy_predictions", "record_valid", "terminated",
                     "truncated", "stop_reason")}
        for row in summaries
    ], "11-S worker's twelve heldout summaries/predict counts differ")
    calls = worker["actual_model_calls"]
    phases = worker["actual_prediction_calls_by_phase"]
    require(calls == {"learn": 1, "save": 1, "load": 2,
                      "predict": 4 + predicted}
            and phases["training_and_finalization"] == 4
            and len(worker["actual_final_probe_batches"]) == 4
            and all(row["shape"] == [32, 99] and row["deterministic"] is True
                    for row in worker["actual_final_probe_batches"])
            and all(phases.get("heldout:" + row["case_id"] + ":" + row["actor"], 0)
                    == row["predict_calls"] for row in summaries),
            "11-S actual model learn/save/load/probe/policy call count differs")
    c, py = worker["C_final"], worker["python"]
    completed = 65536 + sum(row["completed_controls"] for row in summaries)
    require(completed <= 85136
            and c["control_attempts"] == c["control_returns"] == 5 * completed
            and py["control_attempted"] == py["control_completed"] == completed
            and py["native_attempted"] == py["native_returned"] == 5 * completed
            and py["native_failed"] == 0
            and py["fatal_latched"] is False
            and py["clock_advanced_substeps"] == 5 * completed
            and worker["native_guard"]["native_returned"] == 5 * completed
            and worker["native_guard"]["native_checked"] == 5 * completed
            and c["construction_attempts"] == c["construction_returns"] == 2
            and c["violations"] == py["forbidden_entries"] == 0
            and c["phase"] == c["target_model"] == c["target_data"] == 0
            and c["native_construction_caller_verified"] is True
            and worker["control_step_caller_verified_against_closed_E_callsite"] is True,
            "11-S final C/Python/native true 5T accounting differs")
    require(c["ccd_attempts"] == c["ccd_returns"]
            and (c["ccd_attempts"] == 0 and c["first_ccd_caller"] == 0
                 or c["ccd_attempts"] > 0 and c["native_ccd_caller_verified"] is True),
            "11-S final CCD native caller/count differs")
    scores = [score_case(case) for case in canonical]
    aggregate = score_pairs(scores)
    result = {
        "schema": "d1-world-upright-short-rl16-independent-readback-11-v1",
        "run": str(run), "source_closure": origins,
        "offline_reader_source": {"path": str(Path(__file__).resolve()),
                                  **identity(Path(__file__).resolve())},
        "execution_complete": True, "training": training_result,
        "checkpoint": final_checkpoint,
        "heldout_cases": summaries,
        "all_twelve_full_native_controller_contact_records_verified": True,
        "six_initial_state_pairs_bitwise_exact": True,
        "numeric_scores": {"source": str(HERE / "rl16_heldout_score_11.py"),
                           "source_identity": identity(HERE / "rl16_heldout_score_11.py"),
                           "cases": scores, "aggregate": aggregate},
        "all_six_policy_task_qualified_after_raw_readback": bool(
            aggregate["all_six_policy_task_qualified"]),
        "RL_contribution_passed_after_raw_readback": bool(
            aggregate["RL_contribution_passed"]),
        "normal_native_returned": 5 * completed,
        "compiler_native_returned": 2,
        "engine_model_learner_or_policy_imported": False,
        "physics_replayed_or_rescored": False,
        "compact_training_full_contact_force_qualified": False,
    }
    return result, manifest(run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    require(args.run.is_absolute(), "11-S --run must be absolute")
    result, closed = verify(args.run.resolve(strict=True))
    for path in (args.output, args.manifest_output):
        if path is not None:
            require(path.is_absolute() and not path.exists(),
                    "11-S output must be a new absolute path")
    if args.output is None:
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    if args.manifest_output is not None:
        with args.manifest_output.open("x", encoding="utf-8") as stream:
            json.dump(closed, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
