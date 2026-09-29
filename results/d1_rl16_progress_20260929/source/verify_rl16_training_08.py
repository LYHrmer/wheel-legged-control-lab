"""Independent, saved-record-only 08-T readback; never imports the simulator.

The training recorder is compact and cannot prove full native contact forces.
Heldout episodes retain full native/contact records, checked separately here.
The frozen preregistered pure scorer is connected after independent raw checks.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

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

W = Path(__file__).resolve().parent
E_READER_SHA = "c7d7b1ea343f29c6e108650bd707ee14e6515a873d55c3abe02c424a009b1419"
CONTRACT_SHA = "475273fd4517263606af7abd07008cdd6811f9992e6affacf6eb758a9ccbddd7"
SCHEMA = "d1-course-t-rl16-training-heldout-08-v1"
CASES = (
    ("flat_0p6", "flat", .6, 88501),
    ("flat_1p6", "flat", 1.6, 88502),
    ("flat_1p2_yaw", "flat", 1.2, 88503),
    ("bumps_0p4", "bumps", .4, 88504),
    ("rough_0p35", "rough", .35, 88505),
    ("ramp_0p35", "ramp", .35, 88506),
)
PAIR_FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation")
TRAIN_KEYS = (
    "control_index", "episode_index", "episode_tick", "input_observation99",
    "policy_env_input16", "policy_clipped16", "effective_action16",
    "raw_command_vx_vy_yaw_clearance", "servo_command_vx_vy_yaw_clearance",
    "reward_terms9", "reward", "terminated", "truncated", "body_com_vx_mps",
    "body_com_vy_mps", "body_yaw_rate_rps", "base_x_m", "base_y_m",
    "clearance_m", "relative_roll_rad", "relative_pitch_rad",
    "native_nonwheel_contacts", "actuator_delayed5x16", "actuator_applied5x16",
    "stage_safe_torque_nm",
)


def _json(path: Path) -> dict:
    value = read(path)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file() and not path.is_symlink(), f"saved NPZ absent: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _file_matches(path: Path, expected: dict) -> None:
    require(identity(path) == {key: expected[key] for key in ("sha256", "bytes")},
            f"saved source/block hash changed: {path}")


def _source(run: Path) -> tuple[dict, dict]:
    require(identity(W / "verify_course_e_08_03.py")["sha256"] == E_READER_SHA,
            "pure E geometry/controller arithmetic helper changed")
    session = _json(run / "session.json")
    require(session["schema"] == SCHEMA and session["phase"] == "T"
            and session["output_directory"] == str(run)
            and session["contract_sha256"] == CONTRACT_SHA
            and session["control_limit"] == 281344
            and session["normal_native_limit"] == 1406720
            and session["compiler_native_limit"] == 2
            and session["wallclock_limit_s"] == 7200
            and session["training_control_limit"] == 262144
            and session["heldout_control_limit"] == 19200
            and session["segments"] == [262144] + [1600] * 12
            and session["seed"] == 88401 and session["retry_permitted"] is False,
            "T session/prepaid cap is not the frozen protocol")
    frozen = session["source_hashes"]
    require(isinstance(frozen, dict) and len(frozen) >= 100,
            "T frozen input closure absent")
    for name, expected in frozen.items():
        _file_matches(Path(name), expected)
    for name in ("run_rl16_training_08.py", "launch_rl16_training_08.py",
                 "course_impl08/rl16_learning_08.py",
                 "course_impl08/rl16_curriculum_08.py",
                 "course_impl08/rl16_heldout_score_08.py",
                 "rl_heldout_scoring_addendum_08_05.md"):
        require(str(W / name) in frozen, "T critical input not frozen: " + name)
    require(identity(Path(session["contract_path"]))["sha256"] == CONTRACT_SHA,
            "T contract drifted")
    go = W / "astra_rl16_training_go_08.json"
    require(identity(go)["sha256"] == session["go_sha256"], "T actual GO differs")
    decision = _json(go)
    require(decision["decision"] == "GO" and decision["contract_sha256"] == CONTRACT_SHA,
            "T actual GO is not the signed decision")
    for name, expected in decision["inputs"].items():
        _file_matches(Path(name), expected)
    actual_origins = _json(run / "module_origins.json")
    course_root = W / "course_impl08"
    require(actual_origins.get("engine_binding") == session["binding_module"]
            and all(path in frozen and (
                path == session["binding_module"]
                if name == "engine_binding" else Path(path).parent == course_root
            ) for name, path in actual_origins.items()),
            "T actual imported engine/course module origins are unfrozen")
    learning_origins = _json(run / "training/learning_module_origins.json")
    require(set(learning_origins) == {"rl16_curriculum_08", "rl16_learning_08"}
            and all(path in frozen and Path(path).parent == course_root
                    for path in learning_origins.values()),
            "T actual curriculum/learning module origin differs")
    for name in ("learning_dependency_origins_before_physics.json",
                 "learning_dependency_origins_before_heldout.json"):
        path = run / "training" / name if name.endswith("before_physics.json") else run / name
        if path.is_file():
            loaded = _json(path)
            origins = loaded["module_origins"]
            aliases = loaded["synthetic_module_aliases"]
            require(all(origin in frozen for origin in origins.values())
                    and all(any(module == root or module.startswith(root + ".")
                                for root in ("torch", "stable_baselines3", "numpy", "gymnasium"))
                            for module in origins)
                    and set(aliases) == {"torch.ops", "torch.classes"}
                    and all(alias["synthetic_module_alias"] is True
                            and alias["identity_verified"] is True
                            and alias["defining_source"] == origins[key]
                            and alias["defining_source"] in frozen
                            for key, alias in aliases.items()),
                    "T actual loaded learning dependency is outside frozen origins")
    return session, {"verified_frozen_input_count": len(frozen),
                     "go_input_count": len(decision["inputs"]),
                     "actual_module_origins_checked": len(actual_origins),
                     "go_sha256": session["go_sha256"],
                     "pure_E_helper_sha256": E_READER_SHA}


def _boundary_delta(before: dict, after: dict, controls: int, *, compiler: int = 0) -> None:
    cb, ca = before["C_state"], after["C_state"]
    pb, pa = before["python"], after["python"]
    require(ca["construction_returns"] - cb["construction_returns"] == compiler
            and ca["construction_attempts"] - cb["construction_attempts"] == compiler
            and ca["control_attempts"] - cb["control_attempts"] == 5 * controls
            and ca["control_returns"] - cb["control_returns"] == 5 * controls
            and pa["control_attempted"] - pb["control_attempted"] == controls
            and pa["control_completed"] - pb["control_completed"] == controls
            and pa["native_attempted"] - pb["native_attempted"] == 5 * controls
            and pa["native_returned"] - pb["native_returned"] == 5 * controls,
            "T C/Python segment delta is not exact 5T without refund")


def _read_control_rows(folder: Path, blocks: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for index, block in enumerate(blocks):
        require(block["file"] == f"control_records_{index:04d}.jsonl.gz"
                and 0 < block["rows"] <= 200, "heldout control block ordering differs")
        path = folder / block["file"]
        _file_matches(path, block)
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            part = [json.loads(line) for line in stream]
        require(len(part) == block["rows"], "heldout block row count differs")
        rows.extend(part)
    return rows


def _training_schedule(folder: Path, episode_index: int) -> dict:
    saved = _json(folder / f"training_episode_{episode_index:06d}_schedule.json")
    require(saved["episode_index"] == episode_index
            and saved["schedule_seed"] == 88401
            and len(saved["raw_commands"]) == 1600
            and saved["nominal_path_precheck"]["passed"] is True,
            "training seeded episode schedule/precheck missing")
    level, local = saved["level"], saved["level_episode_index"]
    require(type(level) is int and level in range(4)
            and type(local) is int and local >= 0,
            "training curriculum level/local episode invalid")
    terrains = (("flat",), ("flat",), ("flat", "bumps"),
                ("flat", "bumps", "rough", "ramp"))[level]
    require(saved["terrain"] == terrains[local % len(terrains)],
            "training terrain differs from fixed level episode cycle")
    target = saved["target_speed_mps"]
    if saved["terrain"] == "flat":
        ceiling = (.4, .9, 1.6, 1.6)[level]
        flat_ordinal = saved["flat_episode_ordinal"]
        require(saved["spawn_position_m"] == [-8.0, -4.7, .455]
                and type(flat_ordinal) is int and flat_ordinal >= 0
                and 0 <= target <= ceiling
                and ((flat_ordinal + 1) % 4 != 0 or target == ceiling)
                and (saved["yaw_amplitude_rps"] == 0.0 if level == 0
                     else abs(saved["yaw_amplitude_rps"]) <= .3),
                "training flat target/yaw/every-fourth ceiling differs")
    else:
        require(saved["flat_episode_ordinal"] is None
                and .2 <= target <= .4
                and saved["yaw_amplitude_rps"] == 0.0,
                "training terrain target/yaw differs")
    # Training yaw amplitude is seeded in [-.3,.3], not only heldout +.3.
    ramp = math.ceil(saved["target_speed_mps"] / .005 - 1e-12)
    begin, end = 175 + ramp, 575 + ramp
    amplitude = saved["yaw_amplitude_rps"]
    for tick, row in enumerate(saved["raw_commands"]):
        hold_tick = tick - begin
        expected_yaw = (amplitude if hold_tick < 100 or hold_tick >= 300
                        else -amplitude) if 0 <= hold_tick < 400 else 0.0
        require(row == {
            "forward_velocity_mps": 0.0 if tick < 175 or tick >= end else saved["target_speed_mps"],
            "lateral_velocity_mps": 0.0, "yaw_rate_rps": expected_yaw,
            "clearance_m": .455, "jump_requested": False,
        }, "training raw seeded schedule changed")
    import hashlib

    encoded = [[row[key] for key in ("forward_velocity_mps", "lateral_velocity_mps",
                                     "yaw_rate_rps", "clearance_m", "jump_requested")]
               for row in saved["raw_commands"]]
    digest = hashlib.sha256(json.dumps(encoded, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()
    require(digest == saved["raw_command_sha256"],
            "training seeded schedule hash differs")
    return saved


def _curriculum_events(folder: Path, episodes: dict[int, dict], full: bool) -> dict:
    path = folder / "curriculum_events.ndjson"
    require(path.is_file() and not path.is_symlink(), "T curriculum events absent")
    with path.open(encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream]
    require(events and events[0]["event"] == "curriculum_initialized"
            and events[0]["schedule_seed"] == 88401
            and events[0]["measurement_seed_stream_seed"] == 88402,
            "T curriculum seeded event origin differs")
    level = resets = closes = 0
    recent: dict[int, list[bool]] = {index: [] for index in range(4)}
    level_episode_counts = [0, 0, 0, 0]
    flat_episode_counts = [0, 0, 0, 0]
    promotions = []
    for event in events[1:]:
        kind = event["event"]
        if kind == "level_promoted":
            old = event["from"]
            window = recent[old][-10:]
            require(old == level and event["to"] == old+1
                    and event["completed_global_controls"] >= 65536*(old+1)
                    and len(window) == 10 and sum(window) >= 8
                    and event["recent_closed_episodes"] == 10
                    and event["recent_successes"] == sum(window),
                    "T curriculum promotion was not earned by recent ten episodes")
            level += 1
            promotions.append({"from": old, "to": level,
                               "control_index": event["completed_global_controls"]})
        elif kind == "episode_reset":
            schedule = episodes[resets]["schedule"]
            boundary_level = min(schedule["start_completed_global_controls"] // 65536, 3)
            recent_window = recent[level][-10:]
            require(not (boundary_level > level and len(recent_window) == 10
                         and sum(recent_window) >= 8),
                    "T eligible curriculum promotion was skipped before reset")
            require(event["level"] == schedule["level"] == level
                    and schedule["level_episode_index"] == level_episode_counts[level]
                    and event["case_id"] == schedule["case_id"]
                    and event["terrain"] == schedule["terrain"]
                    and event["measurement_seed"] == schedule["measurement_seed"]
                    and event["completed_global_controls"]
                    == schedule["start_completed_global_controls"],
                    "T event/actual episode seeded reset differs")
            level_episode_counts[level] += 1
            if schedule["terrain"] == "flat":
                require(schedule["flat_episode_ordinal"] == flat_episode_counts[level],
                        "T fixed every-fourth flat ceiling ordinal differs")
                flat_episode_counts[level] += 1
            resets += 1
        elif kind == "episode_closed":
            actual = episodes[closes]
            require(event["episode_index"] == closes
                    and event["end_completed_global_controls"] == actual["end"]
                    and event["completed_controls"] == actual["count"]
                    and close(event["drive_sse"], actual["drive_sse"], atol=1e-7)
                    and event["drive_count"] == actual["drive_count"]
                    and close(event["reward_sum"], actual["reward_sum"], atol=1e-7)
                    and event["terminated"] == actual["terminated"]
                    and event["truncated"] == actual["truncated"],
                    "T saved training episode result differs from compact controls")
            rms = (math.sqrt(actual["drive_sse"] / actual["drive_count"])
                   if actual["drive_count"] else None)
            success = bool(not actual["terminated"] and actual["truncated"]
                           and rms is not None and rms <= .12)
            require(event["promotion_eligible"] is success,
                    "T actual episode promotion success flag differs")
            recent[level].append(success)
            closes += 1
        elif kind == "zero_control_postbudget_reset":
            require(event["completed_global_controls"] == 262144,
                    "T unbudgeted reset was not after the sealed final control")
        elif kind == "training_sealed":
            require(event is events[-1]
                    and (not full or event["completed_controls"] == 262144),
                    "T training seal event/count differs")
        else:
            raise ValueError("unexpected T curriculum event: " + kind)
    require(resets == len(episodes) and closes <= resets and level <= 3,
            "T curriculum event/episode count or promotion level differs")
    return {"seeded_episode_resets": resets, "closed_episodes": closes,
            "level_reached": level, "promotions": promotions}


def _training(run: Path, full: bool) -> dict:
    folder = run / "training"
    manifest = _json(folder / "training_blocks_manifest.json")
    segment = _json(folder / "training_segment_receipt.json")
    before = _json(folder / "boundary_before.json")
    after = segment["boundary_after"]
    n = int(manifest["completed_controls"])
    require(n == segment["completed_controls"] == segment["curriculum_completed_controls"]
            and n <= 262144 and manifest["gaussian_records"] == n
            and segment["gaussian_records"] == n
            and manifest["pending_gaussian_control_index"] is None,
            "training compact recorder/C curriculum count mismatch")
    _boundary_delta(before, after, n)
    require(segment["native_segment"]["native_returned"] == 5 * n
            and segment["native_segment"]["native_attempted"] == 5 * n,
            "training compact native count differs")
    blocks, gaussian = manifest["numeric_blocks"], manifest["gaussian_blocks"]
    require(len(blocks) == len(gaussian) and len(blocks) == math.ceil(n / 1024),
            "training compact block count differs")
    nonzero_effective = clipped_controls = native_nonwheel = 0
    episodes_seen: set[int] = set()
    episodes: dict[int, dict] = {}
    seen = 0
    for index, (left, right) in enumerate(zip(blocks, gaussian)):
        require(left["file"] == f"controls_{index:04d}.npz"
                and right["file"] == f"gaussian_{index:04d}.npz"
                and left["rows"] == right["rows"] == min(1024, n - seen)
                and left["first_control_index"] == right["first_control_index"] == seen
                and left["last_control_index"] == right["last_control_index"]
                == seen + left["rows"] - 1,
                "training numeric/Gaussian block index or length differs")
        _file_matches(folder / "training_numeric_blocks" / left["file"], left)
        _file_matches(folder / "training_gaussian_blocks" / right["file"], right)
        a = _load_npz(folder / "training_numeric_blocks" / left["file"])
        g = _load_npz(folder / "training_gaussian_blocks" / right["file"])
        require(set(TRAIN_KEYS) <= set(a) and all(
            values.shape[0] == left["rows"] and np.isfinite(values).all()
            for values in a.values()
        ), "training compact numeric block field/finite failure")
        require(np.array_equal(a["control_index"], np.arange(seen, seen + left["rows"]))
                and np.array_equal(g["control_index"], a["control_index"])
                and np.array_equal(g["episode_index"], a["episode_index"])
                and np.array_equal(g["episode_tick"], a["episode_tick"])
                and np.array_equal(g["clipped_action16"], a["policy_env_input16"])
                and np.array_equal(a["policy_clipped16"],
                                   np.clip(a["policy_env_input16"], -1, 1))
                and np.array_equal(g["effective_action16"], a["effective_action16"])
                and np.array_equal(np.clip(g["raw_gaussian_action16"], -1, 1),
                                   g["clipped_action16"]),
                "training actual Gaussian/action/compact index chain differs")
        require(np.allclose(a["reward_terms9"].sum(axis=1), a["reward"], rtol=0, atol=1e-9)
                and a["input_observation99"].shape == (left["rows"], 99)
                and a["effective_action16"].shape == (left["rows"], 16)
                and a["actuator_applied5x16"].shape == (left["rows"], 5, 16)
                and np.array_equal(a["actuator_applied5x16"],
                                   a["actuator_delayed5x16"])
                and np.array_equal(a["actuator_applied5x16"],
                                   np.broadcast_to(a["stage_safe_torque_nm"][:, None, :],
                                                   a["actuator_applied5x16"].shape)),
                "training reward/actual compact actuator link differs")
        nonzero_effective += int(np.count_nonzero(np.any(a["effective_action16"] != 0, axis=1)))
        clipped_controls += int(np.count_nonzero(np.any(
            g["raw_gaussian_action16"] != g["clipped_action16"], axis=1)))
        native_nonwheel += int(a["native_nonwheel_contacts"].sum())
        episodes_seen.update(int(x) for x in np.unique(a["episode_index"]))
        for offset in range(left["rows"]):
            episode_index = int(a["episode_index"][offset])
            tick = int(a["episode_tick"][offset])
            global_index = seen + offset
            if episode_index not in episodes:
                schedule = _training_schedule(folder, episode_index)
                require(tick == 0 and schedule["start_completed_global_controls"]
                        == global_index and episode_index == len(episodes),
                        "T seeded training episode did not start at the saved control")
                episodes[episode_index] = {
                    "schedule": schedule, "count": 0, "end": global_index,
                    "drive_sse": 0.0, "drive_count": 0, "reward_sum": 0.0,
                    "terminated": False, "truncated": False,
                    "servo_vx": 0.0, "servo_yaw": 0.0,
                }
            episode = episodes[episode_index]
            schedule = episode["schedule"]
            require(tick == episode["count"] and global_index == episode["end"]
                    and tick < 1600 and not episode["terminated"]
                    and not episode["truncated"],
                    "T compact controls skipped/repeated an episode tick")
            raw = schedule["raw_commands"][tick]
            raw_values = [raw[key] for key in (
                "forward_velocity_mps", "lateral_velocity_mps", "yaw_rate_rps", "clearance_m")]
            episode["servo_vx"] += max(-.005, min(.005,
                                                  raw_values[0]-episode["servo_vx"]))
            episode["servo_yaw"] += max(-.006, min(.006,
                                                   raw_values[2]-episode["servo_yaw"]))
            require(close(a["raw_command_vx_vy_yaw_clearance"][offset], raw_values)
                    and close(a["servo_command_vx_vy_yaw_clearance"][offset],
                              [episode["servo_vx"], 0.0, episode["servo_yaw"], .455])
                    and not (a["terminated"][offset] and a["truncated"][offset]),
                    "T actual compact raw/consumed schedule or terminal differs")
            if raw_values[0] == raw_values[2] == 0.0:
                require(not np.any(a["effective_action16"][offset]),
                        "T residual was physically applied after raw release")
            else:
                require(np.array_equal(a["effective_action16"][offset],
                                       a["policy_clipped16"][offset]),
                        "T actual residual action differs from clipped live policy input")
            if episode["servo_vx"] > 0.0:
                diff = float(a["body_com_vx_mps"][offset]) - episode["servo_vx"]
                episode["drive_sse"] += diff*diff
                episode["drive_count"] += 1
            episode["reward_sum"] += float(a["reward"][offset])
            episode["terminated"] = bool(a["terminated"][offset])
            episode["truncated"] = bool(a["truncated"][offset])
            episode["count"] += 1
            episode["end"] += 1
        seen += left["rows"]
    require(seen == n, "training compact blocks do not cover actual completed controls")
    curriculum = _curriculum_events(folder, episodes, full)
    require(curriculum["level_reached"] == manifest["level_reached"]
            and curriculum["closed_episodes"] == manifest["closed_episodes"],
            "T curriculum promotion/closed counts differ from numeric trace")
    learning = _json(folder / "learning_receipt.json") if (folder / "learning_receipt.json").is_file() else None
    if full:
        require(n == 262144 and manifest["complete"] is True and learning is not None,
                "full T run lacks 262144 controls and learning receipt")
        actual = learning["actual"]
        require(learning["status"] == "complete"
                and learning["qualified_for_final_checkpoint"] is True
                and learning["violations"] == learning["failures"] == []
                and [actual[key] for key in ("num_timesteps", "transitions",
                                             "train_calls", "epochs", "optimizer_steps", "rollouts")]
                == [262144, 262144, 256, 1024, 4096, 256]
                and len(learning["updates"]) == len(learning["rollouts"]) == 256
                and learning["initial_hashes"]["policy_state"]
                != learning["final_hashes"]["policy_state"]
                and learning["initial_hashes"]["optimizer_state"]
                != learning["final_hashes"]["optimizer_state"]
                and nonzero_effective > 0,
                "T learned-count/parameter-change evidence fails independent check")
        for index, update in enumerate(learning["updates"]):
            require(update["update_index"] == index
                    and update["num_timesteps"] == (index + 1) * 1024
                    and update["timesteps_delta"] == 1024
                    and update["n_updates_delta"] == 4
                    and update["optimizer_steps"] == 16
                    and update["nonfinite_grad_steps"] == 0
                    and update["missing_grad_params"] == 0
                    and all(math.isfinite(update[key]) for key in (
                        "train/loss", "train/value_loss", "train/policy_gradient_loss",
                        "train/entropy_loss", "train/approx_kl", "train/clip_fraction",
                    )), "T actual PPO update/loss/optimizer evidence differs")
            progress = _json(folder / f"progress_rollout_{index:04d}.json")
            require(progress["rollout"]["rollout_index"] == index
                    and progress["completed_controls"] == (index+1)*1024
                    and progress["gaussian_records"] == progress["completed_controls"]
                    and progress["optimizer_steps_before_pending_update"] == 16*index,
                    "T durable rollout boundary/actual update ordering differs")
    return {"controls": n, "numeric_blocks": len(blocks), "gaussian_blocks": len(gaussian),
            "episode_ids_seen": len(episodes_seen), "nonzero_effective_action_controls": nonzero_effective,
            "Gaussian_clipped_controls": clipped_controls,
            "curriculum_trace": curriculum,
            "compact_interval_nonwheel_counts": native_nonwheel,
            "PPO_actual_counts_checked": bool(full),
            "full_native_contact_force_qualification_available": False,
            "reason_full_force_unavailable": "training uses compact native blocks"}


def _expected_raw(tick: int, speed: float, yaw_case: bool) -> dict:
    ramp = math.ceil(speed / .005 - 1e-12)
    begin, end = 175 + ramp, 175 + ramp + 400
    yaw = 0.0
    if yaw_case and begin <= tick < end:
        hold_tick = tick - begin
        yaw = .3 if hold_tick < 100 or hold_tick >= 300 else -.3
    return {"forward_velocity_mps": 0.0 if tick < 175 or tick >= end else speed,
            "lateral_velocity_mps": 0.0, "yaw_rate_rps": yaw,
            "clearance_m": .455, "jump_requested": False}


def _ground_hit(rows: dict[int, dict], x: float, y: float) -> tuple[float, np.ndarray, int]:
    """Independent scalar vertical slab ray over the actual 92 compiled primitives."""
    height, normal, gid = 0.0, np.asarray((0.0, 0.0, 1.0)), -1
    for geom_id, row in rows.items():
        if row["type"] == "plane":
            require(row["name"] == "floor" and close(row["position_m"], (0, 0, 0)),
                    "actual compiled floor changed")
            gid = geom_id
            break
    require(gid >= 0, "actual compiled floor absent")
    for geom_id, row in rows.items():
        if row["type"] != "box":
            continue
        center = np.asarray(row["position_m"], dtype=np.float64)
        half = np.asarray(row["size_m"], dtype=np.float64)
        rotation = _rotation(row["quaternion_wxyz"])
        dx, dy = x - center[0], y - center[1]
        lower, upper, face = -math.inf, math.inf, None
        for axis in range(3):
            slope = float(rotation[2, axis])
            offset = float(rotation[0, axis] * dx + rotation[1, axis] * dy)
            if abs(slope) <= 1e-14:
                if abs(offset) > half[axis]:
                    face = None
                    break
                continue
            a, b = ((-half[axis] - offset) / slope + center[2],
                    (half[axis] - offset) / slope + center[2])
            lower = max(lower, min(a, b))
            if max(a, b) < upper:
                upper = max(a, b)
                face = math.copysign(1.0, slope) * rotation[:, axis]
        if face is not None and lower <= upper and upper > height:
            require(face[2] > 0, "compiled box vertical hit lacks upward normal")
            height, normal, gid = upper, face, geom_id
    return height, normal, gid


def _terrain_reference(normal: np.ndarray, yaw: float) -> tuple[float, float]:
    # Same geometric meaning as Rz(yaw) Ry(pitch) Rx(roll), independently evaluated.
    slope_x = -normal[0] / normal[2]
    slope_y = -normal[1] / normal[2]
    forward = slope_x * math.cos(yaw) + slope_y * math.sin(yaw)
    lateral = -slope_x * math.sin(yaw) + slope_y * math.cos(yaw)
    return math.atan2(lateral, math.sqrt(1 + forward * forward)), -math.atan(forward)


def _yaw_from_qpos(qpos: np.ndarray) -> float:
    w, x, y, z = map(float, qpos[3:7])
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def _schedule(folder: Path, case_id: str, actor: str, terrain: str,
              speed: float, seed: int) -> dict:
    schedule = _json(folder / "schedule.json")
    require((schedule["case_id"], schedule["actor"], schedule["seed"],
             schedule["terrain"]) == (case_id, actor, seed, terrain)
            and len(schedule["raw_commands"]) == 1600,
            "heldout saved preregistered schedule identity differs")
    require(all(row == _expected_raw(tick, speed, case_id == "flat_1p2_yaw")
                for tick, row in enumerate(schedule["raw_commands"])),
            "heldout raw 175/ramp/400/yaw schedule differs")
    encoded = [[row[field] for field in ("forward_velocity_mps", "lateral_velocity_mps",
                                         "yaw_rate_rps", "clearance_m", "jump_requested")]
               for row in schedule["raw_commands"]]
    import hashlib

    digest = hashlib.sha256(json.dumps(encoded, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()
    require(digest == schedule["raw_command_sha256"],
            "heldout raw schedule hash differs from saved commands")
    return schedule


def _verify_case(run: Path, spec: tuple, actor: str, binding: dict,
                 expected_geometry: dict) -> tuple[dict, dict]:
    case_id, terrain, speed, seed = spec
    folder = run / "heldout" / f"{case_id}_{actor}"
    receipt = _json(folder / "case_receipt.json")
    schedule = _schedule(folder, case_id, actor, terrain, speed, seed)
    reset = _json(folder / "reset_receipt.json")
    before = _json(folder / "boundary_before.json")
    rows = _read_control_rows(folder, receipt["controller_record_blocks"])
    guard = receipt["native_segment"]
    native = _native_rows(folder, guard)
    states = _load_npz(folder / "states.npz")
    initial = _load_npz(folder / "initial_state.npz")
    require(_json(folder / "geometry_manifest.json") == expected_geometry,
            "heldout actual compiled 92 geom manifest changed between cases")
    geometry = geom_map(folder, binding)
    n = len(rows)
    require(receipt["schema"] == "d1-course-rl16-heldout-episode-record-v1"
            and (receipt["case_id"], receipt["actor"], receipt["seed"],
                 receipt["terrain"]) == (case_id, actor, seed, terrain)
            and receipt["raw_command_sha256"] == schedule["raw_command_sha256"]
            and receipt["completed_controls"] == n <= 1600
            and receipt["control_cap"] == 1600
            and receipt["failure"] is None and receipt["record_valid"] is True
            and receipt["termination_recorded"] is True
            and guard["mode"] == "heldout"
            and guard["native_attempted"] == guard["native_returned"] == len(native) == 5*n
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
    _boundary_delta(before, receipt["boundary_after"], n)
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
        "target_speed_mps": speed, "control_cap": 1600,
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
        "native_nonwheel_ground_candidate_count": [],
        "positive_wheel_load_native_counts_by_family": {},
        "warning_count": 0, "geometry_invalid_count": 0,
        "map_escape_count": 0, "fall_count": 0,
    }
    family_loads: Counter[str] = Counter()
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
                and close(record["reward"], sum(info["reward_terms"].values())),
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
            canonical["native_nonwheel_ground_candidate_count"].append(
                interval_candidates
            )
            canonical["map_escape_count"] += int(
                abs(float(native_qpos[0])) > 11 or abs(float(native_qpos[1])) > 6
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
                          metrics["relative_pitch_rad"], atol=1e-6),
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
    canonical["fall_count"] = int(receipt["stop_reason"] == "fall_or_low_clearance")
    require(receipt["terminated"] == rows[-1]["terminated"]
            and receipt["truncated"] == rows[-1]["truncated"]
            and receipt["stop_reason"] == rows[-1]["info"]["terminal_reason"],
            "heldout genuine terminal state differs from final recorded control")
    summary = {
        "case_id": case_id, "actor": actor, "seed": seed, "terrain": terrain,
        "completed_controls": n, "native_returned": len(native),
        "predict_calls": receipt["policy_predictions"],
        "full_1600_controls": n == 1600,
        "legitimate_task_failure": bool(receipt["terminated"] and n < 1600),
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


def _checkpoint(run: Path, learning: dict) -> dict:
    manifest = _json(run / "final_checkpoint_manifest.json")
    folder = run / "final_checkpoint"
    require(manifest["folder"] == str(folder)
            and manifest["trusted_fields"]["observation_size"] == 99
            and manifest["trusted_fields"]["action_size"] == 16
            and manifest["trusted_fields"]["run_id"] == "rl16_training_08_01",
            "T final checkpoint identity/schema differs")
    for name, expected in manifest["files"].items():
        _file_matches(folder / name, expected)
    metadata = folder / "final_metadata.json"
    require(identity(metadata)["sha256"] == manifest["metadata_sha256"],
            "T final model metadata hash differs")
    document = _json(metadata)
    require(document["files"] == manifest["files"]
            and document["training_receipt"] == learning
            and document["observation_size"] == 99
            and document["action_size"] == 16,
            "T final checkpoint metadata/training receipt differs")
    reload = _json(run / "final_checkpoint_reload.json")
    first_reload = manifest["reload_verification"]
    require(first_reload["reference_policy_byte_exact"] is True
            and reload["reference_policy_byte_exact"] is None
            and all(reload[key] == value for key, value in first_reload.items()
                    if key != "reference_policy_byte_exact")
            and reload["matches_training_final_policy_state"] is True,
            "T strict final model reload/probe receipt differs")
    probe_origin = _json(run / "training/final_probe_origin.json")
    source = Path(probe_origin["source_file"])
    require(source == run / "training/training_numeric_blocks/controls_0000.npz"
            and identity(source)["sha256"] == probe_origin["source_file_sha256"]
            and probe_origin["rows"] == [0, 32]
            and probe_origin["additional_physical_probe_calls"] == 0,
            "T fixed first32 actual-observation probe origin differs")
    blocks = _load_npz(source)
    probe_bytes = (folder / "final_probe_observations_f32.bin").read_bytes()
    require(probe_bytes == np.ascontiguousarray(
                blocks["input_observation99"][:32], dtype=np.float32,
            ).tobytes(), "T strict probe inputs differ from first32 trained observations")
    return {"metadata_sha256": manifest["metadata_sha256"],
            "files": manifest["files"], "strict_reload_receipt_verified": True,
            "first32_actual_observation_probe_verified": True,
            "model_reloaded_by_offline_verifier": False}


def _pair_exact(run: Path, specs: tuple, rows: list[dict]) -> None:
    require(len(rows) == 12, "T heldout has fewer than six full pairs")
    for pair_index, spec in enumerate(specs):
        zero, policy = rows[2*pair_index:2*pair_index+2]
        require((zero["case_id"], policy["case_id"]) == (spec[0], spec[0])
                and (zero["actor"], policy["actor"]) == ("zero", "final_policy")
                and zero["seed"] == policy["seed"] == spec[3]
                and zero["actual_model_address"] == policy["actual_model_address"]
                and zero["actual_data_address"] == policy["actual_data_address"],
                "T preregistered zero/final matched pair order/seed/model differs")
        initial = []
        for actor in ("zero", "final_policy"):
            folder = run / "heldout" / f"{spec[0]}_{actor}"
            reset = _json(folder / "reset_receipt.json")
            require(reset["exact_initial_pair"] is (actor == "final_policy")
                    and reset["paired_with"] == ("zero" if actor == "final_policy" else None),
                    "T same-initial-pair reset receipt differs")
            initial.append(_load_npz(folder / "initial_state.npz"))
        require(all(equal(initial[0][field], initial[1][field])
                    for field in PAIR_FIELDS),
                "T zero/final initial five arrays differ bitwise")


def _score_adapter(canonical_cases: list[dict]) -> dict:
    """Single pure-score seam, reached only after all twelve raw cases pass."""
    path = W / "course_impl08/rl16_heldout_score_08.py"
    spec = importlib.util.spec_from_file_location("rl16_heldout_pure_score_08", path)
    require(spec is not None and spec.loader is not None,
            "frozen pure scorer source cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(callable(module.score_case) and callable(module.score_pairs),
            "frozen pure scorer public interface differs")
    case_scores = [module.score_case(case) for case in canonical_cases]
    aggregate = module.score_pairs(case_scores)
    return {"source": str(path), "source_identity": identity(path),
            "cases": case_scores, "aggregate": aggregate}


def _closed_manifest(run: Path) -> dict:
    files = {}
    for path in sorted(run.rglob("*")):
        require(not path.is_symlink(), "T saved run contains a symlink")
        if path.is_file():
            files[path.relative_to(run).as_posix()] = identity(path)
    return {"schema": "d1-course-t-closed-file-manifest-v1", "run": str(run),
            "file_count": len(files),
            "total_bytes": sum(row["bytes"] for row in files.values()),
            "files": files}


def verify(run: Path) -> tuple[dict, dict]:
    require(run.is_absolute() and run.is_dir(), "T --run must be an absolute saved directory")
    session, source = _source(run)
    worker = _json(run / "worker_receipt.json")
    launcher = _json(run / "launcher_receipt.json")
    initial = _json(run / "runtime_initial.json")
    construction = _json(run / "construction_receipt.json")
    require(launcher["exit_code"] == 0
            and launcher["failure"] is None
            and launcher["source_hash_mismatches"] == []
            and launcher["worker_exited"] is True
            and launcher["owned_worker_cleanup"]["no_orphans"] is True
            and launcher["fully_reserved_budget_closed"] is True
            and launcher["retry_permitted"] is False,
            "T external launcher exit, source closure, worker cleanup or budget failed")
    require(worker["schema"] == launcher["schema"] == SCHEMA
            and worker["retry_permitted"] is False
            and worker["warnings"] == []
            and initial["proof"]["passed"] is True
            and initial["proof"]["dso_path"] == session["library"]
            and initial["proof"]["dso_sha256"]
            == session["source_hashes"][session["library"]]["sha256"]
            and len(initial["proof"]["jump_slots"]) == 4
            and all(slot["passed"] is True for slot in initial["proof"]["jump_slots"])
            and construction["C_state"]["construction_attempts"]
            == construction["C_state"]["construction_returns"] == 2
            and construction["nominal_cache"]["misses"] == 1,
            "T worker/launcher/GOT/actual cold2 provenance differs")
    binding = check_binding(construction["actual_geometry_binding"])
    require(initial["C_state"]["control_returns"] == 0,
            "T actual C counter not cold before training")
    cfinal, ledger = worker["C_final"], worker["python"]
    require(cfinal is not None and ledger is not None
            and cfinal["construction_attempts"] == cfinal["construction_returns"] == 2
            and cfinal["control_attempts"] <= 1406720
            and ledger["control_attempted"] <= 281344
            and ledger["native_attempted"] <= 1406720
            and cfinal["violations"] == ledger["forbidden_entries"] == 0
            and cfinal["phase"] == cfinal["target_model"] == cfinal["target_data"] == 0,
            "T final actual C/Python budget or phase differs")
    if cfinal["ccd_attempts"]:
        require(cfinal["ccd_attempts"] == cfinal["ccd_returns"]
                and cfinal["native_ccd_caller_verified"] is True,
                "T nonzero CCD lacks native caller proof")
    else:
        require(cfinal["first_ccd_caller"] == 0,
                "T no-CCD run claims a caller")
    require(cfinal["native_construction_caller_verified"] is True
            and worker["control_step_caller_verified_against_closed_E_callsite"] is True,
            "T true C construction/control callsite proof absent")
    full = worker["execution_complete"] is True and worker["failure"] is None
    training = _training(run, full)
    cases: list[dict] = []
    canonical_cases: list[dict] = []
    checkpoint = None
    if full:
        learning = _json(run / "training/learning_receipt.json")
        checkpoint = _checkpoint(run, learning)
        mode = _json(run / "heldout_mode_transition.json")
        require(mode["from"] == "train" and mode["to"] == "eval"
                and mode["numerical_controller_unchanged"] is True
                and mode["one_actual_model_data_pair"] == [
                    construction["course_plant_model_address"],
                    construction["course_plant_data_address"],
                ], "T heldout used another model/data or numerical controller")
        for spec in CASES:
            for actor in ("zero", "final_policy"):
                result, canonical = _verify_case(
                    run, spec, actor, binding, construction["compiled_geometry"],
                )
                require([result[key] for key in ("actual_model_address",
                                                 "actual_data_address")] ==
                        mode["one_actual_model_data_pair"],
                        "T heldout case switched actual compiled model/data")
                cases.append(result)
                canonical_cases.append(canonical)
        _pair_exact(run, CASES, cases)
        require(worker["heldout_cases"] == [
            {key: _json(run / "heldout" / f"{row['case_id']}_{row['actor']}"
                        / "case_receipt.json")[key]
             for key in ("case_id", "actor", "seed", "terrain", "completed_controls",
                         "policy_predictions", "record_valid", "terminated", "truncated",
                         "stop_reason")}
            for row in cases
        ], "T worker heldout summary differs from 12 actual case receipts")
    completed = training["controls"] + sum(row["completed_controls"] for row in cases)
    counts = (cfinal["control_attempts"] == cfinal["control_returns"] == 5*completed
              and ledger["control_attempted"] == ledger["control_completed"] == completed
              and ledger["native_attempted"] == ledger["native_returned"] == 5*completed
              and worker["native_guard"]["native_returned"] == 5*completed
              and worker["native_guard"]["native_checked"] == 5*completed)
    require(counts, "T final C/Python/native 5T actual totals differ")
    numeric_scores = _score_adapter(canonical_cases) if len(cases) == 12 else None
    combined_physical_and_numeric = full and len(cases) == 12 and counts
    policy_qualified = bool(combined_physical_and_numeric and numeric_scores
                            and numeric_scores["aggregate"]["all_six_policy_task_qualified"])
    contribution = bool(combined_physical_and_numeric and numeric_scores
                        and numeric_scores["aggregate"]["RL_contribution_passed"])
    result = {
        "schema": "d1-course-t-independent-saved-readback-v1",
        "run": str(run), "source_closure": source,
        "offline_reader_source": {"path": str(Path(__file__).resolve()),
                                  **identity(Path(__file__).resolve())},
        "actual_go_sha256": session["go_sha256"],
        "execution_complete": full and len(cases) == 12,
        "training": training, "checkpoint": checkpoint,
        "heldout_cases": cases,
        "all_12_heldout_full_native_contact_controller_evidence": len(cases) == 12,
        "all_6_initial_pairs_bitwise_exact": len(cases) == 12,
        "numeric_scores": numeric_scores,
        "all_six_policy_task_qualified_after_independent_raw_readback": policy_qualified,
        "RL_contribution_passed_after_independent_raw_readback": contribution,
        "numeric_scores_pending_reason": None if numeric_scores else (
            "T is not a complete 12-case raw-verified execution"
        ),
        "normal_native_returned": ledger["native_returned"],
        "compiler_native_returned": cfinal["construction_returns"],
        "engine_or_policy_imported": False,
        "physics_replayed_or_rescored": False,
    }
    return result, _closed_manifest(run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    require(args.run.is_absolute(), "T --run must be absolute")
    result, manifest = verify(args.run.resolve(strict=True))
    for target in (args.output, args.manifest_output):
        if target is not None:
            require(target.is_absolute() and not target.exists(),
                    "T offline output must be a new exclusive absolute path")
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
