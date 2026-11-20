"""Paired first-return wheel-track measurements on archival poses and new zero-action runs."""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from scripts import terrain_preview84_mount_audit as archived
from scripts import terrain_preview84_trials as trial
from wheel_legged_control.d1 import terrain_preview as preview
from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.training_terrain import TrainingTerrainConfig

ANGLES = (74, 82)
ORIGINS_M = preview.ORIGINS + np.array([0.16, 0.0, 0.0])


def write_row(stream, row):
    stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()


class RangeMount:
    def __init__(self, near_deg):
        angles = preview.DEPRESSION_RAD.copy()
        angles[0] = math.radians(near_deg)
        self.directions = np.column_stack((np.cos(angles), np.zeros(4), -np.sin(angles)))

    def acquire(self, plant):
        data, base = plant.measurement_data, plant.base_body_id
        rotation = data.xmat[base].reshape(3, 3)
        origins = ORIGINS_M @ rotation.T + data.xpos[base]
        directions = self.directions @ rotation.T
        raw = np.empty((2, 4))
        hit_ids = np.full((2, 4), -1, dtype=np.int32)
        for lane, origin in enumerate(origins):
            for beam, direction in enumerate(directions):
                raw[lane, beam] = mujoco.mj_ray(
                    plant.model, data, origin, direction, None, True, -1,
                    hit_ids[lane, beam:beam + 1],
                )
        return raw, hit_ids

    def frame(self, sequence, captured, travelled, raw, pitch, roll):
        distances = np.where(
            (raw >= archived.MIN_RANGE_M) & (raw <= archived.MAX_RANGE_M), raw, np.nan,
        )
        xyz = ORIGINS_M[:, None, :] + distances[..., None] * self.directions
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        z_roll = sr * xyz[..., 1] + cr * xyz[..., 2]
        x = cp * xyz[..., 0] + sp * z_roll
        z = -sp * xyz[..., 0] + cp * z_roll
        return preview.RangeFrame(
            sequence, captured, captured + preview.LATENCY_S, travelled, distances, x, z,
        )


def ground_only(frame, ground_hits):
    """Build a scoring-only view. The controller never receives simulator hit identities."""
    return replace(
        frame,
        x_m=np.where(ground_hits, frame.x_m, np.nan),
        z_m=np.where(ground_hits, frame.z_m, np.nan),
    )


def capture_truth(mount, raw, ids, base_xyz, rotation, heading, floor_id):
    """Score-only first-hit positions and their projection on the capture heading."""
    points_body = ORIGINS_M[:, None, :] + raw[..., None] * mount.directions
    relative_world = points_body @ rotation.T
    world = relative_world + base_xyz
    in_range_floor = (
        (ids == floor_id) & (raw >= archived.MIN_RANGE_M) & (raw <= archived.MAX_RANGE_M)
    )
    forward = relative_world @ heading
    x = np.where(in_range_floor, forward, np.nan)
    z = np.where(in_range_floor, world[..., 2], np.nan)
    audit = {
        "first_hit_world_xyz_m": [
            [point.tolist() if distance >= 0 else None for point, distance in zip(track, distances)]
            for track, distances in zip(world, raw)
        ],
        "in_range_floor_forward_from_base_m": archived.grid(x),
        "in_range_floor_world_z_m": archived.grid(z),
    }
    return (x, z), audit


def matched_chords(frame, query, floor_coordinates):
    """Look up the true floor chord between the exact pair selected in each estimated track."""
    starts, slopes = [None, None], [None, None]
    if frame is None:
        return starts, slopes
    floor_x, floor_z = floor_coordinates
    for lane, (x, z) in enumerate(zip(frame.x_m, frame.z_m)):
        inside = (x[:-1] <= query) & (query < x[1:]) & (np.diff(x) >= archived.MIN_SEGMENT_M)
        candidates = np.flatnonzero(inside & np.isfinite(z[:-1]) & np.isfinite(z[1:]))
        if len(candidates) != 1:
            continue
        i = int(candidates[0])
        starts[lane] = i
        dx = floor_x[lane, i + 1] - floor_x[lane, i]
        dz = floor_z[lane, i + 1] - floor_z[lane, i]
        if math.isfinite(dz) and dx > 0.0:
            slopes[lane] = math.atan2(dz, dx)
    return starts, slopes


def corroborated_grade(measured, chords):
    """Conservative signed grade shared by the online reading and both true floor chords."""
    if measured is None or any(v is None for v in chords):
        return None
    grades = [measured, *chords]
    if min(grades) > 0:
        return min(grades)
    if max(grades) < 0:
        return max(grades)
    return 0.0


def archive_case(directory, plant, mounts, stream):
    archived.set_terrain(plant, directory / "terrain.npz")
    with np.load(directory / "states.npz", allow_pickle=False) as states:
        times, qpos, qvel = (states[key] for key in ("time_s", "qpos", "qvel"))
    if len(times) != len(qpos) or len(times) != len(qvel) or np.any(np.diff(times) <= 0):
        raise ValueError(f"invalid archived clock in {directory}")
    counters = {angle: Counter() for angle in ANGLES}
    seen = set()
    with (directory / "frames.jsonl").open() as frames:
        for line in frames:
            packet = json.loads(line)
            captured, sequence = packet["captured_s"], packet["sequence"]
            index = int(np.searchsorted(times, captured))
            if index >= len(times) or abs(times[index] - captured) > 1e-10 or sequence in seen:
                raise ValueError(f"unmatched archive packet in {directory}")
            seen.add(sequence)
            data = plant.measurement_data
            data.qpos[:], data.qvel[:], data.time = qpos[index], qvel[index], captured
            mujoco.mj_forward(plant.model, data)
            rotation = data.xmat[plant.base_body_id].reshape(3, 3)
            roll = math.atan2(rotation[2, 1], rotation[2, 2])
            pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[2, 1], rotation[2, 2]))
            query = packet["query_x_m"]
            row = {
                "case": directory.name, "sequence": sequence, "pose_index": index,
                "captured_s": captured, "query_x_m": query, "first_used_s": packet["control_s"],
                "mounts": {},
            }
            for angle, mount in mounts.items():
                raw, ids = mount.acquire(plant)
                frame = mount.frame(sequence, captured, 0.0, raw, pitch, roll)
                ground = ground_only(frame, ids == plant.floor_geom_id)
                observed = archived.coverage(frame.x_m, frame.z_m, query)
                floor = archived.coverage(ground.x_m, ground.z_m, query)
                counter = counters[angle]
                counter["frames"] += 1
                counter["both_tracks_covered"] += int(observed["covered_both_tracks"])
                counter["valid"] += int(observed["passes_grade_agreement"])
                counter["ground_covered"] += int(floor["covered_both_tracks"])
                counter["ground_valid"] += int(floor["passes_grade_agreement"])
                row["mounts"][str(angle)] = {
                    "raw_range_m": raw.tolist(), "first_hit_geom": ids.tolist(),
                    "x_m": archived.grid(frame.x_m), "z_m": archived.grid(frame.z_m),
                    "coverage": observed, "ground_coverage": floor,
                }
            write_row(stream, row)
    return {str(k): dict(v) for k, v in counters.items()}


def run_offline(trials_dir: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    trial.write_json(output / "started.json", {"status_until_summary": "PARTIAL"})
    cases = []
    try:
        _, groups = archived.check_groups(trials_dir)
        plant = D1Plant(training_terrain=TrainingTerrainConfig(), sampling_mode="synchronized")
        mounts = {angle: RangeMount(angle) for angle in ANGLES}
        pooled = {str(angle): Counter() for angle in ANGLES}
        with (output / "archive_frames.jsonl").open("x") as stream:
            for name, hashes in groups.items():
                counts = archive_case(trials_dir / "cases" / (name + "_off"), plant, mounts, stream)
                for angle, values in counts.items():
                    pooled[angle].update(values)
                cases.append({"case": name, "counts": counts, "source_sha256": hashes})
        reference = pooled["74"]
        matches = (reference["frames"], reference["ground_covered"], reference["ground_valid"]) == (
            308, 82, 80,
        )
        result = {
            "status": "COMPLETE", "attitude_source": "archived true roll and pitch at capture",
            "passed_meaning": "74-degree reference reproduces the archived 308/82/80 geometry counts",
            "passed": matches, "cases": cases, "pool": pooled,
            "task_steps": 0, "native_steps": 0, "training_updates": 0,
        }
        trial.write_json(output / "summary.json", result)
        return result
    except BaseException as exc:
        trial.write_json(output / "partial.json", {"status": "PARTIAL", "cases": cases,
                                                   "error": repr(exc)})
        raise


def reading_row(reading, ground_reading, floor_coordinates):
    starts, chords = matched_chords(reading.frame, reading.query_x_m, floor_coordinates)
    corroborated = corroborated_grade(reading.slope_rad, chords)
    return {
        "sequence": reading.sequence, "captured_s": reading.captured_s,
        "available_s": reading.available_s, "query_x_m": reading.query_x_m,
        "reason": reading.reason, "slope_rad": reading.slope_rad,
        "left_rad": reading.left_slope_rad, "right_rad": reading.right_slope_rad,
        "ground_reason": ground_reading.reason, "ground_slope_rad": ground_reading.slope_rad,
        "online_pair_start_beam_by_track": starts, "true_floor_chord_rad_by_track": chords,
        "corroborated_slope_rad": corroborated,
    }


class PairedRangeClock:
    """Both shadow sensors share exposures and receipt times on one physical trajectory."""

    def __init__(self):
        self.mounts = {str(angle): RangeMount(angle) for angle in ANGLES}
        self.pending = deque()
        self.current = {}
        self.next_sequence = 0

    def observe(self, ep, command, captures):
        state, plant = ep.state, ep.plant
        now = float(state.control_time_s)
        if (abs(plant.data.time - now) > 1e-9
                or abs(state.measurement_time_s - max(0.0, now - 2 * trial.DT)) > 1e-9):
            raise ValueError("sensor clock or estimator delay mismatch")
        roll, pitch = map(float, state.base_rpy[:2])
        travelled = float(state.base_position[0] - trial.INITIAL_POSITION[0])
        front = float(
            (state.foot_position[:2].mean(axis=0) - state.base_position) @ state.base_rotation[:, 0]
        )
        if now + 1e-12 >= self.next_sequence / preview.RATE_HZ:
            seq = self.next_sequence
            frames = {}
            rotation = plant.measurement_data.xmat[plant.base_body_id].reshape(3, 3)
            base_xyz = plant.measurement_data.xpos[plant.base_body_id]
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            heading = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            capture = {
                "sequence": seq, "captured_s": now, "available_s": now + preview.LATENCY_S,
                "estimator_s": state.measurement_time_s, "estimated_roll_rad": roll,
                "estimated_pitch_rad": pitch, "estimated_travelled_m": travelled, "mounts": {},
                "score_only_truth": {
                    "base_xyz_world_m": base_xyz.tolist(), "body_rotation": rotation.tolist(),
                    "forward_heading_world": heading.tolist(), "mounts": {},
                },
            }
            for angle, mount in self.mounts.items():
                raw, ids = mount.acquire(plant)
                frame = mount.frame(seq, now, travelled, raw, pitch, roll)
                coordinates, truth = capture_truth(
                    mount, raw, ids, base_xyz, rotation, heading, plant.floor_geom_id,
                )
                frames[angle] = frame, ground_only(frame, ids == plant.floor_geom_id), coordinates
                capture["score_only_truth"]["mounts"][angle] = truth
                capture["mounts"][angle] = {
                    "raw_range_m": raw.tolist(), "first_hit_geom": ids.tolist(),
                    "floor_geom_id": plant.floor_geom_id,
                    "local_x_m": archived.grid(frame.x_m), "local_z_m": archived.grid(frame.z_m),
                }
            write_row(captures, capture)
            self.pending.append((now + preview.LATENCY_S, frames))
            self.next_sequence = math.floor((now + 1e-12) * preview.RATE_HZ) + 1
        while self.pending and self.pending[0][0] <= now + 1e-12:
            _, self.current = self.pending.popleft()
        observations = {}
        for angle in self.mounts:
            frame, floor_frame, coordinates = self.current.get(angle, (None, None, None))
            args = now, travelled, front, command.forward_velocity_mps
            measured = preview.query_frame(frame, *args)
            ground = preview.query_frame(floor_frame, *args)
            observations[angle] = reading_row(measured, ground, coordinates)
        return observations


def summarize_mount(rows, angle, direction, contact, *, mode="measured"):
    slope_field, reason_field = {
        "measured": ("slope_rad", "reason"),
        "floor_returns": ("ground_slope_rad", "ground_reason"),
        "matched_chords": ("corroborated_slope_rad", "reason"),
    }[mode]
    delivered, valid_packets, first_read_valid = set(), set(), set()
    wrong_packets, early_wrong_packets, correct_early_packets = set(), set(), set()
    first_valid = first_correct = first_confirmed = first_streak_start = qualifying_seq = None
    current_seq, previous_step, streak_start = None, None, None
    valid_ticks = wrong_ticks = early_wrong_ticks = longest = streak = 0
    for row in rows:
        now, step = row["control_s"], row["step"]
        read = row["mounts"][angle]
        seq, grade = read["sequence"], read[slope_field]
        fresh = seq is not None and (
            read["captured_s"] <= read["available_s"] <= now + 1e-12
            and now - read["captured_s"] <= preview.MAX_AGE_S + 1e-12
        )
        valid = fresh and read[reason_field] == "valid" and grade is not None
        if seq is not None:
            if seq not in delivered and valid:
                first_read_valid.add(seq)
            delivered.add(seq)
        if valid:
            valid_packets.add(seq)
            valid_ticks += 1
            if first_valid is None:
                first_valid = now
        early = contact is not None and now <= contact - trial.DT + 1e-9
        if valid and direction * grade <= -trial.VISIBLE_SLOPE_RAD:
            wrong_packets.add(seq)
            wrong_ticks += 1
            if early:
                early_wrong_packets.add(seq)
                early_wrong_ticks += 1
        correct = valid and direction * grade >= trial.VISIBLE_SLOPE_RAD
        if correct and first_correct is None:
            first_correct = now
        if early and correct:
            correct_early_packets.add(seq)
            if seq != current_seq or previous_step != step - 1 or streak == 0:
                streak, streak_start = 0, now
            streak += 1
            longest = max(longest, streak)
            if streak == trial.CONFIRMATION_TICKS and first_confirmed is None:
                first_confirmed, first_streak_start, qualifying_seq = now, streak_start, seq
        else:
            streak = 0
        current_seq, previous_step = seq, step
    return {
        "delivered_packets": len(delivered),
        "both_tracks_valid_at_first_receipt": len(first_read_valid),
        "both_tracks_valid_packets_any_tick": len(valid_packets),
        "both_tracks_valid_packet_fraction": len(valid_packets) / len(delivered) if delivered else None,
        "valid_ticks": valid_ticks, "wrong_direction_packets": len(wrong_packets),
        "wrong_direction_ticks": wrong_ticks, "precontact_wrong_packets": len(early_wrong_packets),
        "precontact_wrong_ticks": early_wrong_ticks,
        "precontact_correct_packets": len(correct_early_packets),
        "first_valid_s": first_valid, "first_correct_s": first_correct,
        "first_qualifying_packet": qualifying_seq, "first_streak_start_s": first_streak_start,
        "first_confirmation_s": first_confirmed, "longest_same_packet_precontact_ticks": longest,
        "passed": first_confirmed is not None,
    }


def first_window_ground_evidence(rows, angle, direction, measured):
    sequence = measured["first_qualifying_packet"]
    if sequence is None:
        return None
    begin, end = measured["first_streak_start_s"], measured["first_confirmation_s"]
    samples = []
    for row in rows:
        if begin <= row["control_s"] <= end:
            read = row["mounts"][angle]
            if read["sequence"] != sequence:
                raise ValueError("registered first window crosses a packet boundary")
            grade = read["corroborated_slope_rad"]
            samples.append({
                "control_s": row["control_s"], "query_x_m": read["query_x_m"],
                "captured_s": read["captured_s"], "measured_slope_rad": read["slope_rad"],
                "online_pair_start_beam_by_track": read["online_pair_start_beam_by_track"],
                "true_floor_chord_rad_by_track": read["true_floor_chord_rad_by_track"],
                "correct_on_both_floor_chords": grade is not None and direction * grade >= trial.VISIBLE_SLOPE_RAD,
            })
    if len(samples) != trial.CONFIRMATION_TICKS:
        raise ValueError("registered first window is incomplete")
    return {"sequence": sequence, "ticks": samples,
            "all_three_confirmed": all(s["correct_on_both_floor_chords"] for s in samples)}


def run_new_case(output, terrain, seed):
    from scripts.terrain_preview85_common import baseline_command, new_episode, physical_sample

    directory = output / f"{terrain}_{seed}"
    directory.mkdir()
    trial.write_json(directory / "started.json",
                     {"terrain": terrain, "seed": seed, "status_until_metrics": "PARTIAL"})
    ep = new_episode(terrain, seed)
    pair = PairedRangeClock()
    rows = []
    termination = "time_limit"
    with (directory / "captures.jsonl").open("x") as captures, (directory / "ticks.jsonl").open("x") as ticks:
        for step in range(1, trial.STEPS + 1):
            now = float(ep.state.control_time_s)
            command = baseline_command(ep)
            measurements = pair.observe(ep, command, captures)
            torque = ep.controller.compute(command, ep.state)
            ep.plant.step(torque)
            physical = physical_sample(ep, torque)
            row = {
                **physical, "control_s": now, "step": step, "mounts": measurements,
                "requested_torque_nm": torque.tolist(),
                "applied_torque_nm": ep.plant.data.ctrl[ep.plant.actuator_ids].tolist(),
            }
            write_row(ticks, row)
            rows.append(row)
            if physical["termination_reason"]:
                termination = physical["termination_reason"]
                break
            ep.state = ep.source.read()
    contact = next((r["time_s"] for r in rows
                    if r["front_contact_x_m"] is not None and r["front_contact_x_m"] >= ep.toe), None)
    crossed = next((r["time_s"] for r in rows if contact is not None and r["time_s"] >= contact
                    and r["front_track_x_m"] >= ep.toe + trial.SCORE_DISTANCE_M), None)
    direction = 1 if terrain == "up" else -1
    readings = {str(a): summarize_mount(rows, str(a), direction, contact) for a in ANGLES}
    ground = {
        str(a): summarize_mount(rows, str(a), direction, contact, mode="floor_returns") for a in ANGLES
    }
    chords = {
        str(a): summarize_mount(rows, str(a), direction, contact, mode="matched_chords") for a in ANGLES
    }
    first_windows = {
        str(a): first_window_ground_evidence(rows, str(a), direction, readings[str(a)]) for a in ANGLES
    }
    window = [r["pitch_error_rad"] for r in rows
              if crossed is not None and contact <= r["time_s"] <= crossed]
    metrics = {
        "terrain": terrain, "seed": seed, "task_steps": len(rows), "termination_reason": termination,
        "full_horizon": termination == "time_limit" and len(rows) == trial.STEPS,
        "toe_x_m": ep.toe, "first_front_contact_s": contact, "crossed_window_s": crossed,
        "zero_window_complete": crossed is not None,
        "zero_window_pitch_rmse_deg": math.degrees(math.sqrt(np.mean(np.square(window)))) if window else None,
        "torque_bound_violations": sum(r["torque_bound_violations"] for r in rows),
        "joint_torque_limits_nm": ep.plant.actuator_torque_limit_nm.tolist(),
        "mounts": readings, "true_ground_only_diagnostic": ground,
        "matched_ground_chord_diagnostic": chords, "first_registered_window_ground_check": first_windows,
        "passed": readings["82"]["passed"],
    }
    trial.write_json(directory / "metrics.json", metrics)
    return metrics


def run_sensor_cases(output: Path, seeds: tuple[int, ...]):
    if len(seeds) != 3 or len(set(seeds)) != 3 or set(seeds).intersection(trial.SEEDS):
        raise ValueError("three distinct new noise seeds are required")
    output.mkdir(parents=True, exist_ok=False)
    trial.write_json(output / "started.json", {"status_until_summary": "PARTIAL", "seeds": list(seeds)})
    cases = []
    active = None
    try:
        for terrain in ("up", "down"):
            for seed in seeds:
                active = f"{terrain}_{seed}"
                cases.append(run_new_case(output, terrain, seed))
        result = {
            "status": "COMPLETE", "cases": cases, "passed": all(c["passed"] for c in cases),
            "task_steps": sum(c["task_steps"] for c in cases), "training_updates": 0,
            "correct_precontact_cases_by_angle": {
                str(a): sum(c["mounts"][str(a)]["passed"] for c in cases) for a in ANGLES
            },
            "true_chord_correct_precontact_cases_by_angle": {
                str(a): sum(c["matched_ground_chord_diagnostic"][str(a)]["passed"] for c in cases)
                for a in ANGLES
            },
            "first_registered_windows_corroborated_by_angle": {
                str(a): sum(bool(c["first_registered_window_ground_check"][str(a)]
                                 and c["first_registered_window_ground_check"][str(a)]["all_three_confirmed"])
                            for c in cases) for a in ANGLES
            },
            "zero_window_complete_cases": sum(c["zero_window_complete"] for c in cases),
            "falls": sum(c["termination_reason"] == "fallen" for c in cases),
        }
        trial.write_json(output / "summary.json", result)
        return result
    except BaseException as exc:
        trial.write_json(output / "partial.json", {"status": "PARTIAL", "active_case": active,
                                                   "completed_cases": cases, "error": repr(exc)})
        raise
