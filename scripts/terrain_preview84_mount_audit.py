"""Trace two fixed rangefinder mounts on the saved 84 trajectories without stepping."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.terrain_preview import (
    DIRECTIONS,
    MAX_AGE_S,
    MAX_GRADE_RAD,
    ORIGINS,
    TRACK_AGREEMENT_RAD,
)
from wheel_legged_control.d1.training_terrain import TrainingTerrainConfig

ARMS = ("off", "on", "inverted")
SEEDS = (8417, 8429, 8443)
TERRAINS = ("flat", "up", "down")
GROUP_FILES = ("terrain.npz", "states.npz", "poses.raw", "frames.jsonl", "telemetry.csv")
MOUNTS = {"registered": 0.0, "forward_0p16m": 0.16}
MIN_SEGMENT_M = 0.035
MIN_RANGE_M = 0.14
MAX_RANGE_M = 1.45


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, allow_nan=False, indent=2, sort_keys=True)
        f.write("\n")


def check_groups(trials):
    summary = read_json(trials / "summary.json")
    expected = {(t, s, a) for t in TERRAINS for s in SEEDS for a in ARMS}
    cases = {(c["terrain"], c["seed"], c["arm"]): c for c in summary["cases"]}
    if summary["status"] != "COMPLETE" or len(summary["cases"]) != 27 or set(cases) != expected:
        raise ValueError("audit requires the 27 registered terminal records")
    groups = {}
    for terrain in TERRAINS:
        for seed in SEEDS:
            group = f"{terrain}_{seed}"
            hashes = {}
            for name in GROUP_FILES:
                digests = [sha256(trials / "cases" / f"{group}_{arm}" / name) for arm in ARMS]
                if len(set(digests)) != 1:
                    raise ValueError(
                        f"arms differ for {group}/{name}; paired audit cannot collapse them"
                    )
                hashes[name] = digests[0]
            groups[group] = hashes
    return cases, groups


def set_terrain(plant, path):
    field = int(plant.model.geom_dataid[plant.floor_geom_id])
    address = int(plant.model.hfield_adr[field])
    shape = (int(plant.model.hfield_nrow[field]), int(plant.model.hfield_ncol[field]))
    with np.load(path, allow_pickle=False) as archive:
        saved = archive["hfield"]
        if saved.shape != shape or not np.isfinite(saved).all():
            raise ValueError("invalid saved collision field")
        plant.model.hfield_data[address : address + saved.size] = saved.reshape(-1)


def part_names(plant):
    model = plant.model
    bodies = {plant.base_body_id}
    for body in range(1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    parts = []
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        parts.append(
            {
                "geom_id": geom,
                "geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom),
                "body_id": body,
                "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body),
                "object": "self" if body in bodies else "environment",
                "ray_geom_group": int(model.geom_group[geom]),
                "collision_type": int(model.geom_contype[geom]),
            }
        )
    return parts


def trace(plant, parts, dx):
    data = plant.measurement_data
    rotation = data.xmat[plant.base_body_id].reshape(3, 3)
    body_origin = data.xpos[plant.base_body_id]
    # Remove yaw only. This levels the hits with the recorded true body tilt
    # at capture, an offline geometry diagnostic independent of the estimator.
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    heading = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    origins = (ORIGINS + np.array([dx, 0, 0])) @ rotation.T
    directions = DIRECTIONS @ rotation.T
    geom_ids = np.full(1, -1, dtype=np.int32)
    ranges = np.full((2, 4), np.nan)
    x = np.full((2, 4), np.nan)
    z = np.full((2, 4), np.nan)
    environment = np.zeros((2, 4), dtype=bool)
    rays = []
    for lane, origin in enumerate(origins):
        for beam, direction in enumerate(directions):
            geom_ids[0] = -1
            distance = float(
                mujoco.mj_ray(
                    plant.model, data, body_origin + origin, direction, None, True, -1, geom_ids
                )
            )
            geom = int(geom_ids[0])
            if distance < 0:
                if geom != -1:
                    raise ValueError("ray returned a geom for a miss")
                details = {
                    "geom_id": None,
                    "geom": None,
                    "body_id": None,
                    "body": None,
                    "object": "none",
                }
                band = "no_return"
            else:
                if not math.isfinite(distance) or not 0 <= geom < len(parts):
                    raise ValueError("invalid MuJoCo first return")
                details = parts[geom]
                band = (
                    "too_near"
                    if distance < MIN_RANGE_M
                    else "too_far"
                    if distance > MAX_RANGE_M
                    else "within"
                )
                if band == "within":
                    point = origin + distance * direction
                    ranges[lane, beam] = distance
                    x[lane, beam] = point @ heading
                    z[lane, beam] = point[2]
                    environment[lane, beam] = details["object"] == "environment"
            rays.append(
                {"lane": lane, "beam": beam, "raw_range_m": distance, "range_band": band, **details}
            )
    return rays, ranges, x, z, environment


def lane_segment(x, z, query):
    mask = np.isfinite(x) & np.isfinite(z)
    adjacent = mask[:-1] & mask[1:]
    if not adjacent.any():
        return None, False, "no_adjacent_pair"
    widths = np.diff(x)
    usable = adjacent & (widths >= MIN_SEGMENT_M)
    if not usable.any():
        return None, True, "no_wide_ordered_pair"
    inside = np.flatnonzero(usable & (x[:-1] <= query) & (query < x[1:]))
    if len(inside) == 1:
        i = inside.item()
        return math.atan2(z[i + 1] - z[i], widths[i]), True, "covered"
    if len(inside) > 1:
        return None, True, "ambiguous_pair"
    if query < min(x[:-1][usable]):
        return None, True, "query_nearer_than_measured"
    if query >= max(x[1:][usable]):
        return None, True, "query_farther_than_measured"
    return None, True, "query_in_internal_gap"


def coverage(x, z, query):
    lanes = [lane_segment(xs, zs, query) for xs, zs in zip(x, z)]
    grades = [g for g, _, _ in lanes]
    covered = all(g is not None for g in grades)
    valid = (
        covered
        and max(abs(g) for g in grades) <= MAX_GRADE_RAD
        and abs(grades[0] - grades[1]) <= TRACK_AGREEMENT_RAD
    )
    return {
        "adjacent_returns_both_tracks": all(adj for _, adj, _ in lanes),
        "covered_both_tracks": covered,
        "passes_grade_agreement": valid,
        "lane_reason": [why for _, _, why in lanes],
        "lane_grade_rad": grades,
    }


def grid(values):
    return [[None if not np.isfinite(v) else float(v) for v in lane] for lane in values]


def count_panel(counts, panel, suffix):
    for field in ("adjacent_returns_both_tracks", "covered_both_tracks", "passes_grade_agreement"):
        counts[f"{field}_{suffix}"] += int(panel[field])
    for lane, reason in enumerate(panel["lane_reason"]):
        counts[f"lane{lane}_{reason}_{suffix}"] += 1


def case_replay(trials, group, hashes, plant, parts, stream, metrics):
    directory = trials / "cases" / (group + "_off")
    set_terrain(plant, directory / "terrain.npz")
    with (directory / "telemetry.csv").open(newline="") as f:
        telemetry = list(csv.DictReader(f))
    with (directory / "frames.jsonl").open() as f:
        frames = [json.loads(line) for line in f]
    with np.load(directory / "states.npz", allow_pickle=False) as archive:
        times, qpos, qvel = (archive[name] for name in ("time_s", "qpos", "qvel"))
    if (
        qpos.shape != (len(telemetry) + 1, plant.model.nq)
        or qvel.shape != (len(telemetry) + 1, plant.model.nv)
        or times.shape != (len(telemetry) + 1,)
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(f"invalid state trajectory for {group}")
    if len(telemetry) != metrics["steps"]:
        raise ValueError(f"telemetry length disagrees with terminal metrics for {group}")
    captured_geometry = {mount: {} for mount in MOUNTS}
    counts = {mount: Counter() for mount in MOUNTS}
    histogram = {mount: Counter() for mount in MOUNTS}
    body_counts = {mount: Counter() for mount in MOUNTS}
    max_range_error = 0.0
    for frame in frames:
        captured, sequence = float(frame["captured_s"]), int(frame["sequence"])
        index = int(np.searchsorted(times, captured))
        if (
            index >= len(times)
            or abs(times[index] - captured) > 1e-10
            or sequence in captured_geometry["registered"]
        ):
            raise ValueError(f"unmatched or duplicate capture {group}/{sequence}")
        data = plant.measurement_data
        data.qpos[:] = qpos[index]
        data.qvel[:] = qvel[index]
        data.time = captured
        mujoco.mj_forward(plant.model, data)
        probe = {
            "case": group,
            "sequence": sequence,
            "capture_pose_index": index,
            "capture_s": captured,
            "first_used_s": frame["control_s"],
            "frame_query_x_m": frame["query_x_m"],
            "mounts": {},
        }
        for mount, shift in MOUNTS.items():
            rays, distance, x, z, env = trace(plant, parts, shift)
            if mount == "registered":
                reference = np.asarray(frame["range_m"], dtype=float)
                if reference.shape != distance.shape or not np.array_equal(
                    np.isfinite(distance), np.isfinite(reference)
                ):
                    raise ValueError(
                        f"first-return masks differ from registered data at {group}/{sequence}"
                    )
                finite = np.isfinite(reference)
                if finite.any():
                    error = float(np.max(np.abs(distance[finite] - reference[finite])))
                    max_range_error = max(max_range_error, error)
                    if error > 1e-10:
                        raise ValueError(f"registered range reconstruction differs by {error} m")
            query = float(frame["query_x_m"])
            observed = coverage(x, z, query)
            env_x = np.where(env, x, np.nan)
            env_z = np.where(env, z, np.nan)
            ground = coverage(env_x, env_z, query)
            captured_geometry[mount][sequence] = (
                x,
                z,
                env_x,
                env_z,
                captured,
                float(frame["available_s"]),
            )
            count = counts[mount]
            count["frames"] += 1
            count["frames_with_any_in_range_return"] += int(np.isfinite(distance).any())
            count["frames_with_any_in_range_environment_return"] += int(env.any())
            count_panel(count, observed, "frames")
            count_panel(count, ground, "environment_frames")
            for ray in rays:
                histogram[mount][f"{ray['object']}/{ray['range_band']}"] += 1
                body_counts[mount][f"{ray['object']}/{ray['body']}/{ray['range_band']}"] += 1
            probe["mounts"][mount] = {
                "rays": rays,
                "in_range_levelled_x_m": grid(x),
                "in_range_levelled_z_m": grid(z),
                "coverage": observed,
                "environment_only_coverage": ground,
            }
        stream.write(json.dumps(probe, allow_nan=False) + "\n")
        stream.flush()
    for row in telemetry:
        query = float(row["preview_query_x_m"])
        now = float(row["control_s"])
        room = float(row["force_headroom_n"]) > 0.0
        for mount in MOUNTS:
            count = counts[mount]
            count["control_ticks"] += 1
            count["positive_longitudinal_room_ticks"] += int(room)
            if not row["preview_sequence"]:
                count["no_frame_yet_ticks"] += 1
                continue
            sequence = int(row["preview_sequence"])
            x, z, env_x, env_z, captured, available = captured_geometry[mount][sequence]
            if not captured <= available <= now + 1e-9 or now - captured > MAX_AGE_S + 1e-9:
                count["unusable_time_ticks"] += 1
                continue
            observed = coverage(x, z, query)
            ground = coverage(env_x, env_z, query)
            count_panel(count, observed, "ticks")
            count_panel(count, ground, "environment_ticks")
            count["qualified_geometry_and_positive_room_ticks"] += int(
                ground["passes_grade_agreement"] and room
            )
            early = (
                metrics["first_front_contact_s"] is not None
                and now < metrics["first_front_contact_s"] - 1e-9
            )
            count["qualified_geometry_room_before_contact_ticks"] += int(
                ground["passes_grade_agreement"] and room and early
            )
    return {
        "case": group,
        "paired_hashes": hashes,
        "registered_ranges_max_error_m": max_range_error,
        "loaded_capture_poses": len(frames),
        "status_of_original_case": metrics["termination_reason"],
        "mount_counts": counts,
        "ray_counts": histogram,
        "hit_body_counts": body_counts,
    }


def audit(run_root, output):
    trials = run_root / "trials" if (run_root / "trials" / "summary.json").is_file() else run_root
    output.mkdir(parents=True, exist_ok=False)
    declaration = {
        "status": "STARTED",
        "run_root": str(trials.resolve()),
        "shifts_x_m": MOUNTS,
        "only_variable": "translate both sensor origins by +0.16 m in body x",
        "native_integration_steps": 0,
        "training_updates": 0,
        "geometry": "stored pose at capture, true gravity-level frame for both mounts; original query positions at receipt and each control tick",
        "summary_input_sha256": sha256(trials / "summary.json"),
        "audit_source_sha256": sha256(Path(__file__)),
    }
    write_json(output / "started.json", declaration)
    complete = []
    try:
        original, groups = check_groups(trials)
        plant = D1Plant(training_terrain=TrainingTerrainConfig(), sampling_mode="synchronized")
        parts = part_names(plant)
        with (output / "probes.jsonl").open("x") as probes:
            for terrain in TERRAINS:
                for seed in SEEDS:
                    group = f"{terrain}_{seed}"
                    complete.append(
                        case_replay(
                            trials,
                            group,
                            groups[group],
                            plant,
                            parts,
                            probes,
                            original[terrain, seed, "off"],
                        )
                    )
        pooled = {}
        for key in ("mount_counts", "ray_counts", "hit_body_counts"):
            pooled[key] = {}
            for mount in MOUNTS:
                total = Counter()
                for case in complete:
                    total.update(case[key][mount])
                pooled[key][mount] = dict(total)
        result = {
            **declaration,
            "status": "COMPLETE",
            "paired_terminal_cases": 27,
            "distinct_pose_series": len(complete),
            "pose_forward_evaluations": sum(c["loaded_capture_poses"] for c in complete),
            "pool": pooled,
            "raw_probe_rows_sha256": sha256(output / "probes.jsonl"),
            "cases": complete,
            "max_registered_range_error_m": max(
                c["registered_ranges_max_error_m"] for c in complete
            ),
            "limitations": "The alternative is evaluated on unchanged archived trajectories and queries, using exact capture attitude. Self hits can pass range limits but are not terrain. No control gain or robustness is inferred.",
        }
        write_json(output / "summary.json", result)
        return result
    except BaseException as exc:
        write_json(
            output / "partial.json",
            {
                "status": "PARTIAL",
                "finished_pose_series": [c["case"] for c in complete],
                "error": repr(exc),
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.run_root, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "frames": result["pose_forward_evaluations"],
                "mount_counts": result["pool"]["mount_counts"],
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
