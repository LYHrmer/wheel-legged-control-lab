#!/usr/bin/env python3
"""Headless calls through the actual frozen GUI runner and torque controller."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def summary(directory, case, commands, limits):
    with np.load(directory / "states.npz", allow_pickle=False) as data:
        pos, vel, times, torque = (data[key] for key in
                                  ("qpos", "qvel", "segment_time_s", "applied_torque_nm"))
    with (directory / "telemetry.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    w, x, y, z = pos[:, 3:7].T
    roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = np.arcsin(np.clip(2*(w*y-z*x), -1, 1))
    yaw = np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    velocity = np.diff(pos[:, :3], axis=0)/np.diff(times)[:, None]
    release = commands.release_time
    release_index = None if release is None else round(release/.01)
    transitions = []
    for index, row in enumerate(rows):
        if index == 0 or any(row[key] != rows[index-1][key] for key in
                             ("applied_controller", "side_phase", "side_failure")):
            transitions.append({"tick": index+1, "before_time_s": float(row["segment_before_time_s"]),
                                **{key: row[key] for key in ("applied_controller", "side_phase",
                                   "side_failure", "side_done", "side_success")}})
    recovery_indices = [i for i, row in enumerate(rows) if row["safety_mode"] == "recovery"]
    out = {
        "case": case, "steps": len(rows), "integrated_duration_s": float(times[-1]),
        "initial_position_m": pos[0, :3].tolist(), "final_position_m": pos[-1, :3].tolist(),
        "max_abs_roll_pitch_rad": float(max(np.abs(roll).max(), np.abs(pitch).max())),
        "max_abs_yaw_from_initial_rad": float(np.abs(yaw-yaw[0]).max()),
        "max_abs_lateral_from_initial_m": float(np.abs(pos[:, 1]-pos[0, 1]).max()),
        "final_lateral_from_initial_m": float(pos[-1, 1]-pos[0, 1]),
        "release_time_s": release, "first_recovery_time_s": None if not recovery_indices else
        float(times[recovery_indices[0]]),
        "finite": bool(np.isfinite(pos).all() and np.isfinite(vel).all() and np.isfinite(torque).all()),
        "torques_within_limits": bool(np.all(np.abs(torque) <= limits+1e-10)),
        "torque_saturation_fraction": float(np.mean(np.abs(torque) >= .98*limits)),
        "traction_boost_ticks": sum(row["traction_mode"] == "boost" for row in rows),
        "side_transition_events": transitions,
        "raw_manifest_sha256": digest(directory / "manifest.json"),
    }
    if release_index is not None:
        stop = next((i for i in range(release_index, len(velocity)-19)
                     if np.all(np.abs(velocity[i:i+20, 0]) < .05)), None)
        out.update({
            "release_position_m": pos[release_index, :3].tolist(),
            "after_release_xyz_displacement_m": (pos[-1, :3]-pos[release_index, :3]).tolist(),
            "last_second_mean_abs_world_velocity_mps": np.abs(velocity[-100:]).mean(axis=0).tolist(),
            "time_to_vx_below_0p05_for_0p2s_s": None if stop is None else float(times[stop]-release),
            "requested_forward_zero_after_release": all(float(row["requested_forward_mps"]) == 0.
                                                         for row in rows[release_index:]),
        })
    if case["kind"] == "forward":
        start = 200
        steady = velocity[release_index-200:release_index, 0]
        requests = np.array([float(row["requested_forward_mps"]) for row in rows])
        out.update({
            "steady_last_2s_world_vx_mps": float(steady.mean()),
            "drive_world_x_progress_m": float(pos[release_index, 0]-pos[start, 0]),
            "drive_world_vx_tracking_rmse_mps": float(np.sqrt(np.mean(
                (velocity[start:release_index, 0]-requests[start:release_index])**2))),
        })
        checks = {
            "finite_and_torque_limits": out["finite"] and out["torques_within_limits"],
            "no_recovery": not recovery_indices,
            "attitude_below_0p3_rad": out["max_abs_roll_pitch_rad"] < .3,
            "yaw_drift_below_0p15_rad": out["max_abs_yaw_from_initial_rad"] < .15,
            "lateral_drift_below_0p10_m": out["max_abs_lateral_from_initial_m"] < .10,
            "steady_vx_within_25_percent": abs(steady.mean()/case["target_mps"]-1) <= .25,
            "release_immediately_zero": out["requested_forward_zero_after_release"],
            "last_second_abs_vx_below_0p05": out["last_second_mean_abs_world_velocity_mps"][0] < .05,
        }
        out["diagnostic_checks"] = checks
        out["straight_and_brake_diagnostic_pass"] = all(checks.values())
    else:
        handoffs = [i for i in range(1, len(rows)) if rows[i]["applied_controller"] == "legacy"
                    and rows[i-1]["applied_controller"] != "legacy"]
        out["side_handoffs"] = []
        for i in handoffs:
            # State i is exactly the last side-controlled integrated state.
            end = min(len(pos)-1, i+400)
            direction = 1 if case["key"] == "A" else -1
            signed_handoff_y = direction*(pos[i, 1]-pos[200, 1])
            retained_y = direction*(pos[end, 1]-pos[200, 1])
            out["side_handoffs"].append({
                "time_s": float(times[i]), "phase": rows[i-1]["side_phase"],
                "failure": rows[i-1]["side_failure"], "success": rows[i-1]["side_success"],
                "position_m": pos[i, :3].tolist(), "measured_after_seconds": float(times[end]-times[i]),
                "world_xy_displacement_after_handoff_m": (pos[end, :2]-pos[i, :2]).tolist(),
                "signed_lateral_at_handoff_m": float(signed_handoff_y),
                "signed_lateral_retained_m": float(retained_y),
                "retention_fraction": None if abs(signed_handoff_y) < 1e-6 else float(retained_y/signed_handoff_y),
                "wheel_torque_jump_nm": (torque[i, [3, 7, 11, 15]]-torque[i-1, [3, 7, 11, 15]]).tolist(),
                "all_motor_torque_jump_l2_nm": float(np.linalg.norm(torque[i]-torque[i-1])),
            })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset", choices=("forward", "side"), required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from scripts.d1_course_controls import CourseKeyboardCommands
    from scripts.run_d1_course_drive import run
    from wheel_legged_control.d1.interactive import D1InteractiveSimulation

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.subset == "forward":
        cases = [{"name": f"{arena}_{zone}_gear{gear}", "kind": "forward", "arena": arena,
                  "zone": zone, "gear": gear, "key": "W", "target_mps": target,
                  "release_seconds": 10. if arena == "flat" else 14.,
                  "duration_seconds": 14. if arena == "flat" else 18.}
                 for arena, zone in (("flat", "start"), ("course", "rough"),
                                     ("course", "ramp"), ("course", "stairs"))
                 for gear, target in enumerate(CourseKeyboardCommands.forward_gears_mps, 1)]
    else:
        cases = [{"name": f"course_start_{key}_{mode}", "kind": "side", "arena": "course",
                  "zone": "start", "gear": 1, "key": key, "release_mode": mode,
                  "release_seconds": 3. if mode == "short" else None,
                  "duration_seconds": 10. if mode == "short" else 22.}
                 for key in ("A", "D") for mode in ("short", "one_cycle")]
    write_json(output / "protocol.json", {
        "schema": "d1-gui-stability-baseline-v1", "helper_sha256": digest(__file__),
        "cases": cases, "settle_seconds": 2., "physics_changes": "none",
        "runner": "scripts.run_d1_course_drive.run", "baseline": "lqr", "state_mode": "oracle",
        "contact_allocation": "legacy", "side_step_profile": "fast", "sampling": "legacy_mixed",
        "input": "CourseKeyboardCommands subclass supplies held sets at each genuine runner tick; no GUI or input injection to the desktop",
        "world_velocity": "finite difference of integrated world position, not the legacy body-COM velocity observation",
        "side_release": "short: release after 1s hold; one_cycle: release on first tick after side_active becomes false, including failed cycles",
        "checks": "diagnostic thresholds frozen before run, not certified task acceptance: yaw<.15rad, lateral<.10m, roll/pitch<.30rad, steady vx ±25%, final mean |vx|<.05m/s",
        "ranked_provisional_hypotheses": [
            "Heading regulation has no cross-track reference, so rough asymmetric contacts can cause lateral drift despite modest yaw error.",
            "Side landing is safe but legacy handoff restores nominal leg shape and discards its own references/torque memory; stable gained lateral displacement may shrink after handoff.",
            "A short A/D press aborts during a support shift; apparent lateral movement may be temporary COM repositioning rather than a completed 3cm step.",
            "The keyboard target reaches each gear, but stuck/terrain guards and traction dynamics prevent obstacle progress; changing keyboard repeat is insufficient.",
        ],
    })
    summaries = []
    for case in cases:
        simulation = D1InteractiveSimulation(arena=case["arena"], baseline="lqr", state_mode="oracle")

        class ScheduledCommands(CourseKeyboardCommands):
            def __init__(self):
                super().__init__(clock=lambda: float(simulation.plant.data.time))
                self.release_time = case["release_seconds"]
                self.seen_side_active = False
                self.input_events = []
                self.previous_keys = None

            def __call__(self, seconds):
                tick = round(seconds/.01)
                keys = set()
                if case["gear"] >= 2 and tick == 100:
                    keys.add(340)
                if case["gear"] >= 3 and tick == 102:
                    keys.add(344)
                if self.side_active:
                    self.seen_side_active = True
                if case["kind"] == "side" and self.release_time is None and self.seen_side_active and not self.side_active:
                    self.release_time = tick*.01
                if tick >= 200 and (self.release_time is None or tick < round(self.release_time/.01)):
                    keys.add(ord(case["key"]))
                if keys != self.previous_keys:
                    self.input_events.append({"time_s": tick*.01, "held_keys": sorted(keys)})
                    self.previous_keys = keys.copy()
                self.update_pressed(keys)
                return super().__call__(seconds)

        commands = ScheduledCommands()
        directory = output / case["name"]
        run(simulation, directory, seconds=case["duration_seconds"], zone=case["zone"],
            commands=commands, viewer_factory=None, realtime=False, side_step_profile="fast")
        result = summary(directory, case, commands, simulation.plant.actuator_torque_limit_nm)
        summaries.append(result)
        write_json(output / f"{case['name']}.analysis.json", result)
        write_json(output / f"{case['name']}.input.json", commands.input_events)
        print(json.dumps({"case": case["name"], "steps": result["steps"], "release": result["release_time_s"],
                          "straight_pass": result.get("straight_and_brake_diagnostic_pass"),
                          "handoffs": result.get("side_handoffs")}, allow_nan=False), flush=True)
    write_json(output / "summary.json", summaries)
    write_json(output / "manifest.json", {str(path.relative_to(output)): {"sha256": digest(path),
               "bytes": path.stat().st_size} for path in sorted(output.rglob("*")) if path.is_file()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
