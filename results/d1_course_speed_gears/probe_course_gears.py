"""Actual headless course-runner speed-gear trials, with no plant-state writes.

Uses a simulated input clock and the same runner/adapter/plant as keyboard driving.
The injected viewer supplies held key sets; it never creates a desktop window.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def summarize(output, *, target, drive_start, release_time, limits):
    with np.load(output / "states.npz", allow_pickle=False) as source:
        qpos, qvel = source["qpos"], source["qvel"]
        times, torque = source["segment_time_s"], source["applied_torque_nm"]
        finite = bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()
                      and np.isfinite(torque).all())
    with (output / "telemetry.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    # Positions are post-integration values. Derived live MuJoCo measurements
    # retain legacy timing, so report independent finite-difference progress.
    vx = np.diff(qpos[:, 0]) / np.diff(times)
    before = times[:-1]
    steady = (before >= release_time - 2 - 1e-8) & (before < release_time - 1e-8)
    final = before >= times[-1] - 1 - 1e-8
    w, x, y, z = qpos[:, 3:7].T
    roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1, 1))
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    start = int(np.argmin(np.abs(times-drive_start)))
    release = int(np.argmin(np.abs(times-release_time)))
    settled = next((i for i in range(release, len(vx)-19)
                    if bool(np.all(np.abs(vx[i:i+20]) < .05))), None)
    request_after_release = [float(r["requested_forward_mps"]) for r in rows[release:]]
    max_roll_pitch = float(max(np.abs(roll).max(), np.abs(pitch).max()))
    has_fallen = bool(np.any(np.abs(roll) > .7) or np.any(np.abs(pitch) > .7)
                      or any(r["safety_mode"] == "fallen_recovery" for r in rows))
    speed = float(np.mean(vx[steady]))
    final_abs_speed = float(np.mean(np.abs(vx[final])))
    checks = {
        "finite": finite,
        "no_fall": not has_fallen,
        "attitude_within_0_30_rad": max_roll_pitch < .30,
        "last_2s_speed_within_25_percent_of_target": .75 <= speed/target <= 1.25,
        "release_command_immediately_zero": all(v == 0 for v in request_after_release),
        "last_1s_mean_abs_speed_below_0_05_mps": final_abs_speed < .05,
        "torques_within_unchanged_limits": bool(np.all(np.abs(torque) <= limits+1e-10)),
    }
    return {
        "target_mps": target,
        "integrated_duration_s": float(times[-1]),
        "steady_last_2s_world_vx_mps": speed,
        "drive_world_x_progress_m": float(qpos[release, 0]-qpos[start, 0]),
        "release_world_x_drift_m": float(qpos[-1, 0]-qpos[release, 0]),
        "last_1s_mean_abs_world_vx_mps": final_abs_speed,
        "time_to_0_05_mps_for_0_20s_after_release_s": (
            None if settled is None else float(times[settled]-release_time)),
        "max_abs_roll_pitch_rad": max_roll_pitch,
        "max_abs_yaw_rad": float(np.abs(yaw).max()),
        "max_abs_torque_nm": float(np.abs(torque).max()),
        "torque_saturation_fraction": float(np.mean(np.abs(torque) >= .98*limits)),
        "final_position_m": qpos[-1, :3].tolist(),
        "checks": checks,
        "flat_tracking_criteria_pass": all(checks.values()),
        "raw_manifest_sha256": sha(output / "manifest.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from scripts.d1_course_controls import CourseKeyboardCommands
    from scripts.run_d1_course_drive import run
    from wheel_legged_control.d1.interactive import D1InteractiveSimulation

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = [("flat", "start", gear, direction) for direction in (1, -1)
             for gear in (1, 2, 3)]
    cases += [("course", zone, gear, 1) for zone in ("rough", "ramp", "stairs")
              for gear in (1, 2, 3)]
    dump(output / "protocol.json", {
        "schema": "d1-course-gear-probe-v1",
        "probe_sha256": sha(__file__),
        "forward_gears_mps": CourseKeyboardCommands.forward_gears_mps,
        "reverse_gears_mps": CourseKeyboardCommands.reverse_gears_mps,
        "cases": cases,
        "settle_seconds": 2,
        "flat_drive_seconds": 8,
        "course_drive_seconds": 12,
        "release_seconds": 4,
        "input_clock": "simulation time; keys updated once per existing 0.01s control tick",
        "model_changes": "none; standard flat/course arenas and named spawn resets only",
        "gui": "none; injected viewer provides input without any GLFW/X11 calls",
        "flat_acceptance": ["finite; no fall; max abs roll/pitch < .30 rad",
                            "last 2s mean world vx within 25% of signed target",
                            "release immediately clears target; last 1s abs vx mean < .05 m/s",
                            "all applied torques within unchanged model limits"],
        "course_acceptance": "no traversal certification; report actual progress and failures",
        "policy": "none; default LQR/VMC course path and oracle state",
    })
    summaries = {}
    for arena, zone, gear, direction in cases:
        name = f"{arena}_{zone}_{'forward' if direction > 0 else 'reverse'}_gear{gear}"
        simulation = D1InteractiveSimulation(arena=arena)
        clock = [0.0]
        commands = CourseKeyboardCommands(clock=lambda: clock[0])
        release_time = 10 if arena == "flat" else 14

        class ProbeViewer:
            def __init__(self, model, data, commands, **_):
                self.commands, self.events = commands, []
                self.cam = SimpleNamespace(lookat=np.zeros(3))

            def poll(self, seconds):
                clock[0] = seconds
                tick = round(seconds/.01)
                keys = set()
                if gear >= 2 and tick == 100:
                    keys.add(340)
                if gear >= 3 and tick == 102:
                    keys.add(344)
                if 200 <= tick < round(release_time/.01):
                    keys.add(ord("W") if direction > 0 else ord("S"))
                self.commands.update_pressed(keys)

            def is_running(self):
                return True

            def lock(self):
                return nullcontext()

            def set_status(self, *_):
                pass

            def sync(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        run(simulation, output/name, seconds=release_time+4, zone=zone,
            commands=commands, viewer_factory=ProbeViewer, realtime=False)
        magnitude = (commands.forward_gears_mps if direction > 0
                     else commands.reverse_gears_mps)[gear-1]
        summaries[name] = summarize(output/name, target=direction*magnitude,
                                    drive_start=2, release_time=release_time,
                                    limits=simulation.plant.actuator_torque_limit_nm)
        dump(output/f"{name}.json", summaries[name])
        print(name, json.dumps(summaries[name], sort_keys=True), flush=True)
    dump(output/"summary.json", summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
