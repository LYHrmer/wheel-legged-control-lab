#!/usr/bin/env python3
"""A single controller-memory intervention at release; physics/gains unchanged."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from probe_gui_baseline import digest, summary, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from scripts.d1_course_controls import CourseKeyboardCommands
    from scripts.run_d1_course_drive import run
    from wheel_legged_control.d1.interactive import D1InteractiveSimulation

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = [{"name": f"{arena}_{zone}_gear{gear}", "kind": "forward", "arena": arena,
              "zone": zone, "gear": gear, "key": "W", "target_mps": target,
              "release_seconds": release, "duration_seconds": release+4.}
             for arena, zone, gear, target, release in
             (("flat", "start", 3, .5, 10.), ("course", "ramp", 2, .4, 14.))]
    write_json(output / "protocol.json", {
        "schema": "d1-release-reference-diagnostic-v1", "cases": cases,
        "variants": ["unchanged", "anchor_once_at_release"], "helper_sha256": digest(__file__),
        "metric_helper_sha256": digest(Path(__file__).with_name("probe_gui_baseline.py")),
        "single_intervention": "Before computing the first zero-speed release tick, set only controller._distance_reference_m=controller._distance_m. No other controller state, source, gain, qpos, qvel, contact or simulation timing is altered.",
        "falsifiable_prediction": "If integrated distance-reference error drives release motion, anchoring it once changes post-release drift and settling while pre-release physical states remain exactly identical.",
        "scope": "read-only repository, work-only causal diagnostic; not a proposed full controller or certified improvement",
    })
    results = []
    for case in cases:
        for variant in ("unchanged", "anchor_once_at_release"):
            simulation = D1InteractiveSimulation(arena=case["arena"], baseline="lqr", state_mode="oracle")

            class Commands(CourseKeyboardCommands):
                def __init__(self, sim, spec, method):
                    self.simulation, self.case, self.method = sim, spec, method
                    super().__init__(clock=lambda: float(self.simulation.plant.data.time))
                    self.release_time = spec["release_seconds"]
                    self.records = []

                def __call__(self, seconds):
                    tick = round(seconds/.01)
                    keys = set()
                    if self.case["gear"] >= 2 and tick == 100:
                        keys.add(340)
                    if self.case["gear"] >= 3 and tick == 102:
                        keys.add(344)
                    if 200 <= tick < round(self.release_time/.01):
                        keys.add(ord("W"))
                    self.update_pressed(keys)
                    command = super().__call__(seconds)
                    controller = self.simulation.teleop.controller
                    before = float(controller._distance_reference_m-controller._distance_m)
                    changed = self.method == "anchor_once_at_release" and tick == round(self.release_time/.01)
                    if changed:
                        controller._distance_reference_m = controller._distance_m
                    self.records.append({
                        "pre_compute_time_s": seconds, "requested_forward_mps": command.forward_velocity_mps,
                        "distance_m": float(controller._distance_m),
                        "distance_reference_m": float(controller._distance_reference_m),
                        "distance_reference_error_before_intervention_m": before,
                        "distance_reference_error_after_intervention_m": float(controller._distance_reference_m-controller._distance_m),
                        "previous_tick_longitudinal_force_n": float(controller.last_longitudinal_force_n),
                        "intervention_applied": changed,
                    })
                    return command

            commands = Commands(simulation, case, variant)
            directory = output / f"{case['name']}_{variant}"
            run(simulation, directory, seconds=case["duration_seconds"], zone=case["zone"],
                commands=commands, viewer_factory=None, realtime=False, side_step_profile="fast")
            result = summary(directory, case, commands, simulation.plant.actuator_torque_limit_nm)
            result["variant"] = variant
            release = commands.records[round(case["release_seconds"]/.01)]
            result["reference_at_release"] = release
            write_json(output / f"{directory.name}.memory.json", commands.records)
            write_json(output / f"{directory.name}.analysis.json", result)
            results.append(result)
            print(json.dumps({"case": case["name"], "variant": variant,
                              "release_reference_error": release["distance_reference_error_before_intervention_m"],
                              "release_dx": result["after_release_xyz_displacement_m"][0],
                              "last_second_abs_vx": result["last_second_mean_abs_world_velocity_mps"][0],
                              "settling_s": result["time_to_vx_below_0p05_for_0p2s_s"]}), flush=True)
    write_json(output / "summary.json", results)
    write_json(output / "manifest.json", {str(path.relative_to(output)): {"sha256": digest(path),
               "bytes": path.stat().st_size} for path in sorted(output.rglob("*")) if path.is_file()})


if __name__ == "__main__":
    main()
