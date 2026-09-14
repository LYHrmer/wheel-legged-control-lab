#!/usr/bin/env python3
"""Work-only single-variable study of the retained-reference removal fraction."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from probe_gui_baseline import digest, summary, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from scripts import d1_course_side_step
    from scripts.d1_course_braking import CourseBrakeReference
    from scripts.d1_course_controls import CourseKeyboardCommands
    from scripts.run_d1_course_drive import run
    from wheel_legged_control.d1.interactive import D1InteractiveSimulation

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = [{"name": f"{arena}_{zone}_gear{gear}", "kind": "forward", "arena": arena,
              "zone": zone, "gear": gear, "key": "W", "target_mps": target,
              "release_seconds": release, "duration_seconds": release+4.}
             for arena, zone, gear, target, release in
             (("flat", "start", 3, .5, 10.), ("course", "ramp", 2, .4, 14.),
              ("course", "rough", 3, .5, 14.), ("course", "stairs", 3, .5, 14.))]
    write_json(output / "protocol.json", {
        "schema": "d1-braking-reference-fraction-study-v1", "cases": cases, "alphas": [0., .5, 1.],
        "helper_sha256": digest(__file__),
        "metric_helper_sha256": digest(Path(__file__).with_name("probe_gui_baseline.py")),
        "intervention": "Only the removal fraction on the existing eligible moving-to-zero edge changes. new_reference=distance+(1-alpha)*(old_reference-distance). alpha0 restores old reference exactly, alpha1 keeps the production full anchor, alpha.5 retains half the previous error. No physical state or gains change.",
        "in_process_subclass": "The work helper temporarily supplies a CourseBrakeReference subclass to the course adapter factory, after the real core validates and consumes the edge; it adjusts reference before any compute or physics step. The repo files and recorded physics model are unchanged. Alpha0 returns no reanchor event/count because its net reference is unchanged.",
        "input": "settle2s; W same timed ramp/gears; release same time; observe4s; no GUI",
        "scope": "candidate comparison, not a selected default or obstacle traversal certificate",
    })
    all_results = []
    original_factory = d1_course_side_step.CourseBrakeReference
    try:
        for case in cases:
            for alpha in (0., .5, 1.):
                records = []

                class FractionalBrake(CourseBrakeReference):
                    def __init__(self, fraction=alpha, record_sink=records):
                        super().__init__()
                        self.fraction, self.record_sink = fraction, record_sink

                    def prepare(self, controller, requested_forward_mps, *, eligible):
                        old_reference = controller.control_memory.distance_reference_m
                        result = super().prepare(controller, requested_forward_mps, eligible=eligible)
                        if result["brake_reference_reanchored"]:
                            distance = controller.control_memory.distance_m
                            if self.fraction == 0.:
                                controller._distance_reference_m = old_reference
                                self._event_count -= 1
                            elif self.fraction != 1.:
                                controller._distance_reference_m = distance+(1-self.fraction)*(old_reference-distance)
                            new_error = controller._distance_reference_m-distance
                            self.record_sink.append({"alpha": self.fraction, "old_reference_m": old_reference,
                                "distance_m": distance, "old_error_m": old_reference-distance,
                                "new_error_m": new_error})
                            result = self._diagnostics(self.fraction != 0., old_reference-distance, new_error)
                        return result

                d1_course_side_step.CourseBrakeReference = FractionalBrake
                simulation = D1InteractiveSimulation(arena=case["arena"], baseline="lqr", state_mode="oracle")

                class Commands(CourseKeyboardCommands):
                    def __init__(self, sim, spec):
                        self.simulation, self.case = sim, spec
                        super().__init__(clock=lambda: float(self.simulation.plant.data.time))
                        self.release_time = spec["release_seconds"]

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
                        return super().__call__(seconds)

                commands = Commands(simulation, case)
                name = f"{case['name']}_alpha{str(alpha).replace('.', 'p')}"
                directory = output / name
                run(simulation, directory, seconds=case["duration_seconds"], zone=case["zone"],
                    commands=commands, viewer_factory=None, realtime=False, side_step_profile="fast")
                result = summary(directory, case, commands, simulation.plant.actuator_torque_limit_nm)
                result.update(alpha=alpha, release_reference_events=records)
                assert len(records) == 1, "The study requires exactly one actual release edge"
                all_results.append(result)
                write_json(output / f"{name}.analysis.json", result)
                print(json.dumps({"case": case["name"], "alpha": alpha,
                    "dx": result["after_release_xyz_displacement_m"][0],
                    "last1_abs_vx": result["last_second_mean_abs_world_velocity_mps"][0],
                    "settling_s": result["time_to_vx_below_0p05_for_0p2s_s"],
                    "reference": records[0]}, allow_nan=False), flush=True)
    finally:
        d1_course_side_step.CourseBrakeReference = original_factory
    write_json(output / "summary.json", all_results)
    write_json(output / "manifest.json", {str(path.relative_to(output)): {"sha256": digest(path),
        "bytes": path.stat().st_size} for path in sorted(output.rglob("*")) if path.is_file()})


if __name__ == "__main__":
    main()
