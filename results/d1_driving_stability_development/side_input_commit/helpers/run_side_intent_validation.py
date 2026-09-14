"""Fixed finite MuJoCo checks through the actual course input and runner path."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

CASES = [
    {"name": "tap_A", "duration": 20., "zone": "start", "kind": "tap", "direction": 1},
    {"name": "tap_D", "duration": 20., "zone": "start", "kind": "tap", "direction": -1},
    {"name": "held_second", "duration": 34., "zone": "start", "kind": "held", "direction": 1},
    {"name": "reverse_second", "duration": 36., "zone": "start", "kind": "reverse", "direction": 1},
    {"name": "cancel_shift", "duration": 20., "zone": "start", "kind": "cancel_shift", "direction": 1},
    {"name": "cancel_swing", "duration": 25., "zone": "start", "kind": "cancel_swing", "direction": 1},
    {"name": "nonflat_guard", "duration": 6., "zone": "ramp", "kind": "tap", "direction": 1},
    {"name": "jump_inhibition", "duration": 10., "zone": "start", "kind": "jump", "direction": 1},
    {"name": "reset_inhibition", "duration": 28., "zone": "start", "kind": "reset", "direction": 1},
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def analyse(directory, case, trace, cycles, cancel_time):
    with np.load(directory / "states.npz", allow_pickle=False) as states:
        qpos, qvel, torque = states["qpos"], states["qvel"], states["applied_torque_nm"]
        segments = states["segment_id"]
    with (directory / "telemetry.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    checks = []
    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
    check("exact_physical_step_count", len(rows) == len(trace) == len(torque) == round(case["duration"]/.01))
    check("finite_states_and_torques", np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(torque).all())
    check("one_owner_one_step", all(abs(t["after_time_s"]-t["before_time_s"]-.01) < 1e-8 for t in trace))
    check("torques_within_actual_limits", all(t["torque_within_limits"] for t in trace))
    for cycle in cycles:
        if "end_tick" not in cycle:
            cycle["unfinished_or_explicitly_reset"] = True
            continue
        start, end = cycle["start_tick"], cycle["end_tick"]
        start_row, end_row = rows[start], rows[end-1]
        before, after = int(start_row["state_before_index"]), int(end_row["state_after_index"])
        first, last = qpos[before], qpos[after]
        w, x, y, z = first[3:7]
        yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        left = np.array([-math.sin(yaw), math.cos(yaw)])
        cycle["signed_body_lateral_m"] = float(cycle["direction"] * np.dot(last[:2]-first[:2], left))
        cycle["cycle_seconds"] = (end-start)*.01
        cycle["same_segment"] = bool(segments[before] == segments[after])
        completion = trace[end-1]
        cycle["safe_completion_contacts"] = all(completion["wheel_contacts_after"])
        cycle["completion_speed_mps"] = completion["base_origin_speed_after_mps"]
        cycle["completion_support_margin_m"] = completion["support_margin_m"]
        cycle["completion_contact_dwell_s"] = completion["contact_dwell_s"]
        cycle["completion_pending_leg"] = completion["pending_leg"]
        check(f"cycle{cycle['index']}_side_owned_through_completion",
              all(t["torque_source"] != "legacy" for t in trace[start:end]))
        check(f"cycle{cycle['index']}_four_contacts_before_handoff", cycle["safe_completion_contacts"])
        check(f"cycle{cycle['index']}_no_pending_leg_before_handoff", cycle["completion_pending_leg"] is None)
        if cycle["success"]:
            check(f"cycle{cycle['index']}_duration_within_15_seconds", cycle["cycle_seconds"] <= 15.)
            check(f"cycle{cycle['index']}_signed_lateral_2p5_to_5cm", .025 <= cycle["signed_body_lateral_m"] <= .05)
        if case["name"] in ("tap_A", "tap_D") and end+400 <= len(rows):
            retained_index = int(rows[end+399]["state_after_index"])
            retained = float(cycle["direction"] * np.dot(qpos[retained_index, :2]-first[:2], left))
            signed = cycle["signed_body_lateral_m"]
            cycle["retained_lateral_after_4s_m"] = retained
            cycle["retained_fraction_after_4s"] = retained/signed if signed > 0 else None
            trajectory = qpos[after:retained_index+1, :2]
            signed_track = cycle["direction"] * ((trajectory-first[:2]) @ left)
            cycle["max_handoff_rollback_m"] = float(max(0., signed-np.min(signed_track)))
            check("handoff_4s_same_segment", segments[retained_index] == segments[after])
            check("handoff_retains_at_least_90percent", signed > 0 and retained >= .9*signed)
            check("handoff_rollback_at_most_5mm", cycle["max_handoff_rollback_m"] <= .005)

    completed = [cycle for cycle in cycles if "end_tick" in cycle]
    if case["name"] in ("tap_A", "tap_D"):
        check("exactly_one_successful_cycle", len(cycles) == len(completed) == 1 and completed[0]["success"])
        check("ordinary_release_did_not_cancel", not any(t["failure"] == "cancelled" for t in trace))
    elif case["kind"] in ("held", "reverse"):
        check("exactly_two_successful_cycles_no_third", len(cycles) == len(completed) == 2 and all(c["success"] for c in completed))
        expected = [1, -1] if case["kind"] == "reverse" else [1, 1]
        check("directions_only_change_at_cycle_boundary", [c["direction"] for c in cycles] == expected)
        if len(completed) == 2:
            check("second_starts_after_first_completion", completed[1]["start_tick"] >= completed[0]["end_tick"])
    elif case["kind"].startswith("cancel"):
        check("explicit_cancel_was_executed", cancel_time is not None)
        check("single_aborted_cycle_no_restart", len(cycles) == len(completed) == 1 and not completed[0]["success"] and completed[0]["failure"] == "cancelled")
        if cancel_time is not None:
            packet = rows[round(cancel_time/.01)]
            check("cancel_packet_raw_held_preserved", packet["side_raw_held_direction"] == "1" and packet["side_resolved_direction"] == "0")
    elif case["name"] == "nonflat_guard":
        check("actual_nonflat_start_guard_unchanged", len(cycles) == 0 and all(t["torque_source"] == "legacy" for t in trace))
    elif case["kind"] == "jump":
        check("actual_jump_was_active", any(r["jump_phase"] != "ready" for r in rows))
        check("side_input_during_jump_never_delayed", len(cycles) == 0)
        check("side_tap_saw_mode_inhibition", any(r["side_press_direction"] == "1" and r["side_start_inhibited"] == "True" for r in rows))
    elif case["kind"] == "reset":
        check("reset_created_distinct_segment", len(set(segments.tolist())) == 2)
        check("no_old_held_restarted_after_reset", not any(300 <= c["start_tick"] < 800 for c in cycles))
        check("fresh_tap_after_release_started_new_segment_cycle", any(c["start_tick"] >= 800 and c.get("success") for c in cycles))
        reset_row = rows[300]
        check("reset_packet_cannot_fake_a_release", reset_row["side_sampled_after_reset"] == "False" and reset_row["side_blocked_until_release"] == "True")
    return {"case": case, "steps": len(rows), "cycles": cycles, "cancel_time_s": cancel_time,
            "checks": checks, "failed_checks": [c["name"] for c in checks if not c["passed"]],
            "all_passed": all(c["passed"] for c in checks), "manual_gui_tested": False}


def run_case(runner, simulation_class, commands_class, output, case):
    original = runner.CourseSideStepDrive
    original_json = runner._json
    trace, cycles = [], []
    box, clock = {}, [0.]
    simulation = simulation_class(arena="course", baseline="lqr", state_mode="oracle")
    commands = commands_class(clock=lambda: clock[0])
    cancel_time = None

    class ObservedDrive(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            box["drive"] = self

        def step(self, requested, direction):
            was_active = self.active
            before_time = float(self.plant.data.time)
            before_phase = self.side.phase
            status, torque, info = super().step(requested, direction)
            index = len(trace)
            if not was_active and info["torque_source"] != "legacy":
                cycles.append({"index": len(cycles)+1, "start_tick": index,
                               "direction": int(self.side.direction)})
            if info["torque_source"] != "legacy" and self.side.done and not self.active:
                cycles[-1].update(end_tick=index+1, success=bool(self.side.success), failure=self.side.failure)
            trace.append({"tick": index+1, "before_time_s": before_time,
                          "after_time_s": float(self.plant.data.time), "phase_before": before_phase,
                          "phase_after": self.side.phase, "torque_source": info["torque_source"],
                          "failure": self.side.failure, "active_after": bool(self.active),
                          "wheel_contacts_after": self.side._contacts().tolist(),
                          "base_origin_speed_after_mps": float(np.linalg.norm(self.plant.base_origin_velocity())),
                          "support_margin_m": float(self.side.margin),
                          "contact_dwell_s": float(self.side.contact_dwell), "pending_leg": self.side.leg,
                          "torque_within_limits": bool(np.all(np.abs(torque) <= self.plant.actuator_torque_limit_nm + 1e-10))})
            return status, torque, info

    class ScriptedViewer:
        def __init__(self, model, data, supplied_commands, **kwargs):
            self.commands = supplied_commands
            self.events, self.previous = [], set()
            self.cam = type("Camera", (), {"lookat": np.zeros(3)})()

        def emit(self, key, action, seconds):
            self.events.append({"type": "key", "key": key, "action": action,
                                "simulation_time_s": float(simulation.plant.data.time), "script_time_s": seconds})
            self.commands.handle_key_event(key, action)

        def poll(self, seconds):
            nonlocal cancel_time
            clock[0] = seconds
            tick, kind = round(seconds/.01), case["kind"]
            keys = set()
            if kind == "tap" and 200 <= tick < 300:
                keys.add(ord("A") if case["direction"] > 0 else ord("D"))
            elif kind in ("held", "reverse") and tick >= 200 and len(cycles) < 2:
                keys.add(ord("D") if kind == "reverse" and tick >= 300 else ord("A"))
            elif kind.startswith("cancel") and tick >= 200:
                if cancel_time is None or seconds < cancel_time+3.:
                    keys.add(ord("A"))
                phase = box["drive"].side.phase
                trigger = phase == "shift" and tick >= 300 if kind == "cancel_shift" else phase == "swing"
                if cancel_time is None and box["drive"].active and trigger:
                    cancel_time = seconds
                    self.emit(ord("X"), 1, seconds)
                    self.emit(ord("X"), 0, seconds)
            elif kind == "jump":
                if tick == 200:
                    self.emit(32, 1, seconds)
                    self.emit(32, 0, seconds)
                if tick == 210:
                    self.emit(ord("A"), 1, seconds)
                    self.emit(ord("A"), 0, seconds)
            elif kind == "reset":
                if 200 <= tick < 500 or 800 <= tick < 900:
                    keys.add(ord("A"))
                if tick == 300:
                    self.emit(ord("R"), 1, seconds)
                    self.emit(ord("R"), 0, seconds)
            for key in sorted(self.previous-keys):
                self.emit(key, 0, seconds)
            for key in sorted(keys-self.previous):
                self.emit(key, 1, seconds)
            self.previous = keys
            self.commands.update_pressed(keys, focused=True)

        def reset_camera(self): pass
        def is_running(self): return True
        def lock(self): return nullcontext()
        def set_status(self, *_): pass
        def sync(self): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass

    def write_actual_protocol(path, payload):
        if Path(path).name == "protocol.json":
            payload = {**payload, "validation_harness_sha256": sha(Path(__file__)),
                       "input_driver": "ScriptedViewer: injected events and held polling; no window",
                       "control_observer": "ObservedDrive delegates unchanged step then only records measured values",
                       "manual_gui_tested": False}
        original_json(path, payload)

    runner.CourseSideStepDrive = ObservedDrive
    runner._json = write_actual_protocol
    try:
        summary = runner.run(simulation, output, seconds=case["duration"], zone=case["zone"],
                             commands=commands, viewer_factory=ScriptedViewer, realtime=False,
                             side_step_profile="fast", brake_reference_mode="legacy",
                             side_input_mode="commit_cycle_v1")
        assert summary["source_unchanged"]
        with (output/"physical_side_trace.jsonl").open("x") as stream:
            for row in trace:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
        result = analyse(output, case, trace, cycles, cancel_time)
        write_json(output/"physical_gate.json", result)
        return result
    finally:
        runner.CourseSideStepDrive = original
        runner._json = original_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", choices=[c["name"] for c in CASES])
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    sys.path.insert(0, str(repo))
    from scripts import run_d1_course_drive as runner
    from scripts.d1_course_controls import CourseKeyboardCommands
    from wheel_legged_control.d1.interactive import D1InteractiveSimulation
    cases = CASES if args.cases is None else [c for c in CASES if c["name"] in args.cases]
    files = [Path(__file__).resolve(), *sorted((repo/"scripts").glob("*.py")),
             *sorted((repo/"src").rglob("*.py"))]
    hashes = {str(path): sha(path) for path in files}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"protocol.json", {
        "schema": "d1-committed-side-input-physical-validation-v1", "cases": cases,
        "side_input_mode": "commit_cycle_v1", "source_sha256": hashes,
        "physical_parameters_changed": False, "manual_gui_tested": False,
        "input": "Scripted GLFW-equivalent PRESS/RELEASE events plus actual held polling feed the real course runner.",
        "gates": {"tap_seconds_max": 15., "signed_lateral_m": [.025, .05],
                  "handoff_seconds": 4., "retention_fraction_min": .9, "rollback_m_max": .005},
        "failure_rule": "Keep every failed gate and all recorded physical states; no gait tuning during this matrix.",
    })
    results = []
    for case in cases:
        try:
            result = run_case(runner, D1InteractiveSimulation, CourseKeyboardCommands,
                              output/case["name"], case)
            results.append(result)
            print(json.dumps({"case": case["name"], "all_passed": result["all_passed"],
                              "failed_checks": result["failed_checks"], "cycles": result["cycles"]}), flush=True)
        except Exception as error:  # noqa: BLE001 - preserve each failed case and continue the fixed matrix
            write_json(output/f"{case['name']}_harness_failure.json", {
                "type": type(error).__name__, "message": str(error)})
            results.append({"case": case, "all_passed": False, "harness_error": str(error)})
            print(json.dumps({"case": case["name"], "harness_error": str(error)}), flush=True)
        gc.collect()
    unchanged = all(sha(path) == digest for path, digest in hashes.items())
    write_json(output/"summary.json", {"results": results, "all_cases_passed": all(r["all_passed"] for r in results),
               "source_unchanged": unchanged, "manual_gui_tested": False})
    write_json(output/"manifest.json", {str(path.relative_to(output)): {"sha256": sha(path), "bytes": path.stat().st_size}
               for path in sorted(output.rglob("*")) if path.is_file()})
    assert unchanged, "source changed during validation"


if __name__ == "__main__":
    main()
