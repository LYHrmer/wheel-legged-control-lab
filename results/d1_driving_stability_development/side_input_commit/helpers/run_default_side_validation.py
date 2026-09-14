"""Confirm the final default through the actual runner and block Space during a cycle."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

import run_side_intent_validation as matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    sys.path.insert(0, str(repo))
    from scripts import run_d1_course_drive as runner
    from scripts.d1_course_controls import CourseKeyboardCommands
    from wheel_legged_control.d1.interactive import D1InteractiveSimulation

    actual_run = runner.run
    assert inspect.signature(actual_run).parameters["side_input_mode"].default == "commit_cycle_v1"
    sources = [Path(__file__), Path(matrix.__file__), *sorted((repo/"scripts").glob("*.py")),
               *sorted((repo/"src").rglob("*.py"))]
    before = {str(p): matrix.sha(p) for p in sources}
    output.mkdir(parents=True, exist_ok=False)
    matrix.write_json(output/"protocol.json", {
        "schema": "d1-final-default-side-input-verification-v1", "source_sha256": before,
        "manual_gui_tested": False, "case": matrix.CASES[0],
        "default_argument_used": True, "extra_event": "Space PRESS/RELEASE at t=3.50 s while the side cycle is active",
    })

    def use_default(*args, **kwargs):
        assert kwargs.pop("side_input_mode") == "commit_cycle_v1"
        original_viewer = kwargs["viewer_factory"]
        class SpaceDuringCycle(original_viewer):
            def poll(self, seconds):
                super().poll(seconds)
                if round(seconds/.01) == 350:
                    self.emit(32, 1, seconds)
                    self.emit(32, 0, seconds)
        kwargs["viewer_factory"] = SpaceDuringCycle
        return actual_run(*args, **kwargs)

    runner.run = use_default
    try:
        result = matrix.run_case(runner, D1InteractiveSimulation, CourseKeyboardCommands,
                                 output/"tap_A", matrix.CASES[0])
    finally:
        runner.run = actual_run
    events = [json.loads(line) for line in (output/"tap_A/keyboard_events.jsonl").read_text().splitlines()]
    summary = json.loads((output/"tap_A/summary.json").read_text())
    checks = {
        "all_tap_gates_passed": result["all_passed"],
        "default_mode_recorded": summary["side_input_mode"] == "commit_cycle_v1",
        "one_space_event_blocked": sum(e["type"] == "jump_blocked" for e in events) == 1,
        "no_jump_queued_after_cycle": summary["jump_phases_observed"] == ["ready"],
        "source_unchanged": all(matrix.sha(p) == digest for p, digest in before.items()),
    }
    matrix.write_json(output/"summary.json", {"checks": checks, "all_passed": all(checks.values()),
                      "steps": summary["steps"], "result": result, "manual_gui_tested": False})
    matrix.write_json(output/"manifest.json", {str(p.relative_to(output)): {"sha256": matrix.sha(p), "bytes": p.stat().st_size}
                      for p in sorted(output.rglob("*")) if p.is_file()})
    print(json.dumps(checks))
    assert all(checks.values())


if __name__ == "__main__":
    main()
