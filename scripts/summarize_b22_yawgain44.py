"""Compact formal-reader summary of the yaw-feedforward exploration versus the 43 zero baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

CASE_FIELDS = ("sse_total", "sse_vx", "sse_yaw", "torque_cost_sum", "drive_completed_ticks")


def summarize(run: Path, setting: str, reference: Path) -> dict:
    new = json.loads((run / setting / "readback.json").read_text())["cases"]
    old = {
        c["case_id"]: c
        for c in json.loads((reference / "readback.json").read_text())["cases"]
        if c["actor"] == "zero"
    }
    tasks, sse_new, sse_old, cost_new, cost_old = {}, 0.0, 0.0, 0.0, 0.0
    for case in new:
        base = old[case["case_id"]]
        a, b = case["rl_terms"], base["rl_terms"]
        tasks[case["case_id"]] = {
            "candidate": {k: a[k] for k in CASE_FIELDS},
            "zero_baseline": {k: b[k] for k in CASE_FIELDS},
            "task_passed": case["task_passed"],
            "stopping_passed": case["stopping"]["passed"],
            "safety_passed": case["safety"]["passed"],
            "time_to_settle_s": case["stopping"].get("time_to_settle_s"),
            "sse_ratio": a["sse_total"] / b["sse_total"],
        }
        sse_new += a["sse_total"]
        sse_old += b["sse_total"]
        cost_new += a["torque_cost_sum"]
        cost_old += b["torque_cost_sum"]
    return {
        "schema": "wheel44-yaw-feedforward-explore-summary-v1",
        "run": str(run),
        "setting": setting,
        "reference_run": str(reference),
        "pooled_sse_candidate": sse_new,
        "pooled_sse_zero_baseline": sse_old,
        "pooled_sse_ratio": sse_new / sse_old,
        "torque_cost_sum_ratio": cost_new / cost_old,
        "all_tasks_passed": all(t["task_passed"] for t in tasks.values()),
        "all_stopping_passed": all(t["stopping_passed"] for t in tasks.values()),
        "all_safety_passed": all(t["safety_passed"] for t in tasks.values()),
        "tasks": tasks,
        "scope": "single seed per task; feedforward gain identified on the same turning task; baseline-controller change, not an RL contribution",
    }


if __name__ == "__main__":
    run, setting, reference, output = sys.argv[1:5]
    Path(output).write_text(
        json.dumps(summarize(Path(run), setting, Path(reference)), indent=2) + "\n"
    )
