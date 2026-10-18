"""Compact summary of the completed B22 symmetry43 physics run.

Reads only the saved terminal JSON files and resources.jsonl; runs no model and no physics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ACTORS = ("zero", "raw42", "mirror43")
SCALARS = ("sse_total", "sse_vx", "sse_yaw", "torque_cost_sum", "drive_completed_ticks")


def sha256(path: Path) -> dict:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def actor_of(case: dict) -> str:
    return "zero" if case["actor"] == "zero" else case["experiment_actor"]


def resource_summary(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    owned = [r["owned"] for r in rows if r.get("owned")]
    events = owned[-1]["events"]
    return {
        "samples": len(rows),
        "min_effective_available_bytes": min(r["effective_available_bytes"] for r in rows),
        "max_owned_current_bytes": max(o["current_bytes"] for o in owned),
        "owned_peak_bytes": max(o.get("peak_bytes") or 0 for o in owned),
        "final_cgroup_events": events,
    }


def summarize(run: Path) -> dict:
    readback = json.loads((run / "readback.json").read_text())
    report = json.loads((run / "report.json").read_text())
    tasks = {}
    for case in readback["cases"]:
        terms = case["rl_terms"]
        entry = tasks.setdefault(
            case["case_id"], {"terrain": case["terrain"], "seed": case["seed"]}
        )
        entry[actor_of(case)] = {
            **{k: terms[k] for k in SCALARS},
            "task_passed": case["task_passed"],
            "completed_controls": case["completed_controls"],
            "stopping_passed": case["stopping"]["passed"],
            "time_to_settle_s": case["stopping"].get("time_to_settle_s"),
            "max_native_forward_excursion_m": case["stopping"].get(
                "max_native_forward_excursion_m"
            ),
            "safety_passed": case["safety"]["passed"],
        }
    for entry in tasks.values():
        for actor in ("raw42", "mirror43"):
            entry[actor]["sse_ratio_to_zero"] = (
                entry[actor]["sse_total"] / entry["zero"]["sse_total"]
            )
    pooled = {}
    for actor in ACTORS:
        sse = sum(t[actor]["sse_total"] for t in tasks.values())
        pooled[actor] = {
            "sse_vx": sum(t[actor]["sse_vx"] for t in tasks.values()),
            "sse_yaw": sum(t[actor]["sse_yaw"] for t in tasks.values()),
            "sse_total": sse,
        }
    for actor in ACTORS:
        pooled[actor]["ratio_to_zero"] = pooled[actor]["sse_total"] / pooled["zero"]["sse_total"]
    gates = {}
    for actor in ("raw42", "mirror43"):
        c = readback["contribution"][actor]
        gates[actor] = {
            "RL_contribution_passed": c["RL_contribution_passed"],
            "all_six_policy_task_qualified": c["all_six_policy_task_qualified"],
            "cost_ratio": c["cost_gate"]["cost_ratio"],
            "cost_gate_passed": c["cost_gate"]["passed"],
            "pooled_sse_ratio": c["branch_b_sse_improvement"]["pooled_sse_ratio"],
            "pairs_no_worse_within_1p02": len(
                c["branch_b_sse_improvement"]["pairs_no_worse_within_1p02"]
            ),
            "branch_b_reasons": c["branch_b_sse_improvement"]["reasons"],
        }
    floors = {
        actor: {
            k: f[k]
            for k in (
                "case_id",
                "numeric_gate_passed",
                "completed_controls",
                "last_drive_vx_rms_error_mps",
                "last_drive_yaw_rms_rps",
                "max_abs_world_roll_deg",
                "max_abs_world_pitch_deg",
            )
        }
        for actor, f in readback["floors"].items()
    }
    return {
        "schema": "wheel43-small-saved-summary-v1",
        "run": str(run),
        "status": readback["status"],
        "contract": readback["execution_contract_id"],
        "source_commit": report["source"]["commit"],
        "identities": {
            name: sha256(run / name)
            for name in ("report.json", "worker_report.json", "readback.json")
        },
        "counts": readback["counts"],
        "worker_elapsed_s": report["worker_elapsed_s"],
        "read_elapsed_s": report["read_elapsed_s"],
        "elapsed_s": report["elapsed_s"],
        "cleanup_complete": report["cleanup_complete"],
        "pooled": pooled,
        "gates": gates,
        "tasks": tasks,
        "floors": floors,
        "resources": resource_summary(run / "resources.jsonl"),
        "automatic_model_replacement": readback["automatic_model_replacement"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.run), indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
