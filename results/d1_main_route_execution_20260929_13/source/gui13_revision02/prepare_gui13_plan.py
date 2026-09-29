"""Prepare a reviewable, non-executable GUI13 draft plan; no engine import."""
from __future__ import annotations

import argparse
from pathlib import Path

from gui13_contract import HERE, PLAN_SCHEMA, SEED, verify_evidence
from launch_gui13 import save_exclusive, sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_id = args.run_id
    if (not run_id or len(run_id) > 48 or
            any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in run_id)):
        raise ValueError("run ID must be a short lowercase/number/underscore token")
    output = args.output.resolve()
    if output.parent != HERE:
        raise ValueError("draft plan must be stored in gui13")
    arms = {}
    for arm in ("headless", "gui"):
        name = f"{run_id}_{arm}"
        arms[arm] = {
            "arm": arm, "actor": "final_policy", "terrain": "flat",
            "seed": SEED, "control_limit": 600, "normal_native_limit": 3000,
            "compiler_native_limit": 2, "wallclock_soft_s": 95,
            "archive_grace_s": 20, "cleanup_grace_s": 5,
            "wallclock_hard_s": 120, "retry_permitted": False,
            "preflight_wall_s": 240, "postcheck_wall_s": 180,
            "outer_host_wall_s": 600,
            "output_directory": str(HERE / name),
            "reservation_path": str(HERE / (name + "_reservation.json")),
        }
    save_exclusive(output, {
        "schema": PLAN_SCHEMA, "decision": "PENDING",
        "review_note": "Change decision to GO only after independent source review",
        "run_id": run_id, "retry_permitted": False,
        "profile": "flat_0p6_final_policy_development_600",
        "original_1600_task_replayed": False,
        "logical_input_source": "scripted_not_hardware_keyboard",
        "evidence": verify_evidence(), "inputs": sources(), "arms": arms,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
