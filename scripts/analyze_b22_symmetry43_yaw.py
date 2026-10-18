"""Read-only yaw diagnostic over the saved symmetry43 run and the 42 training archive.

Streams saved control records; runs no model and no physics. Output is descriptive: it
quantifies how much yaw-differential wheel residual each actor applied, the steady yaw gain
of the closed loop, and how often the 42 training run sampled non-zero yaw commands.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import resource
from pathlib import Path

ACTORS = ("zero", "raw42", "mirror43")
CASES = ("flat_1p2_yaw", "rough_0p35")
YAW_SCALE = 0.4


def records(folder: Path):
    receipt = json.loads((folder / "case_receipt.json").read_text())
    for block in receipt["controller_record_blocks"]:
        with gzip.open(folder / block["file"], "rt") as stream:
            for line in stream:
                yield json.loads(line)


def rms(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else float("nan")


def case_stats(run: Path, windows: dict, case: str, actor: str) -> dict:
    start, stop = windows[case, actor]
    cmd, yaw, diff, common = [], [], [], []
    for rec in records(run / "heldout" / f"{case}_{actor}"):
        if not start <= rec["tick"] < stop:
            continue
        info = rec["info"]
        wheels = info["controller_record"]["calculation"]["wheel_action_offset_rad_s"]
        cmd.append(info["consumed_command"]["yaw_rate_rps"])
        yaw.append(info["metrics"]["body_yaw_rate_rps"])
        diff.append((wheels[1] + wheels[3] - wheels[0] - wheels[2]) / 2)
        common.append(sum(wheels) / 4)
    err = [c - y for c, y in zip(cmd, yaw, strict=True)]
    mc, my = sum(cmd) / len(cmd), sum(yaw) / len(yaw)
    var = sum((c - mc) ** 2 for c in cmd)
    gain = sum((c - mc) * (y - my) for c, y in zip(cmd, yaw, strict=True)) / var if var else None
    split = {}
    if gain is not None:
        offset = my - gain * mc
        # y = gain * command + offset + residual; the least-squares split is exact (cross term 0)
        gap = sum((((1 - gain) * c - offset) / YAW_SCALE) ** 2 for c in cmd)
        resid = sum(
            ((y - gain * c - offset) / YAW_SCALE) ** 2 for c, y in zip(cmd, yaw, strict=True)
        )
        split = {"sse_gain_offset_part": gap, "sse_residual_part": resid}
    return {
        "ticks": len(cmd),
        "sse_yaw": sum((e / YAW_SCALE) ** 2 for e in err),
        "yaw_command_rms": rms(cmd),
        "yaw_measured_rms": rms(yaw),
        "closed_loop_yaw_gain_regression": gain,
        **split,
        "wheel_yaw_differential_offset_rms_rad_s": rms(diff),
        "wheel_common_offset_rms_rad_s": rms(common),
    }


def training_yaw_histogram(training: Path) -> dict:
    bins: collections.Counter = collections.Counter()
    total = small = large = 0
    for path in sorted(training.glob("full_control_records_*.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            for line in stream:
                yaw = json.loads(line)["info"]["consumed_command"]["yaw_rate_rps"]
                bins[round(round(abs(yaw) * 20) / 20, 2)] += 1
                total += 1
                small += abs(yaw) < 0.025
                large += abs(yaw) >= 0.1
    return {
        "controls": total,
        "abs_yaw_command_bins_0p05": {str(k): bins[k] for k in sorted(bins)},
        "controls_abs_yaw_below_0p025": small,
        "controls_abs_yaw_at_least_0p1": large,
        "fraction_abs_yaw_below_0p025": small / total,
        "fraction_abs_yaw_at_least_0p1": large / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("physics_run", type=Path)
    parser.add_argument("training_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
    readback = json.loads((args.physics_run / "readback.json").read_text())
    windows = {
        (c["case_id"], "zero" if c["actor"] == "zero" else c["experiment_actor"]): c["rl_terms"][
            "drive_window"
        ]
        for c in readback["cases"]
    }
    result = {
        "schema": "wheel43-yaw-diagnostic-v1",
        "new_nn_calls": 0,
        "new_physics_steps": 0,
        "cases": {
            case: {actor: case_stats(args.physics_run, windows, case, actor) for actor in ACTORS}
            for case in CASES
        },
        "training_deficit8_yaw_commands": training_yaw_histogram(args.training_dir),
        "limitations": "Descriptive saved-data arithmetic; does not establish causal benefit or a counterfactual closed loop.",
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
