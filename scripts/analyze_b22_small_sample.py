"""Reproducible small-sample diagnostics for the B22 residual fine-tuning study.

Reads saved checkpoints and saved control records only: no model construction, no simulator,
no training. Produces the numbers used by the methodology section: exploration noise (log_std),
residual action magnitude versus the action range, saturation fraction, and the ratio of the
policy mean to its exploration noise.
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import math
import pickle
import struct
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ("flat_0p6", "flat_1p6", "flat_1p2_yaw", "bumps_0p4", "rough_0p35", "ramp_0p45_complete")
CHECKPOINTS = {
    "B22 parent": ROOT / "runtime/d1_b22/checkpoint/final_model.zip",
    "44 ffyaw (s201011)": ROOT
    / "results/b22_yawtrain44_20261018_02/train_ffyaw/final_checkpoint/final_model.zip",
    "46 ffsse (s201011)": ROOT
    / "results/b22_logsse46_20261018_01/train_ffsse/final_checkpoint/final_model.zip",
    "47 ffyaw (s201021)": ROOT
    / "results/b22_seed47_20261018_02/train_ffyaw_s21/final_checkpoint/final_model.zip",
    "47 ffsse (s201021)": ROOT
    / "results/b22_seed47_20261018_02/train_ffsse_s21/final_checkpoint/final_model.zip",
}
EVAL_RUNS = {
    "ffyaw_s21": ROOT / "results/b22_yawff44_arm_ffyaw_s21_20261018_12/FF3.9",
    "ffsse_s21": ROOT / "results/b22_yawff44_arm_ffsse_s21_20261018_12/FF3.9",
}


class _Stub:
    def __init__(self, *args, **kwargs):
        pass


def _rebuild(storage, offset, size, stride, *rest):
    return ("tensor", storage, offset, size)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "_rebuild_tensor_v2":
            return _rebuild
        return (
            collections.OrderedDict if (module, name) == ("collections", "OrderedDict") else _Stub
        )

    def persistent_load(self, pid):
        return pid


def log_std(path: Path) -> dict:
    """Parse ``policy.pth`` without importing torch (torch does not fit the memory budget)."""
    with zipfile.ZipFile(path) as outer:
        import io

        inner = zipfile.ZipFile(io.BytesIO(outer.read("policy.pth")))
        root = inner.namelist()[0].split("/")[0]
        state = _Unpickler(io.BytesIO(inner.read(root + "/data.pkl"))).load()
        tensor = state["log_std"]
        key, offset, count = tensor[1][2], tensor[2], tensor[3][0]
        raw = inner.read(f"{root}/data/{key}")
    values = struct.unpack_from(f"<{count}f", raw, 4 * offset)
    std = [math.exp(v) for v in values]
    return {
        "leg_std_mean": sum(std[:12]) / 12,
        "wheel_std_mean": sum(std[12:]) / 4,
        "std_min": min(std),
        "std_max": max(std),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "log_std_sha256": hashlib.sha256(struct.pack(f"<{count}f", *values)).hexdigest(),
    }


def drive_windows() -> dict:
    readback = json.loads(
        (ROOT / "results/b22_symmetry43_physics_20261018_01/readback.json").read_text()
    )
    return {
        c["case_id"]: c["rl_terms"]["drive_window"]
        for c in readback["cases"]
        if c["actor"] == "zero"
    }


def action_stats(run: Path, case: str, window: tuple[int, int]) -> dict:
    """Residual action magnitude over one case's drive window, both actor forms."""
    rows = []
    start, stop = window
    for actor in ("raw42", "mirror43"):
        folder = run / "heldout" / f"{case}_{actor}"
        receipt = json.loads((folder / "case_receipt.json").read_text())
        for block in receipt["controller_record_blocks"]:
            with gzip.open(folder / block["file"], "rt") as stream:
                for line in stream:
                    record = json.loads(line)
                    if start <= record["tick"] < stop:
                        rows.append(record["policy_input_action"])
    legs = [v for row in rows for v in row[:12]]
    wheels = [v for row in rows for v in row[12:]]

    def stats(values):
        return {
            "mean_abs": sum(abs(v) for v in values) / len(values),
            "rms": math.sqrt(sum(v * v for v in values) / len(values)),
            "saturation_fraction": sum(abs(v) >= 0.999 for v in values) / len(values),
        }

    per_tick_leg_rms = [math.sqrt(sum(v * v for v in row[:12]) / 12) for row in rows]
    return {
        "drive_ticks": len(rows),
        "leg": stats(legs),
        "wheel": stats(wheels),
        "mean_leg_residual_rms_per_tick": sum(per_tick_leg_rms) / len(per_tick_leg_rms),
        "max_leg_residual_rms_per_tick": max(per_tick_leg_rms),
    }


def main() -> int:
    windows = drive_windows()
    result = {
        "schema": "b22-small-sample-diagnostics-v1",
        "new_model_calls": 0,
        "new_physics_steps": 0,
        "checkpoints": {name: log_std(path) for name, path in CHECKPOINTS.items()},
        "residual_actions": {},
    }
    for name, run in EVAL_RUNS.items():
        total = 0
        per_case = {}
        merged = {
            "leg": {"mean_abs": 0.0, "rms": 0.0, "saturation_fraction": 0.0},
            "wheel": {"mean_abs": 0.0, "rms": 0.0, "saturation_fraction": 0.0},
        }
        for case in CASES:
            stats = action_stats(run, case, windows[case])
            per_case[case] = stats
            total += stats["drive_ticks"]
            for key in ("leg", "wheel"):
                for field in ("mean_abs", "rms", "saturation_fraction"):
                    merged[key][field] += stats[key][field] * stats["drive_ticks"]
        for key, values in merged.items():
            for field in values:
                values[field] /= total
        merged["mean_leg_residual_rms_per_tick"] = (
            sum(s["mean_leg_residual_rms_per_tick"] * s["drive_ticks"] for s in per_case.values())
            / total
        )
        merged["max_leg_residual_rms_per_tick"] = max(
            s["max_leg_residual_rms_per_tick"] for s in per_case.values()
        )
        result["residual_actions"][name] = {
            "drive_ticks": total,
            "pooled": merged,
            "per_case": {
                c: {k: v for k, v in s.items() if k != "drive_ticks"} for c, s in per_case.items()
            },
        }
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v["std_min"] for k, v in result["checkpoints"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
