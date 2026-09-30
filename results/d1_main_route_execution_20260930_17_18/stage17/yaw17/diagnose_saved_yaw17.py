"""Offline C17 yaw diagnosis from the frozen C15/C16 records only.

No project modules, policy checkpoints, MuJoCo model, or physics are imported.
Every comparison is descriptive: the three actors generated different states.
The output is a new exclusive JSON file; all source files are read only.
"""

from __future__ import annotations

import argparse
import builtins
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time


_ORIGINAL_IMPORT = builtins.__import__
_FORBIDDEN_IMPORTS = frozenset(("torch", "mujoco", "stable_baselines3", "gym", "gymnasium"))


def _guard_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in _FORBIDDEN_IMPORTS:
        raise RuntimeError(f"model/physics dependency import denied: {name}")
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


builtins.__import__ = _guard_import
import numpy as np


W = Path("/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260924_full_drive01")
C15 = W / "continuation15"
C16 = W / "continuation16" / "eval_01" / "heldout"
ACTORS = ("zero", "global_continue", "grouped_continue")
WHEELS = (3, 7, 11, 15)  # FL, FR, RL, RR
WINDOWS = {
    "full_drive_175_815": (175, 815),
    "hold": (415, 815),
    "positive_1": (415, 515),
    "negative": (515, 715),
    "positive_2": (715, 815),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value, shape=None):
    array = np.asarray(value, dtype=np.float64)
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError(f"invalid finite array: expected {shape}, got {array.shape}")
    return array


def differential(four):
    """Right minus left in FL, FR, RL, RR order."""
    return (four[..., 1] + four[..., 3] - four[..., 0] - four[..., 2]) / 2.0


def stats(values):
    x = finite(values)
    if x.ndim != 1 or not len(x):
        raise ValueError("empty or non-vector statistic")
    return {
        "mean": float(np.mean(x)), "rms": float(np.sqrt(np.mean(x * x))),
        "mean_abs": float(np.mean(np.abs(x))), "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def record_input(path, hashes):
    path = path.resolve(strict=True)
    if not path.is_file() or path in hashes:
        raise ValueError(f"missing or duplicate input {path}")
    hashes[path] = sha256(path)
    return path


def read_json(path, hashes):
    return json.loads(record_input(path, hashes).read_text())


def read_case(actor, hashes):
    folder = C16 / f"flat_1p2_yaw_{actor}"
    schedule = read_json(folder / "schedule.json", hashes)
    receipt = read_json(folder / "case_receipt.json", hashes)
    if (schedule["actor"] != actor or schedule["case_id"] != "flat_1p2_yaw"
            or schedule["control_cap"] != 1600 or receipt["completed_controls"] != 1600
            or receipt["raw_command_sha256"] != schedule["raw_command_sha256"]
            or not receipt["record_valid"]):
        raise ValueError(f"invalid frozen case identity or completeness: {actor}")
    if len(schedule["raw_commands"]) != 1600:
        raise ValueError("short raw command schedule")
    with np.load(record_input(folder / "states.npz", hashes), allow_pickle=False) as states:
        qvel = finite(states["qvel"])
    if qvel.shape[0] != 1601 or qvel.shape[1] <= 5:
        raise ValueError(f"invalid post-control qvel shape {qvel.shape}")
    values = {key: [] for key in (
        "raw_yaw", "servo_yaw", "actual_yaw", "effective_nominal_yaw",
        "raw_wheel_diff", "applied_wheel_diff", "wheel_offset_diff",
        "nominal_target_diff", "wheel_target_diff", "wheel_speed_diff",
        "wheel_error_diff", "base_torque_diff", "request_torque_diff",
        "safe_torque_diff", "wheel_target_clipped", "wheel_torque_clipped",
        "wheel_protected", "wheel_integral_not_committed", "action_disabled",
    )}
    seen = 0
    previous_servo_yaw = 0.0
    for block in receipt["controller_record_blocks"]:
        name = block["file"]
        if not name.startswith("control_records_") or not name.endswith(".jsonl.gz"):
            raise ValueError("unexpected control block path")
        path = record_input(folder / name, hashes)
        rows = 0
        with gzip.open(path, "rt") as source:
            for line in source:
                row = json.loads(line)
                tick = row["tick"]
                if tick != seen or row["actor"] != actor:
                    raise ValueError("nonsequential or mislabeled heldout row")
                info = row["info"]
                ctrl = info["controller_record"]
                calc = ctrl["calculation"]
                raw = info["raw_operator_command"]
                if raw != schedule["raw_commands"][tick]:
                    raise ValueError("raw command differs from frozen schedule")
                servo = float(info["consumed_command"]["yaw_rate_rps"])
                actual = float(qvel[tick + 1, 5])  # C16 scorer uses post-control qvel[5].
                expected_servo = previous_servo_yaw + float(np.clip(
                    float(raw["yaw_rate_rps"]) - previous_servo_yaw, -0.006, 0.006))
                if (abs(actual - float(info["metrics"]["body_yaw_rate_rps"])) > 1e-6
                        or abs(servo - float(ctrl["servo_yaw_rps"])) > 1e-10
                        or abs(servo - expected_servo) > 1e-10
                        or abs(float(raw["yaw_rate_rps"]) - float(ctrl["raw_yaw_rps"])) > 1e-10):
                    raise ValueError("state, servo, or raw yaw disagree with control record")
                previous_servo_yaw = servo
                raw_action = finite(calc["raw_action"], (16,))
                applied_action = finite(calc["applied_action"], (16,))
                wheel_offset = finite(calc["wheel_action_offset_rad_s"], (4,))
                nominal = finite(calc["consumed_nominal_wheel_speed_rad_s"], (4,))
                target = finite(calc["wheel_speed_target_rad_s"], (4,))
                omega = finite(calc["consumed_wheel_omega_rad_s"], (4,))
                error = finite(calc["wheel_speed_error_rad_s"], (4,))
                if (not np.allclose(wheel_offset, 4.0 * applied_action[12:], atol=1e-12, rtol=0)
                        or not np.allclose(error, target - omega, atol=1e-10, rtol=0)
                        or bool(ctrl["action_gate_enabled"]) != bool(calc["action_enabled"])):
                    raise ValueError("action or wheel stage is inconsistent")
                values["raw_yaw"].append(float(raw["yaw_rate_rps"]))
                values["servo_yaw"].append(servo)
                values["actual_yaw"].append(actual)
                values["effective_nominal_yaw"].append(float(ctrl["nominal_effective_yaw_rps"]))
                values["raw_wheel_diff"].append(float(differential(raw_action[12:])))
                values["applied_wheel_diff"].append(float(differential(applied_action[12:])))
                values["wheel_offset_diff"].append(float(differential(wheel_offset)))
                values["nominal_target_diff"].append(float(differential(nominal)))
                values["wheel_target_diff"].append(float(differential(target)))
                values["wheel_speed_diff"].append(float(differential(omega)))
                values["wheel_error_diff"].append(float(differential(error)))
                for source_name, dest in (("base_wheel_torque_nm", "base_torque_diff"),
                                          ("final_request_torque_nm", "request_torque_diff"),
                                          ("safe_torque_nm", "safe_torque_diff")):
                    values[dest].append(float(differential(finite(calc[source_name], (16,))[list(WHEELS)])))
                for source_name, dest in (("wheel_speed_target_clip_mask", "wheel_target_clipped"),
                                          ("torque_clip_mask", "wheel_torque_clipped"),
                                          ("protection_changed_mask", "wheel_protected")):
                    mask = np.asarray(calc[source_name], dtype=bool)
                    if mask.shape != ((4,) if source_name == "wheel_speed_target_clip_mask" else (16,)):
                        raise ValueError("invalid mask shape")
                    values[dest].append(bool(np.any(mask if len(mask) == 4 else mask[list(WHEELS)])))
                commits = np.asarray(calc["wheel_integral_commit_mask"], dtype=bool)
                if commits.shape != (4,):
                    raise ValueError("invalid PI commit mask")
                values["wheel_integral_not_committed"].append(bool(np.any(~commits)))
                values["action_disabled"].append(not bool(calc["action_enabled"]))
                seen += 1
                rows += 1
        if rows != block["rows"]:
            raise ValueError("record block row count differs from receipt")
    if seen != 1600:
        raise ValueError(f"case has {seen} of 1600 control records")
    return {key: np.asarray(value) for key, value in values.items()}


def describe_window(series, begin, end, selected=None):
    indices = np.arange(begin, end)
    if selected is not None:
        selected = np.asarray(selected, dtype=bool)
        if selected.shape != (end - begin,):
            raise ValueError("window selection shape mismatch")
        indices = indices[selected]
    if not len(indices):
        return {"ticks": [begin, end], "count": 0, "available": False}
    error = series["actual_yaw"][indices] - series["servo_yaw"][indices]
    nominal = series["nominal_target_diff"][indices]
    offset = series["wheel_offset_diff"][indices]
    opposed = (np.abs(nominal) > 0.1) & (offset * nominal < 0)
    output = {
        "ticks": [begin, end], "count": len(indices), "available": True,
        "signed_yaw_error_rps": stats(error),
        "yaw_squared_error_sum": float(np.sum(error * error)),
        "yaw_squared_error_share_of_hold": (float(np.sum(error * error) / np.sum(
            np.square(series["actual_yaw"][415:815] - series["servo_yaw"][415:815])))
            if begin >= 415 and end <= 815 else None),
        "nominal_diff_opposed_by_effective_residual_fraction": float(np.mean(opposed)),
        "effective_nominal_yaw_at_0p6_limit_fraction": float(np.mean(
            np.abs(series["effective_nominal_yaw"][indices]) >= 0.6 - 1e-10)),
        "servo_equals_raw_yaw_fraction": float(np.mean(np.abs(
            series["servo_yaw"][indices] - series["raw_yaw"][indices]) <= 1e-9)),
    }
    for key in ("raw_yaw", "servo_yaw", "actual_yaw", "effective_nominal_yaw",
                "raw_wheel_diff", "applied_wheel_diff", "wheel_offset_diff",
                "nominal_target_diff", "wheel_target_diff", "wheel_speed_diff",
                "wheel_error_diff", "base_torque_diff", "request_torque_diff", "safe_torque_diff"):
        output[key] = stats(series[key][indices])
    for key in ("wheel_target_clipped", "wheel_torque_clipped", "wheel_protected",
                "wheel_integral_not_committed", "action_disabled"):
        output[key + "_fraction"] = float(np.mean(series[key][indices]))
    return output


def training_exposure(actor, hashes):
    folder = C15 / f"{actor.split('_')[0]}_01" / "training"
    numeric = folder / "training_numeric_blocks"
    arrays = {key: [] for key in ("control_index", "episode_index", "episode_tick",
        "raw_command_vx_vy_yaw_clearance", "servo_command_vx_vy_yaw_clearance",
        "policy_env_input16", "policy_clipped16", "effective_action16",
        "body_yaw_rate_rps")}
    for block_id in range(16):
        path = record_input(numeric / f"controls_{block_id:04d}.npz", hashes)
        with np.load(path, allow_pickle=False) as block:
            for key in arrays:
                arrays[key].append(np.array(block[key]))
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    if not np.array_equal(arrays["control_index"], np.arange(16384)):
        raise ValueError("training numeric ledger is not complete 0..16383")
    schedule_map = {}
    for episode in np.unique(arrays["episode_index"]):
        schedule = read_json(folder / f"training_episode_{int(episode):06d}_schedule.json", hashes)
        if schedule["episode_index"] != int(episode):
            raise ValueError("training schedule episode mismatch")
        schedule_map[int(episode)] = schedule
    for episode, schedule in schedule_map.items():
        mask = arrays["episode_index"] == episode
        ticks = arrays["episode_tick"][mask]
        expected = np.asarray([[row[k] for k in ("forward_velocity_mps", "lateral_velocity_mps",
            "yaw_rate_rps", "clearance_m")] for row in schedule["raw_commands"]], dtype=float)
        if not np.array_equal(arrays["raw_command_vx_vy_yaw_clearance"][mask], expected[ticks]):
            raise ValueError("training numeric command and saved schedule differ")
    raw = finite(arrays["raw_command_vx_vy_yaw_clearance"], (16384, 4))
    servo = finite(arrays["servo_command_vx_vy_yaw_clearance"], (16384, 4))
    yaw_requested = np.abs(raw[:, 2]) > 1e-12
    raw_neighborhood = (raw[:, 0] >= 1.1) & (np.abs(raw[:, 2]) >= 0.25)
    servo_neighborhood = (servo[:, 0] >= 1.1) & (np.abs(servo[:, 2]) >= 0.25)
    raw_exact_vx = np.abs(raw[:, 0] - 1.2) <= 1e-12
    servo_exact_vx = np.abs(servo[:, 0] - 1.2) <= 1e-12
    raw_exact_yaw = np.abs(np.abs(raw[:, 2]) - 0.3) <= 1e-12
    servo_exact_yaw = np.abs(np.abs(servo[:, 2]) - 0.3) <= 1e-12
    ramp_mask = np.zeros(16384, dtype=bool)
    per_episode = []
    for episode, schedule in sorted(schedule_map.items()):
        mask = arrays["episode_index"] == episode
        if schedule["terrain"] == "ramp":
            ramp_mask |= mask
        choice = schedule["choice"]
        per_episode.append({"episode": episode, "terrain": schedule["terrain"],
            "speed_mps": choice["speed_mps"], "yaw_amplitude_rps": choice["yaw_amplitude_rps"],
            "controls_saved": int(np.sum(mask)), "nonzero_raw_yaw_controls": int(np.sum(mask & yaw_requested))})
    return {
        "controls": 16384, "episodes": per_episode,
        "raw_nonzero_yaw_controls": int(np.sum(yaw_requested)),
        "servo_nonzero_yaw_controls": int(np.sum(np.abs(servo[:, 2]) > 1e-12)),
        "raw_exact_1p2_and_positive_0p3_controls": int(np.sum(raw_exact_vx &
            (np.abs(raw[:, 2] - 0.3) <= 1e-12))),
        "raw_exact_1p2_and_negative_0p3_controls": int(np.sum(raw_exact_vx &
            (np.abs(raw[:, 2] + 0.3) <= 1e-12))),
        "servo_exact_1p2_and_positive_0p3_controls": int(np.sum(servo_exact_vx &
            (np.abs(servo[:, 2] - 0.3) <= 1e-12))),
        "servo_exact_1p2_and_negative_0p3_controls": int(np.sum(servo_exact_vx &
            (np.abs(servo[:, 2] + 0.3) <= 1e-12))),
        "raw_exact_1p2_abs0p3_controls": int(np.sum(raw_exact_vx & raw_exact_yaw)),
        "servo_exact_1p2_abs0p3_controls": int(np.sum(servo_exact_vx & servo_exact_yaw)),
        "raw_neighborhood_vx_ge_1p1_absyaw_ge_0p25_controls": int(np.sum(raw_neighborhood)),
        "servo_neighborhood_vx_ge_1p1_absyaw_ge_0p25_controls": int(np.sum(servo_neighborhood)),
        "ramp_controls_from_saved_schedules": int(np.sum(ramp_mask)),
        "ramp_raw_nonzero_yaw_controls": int(np.sum(ramp_mask & yaw_requested)),
        "ramp_servo_nonzero_yaw_controls": int(np.sum(ramp_mask & (np.abs(servo[:, 2]) > 1e-12))),
        "ramp_raw_neighborhood_controls": int(np.sum(ramp_mask & raw_neighborhood)),
        "yaw_positive_controls": int(np.sum(raw[:, 2] > 0)),
        "yaw_negative_controls": int(np.sum(raw[:, 2] < 0)),
        "mean_abs_effective_wheel_action_when_yaw_requested": (
            float(np.mean(np.abs(arrays["effective_action16"][yaw_requested, 12:])))
            if np.any(yaw_requested) else None),
    }


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != Path(__file__).resolve().parent or output.suffix != ".json":
        raise ValueError("output must be a new JSON file beside this C17 script")
    hashes = {}
    result = {"schema": "saved-yaw17-descriptive-v1", "scope": "C16 three flat_1p2_yaw heldout cases; C15 two 16384-control numeric training ledgers and schedules",
        "cautions": ["Different actors produce different trajectories; same-tick subtraction is not a causal counterfactual.",
                     "Wheel torque and target differences are observed controller quantities, not isolated residual contributions.",
                     "Training exposure and heldout associations do not identify a causal training mechanism."],
        "wheel_order": ["FL", "FR", "RL", "RR"], "differential": "(FR+RR-FL-RL)/2",
        "heldout": {}, "training_exposure": {}}
    for actor in ACTORS:
        series = read_case(actor, hashes)
        windows = {name: describe_window(series, *interval)
                   for name, interval in WINDOWS.items()}
        phases = {}
        for name in ("positive_1", "negative", "positive_2"):
            begin, end = WINDOWS[name]
            reached = np.abs(series["servo_yaw"][begin:end]
                             - series["raw_yaw"][begin:end]) <= 1e-9
            phases[name] = {
                "transition_servo_differs_from_raw": describe_window(series, begin, end, ~reached),
                "plateau_servo_equals_raw": describe_window(series, begin, end, reached),
            }
        result["heldout"][actor] = {"windows": windows, "servo_phase_by_segment": phases}
    for actor in ACTORS[1:]:
        result["training_exposure"][actor] = training_exposure(actor, hashes)
    result["input_sha256"] = {str(path): digest for path, digest in sorted(hashes.items())}
    result["script_sha256"] = sha256(Path(__file__).resolve())
    forbidden_loaded = sorted(_FORBIDDEN_IMPORTS.intersection(
        name.split(".", 1)[0] for name in sys.modules))
    if forbidden_loaded:
        raise RuntimeError(f"forbidden model/physics dependency already loaded: {forbidden_loaded}")
    result["execution"] = {"wall_elapsed_s": time.monotonic() - started,
                           "model_runs": 0, "physics_steps": 0,
                           "forbidden_imports_loaded": forbidden_loaded}
    with output.open("x") as target:
        json.dump(result, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")


if __name__ == "__main__":
    main()
