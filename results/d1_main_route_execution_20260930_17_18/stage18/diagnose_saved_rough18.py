"""Read-only C16/C17 rough comparison. No model, physics, or policy imports.

Run once from any working directory. The sole output is an exclusive JSON file
beside this script. Only the four hard-coded rough control streams are opened.
Native block streams and states are never loaded. Correlations are descriptive.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time


START_NS = time.monotonic_ns()
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
OLD = WORK / "continuation16/eval_01/heldout"
NEW = WORK / "continuation17/qualification_primary_01/heldout"
CASES = (
    ("C16", "zero", OLD / "rough_0p35_zero"),
    ("C16", "grouped_continue", OLD / "rough_0p35_grouped_continue"),
    ("C17", "zero", NEW / "rough_0p35_zero"),
    ("C17", "grouped_continue", NEW / "rough_0p35_grouped_continue"),
)
SOURCE_FILES = (
    WORK / "continuation15/recipes15.py",
    WORK / "continuation17/recipes17.py",
    WORK / "course_impl08/residual16_math_08.py",
    WORK / "continuation17/controller17.py",
    WORK / "continuation17/residual17.py",
    WORK / "continuation16/result_summary_15_16.json",
    WORK / "continuation17/qualification_summary_17.json",
)
OUTPUT = HERE / "saved_rough_diagnosis_18.json"
WHEEL = (3, 7, 11, 15)
N = 1600
HOLD = (245, 645)
TARGET = 0.35
KP = 2.2
RADIUS = 0.087
WINDOWS = {
    "settle_0_175": (0, 175),
    "drive_175_645": (175, 645),
    "drive_rise_175_245": (175, 245),
    "hold_245_645": HOLD,
    "hold_q1_245_345": (245, 345),
    "hold_q2_345_445": (345, 445),
    "hold_q3_445_545": (445, 545),
    "hold_q4_545_645": (545, 645),
    "release_645_1500": (645, 1500),
    "final_1500_1600": (1500, 1600),
}
SERIES = (
    "body_pre_vx", "body_post_vx", "servo_vx", "body_servo_error",
    "wheel_mean_omega", "nominal_mean_target", "wheel_mean_target",
    "wheel_mean_error", "integral_mean_error", "wheel_body_slip_mps",
    "integral_mean_before", "integral_mean_after", "integral_mean_increment",
    "integral_differential_rms", "common_p_nm", "wheel_p_mean_nm",
    "wheel_base_mean_nm", "wheel_request_mean_nm", "wheel_safe_mean_nm",
    "native_requested_mean_nm", "native_applied_mean_nm", "pitch_rad",
    "clearance_m", "ground_height_m", "x_m", "yaw_post_rps",
    "yaw_pre_rps", "yaw_effective_rps", "yaw_unlimited_rps",
    "yaw_limit_active", "yaw_effective_at_limit", "yaw_pre_above_0p15",
    "wheel_clip_fraction", "antiwindup_reject_fraction",
    "support_force_clip_fraction", "nonwheel_contacts", "positive_wheel_load",
)
RUNTIME_COUNTERS = {"model_constructions": 0, "physics_steps": 0,
                    "policy_loads": 0, "training_calls": 0}
FORBIDDEN_MODULE_PREFIXES = (
    "mujoco", "gymnasium", "stable_baselines3", "torch",
    "wheel_legged_control", "world_upright_course_11",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(x: object, name: str) -> float:
    value = float(x)
    if not math.isfinite(value):
        raise ValueError(f"{name}: nonfinite")
    return value


def vector(x: object, length: int, name: str) -> list[float]:
    if not isinstance(x, list) or len(x) != length:
        raise ValueError(f"{name}: expected {length} elements")
    return [finite(v, name) for v in x]


def mean(x: list[float]) -> float:
    return statistics.fmean(x)


def rms(x: list[float]) -> float:
    return math.sqrt(mean([v * v for v in x]))


def stats(x: list[float]) -> dict:
    if not x:
        return {"n": 0}
    return {"n": len(x), "mean": mean(x), "rms": rms(x),
            "std": statistics.pstdev(x), "min": min(x), "max": max(x),
            "start": x[0], "end": x[-1], "end_minus_start": x[-1] - x[0]}


def corr(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or len(a) < 3:
        return None
    ma, mb = mean(a), mean(b)
    aa = [v - ma for v in a]
    bb = [v - mb for v in b]
    denominator = math.sqrt(sum(v*v for v in aa) * sum(v*v for v in bb))
    return sum(x*y for x, y in zip(aa, bb)) / denominator if denominator > 0 else None


def trailing_50(series: list[float]) -> tuple[list[float | None], list[float | None]]:
    """Mean of current and preceding 49 samples; no future samples."""
    low: list[float | None] = [None] * len(series)
    high: list[float | None] = [None] * len(series)
    acc = 0.0
    for tick, value in enumerate(series):
        acc += value
        if tick >= 50:
            acc -= series[tick - 50]
        if tick >= 49:
            low[tick] = acc / 50.0
            high[tick] = value - low[tick]
    return low, high


def windows(rows: list[dict]) -> dict:
    result = {}
    for name, (start, end) in WINDOWS.items():
        part = rows[start:end]
        body_error = [r["body_post_vx"] - TARGET for r in part]
        ids = [r["ground_geom_id"] for r in part]
        names = [r["ground_geom_name"] for r in part]
        counts = {k: names.count(k) for k in sorted(set(names))}
        changes = [start + i for i in range(1, len(ids)) if ids[i] != ids[i-1]]
        result[name] = {
            "tick_interval_half_open": [start, end],
            "body_post_target_error": stats(body_error),
            "body_post_target_rms_mps": rms(body_error),
            "series": {key: stats([r[key] for r in part]) for key in SERIES},
            "ground": {"start_x_m": part[0]["x_m"], "end_x_m": part[-1]["x_m"],
                       "geom_counts": counts, "geom_change_ticks": changes},
        }
    return result


def frequency_description(rows: list[dict]) -> dict:
    """Fixed trailing 50-tick decomposition. Lag is offline description only."""
    names = ("body_servo_error", "wheel_mean_error", "integral_mean_error",
             "integral_mean_after", "integral_mean_increment", "common_p_nm",
             "native_applied_mean_nm", "yaw_post_rps")
    decomposed = {}
    for key in names:
        low, high = trailing_50([r[key] for r in rows])
        decomposed[key] = (low, high)
    start, end = HOLD
    output = {"filter": "trailing 50 controls inclusive; initial 49 undefined",
              "lag_definition": "corr(first[t], second[t+lag]) within hold; lag 0..20 controls; descriptive only",
              "components": {}}
    for key, (low, high) in decomposed.items():
        output["components"][key] = {
            "low": stats([v for v in low[start:end] if v is not None]),
            "high": stats([v for v in high[start:end] if v is not None]),
        }
    pairs = (("body_servo_error", "integral_mean_increment"),
             ("body_servo_error", "integral_mean_after"),
             ("body_servo_error", "native_applied_mean_nm"),
             ("wheel_mean_error", "integral_mean_increment"),
             ("yaw_post_rps", "body_servo_error"))
    output["lag_correlations"] = {}
    for first, second in pairs:
        key = f"{first}__to__{second}"
        output["lag_correlations"][key] = {}
        for component, index in (("low", 0), ("high", 1)):
            a, b = decomposed[first][index], decomposed[second][index]
            output["lag_correlations"][key][component] = [
                {"lag_controls": lag, "n": end-start-lag,
                 "pearson_r": corr(a[start:end-lag], b[start+lag:end])}
                for lag in range(21)
            ]
    # Orthogonal real DFT bins of the fixed 400-control hold at 100 Hz.
    # Each band reports mean-square contribution in the original units squared.
    bands = (("le_1_hz", 0.0, 1.0), ("gt_1_le_5_hz", 1.0, 5.0),
             ("gt_5_le_15_hz", 5.0, 15.0), ("gt_15_hz", 15.0, 50.0))
    output["hold_demeaned_dft_mean_square_by_band"] = {}
    for key in ("body_post_vx", "body_servo_error", "wheel_mean_error",
                "integral_mean_error", "integral_mean_after", "common_p_nm",
                "native_applied_mean_nm", "yaw_post_rps"):
        values = [r[key] for r in rows[start:end]]
        centered = [v-mean(values) for v in values]
        n = len(centered)
        powers = []
        for k in range(n//2+1):
            re = sum(v*math.cos(2*math.pi*k*t/n) for t,v in enumerate(centered))
            im = -sum(v*math.sin(2*math.pi*k*t/n) for t,v in enumerate(centered))
            weight = 1 if k in (0,n//2) else 2
            powers.append(weight*(re*re+im*im)/(n*n))
        band_values = {}
        for name, lo, hi in bands:
            band_values[name] = sum(power for k,power in enumerate(powers)
                                    if (lo < 100*k/n <= hi if lo > 0 else 100*k/n <= hi))
        output["hold_demeaned_dft_mean_square_by_band"][key] = {
            "bands": band_values, "sum": sum(band_values.values()),
            "total_demeaned_mean_square": mean([v*v for v in centered]),
        }
    return output


def read_case(stage: str, actor: str, directory: Path, hashes: dict) -> dict:
    receipt_path = directory / "case_receipt.json"
    schedule_path = directory / "schedule.json"
    geometry_path = directory / "geometry_manifest.json"
    for path in (receipt_path, schedule_path, geometry_path, directory / "initial_state.npz"):
        hashes[str(path)] = digest(path)
    receipt = json.loads(receipt_path.read_text())
    schedule = json.loads(schedule_path.read_text())
    geometry = json.loads(geometry_path.read_text())
    if (receipt["case_id"] != "rough_0p35" or receipt["actor"] != actor
            or receipt["completed_controls"] != N or schedule["case_id"] != "rough_0p35"
            or len(schedule["raw_commands"]) != N):
        raise ValueError(f"{stage}/{actor}: identity/horizon mismatch")
    geom_by_id = {int(g["geom_id"]): g["name"] for g in geometry["world_collision_geoms"]}
    rows: list[dict] = []
    max_identity_error = 0.0
    for block in range(8):
        path = directory / f"control_records_{block:04d}.jsonl.gz"
        hashes[str(path)] = digest(path)
        with gzip.open(path, "rt") as stream:
            for line in stream:
                record = json.loads(line)
                tick = len(rows)
                if record["tick"] != tick or tick >= N or record["actor"] != actor:
                    raise ValueError(f"{stage}/{actor}: nonsequential or wrong-actor tick")
                info = record["info"]
                calc = info["controller_record"]["calculation"]
                adapter = info["controller_record"]
                metric = info["metrics"]
                native_summary = info["native_interval_summary"]
                servo = finite(info["servo_receipt"]["applied"]["forward_velocity_mps"], "servo")
                if info["raw_operator_command"] != schedule["raw_commands"][tick]:
                    raise ValueError(f"{stage}/{actor}: raw schedule mismatch at {tick}")
                if abs(servo - finite(adapter["servo_forward_mps"], "adapter servo")) > 1e-10:
                    raise ValueError(f"{stage}/{actor}: servo mismatch at {tick}")
                body = finite(calc["consumed_body_com_forward_mps"], "body pre")
                target = vector(calc["wheel_speed_target_rad_s"], 4, "wheel target")
                nominal = vector(calc["consumed_nominal_wheel_speed_rad_s"], 4, "nominal wheel")
                omega = vector(calc["consumed_wheel_omega_rad_s"], 4, "wheel omega")
                error = vector(calc["wheel_speed_error_rad_s"], 4, "wheel error")
                ie = (vector(calc["wheel_integral_error_rad_s"], 4, "integral error")
                      if stage == "C17" else error[:])
                ib = vector(calc["wheel_integral_before_nm"], 4, "integral before")
                ia = vector(calc["wheel_integral_after_nm"], 4, "integral after")
                commit = calc["wheel_integral_commit_mask"]
                if len(commit) != 4:
                    raise ValueError("integral commit mask length")
                base = vector(calc["base_wheel_torque_nm"], 16, "wheel base")
                request = vector(calc["final_request_torque_nm"], 16, "wheel request")
                safe = vector(calc["safe_torque_nm"], 16, "wheel safe")
                common = finite(calc["common_wheel_scalar_delta_nm"], "common P")
                active = bool(calc["body_common_p_active"])
                expected_common = KP*(mean(omega)-body/RADIUS) if active else 0.0
                expected_p_i = [KP*(t-o) for t, o in zip(target, omega)]
                expected_base = [p+i for p, i in zip(expected_p_i, ia)]
                expected_total = KP*(mean(target)-(body/RADIUS if active else mean(omega)))+mean(ia)
                identity_errors = ([abs(common-expected_common),
                                    abs(mean([base[j] for j in WHEEL])+common-expected_total)]
                                   + [abs(base[j]-expected_base[k]) for k, j in enumerate(WHEEL)]
                                   + [abs(error[k]-(target[k]-omega[k])) for k in range(4)])
                max_identity_error = max(max_identity_error, *identity_errors)
                if max(identity_errors) > 1e-7:
                    raise ValueError(f"{stage}/{actor}: wheel P/common identity at {tick}")
                if stage == "C17":
                    if abs(finite(calc["consumed_servo_forward_mps"], "calc servo")-servo) > 1e-10:
                        raise ValueError(f"{stage}/{actor}: calc servo mismatch at {tick}")
                    expected_ie = ([e-mean(error)+(servo-body)/RADIUS for e in error]
                                   if active else error)
                    if max(abs(a-b) for a,b in zip(ie,expected_ie)) > 1e-7:
                        raise ValueError(f"{stage}/{actor}: integral-error identity at {tick}")
                traces = record["native_actuator_traces"]
                if not traces:
                    raise ValueError("empty native actuator traces")
                native_req = mean([mean([finite(tr["requested_nm"][j], "native request")
                                         for j in WHEEL]) for tr in traces])
                native_applied = mean([mean([finite(tr["applied_nm"][j], "native applied")
                                             for j in WHEEL]) for tr in traces])
                support_clip = info["controller_record"]["nominal_support"]["force_clip_mask"]
                world_w = vector(info["controller_record"]["nominal_support"]["consumed_base_angular_velocity_world"], 3, "world angular velocity")
                rot = info["controller_record"]["nominal_support"]["consumed_base_rotation_world_from_body"]
                yaw_pre = sum(finite(rot[i][2], "rotation")*world_w[i] for i in range(3))
                yaw_servo = finite(adapter["servo_yaw_rps"], "servo yaw")
                yaw_unlimited = yaw_servo + 4.0*(yaw_servo-yaw_pre)
                yaw_limit = 0.6 if stage == "C16" else finite(adapter["nominal_yaw_limit_rps"], "yaw limit")
                yaw_effective = finite(adapter["nominal_effective_yaw_rps"], "effective yaw")
                if stage == "C17" and abs(yaw_unlimited-finite(adapter["nominal_unlimited_yaw_rps"], "unlimited yaw")) > 1e-7:
                    raise ValueError(f"{stage}/{actor}: nominal yaw identity at {tick}")
                load = native_summary["terrain_family_positive_wheel_load"]
                gid = int(metric["ground_geom_id"])
                ground_name = geom_by_id.get(gid, f"unknown_geom_{gid}")
                out = {
                    "tick": tick, "x_m": finite(metric["x_m"], "x"),
                    "ground_geom_id": gid, "ground_geom_name": ground_name,
                    "ground_height_m": finite(metric["ground_height_m"], "ground height"),
                    "body_pre_vx": body, "body_post_vx": finite(metric["body_com_vx_mps"], "body post"),
                    "servo_vx": servo, "body_servo_error": servo-body,
                    "wheel_target": target, "nominal_wheel_target": nominal, "wheel_omega": omega,
                    "wheel_error": error, "integral_error": ie,
                    "integral_before": ib, "integral_after": ia,
                    "wheel_mean_omega": mean(omega), "nominal_mean_target": mean(nominal),
                    "wheel_mean_target": mean(target), "wheel_mean_error": mean(error),
                    "integral_mean_error": mean(ie), "wheel_body_slip_mps": RADIUS*mean(omega)-body,
                    "integral_mean_before": mean(ib), "integral_mean_after": mean(ia),
                    "integral_mean_increment": mean(ia)-mean(ib),
                    "integral_differential_rms": rms([v-mean(ia) for v in ia]),
                    "common_p_nm": common, "wheel_p_mean_nm": mean(expected_p_i),
                    "wheel_base_mean_nm": mean([base[j] for j in WHEEL]),
                    "wheel_request_mean_nm": mean([request[j] for j in WHEEL]),
                    "wheel_safe_mean_nm": mean([safe[j] for j in WHEEL]),
                    "native_requested_mean_nm": native_req, "native_applied_mean_nm": native_applied,
                    "pitch_rad": finite(metric["absolute_pitch_rad"], "pitch"),
                    "clearance_m": finite(metric["clearance_m"], "clearance"),
                    "yaw_pre_rps": yaw_pre, "yaw_post_rps": finite(metric["body_yaw_rate_rps"], "yaw post"),
                    "yaw_effective_rps": yaw_effective, "yaw_unlimited_rps": yaw_unlimited,
                    "yaw_limit_active": float(abs(yaw_unlimited) > yaw_limit+1e-8),
                    "yaw_effective_at_limit": float(abs(yaw_effective) >= yaw_limit-1e-8),
                    "yaw_pre_above_0p15": float(abs(yaw_pre) > 0.15),
                    "wheel_clip_fraction": mean([float(calc["torque_clip_mask"][j] or calc["protection_changed_mask"][j]) for j in WHEEL]),
                    "antiwindup_reject_fraction": mean([float(not v) for v in commit]),
                    "support_force_clip_fraction": mean([float(v) for v in support_clip]),
                    "nonwheel_contacts": finite(metric["native_interval_nonwheel_contacts"], "nonwheel contacts"),
                    "positive_wheel_load": sum(finite(v, "positive load") for v in load.values()),
                    "positive_wheel_load_by_family": load,
                    "body_common_p_active": active,
                }
                rows.append(out)
    if len(rows) != N:
        raise ValueError(f"{stage}/{actor}: expected {N} controls, got {len(rows)}")
    official_rms = rms([r["body_post_vx"]-TARGET for r in rows[HOLD[0]:HOLD[1]]])
    official_mean = mean([r["body_post_vx"] for r in rows[HOLD[0]:HOLD[1]]])
    return {"stage": stage, "actor": actor, "directory": str(directory),
            "receipt": {"case_id": receipt["case_id"], "actor": receipt["actor"],
                        "seed": receipt["seed"], "completed_controls": receipt["completed_controls"],
                        "raw_command_sha256": receipt["raw_command_sha256"],
                        "checkpoint_sha256": receipt["checkpoint_sha256"]},
            "identity_max_abs_torque_error_nm": max_identity_error,
            "hold_score_from_post_body_com": {"mean_mps": official_mean,
                                                "rms_error_mps": official_rms},
            "windows": windows(rows), "frequency": frequency_description(rows),
            "per_tick": rows}


def main() -> None:
    if any(name == prefix or name.startswith(prefix + ".")
           for name in sys.modules for prefix in FORBIDDEN_MODULE_PREFIXES):
        raise RuntimeError("forbidden model, physics, or policy module loaded")
    hashes = {str(path): digest(path) for path in SOURCE_FILES}
    old_summary = json.loads((WORK / "continuation16/result_summary_15_16.json").read_text())
    new_summary = json.loads((WORK / "continuation17/qualification_summary_17.json").read_text())
    results = [read_case(stage, actor, directory, hashes) for stage, actor, directory in CASES]
    for result in results:
        stage, actor = result["stage"], result["actor"]
        if stage == "C16":
            official = next(r for r in old_summary["tasks"] if r["case_id"] == "rough_0p35" and r["actor"] == actor)
            expected_mean, expected_rms = official["mean_com_vx_mps"], official["hold_vx_rms_mps"]
        else:
            official = new_summary["task_gates"]["primary"]["rough_0p35"][actor]
            expected_mean, expected_rms = official["hold_mean_com_vx_mps"], official["hold_vx_rms_mps"]
        score = result["hold_score_from_post_body_com"]
        if abs(score["mean_mps"]-expected_mean) > 1e-9 or abs(score["rms_error_mps"]-expected_rms) > 1e-9:
            raise ValueError(f"{stage}/{actor}: official score does not close")
        result["official_score"] = {"mean_mps": expected_mean, "rms_error_mps": expected_rms,
                                    "speed_passed": official.get("speed_passed", official.get("task_passed"))}
    hashes[str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
    payload = {
        "schema": "saved-rough-diagnosis-18-v1",
        "scope": "exactly four C16/C17 rough_0p35 zero/grouped control streams; no native blocks or state arrays",
        "window_convention": "half-open control ticks; score uses post-step body-frame base COM vx",
        "drive_interval_half_open": [175, 645], "hold_interval_half_open": list(HOLD),
        "interpretation_limits": [
            "Ground geom is the base ground query, not per-wheel contact attribution.",
            "Positive wheel load is grouped by terrain family, not by individual contact geom.",
            "Old and new trajectories diverge in x and contact timing; cross-run temporal differences are not isolated interventions.",
            "Old yaw unlimited value is reconstructed from saved world angular velocity and rotation; old controller did not record it directly.",
            "Trailing filter and lag correlations describe saved signals; they do not establish causal direction.",
            "Native wheel torques are means of saved native substeps, not mechanical energy.",
        ],
        "source_sha256": hashes,
        "runtime_audit": {**RUNTIME_COUNTERS,
                          "elapsed_wall_s": (time.monotonic_ns()-START_NS)/1e9,
                          "forbidden_modules_loaded": False},
        "cases": results,
    }
    with OUTPUT.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
