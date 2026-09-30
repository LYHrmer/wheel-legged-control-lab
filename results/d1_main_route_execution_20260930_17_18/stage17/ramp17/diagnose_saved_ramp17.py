#!/usr/bin/env python3
"""Read only the six saved C16 ramp/flat cases; never import a model or physics.

Usage: python3 diagnose_saved_ramp17.py --output ramp_saved_diagnosis.json
The fixed windows and case names are from continuation15/recipes15.py.
All COM velocities here are the base body's inertial COM projected into the
visible base_link frame. They are not whole-robot COM or world-X velocities.
Wheel-radius times wheel omega is a kinematic proxy, not COM speed.
"""

from __future__ import annotations

import argparse
import builtins
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from statistics import fmean


W = Path(__file__).resolve().parents[2]
CASE_ROOT = W / "continuation16/eval_01/heldout"
SOURCE_PATHS = (
    Path(__file__).resolve(),
    W / "continuation15/recipes15.py",
    W / "continuation15/score15.py",
    W / "continuation16/final_review_15_16.md",
    W / "continuation16/result_summary_15_16.json",
    W / "course_impl08/full_drive_controller_08.py",
    W / "course_impl08/residual16_math_08.py",
    W / "course_impl08/full_drive_env_08.py",
    W / "upright11/world_upright_course_11.py",
    Path("/home/lyh/wheel-legged-control-lab/src/wheel_legged_control/d1/wheel_leg_controller.py"),
    Path("/home/lyh/wheel-legged-control-lab/src/wheel_legged_control/d1/model.py"),
    Path("/home/lyh/wheel-legged-control-lab/src/wheel_legged_control/d1/state_estimation.py"),
)
ACTORS = ("zero", "global_continue", "grouped_continue")
CASES = {
    "ramp_0p45_complete": (1800, (600, 1000), 1355, .45),
    "flat_0p6": (1600, (295, 695), 695, .6),
}
WHEEL_INDICES = (3, 7, 11, 15)
RADIUS_M = .087
WHEEL_KP = 2.2
FORBIDDEN_IMPORT_ROOTS = frozenset({
    "mujoco", "torch", "stable_baselines3", "gymnasium",
    "wheel_legged_control", "full_drive_env_08", "course_plant_08",
    "full_drive_controller_08", "residual16_math_08",
})


def install_forbidden_import_guard() -> None:
    """Reject accidental runtime model, policy or physics imports."""
    loaded = FORBIDDEN_IMPORT_ROOTS.intersection(name.split(".", 1)[0] for name in sys.modules)
    if loaded:
        raise RuntimeError(f"forbidden module already loaded: {sorted(loaded)}")
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS:
            raise RuntimeError(f"forbidden model/policy/physics import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import


def assert_forbidden_imports_absent() -> None:
    loaded = FORBIDDEN_IMPORT_ROOTS.intersection(name.split(".", 1)[0] for name in sys.modules)
    if loaded:
        raise RuntimeError(f"forbidden runtime module loaded: {sorted(loaded)}")


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path, hashes: dict[str, str]):
    raw = path.read_bytes()
    hashes[str(path)] = sha_bytes(raw)
    return json.loads(raw)


def avg(values: list[float]) -> float | None:
    return fmean(values) if values else None


def rms(values: list[float]) -> float | None:
    return math.sqrt(fmean(v * v for v in values)) if values else None


def mean4(value) -> float:
    if len(value) != 4:
        raise ValueError("expected four wheels")
    return fmean(float(v) for v in value)


def any_nested_true(value) -> bool:
    if isinstance(value, list):
        return any(any_nested_true(v) for v in value)
    return bool(value)


def compressed_rows(folder: Path, n: int, hashes: dict[str, str]):
    expected = 0
    files = sorted(folder.glob("control_records_*.jsonl.gz"))
    if not files:
        raise ValueError(f"no saved control records: {folder}")
    for file in files:
        raw = file.read_bytes()
        hashes[str(file)] = sha_bytes(raw)
        for line in gzip.decompress(raw).splitlines():
            row = json.loads(line)
            if row["tick"] != expected:
                raise ValueError(f"noncontiguous control tick: {file} {row['tick']} != {expected}")
            expected += 1
            yield row
    if expected != n:
        raise ValueError(f"wrong saved control count {folder}: {expected} != {n}")


def measure(row: dict, geom_names: dict[int, str], target: float) -> dict:
    info = row["info"]
    calc = info["controller_record"]["calculation"]
    support = info["controller_record"]["nominal_support"]
    metrics = info["metrics"]
    raw = float(info["raw_operator_command"]["forward_velocity_mps"])
    servo = float(info["servo_receipt"]["applied"]["forward_velocity_mps"])
    body_pre = float(calc["consumed_body_com_forward_mps"])
    body_post = float(metrics["body_com_vx_mps"])
    omega = mean4(calc["consumed_wheel_omega_rad_s"])
    wheel_target = mean4(calc["wheel_speed_target_rad_s"])
    integral = mean4(calc["wheel_integral_after_nm"])
    base_wheel = mean4([calc["base_wheel_torque_nm"][i] for i in WHEEL_INDICES])
    common = float(calc["common_wheel_scalar_delta_nm"])
    p_from_parts = base_wheel + common - integral
    p_from_body = WHEEL_KP * (wheel_target - body_pre / RADIUS_M)
    identity_error = p_from_parts - p_from_body
    target_body_error = RADIUS_M * wheel_target - body_pre
    wheel_tracking_error = RADIUS_M * (wheel_target - omega)
    slip_proxy = RADIUS_M * omega - body_pre
    if not math.isclose(target_body_error, wheel_tracking_error + slip_proxy, abs_tol=1e-10):
        raise ValueError("wheel/body speed decomposition is inconsistent")
    native = row["native_actuator_traces"]
    if len(native) != 5:
        raise ValueError("expected five native actuator traces per control")
    native_wheel = fmean(float(trace["applied_nm"][i]) for trace in native for i in WHEEL_INDICES)
    native_limited = any(abs(float(trace["limited_nm"][i]) - float(trace["requested_nm"][i])) > 1e-9
                         for trace in native for i in WHEEL_INDICES)
    native_delayed = any(abs(float(trace["delayed_nm"][i]) - float(trace["applied_nm"][i])) > 1e-9
                         for trace in native for i in WHEEL_INDICES)
    geom_id = int(metrics["ground_geom_id"])
    native_summary = info["native_interval_summary"]
    if any("geom" in key and "wheel" in key for key in native_summary):
        raise ValueError("new per-geom wheel-load field requires explicit interpretation")
    return {
        "tick": int(row["tick"]),
        "raw_mps": raw,
        "servo_mps": servo,
        "body_pre_mps": body_pre,
        "body_post_mps": body_post,
        "body_error_vs_servo_mps": body_post - servo,
        "body_error_vs_task_mps": body_post - target,
        "wheel_target_equivalent_mps": RADIUS_M * wheel_target,
        "wheel_actual_equivalent_mps": RADIUS_M * omega,
        "wheel_tracking_error_mps": wheel_tracking_error,
        "wheel_body_slip_proxy_mps": slip_proxy,
        "target_body_error_mps": target_body_error,
        "wheel_integral_mean_nm": integral,
        "wheel_p_mean_nm": p_from_body,
        "wheel_request_mean_nm": fmean(float(calc["final_request_torque_nm"][i]) for i in WHEEL_INDICES),
        "wheel_safe_mean_nm": fmean(float(calc["safe_torque_nm"][i]) for i in WHEEL_INDICES),
        "wheel_native_applied_mean_nm": native_wheel,
        "common_wheel_scalar_nm": common,
        "p_identity_abs_error_nm": abs(identity_error),
        "wheel_target_clipped": any_nested_true(calc["wheel_speed_target_clip_mask"]),
        "wheel_torque_clipped": any(bool(calc["torque_clip_mask"][i]) for i in WHEEL_INDICES),
        "wheel_protected": any(bool(calc["protection_changed_mask"][i]) for i in WHEEL_INDICES),
        "wheel_integral_commit_all": all(bool(x) for x in calc["wheel_integral_commit_mask"]),
        "support_force_clipped": any_nested_true(support["force_clip_mask"]),
        "native_wheel_limited": native_limited,
        "native_wheel_delayed": native_delayed,
        "pitch_rad": float(metrics["absolute_pitch_rad"]),
        "pitch_target_error_rad": float(metrics["task_pitch_error_rad"]),
        "x_m": float(metrics["x_m"]),
        "ground_geom": geom_names.get(geom_id, f"unknown:{geom_id}"),
        "ramp_load_count": int(native_summary["terrain_family_positive_wheel_load"].get("ramp", 0)),
        "applied_wheel_residual_mean_rad_s": mean4(calc["wheel_action_offset_rad_s"]),
    }


NUMERIC_FIELDS = (
    "raw_mps", "servo_mps", "body_pre_mps", "body_post_mps",
    "body_error_vs_servo_mps", "body_error_vs_task_mps",
    "wheel_target_equivalent_mps", "wheel_actual_equivalent_mps",
    "wheel_tracking_error_mps", "wheel_body_slip_proxy_mps",
    "target_body_error_mps", "wheel_integral_mean_nm", "wheel_p_mean_nm",
    "wheel_request_mean_nm", "wheel_safe_mean_nm", "wheel_native_applied_mean_nm",
    "common_wheel_scalar_nm", "pitch_rad", "pitch_target_error_rad", "x_m",
    "ramp_load_count", "applied_wheel_residual_mean_rad_s",
)
FLAG_FIELDS = (
    "wheel_target_clipped", "wheel_torque_clipped", "wheel_protected",
    "support_force_clipped", "native_wheel_limited", "native_wheel_delayed",
)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    out = {"count": len(rows), "tick_bounds": [rows[0]["tick"], rows[-1]["tick"] + 1]}
    out["mean"] = {key: avg([r[key] for r in rows]) for key in NUMERIC_FIELDS}
    out["rms_body_error_vs_task_mps"] = rms([r["body_error_vs_task_mps"] for r in rows])
    out["rms_body_error_vs_servo_mps"] = rms([r["body_error_vs_servo_mps"] for r in rows])
    out["flag_fraction"] = {key: avg([float(r[key]) for r in rows]) for key in FLAG_FIELDS}
    out["integral_commit_all_fraction"] = avg([float(r["wheel_integral_commit_all"]) for r in rows])
    out["max_p_identity_abs_error_nm"] = max(r["p_identity_abs_error_nm"] for r in rows)
    out["ground_geom_counts"] = {name: sum(r["ground_geom"] == name for r in rows)
                                 for name in sorted({r["ground_geom"] for r in rows})}
    spans = []
    start = previous = rows[0]["tick"]
    for row in rows[1:]:
        tick = row["tick"]
        if tick != previous + 1:
            spans.append([start, previous + 1])
            start = tick
        previous = tick
    spans.append([start, previous + 1])
    out["tick_spans"] = spans
    return out


def case_report(case: str, actor: str, hashes: dict[str, str]) -> dict:
    n, hold, release, target = CASES[case]
    folder = CASE_ROOT / f"{case}_{actor}"
    schedule = read_json(folder / "schedule.json", hashes)
    geometry = read_json(folder / "geometry_manifest.json", hashes)
    if schedule["case_id"] != case or schedule["actor"] != actor or schedule["control_cap"] != n:
        raise ValueError("case schedule identity differs")
    geom_names = {int(g["geom_id"]): g["name"] for g in geometry["world_collision_geoms"]}
    records = []
    for row in compressed_rows(folder, n, hashes):
        tick = row["tick"]
        if row["actor"] != actor or row["info"]["raw_operator_command"] != schedule["raw_commands"][tick]:
            raise ValueError(f"actual raw command differs from saved schedule at {folder} tick {tick}")
        records.append(measure(row, geom_names, target))
    a, b = hold
    windows = {
        "pre_command": (0, 175),
        "drive_before_hold": (175, a),
        "hold": (a, b),
        "drive_after_hold": (b, release),
        "release_to_end": (release, n),
    }
    windows.update({f"hold_quarter_{i + 1}": (a + 100 * i, a + 100 * (i + 1))
                    for i in range(4)})
    by_ground = {}
    for name in sorted({r["ground_geom"] for r in records}):
        selected = [r for r in records if r["ground_geom"] == name]
        by_ground[name] = summarize(selected)
    ramp_load = [r for r in records if r["ramp_load_count"] > 0]
    return {
        "case_id": case, "actor": actor, "seed": schedule["seed"],
        "raw_command_sha256": schedule["raw_command_sha256"],
        "hold_window": list(hold), "release_tick": release,
        "windows": {name: summarize(records[i:j]) for name, (i, j) in windows.items()},
        "by_base_ground_query_geom": by_ground,
        "ramp_loaded_ticks": summarize(ramp_load),
        "identity_max_abs_error_nm": max(r["p_identity_abs_error_nm"] for r in records),
    }


def main() -> None:
    started = time.monotonic()
    install_forbidden_import_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    output = args.output.resolve(strict=False)
    if output.parent != script_dir or output.suffix != ".json":
        raise ValueError("output must be a new JSON file beside this diagnostic script")
    if output.exists():
        raise FileExistsError(output)
    hashes: dict[str, str] = {}
    for path in SOURCE_PATHS:
        hashes[str(path)] = sha_bytes(path.read_bytes())
    reports = {case: {actor: case_report(case, actor, hashes) for actor in ACTORS}
               for case in CASES}
    for case_reports in reports.values():
        for report in case_reports.values():
            if report["identity_max_abs_error_nm"] > 1e-8:
                raise ValueError("recorded wheel-P identity did not close")
    official = json.loads((W / "continuation16/result_summary_15_16.json").read_text())
    official_ramp = {row["actor"]: row for row in official["tasks"]
                     if row["case_id"] == "ramp_0p45_complete"}
    if set(official_ramp) != set(ACTORS):
        raise ValueError("official C16 ramp summary lacks one of the three actors")
    for actor in ACTORS:
        hold = reports["ramp_0p45_complete"][actor]["windows"]["hold"]
        scored = official_ramp[actor]
        if not math.isclose(hold["mean"]["body_post_mps"], scored["mean_com_vx_mps"], abs_tol=1e-9):
            raise ValueError(f"ramp hold mean differs from official C16 score: {actor}")
        if not math.isclose(hold["rms_body_error_vs_task_mps"], scored["hold_vx_rms_mps"], abs_tol=1e-9):
            raise ValueError(f"ramp hold RMS differs from official C16 score: {actor}")
        speed_passed = (abs(hold["mean"]["body_post_mps"] - .45) <= .04
                        and hold["rms_body_error_vs_task_mps"] <= .05)
        if speed_passed or scored["task_passed"]:
            raise ValueError(f"expected red ramp speed gate from C16: {actor}")
    result = {
        "schema": "d1-stage17-saved-ramp-diagnosis-v1",
        "source_file_sha256": hashes,
        "constants": {"wheel_radius_m": RADIUS_M, "wheel_kp": WHEEL_KP},
        "interpretation": {
            "com_definition": "base-body inertial COM velocity projected into visible base_link axes via model.base_velocity(local=True); controller uses pre-control estimate and score uses post-control truth of the same definition",
            "wheel_body_slip_proxy": "wheel angular speed times nominal radius minus pre-control body COM forward speed; includes kinematic mismatch, and is not a measured slip ratio",
            "p_identity": "mean differential wheel PI P plus common wheel P = Kp*(mean wheel target - pre-control body COM speed/radius)",
            "timing": "controller/calculation uses pre-control state; metrics/body_post uses post-control truth",
            "flat_comparator": "flat target is 0.6 m/s versus ramp 0.45 m/s; compare tracking errors and control components, not absolute velocities",
            "base_ground_query_geom": "geom at the base ground-height query; this is not a per-wheel contact or per-geom positive load classification",
        },
        "unidentifiable_from_saved_fields": [
            "world-frame COM velocity and exact slope-tangent COM velocity (body COM z velocity is not recorded)",
            "individual wheel longitudinal slip/contact force versus ground; control native summary has only aggregate ramp-family positive load, so per-ramp-geom wheel-load phase attribution is unavailable here",
            "counterfactual effect of changing controller gain, leg target, gravity compensation, or policy action",
        ],
        "runtime_audit": {
            "analysis_elapsed_seconds_before_output": time.monotonic() - started,
            "model_loads": 0, "policy_predicts": 0, "physics_calls": 0,
            "native_steps": 0, "backward_calls": 0, "optimizer_steps": 0,
            "forbidden_import_guard_installed": True,
        },
        "cases": reports,
        "ramp_speed_gate": {
            actor: {
                "target_mps": .45,
                "mean_tolerance_mps": .04,
                "rms_limit_mps": .05,
                "mean_body_com_vx_mps": reports["ramp_0p45_complete"][actor]["windows"]["hold"]["mean"]["body_post_mps"],
                "hold_vx_rms_mps": reports["ramp_0p45_complete"][actor]["windows"]["hold"]["rms_body_error_vs_task_mps"],
                "passed": False,
                "matches_official_c16": True,
            }
            for actor in ACTORS
        },
        "paired_hold_ramp_minus_flat": {
            actor: {
                key: reports["ramp_0p45_complete"][actor]["windows"]["hold"]["mean"][key]
                     - reports["flat_0p6"][actor]["windows"]["hold"]["mean"][key]
                for key in (
                    "body_error_vs_task_mps", "body_error_vs_servo_mps",
                    "wheel_tracking_error_mps", "wheel_body_slip_proxy_mps",
                    "target_body_error_mps", "wheel_integral_mean_nm",
                    "wheel_p_mean_nm", "wheel_native_applied_mean_nm",
                    "pitch_target_error_rad", "applied_wheel_residual_mean_rad_s",
                )
            }
            for actor in ACTORS
        },
    }
    assert_forbidden_imports_absent()
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output}; six cases; max wheel-P identity error "
          f"{max(r['identity_max_abs_error_nm'] for c in reports.values() for r in c.values()):.3g} Nm")


if __name__ == "__main__":
    main()
