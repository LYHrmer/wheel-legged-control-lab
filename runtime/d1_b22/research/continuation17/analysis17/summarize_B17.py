"""Pure saved-data summary of the three frozen C17 development workers.

Reads completed archives and independent readbacks only. No policy, controller,
model, engine, or physics module is imported. The fixed output is opened with x.
"""

from __future__ import annotations

import builtins
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time


_IMPORT = builtins.__import__
_FORBIDDEN = frozenset(("torch", "mujoco", "stable_baselines3", "gym", "gymnasium",
                       "wheel_legged_control", "engine_binding"))


def _guard(name, *args, **kwargs):
    if name.partition(".")[0] in _FORBIDDEN:
        raise RuntimeError("model/physics import denied: " + name)
    return _IMPORT(name, *args, **kwargs)


builtins.__import__ = _guard
import numpy as np


C = Path(__file__).resolve().parent.parent
RUNS = {variant: C / f"development_{variant}_01"
        for variant in ("baseline", "yaw", "ramp")}
OUTPUT = C / "development_summary_17.json"
PAIR_FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation")
YAW_SEGMENTS = {"positive_1": (415, 515), "negative": (515, 715),
                "positive_2": (715, 815)}
RAMP_QUARTERS = {f"quarter_{index+1}": (600+100*index, 700+100*index)
                 for index in range(4)}
WHEEL_INDICES = (3, 7, 11, 15)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def track(path: Path, sources: dict[str, str]) -> Path:
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ValueError("missing or linked input: " + str(path))
    sources[str(path)] = sha(path)
    return path


def document(path: Path, sources: dict[str, str]) -> dict:
    value = json.loads(track(path, sources).read_text())
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    return value


def initial(path: Path, sources: dict[str, str]) -> dict[str, np.ndarray]:
    with np.load(track(path, sources), allow_pickle=False) as loaded:
        return {key: np.array(loaded[key]) for key in PAIR_FIELDS}


def bitwise_pair(first: Path, second: Path, sources: dict[str, str]) -> dict:
    a, b = initial(first, sources), initial(second, sources)
    fields = {key: bool(a[key].shape == b[key].shape
                        and a[key].dtype == b[key].dtype
                        and a[key].tobytes(order="C") == b[key].tobytes(order="C"))
              for key in PAIR_FIELDS}
    return {"fields_bitwise_equal": fields, "all_five_equal": all(fields.values())}


def rows(folder: Path, sources: dict[str, str]) -> list[dict]:
    receipt = document(folder / "case_receipt.json", sources)
    result = []
    for index, block in enumerate(receipt["controller_record_blocks"]):
        name = f"control_records_{index:04d}.jsonl.gz"
        if block["file"] != name or not 0 < block["rows"] <= 200:
            raise ValueError("control block order/count differs")
        path = track(folder / name, sources)
        if sha(path) != block["sha256"]:
            raise ValueError("control block receipt hash differs")
        with gzip.open(path, "rt") as source:
            part = [json.loads(line) for line in source]
        if len(part) != block["rows"]:
            raise ValueError("control block row count differs")
        result.extend(part)
    if len(result) != receipt["completed_controls"] or not receipt["record_valid"]:
        raise ValueError("incomplete saved case")
    if any(row["tick"] != index for index, row in enumerate(result)):
        raise ValueError("control ticks are not consecutive")
    return result


def mean(values) -> float:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("nonfinite or empty series")
    return float(np.mean(x))


def rms(values) -> float:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("nonfinite or empty series")
    return float(np.sqrt(np.mean(x*x)))


def yaw_slice(all_rows: list[dict], begin: int, end: int, *, plateau: bool | None) -> dict:
    chosen = []
    for row in all_rows[begin:end]:
        info = row["info"]
        command = info["raw_operator_command"]
        servo = info["consumed_command"]
        reached = abs(float(servo["yaw_rate_rps"]) - float(command["yaw_rate_rps"])) <= 1e-9
        if plateau is None or reached is plateau:
            chosen.append(row)
    if not chosen:
        return {"ticks": [begin, end], "count": 0, "available": False}
    errors, prelimit, clipped, nominal, servo, actual = [], [], [], [], [], []
    for row in chosen:
        info = row["info"]
        adapter = info["controller_record"]
        command_yaw = float(info["consumed_command"]["yaw_rate_rps"])
        body_yaw = float(info["metrics"]["body_yaw_rate_rps"])
        request = command_yaw + float(adapter["nominal_yaw_feedback_gain"])*(
            command_yaw - float(adapter["pre_body_yaw_rate_rps"]))
        new_limit = float(adapter["nominal_yaw_limit_rps"])
        if not new_limit > 0 or not math.isclose(
                float(adapter["nominal_effective_yaw_rps"]),
                float(np.clip(request, -new_limit, new_limit)), abs_tol=1e-8):
            raise ValueError("nominal yaw clip differs from saved pre-state")
        errors.append(body_yaw-command_yaw)
        prelimit.append(request)
        clipped.append(abs(float(adapter["nominal_effective_yaw_rps"])) >= new_limit-1e-9)
        nominal.append(float(adapter["nominal_effective_yaw_rps"]))
        servo.append(command_yaw)
        actual.append(body_yaw)
    return {"ticks": [begin, end], "count": len(chosen), "available": True,
            "signed_error_mean_rps": mean(errors), "error_rms_rps": rms(errors),
            "error_sse": float(np.sum(np.square(errors))),
            "servo_mean_rps": mean(servo), "actual_mean_rps": mean(actual),
            "preclip_nominal_yaw_mean_rps": mean(prelimit),
            "effective_nominal_yaw_mean_rps": mean(nominal),
            "preclip_exceeds_old_0p6_fraction": mean([abs(x) > .6+1e-9 for x in prelimit]),
            "at_new_limit_fraction": mean(clipped)}


def yaw_summary(all_rows: list[dict]) -> dict:
    if len(all_rows) < 815:
        return {"available": False, "completed_controls": len(all_rows),
                "reason": "yaw hold did not complete"}
    complete = yaw_slice(all_rows, 415, 815, plateau=None)
    segments = {}
    for name, (begin, end) in YAW_SEGMENTS.items():
        segments[name] = {
            "all": yaw_slice(all_rows, begin, end, plateau=None),
            "transition": yaw_slice(all_rows, begin, end, plateau=False),
            "plateau": yaw_slice(all_rows, begin, end, plateau=True),
        }
    return {"hold_all_400": complete, "segments": segments}


def wheel_mean16(calc: dict, field: str) -> float:
    value = np.asarray(calc[field], dtype=float)
    if value.shape != (16,) or not np.isfinite(value).all():
        raise ValueError("invalid wheel torque shape")
    return float(np.mean(value[list(WHEEL_INDICES)]))


def wheel_mean4(calc: dict, field: str) -> float:
    value = np.asarray(calc[field], dtype=float)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("invalid wheel vector shape")
    return float(np.mean(value))


def ramp_summary(all_rows: list[dict]) -> dict:
    if len(all_rows) < 1000:
        return {"available": False, "completed_controls": len(all_rows),
                "reason": "ramp hold did not complete"}
    windows = {}
    for name, (begin, end) in {"hold_all_400": (600, 1000), **RAMP_QUARTERS}.items():
        post, servo, integral, proportional, common, target, omega = ([] for _ in range(7))
        torque_request, torque_safe = [], []
        for row in all_rows[begin:end]:
            info = row["info"]
            calc = info["controller_record"]["calculation"]
            post.append(float(info["metrics"]["body_com_vx_mps"]))
            servo.append(float(info["consumed_command"]["forward_velocity_mps"]))
            integral.append(wheel_mean4(calc, "wheel_integral_after_nm"))
            base = wheel_mean16(calc, "base_wheel_torque_nm")
            proportional.append(base-integral[-1])
            common.append(wheel_mean16(calc, "common_wheel_delta_torque_nm"))
            target.append(wheel_mean4(calc, "wheel_speed_target_rad_s"))
            omega.append(wheel_mean4(calc, "consumed_wheel_omega_rad_s"))
            torque_request.append(wheel_mean16(calc, "final_request_torque_nm"))
            torque_safe.append(wheel_mean16(calc, "safe_torque_nm"))
        error = np.asarray(post)-np.asarray(servo)
        windows[name] = {"ticks": [begin, end], "count": end-begin,
            "post_com_vx_mean_mps": mean(post), "servo_vx_mean_mps": mean(servo),
            "post_minus_servo_mean_mps": mean(error), "post_minus_servo_rms_mps": rms(error),
            "wheel_integral_mean_nm": mean(integral),
            "wheel_proportional_mean_nm": mean(proportional),
            "common_wheel_proportional_mean_nm": mean(common),
            "wheel_target_mean_rad_s": mean(target), "wheel_omega_mean_rad_s": mean(omega),
            "wheel_target_minus_omega_mean_rad_s": mean(np.asarray(target)-np.asarray(omega)),
            "final_request_wheel_mean_nm": mean(torque_request),
            "safe_wheel_torque_mean_nm": mean(torque_safe)}
    return windows


def case_score(readback: dict, case_id: str) -> dict:
    matches = [row for row in readback["numeric_scores"]
               if row["case_id"] == case_id and row["experiment_actor"] == "grouped_continue"]
    if len(matches) != 1:
        raise ValueError("missing unique grouped task score: " + case_id)
    row = matches[0]
    return {"task_passed": row["task_passed"], "task_reasons": row["task_reasons"],
            "record_valid": row["record"]["record_valid"],
            "safety_passed": row["safety"]["passed"],
            "final_window_passed": row["final_window"]["passed"],
            "speed_passed": row["speed"]["passed"],
            "yaw_passed": row["yaw"]["passed"],
            "hold_vx_rms_mps": row["speed"].get("rms_com_vx_error_vs_target_mps"),
            "hold_yaw_rms_rps": row["yaw"].get("rms_yaw_rate_error_vs_applied_servo_rps"),
            "drive_sse": row["rl_terms"]["sse_total"]}


def main() -> None:
    started = time.monotonic()
    sources: dict[str, str] = {}
    inputs = {}
    for variant, folder in RUNS.items():
        readback = document(C / f"development_{variant}_readback_17.json", sources)
        host = document(folder / "host_receipt.json", sources)
        session = document(folder / "session.json", sources)
        worker = document(folder / "worker_receipt.json", sources)
        if (readback.get("source_closure_verified") is not True
                or readback.get("checkpoint_ZIP_verified") is not True
                or readback.get("physical_calls_performed_by_reader") != 0
                or readback.get("run") != str(folder.resolve())
                or readback.get("controller_variant") != variant
                or host.get("changed_sources") != []
                or host.get("postcheck_complete") is not True
                or host.get("failure") is not None
                or host.get("exit_code") != 0
                or worker.get("execution_complete") is not True
                or worker.get("failure") is not None
                or session.get("controller_variant") != variant
                or not session.get("source_hashes")):
            raise ValueError("host/worker/independent readback source closure differs: " + variant)
        inputs[variant] = readback
    pairs = {}
    for name, a, b, case_id in (
        ("yaw_initial_baseline_vs_yaw", "baseline", "yaw", "flat_1p2_yaw"),
        ("ramp_initial_baseline_vs_ramp", "baseline", "ramp", "ramp_0p45_complete"),
        ("floor_initial_baseline_vs_yaw", "baseline", "yaw", None),
        ("floor_initial_baseline_vs_ramp", "baseline", "ramp", None)):
        left = RUNS[a] / ("floor/grouped_continue" if case_id is None
                          else f"heldout/{case_id}_grouped_continue") / "initial_state.npz"
        right = RUNS[b] / ("floor/grouped_continue" if case_id is None
                           else f"heldout/{case_id}_grouped_continue") / "initial_state.npz"
        pairs[name] = bitwise_pair(left, right, sources)
        if not pairs[name]["all_five_equal"]:
            raise ValueError("cross-worker initial state differs: " + name)
    result = {"schema": "d1-control-repair-development-B-summary-17-v1",
              "source_closure_verified": True,
              "cross_worker_initial_state": pairs,
              "task_gates": {}, "yaw_diagnostics": {}, "ramp_diagnostics": {},
              "interpretation_limit": "Saved-run association only; the policies and resulting trajectories are not same-state counterfactuals."}
    for variant, readback in inputs.items():
        result["task_gates"][variant] = {}
        for spec in ("flat_1p2_yaw", "ramp_0p45_complete"):
            if any(row["case_id"] == spec and row["experiment_actor"] == "grouped_continue"
                   for row in readback["numeric_scores"]):
                result["task_gates"][variant][spec] = case_score(readback, spec)
        if "flat_1p2_yaw" in result["task_gates"][variant]:
            result["yaw_diagnostics"][variant] = yaw_summary(rows(
                RUNS[variant] / "heldout/flat_1p2_yaw_grouped_continue", sources))
        if "ramp_0p45_complete" in result["task_gates"][variant]:
            result["ramp_diagnostics"][variant] = ramp_summary(rows(
                RUNS[variant] / "heldout/ramp_0p45_complete_grouped_continue", sources))
    result["input_sha256"] = dict(sorted(sources.items()))
    result["script_sha256"] = sha(Path(__file__))
    loaded = {name.partition(".")[0] for name in sys.modules} & _FORBIDDEN
    if loaded:
        raise RuntimeError("forbidden modules loaded: " + repr(sorted(loaded)))
    result["execution"] = {"wall_elapsed_s": time.monotonic()-started,
                           "model_runs": 0, "physics_steps": 0}
    with OUTPUT.open("x") as target:
        json.dump(result, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")


if __name__ == "__main__":
    main()
