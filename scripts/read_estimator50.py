"""Independent stdlib-only streaming audit; no simulator, NumPy or experiment imports."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import struct
import traceback
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "constant_velocity_measured_leg_q"
LEGS = [i for i in range(16) if i % 4 != 3]
CHANGED = {"base_position", "base_rotation", "base_linear_velocity_world",
           "base_angular_velocity_world", "joint_position", "foot_position", "foot_jacobian"}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def flat(value):
    if isinstance(value, list):
        return [number for child in value for number in flat(child)]
    return [value]


def exact(left, right):
    if isinstance(left, list):
        return (isinstance(right, list) and len(left) == len(right) and
                all(exact(a, b) for a, b in zip(left, right, strict=True)))
    if isinstance(left, float):
        return isinstance(right, float) and struct.pack("<d", left) == struct.pack("<d", right)
    return type(left) is type(right) and left == right


def difference(left, right):
    return max((abs(a - b) for a, b in zip(flat(left), flat(right), strict=True)), default=0.0)


def transpose(matrix):
    return [list(row) for row in zip(*matrix, strict=True)]


def multiply(left, right):
    return [[sum(a * b for a, b in zip(row, column, strict=True))
             for column in zip(*right, strict=True)] for row in left]


def matvec(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector, strict=True)) for row in matrix]


def rotation(vector):
    angle = math.sqrt(sum(v * v for v in vector))
    identity = [[float(i == j) for j in range(3)] for i in range(3)]
    if angle == 0.0:
        return identity
    x, y, z = [v / angle for v in vector]
    skew = [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]
    squared = multiply(skew, skew)
    return [[identity[i][j] + math.sin(angle) * skew[i][j] +
             (1.0 - math.cos(angle)) * squared[i][j] for j in range(3)] for i in range(3)]


def expected_control(raw, arm):
    """Independently express the frozen CV equations using scalar Python arithmetic."""
    age = raw["control_time_s"] - raw["measurement_time_s"]
    expected = dict(raw)
    if arm == "none":
        return expected, "disabled", 0.0
    if age <= 1e-12:
        return expected, "bypassed", 0.0
    if age > 0.05 + 1e-12:
        return expected, "horizon_exceeded", 0.0
    delta = [v * age for v in raw["joint_velocity"]]
    joint = [q + dq for q, dq in zip(raw["joint_position"], delta, strict=True)]
    low, high = [-0.785398, -1.8326, -2.775], [0.785398, 3.40339, -0.855]
    if any(abs(delta[i]) > 0.35 or joint[i] < low[i % 4] or joint[i] > high[i % 4]
           for i in LEGS):
        return expected, "kinematic_horizon_exceeded", 0.0
    old_rotation = raw["base_rotation"]
    angular_step = [v * age for v in raw["base_angular_velocity_body"]]
    predicted_rotation = multiply(old_rotation, rotation(angular_step))
    midpoint = multiply(old_rotation, rotation([0.5 * v for v in angular_step]))
    velocity = matvec(midpoint, raw["base_linear_velocity_body"])
    position = [p + v * age for p, v in zip(raw["base_position"], velocity, strict=True)]
    feet, jacobians = [], []
    for leg in range(4):
        offset_world = [p - b for p, b in zip(raw["foot_position"][leg],
                                             raw["base_position"], strict=True)]
        offset_body = matvec(transpose(old_rotation), offset_world)
        jacobian_body = multiply(transpose(old_rotation), raw["foot_jacobian"][leg])
        foot_velocity = matvec(jacobian_body, raw["joint_velocity"][4 * leg:4 * leg + 4])
        predicted_offset = [p + v * age for p, v in zip(offset_body, foot_velocity, strict=True)]
        feet.append([p + b for p, b in zip(position, matvec(predicted_rotation, predicted_offset),
                                           strict=True)])
        jacobians.append(multiply(predicted_rotation, jacobian_body))
    if arm == CANDIDATE:
        for index in LEGS:
            joint[index] = raw["joint_position"][index]
    expected.update(base_position=position, base_rotation=predicted_rotation,
                    base_linear_velocity_world=matvec(predicted_rotation,
                                                       raw["base_linear_velocity_body"]),
                    base_angular_velocity_world=matvec(predicted_rotation,
                                                        raw["base_angular_velocity_body"]),
                    joint_position=joint, foot_position=feet, foot_jacobian=jacobians)
    return expected, "applied", age


def read_npy(archive, name):
    with archive.open(name + ".npy") as handle:
        check(handle.read(6) == b"\x93NUMPY", "invalid NPY magic")
        major, minor = handle.read(2)
        check(major in (1, 2, 3) and minor == 0, "unsupported NPY version")
        length = struct.unpack("<H" if major == 1 else "<I", handle.read(2 if major == 1 else 4))[0]
        header = ast.literal_eval(handle.read(length).decode("utf-8" if major == 3 else "latin1"))
        check(header["descr"] == "<f8" and not header["fortran_order"], "unexpected trajectory dtype")
        payload = handle.read()
        check(len(payload) == math.prod(header["shape"]) * 8, "trajectory payload size mismatch")
        return header["shape"], payload


def verify_recorded_array(archive_arrays, name, index, value):
    shape, payload = archive_arrays[name]
    width = math.prod(shape[1:])
    numbers = flat(value)
    check(len(numbers) == width, f"{name}: record shape mismatch")
    encoded = struct.pack("<" + "d" * width, *numbers)
    check(payload[index * width * 8:(index + 1) * width * 8] == encoded,
          f"{name}: JSONL/NPZ differs at record {index}")


def audit_case(directory, case):
    stem = (f"{case['terrain']}_{case['noise']}_delay{case['delay_steps']}_"
            f"seed{case['seed']}_{case['arm']}")
    with zipfile.ZipFile(directory / (stem + "_trajectory.npz")) as archive:
        arrays = {name: read_npy(archive, name) for name in
                  ("time_s", "qpos", "qvel", "command_torque_nm", "leg_pd_unclipped_nm")}
    count, status_counts = 0, Counter()
    max_prediction, max_pd_error, max_pd_delta, max_pd_identity_error = 0.0, 0.0, 0.0, 0.0
    with (directory / (stem + "_control_state.jsonl")).open() as handle:
        initial = json.loads(next(handle))
        check(initial["record_type"] == "reset", stem + ": missing reset")
        for key in ("qpos", "qvel"):
            verify_recorded_array(arrays, key, 0, initial[key])
        verify_recorded_array(arrays, "time_s", 0, initial["simulation_time_s"])
        for index, line in enumerate(handle, start=1):
            record = json.loads(line)
            check(record["record_type"] == "transition" and record["step"] == index,
                  stem + ": invalid control sequence")
            count += 1
            raw, applied = record["raw_control_state"], record["applied_control_state"]
            expected, status, horizon = expected_control(raw, case["arm"])
            status_counts[status] += 1
            check(record["compensation_status"] == status, stem + ": original guard differs")
            check(record["compensation_horizon_s"] == horizon, stem + ": horizon differs")
            check(set(applied) == set(expected), stem + ": state fields differ")
            for key, value in expected.items():
                if status == "applied" and key in CHANGED:
                    error = difference(value, applied[key])
                    max_prediction = max(max_prediction, error)
                    check(error <= 1e-10, stem + ": CV equation mismatch in " + key)
                else:
                    check(exact(value, applied[key]), stem + ": unchanged field differs: " + key)
            if case["arm"] == CANDIDATE:
                check(all(exact(raw["joint_position"][i], applied["joint_position"][i])
                          for i in LEGS), stem + ": leg q was not retained bitwise")
            times = record["control_input_times"]
            check(times["measurement_time_s"] == raw["measurement_time_s"] and
                  times["control_time_s"] == raw["control_time_s"] and
                  times["state_age_s"] == raw["control_time_s"] - raw["measurement_time_s"] and
                  times["contact_geometry_time_s"] == raw["measurement_time_s"],
                  stem + ": time provenance differs")
            if case["arm"] == CANDIDATE:
                check(times["leg_joint_position_time_s"] == raw["measurement_time_s"],
                      stem + ": candidate leg q time differs")
            nominal = [0.0, 0.8, -1.5, 0.0] * 4
            pd_applied = [40.0 * (nominal[i] - applied["joint_position"][i]) -
                          1.5 * applied["joint_velocity"][i] if i in LEGS else 0.0
                          for i in range(16)]
            pd_raw = [40.0 * (nominal[i] - raw["joint_position"][i]) -
                      1.5 * raw["joint_velocity"][i] if i in LEGS else 0.0 for i in range(16)]
            delta = [a - b for a, b in zip(pd_applied, pd_raw, strict=True)]
            error = max(difference(pd_applied, record["leg_pd_unclipped_nm"]),
                        difference(pd_raw, record["measured_q_leg_pd_unclipped_nm"]),
                        difference(delta, record["leg_pd_delta_from_measured_q_nm"]))
            max_pd_error = max(max_pd_error, error)
            check(error <= 1e-12, stem + ": PD recorded components differ")
            expected_delta = [-40.0 * (applied["joint_position"][i] - raw["joint_position"][i])
                              if i in LEGS else 0.0 for i in range(16)]
            max_pd_identity_error = max(max_pd_identity_error, difference(delta, expected_delta))
            max_pd_delta = max(max_pd_delta, max(abs(v) for v in delta))
            if case["arm"] in ("none", CANDIDATE):
                check(all(v == 0.0 for v in delta), stem + ": unwanted added PD remains")
            for key in ("qpos", "qvel"):
                check(all(math.isfinite(v) for v in record[key]), stem + ": nonfinite state")
                verify_recorded_array(arrays, key, index, record[key])
            verify_recorded_array(arrays, "time_s", index, record["simulation_time_s"])
            verify_recorded_array(arrays, "command_torque_nm", index - 1, record["command_torque_nm"])
            verify_recorded_array(arrays, "leg_pd_unclipped_nm", index - 1, record["leg_pd_unclipped_nm"])
    check(count == case["steps"], stem + ": state record count differs")
    check(dict(status_counts) == case["compensation_status_counts"], stem + ": status counts differ")
    check(arrays["qpos"][0] == (count + 1, 23) and arrays["qvel"][0] == (count + 1, 22),
          stem + ": full state dimensions differ")
    for name in ("command_torque_nm", "leg_pd_unclipped_nm"):
        check(arrays[name][0] == (count, 16), stem + ": torque dimensions differ")
    check(arrays["time_s"][0] == (count + 1,), stem + ": trajectory time count differs")
    return {"case": stem, "arm": case["arm"], "control_records": count,
            "state_records_with_initial": count + 1, "guard_status_counts": dict(status_counts),
            "max_prediction_abs_difference": max_prediction, "max_pd_record_error_nm": max_pd_error,
            "max_pd_delta_identity_error_nm": max_pd_identity_error,
            "max_abs_added_pd_nm": max_pd_delta, "passed": True}


def audit():
    prereg_path = ROOT / "docs/experiments50/estimator_latency50_preregistered.json"
    prereg = json.loads(prereg_path.read_text())
    for relative, expected in prereg["source_sha256"].items():
        check(digest(ROOT / relative) == expected, "source changed: " + relative)
    artifacts, summaries = {}, {}
    for phase in ("offline", "preflight"):
        directory = ROOT / f"results/estimator50_{phase}_20261024_01"
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text())
        check(summary["status"] == "complete", phase + ": execution incomplete")
        check(summary["preregistration_sha256"] == digest(prereg_path), phase + ": prereg differs")
        check(summary["script_sha256"] == digest(ROOT / "scripts/estimator_candidate50.py"),
              phase + ": executed script differs")
        for relative, expected in summary["output_artifact_sha256"].items():
            path = directory / relative
            check(digest(path) == expected, "output artifact changed: " + str(path))
            artifacts[str(path.relative_to(ROOT))] = expected
        summaries[phase] = {"path": str(summary_path.relative_to(ROOT)), "sha256": digest(summary_path)}
    directory = ROOT / "results/estimator50_preflight_20261024_01"
    summary = json.loads((directory / "summary.json").read_text())
    cases = [audit_case(directory, case) for case in summary["cases"]]
    return {"schema": "estimator50-independent-state-readback-v1", "passed": True,
            "preregistration_sha256": digest(prereg_path), "input_source_count": len(prereg["source_sha256"]),
            "verified_output_artifact_count": len(artifacts), "output_artifact_sha256": artifacts,
            "summaries": summaries, "cases": cases,
            "checked_control_records": sum(case["control_records"] for case in cases),
            "checked_state_records_with_initial": sum(case["state_records_with_initial"] for case in cases),
            "verification_boundary": "Recorded state/torque/NPZ consistency and independently expressed CV equations. Unchanged fields and measured candidate leg q use IEEE bit comparison; predicted floating-point matrix values use abs<=1e-10 because scalar Python and NumPy arithmetic can round differently. Executed offline bitwise-CV assertions remain a distinct source of evidence. No physics replay or controller performance qualification is performed.",
            "new_training_controls": 0, "new_simulation_controls": 0, "model_constructions": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), "readback output already exists; never overwrite")
    try:
        result = audit()
    except BaseException:
        result = {"schema": "estimator50-independent-state-readback-v1", "passed": False,
                  "traceback": traceback.format_exc()}
        raise
    finally:
        result["reader_sha256"] = digest(Path(__file__))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("output_artifact_sha256", "cases")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
