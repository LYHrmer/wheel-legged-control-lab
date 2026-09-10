"""Independent, read-only audit of encoder-feedback-v1; never imports the runner."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np

FEEDBACKS = ("synthetic_velocity", "difference_lowpass", "alpha_beta")
PROFILES = ("tracking", "reversal", "stress")
CONDITIONS = {
    "noisy": {"position_noise_std_rad": 0.0002, "quantization_rad": 0.0, "delay_steps": 0},
    "quantized": {"position_noise_std_rad": 0.0002, "quantization_rad": 2 * math.pi / 4096, "delay_steps": 0},
    "delayed": {"position_noise_std_rad": 0.0002, "quantization_rad": 0.0, "delay_steps": 5},
}
SOURCE_NAMES = {
    "scripts/evaluate_encoder_identification.py", "scripts/evaluate_encoder_feedback.py",
    "scripts/evaluate_wheel_control.py", "scripts/run_actuator_identification.py",
    "src/wheel_legged_control/__init__.py", "src/wheel_legged_control/controllers.py",
    "src/wheel_legged_control/model.py", "src/wheel_legged_control/rewards.py",
    "src/wheel_legged_control/encoder_identification.py", "src/wheel_legged_control/encoder_velocity.py",
    "src/wheel_legged_control/actuator_identification.py", "src/wheel_legged_control/actuator_bench.py",
    "src/wheel_legged_control/wheel_control.py", "src/wheel_legged_control/provenance.py", "pyproject.toml",
}
DT = 0.002

if not __debug__:
    raise RuntimeError("Run this independent arithmetic audit without python -O")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with path.open() as stream:
        return json.load(stream)


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def col(rows, name):
    result = np.array([float(row[name]) for row in rows])
    assert np.isfinite(result).all(), name
    return result


def close(actual, expected, label):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11, err_msg=label)


def rms(values):
    return math.sqrt(math.fsum(float(value) ** 2 for value in values) / len(values))


def key(row):
    return row["scenario"], row["condition"], int(row["seed"]), row["profile"], row["feedback"]


def check_case(rows, metric, csv_metric, parameters, duration):
    scenario, condition, seed, profile, feedback = key(metric)
    n, expected_n = len(rows), round(duration / DT)
    assert 0 < n <= expected_n
    sensor = CONDITIONS[condition]
    times = np.arange(n) * DT
    measured_q = col(rows, "position_rad") + np.random.default_rng(
        np.random.SeedSequence((seed, 401))
    ).normal(size=expected_n)[:n] * sensor["position_noise_std_rad"]
    if sensor["quantization_rad"]:
        measured_q = np.rint(measured_q / sensor["quantization_rad"]) * sensor["quantization_rad"]
    np.testing.assert_array_equal(col(rows, "position_measurement_rad"), measured_q)
    close(col(rows, "time_s"), times, "physical clock")
    delivered = np.maximum(0, np.arange(n) - sensor["delay_steps"])
    np.testing.assert_array_equal(col(rows, "delivered_sample_index"), delivered)
    close(col(rows, "sample_time_s"), delivered * DT, "sampling clock")
    close(col(rows, "sample_age_s"), times - delivered * DT, "measurement age")
    np.testing.assert_array_equal(col(rows, "delivered_position_rad"), measured_q[delivered])
    velocity = col(rows, "actual_velocity_rad_s")
    np.testing.assert_array_equal(col(rows, "actual_velocity_at_sample_rad_s"), velocity[delivered])
    np.testing.assert_array_equal(velocity[1:], col(rows, "next_velocity_rad_s")[:-1])
    np.testing.assert_array_equal(col(rows, "position_rad")[1:], col(rows, "next_position_rad")[:-1])
    assert rows[0]["position_rad"] == "0.0" and velocity[0] == 0.0

    v_noise = np.random.default_rng(np.random.SeedSequence((seed, 402))).normal(size=expected_n)[:n] * .002
    reference_sensor = velocity + v_noise
    reference_sensor[0] = 0.0
    qhat, vhat, previous_q, last = measured_q[0], 0.0, measured_q[0], 0
    q_estimates, v_estimates = [], []
    for index, sample in enumerate(delivered):
        assert (rows[index]["measurement_updated"] == "True") == (index == 0 or sample != last)
        if feedback == "synthetic_velocity":
            qhat, vhat = measured_q[sample], reference_sensor[sample]
        elif sample != last:
            if feedback == "difference_lowpass":
                decay = math.exp(-2 * math.pi * .04)
                vhat = decay * vhat + (1 - decay) * (measured_q[sample] - previous_q) / DT
                qhat = measured_q[sample]
            else:
                predicted = qhat + DT * vhat
                residual = measured_q[sample] - predicted
                qhat = predicted + .25 * residual
                vhat += .04 * residual / DT
        previous_q, last = measured_q[sample], sample
        q_estimates.append(qhat)
        v_estimates.append(vhat)
    close(col(rows, "estimated_position_rad"), q_estimates, "observer position")
    close(col(rows, "measured_velocity_rad_s"), v_estimates, "observer velocity")
    if profile == "reversal":
        phase = 2 * math.pi * .35 * times
        inner = 3 * np.sin(phase)
        target = 2.5 * np.tanh(inner)
        acceleration = 2.5 * 3 * 2 * math.pi * .35 * np.cos(phase) / np.cosh(inner) ** 2
    else:
        frequency, amplitude = (.45, 3.) if profile == "tracking" else (1., 18.)
        phase = 2 * math.pi * frequency * times
        target, acceleration = amplitude * np.sin(phase), amplitude * 2 * math.pi * frequency * np.cos(phase)
    close(col(rows, "reference_velocity_rad_s"), target, "paired reference")
    close(col(rows, "reference_acceleration_rad_s2"), acceleration, "paired acceleration")

    integral = 0.0
    for row in rows:
        target_v = float(row["reference_velocity_rad_s"])
        error = target_v - float(row["measured_velocity_rad_s"])
        p = .12 * error
        increment = .6 * error * DT
        candidate_i = integral + increment
        ff = ((.5 * .5 * .087**2 + parameters["armature"]) * float(row["reference_acceleration_rad_s2"])
              + parameters["damping"] * target_v + parameters["coulomb_friction"] * math.tanh(target_v / .05))
        candidate = p + candidate_i + ff
        frozen = (candidate > 2 and increment > 0) or (candidate < -2 and increment < 0)
        integral = integral if frozen else candidate_i
        requested = p + integral + ff
        for field, expected in (("error_rad_s", error), ("proportional_torque_nm", p),
                                ("integral_torque_nm", integral), ("feedforward_torque_nm", ff),
                                ("requested_torque_nm", requested), ("command_torque_nm", max(-2., min(2., requested)))):
            assert math.isclose(float(row[field]), expected, rel_tol=1e-10, abs_tol=1e-11), field
        assert (row["integration_frozen"] == "True") == frozen
        assert (row["saturated"] == "True") == (abs(requested) > 2.)
    command, requested, applied = (col(rows, name) for name in ("command_torque_nm", "requested_torque_nm", "applied_torque_nm"))
    np.testing.assert_array_equal(applied, np.r_[0., 0., command][:n])
    tracking_error = target - velocity
    sample_error = np.array(v_estimates) - velocity[delivered]
    current_error = np.array(v_estimates) - velocity
    close(col(rows, "tracking_error_rad_s"), tracking_error, "time-aligned tracking error")
    close(col(rows, "estimation_error_at_sample_rad_s"), sample_error, "sample-aligned estimation error")
    close(col(rows, "estimation_error_current_rad_s"), current_error, "current estimation error")
    guards = np.abs(col(rows, "next_velocity_rad_s")) > 30.
    assert not guards[:-1].any()
    assert all((row["speed_guard_triggered"] == "True") == bool(guard) for row, guard in zip(rows, guards))
    complete = n == expected_n and not bool(guards.any())
    values = {
        "steps": n, "duration_s": n * DT, "velocity_rmse_rad_s": rms(tracking_error),
        "velocity_mae_rad_s": math.fsum(abs(float(x)) for x in tracking_error) / n,
        "peak_velocity_error_rad_s": max(abs(tracking_error)),
        "integrated_absolute_error_rad": math.fsum(abs(float(x)) for x in tracking_error) * DT,
        "command_torque_rms_nm": rms(command), "applied_torque_rms_nm": rms(applied),
        "peak_requested_torque_nm": max(abs(requested)), "saturation_fraction": sum(abs(requested) > 2.) / n,
        "integration_frozen_fraction": sum(row["integration_frozen"] == "True" for row in rows) / n,
        "estimation_rmse_at_sample_rad_s": rms(sample_error), "estimation_rmse_current_rad_s": rms(current_error),
    }
    for name, value in values.items():
        close(metric[name], value, name)
        close(float(csv_metric[name]), value, "saved " + name)
    assert metric["completed"] is complete and (csv_metric["completed"] == "True") is complete
    assert metric["speed_guard_triggered"] is bool(guards.any())
    assert (csv_metric["speed_guard_triggered"] == "True") is bool(guards.any())
    return n


def audit(directory, allow_subset=False):
    directory = directory.resolve()
    protocol, summary = (read_json(directory / name) for name in ("protocol.json", "summary.json"))
    args = protocol["arguments"]
    assert protocol["schema"] == "encoder-feedback-v1" and protocol["data_origin"] == "synthetic_only"
    assert protocol["calibration_seed"] == 89
    assert protocol["bench"] == {"kind": "wheel", "dt": DT, "torque_limit_nm": 2.}
    assert protocol["controller_all_arms"] == {"kp": .12, "ki": .6, "dt": DT, "torque_limit_nm": 2.}
    assert protocol["generator_truth_not_available_to_observer"] == {"armature": .017, "damping": .105, "coulomb_friction": .043, "delay_steps": 2}
    assert protocol["extra_stribeck_nm"] == .08 and protocol["synthetic_velocity_noise_std_rad_s"] == .002
    assert protocol["feedbacks"] == list(FEEDBACKS) and protocol["profiles"] == list(PROFILES)
    assert protocol["sensor_conditions"] == {name: CONDITIONS[name] for name in args["conditions"]}
    assert protocol["observer_configs"] == {name: {"method": name, "dt": DT, "cutoff_hz": 20., "alpha": .25, "beta": .04} for name in FEEDBACKS[1:]}
    if not allow_subset:
        assert args["scenarios"] == ["matched", "mismatch"] and args["conditions"] == list(CONDITIONS)
        assert args["seeds"] == [101, 103, 107] and args["duration"] == 6. and args["calibration_duration"] == 3.
        assert args["max_delay_steps"] == 4 and args["max_nfev"] == 30
    expected = set(itertools.product(args["scenarios"], args["conditions"], args["seeds"], PROFILES, FEEDBACKS))
    metrics = {key(row): row for row in summary["metrics"]}
    saved_rows = read_rows(directory / "metrics.csv.gz")
    saved = {key(row): row for row in saved_rows}
    assert len(summary["metrics"]) == len(metrics) == protocol["expected_control_cases"] == len(expected)
    assert len(saved_rows) == len(saved)
    assert set(metrics) == set(saved) == expected
    manifest = read_json(directory / "manifest.json")["sha256"]
    disk_files = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    # The owner explicitly added README after the experiment. It is not silently
    # retrofitted into the generation-time manifest or treated as generated data.
    post_experiment_docs = disk_files - set(manifest) - {"manifest.json"}
    assert post_experiment_docs <= {"README.md"}
    assert set(manifest) <= disk_files
    for name, digest in manifest.items():
        path = PurePosixPath(name)
        assert not path.is_absolute() and ".." not in path.parts and "\\" not in name
        assert not (directory / name).is_symlink() and sha256(directory / name) == digest
    source = read_json(directory / "source.json")["sha256"]
    assert set(source) == SOURCE_NAMES
    with tarfile.open(directory / "source.tar.gz") as archive:
        members = archive.getmembers()
        assert len(members) == len({member.name for member in members}) == 15
        assert {member.name for member in members} == SOURCE_NAMES
        for member in members:
            assert member.isfile()
            data = archive.extractfile(member).read()
            assert len(data) == member.size and hashlib.sha256(data).hexdigest() == source[member.name]
    assert read_json(directory / "source_consistency.json") == {"unchanged": True, "changed": []}
    fits, fit_report = {}, []
    for scenario in args["scenarios"]:
        fit = read_json(directory / (scenario + "_calibration") / "fit.json")
        assert fit in summary["fits"] and fit["scenario"] == scenario
        candidates = fit["candidates"]
        assert [row["parameters"]["delay_steps"] for row in candidates] == list(range(args["max_delay_steps"] + 1))
        best = min(candidates, key=lambda row: row["calibration_loss"])
        assert fit["parameters"] == best["parameters"]
        assert fit["fit_status"]["optimizer_success"] is best["optimizer_success"]
        assert all(isinstance(c["nfev"], int) and 1 <= c["nfev"] <= args["max_nfev"] for c in candidates)
        for index in range(2):
            with (directory / (scenario + "_calibration") / ("calibration_" + str(index) + ".csv")).open() as stream:
                meta = json.loads(stream.readline()[2:])
                reader = csv.DictReader(stream)
                calibration = list(reader)
            assert meta == {"schema": "encoder_position_only", "schema_version": 1,
                            "config": {"kind": "wheel", "dt": DT, "torque_limit_nm": 2.},
                            "name": "calibration_" + str(index), "split": "calibration",
                            "sampling": "position_then_command_then_next_position",
                            "initial_command_history": "zero", "initial_state": "known_rest",
                            "known_initial_rest": True}
            assert reader.fieldnames == ["time_s", "command_torque_nm", "position_rad", "next_position_rad"]
            count = round(args["calibration_duration"] / DT)
            assert len(calibration) == count
            close(col(calibration, "time_s"), np.arange(count) * DT, "calibration clock")
            np.testing.assert_array_equal(col(calibration, "position_rad")[1:], col(calibration, "next_position_rad")[:-1])
            first_q = np.random.default_rng(np.random.SeedSequence((89, index))).normal(size=(count + 1, 2))[0, 0] * .0002
            assert float(calibration[0]["position_rad"]) == first_q
        fits[scenario] = fit["parameters"]
        fit_report.append({"scenario": scenario, "parameters": fit["parameters"], "selected_success": best["optimizer_success"], "candidate_nfev": [c["nfev"] for c in candidates], "candidate_success": [c["optimizer_success"] for c in candidates]})
    assert len(summary["fits"]) == len(fits)
    raw_count = 0
    for case, metric in metrics.items():
        scenario, condition, seed, profile, feedback = case
        path = directory / f"{scenario}_{condition}_seed{seed}" / f"{profile}_{feedback}.csv.gz"
        raw_count += check_case(read_rows(path), metric, saved[case], fits[scenario], args["duration"])
    complete = all(row["completed"] for row in metrics.values())
    assert summary["status"] == ("completed" if complete else "contains_incomplete_runs")
    comparisons = []
    for condition, feedback in itertools.product(args["conditions"], FEEDBACKS[1:]):
        selected = [(case, row) for case, row in metrics.items() if case[1] == condition and case[4] == feedback]
        paired = [(case, row, row["velocity_rmse_rad_s"] / metrics[case[:-1] + ("synthetic_velocity",)]["velocity_rmse_rad_s"]) for case, row in selected]
        worst = max(paired, key=lambda item: item[2])
        comparisons.append({"condition": condition, "feedback": feedback, "pairs": len(paired), "rmse_lower_count": sum(ratio < 1 - 1e-10 for _, _, ratio in paired), "rmse_higher_count": sum(ratio > 1 + 1e-10 for _, _, ratio in paired), "median_rmse_ratio_to_reference": float(np.median([ratio for _, _, ratio in paired])), "worst_rmse_ratio": worst[2], "worst_case": list(worst[0]), "max_saturation_fraction": max(row["saturation_fraction"] for _, row in selected), "max_current_estimation_rmse_rad_s": max(row["estimation_rmse_current_rad_s"] for _, row in selected)})
    return {"status": "verified", "directory": str(directory), "full_protocol_required": not allow_subset, "case_count": len(metrics), "raw_rows": raw_count, "source_members": len(source), "manifest_files": len(manifest), "post_experiment_docs_not_in_original_manifest": sorted(post_experiment_docs), "summary_status": summary["status"], "completed_cases": sum(row["completed"] for row in metrics.values()), "guard_cases": sum(row["speed_guard_triggered"] for row in metrics.values()), "fits": fit_report, "paired_comparisons": comparisons, "scope": "Checks raw arithmetic, timing, deterministic sensor draws, pairing, shared fit and archive integrity; does not rerun hidden physical dynamics, prove optimizer quality, or claim hardware/closed-loop robustness."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--allow-subset", action="store_true", help="only for preflight on the existing tiny run")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and (args.output.exists() or args.output.resolve().is_relative_to(args.directory.resolve())):
        parser.error("output must be a new file outside the audited directory")
    report = audit(args.directory, args.allow_subset)
    if args.output is not None:
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))
