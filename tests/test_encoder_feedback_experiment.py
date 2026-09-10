from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import tarfile

import numpy as np
import pytest

from scripts import evaluate_encoder_feedback as experiment
from wheel_legged_control.actuator_bench import ActuatorParameters, BenchConfig
from wheel_legged_control.wheel_control import WheelControlConfig, WheelVelocityController


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def numeric(rows, name):
    return np.asarray([float(row[name]) for row in rows])


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    output = tmp_path_factory.mktemp("encoder_feedback") / "study"
    args = argparse.Namespace(
        output=output,
        seeds=[101],
        scenarios=["matched"],
        conditions=list(experiment.CONDITIONS),
        duration=1.0,
        calibration_duration=0.5,
        max_delay_steps=0,
        max_nfev=1,
    )
    fitter = experiment.fit_encoder_parameters
    seen = []

    def calibration_only(logs, **kwargs):
        assert len(logs) == 2
        assert all(log.split == "calibration" and log.known_initial_rest is True for log in logs)
        assert all(not hasattr(log, "velocities") and not hasattr(log, "truth") for log in logs)
        assert set(kwargs) == {"max_delay_steps", "max_nfev"}
        seen.append(logs)
        return fitter(logs, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(experiment, "fit_encoder_parameters", calibration_only)
        experiment.run(args)
    assert len(seen) == 1
    return output


def test_real_experiment_complete_matrix_and_honest_fit_status(study):
    summary = json.loads((study / "summary.json").read_text())
    protocol = json.loads((study / "protocol.json").read_text())
    assert protocol["expected_control_cases"] == len(summary["metrics"]) == 27
    keys = {(r["condition"], r["profile"], r["feedback"]) for r in summary["metrics"]}
    assert keys == {
        (c, p, f)
        for c in experiment.CONDITIONS
        for p in experiment.PROFILES
        for f in experiment.FEEDBACKS
    }
    assert all(r["steps"] == 500 and r["completed"] for r in summary["metrics"])
    assert summary["status"] == "completed"
    assert len(summary["fits"]) == 1
    assert not summary["fits"][0]["fit_status"]["optimizer_success"]
    assert summary["fits"][0]["candidates"][0]["nfev"] == 1
    assert protocol["calibration_seed"] == 89
    assert protocol["generator_truth_not_available_to_observer"]["delay_steps"] == 2
    assert "not equivalent" in protocol["reference_sensor_scope"]


def test_calibration_is_position_only_and_separate_from_feedback_noise(study):
    directory = study / "matched_calibration"
    assert sorted(p.name for p in directory.iterdir()) == [
        "calibration_0.csv",
        "calibration_1.csv",
        "fit.json",
    ]
    for path in directory.glob("*.csv"):
        with path.open() as stream:
            metadata = json.loads(stream.readline()[2:])
            reader = csv.DictReader(stream)
            rows = list(reader)
        assert metadata["known_initial_rest"] is True
        assert metadata["split"] == "calibration"
        assert set(reader.fieldnames) == {
            "time_s",
            "command_torque_nm",
            "position_rad",
            "next_position_rad",
        }
        assert len(rows) == 250
        assert not any(
            word in json.dumps(metadata) for word in ("velocity", "truth", "armature", "stribeck")
        )


@pytest.mark.parametrize("condition", list(experiment.CONDITIONS))
@pytest.mark.parametrize("feedback", experiment.FEEDBACKS)
def test_raw_feedback_replays_causal_math_sensor_pairing_and_sample_age(study, condition, feedback):
    rows = read_rows(study / f"matched_{condition}_seed101" / f"tracking_{feedback}.csv.gz")
    sensor = experiment.CONDITIONS[condition]
    n, dt = len(rows), 0.002
    true_v = numeric(rows, "actual_velocity_rad_s")
    qnoise = np.random.default_rng(np.random.SeedSequence((101, 401))).normal(size=n)
    vnoise = np.random.default_rng(np.random.SeedSequence((101, 402))).normal(size=n)
    measured_q = numeric(rows, "position_rad") + qnoise * 0.0002
    quantum = sensor["quantization_rad"]
    if quantum:
        measured_q = np.rint(measured_q / quantum) * quantum
    np.testing.assert_array_equal(numeric(rows, "position_measurement_rad"), measured_q)
    delivered = np.maximum(0, np.arange(n) - sensor["delay_steps"])
    np.testing.assert_array_equal(numeric(rows, "delivered_sample_index"), delivered)
    np.testing.assert_allclose(numeric(rows, "sample_time_s"), delivered * dt, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        numeric(rows, "sample_age_s"), (np.arange(n) - delivered) * dt, rtol=0, atol=1e-14
    )
    np.testing.assert_array_equal(numeric(rows, "delivered_position_rad"), measured_q[delivered])
    np.testing.assert_array_equal(
        numeric(rows, "actual_velocity_at_sample_rad_s"), true_v[delivered]
    )
    observed_q, observed_v, previous_q, last = measured_q[0], 0.0, measured_q[0], 0
    expected_velocity, expected_position = [], []
    for k, sample in enumerate(delivered):
        assert (rows[k]["measurement_updated"] == "True") == (k == 0 or sample != last)
        if feedback == "synthetic_velocity":
            observed_q = measured_q[sample]
            observed_v = 0.0 if sample == 0 else true_v[sample] + 0.002 * vnoise[sample]
        elif sample != last:
            if feedback == "difference_lowpass":
                retention = math.exp(-2 * math.pi * 20 * dt)
                observed_v = (
                    retention * observed_v
                    + (1 - retention) * (measured_q[sample] - previous_q) / dt
                )
                observed_q = measured_q[sample]
            else:
                prediction = observed_q + dt * observed_v
                innovation = measured_q[sample] - prediction
                observed_q = prediction + 0.25 * innovation
                observed_v += 0.04 * innovation / dt
            previous_q = measured_q[sample]
        last = sample
        expected_velocity.append(observed_v)
        expected_position.append(observed_q)
    np.testing.assert_allclose(
        numeric(rows, "measured_velocity_rad_s"), expected_velocity, rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        numeric(rows, "estimated_position_rad"), expected_position, rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        numeric(rows, "estimation_error_at_sample_rad_s"),
        np.array(expected_velocity) - true_v[delivered],
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        numeric(rows, "estimation_error_current_rad_s"),
        np.array(expected_velocity) - true_v,
        rtol=1e-11,
        atol=1e-12,
    )


def test_every_control_metric_and_physical_timing_recomputes_from_raw_records(study):
    summary = json.loads((study / "summary.json").read_text())
    compressed = read_rows(study / "metrics.csv.gz")
    assert len(compressed) == len(summary["metrics"])
    for metric, saved in zip(summary["metrics"], compressed):
        rows = read_rows(
            study
            / f"matched_{metric['condition']}_seed101"
            / f"{metric['profile']}_{metric['feedback']}.csv.gz"
        )
        dt = 0.002
        true = numeric(rows, "actual_velocity_rad_s")
        ref = numeric(rows, "reference_velocity_rad_s")
        measured = numeric(rows, "measured_velocity_rad_s")
        error = ref - true
        command = numeric(rows, "command_torque_nm")
        requested = numeric(rows, "requested_torque_nm")
        applied = numeric(rows, "applied_torque_nm")
        np.testing.assert_array_equal(true[1:], numeric(rows, "next_velocity_rad_s")[:-1])
        np.testing.assert_array_equal(
            numeric(rows, "position_rad")[1:], numeric(rows, "next_position_rad")[:-1]
        )
        np.testing.assert_allclose(numeric(rows, "tracking_error_rad_s"), error, rtol=0, atol=1e-14)
        np.testing.assert_allclose(numeric(rows, "error_rad_s"), ref - measured, rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            numeric(rows, "proportional_torque_nm"), 0.12 * (ref - measured), rtol=0, atol=1e-14
        )
        np.testing.assert_array_equal(command, np.clip(requested, -2.0, 2.0))
        np.testing.assert_array_equal(applied, np.r_[0.0, 0.0, command[:-2]])
        computed = {
            "velocity_rmse_rad_s": float(np.sqrt(np.mean(error**2))),
            "velocity_mae_rad_s": float(np.mean(np.abs(error))),
            "peak_velocity_error_rad_s": float(np.max(np.abs(error))),
            "integrated_absolute_error_rad": float(np.sum(np.abs(error)) * dt),
            "command_torque_rms_nm": float(np.sqrt(np.mean(command**2))),
            "applied_torque_rms_nm": float(np.sqrt(np.mean(applied**2))),
            "peak_requested_torque_nm": float(np.max(np.abs(requested))),
            "saturation_fraction": float(np.mean(np.abs(requested) > 2)),
            "integration_frozen_fraction": float(
                np.mean([r["integration_frozen"] == "True" for r in rows])
            ),
            "steps": len(rows),
            "duration_s": len(rows) * dt,
        }
        for timing in ("at_sample", "current"):
            e = numeric(rows, f"estimation_error_{timing}_rad_s")
            computed[f"estimation_rmse_{timing}_rad_s"] = float(np.sqrt(np.mean(e**2)))
        for name, expected in computed.items():
            assert metric[name] == pytest.approx(expected, rel=1e-11, abs=1e-12)
            assert float(saved[name]) == pytest.approx(expected, rel=1e-11, abs=1e-12)
        assert not metric["speed_guard_triggered"]


def test_raw_records_manifest_and_actual_source_archive_are_complete(study):
    manifest = json.loads((study / "manifest.json").read_text())["sha256"]
    files = {str(p.relative_to(study)) for p in study.rglob("*") if p.is_file()}
    assert set(manifest) == files - {"manifest.json"}
    for name, digest in manifest.items():
        assert hashlib.sha256((study / name).read_bytes()).hexdigest() == digest
    source = json.loads((study / "source.json").read_text())["sha256"]
    assert len(source) == 15
    assert set(source) == set(experiment.SOURCE_FILES)
    with tarfile.open(study / "source.tar.gz") as archive:
        assert len(archive.getmembers()) == len(source)
        for member in archive.getmembers():
            assert member.isfile()
            payload = archive.extractfile(member).read()
            assert hashlib.sha256(payload).hexdigest() == source[member.name]
            assert payload == (experiment.ROOT / member.name).read_bytes()
    assert json.loads((study / "source_consistency.json").read_text()) == {
        "unchanged": True,
        "changed": [],
    }
    assert len(list(study.rglob("*.png"))) == 9


def loop(**kwargs):
    return experiment.run_feedback_loop(
        BenchConfig(),
        ActuatorParameters(0.017, 0.105, 0.043, 2),
        WheelVelocityController(WheelControlConfig(), ActuatorParameters()),
        feedback=kwargs.pop("feedback", "alpha_beta"),
        condition=kwargs.pop("condition", "noisy"),
        profile="tracking",
        duration=1.0,
        seed=101,
        **kwargs,
    )


def test_future_measurement_changes_cannot_change_earlier_control(monkeypatch):
    original = loop()
    real_rng = np.random.default_rng

    class FutureNoise:
        def __init__(self, seed):
            self.generator = real_rng(seed)

        def normal(self, *, size):
            values = self.generator.normal(size=size)
            values[200:] += 10.0
            return values

    monkeypatch.setattr(np.random, "default_rng", FutureNoise)
    modified = loop()
    assert original[:200] == modified[:200]
    assert original[200:] != modified[200:]


def test_speed_guard_is_not_reported_as_completed(monkeypatch):
    monkeypatch.setattr(experiment, "SPEED_GUARD_RAD_S", 1e-10)
    rows = loop()
    assert rows[-1]["speed_guard_triggered"]
    assert len(rows) < 500
    assert not experiment.feedback_metrics(rows, 0.002, 1.0)["completed"]


def test_existing_output_is_never_overwritten(study):
    digest = hashlib.sha256((study / "summary.json").read_bytes()).hexdigest()
    with pytest.raises(SystemExit):
        experiment.main(["--output", str(study)])
    assert hashlib.sha256((study / "summary.json").read_bytes()).hexdigest() == digest


@pytest.mark.parametrize(
    "flags",
    [
        ["--duration", "nan"],
        ["--duration", "0.5"],
        ["--calibration-duration", "inf"],
        ["--max-delay-steps", "-1"],
        ["--max-nfev", "0"],
        ["--seeds", "1", "1"],
        ["--seeds", "-1"],
        ["--conditions", "noisy", "noisy"],
        ["--scenarios", "matched", "matched"],
    ],
)
def test_bad_cli_arguments_do_not_create_output(tmp_path, flags):
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit):
        experiment.main(["--output", str(output), *flags])
    assert not output.exists()
