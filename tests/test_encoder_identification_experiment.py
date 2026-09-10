"""Real small-budget experiment checks, not a control-performance benchmark."""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib
import json
import math
import tarfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("tracking", "reversal", "stress")
ARMS = ("pi", "nominal_ff", "position_ff", "position_velocity_ff")
SPLITS = ("calibration", "calibration", "validation", "holdout")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="module")
def cli_module() -> ModuleType:
    return importlib.import_module("scripts.evaluate_encoder_identification")


@pytest.fixture(scope="module")
def experiment(cli_module: ModuleType, tmp_path_factory: pytest.TempPathFactory):
    """Observe the log-only boundary while executing both real fitters and plots."""
    output = tmp_path_factory.mktemp("encoder_experiment") / "run"
    calls = {}
    encoder_type = cli_module.EncoderLog

    class VelocityForbiddenLog(encoder_type):
        @property
        def velocities(self):
            raise AssertionError("position-only fitting accessed measured velocity")

    def observe(name, fitter):
        def checked_fit(logs, **kwargs):
            assert name not in calls
            logs = tuple(logs)
            calls[name] = {"logs": logs, "kwargs": kwargs}
            if name == "position":
                logs = tuple(
                    VelocityForbiddenLog(
                        log.config,
                        log.commands,
                        log.positions,
                        known_initial_rest=log.known_initial_rest,
                        name=log.name,
                        split=log.split,
                    )
                    for log in logs
                )
            return fitter(logs, **kwargs)

        return checked_fit

    with pytest.MonkeyPatch.context() as patch:
        for name, attribute in (
            ("position", "fit_encoder_parameters"),
            ("position_velocity", "fit_actuator_parameters"),
        ):
            patch.setattr(cli_module, attribute, observe(name, getattr(cli_module, attribute)))
        cli_module.main(
            [
                "--output", str(output),
                "--scenario", "matched",
                "--seeds", "17",
                "--calibration-duration", "0.5",
                "--duration", "1",
                "--noise-scale", "1",
                "--max-delay-steps", "0",
                "--max-nfev", "1",
            ]
        )
    return output, calls


@pytest.fixture(scope="module")
def experiment_output(experiment) -> Path:
    return experiment[0]


def test_real_short_run_has_the_complete_declared_matrix(experiment_output: Path) -> None:
    protocol = read_json(experiment_output / "protocol.json")
    summary = read_json(experiment_output / "summary.json")
    assert protocol["schema"] == "position-only-bench-comparison-v1"
    assert protocol["data_origin"] == "synthetic_only"
    assert protocol["arguments"]["seeds"] == [17]
    assert protocol["arguments"]["duration"] == 1.0
    assert protocol["arguments"]["calibration_duration"] == 0.5
    assert protocol["bench"] == {"kind": "wheel", "dt": 0.002, "torque_limit_nm": 2.0}
    assert protocol["controller_all_arms"] == {
        "kp": 0.12, "ki": 0.6, "dt": 0.002, "torque_limit_nm": 2.0,
    }
    assert protocol["expected_control_cases"] == protocol["expected_prediction_rows"] == 12
    control = summary["control_metrics"]
    predictions = summary["prediction_metrics"]
    assert len(control) == len(predictions) == 12
    assert {(row["profile"], row["controller"]) for row in control} == {
        (profile, arm) for profile in PROFILES for arm in ARMS
    }
    assert {(row["trajectory"], row["model"]) for row in predictions} == {
        (f"{split}_{index}", model)
        for index, split in enumerate(SPLITS)
        for model in ("nominal", "position", "position_velocity")
    }
    assert {(row["scenario"], row["seed"]) for row in control + predictions} == {("matched", 17)}
    assert all(row["steps"] == 500 and row["duration_s"] == 1.0 for row in control)
    assert all(row["completed"] for row in control)
    assert summary["status"] == "completed"
    assert "synthetic velocity" in protocol["closed_loop_feedback"]
    assert "not a position-only controller" in protocol["closed_loop_feedback"]
    assert "no population or D1 claim" in summary["claims"]


def test_only_two_calibration_logs_cross_each_real_fitting_boundary(experiment) -> None:
    _, calls = experiment
    assert set(calls) == {"position", "position_velocity"}
    for name, call in calls.items():
        assert call["kwargs"] == {"max_delay_steps": 0, "max_nfev": 1}
        logs = call["logs"]
        assert [log.name for log in logs] == ["calibration_0", "calibration_1"]
        assert [log.split for log in logs] == ["calibration", "calibration"]
        if name == "position":
            assert all(log.known_initial_rest is True for log in logs)
            assert all(not hasattr(log, "velocities") for log in logs)
        else:
            assert all(log.velocities[0] == 0.0 for log in logs)


def test_saved_logs_share_commands_positions_and_initial_rest(
    experiment_output: Path, cli_module: ModuleType,
) -> None:
    destination = experiment_output / "matched_seed17"
    calibration_commands = []
    for index, split in enumerate(SPLITS):
        stem = f"{split}_{index}"
        position_path = destination / f"{stem}_position.csv"
        position = cli_module.load_encoder_log(position_path)
        reference = cli_module.load_actuator_log(destination / f"{stem}_position_velocity.csv")
        assert position.name == reference.name == stem
        assert position.split == reference.split == split
        assert position.config == reference.config
        assert position.commands.shape == reference.commands.shape == (250,)
        assert position.positions.shape == reference.positions.shape == (251,)
        np.testing.assert_array_equal(position.commands, reference.commands)
        np.testing.assert_array_equal(position.positions, reference.positions)
        assert position.known_initial_rest is True
        assert reference.velocities[0] == 0.0
        assert not hasattr(position, "velocities")
        with position_path.open(encoding="utf-8", newline="") as stream:
            metadata = json.loads(stream.readline()[2:])
            header = next(csv.reader(stream))
        assert metadata["known_initial_rest"] is True
        assert not any("velocity" in field or "applied_torque" in field for field in header)
        if split == "calibration":
            calibration_commands.append(position.commands)
        else:
            assert all(not np.array_equal(position.commands, c) for c in calibration_commands)


def test_prediction_metrics_recompute_from_complete_saved_position_rows(
    experiment_output: Path, cli_module: ModuleType,
) -> None:
    destination = experiment_output / "matched_seed17"
    summary = read_json(experiment_output / "summary.json")
    csv_metrics = read_csv_gz(experiment_output / "prediction_metrics.csv.gz")
    assert len(csv_metrics) == 12
    metric_index = {(row["trajectory"], row["model"]): row for row in csv_metrics}
    assert len(metric_index) == 12
    for metric in summary["prediction_metrics"]:
        stem = metric["trajectory"]
        log = cli_module.load_encoder_log(destination / f"{stem}_position.csv")
        rows = read_csv_gz(destination / f"{stem}_predictions.csv.gz")
        assert len(rows) == 251
        np.testing.assert_allclose([float(r["time_s"]) for r in rows], np.arange(251) * 0.002)
        np.testing.assert_array_equal(
            [float(r["measured_position_rad"]) for r in rows], log.positions,
        )
        field = f"{metric['model']}_position_rad"
        assert float(rows[0][field]) == log.positions[0]
        # Initial position is supplied, not predicted: score only samples 1..T.
        errors = [float(row[field]) - float(row["measured_position_rad"]) for row in rows[1:]]
        expected_rmse = math.sqrt(math.fsum(error * error for error in errors) / 250)
        assert metric["position_rmse_rad"] == pytest.approx(expected_rmse, rel=1e-12)
        csv_metric = metric_index[(stem, metric["model"])]
        assert float(csv_metric["position_rmse_rad"]) == metric["position_rmse_rad"]
        assert csv_metric["split"] == log.split


def test_control_metrics_use_current_state_and_paired_reference_and_sensor_noise(
    experiment_output: Path,
) -> None:
    destination = experiment_output / "matched_seed17"
    summary = read_json(experiment_output / "summary.json")
    metrics = {(r["profile"], r["controller"]): r for r in summary["control_metrics"]}
    csv_metrics = read_csv_gz(experiment_output / "control_metrics.csv.gz")
    assert len(csv_metrics) == 12
    csv_index = {(r["profile"], r["controller"]): r for r in csv_metrics}
    assert len(csv_index) == 12
    for profile in PROFILES:
        paired_reference = None
        paired_noise = None
        for arm in ARMS:
            rows = read_csv_gz(destination / f"{profile}_{arm}.csv.gz")
            assert len(rows) == 500
            time = np.array([float(row["time_s"]) for row in rows])
            actual = np.array([float(row["actual_velocity_rad_s"]) for row in rows])
            measured = np.array([float(row["measured_velocity_rad_s"]) for row in rows])
            reference = np.array([float(row["reference_velocity_rad_s"]) for row in rows])
            acceleration = np.array([float(row["reference_acceleration_rad_s2"]) for row in rows])
            next_velocity = np.array([float(row["next_velocity_rad_s"]) for row in rows])
            np.testing.assert_allclose(time, np.arange(500) * 0.002, atol=1e-15, rtol=0)
            np.testing.assert_array_equal(actual[1:], next_velocity[:-1])
            assert actual[0] == float(rows[0]["position_rad"]) == 0.0
            noise = measured - actual
            assert np.max(np.abs(noise)) > 0.0  # Exercise pairing with actual nonzero noise.
            commands = np.column_stack((reference, acceleration))
            if paired_reference is None:
                paired_reference, paired_noise = commands, noise
            else:
                np.testing.assert_array_equal(commands, paired_reference)
                np.testing.assert_allclose(noise, paired_noise, rtol=0, atol=4e-15)
            errors = reference - actual
            np.testing.assert_array_equal(
                [float(row["tracking_error_rad_s"]) for row in rows], errors,
            )
            # The existing controller still consumes the independent velocity sensor.
            np.testing.assert_array_equal(
                [float(row["error_rad_s"]) for row in rows], reference - measured,
            )
            np.testing.assert_allclose(
                [float(row["proportional_torque_nm"]) for row in rows],
                0.12 * (reference - measured), rtol=1e-14, atol=1e-15,
            )
            expected_rmse = math.sqrt(math.fsum(float(e) ** 2 for e in errors) / len(rows))
            metric = metrics[(profile, arm)]
            assert metric["velocity_rmse_rad_s"] == pytest.approx(expected_rmse, rel=1e-12)
            assert metric["duration_s"] == pytest.approx(time[-1] + 0.002)
            assert float(csv_index[(profile, arm)]["velocity_rmse_rad_s"]) == metric["velocity_rmse_rad_s"]
            if profile == "tracking":
                shifted_rmse = float(np.sqrt(np.mean((reference - next_velocity) ** 2)))
                assert abs(shifted_rmse - expected_rmse) > 1e-6


def test_unconverged_fits_are_retained_without_fabricating_success(experiment_output: Path) -> None:
    summary = read_json(experiment_output / "summary.json")
    assert len(summary["fits"]) == 2
    for fit in summary["fits"]:
        saved = read_json(experiment_output / "matched_seed17" / f"fit_{fit['observation']}.json")
        assert saved == fit
        assert len(fit["candidates"]) == 1
        candidate = fit["candidates"][0]
        assert candidate["parameters"]["delay_steps"] == 0
        assert candidate["nfev"] == 1
        assert candidate["optimizer_success"] is False
        assert candidate["message"]
        assert math.isfinite(candidate["calibration_loss"])
        assert fit["parameters"] == candidate["parameters"]
        assert fit["fit_status"]["optimizer_success"] is False
        assert fit["fit_status"]["delay_at_search_boundary"] is True
        assert "not global identifiability" in fit["fit_status"]["diagnostic_scope"]
        assert fit["fit_seconds"] > 0.0


def test_source_archive_and_manifest_match_every_recorded_byte(experiment_output: Path) -> None:
    source = read_json(experiment_output / "source.json")
    assert set(source["sha256"]) == {
        "scripts/evaluate_encoder_identification.py",
        "scripts/evaluate_wheel_control.py",
        "scripts/run_actuator_identification.py",
        "src/wheel_legged_control/encoder_identification.py",
        "src/wheel_legged_control/actuator_identification.py",
        "src/wheel_legged_control/actuator_bench.py",
        "src/wheel_legged_control/wheel_control.py",
        "src/wheel_legged_control/provenance.py",
        "src/wheel_legged_control/__init__.py",
        "src/wheel_legged_control/controllers.py",
        "src/wheel_legged_control/model.py",
        "src/wheel_legged_control/rewards.py",
        "pyproject.toml",
    }
    with tarfile.open(experiment_output / "source.tar.gz", "r:gz") as archive:
        members = archive.getmembers()
        assert len(members) == len(source["sha256"])
        assert {member.name for member in members} == set(source["sha256"])
        for member in members:
            assert member.isfile()
            stream = archive.extractfile(member)
            assert stream is not None
            payload = stream.read()
            assert hashlib.sha256(payload).hexdigest() == source["sha256"][member.name]
            assert payload == (ROOT / member.name).read_bytes()
    assert read_json(experiment_output / "source_consistency.json") == {"unchanged": True, "changed": []}
    manifest = read_json(experiment_output / "manifest.json")
    files = {
        str(path.relative_to(experiment_output))
        for path in experiment_output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert set(manifest["sha256"]) == files
    for relative, digest in manifest["sha256"].items():
        assert hashlib.sha256((experiment_output / relative).read_bytes()).hexdigest() == digest
    for name in ("holdout_prediction.png", "closed_loop.png"):
        payload = (experiment_output / "matched_seed17" / name).read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(payload) > 1000


@pytest.mark.parametrize(
    "options",
    [
        ["--seeds", "17", "17"],
        ["--seeds", "-1"],
        ["--seeds", "0.5"],
        ["--seeds"],
        ["--duration", "0.99"],
        ["--duration", "nan"],
        ["--duration", "inf"],
        ["--calibration-duration", "0.49"],
        ["--calibration-duration", "nan"],
        ["--calibration-duration", "inf"],
        ["--noise-scale", "-0.1"],
        ["--noise-scale", "nan"],
        ["--noise-scale", "inf"],
        ["--max-delay-steps", "-1"],
        ["--max-delay-steps", "0.5"],
        ["--max-nfev", "0"],
        ["--max-nfev", "-1"],
        ["--max-nfev", "nan"],
        ["--scenario", "unknown"],
    ],
)
def test_invalid_cli_arguments_create_no_result_directory(
    cli_module: ModuleType, tmp_path: Path, options: list[str],
) -> None:
    output = tmp_path / "must_not_exist"
    with pytest.raises(SystemExit) as stopped:
        cli_module.main(["--output", str(output), *options])
    assert stopped.value.code == 2
    assert not output.exists()


def test_existing_experiment_cannot_be_overwritten(
    cli_module: ModuleType, experiment_output: Path,
) -> None:
    before = {
        path.relative_to(experiment_output): path.read_bytes()
        for path in experiment_output.rglob("*") if path.is_file()
    }
    with pytest.raises(SystemExit) as stopped:
        cli_module.main(["--output", str(experiment_output)])
    assert stopped.value.code == 2
    after = {
        path.relative_to(experiment_output): path.read_bytes()
        for path in experiment_output.rglob("*") if path.is_file()
    }
    assert after == before
