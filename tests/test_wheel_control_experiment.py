"""Closed-loop protocol checks, not assertions that identification must win."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from wheel_legged_control.actuator_bench import (
    ActuatorBench,
    ActuatorParameters,
    BenchConfig,
)
from wheel_legged_control.actuator_identification import load_actuator_log
from wheel_legged_control.wheel_control import WheelControlConfig, WheelVelocityController

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_wheel_control.py"
PROFILES = ("tracking", "reversal", "stress")
CONTROLLERS = ("pi", "nominal_ff", "identified_ff")


@pytest.fixture(scope="module")
def cli_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location("wheel_control_experiment_cli", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def experiment_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("wheel_control_cli") / "experiment"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / ".local-deps"), environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output),
            "--scenario",
            "matched",
            "--duration",
            "1",
            "--calibration-duration",
            "0.5",
            "--max-delay-steps",
            "0",
            "--max-nfev",
            "1",
            "--noise-scale",
            "1",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Results:" in completed.stdout
    return output


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("profile", PROFILES)
def test_reference_acceleration_matches_analytic_velocity_derivative(
    cli_module: ModuleType, profile: str
) -> None:
    # Off-grid times avoid any intended piecewise-profile join points.
    time = np.asarray((0.0713, 0.3167, 0.7411, 1.1963, 2.7311, 4.7313, 5.3127))
    epsilon = 1e-6
    velocity, acceleration = cli_module.reference(profile, time)
    before, _ = cli_module.reference(profile, time - epsilon)
    after, _ = cli_module.reference(profile, time + epsilon)
    assert velocity.shape == acceleration.shape == time.shape
    assert np.isfinite(velocity).all() and np.isfinite(acceleration).all()
    np.testing.assert_allclose(acceleration, (after - before) / (2 * epsilon), rtol=1e-5, atol=1e-5)


def test_closed_loop_uses_current_state_and_replays_the_same_hidden_plant(
    cli_module: ModuleType,
) -> None:
    config = BenchConfig(dt=0.005)
    truth = ActuatorParameters(0.031, 0.12, 0.06, 3)
    control_config = WheelControlConfig(dt=config.dt, torque_limit_nm=config.torque_limit_nm)
    noise_residuals = []
    for parameters in (None, ActuatorParameters(), ActuatorParameters(0.02, 0.15, 0.04, 1)):
        rows = cli_module.run_closed_loop(
            config,
            truth,
            WheelVelocityController(control_config, parameters),
            profile="tracking",
            duration=1.0,
            seed=91,
            noise_scale=1.0,
            stribeck_friction_nm=0.07,
        )
        assert len(rows) == 200
        np.testing.assert_allclose([row["time_s"] for row in rows], np.arange(200) * config.dt)
        bench = ActuatorBench(config, truth, stribeck_friction_nm=0.07)
        for row in rows:
            assert row["actual_velocity_rad_s"] == pytest.approx(bench.state[1], abs=1e-13)
            assert row["tracking_error_rad_s"] == pytest.approx(
                row["reference_velocity_rad_s"] - row["actual_velocity_rad_s"], abs=1e-13
            )
            assert row["error_rad_s"] == pytest.approx(
                row["reference_velocity_rad_s"] - row["measured_velocity_rad_s"], abs=1e-13
            )
            next_state = bench.step(row["command_torque_nm"])
            assert row["next_velocity_rad_s"] == pytest.approx(next_state[1], abs=1e-13)
            assert row["applied_torque_nm"] == pytest.approx(bench.applied_torque_nm, abs=1e-13)
        noise_residuals.append(
            np.asarray(
                [row["measured_velocity_rad_s"] - row["actual_velocity_rad_s"] for row in rows]
            )
        )
    assert np.linalg.norm(noise_residuals[0]) > 0
    for residual in noise_residuals[1:]:
        np.testing.assert_allclose(residual, noise_residuals[0], rtol=0.0, atol=1e-14)


def test_closed_loop_resets_a_reused_controller_and_is_seed_reproducible(
    cli_module: ModuleType,
) -> None:
    config = BenchConfig()
    controller = WheelVelocityController(WheelControlConfig(), ActuatorParameters())

    def rollout(seed: int) -> list[dict]:
        return cli_module.run_closed_loop(
            config,
            ActuatorParameters(0.017, 0.105, 0.043, 2),
            controller,
            profile="reversal",
            duration=1.0,
            seed=seed,
            noise_scale=1.0,
        )

    first = rollout(17)
    assert first == rollout(17)
    other = rollout(18)
    assert not np.array_equal(
        [row["measured_velocity_rad_s"] for row in first],
        [row["measured_velocity_rad_s"] for row in other],
    )


def test_stress_trace_preserves_limit_and_conditional_integration_evidence(
    cli_module: ModuleType,
) -> None:
    config = BenchConfig()
    control_config = WheelControlConfig()
    rows = cli_module.run_closed_loop(
        config,
        ActuatorParameters(0.017, 0.105, 0.043, 2),
        WheelVelocityController(control_config, ActuatorParameters()),
        profile="stress",
        duration=1.0,
        seed=17,
        noise_scale=0.0,
    )
    assert any(row["saturated"] for row in rows)
    assert any(row["integration_frozen"] for row in rows)
    previous_integral = 0.0
    for row in rows:
        expected_integral = previous_integral
        if not row["integration_frozen"]:
            expected_integral += control_config.ki * row["error_rad_s"] * config.dt
        assert row["integral_torque_nm"] == pytest.approx(expected_integral, abs=1e-13)
        assert row["requested_torque_nm"] == pytest.approx(
            row["proportional_torque_nm"]
            + row["integral_torque_nm"]
            + row["feedforward_torque_nm"],
            abs=1e-13,
        )
        assert row["command_torque_nm"] == pytest.approx(
            np.clip(row["requested_torque_nm"], -config.torque_limit_nm, config.torque_limit_nm),
            abs=1e-13,
        )
        assert abs(row["applied_torque_nm"]) <= config.torque_limit_nm
        previous_integral = row["integral_torque_nm"]


def test_speed_guard_keeps_the_triggering_step_and_reports_partial_metrics(
    cli_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_module, "SPEED_GUARD_RAD_S", 0.0001)
    rows = cli_module.run_closed_loop(
        BenchConfig(),
        ActuatorParameters(),
        WheelVelocityController(WheelControlConfig(), ActuatorParameters()),
        profile="tracking",
        duration=1.0,
        seed=17,
        noise_scale=0.0,
    )
    assert len(rows) == 1
    assert rows[-1]["speed_guard_triggered"] is True
    assert abs(rows[-1]["next_velocity_rad_s"]) > cli_module.SPEED_GUARD_RAD_S
    metrics = cli_module.compute_metrics(rows, 0.002)
    assert metrics["steps"] == 1
    assert metrics["duration_s"] == 0.002
    assert metrics["speed_guard_triggered"] is True


def test_metrics_are_recomputed_from_every_saved_physical_step(
    cli_module: ModuleType, experiment_output: Path
) -> None:
    metric_rows = read_rows(experiment_output / "metrics.csv")
    assert len(metric_rows) == 9
    assert {(row["profile"], row["controller"]) for row in metric_rows} == {
        (profile, controller) for profile in PROFILES for controller in CONTROLLERS
    }
    boolean_fields = {"saturated", "integration_frozen", "speed_guard_triggered"}
    for metric in metric_rows:
        assert metric["scenario"] == "matched"
        rows = read_rows(
            experiment_output / "matched" / f"{metric['profile']}_{metric['controller']}.csv"
        )
        records = [
            {
                key: value == "True" if key in boolean_fields else float(value)
                for key, value in row.items()
            }
            for row in rows
        ]
        assert len(records) == 500
        for row in rows:
            for field in boolean_fields:
                assert row[field] in {"True", "False"}
        for row in records:
            assert np.isfinite(list(row.values())).all()
        reference_velocity = np.asarray([row["reference_velocity_rad_s"] for row in records])
        actual_velocity = np.asarray([row["actual_velocity_rad_s"] for row in records])
        # Independent recomputation does not trust the stored error column.
        error = reference_velocity - actual_velocity
        np.testing.assert_array_equal(error, [row["tracking_error_rad_s"] for row in records])
        command = np.asarray([row["command_torque_nm"] for row in records])
        applied = np.asarray([row["applied_torque_nm"] for row in records])
        expected = {
            "steps": len(records),
            "duration_s": len(records) * 0.002,
            "velocity_rmse_rad_s": np.sqrt(np.mean(error**2)),
            "velocity_mae_rad_s": np.mean(np.abs(error)),
            "peak_velocity_error_rad_s": np.max(np.abs(error)),
            "integrated_absolute_error_rad": np.sum(np.abs(error)) * 0.002,
            "command_torque_rms_nm": np.sqrt(np.mean(command**2)),
            "applied_torque_rms_nm": np.sqrt(np.mean(applied**2)),
            "peak_requested_torque_nm": max(abs(row["requested_torque_nm"]) for row in records),
            "saturation_fraction": np.mean([row["saturated"] for row in records]),
            "integration_frozen_fraction": np.mean([row["integration_frozen"] for row in records]),
        }
        recomputed = cli_module.compute_metrics(records, 0.002)
        for name, value in expected.items():
            assert float(metric[name]) == pytest.approx(value, rel=1e-12, abs=1e-12)
            assert recomputed[name] == pytest.approx(value, rel=1e-12, abs=1e-12)
        assert metric["completed"] == "True"
        assert metric["speed_guard_triggered"] == "False"
        assert recomputed["speed_guard_triggered"] is False
        np.testing.assert_array_equal(
            actual_velocity[1:], [row["next_velocity_rad_s"] for row in records[:-1]]
        )


def test_low_budget_fit_keeps_nonconvergence_and_scope_explicit(experiment_output: Path) -> None:
    experiment = json.loads((experiment_output / "experiment.json").read_text(encoding="utf-8"))
    summary = json.loads((experiment_output / "summary.json").read_text(encoding="utf-8"))
    fit = json.loads((experiment_output / "matched" / "fit.json").read_text(encoding="utf-8"))
    assert experiment["schema_version"] == 1
    assert experiment["data_origin"] == "synthetic_only"
    assert experiment["arguments"]["max_nfev"] == 1
    assert experiment["controller_config_all_conditions"] == {
        "kp": 0.12,
        "ki": 0.6,
        "dt": 0.002,
        "torque_limit_nm": 2.0,
    }
    assert (
        experiment["nominal_parameters"]
        != experiment["generator_truth_not_available_to_fitter_or_controller"]
    )
    assert "no closed-loop metrics" in experiment["protocol"]["fit"]
    assert "not next_velocity" in experiment["protocol"]["metrics"]
    assert "none" in experiment["protocol"]["delay_compensation"]
    assert summary["status"] == "completed"
    assert "no whole-robot or hardware validation" in summary["claims"]
    assert len(fit["candidates"]) == 1
    assert fit["candidates"][0]["nfev"] == 1
    assert fit["candidates"][0]["optimizer_success"] is False
    assert fit["fit_status"]["optimizer_success"] is False
    assert fit["fit_status"]["delay_compensated"] is False
    assert fit["fit_status"]["delay_at_search_boundary"] is True
    assert isinstance(fit["fit_status"]["continuous_parameters_at_bounds"], list)
    assert summary["cases"][0]["fit_status"] == fit["fit_status"]
    assert len(summary["metrics"]) == 9
    # A completed artifact pipeline is not a converged fit or an accuracy claim.


def test_manifest_covers_each_artifact_and_source_fingerprints_are_current(
    experiment_output: Path,
) -> None:
    manifest = json.loads((experiment_output / "manifest.json").read_text(encoding="utf-8"))
    files = {
        str(path.relative_to(experiment_output))
        for path in experiment_output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert set(manifest["sha256"]) == files
    assert {"experiment.json", "metrics.csv", "summary.json"}.issubset(files)
    for relative, expected in manifest["sha256"].items():
        assert hashlib.sha256((experiment_output / relative).read_bytes()).hexdigest() == expected
    experiment = json.loads((experiment_output / "experiment.json").read_text(encoding="utf-8"))
    assert "scripts/evaluate_wheel_control.py" in experiment["source_sha256"]
    assert "src/wheel_legged_control/wheel_control.py" in experiment["source_sha256"]
    for relative, expected in experiment["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_two_calibration_logs_are_distinct_from_closed_loop_profiles(
    experiment_output: Path,
) -> None:
    destination = experiment_output / "matched"
    logs = [load_actuator_log(destination / f"calibration_{index}.csv") for index in range(2)]
    for log in logs:
        assert log.split == "calibration"
        assert len(log.commands) == 250
        assert len(log.positions) == len(log.velocities) == 251
        assert log.config == BenchConfig()
        assert log.name not in PROFILES
    assert not np.array_equal(logs[0].commands, logs[1].commands)
    for profile in PROFILES:
        for controller in CONTROLLERS:
            assert (destination / f"{profile}_{controller}.csv").is_file()
        image = (destination / f"{profile}.png").read_bytes()
        assert image.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(image) > 1000


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--duration", "0.99"),
        ("--duration", "nan"),
        ("--duration", "inf"),
        ("--calibration-duration", "0.49"),
        ("--calibration-duration", "nan"),
        ("--calibration-duration", "inf"),
        ("--noise-scale", "-0.1"),
        ("--noise-scale", "nan"),
        ("--noise-scale", "inf"),
        ("--seed", "-1"),
        ("--seed", "1.5"),
        ("--max-delay-steps", "-1"),
        ("--max-delay-steps", "0.5"),
        ("--max-nfev", "0"),
        ("--max-nfev", "-1"),
        ("--max-nfev", "nan"),
    ],
)
def test_invalid_arguments_do_not_create_an_output_directory(
    cli_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    value: str,
) -> None:
    output = tmp_path / "must_not_exist"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(output), option, value])
    with pytest.raises(SystemExit) as stopped:
        cli_module.main()
    assert stopped.value.code == 2
    assert not output.exists()


def test_cli_refuses_to_overwrite_existing_experiment(
    cli_module: ModuleType, experiment_output: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = {
        path.relative_to(experiment_output): path.read_bytes()
        for path in experiment_output.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(experiment_output)])
    with pytest.raises(SystemExit) as stopped:
        cli_module.main()
    assert stopped.value.code == 2
    after = {
        path.relative_to(experiment_output): path.read_bytes()
        for path in experiment_output.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_help_succeeds_without_output(
    cli_module: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--help"])
    with pytest.raises(SystemExit) as stopped:
        cli_module.main()
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "--output" in help_text
    assert "--calibration-duration" in help_text
    assert "--scenario" in help_text
