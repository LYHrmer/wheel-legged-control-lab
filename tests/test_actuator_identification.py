from __future__ import annotations

import csv
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from wheel_legged_control.actuator_bench import ActuatorBench, ActuatorParameters, BenchConfig
from wheel_legged_control.actuator_identification import (
    ActuatorLog,
    fit_actuator_parameters,
    load_actuator_log,
    predict_actuator_log,
    prediction_metrics,
    save_actuator_log,
)


def _record(
    commands: np.ndarray,
    *,
    parameters: ActuatorParameters | None = None,
    config: BenchConfig | None = None,
    split: str = "calibration",
    initial_position: float = 0.0,
    initial_velocity: float = 0.0,
) -> ActuatorLog:
    config = config if config is not None else BenchConfig()
    parameters = parameters if parameters is not None else ActuatorParameters()
    bench = ActuatorBench(config, parameters)
    bench.reset(position=initial_position, velocity=initial_velocity)
    states = np.asarray([bench.state, *(bench.step(command) for command in commands)])
    return ActuatorLog(
        config=config,
        commands=commands,
        positions=states[:, 0],
        velocities=states[:, 1],
        name="synthetic_trajectory",
        split=split,
    )


def _short_log() -> ActuatorLog:
    return _record(np.asarray([0.25, -0.125, 0.375, -0.5]))


def test_log_copies_input_arrays_and_freezes_fields() -> None:
    commands = np.asarray([0.1, -0.2, 0.3])
    positions = np.asarray([0.0, 0.01, 0.02, 0.03])
    velocities = np.asarray([0.0, 0.1, 0.2, 0.3])
    log = ActuatorLog(BenchConfig(), commands, positions, velocities)
    commands[:] = 99.0
    positions[:] = 99.0
    velocities[:] = 99.0
    np.testing.assert_array_equal(log.commands, [0.1, -0.2, 0.3])
    np.testing.assert_array_equal(log.positions, [0.0, 0.01, 0.02, 0.03])
    np.testing.assert_array_equal(log.velocities, [0.0, 0.1, 0.2, 0.3])
    for array in (log.commands, log.positions, log.velocities):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 1.0
    with pytest.raises(FrozenInstanceError):
        log.split = "holdout"


@pytest.mark.parametrize("split", ["calibration", "validation", "holdout"])
def test_supported_splits_are_preserved(split: str) -> None:
    assert replace(_short_log(), split=split).split == split


@pytest.mark.parametrize("split", ["train", "", "test", None])
def test_unknown_split_is_rejected(split: object) -> None:
    with pytest.raises(ValueError):
        replace(_short_log(), split=split)


@pytest.mark.parametrize("field", ["commands", "positions", "velocities"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_log_values_are_rejected(field: str, bad_value: float) -> None:
    log = _short_log()
    values = getattr(log, field).copy()
    values[1] = bad_value
    with pytest.raises(ValueError):
        replace(log, **{field: values})


@pytest.mark.parametrize("field", ["commands", "positions", "velocities"])
def test_wrong_array_dimensions_are_rejected(field: str) -> None:
    log = _short_log()
    with pytest.raises(ValueError):
        replace(log, **{field: getattr(log, field)[:, None]})


@pytest.mark.parametrize("field", ["positions", "velocities"])
def test_state_sequence_requires_one_more_sample_than_commands(field: str) -> None:
    log = _short_log()
    with pytest.raises(ValueError):
        replace(log, **{field: getattr(log, field)[:-1]})


@pytest.mark.parametrize("steps", [0, 1])
def test_log_requires_at_least_two_commands(steps: int) -> None:
    with pytest.raises(ValueError):
        ActuatorLog(BenchConfig(), np.zeros(steps), np.zeros(steps + 1), np.zeros(steps + 1))


def test_csv_roundtrip_is_exact_and_excludes_hidden_parameters(tmp_path: Path) -> None:
    log = replace(_short_log(), name="轮轴记录", split="holdout")
    path = tmp_path / "measurements.csv"
    save_actuator_log(path, log)
    restored = load_actuator_log(path)
    assert restored.config == log.config
    assert restored.name == log.name
    assert restored.split == log.split
    for field in ("commands", "positions", "velocities"):
        np.testing.assert_array_equal(getattr(restored, field), getattr(log, field))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# ")
    metadata = json.loads(lines[0][2:])
    metadata_text = json.dumps(metadata).lower()
    for forbidden in ("armature", "damping", "coulomb", "stribeck", "truth"):
        assert forbidden not in metadata_text
    assert next(csv.reader([lines[1]])) == [
        "time_s",
        "command_torque_nm",
        "position_rad",
        "velocity_rad_s",
        "next_position_rad",
        "next_velocity_rad_s",
    ]
    assert len(lines) == len(log.commands) + 2


def test_csv_save_refuses_to_overwrite_existing_record(tmp_path: Path) -> None:
    path = tmp_path / "measurements.csv"
    log = _short_log()
    save_actuator_log(path, log)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_actuator_log(path, replace(log, name="replacement"))
    assert path.read_bytes() == before


@pytest.mark.parametrize("header", ["not a header", "# {broken json", "# {}"])
def test_loader_rejects_bad_metadata_header(tmp_path: Path, header: str) -> None:
    path = tmp_path / "bad_header.csv"
    save_actuator_log(path, _short_log())
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = header
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_actuator_log(path)


@pytest.mark.parametrize(
    ("row", "column", "replacement"),
    [
        (0, 0, "0.001"),  # Time must start at zero.
        (1, 0, "0.0021"),  # Samples must follow the configured physics timestep.
        (1, 1, "nan"),
        (1, 3, "inf"),
        (0, 4, "123.0"),  # A row's next state must equal the next row's state.
        (0, 5, "-123.0"),
    ],
)
def test_loader_rejects_nonfinite_or_discontinuous_samples(
    tmp_path: Path, row: int, column: int, replacement: str
) -> None:
    path = tmp_path / "bad_samples.csv"
    save_actuator_log(path, _short_log())
    lines = path.read_text(encoding="utf-8").splitlines()
    record = next(csv.reader([lines[row + 2]]))
    record[column] = replacement
    lines[row + 2] = ",".join(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_actuator_log(path)


def test_replay_starts_from_logged_state_and_is_repeatable() -> None:
    parameters = ActuatorParameters(armature=0.02, damping=0.1, delay_steps=2)
    log = _record(
        np.asarray([0.3, -0.2, 0.4, -0.1]),
        parameters=parameters,
        initial_position=0.4,
        initial_velocity=-0.3,
    )
    first = predict_actuator_log(log, parameters)
    second = predict_actuator_log(log, parameters)
    assert first.shape == (len(log.commands) + 1, 2)
    np.testing.assert_array_equal(first, np.column_stack((log.positions, log.velocities)))
    np.testing.assert_array_equal(second, first)
    first[:] = 99.0
    np.testing.assert_array_equal(predict_actuator_log(log, parameters), second)


def test_prediction_metrics_exclude_the_known_initial_state() -> None:
    log = _short_log()
    predicted = np.column_stack((log.positions, log.velocities))
    predicted[0] += 1000.0
    predicted[1:, 0] += 0.1
    predicted[1:, 1] -= 0.2
    metrics = prediction_metrics(log, predicted)
    assert metrics["position_rmse_rad"] == pytest.approx(0.1)
    assert metrics["velocity_rmse_rad_s"] == pytest.approx(0.2)


def test_prediction_does_not_teacher_force_from_future_measured_states() -> None:
    log = _short_log()
    altered_positions = log.positions.copy()
    altered_velocities = log.velocities.copy()
    altered_positions[1:] += 10.0
    altered_velocities[1:] -= 20.0
    altered = replace(log, positions=altered_positions, velocities=altered_velocities)
    parameters = ActuatorParameters()
    np.testing.assert_array_equal(
        predict_actuator_log(log, parameters), predict_actuator_log(altered, parameters)
    )


@pytest.mark.parametrize("invalid", [np.zeros((4, 2)), np.full((5, 2), np.nan)])
def test_prediction_metrics_reject_invalid_prediction(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError):
        prediction_metrics(_short_log(), invalid)


@pytest.mark.parametrize("split", ["validation", "holdout"])
def test_fitting_refuses_noncalibration_logs(split: str) -> None:
    with pytest.raises(ValueError):
        fit_actuator_parameters([replace(_short_log(), split=split)])


def test_fitting_requires_logs_with_the_same_known_bench_configuration() -> None:
    log = _short_log()
    other = _record(log.commands, config=BenchConfig(kind="pendulum"))
    with pytest.raises(ValueError):
        fit_actuator_parameters([log, other])


def test_fitting_requires_nonempty_data() -> None:
    with pytest.raises(ValueError):
        fit_actuator_parameters([])


def test_noise_free_identification_recovers_parameters_and_predicts_unseen_motion() -> None:
    truth = ActuatorParameters(armature=0.017, damping=0.105, coulomb_friction=0.043, delay_steps=2)
    config = BenchConfig()
    time = np.arange(1000) * config.dt
    commands = (
        0.55 * np.sin(2.0 * np.pi * 0.9 * time)
        + 0.35 * np.sin(2.0 * np.pi * 2.7 * time + 0.3)
        + 0.15 * np.sin(2.0 * np.pi * 6.3 * time - 1.0)
    )
    calibration = _record(commands, parameters=truth, config=config)
    result = fit_actuator_parameters([calibration], max_delay_steps=2, max_nfev=25)
    assert result.parameters.delay_steps == truth.delay_steps
    for name in ("armature", "damping", "coulomb_friction"):
        assert getattr(result.parameters, name) == pytest.approx(getattr(truth, name), rel=0.01)
    assert len(result.candidates) == 3
    assert {candidate["parameters"]["delay_steps"] for candidate in result.candidates} == {0, 1, 2}
    assert all(np.isfinite(candidate["calibration_loss"]) for candidate in result.candidates)
    assert len(result.jacobian_singular_values) == 3
    assert result.jacobian_rank == 3
    assert all(np.isfinite(result.jacobian_singular_values))

    # Different frequencies, phases, and initial motion; never supplied to fit.
    holdout_time = np.arange(750) * config.dt
    holdout_commands = (
        0.45 * np.sin(2.0 * np.pi * 1.4 * holdout_time + 0.7)
        + 0.3 * np.sin(2.0 * np.pi * 3.8 * holdout_time - 0.5)
        + 0.12 * np.sin(2.0 * np.pi * 7.1 * holdout_time)
    )
    holdout = _record(
        holdout_commands,
        parameters=truth,
        config=config,
        split="holdout",
        initial_position=0.2,
        initial_velocity=-0.4,
    )
    fitted = prediction_metrics(holdout, predict_actuator_log(holdout, result.parameters))
    nominal = prediction_metrics(holdout, predict_actuator_log(holdout, ActuatorParameters()))
    assert fitted["position_rmse_rad"] < 1e-3
    assert fitted["velocity_rmse_rad_s"] < 1e-3
    assert fitted["velocity_rmse_rad_s"] < 0.01 * nominal["velocity_rmse_rad_s"]
