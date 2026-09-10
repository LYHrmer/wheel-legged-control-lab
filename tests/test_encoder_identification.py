from __future__ import annotations

import csv
import json
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import numpy as np
import pytest

from wheel_legged_control.actuator_bench import ActuatorBench, ActuatorParameters, BenchConfig
from wheel_legged_control.actuator_identification import (
    PARAMETER_LOWER,
    PARAMETER_NAMES,
    PARAMETER_UPPER,
    POSITION_SCALE_RAD,
    FitResult,
)
from wheel_legged_control.encoder_identification import (
    CSV_COLUMNS,
    EncoderLog,
    encoder_prediction_metrics,
    fit_encoder_parameters,
    load_encoder_log,
    predict_encoder_log,
    save_encoder_log,
)


def _record(
    commands: np.ndarray,
    *,
    parameters: ActuatorParameters | None = None,
    config: BenchConfig | None = None,
    split: str = "calibration",
    initial_position: float = 0.0,
    name: str = "synthetic_trajectory",
) -> EncoderLog:
    """Acquire a log on the real bench; the fixture knows parameters, the log never does."""
    config = config if config is not None else BenchConfig()
    parameters = parameters if parameters is not None else ActuatorParameters()
    bench = ActuatorBench(config, parameters)
    bench.reset(position=initial_position, velocity=0.0)
    positions = [bench.state[0], *(bench.step(command)[0] for command in commands)]
    return EncoderLog(
        config=config,
        commands=commands,
        positions=np.asarray(positions),
        known_initial_rest=True,
        name=name,
        split=split,
    )


def _short_log() -> EncoderLog:
    return _record(np.asarray([0.25, -0.125, 0.375, -0.5]))


def _chirp(steps: int, dt: float, scales: tuple, freqs: tuple, phases: tuple) -> np.ndarray:
    time = np.arange(steps) * dt
    return sum(
        scale * np.sin(2.0 * np.pi * freq * time + phase)
        for scale, freq, phase in zip(scales, freqs, phases, strict=True)
    )


def _rewrite(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_encoder_log_requires_explicit_known_initial_rest() -> None:
    with pytest.raises(TypeError):
        EncoderLog(BenchConfig(), np.zeros(3), np.zeros(4))
    log = EncoderLog(BenchConfig(), np.zeros(3), np.zeros(4), known_initial_rest=True)
    assert log.known_initial_rest is True


@pytest.mark.parametrize("rest", [False, None, 1, "true", np.True_])
def test_rest_declaration_must_be_exactly_true(rest: object) -> None:
    with pytest.raises(ValueError):
        EncoderLog(BenchConfig(), np.zeros(3), np.zeros(4), known_initial_rest=rest)


def test_log_exposes_only_position_fields_without_velocity_or_truth() -> None:
    log = _short_log()
    assert tuple(field.name for field in fields(log)) == (
        "config",
        "commands",
        "positions",
        "known_initial_rest",
        "name",
        "split",
    )
    for hidden in ("velocities", "velocity", "applied_torque_nm", "truth", "parameters"):
        assert not hasattr(log, hidden)


def test_log_copies_input_arrays_and_freezes_fields() -> None:
    commands = np.asarray([0.1, -0.2, 0.3])
    positions = np.asarray([0.0, 0.01, 0.02, 0.03])
    log = EncoderLog(BenchConfig(), commands, positions, known_initial_rest=True)
    commands[:] = 99.0
    positions[:] = 99.0
    np.testing.assert_array_equal(log.commands, [0.1, -0.2, 0.3])
    np.testing.assert_array_equal(log.positions, [0.0, 0.01, 0.02, 0.03])
    for array in (log.commands, log.positions):
        assert array.dtype == np.float64
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


@pytest.mark.parametrize("name", ["", "   ", None])
def test_empty_name_is_rejected(name: object) -> None:
    with pytest.raises(ValueError):
        replace(_short_log(), name=name)


def test_non_bench_config_is_rejected() -> None:
    with pytest.raises(TypeError):
        replace(_short_log(), config={"kind": "wheel", "dt": 0.002, "torque_limit_nm": 2.0})


@pytest.mark.parametrize("field", ["commands", "positions"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_log_values_are_rejected(field: str, bad_value: float) -> None:
    log = _short_log()
    values = getattr(log, field).copy()
    values[1] = bad_value
    with pytest.raises(ValueError):
        replace(log, **{field: values})


@pytest.mark.parametrize("field", ["commands", "positions"])
def test_wrong_array_dimensions_are_rejected(field: str) -> None:
    log = _short_log()
    with pytest.raises(ValueError):
        replace(log, **{field: getattr(log, field)[:, None]})


def test_positions_require_one_more_sample_than_commands() -> None:
    log = _short_log()
    with pytest.raises(ValueError):
        replace(log, positions=log.positions[:-1])
    with pytest.raises(ValueError):
        replace(log, positions=np.r_[log.positions, 0.0])


@pytest.mark.parametrize("steps", [0, 1])
def test_log_requires_at_least_two_commands(steps: int) -> None:
    with pytest.raises(ValueError):
        EncoderLog(BenchConfig(), np.zeros(steps), np.zeros(steps + 1), known_initial_rest=True)


def test_csv_roundtrip_is_exact_and_uses_the_position_only_schema(tmp_path: Path) -> None:
    log = replace(_short_log(), name="轮轴位置记录", split="holdout")
    path = tmp_path / "encoder.csv"
    save_encoder_log(path, log)
    restored = load_encoder_log(path)
    assert restored.config == log.config
    assert restored.name == log.name
    assert restored.split == log.split
    assert restored.known_initial_rest is True
    np.testing.assert_array_equal(restored.commands, log.commands)
    np.testing.assert_array_equal(restored.positions, log.positions)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert next(csv.reader([lines[1]])) == [
        "time_s",
        "command_torque_nm",
        "position_rad",
        "next_position_rad",
    ]
    assert list(CSV_COLUMNS) == next(csv.reader([lines[1]]))
    assert len(lines) == len(log.commands) + 2


def test_metadata_declares_rest_and_zero_history_without_hidden_parameters(
    tmp_path: Path,
) -> None:
    path = tmp_path / "encoder.csv"
    save_encoder_log(path, _short_log())
    metadata = json.loads(path.read_text(encoding="utf-8").splitlines()[0][2:])
    assert metadata["known_initial_rest"] is True
    assert metadata["initial_command_history"] == "zero"
    assert metadata["initial_state"] == "known_rest"
    assert metadata["schema"] == "encoder_position_only"
    text = json.dumps(metadata).lower()
    for forbidden in ("armature", "damping", "coulomb", "stribeck", "truth", "velocit", "delay"):
        assert forbidden not in text


def test_old_q_and_v_csv_is_not_accepted_by_the_position_only_reader(tmp_path: Path) -> None:
    from wheel_legged_control.actuator_identification import ActuatorLog, save_actuator_log

    log = _short_log()
    old = ActuatorLog(log.config, log.commands, log.positions, np.zeros(len(log.positions)))
    path = tmp_path / "old.csv"
    save_actuator_log(path, old)
    with pytest.raises(ValueError):
        load_encoder_log(path)


def test_csv_save_refuses_to_overwrite_existing_record(tmp_path: Path) -> None:
    path = tmp_path / "encoder.csv"
    log = _short_log()
    save_encoder_log(path, log)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_encoder_log(path, replace(log, name="replacement"))
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "header",
    [
        "not a header",
        "# {broken json",
        "# {}",
        "# []",
        '# {"schema": "encoder_position_only"}',
    ],
)
def test_loader_rejects_bad_metadata_header(tmp_path: Path, header: str) -> None:
    path = tmp_path / "bad_header.csv"
    save_encoder_log(path, _short_log())
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = header
    _rewrite(path, lines)
    with pytest.raises(ValueError):
        load_encoder_log(path)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param({"velocity_scale_rad_s": 0.5}, id="extra_velocity_key"),
        pytest.param({"truth_parameters": [0.017, 0.105]}, id="truth_key"),
        pytest.param({"stribeck_friction_nm": 0.08}, id="hidden_plant_key"),
    ],
)
def test_loader_rejects_unknown_metadata_keys(tmp_path: Path, mutate: dict) -> None:
    path = tmp_path / "extra_keys.csv"
    save_encoder_log(path, _short_log())
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(lines[0][2:]) | mutate
    lines[0] = "# " + json.dumps(metadata, sort_keys=True)
    _rewrite(path, lines)
    with pytest.raises(ValueError):
        load_encoder_log(path)


@pytest.mark.parametrize("key", ["known_initial_rest", "initial_command_history", "config", "name"])
def test_loader_rejects_missing_metadata_keys(tmp_path: Path, key: str) -> None:
    path = tmp_path / "missing_key.csv"
    save_encoder_log(path, _short_log())
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(lines[0][2:])
    del metadata[key]
    lines[0] = "# " + json.dumps(metadata, sort_keys=True)
    _rewrite(path, lines)
    with pytest.raises(ValueError):
        load_encoder_log(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("known_initial_rest", False),
        ("known_initial_rest", "true"),
        ("known_initial_rest", 1),
        ("initial_command_history", "unknown"),
        ("initial_state", "moving"),
        ("sampling", "state_then_command_then_next_state"),
        ("schema", "actuator_state"),
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", 1.0),
    ],
)
def test_loader_rejects_altered_acquisition_declarations(
    tmp_path: Path, key: str, value: object
) -> None:
    path = tmp_path / "altered.csv"
    save_encoder_log(path, _short_log())
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(lines[0][2:])
    metadata[key] = value
    lines[0] = "# " + json.dumps(metadata, sort_keys=True)
    _rewrite(path, lines)
    with pytest.raises(ValueError):
        load_encoder_log(path)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"kind": "wheel", "dt": 0.002}, id="missing_limit"),
        pytest.param(
            {"kind": "wheel", "dt": 0.002, "torque_limit_nm": 2.0, "gear": 1.0}, id="unknown_key"
        ),
        pytest.param({"kind": "leg", "dt": 0.002, "torque_limit_nm": 2.0}, id="unknown_kind"),
        pytest.param({"kind": "wheel", "dt": -0.002, "torque_limit_nm": 2.0}, id="bad_dt"),
    ],
)
def test_loader_rejects_bad_config_metadata(tmp_path: Path, config: dict) -> None:
    path = tmp_path / "bad_config.csv"
    save_encoder_log(path, _short_log())
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(lines[0][2:])
    metadata["config"] = config
    lines[0] = "# " + json.dumps(metadata, sort_keys=True)
    _rewrite(path, lines)
    with pytest.raises(ValueError):
        load_encoder_log(path)


def test_loader_rejects_duplicate_metadata_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.csv"
    save_encoder_log(path, _short_log())
    lines = path.read_text(encoding="utf-8").splitlines()
    body = lines[0][3:]
    lines[0] = '# {"known_initial_rest": false, ' + body
    _rewrite(path, lines)
    with pytest.raises(ValueError):
        load_encoder_log(path)


@pytest.mark.parametrize(("key", "value"), [("name", 5), ("split", "train"), ("name", "  ")])
def test_loader_rejects_bad_name_or_split_metadata(tmp_path: Path, key: str, value: object) -> None:
    path = tmp_path / "bad_identity.csv"
    save_encoder_log(path, _short_log())
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(lines[0][2:])
    metadata[key] = value
    lines[0] = "# " + json.dumps(metadata, sort_keys=True)
    _rewrite(path, lines)
    with pytest.raises(ValueError):
        load_encoder_log(path)


def test_loader_rejects_nonfinite_metadata(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite_metadata.csv"
    save_encoder_log(path, _short_log())
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(lines[0][2:])
    metadata["config"]["dt"] = float("nan")
    lines[0] = "# " + json.dumps(metadata, sort_keys=True)
    _rewrite(path, lines)
    with pytest.raises(ValueError):
        load_encoder_log(path)


@pytest.mark.parametrize(
    ("row", "column", "replacement"),
    [
        (0, 0, "0.001"),  # Time must start at zero.
        (1, 0, "0.0021"),  # Samples must follow the configured physics timestep.
        (1, 1, "nan"),
        (1, 2, "inf"),
        (0, 3, "123.0"),  # A row's next position must equal the next row's position.
    ],
)
def test_loader_rejects_broken_clock_or_discontinuous_positions(
    tmp_path: Path, row: int, column: int, replacement: str
) -> None:
    path = tmp_path / "bad_samples.csv"
    save_encoder_log(path, _short_log())
    lines = path.read_text(encoding="utf-8").splitlines()
    record = next(csv.reader([lines[row + 2]]))
    record[column] = replacement
    lines[row + 2] = ",".join(record)
    _rewrite(path, lines)
    with pytest.raises(ValueError):
        load_encoder_log(path)


@pytest.mark.parametrize(
    "columns",
    [
        "time_s,command_torque_nm,position_rad,velocity_rad_s,next_position_rad",
        "time_s,command_torque_nm,position_rad",
        "t,command_torque_nm,position_rad,next_position_rad",
    ],
)
def test_loader_rejects_wrong_columns(tmp_path: Path, columns: str) -> None:
    path = tmp_path / "bad_columns.csv"
    save_encoder_log(path, _short_log())
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = columns
    _rewrite(path, lines)
    with pytest.raises(ValueError):
        load_encoder_log(path)


def test_loader_rejects_short_or_malformed_rows(tmp_path: Path) -> None:
    path = tmp_path / "short_row.csv"
    save_encoder_log(path, _short_log())
    lines = path.read_text(encoding="utf-8").splitlines()
    _rewrite(path, [*lines[:-1], "0.006,0.375"])
    with pytest.raises(ValueError):
        load_encoder_log(path)
    other = tmp_path / "text_row.csv"
    save_encoder_log(other, _short_log())
    lines = other.read_text(encoding="utf-8").splitlines()
    _rewrite(other, [*lines[:-1], "0.006,0.375,abc,0.0"])
    with pytest.raises(ValueError):
        load_encoder_log(other)


def test_loader_requires_at_least_two_rows(tmp_path: Path) -> None:
    path = tmp_path / "one_row.csv"
    save_encoder_log(path, _short_log())
    lines = path.read_text(encoding="utf-8").splitlines()
    _rewrite(path, lines[:3])
    with pytest.raises(ValueError):
        load_encoder_log(path)


def test_replay_reproduces_the_recording_bench_and_is_repeatable() -> None:
    parameters = ActuatorParameters(armature=0.02, damping=0.1, delay_steps=2)
    log = _record(np.asarray([0.3, -0.2, 0.4, -0.1]), parameters=parameters, initial_position=0.4)
    first = predict_encoder_log(log, parameters)
    second = predict_encoder_log(log, parameters)
    assert first.shape == (len(log.commands) + 1,)
    np.testing.assert_array_equal(first, log.positions)
    np.testing.assert_array_equal(second, first)
    first[:] = 99.0
    np.testing.assert_array_equal(predict_encoder_log(log, parameters), second)


def test_replay_returns_positions_only_and_starts_at_the_first_sample() -> None:
    log = _record(np.asarray([0.2, -0.3, 0.1]), initial_position=-0.7)
    prediction = predict_encoder_log(log, ActuatorParameters())
    assert prediction.ndim == 1
    assert prediction[0] == pytest.approx(log.positions[0])


def test_replay_resets_the_command_queue_so_delay_shifts_the_response() -> None:
    commands = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0])
    log = _record(commands)
    undelayed = predict_encoder_log(log, ActuatorParameters(delay_steps=0))
    delayed = predict_encoder_log(log, ActuatorParameters(delay_steps=1))
    assert undelayed[1] != pytest.approx(delayed[1], abs=1e-12)
    assert delayed[1] == pytest.approx(log.positions[0], abs=1e-12)
    np.testing.assert_allclose(delayed[2:], undelayed[1:-1], rtol=0, atol=1e-12)


def test_prediction_does_not_teacher_force_from_later_measured_positions() -> None:
    log = _short_log()
    altered = log.positions.copy()
    altered[1:] += 10.0
    parameters = ActuatorParameters()
    np.testing.assert_array_equal(
        predict_encoder_log(log, parameters),
        predict_encoder_log(replace(log, positions=altered), parameters),
    )


def test_metrics_exclude_the_known_initial_sample() -> None:
    log = _short_log()
    predicted = log.positions.copy()
    predicted[0] += 1000.0
    predicted[1:] += 0.1
    metrics = encoder_prediction_metrics(log, predicted)
    assert set(metrics) == {"position_rmse_rad"}
    assert metrics["position_rmse_rad"] == pytest.approx(0.1)


def test_metrics_are_zero_for_an_exact_replay() -> None:
    log = _short_log()
    metrics = encoder_prediction_metrics(log, predict_encoder_log(log, ActuatorParameters()))
    assert metrics["position_rmse_rad"] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize(
    "invalid",
    [np.zeros(4), np.full(5, np.nan), np.zeros((5, 2)), np.zeros(6)],
)
def test_metrics_reject_invalid_prediction(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError):
        encoder_prediction_metrics(_short_log(), invalid)


@pytest.mark.parametrize("split", ["validation", "holdout"])
def test_fitting_refuses_noncalibration_logs(split: str) -> None:
    with pytest.raises(ValueError):
        fit_encoder_parameters([replace(_short_log(), split=split)])


def test_fitting_requires_logs_with_the_same_known_bench_configuration() -> None:
    log = _short_log()
    other = _record(log.commands, config=BenchConfig(kind="pendulum"))
    with pytest.raises(ValueError):
        fit_encoder_parameters([log, other])


def test_fitting_requires_nonempty_encoder_logs() -> None:
    with pytest.raises(ValueError):
        fit_encoder_parameters([])
    with pytest.raises(ValueError):
        fit_encoder_parameters([object()])


def test_fitting_requires_varying_excitation() -> None:
    with pytest.raises(ValueError):
        fit_encoder_parameters([_record(np.zeros(5))])


@pytest.mark.parametrize(
    ("budget", "value"),
    [
        ("max_delay_steps", -1),
        ("max_delay_steps", 1.5),
        ("max_delay_steps", True),
        ("max_nfev", 0),
        ("max_nfev", 2.5),
    ],
)
def test_fitting_rejects_invalid_budgets(budget: str, value: object) -> None:
    with pytest.raises(ValueError):
        fit_encoder_parameters([_short_log()], **{budget: value})


def test_fitting_enumerates_every_integer_delay_and_reports_all_candidates() -> None:
    log = _record(np.asarray([0.6, -0.4, 0.5, -0.3, 0.2, -0.1]))
    result = fit_encoder_parameters([log], max_delay_steps=2, max_nfev=3)
    assert isinstance(result, FitResult)
    assert [candidate["parameters"]["delay_steps"] for candidate in result.candidates] == [0, 1, 2]
    for candidate in result.candidates:
        assert set(candidate) == {
            "parameters",
            "calibration_loss",
            "optimizer_success",
            "nfev",
            "message",
        }
        assert np.isfinite(candidate["calibration_loss"])
        assert isinstance(candidate["optimizer_success"], bool)
        assert 1 <= candidate["nfev"] <= 3
        assert candidate["message"]
    losses = [candidate["calibration_loss"] for candidate in result.candidates]
    assert result.candidates[int(np.argmin(losses))]["parameters"] == {
        name: getattr(result.parameters, name) for name in (*PARAMETER_NAMES, "delay_steps")
    }


def test_fitted_parameters_stay_inside_the_reused_bounds() -> None:
    log = _record(np.asarray([0.7, -0.5, 0.4, -0.2, 0.3, -0.6]))
    result = fit_encoder_parameters([log], max_delay_steps=1, max_nfev=4)
    values = np.asarray([getattr(result.parameters, name) for name in PARAMETER_NAMES])
    assert np.all(values >= PARAMETER_LOWER - 1e-12)
    assert np.all(values <= PARAMETER_UPPER + 1e-12)
    for candidate in result.candidates:
        candidate_values = np.asarray([candidate["parameters"][name] for name in PARAMETER_NAMES])
        assert np.all(candidate_values >= PARAMETER_LOWER - 1e-12)
        assert np.all(candidate_values <= PARAMETER_UPPER + 1e-12)


def test_residual_scaling_matches_the_declared_position_scale() -> None:
    first = _record(np.asarray([0.5, -0.4, 0.3, -0.2]))
    second = _record(np.asarray([0.6, -0.1]))
    # At the fixed initial parameters the true bench prediction is unchanged.
    # Deliberate position offsets give residuals [1,2,3,4] and [5,6] after
    # division by .05 rad: sum of squares = 91 over six commands, not two logs.
    first = replace(first, positions=first.positions + np.asarray([0, 0.05, 0.10, 0.15, 0.20]))
    second = replace(second, positions=second.positions + np.asarray([0, 0.25, 0.30]))
    assert POSITION_SCALE_RAD == 0.05
    result = fit_encoder_parameters([first, second], max_delay_steps=0, max_nfev=1)
    assert result.candidates[0]["calibration_loss"] == pytest.approx(91.0 / 6.0, rel=1e-12)


@pytest.mark.parametrize("field", ["commands", "positions"])
def test_complex_log_values_cannot_silently_drop_the_imaginary_component(field: str) -> None:
    log = _short_log()
    values = getattr(log, field).astype(complex)
    values[1] += 0.25j
    with pytest.raises(ValueError):
        replace(log, **{field: values})


def test_complex_prediction_cannot_silently_drop_the_imaginary_component() -> None:
    log = _short_log()
    with pytest.raises(ValueError):
        encoder_prediction_metrics(log, log.positions.astype(complex) + 0.25j)


def test_fit_result_reports_local_jacobian_diagnostics() -> None:
    log = _record(_chirp(400, 0.002, (0.5, 0.3), (1.1, 3.3), (0.0, 0.4)))
    result = fit_encoder_parameters([log], max_delay_steps=1, max_nfev=6)
    assert len(result.jacobian_singular_values) == len(PARAMETER_NAMES)
    assert all(np.isfinite(value) for value in result.jacobian_singular_values)
    assert 0 <= result.jacobian_rank <= len(PARAMETER_NAMES)


def test_noise_free_identification_recovers_parameters_and_predicts_unseen_motion() -> None:
    truth = ActuatorParameters(armature=0.017, damping=0.105, coulomb_friction=0.043, delay_steps=2)
    config = BenchConfig()
    commands = _chirp(1000, config.dt, (0.55, 0.35, 0.15), (0.9, 2.7, 6.3), (0.0, 0.3, -1.0))
    calibration = _record(commands, parameters=truth, config=config)
    result = fit_encoder_parameters([calibration], max_delay_steps=2, max_nfev=25)
    assert result.parameters.delay_steps == truth.delay_steps
    for name in PARAMETER_NAMES:
        assert getattr(result.parameters, name) == pytest.approx(getattr(truth, name), rel=0.01)
    assert len(result.candidates) == 3
    assert result.jacobian_rank == 3

    # Different frequencies, phases and initial angle; never supplied to the fitter.
    holdout_commands = _chirp(750, config.dt, (0.45, 0.3, 0.12), (1.4, 3.8, 7.1), (0.7, -0.5, 0.0))
    holdout = _record(
        holdout_commands,
        parameters=truth,
        config=config,
        split="holdout",
        initial_position=0.2,
        name="holdout_trajectory",
    )
    fitted = encoder_prediction_metrics(holdout, predict_encoder_log(holdout, result.parameters))
    nominal = encoder_prediction_metrics(
        holdout, predict_encoder_log(holdout, ActuatorParameters())
    )
    assert fitted["position_rmse_rad"] < 1e-3
    assert fitted["position_rmse_rad"] < 0.01 * nominal["position_rmse_rad"]
