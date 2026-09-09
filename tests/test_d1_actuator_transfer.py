"""Synthetic torque-channel fit, causal compensation and short real D1 integration."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_d1_actuator_transfer.py"


@pytest.fixture(scope="module")
def transfer():
    spec = importlib.util.spec_from_file_location("d1_actuator_transfer_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def calibration(transfer):
    log, truth = transfer.record_synthetic_log("calibration", seed=10017, samples=2000)
    fitted = transfer.fit_torque_channel(log)
    return log, truth, fitted


def test_identification_recovers_synthetic_gain_tau_and_physics_delay(transfer, calibration):
    log, truth, result = calibration
    assert result["config"]["delay_steps"] == 2
    assert result["config"]["physics_dt_s"] == 0.002
    np.testing.assert_allclose(result["config"]["gain"], transfer.SYNTHETIC_WHEELS.gain, atol=0.01)
    np.testing.assert_allclose(
        result["config"]["time_constant_s"], transfer.SYNTHETIC_WHEELS.time_constant_s, atol=0.0005
    )
    assert np.sqrt(np.mean((log.measured_nm - truth) ** 2)) == pytest.approx(0.01, abs=0.0004)
    assert len(result["candidates"]) == 5
    assert all(len(item["optimizers"]) == 4 for item in result["candidates"])
    assert "armature" not in result["config"] and "coulomb_friction" not in result["config"]


def test_validation_is_a_new_waveform_and_cannot_select_parameters(transfer, calibration):
    log, _, fitted = calibration
    validation, _ = transfer.record_synthetic_log("validation", seed=20017, samples=1500)
    with pytest.raises(ValueError, match="only a calibration"):
        transfer.fit_torque_channel(validation)
    prediction = transfer.channel_response(
        validation.commands_nm, transfer.ActuatorChannelConfig(**fitted["config"])
    )
    assert np.max(np.sqrt(np.mean((prediction - validation.measured_nm) ** 2, axis=0))) < 0.012
    assert not np.array_equal(
        validation.commands_nm, log.commands_nm[: len(validation.commands_nm)]
    )
    with pytest.raises(ValueError):
        log.commands_nm[0, 0] = 10


def test_excitation_rng_is_local_and_repeatable(transfer):
    np.random.seed(91)
    expected = np.random.random(5)
    np.random.seed(91)
    first = transfer.excitation("calibration", seed=17, samples=200)
    np.testing.assert_array_equal(np.random.random(5), expected)
    np.testing.assert_array_equal(first, transfer.excitation("calibration", seed=17, samples=200))
    assert not np.array_equal(first, transfer.excitation("calibration", seed=18, samples=200))
    assert np.max(np.abs(first)) <= 3


def test_analytic_fit_response_matches_real_channel_not_a_different_time_convention(transfer):
    commands = transfer.excitation("calibration", seed=7, samples=120)
    config = transfer.SYNTHETIC_WHEELS
    actual = transfer.channel_response(commands, config)
    for axis in range(4):
        expected = transfer._linear_response(
            commands[:, axis], config.gain[axis], config.time_constant_s[axis], config.delay_steps
        )
        np.testing.assert_allclose(actual[:, axis], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("delay", [0, 2, 5])
def test_inverse_hand_calculation_cancels_lag_but_keeps_whole_pure_delay(transfer, delay):
    # alpha=0.5, g=0.8: first desired 1 Nm requires 2.5 Nm, then 1.25 Nm.
    config = transfer.ActuatorChannelConfig(
        n_axes=4,
        gain=0.8,
        time_constant_s=-0.002 / np.log(0.5),
        delay_steps=delay,
        torque_limit_nm=12,
    )
    inverse, channel = transfer.CausalTorqueInverse(config), transfer.ActuatorChannel(config)
    desired = np.asarray([1.0, 1.0, 0.4, -0.2, -0.2, 0.3, 0.1, 0.0])
    drive, applied = [], []
    for target in desired:
        request = inverse.step(np.full(4, target))
        drive.append(request)
        applied.append(channel.step(request, np.zeros(4)).applied_nm)
    np.testing.assert_allclose(drive[0], 2.5)
    np.testing.assert_allclose(drive[1], 1.25)
    expected = np.r_[np.zeros(delay), desired][: len(desired)]
    np.testing.assert_allclose(
        np.asarray(applied), np.repeat(expected[:, None], 4, axis=1), atol=1e-14
    )


def test_inverse_does_not_see_future_reference_or_true_channel_parameters(transfer):
    fitted = transfer.ActuatorChannelConfig(n_axes=4, gain=0.9, time_constant_s=0.01, delay_steps=3)
    first, second = transfer.CausalTorqueInverse(fitted), transfer.CausalTorqueInverse(fitted)
    prefix = (0.0, 0.1, 0.5, 0.3)
    requests = [first.step(np.full(4, value)) for value in prefix]
    other = [second.step(np.full(4, value)) for value in prefix]
    first.step(np.full(4, 100.0))
    second.step(np.full(4, -100.0))
    np.testing.assert_array_equal(requests, other)
    # The objects accept only fitted config + desired values, never a plant.
    assert first.fitted is fitted and second.fitted is fitted
    first.reset()
    np.testing.assert_array_equal(first.step(np.zeros(4)), np.zeros(4))


def test_saturation_prevents_exact_cancellation_and_is_not_hidden(transfer):
    config = transfer.ActuatorChannelConfig(
        n_axes=4, gain=0.8, time_constant_s=0.02, torque_limit_nm=12
    )
    inverse, physical = transfer.CausalTorqueInverse(config), transfer.ActuatorChannel(config)
    drive = inverse.step(np.full(4, 6.0))
    assert np.min(drive) > 12.0
    trace = physical.step(drive, np.zeros(4))
    np.testing.assert_array_equal(trace.limited_nm, np.full(4, 12.0))
    assert np.max(trace.applied_nm) < 6.0


def test_mixed_channel_delays_only_wheels_and_never_adds_velocity_friction(transfer):
    channel = transfer.WheelOnlyChannel(transfer.SYNTHETIC_WHEELS)
    torque = np.full(16, 1.0)
    first = channel.step(torque, np.arange(16) * 100.0)
    np.testing.assert_array_equal(first.applied_nm[transfer.LEGS], torque[transfer.LEGS])
    np.testing.assert_array_equal(first.applied_nm[transfer.WHEELS], np.zeros(4))
    channel.step(torque, np.zeros(16))
    third = channel.step(torque, np.zeros(16))
    assert np.all(third.applied_nm[transfer.WHEELS] > 0)
    metadata = channel.topology_metadata
    np.testing.assert_array_equal(
        np.asarray(metadata["actual_delay_physics_steps_per_axis"])[transfer.LEGS], 0
    )
    np.testing.assert_array_equal(
        np.asarray(metadata["actual_delay_physics_steps_per_axis"])[transfer.WHEELS], 2
    )
    channel.reset()
    np.testing.assert_array_equal(channel.step(torque, np.zeros(16)).applied_nm, first.applied_nm)
    assert channel.physics_step_count == 1


def test_real_wheel_velocity_loop_with_real_torque_channel_and_no_extra_friction(
    transfer, calibration
):
    from wheel_legged_control.actuator_bench import ActuatorBench, ActuatorParameters, BenchConfig
    from wheel_legged_control.wheel_control import WheelControlConfig, WheelVelocityController

    fitted = transfer.ActuatorChannelConfig(**calibration[2]["config"])
    inverse = transfer.CausalTorqueInverse(fitted)
    channel = transfer.ActuatorChannel(transfer.SYNTHETIC_WHEELS)
    bench = ActuatorBench(
        BenchConfig(torque_limit_nm=12.0),
        ActuatorParameters(armature=0.01, damping=0.08, coulomb_friction=0.0),
    )
    controller = WheelVelocityController(WheelControlConfig(torque_limit_nm=12.0))
    for _ in range(300):
        command = controller.compute(1.0, 0.0, bench.state[1])
        drive = inverse.step(np.full(4, command.command_torque_nm))
        applied = channel.step(drive, np.full(4, bench.state[1])).applied_nm[0]
        state = bench.step(float(applied))
        assert np.isfinite(state).all()
    assert abs(state[1] - 1.0) < 0.1


@pytest.mark.parametrize("case", ("ideal", "nonideal", "fitted_compensation"))
def test_real_d1_short_run_uses_installed_physics_channel_and_preserves_leg_passthrough(
    transfer,
    calibration,
    tmp_path,
    case,
):
    fitted = transfer.ActuatorChannelConfig(**calibration[2]["config"])
    directory = tmp_path / case
    result, rows = transfer.evaluate_case(
        directory, case=case, fitted_config=fitted, duration_s=0.03, terrain_index=0
    )
    assert len(rows) == 3 and result["physics_steps"] == 15
    assert result["terminal_reason"] == "time_limit"
    params = json.loads((directory / "parameters.json").read_text())
    topology = params["installed_channel"]
    assert params["environment_reset_metadata"]["source_schema"] == "d1-synchronized-oracle-v1"
    expected_delay = 0 if case == "ideal" else 2
    np.testing.assert_array_equal(
        np.asarray(topology["actual_delay_physics_steps_per_axis"])[transfer.WHEELS], expected_delay
    )
    assert (topology["fitted_inverse_config"] is not None) == (case == "fitted_compensation")
    with np.load(directory / "physical_trace.npz", allow_pickle=False) as trace:
        assert trace["applied_nm"].shape == (15, 16)
        np.testing.assert_array_equal(
            trace["controller_requested_nm"][:, transfer.LEGS],
            trace["applied_nm"][:, transfer.LEGS],
        )
        np.testing.assert_allclose(
            np.asarray([row["mechanical_power_w"] for row in rows]),
            np.abs(trace["applied_nm"] * trace["joint_velocity_rad_s"])
            .sum(axis=1)
            .reshape(3, 5)
            .mean(axis=1),
        )
        if case != "ideal":
            np.testing.assert_array_equal(trace["applied_nm"][:2, transfer.WHEELS], 0)
    with pytest.raises(FileExistsError):
        transfer.evaluate_case(directory, case=case, fitted_config=fitted, duration_s=0.03)


def test_summary_keeps_failures_even_when_short_prefix_tracking_is_good(transfer):
    rows = [
        {
            "time_s": 0.01,
            "velocity_error_mps": 0.06,
            "yaw_rate_error_rps": 0.04,
            "height_error_m": 0.01,
            "roll_error_rad": 0.03,
            "pitch_error_rad": 0.04,
            "mechanical_power_w": 3.0,
            "torque_input_clipped_fraction": 0.25,
        }
    ]
    result = transfer.summarize(
        rows, {"terminal_reason": "fall_or_body_contact", "terrain_exposure": {}}
    )
    assert not result["completed"] and not result["quality_pass"]
    assert result["attitude_rmse_rad"] == pytest.approx(0.05)
    assert result["drive_input_clipped_fraction"] == 0.25


@pytest.mark.parametrize("duration", [0, -1, 0.015, float("nan"), float("inf")])
def test_invalid_cli_protocol_fails_before_output_creation(transfer, tmp_path, duration):
    output = tmp_path / "bad"
    with pytest.raises(ValueError):
        transfer.run(output, duration_s=duration)
    assert not output.exists()


def test_no_source_snapshot_or_calibration_if_output_already_exists(
    transfer, tmp_path, monkeypatch
):
    def fail(*args, **kwargs):
        raise AssertionError("must reject before source or calibration")

    monkeypatch.setattr(transfer, "_source_snapshot", fail)
    with pytest.raises(FileExistsError):
        transfer.run(tmp_path)


def test_unexcited_or_nonfinite_fit_logs_rejected(transfer):
    log = transfer.TorqueLog(np.zeros((40, 4)), np.zeros((40, 4)), "calibration")
    with pytest.raises(ValueError, match="varying"):
        transfer.fit_torque_channel(log)
    with pytest.raises(ValueError, match="finite"):
        transfer.TorqueLog(np.full((40, 4), np.nan), np.zeros((40, 4)), "calibration")
    with pytest.raises(ValueError, match="clock"):
        transfer.TorqueLog(np.zeros((40, 4)), np.zeros((40, 4)), "calibration", physics_dt_s=0.01)
