"""The same torque-channel interface serves one-axis benches and all 16 D1 joints."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from wheel_legged_control.d1.actuator_channel import (
    ActuatorChannel,
    ActuatorChannelConfig,
)


def test_default_channel_is_exact_float64_identity_at_all_16_axes():
    channel = ActuatorChannel(ActuatorChannelConfig())
    assert channel.config.n_axes == 16
    velocity = np.linspace(-2, 2, 16)
    for requested in (np.full(16, 1e16), np.linspace(-1e-12, 1e-12, 16), np.arange(-8, 8.0)):
        trace = channel.step(requested, velocity)
        for name in ("requested_nm", "limited_nm", "delayed_nm", "applied_nm"):
            value = getattr(trace, name)
            assert value.dtype == np.float64
            np.testing.assert_array_equal(value, requested)
        np.testing.assert_array_equal(trace.joint_velocity_rps, velocity)


@pytest.mark.parametrize("delay", (0, 1, 2, 5))
def test_delay_is_counted_in_physics_steps_with_zero_prehistory(delay):
    channel = ActuatorChannel(ActuatorChannelConfig(n_axes=1, delay_steps=delay))
    command = [1.0, -2.0, 3.0, 0.0, 0.0, 0.0, 0.0]
    expected = ([0.0] * delay + command)[: len(command)]
    observed = [channel.step([value], [0.0]).applied_nm[0] for value in command]
    assert observed == expected


@pytest.mark.parametrize("dt", (0.001, 0.002, 0.005))
def test_lag_uses_exact_physics_dt_and_supports_heterogeneous_time_constants(dt):
    channel = ActuatorChannel(
        ActuatorChannelConfig(n_axes=3, physics_dt_s=dt, time_constant_s=(0, 0.01, 0.025))
    )
    command = np.asarray((0.7, -2.0, 3.0))
    for step in range(1, 11):
        expected = command * np.asarray(
            (1.0, -np.expm1(-step * dt / 0.01), -np.expm1(-step * dt / 0.025))
        )
        trace = channel.step(command, np.zeros(3))
        np.testing.assert_allclose(trace.applied_nm, expected, rtol=1e-14, atol=1e-15)
        np.testing.assert_array_equal(trace.delayed_nm, command)


def test_gain_does_not_scale_limits_and_trace_distinguishes_each_stage():
    channel = ActuatorChannel(
        ActuatorChannelConfig(n_axes=2, torque_limit_nm=(2.0, 1.0), gain=(3.0, 0.5), delay_steps=1)
    )
    first = channel.step((5.0, -4.0), (0.0, 0.0))
    np.testing.assert_array_equal(first.requested_nm, (5.0, -4.0))
    np.testing.assert_array_equal(first.limited_nm, (2.0, -1.0))
    np.testing.assert_array_equal(first.delayed_nm, (0.0, 0.0))
    second = channel.step((0.0, 0.0), (0.0, 0.0))
    np.testing.assert_array_equal(second.delayed_nm, (2.0, -1.0))
    np.testing.assert_array_equal(second.applied_nm, (2.0, -0.5))


def test_velocity_is_recorded_without_implicitly_adding_or_subtracting_friction():
    channel = ActuatorChannel(ActuatorChannelConfig(n_axes=3))
    trace = channel.step((0.0, 1.0, -2.0), (9.0, -8.0, 7.0))
    np.testing.assert_array_equal(trace.applied_nm, (0.0, 1.0, -2.0))
    np.testing.assert_array_equal(trace.joint_velocity_rps, (9.0, -8.0, 7.0))


def test_zero_gain_is_allowed_without_changing_delayed_signal():
    channel = ActuatorChannel(ActuatorChannelConfig(n_axes=1, gain=0.0))
    trace = channel.step((2.0,), (0.0,))
    np.testing.assert_array_equal(trace.delayed_nm, (2.0,))
    np.testing.assert_array_equal(trace.applied_nm, (0.0,))


def test_reset_clears_queue_and_lag_and_repeats_initial_response():
    channel = ActuatorChannel(ActuatorChannelConfig(n_axes=1, delay_steps=2, time_constant_s=0.01))
    original = [channel.step([1.0], [0.0]).applied_nm.copy() for _ in range(8)]
    assert channel.reset() is None
    replay = [channel.step([1.0], [0.0]).applied_nm.copy() for _ in range(8)]
    np.testing.assert_array_equal(replay, original)
    channel.reset()
    np.testing.assert_array_equal(channel.step([0.0], [0.0]).applied_nm, (0.0,))


def test_trace_is_frozen_and_its_arrays_cannot_be_made_writable():
    channel = ActuatorChannel(ActuatorChannelConfig(n_axes=2))
    torque, velocity = np.asarray((1.0, 2.0)), np.asarray((3.0, 4.0))
    trace = channel.step(torque, velocity)
    torque[:] = 99
    velocity[:] = 99
    channel.step((5.0, 6.0), (7.0, 8.0))
    channel.reset()
    np.testing.assert_array_equal(trace.requested_nm, (1.0, 2.0))
    np.testing.assert_array_equal(trace.joint_velocity_rps, (3.0, 4.0))
    for name in ("requested_nm", "limited_nm", "delayed_nm", "applied_nm", "joint_velocity_rps"):
        value = getattr(trace, name)
        with pytest.raises(ValueError):
            value[0] = 0
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        trace.applied_nm = np.zeros(2)


def test_configuration_copies_sequences_into_immutable_values():
    gains = [1.0, 2.0]
    config = ActuatorChannelConfig(
        n_axes=2, gain=gains, time_constant_s=[0, 0.01], torque_limit_nm=[2.0, 3.0]
    )
    gains[0] = 100
    assert config.gain == (1.0, 2.0)
    assert config.time_constant_s == (0.0, 0.01)
    assert config.torque_limit_nm == (2.0, 3.0)
    with pytest.raises(FrozenInstanceError):
        config.gain = 2.0
    channel = ActuatorChannel(config)
    with pytest.raises(AttributeError):
        channel.config = ActuatorChannelConfig(n_axes=2)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("n_axes", 0),
        ("n_axes", True),
        ("n_axes", 2.0),
        ("physics_dt_s", 0),
        ("physics_dt_s", np.inf),
        ("physics_dt_s", True),
        ("delay_steps", -1),
        ("delay_steps", 1.5),
        ("delay_steps", False),
        ("time_constant_s", -0.1),
        ("time_constant_s", np.nan),
        ("gain", -1),
        ("gain", np.inf),
        ("gain", True),
        ("gain", (1.0,)),
        ("gain", (1.0, 1j)),
        ("gain", [1.0, True]),
        ("torque_limit_nm", 0),
        ("torque_limit_nm", -1),
        ("torque_limit_nm", np.inf),
        ("torque_limit_nm", "2.0"),
    ],
)
def test_configuration_rejects_invalid_units_dimensions_and_nonfinite_values(field, bad):
    kwargs = {"n_axes": 2, field: bad}
    with pytest.raises((TypeError, ValueError)):
        ActuatorChannelConfig(**kwargs)


@pytest.mark.parametrize(
    "bad", [1.0, (1.0,), ((1.0, 2.0),), (np.nan, 0), (0, np.inf), (1, 2j), (1.0, True), ("1", "2")]
)
@pytest.mark.parametrize("argument", ("torque", "velocity"))
def test_invalid_step_input_does_not_advance_channel_state(bad, argument):
    config = ActuatorChannelConfig(n_axes=2, delay_steps=1, time_constant_s=0.01)
    channel, control = ActuatorChannel(config), ActuatorChannel(config)
    channel.step((1.0, -1.0), (0.0, 0.0))
    control.step((1.0, -1.0), (0.0, 0.0))
    torque = bad if argument == "torque" else (2.0, 2.0)
    velocity = bad if argument == "velocity" else (0.0, 0.0)
    with pytest.raises((TypeError, ValueError)):
        channel.step(torque, velocity)
    observed = channel.step((3.0, -2.0), (0.0, 0.0))
    expected = control.step((3.0, -2.0), (0.0, 0.0))
    np.testing.assert_array_equal(observed.delayed_nm, expected.delayed_nm)
    np.testing.assert_array_equal(observed.applied_nm, expected.applied_nm)


def test_overflow_rejection_does_not_commit_lag_state():
    config = ActuatorChannelConfig(n_axes=1, gain=1e308, time_constant_s=0.002)
    channel, control = ActuatorChannel(config), ActuatorChannel(config)
    with pytest.raises((ValueError, FloatingPointError, OverflowError)):
        channel.step((100.0,), (0.0,))
    actual = channel.step((0.1,), (0.0,))
    expected = control.step((0.1,), (0.0,))
    np.testing.assert_array_equal(actual.applied_nm, expected.applied_nm)


@pytest.mark.parametrize("kind", ("wheel", "pendulum"))
def test_one_axis_channel_matches_existing_bench_motor_delay_without_double_friction(kind):
    pytest.importorskip("mujoco")
    from wheel_legged_control.actuator_bench import (
        ActuatorBench,
        ActuatorParameters,
        BenchConfig,
    )

    bench_config = BenchConfig(kind=kind, dt=0.002, torque_limit_nm=2.0)
    reference = ActuatorBench(bench_config, ActuatorParameters(delay_steps=2))
    external_channel_plant = ActuatorBench(bench_config, ActuatorParameters(delay_steps=0))
    channel = ActuatorChannel(
        ActuatorChannelConfig(n_axes=1, physics_dt_s=0.002, delay_steps=2, torque_limit_nm=2.0)
    )
    reference.reset(position=0.15, velocity=-0.3)
    external_channel_plant.reset(position=0.15, velocity=-0.3)
    for command in 3.0 * np.sin(np.arange(150) * 0.2):
        trace = channel.step([command], [external_channel_plant.state[1]])
        expected = reference.step(command)
        observed = external_channel_plant.step(trace.applied_nm[0])
        np.testing.assert_array_equal(trace.applied_nm[0], reference.applied_torque_nm)
        np.testing.assert_array_equal(observed, expected)
