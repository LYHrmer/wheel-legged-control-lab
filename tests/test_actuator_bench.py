from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from wheel_legged_control.actuator_bench import (
    FRICTION_SMOOTHING_RAD_S,
    GRAVITY_M_S2,
    PENDULUM_LENGTH_M,
    PENDULUM_MASS_KG,
    PENDULUM_PIVOT_INERTIA_KG_M2,
    WHEEL_INERTIA_KG_M2,
    ActuatorBench,
    ActuatorParameters,
    BenchConfig,
)


def _free_rotor(*, delay_steps: int = 0) -> ActuatorBench:
    return ActuatorBench(
        BenchConfig(),
        ActuatorParameters(
            armature=0.01, damping=0.0, coulomb_friction=0.0, delay_steps=delay_steps
        ),
    )


@pytest.mark.parametrize("delay_steps", [0, 1, 3])
def test_delay_uses_exact_physics_step_history(delay_steps: int) -> None:
    bench = _free_rotor(delay_steps=delay_steps)
    commands = [0.2, -0.4, 0.6, 0.0, 0.0, 0.0]
    expected = ([0.0] * delay_steps + commands)[: len(commands)]
    measured = []
    for command in commands:
        previous_velocity = bench.state[1]
        state = bench.step(command)
        measured.append(bench.applied_torque_nm)
        inertia = WHEEL_INERTIA_KG_M2 + bench.parameters.armature
        assert state[1] - previous_velocity == pytest.approx(
            bench.applied_torque_nm * bench.config.dt / inertia
        )
    assert measured == expected


def test_reset_clears_delay_history_and_state_is_a_copy() -> None:
    bench = _free_rotor(delay_steps=2)
    bench.step(1.0)
    bench.step(1.0)
    bench.reset(position=0.3, velocity=-0.2)
    snapshot = bench.state
    snapshot[:] = 99.0
    np.testing.assert_array_equal(bench.state, [0.3, -0.2])
    assert bench.applied_torque_nm == 0.0
    for _ in range(2):
        bench.step(0.0)
        assert bench.applied_torque_nm == 0.0
        assert bench.state[1] == pytest.approx(-0.2)


@pytest.mark.parametrize("command", [-100.0, 100.0])
def test_torque_is_clipped_before_delay(command: float) -> None:
    bench = _free_rotor(delay_steps=1)
    bench.step(command)
    assert bench.applied_torque_nm == 0.0
    bench.step(0.0)
    expected_torque = np.sign(command) * bench.config.torque_limit_nm
    assert bench.applied_torque_nm == expected_torque
    assert bench.state[1] == pytest.approx(
        expected_torque * bench.config.dt / (WHEEL_INERTIA_KG_M2 + bench.parameters.armature)
    )


def test_free_rotor_matches_constant_torque_acceleration() -> None:
    bench = _free_rotor()
    torque = 0.2
    steps = 250
    for _ in range(steps):
        state = bench.step(torque)
    acceleration = torque / (WHEEL_INERTIA_KG_M2 + bench.parameters.armature)
    assert state[1] == pytest.approx(acceleration * steps * bench.config.dt, rel=1e-12)
    # Semi-implicit position integration uses each step's updated velocity.
    assert state[0] == pytest.approx(
        acceleration * bench.config.dt**2 * steps * (steps + 1) / 2.0, rel=1e-12
    )


def test_viscous_damping_reduces_energy_and_matches_implicit_update() -> None:
    parameters = ActuatorParameters(armature=0.01, damping=0.08, coulomb_friction=0.0)
    bench = ActuatorBench(BenchConfig(), parameters)
    bench.reset(velocity=4.0)
    inertia = WHEEL_INERTIA_KG_M2 + parameters.armature
    velocities = [bench.state[1]]
    for _ in range(200):
        velocities.append(bench.step(0.0)[1])
    ratio = inertia / (inertia + parameters.damping * bench.config.dt)
    np.testing.assert_allclose(velocities, 4.0 * ratio ** np.arange(201), rtol=1e-12)
    energy = 0.5 * inertia * np.square(velocities)
    assert np.all(np.diff(energy) < 0.0)


def test_wheel_angle_is_continuous_across_full_turns() -> None:
    bench = _free_rotor()
    bench.reset(position=2.0 * np.pi - 0.01, velocity=10.0)
    state = bench.step(0.0)
    assert state[0] == pytest.approx(2.0 * np.pi + 0.01)
    bench.reset(position=-2.0 * np.pi + 0.01, velocity=-10.0)
    assert bench.step(0.0)[0] == pytest.approx(-2.0 * np.pi - 0.01)


def test_pendulum_gravity_matches_known_load_and_downward_equilibrium() -> None:
    parameters = ActuatorParameters(armature=0.01, damping=0.0, coulomb_friction=0.0)
    bench = ActuatorBench(BenchConfig(kind="pendulum"), parameters)
    np.testing.assert_allclose(bench.step(0.0), [0.0, 0.0], atol=1e-15)
    angle = 0.2
    bench.reset(position=angle)
    state = bench.step(0.0)
    torque = -PENDULUM_MASS_KG * GRAVITY_M_S2 * PENDULUM_LENGTH_M / 2.0 * np.sin(angle)
    expected_velocity = (
        torque * bench.config.dt / (PENDULUM_PIVOT_INERTIA_KG_M2 + parameters.armature)
    )
    assert state[1] == pytest.approx(expected_velocity, rel=1e-12)
    assert state[0] < angle


@pytest.mark.parametrize("velocity", [-1.0, 1.0])
def test_smooth_coulomb_friction_is_applied_once_and_opposes_motion(velocity: float) -> None:
    parameters = ActuatorParameters(armature=0.01, damping=0.0, coulomb_friction=0.025)
    bench = ActuatorBench(BenchConfig(), parameters)
    bench.reset(velocity=velocity)
    state = bench.step(0.0)
    friction = -parameters.coulomb_friction * np.tanh(velocity / FRICTION_SMOOTHING_RAD_S)
    assert state[1] == pytest.approx(
        velocity + friction * bench.config.dt / (WHEEL_INERTIA_KG_M2 + parameters.armature)
    )
    assert abs(state[1]) < abs(velocity)


def test_hidden_stribeck_term_adds_low_speed_friction() -> None:
    parameters = ActuatorParameters(damping=0.0)
    nominal = ActuatorBench(BenchConfig(), parameters)
    hidden = ActuatorBench(BenchConfig(), parameters, stribeck_friction_nm=0.04)
    for bench in (nominal, hidden):
        bench.reset(velocity=0.1)
    assert 0.0 < hidden.step(0.0)[1] < nominal.step(0.0)[1]


@pytest.mark.parametrize("field", ["armature", "damping", "coulomb_friction"])
@pytest.mark.parametrize("value", [-0.01, np.nan, np.inf, True, "0.1"])
def test_invalid_physical_parameters_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        ActuatorParameters(**{field: value})


@pytest.mark.parametrize("value", [-1, 0.5, np.nan, True, "1"])
def test_delay_must_be_a_nonnegative_integer(value: object) -> None:
    with pytest.raises(ValueError):
        ActuatorParameters(delay_steps=value)


@pytest.mark.parametrize("field", ["dt", "torque_limit_nm"])
@pytest.mark.parametrize("value", [0.0, -0.01, np.nan, np.inf, True, "0.1"])
def test_invalid_config_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        BenchConfig(**{field: value})


@pytest.mark.parametrize("kind", ["leg", None, ["wheel"]])
def test_invalid_kind_is_rejected(kind: object) -> None:
    with pytest.raises(ValueError):
        BenchConfig(kind=kind)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, "0.1", [0.1]])
def test_invalid_commands_and_initial_states_do_not_mutate_bench(value: object) -> None:
    bench = _free_rotor()
    original = bench.state
    with pytest.raises(ValueError):
        bench.step(value)
    with pytest.raises(ValueError):
        bench.reset(position=value)
    with pytest.raises(ValueError):
        bench.reset(velocity=value)
    np.testing.assert_array_equal(bench.state, original)


@pytest.mark.parametrize("value", [-0.1, np.nan, np.inf, True])
def test_invalid_stribeck_is_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        ActuatorBench(BenchConfig(), ActuatorParameters(), stribeck_friction_nm=value)


def test_configuration_records_are_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        BenchConfig().dt = 0.1
    with pytest.raises(FrozenInstanceError):
        ActuatorParameters().damping = 1.0


def test_constructor_requires_typed_configuration() -> None:
    with pytest.raises(TypeError):
        ActuatorBench({}, ActuatorParameters())
    with pytest.raises(TypeError):
        ActuatorBench(BenchConfig(), {})


def test_zero_armature_uses_the_load_inertia_without_hidden_extra_inertia() -> None:
    bench = ActuatorBench(
        BenchConfig(), ActuatorParameters(armature=0.0, damping=0.0, coulomb_friction=0.0)
    )
    assert bench.step(0.1)[1] == pytest.approx(0.1 * bench.config.dt / WHEEL_INERTIA_KG_M2)
