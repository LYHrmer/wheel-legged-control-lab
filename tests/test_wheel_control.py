from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from math import tanh

import numpy as np
import pytest

from wheel_legged_control.actuator_bench import (
    FRICTION_SMOOTHING_RAD_S,
    WHEEL_INERTIA_KG_M2,
    ActuatorBench,
    ActuatorParameters,
    BenchConfig,
)
from wheel_legged_control.wheel_control import WheelControlConfig, WheelVelocityController


def test_pi_uses_current_error_and_accepted_integral_in_current_command() -> None:
    controller = WheelVelocityController(WheelControlConfig(kp=0.2, ki=0.5, dt=0.1))
    first = controller.compute(2.0, 99.0, 0.5)
    assert first.error_rad_s == 1.5
    assert first.proportional_torque_nm == pytest.approx(0.3)
    assert first.integral_torque_nm == pytest.approx(0.075)
    assert first.feedforward_torque_nm == 0.0
    assert first.requested_torque_nm == pytest.approx(0.375)
    assert first.command_torque_nm == first.requested_torque_nm
    assert not first.saturated
    assert not first.integration_frozen
    second = controller.compute(2.0, 0.0, 1.0)
    assert second.integral_torque_nm == pytest.approx(0.125)
    assert second.command_torque_nm == pytest.approx(0.325)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_feedforward_uses_known_load_and_model_parameters(direction: float) -> None:
    parameters = ActuatorParameters(armature=0.017, damping=0.105, coulomb_friction=0.043)
    controller = WheelVelocityController(WheelControlConfig(kp=0.0, ki=0.0), parameters)
    reference, acceleration = direction * 0.02, direction * 3.0
    expected = (
        (WHEEL_INERTIA_KG_M2 + parameters.armature) * acceleration
        + parameters.damping * reference
        + parameters.coulomb_friction * tanh(reference / FRICTION_SMOOTHING_RAD_S)
    )
    output = controller.compute(reference, acceleration, 100.0)
    assert output.feedforward_torque_nm == pytest.approx(expected)
    assert output.command_torque_nm == pytest.approx(expected)
    assert controller.compute(reference, acceleration, -100.0).command_torque_nm == pytest.approx(
        expected
    )


def test_load_inertia_is_an_explicit_known_quantity() -> None:
    parameters = ActuatorParameters(armature=0.02, damping=0.0, coulomb_friction=0.0)
    controller = WheelVelocityController(
        WheelControlConfig(kp=0.0, ki=0.0), parameters, load_inertia_kg_m2=0.03
    )
    assert controller.compute(0.0, 2.0, 0.0).feedforward_torque_nm == pytest.approx(0.1)


def test_delay_parameter_does_not_claim_inverse_delay_compensation() -> None:
    parameters = ActuatorParameters()
    controllers = [
        WheelVelocityController(WheelControlConfig(), replace(parameters, delay_steps=delay))
        for delay in (0, 9)
    ]
    for reference, acceleration, measured in ((1.0, 2.0, 0.8), (0.3, -1.0, 0.4)):
        assert controllers[0].compute(reference, acceleration, measured) == controllers[1].compute(
            reference, acceleration, measured
        )


def test_pi_and_two_feedforward_models_share_feedback_gains_and_timing() -> None:
    config = WheelControlConfig()
    controllers = [
        WheelVelocityController(config, parameters)
        for parameters in (
            None,
            ActuatorParameters(),
            ActuatorParameters(armature=0.017, damping=0.105, coulomb_friction=0.043),
        )
    ]
    for reference, acceleration, measured in ((1.0, 2.0, 0.8), (0.3, -1.0, 0.4)):
        outputs = [item.compute(reference, acceleration, measured) for item in controllers]
        for output in outputs[1:]:
            assert output.error_rad_s == outputs[0].error_rad_s
            assert output.proportional_torque_nm == outputs[0].proportional_torque_nm
            assert output.integral_torque_nm == outputs[0].integral_torque_nm
        for output in outputs:
            assert not output.saturated
            assert output.command_torque_nm == pytest.approx(
                output.proportional_torque_nm
                + output.integral_torque_nm
                + output.feedforward_torque_nm
            )


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_saturated_feedback_does_not_accumulate_outward_integral(direction: float) -> None:
    config = WheelControlConfig(kp=2.0, ki=1.0, dt=0.1, torque_limit_nm=1.0)
    controller = WheelVelocityController(config)
    for _ in range(100):
        output = controller.compute(direction, 0.0, 0.0)
        assert output.command_torque_nm == direction
        assert output.requested_torque_nm == 2.0 * direction
        assert output.integral_torque_nm == 0.0
        assert output.saturated
        assert output.integration_frozen
    recovered = controller.compute(0.0, 0.0, 0.0)
    assert recovered.command_torque_nm == 0.0
    assert not recovered.integration_frozen


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_rejected_candidate_is_recomputed_before_output(direction: float) -> None:
    config = WheelControlConfig(kp=0.0, ki=1.0, dt=1.0, torque_limit_nm=1.0)
    controller = WheelVelocityController(config)
    controller.compute(0.8 * direction, 0.0, 0.0)
    output = controller.compute(0.3 * direction, 0.0, 0.0)
    assert output.integral_torque_nm == pytest.approx(0.8 * direction)
    assert output.command_torque_nm == pytest.approx(0.8 * direction)
    assert output.integration_frozen
    assert not output.saturated


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_inward_integral_is_accepted_even_while_total_remains_saturated(direction: float) -> None:
    config = WheelControlConfig(kp=0.0, ki=1.0, dt=1.0, torque_limit_nm=1.0)
    parameters = ActuatorParameters(armature=0.0, damping=2.0, coulomb_friction=0.0)
    controller = WheelVelocityController(config, parameters)
    controller.compute(0.0, 0.0, -0.8 * direction)
    output = controller.compute(direction, 0.0, 1.1 * direction)
    assert output.integral_torque_nm == pytest.approx(0.7 * direction)
    assert output.requested_torque_nm == pytest.approx(2.7 * direction)
    assert output.command_torque_nm == direction
    assert output.saturated
    assert not output.integration_frozen
    recovered = controller.compute(0.0, 0.0, 0.5 * direction)
    assert recovered.command_torque_nm == pytest.approx(0.2 * direction)


def test_exact_limit_is_not_saturation_or_a_reason_to_freeze_integral() -> None:
    controller = WheelVelocityController(
        WheelControlConfig(kp=0.0, ki=1.0, dt=1.0, torque_limit_nm=1.0)
    )
    output = controller.compute(1.0, 0.0, 0.0)
    assert output.command_torque_nm == 1.0
    assert output.integral_torque_nm == 1.0
    assert not output.saturated
    assert not output.integration_frozen


def test_reset_clears_integral_without_affecting_other_controller_or_snapshot() -> None:
    config = WheelControlConfig(kp=0.0, ki=1.0, dt=0.1)
    first, second = WheelVelocityController(config), WheelVelocityController(config)
    snapshot = first.compute(1.0, 0.0, 0.0)
    second.compute(1.0, 0.0, 0.0)
    first.reset()
    assert first.compute(0.0, 0.0, 0.0).command_torque_nm == 0.0
    assert second.compute(0.0, 0.0, 0.0).command_torque_nm == pytest.approx(0.1)
    assert snapshot.integral_torque_nm == pytest.approx(0.1)
    with pytest.raises(FrozenInstanceError):
        snapshot.integral_torque_nm = 9.0
    with pytest.raises(FrozenInstanceError):
        config.kp = 9.0


@pytest.mark.parametrize("field", ["kp", "ki", "dt", "torque_limit_nm"])
@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan"), True, "0.1"])
def test_config_rejects_invalid_scalars(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        WheelControlConfig(**{field: value})


@pytest.mark.parametrize("field", ["dt", "torque_limit_nm"])
def test_config_rejects_zero_timing_and_limit(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        WheelControlConfig(**{field: 0.0})


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True, "1"])
def test_constructor_rejects_invalid_load_inertia(value: object) -> None:
    with pytest.raises(ValueError, match="load_inertia_kg_m2"):
        WheelVelocityController(WheelControlConfig(), load_inertia_kg_m2=value)


def test_constructor_rejects_wrong_config_or_parameter_type() -> None:
    with pytest.raises(TypeError, match="config"):
        WheelVelocityController({})
    with pytest.raises(TypeError, match="feedforward_parameters"):
        WheelVelocityController(WheelControlConfig(), {})


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1", [1.0]])
def test_compute_rejects_invalid_scalar_inputs_without_advancing_integral(
    index: int, value: object
) -> None:
    controller = WheelVelocityController(WheelControlConfig(kp=0.0, ki=1.0, dt=0.1))
    controller.compute(1.0, 0.0, 0.0)
    arguments = [1.0, 0.0, 0.0]
    arguments[index] = value
    with pytest.raises(ValueError, match="finite real scalar"):
        controller.compute(*arguments)
    assert controller.compute(0.0, 0.0, 0.0).integral_torque_nm == pytest.approx(0.1)


def test_nonfinite_arithmetic_is_rejected_before_changing_state() -> None:
    controller = WheelVelocityController(WheelControlConfig(kp=0.0, ki=1.0, dt=0.1))
    controller.compute(1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="arithmetic"):
        controller.compute(1e308, 0.0, -1e308)
    assert controller.compute(0.0, 0.0, 0.0).integral_torque_nm == pytest.approx(0.1)


def test_numpy_real_scalars_are_accepted_and_outputs_remain_python_scalars() -> None:
    controller = WheelVelocityController(WheelControlConfig(kp=np.float64(0.1)))
    output = controller.compute(np.float64(1.0), np.float32(0.0), np.int64(0))
    assert type(output.error_rad_s) is float
    assert type(output.command_torque_nm) is float
    assert type(output.saturated) is bool


def test_constant_speed_pi_closes_the_loop_on_a_nonideal_wheel() -> None:
    bench = ActuatorBench(
        BenchConfig(),
        ActuatorParameters(armature=0.017, damping=0.105, coulomb_friction=0.043, delay_steps=2),
    )
    controller = WheelVelocityController(WheelControlConfig())
    for _ in range(1500):
        output = controller.compute(3.0, 0.0, bench.state[1])
        bench.step(output.command_torque_nm)
        assert abs(output.command_torque_nm) <= bench.config.torque_limit_nm
        assert np.isfinite(bench.state).all()
    assert bench.state[1] == pytest.approx(3.0, abs=0.02)
