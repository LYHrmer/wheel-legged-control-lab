import numpy as np

from wheel_legged_control.model import ACTION_SIZE, STATE_SIZE, WheelLeggedPlant


def test_nominal_control_is_an_equilibrium() -> None:
    plant = WheelLeggedPlant()
    acceleration = plant.acceleration(np.zeros(STATE_SIZE), plant.equilibrium_control)
    np.testing.assert_allclose(acceleration, 0.0, atol=1e-9)


def test_linearization_has_expected_shape_and_unstable_mode() -> None:
    plant = WheelLeggedPlant()
    a, b = plant.linearize()
    assert a.shape == (STATE_SIZE, STATE_SIZE)
    assert b.shape == (STATE_SIZE, ACTION_SIZE)
    assert np.max(np.abs(np.linalg.eigvals(a))) > 1.0


def test_step_returns_finite_state_and_respects_control_shape() -> None:
    plant = WheelLeggedPlant()
    plant.reset(np.array([0.0, 0.04, 0.0, 0.0, 0.0, 0.0]))
    state = plant.step(plant.equilibrium_control)
    assert state.shape == (STATE_SIZE,)
    assert np.all(np.isfinite(state))

