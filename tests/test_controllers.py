import numpy as np

from wheel_legged_control.controllers import (
    LinearMPCController,
    LQRController,
    TrackingCommand,
)
from wheel_legged_control.model import WheelLeggedPlant


def test_lqr_closed_loop_is_stable() -> None:
    controller = LQRController(WheelLeggedPlant())
    assert np.max(np.abs(controller.closed_loop_eigenvalues)) < 1.0


def test_lqr_recovers_from_small_pitch_error() -> None:
    plant = WheelLeggedPlant()
    initial_state = np.array([0.0, np.deg2rad(5.0), 0.0, 0.0, 0.0, 0.0])
    plant.reset(initial_state)
    controller = LQRController(plant)
    controller.reset(initial_state)
    for _ in range(200):
        plant.step(controller.compute(plant.state(), TrackingCommand()))
    assert abs(plant.state()[1]) < np.deg2rad(0.2)


def test_mpc_returns_bounded_control() -> None:
    plant = WheelLeggedPlant()
    state = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0])
    plant.reset(state)
    controller = LinearMPCController(plant, horizon=10)
    controller.reset(state)
    control = controller.compute(state, TrackingCommand(velocity_mps=0.4))
    assert np.all(np.isfinite(control))
    assert np.all(control >= plant.actuator_low)
    assert np.all(control <= plant.actuator_high)
