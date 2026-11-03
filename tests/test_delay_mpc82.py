"""Synthetic delay-prediction checks; no plant instance or integration required."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("mujoco")
from wheel_legged_control.controllers import LinearMPCController, TrackingCommand
from wheel_legged_control.delay_mpc82 import DelayMPC82, prediction_matrices


class FakePlant:
    def __init__(self):
        self.a = np.diag([0.9, 0.85, 0.8, 0.75, 0.7, 0.65])
        self.b = np.array(
            [[0.02, 0.01], [0.01, 0.02], [0.0, 0.03], [0.04, 0.0], [0.02, 0.01], [0.0, 0.05]]
        )
        self.control_dt = 0.02
        self.calls = 0

    def linearize(self):
        self.calls += 1
        return (self.a.copy(), self.b.copy())

    @property
    def equilibrium_control(self):
        return np.array([0.0, 100.0])

    @property
    def actuator_low(self):
        return np.array([-80.0, 0.0])

    @property
    def actuator_high(self):
        return np.array([80.0, 260.0])


@pytest.mark.parametrize("delay", [0, 2])
def test_condensed_predictions_match_explicit_delayed_recursion(delay):
    plant = FakePlant()
    horizon = 8
    (sx, su) = prediction_matrices(plant.a, plant.b, horizon, delay)
    state = np.array([0.1, -0.04, 0.03, 0.2, -0.07, 0.08])
    pending = [np.array([0.7, -0.3]), np.array([-0.2, 0.6])][:delay]
    inputs = np.arange(horizon * 2, dtype=float).reshape(horizon, 2) * 0.03 - 0.2
    extended = np.r_[state, np.asarray(pending).ravel()]
    expected = []
    queue = [p.copy() for p in pending]
    for control in inputs:
        queue.append(control)
        applied = queue.pop(0)
        state = plant.a @ state + plant.b @ applied
        expected.append(state.copy())
    np.testing.assert_allclose(
        sx @ extended + su @ inputs.ravel(), np.asarray(expected).ravel(), rtol=1e-13, atol=1e-14
    )
    if delay:
        assert np.count_nonzero(su[:12]) == 0
        np.testing.assert_array_equal(su[12:18, :2], plant.b)
        np.testing.assert_array_equal(sx[:6, 6:8], plant.b)
        np.testing.assert_array_equal(sx[:6, 8:10], np.zeros((6, 2)))


def test_zero_delay_commands_and_warm_start_are_bit_exact_frozen_parent():
    plant = FakePlant()
    (old, new) = (LinearMPCController(plant), DelayMPC82(plant, delay_steps=0))
    state = np.array([0.1, 0.02, -0.01, 0.1, -0.01, 0.02])
    old.reset(state)
    new.reset(state)
    for index in range(6):
        command = TrackingCommand(0.15 if index < 3 else -0.2, 0.025)
        measured = state + np.arange(6) * index * 0.0001
        np.testing.assert_array_equal(
            new.compute(measured, command), old.compute(measured, command)
        )
        np.testing.assert_array_equal(new._solution, old._solution)
        assert new._x_reference == old._x_reference
        assert new.pending_controls.shape == (0, 2)


def test_delay_queue_contains_whole_issued_control_and_resets_to_equilibrium():
    plant = FakePlant()
    controller = DelayMPC82(plant, delay_steps=2)
    state = np.array([0.01, 0.02, -0.015, 0.02, 0.0, 0.0])
    controller.reset(state)
    queue = [plant.equilibrium_control.copy() for _ in range(2)]
    issued = []
    for index in range(5):
        before = np.asarray(queue).copy()
        control = controller.compute(state, TrackingCommand(0.1, 0.02))
        np.testing.assert_array_equal(controller.last_pending_before, before)
        queue.append(control.copy())
        applied = queue.pop(0)
        np.testing.assert_array_equal(controller.pending_controls, np.asarray(queue))
        if index < 2:
            np.testing.assert_array_equal(applied, plant.equilibrium_control)
        else:
            np.testing.assert_array_equal(applied, issued[index - 2])
        issued.append(control.copy())
        state = plant.a @ state + plant.b @ (applied - plant.equilibrium_control)
    exposed = controller.pending_controls
    exposed[:] = 12345
    assert not np.any(controller.pending_controls == 12345)
    controller.reset(state)
    np.testing.assert_array_equal(
        controller.pending_controls, np.tile(plant.equilibrium_control, (2, 1))
    )
    assert np.count_nonzero(controller._solution) == 0


def test_only_prediction_changes_cost_model_constraints_target_and_horizon_stay():
    plant = FakePlant()
    (old, new) = (LinearMPCController(plant), DelayMPC82(plant, delay_steps=2))
    for name in ("a", "b", "q", "r", "terminal_q", "_qbar", "_rbar"):
        np.testing.assert_array_equal(getattr(new, name), getattr(old, name))
    assert new.horizon == old.horizon == 20
    assert new._solution.shape == old._solution.shape == (40,)
    assert new._sx.shape == (120, 10) and new._su.shape == (120, 40)
    assert new.terminal_q.shape == (6, 6) and new.q[2, 2] == 90.0
    state = np.array([0.04, 0.01, 0.0, 0.03, 0.0, 0.0])
    command = TrackingCommand(0.2, 0.03)
    np.testing.assert_array_equal(
        new._target_trajectory(np.r_[state, np.zeros(4)], command),
        old._target_trajectory(state, command),
    )
    original_a = new.a.copy()
    plant.a[:] = 0.0
    np.testing.assert_array_equal(new.a, original_a)
    assert plant.calls == 2


@pytest.mark.parametrize("bad", [-1, 1, 3, True, 2.0])
def test_unknown_delay_is_rejected(bad):
    with pytest.raises(ValueError, match="delay"):
        DelayMPC82(FakePlant(), delay_steps=bad)


def test_invalid_state_and_matrices_fail_explicitly():
    controller = DelayMPC82(FakePlant(), delay_steps=2)
    for state in (np.zeros(10), np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.nan])):
        with pytest.raises(ValueError, match="physical6"):
            controller.compute(state, TrackingCommand())
    with pytest.raises(ValueError, match="6x6"):
        prediction_matrices(np.eye(7), np.zeros((7, 2)), 20, 2)
