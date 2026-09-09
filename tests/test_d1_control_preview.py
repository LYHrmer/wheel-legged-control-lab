from dataclasses import replace

import numpy as np
import pytest

from wheel_legged_control.d1 import hierarchical
from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.hierarchical import D1LQRVMCController, D1MPCVMCController
from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource


@pytest.fixture
def plant():
    return D1Plant(sampling_mode="synchronized")


@pytest.mark.parametrize("controller_type", (D1LQRVMCController, D1MPCVMCController))
def test_preview_does_not_advance_memory_and_compute_consumes_once(plant, controller_type):
    controller = controller_type(plant)
    state = D1MujocoTruthStateSource(plant).reset()
    command = D1Command(forward_velocity_mps=0.12)
    memory = controller.control_memory
    proposal = controller.preview_baseline(command, state)
    for _ in range(3):
        assert controller.preview_baseline(command, state) is proposal
        assert controller.control_memory == memory
    assert controller.last_longitudinal_force_n == 0.0
    controller.compute(command, state)
    assert controller.control_memory.distance_reference_m == pytest.approx(0.0012)
    with pytest.raises(RuntimeError, match="already been consumed"):
        controller.compute(command, state)
    with pytest.raises(RuntimeError, match="already been consumed"):
        controller.preview_baseline(command, state)
    controller.reset()
    assert controller.control_memory == memory
    assert controller.preview_baseline(command, state).tick == 0


def test_lqr_preview_matches_hand_calculation_not_previous_total_force(plant):
    controller = D1LQRVMCController(plant)
    state = D1MujocoTruthStateSource(plant).reset()
    command = D1Command(forward_velocity_mps=0.08, base_height_m=0.46, pitch_rad=0.01)
    controller.last_longitudinal_force_n = -123.0
    memory = controller.control_memory
    proposal = controller.preview_baseline(command, state, vertical_feedforward_force_n=600.0)
    v, omega = state.base_velocity(local=True)
    next_distance = memory.distance_m + v[0] * plant.control_dt
    next_reference = np.clip(
        memory.distance_reference_m + 0.08 * plant.control_dt,
        next_distance - 0.55,
        next_distance + 0.55,
    )
    error = np.asarray(
        (
            next_distance - next_reference,
            state.base_rpy[1] - controller.linear_model.operating_state[1] - 0.01,
            v[0] - 0.08,
            omega[1],
        )
    )
    assert proposal.baseline.longitudinal_force_n == pytest.approx(
        -float((controller.gain @ error).item())
    )
    assert proposal.baseline.longitudinal_force_n != -123.0
    expected_support = (
        controller.low_level.total_mass_kg * controller.low_level.gravity_mps2
        + controller.low_level.height_kp * (0.46 - state.base_position[2])
        - controller.low_level.height_kd * state.base_linear_velocity_world[2]
    )
    assert proposal.baseline.support_vertical_force_n == pytest.approx(expected_support)
    assert proposal.baseline.vertical_feedforward_force_n == 500.0
    controller.compute(
        command, state, residual_force_n=np.asarray((7.0, 2.0)), vertical_feedforward_force_n=600.0
    )
    assert controller.last_longitudinal_force_n == np.clip(
        proposal.baseline.longitudinal_force_n + 7, -180, 180
    )
    assert controller.last_vertical_residual_n == 502.0


def test_mpc_preview_solves_once_and_does_not_shift_warmstart(plant, monkeypatch):
    calls = []
    original = hierarchical.minimize

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(hierarchical, "minimize", counted)
    controller = D1MPCVMCController(plant)
    state = D1MujocoTruthStateSource(plant).reset()
    initial = controller._solution.copy()
    command = D1Command(forward_velocity_mps=0.15)
    controller.preview_baseline(command, state)
    controller.preview_baseline(command, state)
    np.testing.assert_array_equal(controller._solution, initial)
    assert controller.last_iterations == 0 and controller.last_solve_ms == 0.0
    controller.compute(command, state)
    assert len(calls) == 1
    assert controller.last_iterations > 0
    assert controller.last_solve_ms > 0


@pytest.mark.parametrize("controller_type", (D1LQRVMCController, D1MPCVMCController))
@pytest.mark.parametrize("change", ("command", "state", "feedforward"))
def test_pending_proposal_rejects_changed_inputs(plant, controller_type, change):
    controller = controller_type(plant)
    state = D1MujocoTruthStateSource(plant).reset()
    command = D1Command()
    controller.preview_baseline(command, state)
    new_command = replace(command, forward_velocity_mps=0.1) if change == "command" else command
    new_state = replace(state) if change == "state" else state
    ff = 1.0 if change == "feedforward" else 0.0
    with pytest.raises(RuntimeError, match="stale"):
        controller.compute(new_command, new_state, vertical_feedforward_force_n=ff)


@pytest.mark.parametrize("controller_type", (D1LQRVMCController, D1MPCVMCController))
def test_real_short_trajectory_preview_path_matches_direct_compute_exactly(plant, controller_type):
    direct, preview = controller_type(plant), controller_type(plant)
    source = D1MujocoTruthStateSource(plant)
    state = source.reset()
    for index in range(40):
        command = D1Command(forward_velocity_mps=0.04 if index < 20 else -0.02)
        residual = np.asarray((2.0 * np.sin(index * 0.2), 1.0))
        expected = direct.compute(command, state, residual_force_n=residual)
        proposal = preview.preview_baseline(command, state)
        assert preview.preview_baseline(command, state) is proposal
        actual = preview.compute(command, state, residual_force_n=residual)
        np.testing.assert_array_equal(actual, expected)
        assert preview.control_memory == direct.control_memory
        assert preview.last_longitudinal_force_n == direct.last_longitudinal_force_n
        if controller_type is D1MPCVMCController:
            np.testing.assert_array_equal(preview._solution, direct._solution)
        plant.step(expected)
        state = source.read()
    assert state.control_time_s == pytest.approx(0.4)
