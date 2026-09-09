"""Action semantics and actual closed-loop checks through the v3 controller seam."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("mujoco")
from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.model import (
    JOINT_POSITION_HIGH,
    JOINT_POSITION_LOW,
    JOINT_TORQUE_LIMIT,
    D1Plant,
)
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource
from wheel_legged_control.d1.wheel_leg_controller import (
    D1WheelLegController,
    build_leg_extension_table,
)


@pytest.fixture(scope="module")
def controller():
    return D1WheelLegController(control_dt=0.01)


@pytest.fixture
def plant_state():
    plant = D1Plant(sampling_mode="synchronized")
    source = D1MujocoTruthStateSource(plant)
    return plant, source, source.reset()


def test_nominal_targets_use_world_height_minus_estimated_ground(controller, plant_state):
    _, _, state = plant_state
    flat = controller.nominal_targets(D1Command(), state, ground_height_m=0.0)
    raised = controller.nominal_targets(D1Command(base_height_m=0.655), state, ground_height_m=0.2)
    np.testing.assert_allclose(
        flat.nominal_joint_target_rad, raised.nominal_joint_target_rad, atol=1e-12
    )
    assert flat.nominal_joint_target_rad.shape == (16,)
    assert flat.nominal_wheel_speed_rad_s.shape == (4,)


@pytest.mark.parametrize("leg", range(4))
def test_leg_action_changes_only_its_geometric_target(controller, plant_state, leg):
    _, _, state = plant_state
    controller.compute(D1Command(), state, np.zeros(8))
    baseline = controller.last_result
    action = np.zeros(8)
    action[leg] = 0.5
    controller.compute(D1Command(), state, action)
    changed = controller.last_result
    np.testing.assert_allclose(
        changed.leg_extension_target_m - baseline.leg_extension_target_m,
        np.eye(4)[leg] * 0.02,
        atol=1e-12,
    )
    other = np.ones(16, dtype=bool)
    other[4 * leg : 4 * leg + 4] = False
    np.testing.assert_array_equal(changed.joint_target_rad[other], baseline.joint_target_rad[other])
    assert np.max(np.abs(changed.joint_target_rad - baseline.joint_target_rad)) > 0.01
    np.testing.assert_array_equal(
        changed.wheel_speed_target_rad_s, baseline.wheel_speed_target_rad_s
    )


def test_wheel_targets_include_command_differential_and_residual(controller, plant_state):
    _, _, state = plant_state
    command = D1Command(forward_velocity_mps=0.25, yaw_rate_rps=0.15)
    baseline = controller.nominal_targets(command, state)
    speed = baseline.nominal_wheel_speed_rad_s
    assert speed[0] < speed[1] and speed[2] < speed[3]
    action = np.zeros(8)
    action[6] = -0.5
    controller.compute(command, state, action)
    np.testing.assert_allclose(
        controller.last_result.wheel_speed_target_rad_s - speed, [0, 0, -2, 0], atol=1e-12
    )


@pytest.mark.parametrize(
    "action", [np.zeros(7), np.zeros(9), np.full(8, np.nan), np.full(8, np.inf)]
)
def test_invalid_actions_fail_before_torque(controller, plant_state, action):
    _, _, state = plant_state
    with pytest.raises(ValueError):
        controller.compute(D1Command(), state, action)


def test_control_depends_only_on_snapshot_not_live_plant(controller, plant_state):
    plant, _, state = plant_state
    action = np.linspace(-0.2, 0.2, 8)
    controller.reset()
    before = controller.compute(D1Command(), state, action)
    plant.data.qpos[:] = 100
    plant.data.qvel[:] = 100
    plant.model.body_mass[:] = 100
    controller.reset()
    after = controller.compute(D1Command(), state, action)
    np.testing.assert_array_equal(after, before)


def test_targets_and_torques_obey_bounds(controller, plant_state):
    _, _, state = plant_state
    torque = controller.compute(D1Command(), state, np.full(8, 3.0))
    report = controller.last_result
    assert np.all(np.abs(torque) <= JOINT_TORQUE_LIMIT)
    assert np.all(report.joint_target_rad >= JOINT_POSITION_LOW)
    assert np.all(report.joint_target_rad <= JOINT_POSITION_HIGH)
    np.testing.assert_array_equal(report.clipped_action, np.ones(8))
    assert np.all(np.abs(report.wheel_speed_target_rad_s) <= 30)


def test_actual_zero_action_stands_for_four_seconds(controller, plant_state):
    plant, source, state = plant_state
    controller.reset()
    history = []
    for _ in range(400):
        torque = controller.compute(D1Command(), state, np.zeros(8))
        plant.step(torque)
        state = source.read()
        assert not state.has_fallen()
        history.append((state.base_position[2] - 0.455, *state.base_rpy[:2]))
    history = np.asarray(history)
    assert np.sqrt(np.mean(history[-100:, 0] ** 2)) <= 0.025
    assert np.max(np.abs(history[:, 1:])) <= 0.1


def test_wheel_integral_is_public_single_update_and_preview_is_pure(controller, plant_state):
    _, _, state = plant_state
    controller.reset()
    before = controller.control_memory
    command = D1Command(forward_velocity_mps=0.1)
    targets = controller.nominal_targets(command, state)
    np.testing.assert_array_equal(
        controller.control_memory.wheel_integral_nm, before.wheel_integral_nm
    )
    controller.compute(command, state, np.zeros(8))
    expected = 3.0 * targets.nominal_wheel_speed_rad_s * 0.01
    np.testing.assert_allclose(controller.control_memory.wheel_integral_nm, expected, atol=1e-12)
    np.testing.assert_array_equal(
        controller.last_result.memory_before.wheel_integral_nm, np.zeros(4)
    )
    np.testing.assert_array_equal(controller.last_result.memory_after.wheel_integral_nm, expected)
    with pytest.raises(ValueError):
        controller.control_memory.wheel_integral_nm[0] = 1.0
    controller.reset()
    np.testing.assert_array_equal(controller.control_memory.wheel_integral_nm, np.zeros(4))


def test_wheel_integral_does_not_wind_up_during_outward_torque_saturation(controller, plant_state):
    _, _, state = plant_state
    controller.reset()
    velocity = state.joint_velocity.copy()
    velocity[[3, 7, 11, 15]] = -30.0
    state = replace(state, joint_velocity=velocity)
    torque = controller.compute(D1Command(forward_velocity_mps=1.0), state, np.zeros(8))
    np.testing.assert_array_equal(controller.control_memory.wheel_integral_nm, np.zeros(4))
    np.testing.assert_array_equal(torque[[3, 7, 11, 15]], np.full(4, 12.0))


def test_nominal_ik_precompute_restores_scratch_and_interpolation_matches_fk(plant_state):
    plant, source, initial = plant_state
    qpos, qvel = plant.simulation_state()
    grid, table, offsets, error = build_leg_extension_table(plant)
    np.testing.assert_array_equal(plant.data.qpos, qpos)
    np.testing.assert_array_equal(plant.data.qvel, qvel)
    assert error < 1e-5
    assert table.shape == (4, 161, 3)
    extension = 0.01337  # Deliberately not a tabulated grid point.
    q = initial.joint_position.copy()
    for leg in range(4):
        q[4 * leg : 4 * leg + 3] = [
            np.interp(extension, grid, table[leg, :, joint]) for joint in range(3)
        ]
    plant.reset(joint_position=q)
    state = source.reset()
    observed = state.foot_offset_world @ state.base_rotation
    requested = offsets.copy()
    requested[:, 2] -= extension
    np.testing.assert_allclose(observed, requested, rtol=0, atol=1e-5)


@pytest.mark.parametrize("direction", (-1, 1))
def test_actual_zero_residual_turn_reaches_seventy_percent_commanded_progress(
    controller, plant_state, direction
):
    plant, source, state = plant_state
    controller.reset()
    integrated_command = 0.0
    initial_yaw = float(state.base_rpy[2])
    yaw_errors = []
    for step in range(800):
        ramp = float(np.clip((step * 0.01 - 1.0) / 0.5, 0.0, 1.0))
        command = D1Command(forward_velocity_mps=0.15 * ramp, yaw_rate_rps=direction * 0.15 * ramp)
        integrated_command += command.yaw_rate_rps * 0.01
        plant.step(controller.compute(command, state, np.zeros(8)))
        state = source.read()
        assert not state.has_fallen()
        yaw_errors.append(float(state.base_angular_velocity_body[2] - command.yaw_rate_rps))
    progress = (float(state.base_rpy[2]) - initial_yaw) / integrated_command
    assert progress >= 0.70, {"turn_progress_fraction": progress}
    assert np.sqrt(np.mean(np.square(yaw_errors[-100:]))) <= 0.05


def test_yaw_feedback_is_pure_bounded_and_changes_only_wheel_targets(plant_state):
    _, _, state = plant_state
    command = D1Command(forward_velocity_mps=0.15, yaw_rate_rps=0.15)
    baseline = D1WheelLegController(yaw_feedback_gain=0)
    feedback = D1WheelLegController(yaw_feedback_gain=4)
    angular_body = np.asarray((0.0, 0.0, -0.2))
    state = replace(
        state,
        base_angular_velocity_body=angular_body,
        base_angular_velocity_world=state.base_rotation @ angular_body,
    )
    nominal = baseline.nominal_targets(command, state)
    adjusted = feedback.nominal_targets(command, state)
    assert adjusted.effective_yaw_request_rps == 0.6
    np.testing.assert_array_equal(
        adjusted.nominal_joint_target_rad, nominal.nominal_joint_target_rad
    )
    assert adjusted.nominal_wheel_speed_rad_s[0] < nominal.nominal_wheel_speed_rad_s[0]
    assert adjusted.nominal_wheel_speed_rad_s[1] > nominal.nominal_wheel_speed_rad_s[1]
    np.testing.assert_array_equal(feedback.control_memory.wheel_integral_nm, np.zeros(4))
    assert feedback.control_memory.yaw_integral_nm is None


@pytest.mark.parametrize("gain", (-1.0, np.nan, np.inf))
def test_invalid_yaw_feedback_gain_rejected(gain):
    with pytest.raises(ValueError):
        D1WheelLegController(yaw_feedback_gain=gain)
