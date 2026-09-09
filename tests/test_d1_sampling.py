"""Physical sampling phase checks for the D1 control loop."""

import mujoco
import numpy as np
import pytest

from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.sensor_estimation import D1MujocoSensorSource, D1SensorStateSource
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource


def test_control_sample_origin_matches_generalized_position():
    plant = D1Plant(sampling_mode="synchronized")
    plant.step(np.zeros(16), push_force_world_n=np.asarray((25.0, 0.0, 0.0)))
    # Free-base translation is the visible body origin, not its displaced COM.
    np.testing.assert_allclose(plant.base_position, plant.data.qpos[:3], atol=1e-12, rtol=0)


def test_legacy_mode_explicitly_preserves_historical_phase():
    plant = D1Plant(sampling_mode="legacy_mixed")
    plant.step(np.zeros(16))
    assert plant.measurement_data is plant.data
    assert np.max(np.abs(plant.base_position - plant.data.qpos[:3])) > 1e-5
    np.testing.assert_allclose(
        plant.base_position, plant.data.qpos[:3] - 0.002 * plant.data.qvel[:3], atol=1e-12
    )


def test_synchronized_sensor_and_truth_share_post_integration_pose_and_time():
    plant = D1Plant(sampling_mode="synchronized")
    truth = D1MujocoTruthStateSource(plant)
    sensor = D1MujocoSensorSource(plant)
    truth.reset()
    sensor.reset()
    for _ in range(5):
        plant.step(np.zeros(16), push_torque_world_nm=np.asarray((0.0, 0.5, 0.1)))
        state, packet = truth.read(), sensor.read()
        assert state.control_time_s == packet.time_s == plant.data.time
        np.testing.assert_array_equal(state.joint_position, plant.data.qpos[plant.qpos_addresses])
        np.testing.assert_array_equal(packet.joint_position_rad, state.joint_position)
        np.testing.assert_array_equal(packet.joint_velocity_rad_s, state.joint_velocity)
        np.testing.assert_array_equal(packet.gyro_rad_s, state.base_angular_velocity_body)
        np.testing.assert_array_equal(state.base_position, plant.data.qpos[:3])
        expected_rotation = np.zeros(9)
        mujoco.mju_quat2Mat(expected_rotation, plant.data.qpos[3:7])
        np.testing.assert_allclose(state.base_rotation, expected_rotation.reshape(3, 3), atol=1e-15)


@pytest.mark.parametrize("measure_contacts", [False, True])
def test_sampling_and_sensor_reads_do_not_change_physics_trajectory(measure_contacts):
    legacy, sampled = D1Plant(), D1Plant(sampling_mode="synchronized")
    source = D1MujocoSensorSource(sampled)
    source.reset()
    rng = np.random.default_rng(1729)
    for _ in range(20):
        torque = rng.uniform(-1, 1, 16)
        push = rng.uniform(-10, 10, 3)
        for plant in (legacy, sampled):
            # Moments only compare at the same reference point. The two modes
            # deliberately publish different legacy/synchronized body origins.
            plant.step(
                torque,
                push_force_world_n=push,
                measure_contact_wrench=measure_contacts,
                contact_wrench_reference_world_m=np.asarray((0.0, 0.0, 0.455)),
            )
        warmstart = sampled.data.qacc_warmstart.copy()
        source.read()
        source.read()  # The new provider will prevent this extra sampling too.
        np.testing.assert_array_equal(sampled.data.qacc_warmstart, warmstart)
        np.testing.assert_array_equal(sampled.data.qpos, legacy.data.qpos)
        np.testing.assert_array_equal(sampled.data.qvel, legacy.data.qvel)
        np.testing.assert_array_equal(sampled.data.qacc_warmstart, legacy.data.qacc_warmstart)
        np.testing.assert_array_equal(
            sampled.last_control_interval_contact_wrench.wrench_world,
            legacy.last_control_interval_contact_wrench.wrench_world,
        )


def test_sensor_sample_retains_applied_push_when_live_buffer_is_cleared():
    plant = D1Plant(sampling_mode="synchronized")
    plant.reset(base_position=np.asarray((0.0, 0.0, 2.0)))
    force, moment = np.asarray((30.0, 0.0, 0.0)), np.asarray((0.0, 1.0, 0.0))
    plant.step(np.zeros(16), push_force_world_n=force, push_torque_world_nm=moment)
    np.testing.assert_array_equal(plant.data.xfrc_applied, 0)
    np.testing.assert_array_equal(
        plant.measurement_data.xfrc_applied[plant.base_body_id], np.r_[force, moment]
    )
    with_push = D1MujocoSensorSource(plant).reset().specific_force_m_s2
    reference = mujoco.MjData(plant.model)
    mujoco.mj_copyData(reference, plant.model, plant.measurement_data)
    reference.xfrc_applied[:] = 0
    mujoco.mj_forward(plant.model, reference)
    mujoco.mj_rnePostConstraint(plant.model, reference)
    acceleration = np.zeros(6)
    mujoco.mj_objectAcceleration(
        plant.model, reference, mujoco.mjtObj.mjOBJ_BODY, plant.base_body_id, acceleration, 0
    )
    without_push = reference.xmat[plant.base_body_id].reshape(3, 3).T @ acceleration[3:]
    assert np.linalg.norm(with_push - without_push) > 0.1


def test_origin_velocity_matches_position_derivative_with_rotated_com_offset():
    plant = D1Plant(sampling_mode="synchronized")
    plant.data.qpos[3:7] = (0.9, 0.2, -0.1, 0.3)
    plant.data.qpos[3:7] /= np.linalg.norm(plant.data.qpos[3:7])
    plant.data.qvel[:6] = (0.2, -0.1, 0.3, 0.7, -0.4, 0.5)
    plant.refresh_measurements()
    state = D1MujocoTruthStateSource(plant).reset()
    offset = plant.model.body_ipos[plant.base_body_id]
    reference = mujoco.MjData(plant.model)
    mujoco.mj_copyData(reference, plant.model, plant.measurement_data)
    dt = 1e-6
    mujoco.mj_integratePos(plant.model, reference.qpos, reference.qvel, dt)
    mujoco.mj_kinematics(plant.model, reference)
    derivative = (reference.xpos[plant.base_body_id] - state.base_position) / dt
    np.testing.assert_allclose(plant.base_origin_velocity(), derivative, atol=1e-9)
    np.testing.assert_allclose(state.base_origin_velocity(offset), derivative, atol=1e-9)
    np.testing.assert_allclose(
        state.base_origin_velocity(offset, local=True),
        state.base_rotation.T @ derivative,
        atol=1e-9,
    )
    assert np.linalg.norm(state.base_linear_velocity_world - derivative) > 0.005


def test_reset_and_state_restore_refresh_measurement_copy():
    plant = D1Plant(sampling_mode="synchronized")
    plant.step(np.zeros(16))
    qpos, qvel = plant.simulation_state()
    qpos[:3] = (1, -1, 0.6)
    plant.set_simulation_state(qpos, qvel)
    np.testing.assert_array_equal(plant.base_position, qpos[:3])
    plant.reset()
    assert plant.measurement_data.time == 0
    np.testing.assert_array_equal(plant.base_position, (0, 0, 0.455))


def test_invalid_sampling_mode_is_rejected_before_model_compilation():
    with pytest.raises(ValueError, match="sampling_mode"):
        D1Plant(sampling_mode="estimated")


@pytest.mark.parametrize("source_class", [D1MujocoTruthStateSource, D1SensorStateSource])
def test_wheel_center_position_and_jacobian_use_the_same_reference(source_class):
    plant = D1Plant(sampling_mode="synchronized")
    state = source_class(plant).reset()
    # Rotating a circular wheel about its axle does not translate its center.
    np.testing.assert_allclose(state.foot_jacobian[:, :, 3], 0, atol=1e-12)
    data = mujoco.MjData(plant.model)
    epsilon = 1e-7
    for leg, body in enumerate(plant.wheel_body_ids_by_leg):
        for column in range(3):
            mujoco.mj_copyData(data, plant.model, plant.measurement_data)
            data.qpos[plant.qpos_addresses[4 * leg + column]] += epsilon
            mujoco.mj_kinematics(plant.model, data)
            derivative = (data.xpos[body] - state.foot_position[leg]) / epsilon
            np.testing.assert_allclose(state.foot_jacobian[leg, :, column], derivative, atol=3e-8)
