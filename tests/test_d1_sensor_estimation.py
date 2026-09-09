from __future__ import annotations

from dataclasses import fields, replace

import mujoco
import numpy as np
import pytest

from wheel_legged_control.d1.controllers import D1Command, D1VMCController
from wheel_legged_control.d1.model import NOMINAL_JOINT_POSITION, D1Plant
from wheel_legged_control.d1.sensor_estimation import (
    D1MujocoSensorSource,
    D1ProprioceptiveEstimator,
    D1SensorMeasurements,
    D1SensorNoise,
    D1SensorStateSource,
)
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource


def _measurement(sequence=0, **overrides):
    values = {
        "sequence": sequence,
        "time_s": sequence * 0.01,
        "gyro_rad_s": np.zeros(3),
        "specific_force_m_s2": np.asarray((0.0, 0.0, 9.81)),
        "joint_position_rad": NOMINAL_JOINT_POSITION,
        "joint_velocity_rad_s": np.zeros(16),
        "wheel_contact": np.ones(4, dtype=bool),
    }
    values.update(overrides)
    return D1SensorMeasurements(**values)


def test_measurements_are_immutable_and_cannot_contain_base_truth():
    sample = _measurement()
    assert {field.name for field in fields(sample)} == {
        "sequence",
        "time_s",
        "gyro_rad_s",
        "specific_force_m_s2",
        "joint_position_rad",
        "joint_velocity_rad_s",
        "wheel_contact",
    }
    for name in (
        "gyro_rad_s",
        "specific_force_m_s2",
        "joint_position_rad",
        "joint_velocity_rad_s",
        "wheel_contact",
    ):
        with pytest.raises(ValueError):
            getattr(sample, name).setflags(write=True)
    with pytest.raises(ValueError, match="finite"):
        _measurement(gyro_rad_s=[np.nan, 0, 0])
    with pytest.raises(ValueError, match="shape"):
        _measurement(joint_position_rad=[0])


def test_mujoco_object_acceleration_is_specific_force_not_acceleration_plus_g():
    xml = '<mujoco><worldbody><body name="base"><freejoint/><geom type="sphere" size=".1" mass="1"/><site name="imu"/></body></worldbody><sensor><accelerometer site="imu"/></sensor></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for support, expected in ((0.0, 0.0), (9.81, 9.81)):
        data.xfrc_applied[1, 2] = support
        mujoco.mj_forward(model, data)
        mujoco.mj_rnePostConstraint(model, data)
        acceleration = np.zeros(6)
        mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_BODY, 1, acceleration, 0)
        np.testing.assert_allclose(acceleration[3:], [0, 0, expected], atol=1e-12)
        np.testing.assert_allclose(acceleration[3:], data.sensordata, atol=1e-12)


def test_com_imu_axes_match_real_site_sensor_with_displaced_rotated_inertia():
    xml = '<mujoco><worldbody><body name="base" quat=".97 .1 -.1 .15"><freejoint/><inertial pos=".03 .001 -.015" quat=".9 .1 .3 -.2" mass="2" diaginertia=".1 .2 .25"/><geom type="sphere" size=".1"/><site name="imu" pos=".03 .001 -.015"/></body></worldbody><sensor><accelerometer site="imu"/><gyro site="imu"/></sensor></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qvel[:] = [0.4, -0.2, 0.1, 0.3, 0.5, -0.2]
    data.xfrc_applied[1] = [2, -1, 19.62, 0.2, 0.1, -0.1]
    mujoco.mj_forward(model, data)
    mujoco.mj_rnePostConstraint(model, data)
    acceleration, velocity = np.zeros(6), np.zeros(6)
    mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_BODY, 1, acceleration, 0)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, 1, velocity, 0)
    rotation = data.xmat[1].reshape(3, 3)
    np.testing.assert_allclose(rotation.T @ acceleration[3:], data.sensordata[:3], atol=1e-10)
    np.testing.assert_allclose(rotation.T @ velocity[:3], data.sensordata[3:], atol=1e-10)


def test_sensor_collector_freefall_and_visible_body_axes():
    plant = D1Plant()
    quaternion = np.empty(4)
    mujoco.mju_euler2Quat(quaternion, np.asarray((0.2, -0.1, 0.4)), "xyz")
    plant.reset(base_position=np.asarray((0.0, 0.0, 2.0)), base_quaternion=quaternion)
    qpos, qvel = plant.simulation_state()
    qvel[3:6] = [0.2, -0.1, 0.3]
    plant.set_simulation_state(qpos, qvel)
    measurement = D1MujocoSensorSource(plant).reset()
    rotation = plant.data.xmat[plant.base_body_id].reshape(3, 3)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        plant.model, plant.data, mujoco.mjtObj.mjOBJ_BODY, plant.base_body_id, velocity, 0
    )
    np.testing.assert_allclose(measurement.gyro_rad_s, rotation.T @ velocity[:3], atol=1e-12)
    assert not measurement.wheel_contact.any()
    # Internal centripetal accelerations are possible when the articulated body
    # rotates. Check zero-angular-rate free fall separately.
    plant.reset(base_position=np.asarray((0.0, 0.0, 2.0)))
    np.testing.assert_allclose(
        D1MujocoSensorSource(plant).reset().specific_force_m_s2, 0, atol=1e-10
    )


def test_noisy_measurements_reseed_and_bias_is_measurement_only():
    plant = D1Plant()
    noise = D1SensorNoise(
        gyro_std_rad_s=0.01,
        accelerometer_std_m_s2=0.03,
        encoder_position_std_rad=0.001,
        gyro_bias_rad_s=(0.0, 0.0, 0.02),
    )
    source = D1MujocoSensorSource(plant, noise=noise)
    first, second = source.reset(seed=7), source.reset(seed=7)
    for name in ("gyro_rad_s", "specific_force_m_s2", "joint_position_rad"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert not np.array_equal(first.gyro_rad_s, source.reset(seed=8).gyro_rad_s)
    biased = D1MujocoSensorSource(plant, noise=D1SensorNoise(gyro_bias_rad_s=(0, 0, 0.02))).reset()
    np.testing.assert_allclose(biased.gyro_rad_s, (0, 0, 0.02), atol=1e-12)


def test_loaded_contact_margin_is_not_misreported_as_airborne():
    plant = D1Plant()
    source = D1MujocoTruthStateSource(plant)
    controller = D1VMCController(plant)
    state = source.reset()
    for _ in range(100):
        plant.step(controller.compute(D1Command(), state))
        state = source.read()
    assert all(contact.dist > 0 for contact in plant.data.contact)
    assert D1MujocoSensorSource(plant).reset().wheel_contact.all()


def test_stationary_kinematics_and_ground_plane_from_encoders():
    estimator = D1ProprioceptiveEstimator()
    initial = estimator.reset(_measurement())
    state = estimator.update(_measurement(1))
    truth = D1MujocoTruthStateSource(D1Plant()).reset()
    np.testing.assert_allclose(state.base_rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(state.base_linear_velocity_world, 0, atol=1e-12)
    np.testing.assert_allclose(state.foot_position, truth.foot_position, atol=1e-12)
    np.testing.assert_allclose(state.foot_jacobian, truth.foot_jacobian, atol=1e-12)
    # Nominal placement is 2.6 mm above contact; synthetic contact flags are an
    # explicit fixture, so the fitted plane is above z=0 by that known gap.
    reference = estimator.ground_reference(state)
    assert reference.height_m == pytest.approx(
        np.mean(initial.foot_position[:, 2]) - 0.087, abs=1e-6
    )
    assert estimator.diagnostics["support_plane_observable"]


def test_tilted_support_plane_comes_from_estimated_pose_and_fk_not_map():
    estimator = D1ProprioceptiveEstimator()
    state = estimator.reset(_measurement(), initial_rpy=(0.0, -0.07, 0.0))
    ground = estimator.ground_reference(state)
    assert ground.pitch_rad == pytest.approx(-0.07, abs=2e-6)
    assert ground.roll_rad == pytest.approx(0.0, abs=2e-6)
    np.testing.assert_allclose(
        state.wheel_contact_normal[0], [-np.sin(0.07), 0, np.cos(0.07)], atol=2e-6
    )


def test_rolling_odometry_includes_com_to_origin_and_contact_lever_arms():
    estimator = D1ProprioceptiveEstimator(attitude_time_constant_s=1e12)
    contact = np.asarray((True, False, False, False))
    estimator.reset(_measurement(wheel_contact=contact))
    omega = np.asarray((0.0, 0.2, 0.0))
    sample = _measurement(1, gyro_rad_s=omega, wheel_contact=contact)
    state = estimator.update(sample)
    rotation = state.base_rotation
    world_omega = rotation @ omega
    com_offset = rotation @ estimator.model.body_ipos[estimator._base]
    arm = state.wheel_contact_point[0] - state.base_position
    expected_odometry = -np.cross(world_omega, arm - com_offset)
    expected_inertial = (rotation @ sample.specific_force_m_s2 + [0, 0, -9.81]) * 0.01
    alpha = 0.01 / (0.02 + 0.01)
    np.testing.assert_allclose(
        state.base_linear_velocity_world,
        expected_inertial * (1 - alpha) + expected_odometry * alpha,
        atol=1e-12,
    )
    assert np.linalg.norm(np.cross(world_omega, com_offset)) > 0.005
    origin_without_com = -np.cross(world_omega, arm)
    assert np.linalg.norm(expected_odometry - origin_without_com) > 0.005


def test_wheel_spin_has_forward_rolling_velocity():
    estimator = D1ProprioceptiveEstimator()
    estimator.reset(_measurement())
    velocity = np.zeros(16)
    velocity[[3, 7, 11, 15]] = 1.0
    state = estimator.update(_measurement(1, joint_velocity_rad_s=velocity))
    assert state.base_linear_velocity_world[0] == pytest.approx(0.087 / 3.0, abs=1e-5)


def test_accelerometer_corrects_tilt_but_does_not_observe_yaw():
    estimator = D1ProprioceptiveEstimator(attitude_time_constant_s=0.3)
    state = estimator.reset(_measurement(), initial_rpy=(0.08, -0.06, 0.4))
    initial = np.linalg.norm(state.base_rpy[:2])
    for i in range(1, 151):
        state = estimator.update(_measurement(i))
    assert np.linalg.norm(state.base_rpy[:2]) < initial * 0.02
    assert state.base_rpy[2] == pytest.approx(0.4, abs=0.004)


def test_no_contact_is_inertial_dead_reckoning_not_truth_fallback():
    estimator = D1ProprioceptiveEstimator()
    estimator.reset(
        _measurement(wheel_contact=np.zeros(4, dtype=bool), specific_force_m_s2=np.zeros(3)),
        initial_position=(1, 2, 3),
    )
    state = estimator.update(
        _measurement(1, wheel_contact=np.zeros(4, dtype=bool), specific_force_m_s2=np.zeros(3))
    )
    assert state.base_linear_velocity_world[2] == pytest.approx(-0.0981)
    assert state.base_position[2] == pytest.approx(3 - 0.000981)
    assert estimator.diagnostics["support_plane_status"] == "no_contact_dead_reckoning"
    assert not state.wheel_contact_point.any()


def test_recorded_measurements_replay_identically_after_live_truth_is_destroyed(monkeypatch):
    plant = D1Plant()
    measurements = D1MujocoSensorSource(plant)
    truth = D1MujocoTruthStateSource(plant)
    controller = D1VMCController(plant)
    packets, states = [measurements.reset()], []
    state = truth.reset()
    estimator = D1ProprioceptiveEstimator()
    states.append(estimator.reset(packets[0]))
    for _ in range(30):
        plant.step(controller.compute(D1Command(), state))
        state = truth.read()
        packets.append(measurements.read())
        states.append(estimator.update(packets[-1]))
    plant.reset(base_position=np.asarray((99.0, -99.0, 50.0)))

    def forbidden(*args, **kwargs):
        raise AssertionError("live state access in fusion")

    monkeypatch.setattr(D1Plant, "base_position", property(forbidden))
    monkeypatch.setattr(D1Plant, "base_velocity", forbidden)
    monkeypatch.setattr(D1Plant, "training_ground_reference", forbidden)
    monkeypatch.setattr(mujoco, "mj_objectVelocity", forbidden)
    monkeypatch.setattr(mujoco, "mj_objectAcceleration", forbidden)
    replay = D1ProprioceptiveEstimator()
    assert replay.model is not plant.model
    assert replay.data is not plant.data
    replayed = [replay.reset(packets[0])]
    replayed.extend(replay.update(packet) for packet in packets[1:])
    for expected, actual in zip(states, replayed, strict=True):
        for field in fields(actual):
            np.testing.assert_array_equal(
                getattr(actual, field.name), getattr(expected, field.name)
            )


def test_source_initialization_is_explicit_prior_not_live_spawn():
    plant = D1Plant()
    plant.reset(base_position=np.asarray((4.0, -2.0, 1.0)))
    source = D1SensorStateSource(plant, initial_position=(1.0, 0.0, 0.46))
    state = source.reset(seed=4)
    np.testing.assert_array_equal(state.base_position, [1.0, 0.0, 0.46])
    np.testing.assert_array_equal(state.base_linear_velocity_world, 0)


def test_measurement_delay_holds_initial_packet_without_hidden_warmup():
    plant = D1Plant()
    source = D1SensorStateSource(plant, delay_steps=2)
    initial = source.reset(seed=7)
    assert plant.data.time == 0.0
    outputs = []
    for _ in range(3):
        plant.step(np.zeros(16))
        outputs.append(source.read())
    assert [state.sequence for state in outputs] == [1, 2, 3]
    assert [state.measurement_time_s for state in outputs] == pytest.approx([0, 0, 0.01])
    assert [state.age_s for state in outputs] == pytest.approx([0.01, 0.02, 0.02])
    np.testing.assert_array_equal(outputs[0].base_position, initial.base_position)
    np.testing.assert_array_equal(outputs[1].base_position, initial.base_position)
    assert source.latest_measurement.time_s == pytest.approx(0.03)
    assert outputs[-1].base_position[2] != initial.base_position[2]


@pytest.mark.parametrize("delay", [-1, 0.5, True])
def test_invalid_measurement_delay_rejected(delay):
    with pytest.raises(ValueError, match="delay_steps"):
        D1SensorStateSource(D1Plant(), delay_steps=delay)


def test_delayed_actual_closed_loop_has_exact_measurement_replay():
    from scripts.evaluate_d1_sensor_estimation import run_case

    summary, rows, packets = run_case(
        mode="closed_loop",
        noise_name="noise",
        terrain_name="flat",
        seed=17,
        duration_s=0.2,
        delay_steps=2,
    )
    assert summary["delay_steps"] == 2
    assert summary["replay_bitwise_equal"]
    assert rows[-1]["measurement_age_s"] == pytest.approx(0.02)
    assert packets["time_s"][-1] == pytest.approx(0.2)


@pytest.mark.parametrize("time,sequence", [(0.0, 1), (0.11, 1), (0.01, 0)])
def test_nonincreasing_or_stale_measurements_rejected(time, sequence):
    estimator = D1ProprioceptiveEstimator()
    estimator.reset(_measurement())
    with pytest.raises(ValueError, match="increasing"):
        estimator.update(replace(_measurement(), time_s=time, sequence=sequence))


def test_actual_closed_loop_observation_and_controller_share_estimated_state(monkeypatch):
    from scripts import evaluate_d1_sensor_estimation as experiment

    observed_states = []
    controlled_states = []
    encode = experiment.encode_d1_observation
    compute = experiment.D1LQRVMCController.compute

    def observe(**kwargs):
        observed_states.append(kwargs["state"])
        return encode(**kwargs)

    def control(self, command, state, *args, **kwargs):
        controlled_states.append(state)
        return compute(self, command, state, *args, **kwargs)

    monkeypatch.setattr(experiment, "encode_d1_observation", observe)
    monkeypatch.setattr(experiment.D1LQRVMCController, "compute", control)
    summary, rows, _ = experiment.run_case(
        mode="closed_loop", noise_name="bias", terrain_name="flat", seed=17, duration_s=0.1
    )
    assert summary["replay_bitwise_equal"]
    assert summary["steps"] == 10
    assert len(observed_states) == len(controlled_states) == 10
    assert all(
        observed is controlled
        for observed, controlled in zip(observed_states, controlled_states, strict=True)
    )
    # A known uncalibrated gyro bias is present in the actual feedback object.
    assert controlled_states[0].base_angular_velocity_body[2] != 0.0
    assert any(abs(row["estimate_pitch_rad"] - row["truth_pitch_rad"]) > 1e-6 for row in rows)
