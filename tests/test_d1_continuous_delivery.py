"""Delivery checks for the sensor-command45 route; no training is performed."""

from dataclasses import replace
from functools import partial

import numpy as np

from wheel_legged_control.d1.continuous_task import ContinuousTaskConfig, D1ContinuousTask
from wheel_legged_control.d1.sensor_estimation import D1SensorNoise, D1SensorStateSource


def test_sensor_step_observation_and_torque_ignore_evaluation_truth(monkeypatch):
    """Change evaluation truth, holding sensors and the physical trajectory fixed.

    This exercises actual MuJoCo sensor acquisition, estimator updates, command45
    observations, and LQR/VMC+PI torque computation. It does not test estimator
    accuracy or claim that reward/termination should ignore truth.
    """
    task = D1ContinuousTask(
        ContinuousTaskConfig(ground_reference_mode="estimated", observation_layout="command45"),
        state_source_factory=partial(
            D1SensorStateSource,
            noise=D1SensorNoise(
                gyro_std_rad_s=0.002,
                accelerometer_std_m_s2=0.03,
                encoder_position_std_rad=0.0005,
                encoder_velocity_std_rad_s=0.005,
            ),
            initial_position=(-3.8, 0, 0.455),
            initial_rpy=(0, 0, 0),
        ),
    )
    original_step = task.plant.step
    torques = []

    def record_step(torque):
        torques.append(torque.copy())
        original_step(torque)

    monkeypatch.setattr(task.plant, "step", record_step)
    original_truth_read = task._truth.read
    original_truth_ground = task._truth_ground

    def poison_truth():
        truth = original_truth_read()
        # Keep the poisoned state's body/world velocity contract consistent.
        # Only evaluation consumes this state, so control must stay unchanged.
        return replace(
            truth,
            base_position=truth.base_position + (0, 0, 0.03),
            base_linear_velocity_body=truth.base_linear_velocity_body + (0.3, 0, 0),
            base_linear_velocity_world=truth.base_linear_velocity_world
            + truth.base_rotation @ np.asarray((0.3, 0, 0)),
        )

    def poison_ground(x, y):
        return replace(original_truth_ground(x, y), height_m=0.02)

    def rollout():
        obs, _ = task.reset(seed=17)
        observations, rewards, telemetry = [obs], [], []
        torques.clear()
        for index in range(8):
            action = np.asarray((0.1 * index, -0.2), dtype=np.float32)
            obs, reward, terminated, truncated, info = task.step(action)
            assert not terminated and not truncated
            observations.append(obs)
            rewards.append(reward)
            telemetry.append(info)
        return np.asarray(observations), np.asarray(torques), np.asarray(rewards), telemetry

    nominal = rollout()
    monkeypatch.setattr(task._truth, "read", poison_truth)
    monkeypatch.setattr(task, "_truth_ground", poison_ground)
    poisoned = rollout()
    np.testing.assert_array_equal(nominal[0], poisoned[0])
    np.testing.assert_array_equal(nominal[1], poisoned[1])
    assert not np.array_equal(nominal[2], poisoned[2])
    assert nominal[3][0]["z_m"] != poisoned[3][0]["z_m"]
    assert nominal[3][0]["ground_height_m"] != poisoned[3][0]["ground_height_m"]
    assert task.source.estimator.data is not task.plant.data
