"""IMU/encoder measurements and a replayable D1 proprioceptive estimator.

Only ``D1MujocoSensorSource`` has access to the live simulator. The estimator
owns a separate nominal model and MjData for forward kinematics. It assumes
binary wheel-contact switches, circular wheels, and approximately rolling
contacts. It is a complementary observer, not a slip-aware EKF.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from numbers import Integral

import mujoco
import numpy as np

from .model import D1_JOINT_NAMES, LEG_PREFIXES, D1Plant, build_d1_model
from .state_estimation import D1StateEstimate, _immutable_array
from .training_terrain import TrainingGroundReference


@dataclass(frozen=True, slots=True)
class D1SensorMeasurements:
    """IMU at the base COM, with axes aligned to visible ``base_link``.

    Specific force is acceleration minus gravity: zero in free fall and +g
    along body z when upright and supported. Encoder velocities are motor-side
    speed measurements (not base velocity). Contact switches report only bools.
    There is deliberately no base pose, base velocity, or terrain measurement.
    """

    sequence: int
    time_s: float
    gyro_rad_s: np.ndarray
    specific_force_m_s2: np.ndarray
    joint_position_rad: np.ndarray
    joint_velocity_rad_s: np.ndarray
    wheel_contact: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        if not np.isfinite(self.time_s):
            raise ValueError("time_s must be finite")
        for name, shape, dtype in (
            ("gyro_rad_s", (3,), np.float64),
            ("specific_force_m_s2", (3,), np.float64),
            ("joint_position_rad", (16,), np.float64),
            ("joint_velocity_rad_s", (16,), np.float64),
            ("wheel_contact", (4,), np.bool_),
        ):
            object.__setattr__(
                self, name, _immutable_array(getattr(self, name), shape=shape, dtype=dtype)
            )


@dataclass(frozen=True, slots=True)
class D1SensorNoise:
    """Per-sample standard deviations and constant, uncalibrated IMU biases."""

    gyro_std_rad_s: float = 0.0
    accelerometer_std_m_s2: float = 0.0
    encoder_position_std_rad: float = 0.0
    encoder_velocity_std_rad_s: float = 0.0
    gyro_bias_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    accelerometer_bias_m_s2: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name in (
            "gyro_std_rad_s",
            "accelerometer_std_m_s2",
            "encoder_position_std_rad",
            "encoder_velocity_std_rad_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("gyro_bias_rad_s", "accelerometer_bias_m_s2"):
            value = _immutable_array(getattr(self, name), shape=(3,))
            object.__setattr__(self, name, tuple(float(x) for x in value))


class D1MujocoSensorSource:
    """Simulate allowed sensors; never publish pose, linear velocity or normals.

    MuJoCo's post-constraint object acceleration is already specific force.
    Calling mj_rnePostConstraint is necessary without XML force/IMU sensors.
    Four ideal load switches use summed normal load > 1 N. Neither normal
    directions nor force magnitudes leave this collector. MuJoCo's contact
    margin can transmit load at positive distance, so distance is not a switch.
    """

    def __init__(
        self, plant: D1Plant, *, noise: D1SensorNoise | None = None, seed: int | None = None
    ) -> None:
        self.plant = plant
        self.noise = D1SensorNoise() if noise is None else noise
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._sequence = -1
        self._wheel_index = {body: i for i, body in enumerate(plant.wheel_body_ids_by_leg)}

    def reset(self, *, seed: int | None = None) -> D1SensorMeasurements:
        self._rng = np.random.default_rng(self._seed if seed is None else seed)
        self._sequence = 0
        return self._capture()

    def read(self) -> D1SensorMeasurements:
        if self._sequence < 0:
            raise RuntimeError("reset must precede read")
        self._sequence += 1
        return self._capture()

    def _capture(self) -> D1SensorMeasurements:
        plant, noise = self.plant, self.noise
        data = plant.measurement_data
        rotation = data.xmat[plant.base_body_id].reshape(3, 3)
        velocity, acceleration = np.zeros(6), np.zeros(6)
        mujoco.mj_objectVelocity(
            plant.model, data, mujoco.mjtObj.mjOBJ_BODY, plant.base_body_id, velocity, 0
        )
        mujoco.mj_rnePostConstraint(plant.model, data)
        mujoco.mj_objectAcceleration(
            plant.model, data, mujoco.mjtObj.mjOBJ_BODY, plant.base_body_id, acceleration, 0
        )
        contact_load = np.zeros(4)
        for contact_index, contact in enumerate(data.contact):
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            other = (
                geom2
                if geom1 in plant.terrain_geom_ids
                else geom1
                if geom2 in plant.terrain_geom_ids
                else None
            )
            if other is not None:
                index = self._wheel_index.get(int(plant.model.geom_bodyid[other]))
                if index is not None:
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(plant.model, data, contact_index, wrench)
                    contact_load[index] += max(0.0, float(wrench[0]))
        return D1SensorMeasurements(
            sequence=self._sequence,
            time_s=float(data.time),
            gyro_rad_s=rotation.T @ velocity[:3]
            + noise.gyro_bias_rad_s
            + self._rng.normal(0, noise.gyro_std_rad_s, 3),
            specific_force_m_s2=rotation.T @ acceleration[3:]
            + noise.accelerometer_bias_m_s2
            + self._rng.normal(0, noise.accelerometer_std_m_s2, 3),
            joint_position_rad=plant.joint_position
            + self._rng.normal(0, noise.encoder_position_std_rad, 16),
            joint_velocity_rad_s=plant.joint_velocity
            + self._rng.normal(0, noise.encoder_velocity_std_rad_s, 16),
            wheel_contact=contact_load > 1.0,
        )


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = (
        np.cos(roll),
        np.sin(roll),
        np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw),
        np.sin(yaw),
    )
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
    )


class D1ProprioceptiveEstimator:
    """Complementary IMU attitude and inertial/rolling-constraint velocity.

    Initialization is an explicit known placement prior, not a simulator query.
    There is no absolute position/yaw correction and no automatic bias estimate.
    A three-or-more-contact wheel-center plane supplies a local ground estimate;
    without sufficient non-collinear support it retains its last world normal.
    """

    schema = "d1-imu-encoder-complementary-v1"

    def __init__(
        self,
        *,
        attitude_time_constant_s: float = 1.0,
        velocity_time_constant_s: float = 0.02,
        wheel_center_jacobians: bool = False,
    ) -> None:
        for value in (attitude_time_constant_s, velocity_time_constant_s):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("filter time constants must be finite and positive")
        self.attitude_time_constant_s = float(attitude_time_constant_s)
        self.velocity_time_constant_s = float(velocity_time_constant_s)
        self.wheel_center_jacobians = bool(wheel_center_jacobians)
        if self.wheel_center_jacobians:
            self.schema = "d1-imu-encoder-complementary-wheel-center-v2"
        self.model = build_d1_model()
        self.data = mujoco.MjData(self.model)
        self._base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._joints = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in D1_JOINT_NAMES
            ]
        )
        self._qadr = self.model.jnt_qposadr[self._joints]
        self._dadr = self.model.jnt_dofadr[self._joints]
        self._feet = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot")
            for leg in LEG_PREFIXES
        ]
        self._com_offset = self.model.body_ipos[self._base].copy()
        self._wheel_radius_m = 0.087
        self._last_measurement: D1SensorMeasurements | None = None
        self._state: D1StateEstimate | None = None
        self._normal = np.asarray((0.0, 0.0, 1.0))
        self._plane_offset = 0.0
        self.diagnostics: dict[str, float | int | bool | str] = {}

    def reset(
        self,
        measurement: D1SensorMeasurements,
        *,
        initial_position: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.455),
        initial_rpy: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    ) -> D1StateEstimate:
        position = _immutable_array(initial_position, shape=(3,)).copy()
        self._rotation = _rotation_from_rpy(_immutable_array(initial_rpy, shape=(3,)))
        self._position = position
        self._velocity = np.zeros(3)
        self._normal = np.asarray((0.0, 0.0, 1.0))
        self._plane_offset = 0.0  # Known initial world datum, not inferred altitude.
        self._last_measurement = measurement
        self.diagnostics = {
            "attitude_accelerometer_used": False,
            "odometry_contacts": 0,
            "support_plane_observable": False,
            "velocity_innovation_mps": 0.0,
        }
        self._state = self._snapshot(measurement, dt=0.0)
        return self._state

    def update(self, measurement: D1SensorMeasurements) -> D1StateEstimate:
        previous = self._last_measurement
        if previous is None:
            raise RuntimeError("reset must precede update")
        dt = measurement.time_s - previous.time_s
        if measurement.sequence <= previous.sequence or not 0.0 < dt <= 0.1:
            raise ValueError("measurements require increasing sequence/time and dt <= 0.1 s")
        old_rotation = self._rotation.copy()
        gyro = 0.5 * (previous.gyro_rad_s + measurement.gyro_rad_s)
        force_norm = float(np.linalg.norm(measurement.specific_force_m_s2))
        accelerometer_used = abs(force_norm - 9.81) < 1.5 and measurement.wheel_contact.any()
        if accelerometer_used:
            measured_up = measurement.specific_force_m_s2 / force_norm
            predicted_up = self._rotation.T @ np.asarray((0.0, 0.0, 1.0))
            gyro = gyro + np.cross(measured_up, predicted_up) / self.attitude_time_constant_s
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, self._rotation.reshape(9))
        mujoco.mju_quatIntegrate(quaternion, gyro, dt)
        mujoco.mju_quat2Mat(self._rotation.reshape(9), quaternion)
        inertial_acceleration = self._rotation @ measurement.specific_force_m_s2 + np.asarray(
            (0.0, 0.0, -9.81)
        )
        self._velocity += inertial_acceleration * dt
        self.diagnostics["attitude_accelerometer_used"] = bool(accelerometer_used)
        # Position refers to the visible origin, velocity to its displaced COM.
        old_com = self._position + old_rotation @ self._com_offset
        self._snapshot(measurement, dt=dt)
        self._position = old_com + self._velocity * dt - self._rotation @ self._com_offset
        self._last_measurement = measurement
        self._state = self._snapshot(measurement, dt=0.0)
        return self._state

    def _kinematics(self, measurement: D1SensorMeasurements) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:3] = self._position
        mujoco.mju_mat2Quat(self.data.qpos[3:7], self._rotation.reshape(9))
        self.data.qpos[self._qadr] = measurement.joint_position_rad
        # Only FK/Jacobians: no collision detection, dynamics or live MjData.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        centers = self.data.xpos[self._feet].copy()
        jacobians = np.zeros((4, 3, 4))
        for index, foot in enumerate(self._feet):
            jac = np.zeros((3, self.model.nv))
            jacobian_function = (
                mujoco.mj_jacBody if self.wheel_center_jacobians else mujoco.mj_jacBodyCom
            )
            jacobian_function(self.model, self.data, jac, None, foot)
            jacobians[index] = jac[:, self._dadr[4 * index : 4 * index + 4]]
        return centers, jacobians

    def _snapshot(self, measurement: D1SensorMeasurements, *, dt: float) -> D1StateEstimate:
        centers, foot_jacobians = self._kinematics(measurement)
        selected = centers[measurement.wheel_contact]
        plane_observable = False
        if len(selected) >= 3:
            _, singular, vh = np.linalg.svd(selected - selected.mean(axis=0), full_matrices=False)
            if singular[1] > 0.02:
                normal = vh[-1]
                if normal[2] < 0:
                    normal = -normal
                if normal[2] > 0.5:
                    self._normal = normal
                    plane_observable = True
        if len(selected):
            self._plane_offset = float(np.mean(selected @ self._normal) - self._wheel_radius_m)
        contact_points = np.zeros((4, 3))
        contact_normals = np.zeros((4, 3))
        contact_jacobians = np.zeros((4, 3, 4))
        velocities = []
        omega_world = self._rotation @ measurement.gyro_rad_s
        for index in np.flatnonzero(measurement.wheel_contact):
            point = centers[index] - self._wheel_radius_m * self._normal
            contact_points[index], contact_normals[index] = point, self._normal
            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jac, None, point, self._feet[index])
            indices = slice(4 * index, 4 * index + 4)
            contact_jacobians[index] = jac[:, self._dadr[indices]]
            # v_contact = v_origin + omega x r + J_joint qdot = 0.
            velocity_origin = (
                -np.cross(omega_world, point - self._position)
                - contact_jacobians[index] @ measurement.joint_velocity_rad_s[indices]
            )
            velocities.append(
                velocity_origin + np.cross(omega_world, self._rotation @ self._com_offset)
            )
        if dt > 0.0 and velocities:
            odometry = np.mean(velocities, axis=0)
            self.diagnostics["velocity_innovation_mps"] = float(
                np.linalg.norm(odometry - self._velocity)
            )
            alpha = dt / (self.velocity_time_constant_s + dt)
            self._velocity += alpha * (odometry - self._velocity)
        self.diagnostics["odometry_contacts"] = len(velocities)
        self.diagnostics["support_plane_observable"] = plane_observable
        self.diagnostics["support_plane_status"] = (
            "wheel_center_fit"
            if plane_observable
            else "retained_normal"
            if velocities
            else "no_contact_dead_reckoning"
        )
        return D1StateEstimate(
            sequence=measurement.sequence,
            control_time_s=measurement.time_s,
            measurement_time_s=measurement.time_s,
            base_position=self._position,
            base_rotation=self._rotation,
            base_linear_velocity_body=self._rotation.T @ self._velocity,
            base_angular_velocity_body=measurement.gyro_rad_s,
            base_linear_velocity_world=self._velocity,
            base_angular_velocity_world=omega_world,
            joint_position=measurement.joint_position_rad,
            joint_velocity=measurement.joint_velocity_rad_s,
            foot_position=centers,
            foot_jacobian=foot_jacobians,
            wheel_contact=measurement.wheel_contact,
            wheel_contact_point=contact_points,
            wheel_contact_normal=contact_normals,
            wheel_contact_jacobian=contact_jacobians,
            undesired_ground_contacts=0,
        )

    def ground_reference(self, state: D1StateEstimate) -> TrainingGroundReference:
        """Local support plane in the drifting estimator world, not a map query."""
        nx, ny, nz = self._normal
        x, y = state.base_position[:2]
        return TrainingGroundReference(
            height_m=float((self._plane_offset - nx * x - ny * y) / nz),
            pitch_rad=float(np.arctan2(nx, nz)),
            roll_rad=float(np.arctan2(-ny, nz)),
        )


class D1SensorStateSource:
    """Read/reset seam, optionally delaying measurements before fusion.

    Startup repeats the initial packet until a delayed packet is available; no
    hidden simulation warmup or truth extrapolation is performed. Publication
    time is current control time and measurement_time identifies the used packet.
    """

    schema = D1ProprioceptiveEstimator.schema

    def __init__(
        self,
        plant: D1Plant,
        *,
        noise: D1SensorNoise | None = None,
        seed: int | None = None,
        initial_position: tuple[float, float, float] = (0.0, 0.0, 0.455),
        initial_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
        delay_steps: int = 0,
    ) -> None:
        if (
            isinstance(delay_steps, bool)
            or not isinstance(delay_steps, Integral)
            or delay_steps < 0
        ):
            raise ValueError("delay_steps must be a non-negative integer")
        self.measurements = D1MujocoSensorSource(plant, noise=noise, seed=seed)
        self.estimator = D1ProprioceptiveEstimator(
            wheel_center_jacobians=plant.sampling_mode == "synchronized"
        )
        self.schema = self.estimator.schema
        self.delay_steps = int(delay_steps)
        self._queue: deque[D1SensorMeasurements] = deque()
        self._latest_measurement: D1SensorMeasurements | None = None
        self._fusion_state: D1StateEstimate | None = None
        self.initial_position = tuple(
            float(x) for x in _immutable_array(initial_position, shape=(3,))
        )
        self.initial_rpy = tuple(float(x) for x in _immutable_array(initial_rpy, shape=(3,)))

    def reset(self, *, seed: int | None = None) -> D1StateEstimate:
        self._latest_measurement = self.measurements.reset(seed=seed)
        self._queue = deque([self._latest_measurement])
        self._fusion_state = self.estimator.reset(
            self._latest_measurement,
            initial_position=self.initial_position,
            initial_rpy=self.initial_rpy,
        )
        return self._fusion_state

    def read(self) -> D1StateEstimate:
        if self._fusion_state is None:
            raise RuntimeError("reset must precede read")
        current = self.measurements.read()
        self._latest_measurement = current
        self._queue.append(current)
        while len(self._queue) > self.delay_steps + 1:
            self._queue.popleft()
        packet = self._queue[0]
        if packet.sequence > self._fusion_state.sequence:
            self._fusion_state = self.estimator.update(packet)
        return replace(
            self._fusion_state,
            sequence=current.sequence,
            control_time_s=current.time_s,
            measurement_time_s=packet.time_s,
        )

    @property
    def latest_measurement(self) -> D1SensorMeasurements:
        """Most recently captured immutable packet, for recording/replay only."""
        if self._latest_measurement is None:
            raise RuntimeError("reset must precede reading measurements")
        return self._latest_measurement

    def ground_reference(self, state: D1StateEstimate) -> TrainingGroundReference:
        return self.estimator.ground_reference(state)
