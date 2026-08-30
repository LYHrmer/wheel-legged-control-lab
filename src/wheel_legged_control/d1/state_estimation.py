"""Control-ready D1 state snapshots and MuJoCo estimator-channel sources.

The noisy source in this first version perturbs and delays complete MuJoCo truth
snapshots.  It is an estimator-channel model for robustness experiments, not a
sensor-fusion algorithm, observer, or EKF.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from .model import D1_JOINT_NAMES, LEG_PREFIXES, D1Plant


def _immutable_array(
    value: np.ndarray,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Copy an array into immutable byte-backed storage and validate it once."""

    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError("state arrays must contain only finite values")
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(shape)


@dataclass(frozen=True, slots=True)
class D1StateEstimate:
    """Immutable, control-ready estimate using explicit world and body frames.

    ``base_rotation`` maps body-frame vectors into the world frame.  Base
    velocities are stored in both the visible ``base_link`` and world frames.
    Foot positions and translational Jacobians are expressed in the world frame;
    each Jacobian contains the four columns for that leg in D1 joint order.
    Contact-point rows are zero when the corresponding contact flag is false.
    """

    sequence: int
    control_time_s: float
    measurement_time_s: float
    base_position: np.ndarray
    base_rotation: np.ndarray
    base_linear_velocity_body: np.ndarray
    base_angular_velocity_body: np.ndarray
    base_linear_velocity_world: np.ndarray
    base_angular_velocity_world: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    foot_position: np.ndarray
    foot_jacobian: np.ndarray
    wheel_contact: np.ndarray
    wheel_contact_point: np.ndarray
    undesired_ground_contacts: int

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        times = np.asarray((self.control_time_s, self.measurement_time_s))
        if not np.isfinite(times).all():
            raise ValueError("state times must be finite")
        if self.measurement_time_s > self.control_time_s + 1e-12:
            raise ValueError("measurement time cannot be later than control time")
        if self.undesired_ground_contacts < 0:
            raise ValueError("undesired_ground_contacts must be non-negative")

        specifications = {
            "base_position": ((3,), np.float64),
            "base_rotation": ((3, 3), np.float64),
            "base_linear_velocity_body": ((3,), np.float64),
            "base_angular_velocity_body": ((3,), np.float64),
            "base_linear_velocity_world": ((3,), np.float64),
            "base_angular_velocity_world": ((3,), np.float64),
            "joint_position": ((len(D1_JOINT_NAMES),), np.float64),
            "joint_velocity": ((len(D1_JOINT_NAMES),), np.float64),
            "foot_position": ((len(LEG_PREFIXES), 3), np.float64),
            "foot_jacobian": ((len(LEG_PREFIXES), 3, 4), np.float64),
            "wheel_contact": ((len(LEG_PREFIXES),), np.bool_),
            "wheel_contact_point": ((len(LEG_PREFIXES), 3), np.float64),
        }
        for name, (shape, dtype) in specifications.items():
            object.__setattr__(
                self,
                name,
                _immutable_array(getattr(self, name), shape=shape, dtype=dtype),
            )

        rotation = self.base_rotation
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=1e-8
        ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8, rtol=1e-8):
            raise ValueError("base_rotation must be a proper orthonormal rotation")
        for body_velocity, world_velocity in (
            (self.base_linear_velocity_body, self.base_linear_velocity_world),
            (self.base_angular_velocity_body, self.base_angular_velocity_world),
        ):
            if not np.allclose(
                rotation @ body_velocity,
                world_velocity,
                atol=1e-8,
                rtol=1e-8,
            ):
                raise ValueError("body- and world-frame velocities must agree")

    @property
    def age_s(self) -> float:
        return float(self.control_time_s - self.measurement_time_s)

    @property
    def base_rpy(self) -> np.ndarray:
        rotation = self.base_rotation
        pitch = np.arcsin(-np.clip(rotation[2, 0], -1.0, 1.0))
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.asarray((roll, pitch, yaw), dtype=np.float64)

    @property
    def projected_gravity_body(self) -> np.ndarray:
        return self.base_rotation.T @ np.asarray((0.0, 0.0, -1.0))

    @property
    def foot_offset_world(self) -> np.ndarray:
        return self.foot_position - self.base_position

    @property
    def wheel_ground_contacts(self) -> int:
        return int(np.count_nonzero(self.wheel_contact))

    def base_velocity(self, *, local: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return copied linear and angular velocity in body or world frame."""

        if local:
            return (
                self.base_linear_velocity_body.copy(),
                self.base_angular_velocity_body.copy(),
            )
        return (
            self.base_linear_velocity_world.copy(),
            self.base_angular_velocity_world.copy(),
        )

    def reduced_state(self) -> np.ndarray:
        """Return ``[x, pitch, z, vx, pitch_rate, vz]`` in world coordinates."""

        linear = self.base_linear_velocity_world
        angular = self.base_angular_velocity_world
        return np.asarray(
            (
                self.base_position[0],
                self.base_rpy[1],
                self.base_position[2],
                linear[0],
                angular[1],
                linear[2],
            ),
            dtype=np.float64,
        )

    def has_fallen(self) -> bool:
        roll, pitch, _ = self.base_rpy
        return bool(self.base_position[2] < 0.22 or abs(roll) > 0.85 or abs(pitch) > 0.85)


class D1MujocoTruthStateSource:
    """Read exact D1 state snapshots from a caller-owned MuJoCo plant."""

    def __init__(self, plant: D1Plant) -> None:
        self.plant = plant
        self._foot_body_ids = tuple(
            mujoco.mj_name2id(
                plant.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"{leg}_foot",
            )
            for leg in LEG_PREFIXES
        )
        self._body_to_foot_index = {
            body_id: index for index, body_id in enumerate(self._foot_body_ids)
        }
        self._leg_dof_addresses = tuple(
            plant.dof_addresses[
                np.asarray(
                    [index for index, name in enumerate(D1_JOINT_NAMES) if name.startswith(leg)],
                    dtype=np.int32,
                )
            ]
            for leg in LEG_PREFIXES
        )
        self._sequence = -1
        self._was_reset = False

    def _capture(self) -> D1StateEstimate:
        plant = self.plant
        foot_position = np.asarray(
            [plant.data.xpos[body_id] for body_id in self._foot_body_ids],
            dtype=np.float64,
        )
        foot_jacobian = np.empty((len(LEG_PREFIXES), 3, 4), dtype=np.float64)
        for index, (body_id, dof_addresses) in enumerate(
            zip(self._foot_body_ids, self._leg_dof_addresses, strict=True)
        ):
            translation = np.zeros((3, plant.model.nv), dtype=np.float64)
            rotation = np.zeros_like(translation)
            mujoco.mj_jacBodyCom(
                plant.model,
                plant.data,
                translation,
                rotation,
                body_id,
            )
            foot_jacobian[index] = translation[:, dof_addresses]

        contact_points: list[list[np.ndarray]] = [[] for _ in range(len(LEG_PREFIXES))]
        for contact in plant.data.contact:
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in plant.terrain_geom_ids:
                wheel_geom = geom2
            elif geom2 in plant.terrain_geom_ids:
                wheel_geom = geom1
            else:
                continue
            foot_index = self._body_to_foot_index.get(int(plant.model.geom_bodyid[wheel_geom]))
            if foot_index is not None:
                contact_points[foot_index].append(np.asarray(contact.pos).copy())

        wheel_contact = np.asarray(
            [bool(points) for points in contact_points],
            dtype=np.bool_,
        )
        wheel_contact_point = np.zeros((len(LEG_PREFIXES), 3), dtype=np.float64)
        for index, points in enumerate(contact_points):
            if points:
                wheel_contact_point[index] = np.mean(points, axis=0)

        linear_body, angular_body = plant.base_velocity(local=True)
        linear_world, angular_world = plant.base_velocity(local=False)
        time_s = float(plant.data.time)
        return D1StateEstimate(
            sequence=self._sequence,
            control_time_s=time_s,
            measurement_time_s=time_s,
            base_position=plant.base_position,
            base_rotation=plant.data.xmat[plant.base_body_id].reshape(3, 3),
            base_linear_velocity_body=linear_body,
            base_angular_velocity_body=angular_body,
            base_linear_velocity_world=linear_world,
            base_angular_velocity_world=angular_world,
            joint_position=plant.joint_position,
            joint_velocity=plant.joint_velocity,
            foot_position=foot_position,
            foot_jacobian=foot_jacobian,
            wheel_contact=wheel_contact,
            wheel_contact_point=wheel_contact_point,
            undesired_ground_contacts=plant.undesired_ground_contacts,
        )

    def reset(self, *, seed: int | None = None) -> D1StateEstimate:
        """Reset source-local ordering; the caller remains responsible for plant reset."""

        del seed
        self._sequence = 0
        self._was_reset = True
        return self._capture()

    def read(self) -> D1StateEstimate:
        if not self._was_reset:
            raise RuntimeError("D1 state source must be reset before read")
        self._sequence += 1
        return self._capture()


@dataclass(frozen=True, slots=True)
class D1EstimatorImpairments:
    """Field-level noise and integer-step delay for the estimator channel."""

    delay_steps: int = 0
    base_position_std_m: float = 0.0
    base_rotation_std_rad: float = 0.0
    base_linear_velocity_std_mps: float = 0.0
    base_angular_velocity_std_radps: float = 0.0
    joint_position_std_rad: float = 0.0
    joint_velocity_std_radps: float = 0.0
    foot_position_std_m: float = 0.0
    foot_jacobian_std: float = 0.0
    contact_point_std_m: float = 0.0
    contact_flip_probability: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.delay_steps, bool) or not isinstance(self.delay_steps, int):
            raise TypeError("delay_steps must be an integer")
        if self.delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        standard_deviations = (
            self.base_position_std_m,
            self.base_rotation_std_rad,
            self.base_linear_velocity_std_mps,
            self.base_angular_velocity_std_radps,
            self.joint_position_std_rad,
            self.joint_velocity_std_radps,
            self.foot_position_std_m,
            self.foot_jacobian_std,
            self.contact_point_std_m,
        )
        if not np.isfinite(standard_deviations).all() or any(
            value < 0.0 for value in standard_deviations
        ):
            raise ValueError("noise standard deviations must be finite and non-negative")
        probability = self.contact_flip_probability
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("contact_flip_probability must be in [0, 1]")


def make_default_d1_estimator_impairments(
    *,
    delay_steps: int = 0,
    noise_scale: float = 1.0,
) -> D1EstimatorImpairments:
    """Return the shared D1 robustness profile used by training and teleoperation."""

    if not np.isfinite(noise_scale) or noise_scale < 0.0:
        raise ValueError("noise_scale must be finite and non-negative")
    scale = float(noise_scale)
    return D1EstimatorImpairments(
        delay_steps=delay_steps,
        base_position_std_m=0.002 * scale,
        base_rotation_std_rad=0.004 * scale,
        base_linear_velocity_std_mps=0.020 * scale,
        base_angular_velocity_std_radps=0.010 * scale,
        joint_position_std_rad=0.001 * scale,
        joint_velocity_std_radps=0.010 * scale,
        foot_position_std_m=0.001 * scale,
        foot_jacobian_std=0.001 * scale,
        contact_point_std_m=0.002 * scale,
        contact_flip_probability=min(0.005 * scale, 1.0),
    )


def _rotation_from_vector(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    if angle == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z = rotation_vector / angle
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


class D1NoisyDelayedStateSource:
    """Delay and perturb complete truth snapshots as an estimator-channel model.

    This Adapter is intentionally not presented as an EKF or a realistic IMU
    and encoder fusion pipeline.  It is a deterministic robustness-test seam
    until a proprioceptive estimator is implemented.
    """

    def __init__(
        self,
        plant: D1Plant,
        *,
        impairments: D1EstimatorImpairments | None = None,
        seed: int | None = None,
    ) -> None:
        self.truth_source = D1MujocoTruthStateSource(plant)
        self.impairments = D1EstimatorImpairments() if impairments is None else impairments
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._delay_buffer: deque[D1StateEstimate] = deque()
        self._sequence = -1
        self._was_reset = False

    def _normal(self, shape: tuple[int, ...], standard_deviation: float) -> np.ndarray:
        if standard_deviation == 0.0:
            return np.zeros(shape, dtype=np.float64)
        return self._rng.normal(0.0, standard_deviation, size=shape)

    def _perturb(self, state: D1StateEstimate) -> D1StateEstimate:
        config = self.impairments
        rotation_noise = _rotation_from_vector(self._normal((3,), config.base_rotation_std_rad))
        wheel_contact = state.wheel_contact.copy()
        if config.contact_flip_probability > 0.0:
            wheel_contact ^= self._rng.random(len(LEG_PREFIXES)) < config.contact_flip_probability

        contact_point = state.wheel_contact_point.copy()
        contact_point += self._normal(contact_point.shape, config.contact_point_std_m)
        newly_contacting = wheel_contact & ~state.wheel_contact
        contact_point[newly_contacting] = state.foot_position[newly_contacting]
        contact_point[~wheel_contact] = 0.0

        base_rotation = state.base_rotation @ rotation_noise
        linear_body = state.base_linear_velocity_body + self._normal(
            (3,), config.base_linear_velocity_std_mps
        )
        angular_body = state.base_angular_velocity_body + self._normal(
            (3,), config.base_angular_velocity_std_radps
        )
        return D1StateEstimate(
            sequence=state.sequence,
            control_time_s=state.control_time_s,
            measurement_time_s=state.measurement_time_s,
            base_position=state.base_position + self._normal((3,), config.base_position_std_m),
            base_rotation=base_rotation,
            base_linear_velocity_body=linear_body,
            base_angular_velocity_body=angular_body,
            base_linear_velocity_world=base_rotation @ linear_body,
            base_angular_velocity_world=base_rotation @ angular_body,
            joint_position=state.joint_position
            + self._normal(state.joint_position.shape, config.joint_position_std_rad),
            joint_velocity=state.joint_velocity
            + self._normal(state.joint_velocity.shape, config.joint_velocity_std_radps),
            foot_position=state.foot_position
            + self._normal(state.foot_position.shape, config.foot_position_std_m),
            foot_jacobian=state.foot_jacobian
            + self._normal(state.foot_jacobian.shape, config.foot_jacobian_std),
            wheel_contact=wheel_contact,
            wheel_contact_point=contact_point,
            undesired_ground_contacts=state.undesired_ground_contacts,
        )

    def reset(self, *, seed: int | None = None) -> D1StateEstimate:
        effective_seed = self.seed if seed is None else seed
        self._rng = np.random.default_rng(effective_seed)
        measurement = self._perturb(self.truth_source.reset())
        self._delay_buffer = deque([measurement] * self.impairments.delay_steps)
        self._sequence = 0
        self._was_reset = True
        return replace(measurement, sequence=0)

    def read(self) -> D1StateEstimate:
        if not self._was_reset:
            raise RuntimeError("D1 state source must be reset before read")
        truth = self.truth_source.read()
        measurement = self._perturb(truth)
        if self.impairments.delay_steps:
            self._delay_buffer.append(measurement)
            delivered = self._delay_buffer.popleft()
        else:
            delivered = measurement
        self._sequence += 1
        return replace(
            delivered,
            sequence=self._sequence,
            control_time_s=truth.control_time_s,
        )


def make_d1_state_source(
    plant: D1Plant,
    *,
    impairments: D1EstimatorImpairments | None = None,
    seed: int | None = None,
) -> D1MujocoTruthStateSource | D1NoisyDelayedStateSource:
    """Build the simple truth source by default, or the channel model on request."""

    if impairments is None:
        return D1MujocoTruthStateSource(plant)
    return D1NoisyDelayedStateSource(
        plant,
        impairments=impairments,
        seed=seed,
    )
