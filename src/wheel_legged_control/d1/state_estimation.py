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

from .model import (
    D1_JOINT_NAMES,
    JOINT_POSITION_HIGH,
    JOINT_POSITION_LOW,
    LEG_PREFIXES,
    D1Plant,
)


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
    Linear velocities refer to the base inertial COM, while ``base_position``
    refers to the visible body origin (a legacy convention preserved for control).
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
    wheel_contact_normal: np.ndarray
    wheel_contact_jacobian: np.ndarray
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
            "wheel_contact_normal": ((len(LEG_PREFIXES), 3), np.float64),
            "wheel_contact_jacobian": ((len(LEG_PREFIXES), 3, 4), np.float64),
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

    def base_origin_velocity(
        self, com_offset_body_m: np.ndarray, *, local: bool = False
    ) -> np.ndarray:
        """Convert measured COM velocity to the visible origin's derivative.

        The offset is an explicit nominal model parameter, not live truth.
        Pass it in base_link axes; rotational motion contributes omega cross r.
        """

        offset = _immutable_array(com_offset_body_m, shape=(3,))
        world = self.base_linear_velocity_world - np.cross(
            self.base_angular_velocity_world, self.base_rotation @ offset
        )
        return self.base_rotation.T @ world if local else world

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

    def _contact_point_jacobian(self, foot_index: int, point_world_m: np.ndarray) -> np.ndarray:
        translation = np.zeros((3, self.plant.model.nv), dtype=np.float64)
        rotation = np.zeros_like(translation)
        mujoco.mj_jac(
            self.plant.model,
            self.plant.measurement_data,
            translation,
            rotation,
            point_world_m,
            self._foot_body_ids[foot_index],
        )
        return translation[:, self._leg_dof_addresses[foot_index]]

    def _capture(self) -> D1StateEstimate:
        plant = self.plant
        foot_position = np.asarray(
            [plant.measurement_data.xpos[body_id] for body_id in self._foot_body_ids],
            dtype=np.float64,
        )
        foot_jacobian = np.empty((len(LEG_PREFIXES), 3, 4), dtype=np.float64)
        for index, (body_id, dof_addresses) in enumerate(
            zip(self._foot_body_ids, self._leg_dof_addresses, strict=True)
        ):
            translation = np.zeros((3, plant.model.nv), dtype=np.float64)
            rotation = np.zeros_like(translation)
            jacobian_function = (
                mujoco.mj_jacBody
                if plant.sampling_mode == "synchronized"
                else mujoco.mj_jacBodyCom
            )
            jacobian_function(
                plant.model,
                plant.measurement_data,
                translation,
                rotation,
                body_id,
            )
            foot_jacobian[index] = translation[:, dof_addresses]

        contact_points: list[list[np.ndarray]] = [[] for _ in range(len(LEG_PREFIXES))]
        contact_normals: list[list[np.ndarray]] = [[] for _ in range(len(LEG_PREFIXES))]
        for contact in plant.measurement_data.contact:
            if int(contact.efc_address) < 0:
                continue
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
                point = np.asarray(contact.pos).copy()
                normal = np.asarray(contact.frame).reshape(3, 3)[0].copy()
                if np.dot(normal, foot_position[foot_index] - point) < 0.0:
                    normal *= -1.0
                contact_points[foot_index].append(point)
                contact_normals[foot_index].append(normal)

        wheel_contact = np.asarray(
            [bool(points) for points in contact_points],
            dtype=np.bool_,
        )
        wheel_contact_point = np.zeros((len(LEG_PREFIXES), 3), dtype=np.float64)
        wheel_contact_normal = np.zeros((len(LEG_PREFIXES), 3), dtype=np.float64)
        wheel_contact_jacobian = np.zeros((len(LEG_PREFIXES), 3, 4), dtype=np.float64)
        for index, points in enumerate(contact_points):
            if points:
                wheel_contact_point[index] = np.mean(points, axis=0)
                mean_normal = np.mean(contact_normals[index], axis=0)
                wheel_contact_normal[index] = mean_normal / np.linalg.norm(mean_normal)
                wheel_contact_jacobian[index] = self._contact_point_jacobian(
                    index,
                    wheel_contact_point[index],
                )

        linear_body, angular_body = plant.base_velocity(local=True)
        linear_world, angular_world = plant.base_velocity(local=False)
        time_s = float(plant.measurement_data.time)
        return D1StateEstimate(
            sequence=self._sequence,
            control_time_s=time_s,
            measurement_time_s=time_s,
            base_position=plant.base_position,
            base_rotation=plant.measurement_data.xmat[plant.base_body_id].reshape(3, 3),
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
            wheel_contact_normal=wheel_contact_normal,
            wheel_contact_jacobian=wheel_contact_jacobian,
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
    contact_normal_std_rad: float = 0.0
    contact_jacobian_std: float = 0.0
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
            self.contact_normal_std_rad,
            self.contact_jacobian_std,
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
        contact_normal_std_rad=0.004 * scale,
        contact_jacobian_std=0.001 * scale,
        contact_flip_probability=min(0.005 * scale, 1.0),
    )


def _rotation_from_vector(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    if angle == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z = rotation_vector / angle
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


@dataclass(frozen=True, slots=True)
class D1LatencyCompensationResult:
    """State plus provenance for one short-horizon latency compensation step."""

    control_state: D1StateEstimate
    method: str
    source_age_s: float
    applied_horizon_s: float
    estimate_time_s: float
    status: str


def compensate_d1_state_constant_velocity(
    measured: D1StateEstimate,
    *,
    max_horizon_s: float = 0.05,
    max_leg_joint_delta_rad: float = 0.35,
) -> D1LatencyCompensationResult:
    """Extrapolate a delayed snapshot with constant body twist and joint velocity.

    Contact flags and contact points remain measured values.  The function is a
    short-horizon latency compensator, not a sensor observer or contact predictor.
    """

    if not np.isfinite(max_horizon_s) or max_horizon_s <= 0.0:
        raise ValueError("max_horizon_s must be finite and positive")
    if not np.isfinite(max_leg_joint_delta_rad) or max_leg_joint_delta_rad <= 0.0:
        raise ValueError("max_leg_joint_delta_rad must be finite and positive")
    age_s = measured.age_s
    common = {
        "method": "constant_body_twist_first_order",
        "source_age_s": age_s,
    }
    if age_s <= 1e-12:
        return D1LatencyCompensationResult(
            control_state=measured,
            applied_horizon_s=0.0,
            estimate_time_s=measured.measurement_time_s,
            status="bypassed",
            **common,
        )
    if age_s > max_horizon_s + 1e-12:
        return D1LatencyCompensationResult(
            control_state=measured,
            applied_horizon_s=0.0,
            estimate_time_s=measured.measurement_time_s,
            status="horizon_exceeded",
            **common,
        )

    leg_indices = np.asarray(
        [index for index in range(len(D1_JOINT_NAMES)) if index % 4 != 3],
        dtype=np.int32,
    )
    joint_delta = measured.joint_velocity * age_s
    predicted_joint_position = measured.joint_position + joint_delta
    invalid_leg_prediction = bool(
        np.any(np.abs(joint_delta[leg_indices]) > max_leg_joint_delta_rad)
        or np.any(predicted_joint_position[leg_indices] < JOINT_POSITION_LOW[leg_indices])
        or np.any(predicted_joint_position[leg_indices] > JOINT_POSITION_HIGH[leg_indices])
    )
    if invalid_leg_prediction:
        return D1LatencyCompensationResult(
            control_state=measured,
            applied_horizon_s=0.0,
            estimate_time_s=measured.measurement_time_s,
            status="kinematic_horizon_exceeded",
            **common,
        )

    rotation = measured.base_rotation
    angular_step = measured.base_angular_velocity_body * age_s
    predicted_rotation = rotation @ _rotation_from_vector(angular_step)
    midpoint_rotation = rotation @ _rotation_from_vector(0.5 * angular_step)
    predicted_position = (
        measured.base_position + midpoint_rotation @ measured.base_linear_velocity_body * age_s
    )

    predicted_foot_position = np.empty_like(measured.foot_position)
    predicted_foot_jacobian = np.empty_like(measured.foot_jacobian)
    for leg_index in range(len(LEG_PREFIXES)):
        joint_slice = slice(4 * leg_index, 4 * (leg_index + 1))
        foot_offset_body = rotation.T @ (measured.foot_position[leg_index] - measured.base_position)
        jacobian_body = rotation.T @ measured.foot_jacobian[leg_index]
        predicted_offset_body = (
            foot_offset_body + (jacobian_body @ measured.joint_velocity[joint_slice]) * age_s
        )
        predicted_foot_position[leg_index] = (
            predicted_position + predicted_rotation @ predicted_offset_body
        )
        predicted_foot_jacobian[leg_index] = predicted_rotation @ jacobian_body

    control_state = replace(
        measured,
        base_position=predicted_position,
        base_rotation=predicted_rotation,
        base_linear_velocity_world=(predicted_rotation @ measured.base_linear_velocity_body),
        base_angular_velocity_world=(predicted_rotation @ measured.base_angular_velocity_body),
        joint_position=predicted_joint_position,
        foot_position=predicted_foot_position,
        foot_jacobian=predicted_foot_jacobian,
    )
    return D1LatencyCompensationResult(
        control_state=control_state,
        applied_horizon_s=age_s,
        estimate_time_s=measured.control_time_s,
        status="applied",
        **common,
    )


def prepare_d1_control_state(
    measured: D1StateEstimate,
    *,
    latency_compensation: str = "none",
) -> D1LatencyCompensationResult:
    """Apply the configured latency treatment with one shared mode switch."""

    if latency_compensation == "none":
        return D1LatencyCompensationResult(
            control_state=measured,
            method="none",
            source_age_s=measured.age_s,
            applied_horizon_s=0.0,
            estimate_time_s=measured.measurement_time_s,
            status="disabled",
        )
    if latency_compensation == "constant_velocity":
        return compensate_d1_state_constant_velocity(measured)
    raise ValueError("latency_compensation must be 'none' or 'constant_velocity'")


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
        contact_normal = state.wheel_contact_normal.copy()
        contact_jacobian = state.wheel_contact_jacobian.copy()
        newly_contacting = wheel_contact & ~state.wheel_contact
        contact_normal[newly_contacting] = np.asarray((0.0, 0.0, 1.0))
        contact_point[newly_contacting] = (
            state.foot_position[newly_contacting]
            - self.truth_source.plant.wheel_radius_m * contact_normal[newly_contacting]
        )
        contact_jacobian[newly_contacting] = state.foot_jacobian[newly_contacting]
        for index in np.flatnonzero(newly_contacting):
            contact_jacobian[index] = self.truth_source._contact_point_jacobian(
                int(index),
                contact_point[index],
            )
        for index in np.flatnonzero(wheel_contact):
            contact_normal[index] = _rotation_from_vector(
                self._normal((3,), config.contact_normal_std_rad)
            ) @ contact_normal[index]
            contact_normal[index] /= np.linalg.norm(contact_normal[index])
            if np.dot(
                contact_normal[index],
                state.foot_position[index] - contact_point[index],
            ) < 0.0:
                contact_normal[index] *= -1.0
            contact_jacobian[index] += self._normal(
                (3, 4),
                config.contact_jacobian_std,
            )
        contact_point[~wheel_contact] = 0.0
        contact_normal[~wheel_contact] = 0.0
        contact_jacobian[~wheel_contact] = 0.0

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
            wheel_contact_normal=contact_normal,
            wheel_contact_jacobian=contact_jacobian,
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
