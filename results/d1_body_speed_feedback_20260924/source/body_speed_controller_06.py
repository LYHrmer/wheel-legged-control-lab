"""Cooperative body-COM-speed wheel-P stage after the original single PI update."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from body_speed_math_06 import (
    WHEEL_KP,
    WHEEL_RADIUS_M,
    WHEELS,
    body_common_increment,
    finite_array,
    finite_scalar,
)
from drive_damping_controller_04 import (
    DriveDampingStageController,
    DriveWheelLegResult,
    assert_frozen_arithmetic_identity,
)
from drive_damping_math_04 import protect_requested

from scripts.d1_rolling_residual_env import RollingResidualCompositionController
from scripts.d1_turn_yaw_authority import WHEEL_RADIUS_M as ORIGINAL_WHEEL_RADIUS_M

BODY_RECORD_SCHEMA = "d1-drive-body-common-p-record-v1"
BODY_CONTROL_SCHEMA = "d1-rolling-shared-leg-drive-jx-body-common-p-stop-turn-v1"
BODY_TASK_SCHEMA = "d1-rolling-residual-drive-jx-body-common-p-transfer-v1"


def _readonly(value: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = finite_array(value, shape, label)
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.float64).reshape(shape)


@dataclass(frozen=True)
class BodyCommonPRecord:
    schema: str
    active: bool
    raw_forward_mps: float
    servo_forward_mps: float
    control_time_s: float
    provider_state_sequence: int
    provider_state_age_s: float
    base_linear_velocity_body: np.ndarray
    base_rotation: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    wheel_omega_rad_s: np.ndarray
    mean_omega_rad_s: float
    body_equivalent_omega_rad_s: float
    wheel_kp: float
    wheel_radius_m: float
    common_error_rad_s: float
    scalar_delta_nm: float
    delta_torque_nm: np.ndarray
    parent_requested_torque_nm: np.ndarray
    parent_protected_torque_nm: np.ndarray
    parent_wheel_nm: np.ndarray
    total_requested_torque_nm: np.ndarray
    total_protected_torque_nm: np.ndarray
    total_wheel_nm: np.ndarray
    actual_protected_increment_nm: np.ndarray
    total_torque_limited: np.ndarray
    original_pi_integral_before_nm: np.ndarray
    original_pi_integral_after_nm: np.ndarray
    nominal_wheel_speed_rad_s: np.ndarray
    wheel_speed_target_rad_s: np.ndarray

    def __post_init__(self) -> None:
        if self.schema != BODY_RECORD_SCHEMA or type(self.active) is not bool:
            raise ValueError("body-P record schema or gate differs")
        if type(self.provider_state_sequence) is not int:
            raise TypeError("provider sequence must be an integer")
        for name in ("raw_forward_mps", "servo_forward_mps", "control_time_s",
                     "provider_state_age_s", "mean_omega_rad_s",
                     "body_equivalent_omega_rad_s", "wheel_kp", "wheel_radius_m",
                     "common_error_rad_s", "scalar_delta_nm"):
            object.__setattr__(self, name, finite_scalar(getattr(self, name), name))
        for name, shape in (
            ("base_linear_velocity_body", (3,)), ("base_rotation", (3, 3)),
            ("joint_position", (16,)), ("joint_velocity", (16,)),
            ("wheel_omega_rad_s", (4,)), ("delta_torque_nm", (16,)),
            ("parent_requested_torque_nm", (16,)),
            ("parent_protected_torque_nm", (16,)), ("parent_wheel_nm", (16,)),
            ("total_requested_torque_nm", (16,)),
            ("total_protected_torque_nm", (16,)), ("total_wheel_nm", (16,)),
            ("actual_protected_increment_nm", (16,)),
            ("original_pi_integral_before_nm", (4,)),
            ("original_pi_integral_after_nm", (4,)),
            ("nominal_wheel_speed_rad_s", (4,)),
            ("wheel_speed_target_rad_s", (4,)),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), shape, name))
        limited = np.asarray(self.total_torque_limited)
        if limited.shape != (16,) or limited.dtype.kind != "b":
            raise TypeError("total_torque_limited must be a bool[16] array")
        object.__setattr__(self, "total_torque_limited",
                           np.frombuffer(np.ascontiguousarray(limited).tobytes(),
                                         dtype=np.bool_))
        if np.any(self.delta_torque_nm[[i for i in range(16) if i not in WHEELS]] != 0.0):
            raise ValueError("body-P increment must have zero leg channels")


@dataclass(frozen=True)
class BodyWheelLegResult(DriveWheelLegResult):
    body_common_p: BodyCommonPRecord


class BodyCommonPStageController(DriveDampingStageController):
    """Read the real drive result, then add a common wheel-only P request."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.wheel_kp != WHEEL_KP or self.wheel_ki != 3.0 or WHEEL_RADIUS_M != ORIGINAL_WHEEL_RADIUS_M:
            raise RuntimeError("common-P constants differ from original PI/radius")
        self.last_body_common_p: BodyCommonPRecord | None = None

    def reset(self) -> None:
        self.last_body_common_p = None
        super().reset()

    def compute(self, command, state, action, *, ground_height_m: float = 0.0):
        checked = self._checked_zero_action(action)
        binding = self._require_binding(command, state)
        raw_forward = finite_scalar(binding.forward_velocity_mps, "raw forward")
        servo_forward = finite_scalar(command.forward_velocity_mps, "servo forward")
        body_velocity = finite_array(state.base_linear_velocity_body, (3,), "body velocity").copy()
        rotation = finite_array(state.base_rotation, (3, 3), "base rotation").copy()
        position = finite_array(state.joint_position, (16,), "joint position").copy()
        velocity = finite_array(state.joint_velocity, (16,), "joint velocity").copy()
        time_s = finite_scalar(state.control_time_s, "control time")
        age_s = finite_scalar(state.age_s, "provider age")
        if type(state.sequence) is not int:
            raise TypeError("provider sequence must be an integer")
        omega = velocity[list(WHEELS)].copy()
        increment = body_common_increment(omega, body_velocity, raw_forward_mps=raw_forward)

        parent_torque = super().compute(command, state, checked,
                                        ground_height_m=ground_height_m)
        parent = self.last_result
        if not isinstance(parent, DriveWheelLegResult):
            raise TypeError("drive stage did not publish its actual result")
        delta = increment["delta_torque_nm"]
        if increment["active"]:
            requested = parent.requested_torque_nm + delta
            protected, limited = protect_requested(requested, position, velocity)
            wheel_nm = parent.wheel_nm + delta
        else:
            requested = parent.requested_torque_nm
            protected = parent.torque_nm
            limited = parent.torque_limited
            wheel_nm = parent.wheel_nm
        record = BodyCommonPRecord(
            schema=BODY_RECORD_SCHEMA, active=increment["active"],
            raw_forward_mps=raw_forward, servo_forward_mps=servo_forward,
            control_time_s=time_s, provider_state_sequence=state.sequence,
            provider_state_age_s=age_s,
            base_linear_velocity_body=body_velocity, base_rotation=rotation,
            joint_position=position, joint_velocity=velocity,
            wheel_omega_rad_s=omega,
            mean_omega_rad_s=increment["mean_omega_rad_s"],
            body_equivalent_omega_rad_s=increment["body_equivalent_omega_rad_s"],
            wheel_kp=WHEEL_KP, wheel_radius_m=WHEEL_RADIUS_M,
            common_error_rad_s=increment["common_error_rad_s"],
            scalar_delta_nm=increment["scalar_delta_nm"], delta_torque_nm=delta,
            parent_requested_torque_nm=parent.requested_torque_nm,
            parent_protected_torque_nm=parent.torque_nm,
            parent_wheel_nm=parent.wheel_nm,
            total_requested_torque_nm=requested,
            total_protected_torque_nm=protected,
            total_wheel_nm=wheel_nm,
            actual_protected_increment_nm=protected - parent.torque_nm,
            total_torque_limited=limited,
            original_pi_integral_before_nm=parent.memory_before.wheel_integral_nm,
            original_pi_integral_after_nm=parent.memory_after.wheel_integral_nm,
            nominal_wheel_speed_rad_s=parent.nominal_wheel_speed_rad_s,
            wheel_speed_target_rad_s=parent.wheel_speed_target_rad_s,
        )
        payload = {field.name: getattr(parent, field.name) for field in fields(DriveWheelLegResult)}
        payload.update(wheel_nm=wheel_nm, requested_torque_nm=requested,
                       torque_nm=protected, torque_limited=limited)
        result = BodyWheelLegResult(**payload, body_common_p=record)
        self.last_result = result
        self.last_body_common_p = record
        return protected.copy() if increment["active"] else parent_torque


class BodyCommonPRollingController(RollingResidualCompositionController,
                                   BodyCommonPStageController):
    def __init__(self, **kwargs) -> None:
        assert_frozen_arithmetic_identity()
        super().__init__(**kwargs)
        if not self.enabled:
            raise ValueError("body-P transfer requires enabled composition")
        self.control_schema = BODY_CONTROL_SCHEMA


EXPECTED_MRO = (
    "BodyCommonPRollingController", "RollingResidualCompositionController",
    "StopTurnCompositionController", "BodyCommonPStageController",
    "DriveDampingStageController", "TurnYawAuthorityController",
    "D1WheelLegController",
)


def assert_cooperative_mro() -> tuple[str, ...]:
    names = tuple(cls.__name__ for cls in BodyCommonPRollingController.__mro__[:7])
    if names != EXPECTED_MRO:
        raise RuntimeError(f"body-P cooperative MRO differs: {names}")
    return names
