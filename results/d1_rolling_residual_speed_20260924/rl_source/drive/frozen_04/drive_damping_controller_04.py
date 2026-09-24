"""One cooperative drive damping stage beneath the frozen stop/turn controller.

MRO: DriveRolling -> RollingResidual -> StopTurn -> DriveStage -> TurnYaw ->
D1WheelLeg. The original wheel PI and parent controller compute run once.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from drive_damping_math_04 import (
    GAIN_NSPM,
    POSITION_HIGH,
    POSITION_LOW,
    TORQUE_LIMIT,
    VELOCITY_LIMIT,
    drive_increment,
    protect_requested,
)
from scripts.d1_rolling_residual_env import RollingResidualCompositionController
from scripts.d1_stop_leg_damping import STOP_LEG_DAMPING_NSPM
from scripts.d1_turn_yaw_authority import TurnYawAuthorityController
from wheel_legged_control.d1.model import (
    JOINT_POSITION_HIGH,
    JOINT_POSITION_LOW,
    JOINT_TORQUE_LIMIT,
    JOINT_VELOCITY_LIMIT,
)
from wheel_legged_control.d1.wheel_leg_controller import D1WheelLegResult

DRIVE_DAMPING_RECORD_SCHEMA = "d1-drive-jx-leg-damping-record-v1"
DRIVE_ROLLING_CONTROL_SCHEMA = "d1-rolling-shared-leg-drive-jx-stop-turn-v1"
DRIVE_ROLLING_TASK_SCHEMA = "d1-rolling-residual-drive-jx-transfer-v1"


def assert_frozen_arithmetic_identity() -> None:
    """The pure helper must use the exact original limits and imported gain."""
    if (GAIN_NSPM != STOP_LEG_DAMPING_NSPM
            or not np.array_equal(TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
            or not np.array_equal(VELOCITY_LIMIT, JOINT_VELOCITY_LIMIT)
            or not np.array_equal(POSITION_LOW, JOINT_POSITION_LOW)
            or not np.array_equal(POSITION_HIGH, JOINT_POSITION_HIGH)):
        raise RuntimeError("new arithmetic differs from the frozen gain or protection arrays")


def _readonly(value: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must have finite shape {shape}")
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(shape)


@dataclass(frozen=True)
class DriveDampingRecord:
    schema: str
    active: bool
    raw_forward_mps: float
    servo_forward_mps: float
    control_time_s: float
    provider_state_sequence: int
    provider_state_age_s: float
    damping_nspm: float
    base_rotation: np.ndarray
    foot_jacobian: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    projected_jx: np.ndarray
    relative_forward_mps: np.ndarray
    delta_torque_nm: np.ndarray
    base_leg_pd_nm: np.ndarray
    base_requested_torque_nm: np.ndarray
    base_protected_torque_nm: np.ndarray
    total_requested_torque_nm: np.ndarray
    total_protected_torque_nm: np.ndarray
    actual_protected_increment_nm: np.ndarray
    raw_joint_power_w: float

    def __post_init__(self) -> None:
        if self.schema != DRIVE_DAMPING_RECORD_SCHEMA or type(self.active) is not bool:
            raise ValueError("driving damping record schema or gate differs")
        for name in ("raw_forward_mps", "servo_forward_mps", "control_time_s",
                     "provider_state_age_s", "damping_nspm", "raw_joint_power_w"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if type(self.provider_state_sequence) is not int:
            raise TypeError("provider_state_sequence must be an integer")
        for name, shape in (
            ("base_rotation", (3, 3)), ("foot_jacobian", (4, 3, 4)),
            ("joint_position", (16,)), ("joint_velocity", (16,)),
            ("projected_jx", (4, 3)), ("relative_forward_mps", (4,)),
            ("delta_torque_nm", (16,)), ("base_leg_pd_nm", (16,)),
            ("base_requested_torque_nm", (16,)),
            ("base_protected_torque_nm", (16,)),
            ("total_requested_torque_nm", (16,)),
            ("total_protected_torque_nm", (16,)),
            ("actual_protected_increment_nm", (16,)),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), shape, name))
        if np.any(self.delta_torque_nm[[3, 7, 11, 15]] != 0.0):
            raise ValueError("drive stage must not alter wheel torque")


@dataclass(frozen=True)
class DriveWheelLegResult(D1WheelLegResult):
    """Original result fields plus the actual intermediate drive-stage record."""

    drive_damping: DriveDampingRecord


class DriveDampingStageController(TurnYawAuthorityController):
    """Cooperative stage between frozen StopTurn and TurnYaw classes."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.last_drive_damping: DriveDampingRecord | None = None

    def reset(self) -> None:
        self.last_drive_damping = None
        super().reset()

    def compute(self, command, state, action, *, ground_height_m: float = 0.0):
        checked = self._checked_zero_action(action)
        binding = self._require_binding(command, state)
        raw_forward = float(binding.forward_velocity_mps)
        rotation = np.asarray(state.base_rotation, dtype=np.float64).copy()
        jacobian = np.asarray(state.foot_jacobian, dtype=np.float64).copy()
        position = np.asarray(state.joint_position, dtype=np.float64).copy()
        velocity = np.asarray(state.joint_velocity, dtype=np.float64).copy()
        increment = drive_increment(rotation, jacobian, velocity,
                                    raw_forward_mps=raw_forward)
        base_torque = super().compute(command, state, checked,
                                      ground_height_m=ground_height_m)
        base = self.last_result
        if not isinstance(base, D1WheelLegResult):
            raise TypeError("original parent controller did not publish its result")
        delta = increment["delta_torque_nm"]
        if increment["active"]:
            requested = base.requested_torque_nm + delta
            protected, limited = protect_requested(requested, position, velocity)
            leg_pd = base.leg_pd_nm + delta
        else:
            requested = base.requested_torque_nm
            protected = base.torque_nm
            limited = base.torque_limited
            leg_pd = base.leg_pd_nm
        actual_increment = protected - base.torque_nm
        record = DriveDampingRecord(
            schema=DRIVE_DAMPING_RECORD_SCHEMA,
            active=increment["active"],
            raw_forward_mps=raw_forward,
            servo_forward_mps=float(command.forward_velocity_mps),
            control_time_s=float(state.control_time_s),
            provider_state_sequence=int(state.sequence),
            provider_state_age_s=float(state.age_s),
            damping_nspm=STOP_LEG_DAMPING_NSPM,
            base_rotation=rotation,
            foot_jacobian=jacobian,
            joint_position=position,
            joint_velocity=velocity,
            projected_jx=increment["projected_jx"],
            relative_forward_mps=increment["relative_forward_mps"],
            delta_torque_nm=delta,
            base_leg_pd_nm=base.leg_pd_nm,
            base_requested_torque_nm=base.requested_torque_nm,
            base_protected_torque_nm=base.torque_nm,
            total_requested_torque_nm=requested,
            total_protected_torque_nm=protected,
            actual_protected_increment_nm=actual_increment,
            raw_joint_power_w=increment["raw_joint_power_w"],
        )
        payload = {field.name: getattr(base, field.name) for field in fields(D1WheelLegResult)}
        payload.update(leg_pd_nm=leg_pd, requested_torque_nm=requested,
                       torque_nm=protected, torque_limited=limited)
        result = DriveWheelLegResult(**payload, drive_damping=record)
        self.last_result = result
        self.last_drive_damping = record
        return protected.copy() if increment["active"] else base_torque


class DriveRollingController(RollingResidualCompositionController,
                             DriveDampingStageController):
    """Exact cooperative composition of shared-leg, stop, drive and turn stages."""

    def __init__(self, **kwargs) -> None:
        assert_frozen_arithmetic_identity()
        super().__init__(**kwargs)
        if not self.enabled:
            raise ValueError("the transferred drive candidate requires enabled composition")
        self.control_schema = DRIVE_ROLLING_CONTROL_SCHEMA


EXPECTED_MRO = (
    "DriveRollingController", "RollingResidualCompositionController",
    "StopTurnCompositionController", "DriveDampingStageController",
    "TurnYawAuthorityController", "D1WheelLegController",
)


def assert_cooperative_mro() -> tuple[str, ...]:
    names = tuple(cls.__name__ for cls in DriveRollingController.__mro__[:6])
    if names != EXPECTED_MRO:
        raise RuntimeError(f"drive controller MRO differs from the one reviewed: {names}")
    return names
