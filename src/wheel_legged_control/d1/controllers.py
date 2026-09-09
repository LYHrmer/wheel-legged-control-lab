"""Low-level joint and virtual-model control for the full-body D1 plant."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact_allocation import (
    D1AllocationStatus,
    D1ContactAllocationRequest,
    D1ContactAllocator,
    D1WrenchTrackingStatus,
)
from .model import (
    D1_JOINT_NAMES,
    JOINT_TORQUE_LIMIT,
    LEG_PREFIXES,
    NOMINAL_JOINT_POSITION,
    D1Plant,
)
from .state_estimation import D1StateEstimate


def _heading_rotation(yaw_rad: float) -> np.ndarray:
    """Map yaw-aligned horizontal-frame vectors into the world frame."""

    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    return np.asarray(
        ((cos_yaw, -sin_yaw, 0.0), (sin_yaw, cos_yaw, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class D1Command:
    """Body command; height and vertical velocity are world-z references.

    Nonzero vertical-velocity feedforward is supported by inverse dynamics only.
    Legacy VMC paths reject it rather than silently ignoring a requested motion.
    """

    forward_velocity_mps: float = 0.0
    yaw_rate_rps: float = 0.0
    base_height_m: float = 0.455
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    base_vertical_velocity_mps: float = 0.0


@dataclass(frozen=True)
class D1ControlBreakdown:
    """Auditable torque decomposition returned by :class:`D1VMCController`."""

    leg_pd_nm: np.ndarray
    support_nm: np.ndarray
    wheel_velocity_nm: np.ndarray
    high_level_nm: np.ndarray
    total_nm: np.ndarray
    support_force_n: np.ndarray
    contact_force_world_n: np.ndarray
    wrench_reference_position_world_m: np.ndarray
    desired_wrench_world: np.ndarray
    achieved_wrench_world: np.ndarray
    allocation_status: D1AllocationStatus | None
    allocation_wrench_tracking_status: D1WrenchTrackingStatus | None
    allocation_status_reason: str
    allocation_solve_ms: float
    allocation_constraint_violation: float


class D1VMCController:
    """Joint PD plus virtual-model support and wheel-speed control.

    The controller deliberately exposes two residual high-level channels:
    longitudinal ground force and total vertical support-force correction.
    LQR, MPC, or a learned residual can share those same bounded channels.
    """

    def __init__(
        self,
        plant: D1Plant,
        *,
        contact_allocator: D1ContactAllocator | None = None,
    ) -> None:
        # Keep only nominal model constants.  Runtime feedback must arrive in a
        # D1StateEstimate so randomized MuJoCo truth cannot leak into control.
        self.total_mass_kg = plant.nominal_total_mass_kg
        self.nominal_base_com_offset_body_m = plant.model.body_ipos[plant.base_body_id].copy()
        self.nominal_base_com_offset_body_m.setflags(write=False)
        self.gravity_mps2 = abs(float(plant.model.opt.gravity[2]))
        self.wheel_radius_m = float(plant.wheel_radius_m)
        self.leg_kp = np.tile((40.0, 40.0, 40.0, 0.0), len(LEG_PREFIXES))
        self.leg_kd = np.tile((1.5, 1.5, 1.5, 0.0), len(LEG_PREFIXES))
        self.wheel_velocity_gain = 0.55
        self.yaw_rate_gain = 5.0
        self.contact_allocator = contact_allocator
        self.height_kp = 900.0
        self.height_kd = 180.0
        self.roll_kp = 180.0
        self.roll_kd = 24.0
        self.pitch_kp = 180.0
        self.pitch_kd = 24.0
        self._leg_indices = {
            leg: np.asarray(
                [index for index, name in enumerate(D1_JOINT_NAMES) if name.startswith(leg)],
                dtype=np.int32,
            )
            for leg in LEG_PREFIXES
        }
        self._wheel_indices = np.asarray((3, 7, 11, 15), dtype=np.int32)
        zeros_torque = np.zeros(len(D1_JOINT_NAMES), dtype=np.float64)
        self._last = D1ControlBreakdown(
            leg_pd_nm=zeros_torque.copy(),
            support_nm=zeros_torque.copy(),
            wheel_velocity_nm=zeros_torque.copy(),
            high_level_nm=zeros_torque.copy(),
            total_nm=zeros_torque.copy(),
            support_force_n=np.zeros(len(LEG_PREFIXES), dtype=np.float64),
            contact_force_world_n=np.zeros((len(LEG_PREFIXES), 3), dtype=np.float64),
            wrench_reference_position_world_m=np.zeros(3, dtype=np.float64),
            desired_wrench_world=np.zeros(6, dtype=np.float64),
            achieved_wrench_world=np.zeros(6, dtype=np.float64),
            allocation_status=None,
            allocation_wrench_tracking_status=None,
            allocation_status_reason="legacy",
            allocation_solve_ms=0.0,
            allocation_constraint_violation=0.0,
        )

    @property
    def last_breakdown(self) -> D1ControlBreakdown:
        return self._last

    def reset(self) -> None:
        if self.contact_allocator is not None:
            self.contact_allocator.reset()
        zeros = np.zeros(len(D1_JOINT_NAMES), dtype=np.float64)
        self._last = D1ControlBreakdown(
            leg_pd_nm=zeros.copy(),
            support_nm=zeros.copy(),
            wheel_velocity_nm=zeros.copy(),
            high_level_nm=zeros.copy(),
            total_nm=zeros.copy(),
            support_force_n=np.zeros(len(LEG_PREFIXES), dtype=np.float64),
            contact_force_world_n=np.zeros((len(LEG_PREFIXES), 3), dtype=np.float64),
            wrench_reference_position_world_m=np.zeros(3, dtype=np.float64),
            desired_wrench_world=np.zeros(6, dtype=np.float64),
            achieved_wrench_world=np.zeros(6, dtype=np.float64),
            allocation_status=None,
            allocation_wrench_tracking_status=None,
            allocation_status_reason="legacy",
            allocation_solve_ms=0.0,
            allocation_constraint_violation=0.0,
        )

    def _desired_support_wrench(
        self,
        state: D1StateEstimate,
        command: D1Command,
        vertical_force_offset_n: float,
    ) -> tuple[float, np.ndarray]:
        linear_velocity = state.base_linear_velocity_world
        roll, pitch, yaw = state.base_rpy
        heading_rotation = _heading_rotation(float(yaw))
        angular_velocity_heading = heading_rotation.T @ state.base_angular_velocity_world
        total_upward_force = (
            self.total_mass_kg * self.gravity_mps2
            + self.height_kp * (command.base_height_m - state.base_position[2])
            - self.height_kd * linear_velocity[2]
            + vertical_force_offset_n
        )
        desired_moment_heading = np.asarray(
            (
                self.roll_kp * (command.roll_rad - roll)
                - self.roll_kd * angular_velocity_heading[0],
                self.pitch_kp * (command.pitch_rad - pitch)
                - self.pitch_kd * angular_velocity_heading[1],
                0.0,
            ),
            dtype=np.float64,
        )
        return total_upward_force, heading_rotation @ desired_moment_heading

    def _support_torque(
        self,
        state: D1StateEstimate,
        command: D1Command,
        vertical_force_offset_n: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        total_upward_force, desired_moment_world = self._desired_support_wrench(
            state,
            command,
            vertical_force_offset_n,
        )
        mass = self.total_mass_kg

        wheel_positions = state.foot_offset_world
        allocation = np.vstack(
            (
                np.ones(len(LEG_PREFIXES)),
                wheel_positions[:, 1],
                -wheel_positions[:, 0],
            )
        )
        desired_wrench = np.asarray(
            (total_upward_force, desired_moment_world[0], desired_moment_world[1]),
            dtype=np.float64,
        )
        upward_forces = np.linalg.lstsq(allocation, desired_wrench, rcond=None)[0]
        upward_forces = np.clip(upward_forces, 0.0, mass * 9.81 * 0.65)

        torque = np.zeros(len(D1_JOINT_NAMES), dtype=np.float64)
        for leg_index, (leg, force_n) in enumerate(zip(LEG_PREFIXES, upward_forces, strict=True)):
            indices = self._leg_indices[leg]
            foot_force = np.asarray((0.0, 0.0, -force_n), dtype=np.float64)
            torque[indices] += state.foot_jacobian[leg_index].T @ foot_force
        torque[self._wheel_indices] = 0.0
        return torque, upward_forces

    def compute(
        self,
        command: D1Command,
        state: D1StateEstimate,
        *,
        longitudinal_force_n: float = 0.0,
        vertical_force_offset_n: float = 0.0,
    ) -> np.ndarray:
        """Compute all 16 commanded joint torques.

        ``longitudinal_force_n`` and ``vertical_force_offset_n`` are total-body
        effort corrections.  Positive longitudinal force drives forward;
        positive vertical correction asks the wheels to support more load.
        """

        if command.base_vertical_velocity_mps != 0.0:
            raise ValueError("vertical velocity feedforward requires the inverse dynamics controller")

        joint_position = state.joint_position
        joint_velocity = state.joint_velocity
        leg_pd = self.leg_kp * (NOMINAL_JOINT_POSITION - joint_position)
        leg_pd -= self.leg_kd * joint_velocity
        leg_pd[self._wheel_indices] = 0.0

        wheel_velocity = np.zeros(len(D1_JOINT_NAMES), dtype=np.float64)
        heading_rotation = _heading_rotation(float(state.base_rpy[2]))
        wheel_positions_heading = state.foot_offset_world @ heading_rotation
        for leg_index, index in enumerate(self._wheel_indices):
            lateral_position = wheel_positions_heading[leg_index, 1]
            target_linear_velocity = (
                command.forward_velocity_mps - command.yaw_rate_rps * lateral_position
            )
            target_wheel_velocity = target_linear_velocity / self.wheel_radius_m
            wheel_velocity[index] = self.wheel_velocity_gain * (
                target_wheel_velocity - joint_velocity[index]
            )
        local_angular_velocity = state.base_angular_velocity_body
        yaw_torque = float(
            np.clip(
                self.yaw_rate_gain * (command.yaw_rate_rps - local_angular_velocity[2]),
                -4.0,
                4.0,
            )
        )
        wheel_velocity[np.asarray((0, 8)) + 3] -= yaw_torque
        wheel_velocity[np.asarray((4, 12)) + 3] += yaw_torque

        if self.contact_allocator is None:
            support, upward_forces = self._support_torque(
                state,
                command,
                vertical_force_offset_n,
            )
            high_level = np.zeros(len(D1_JOINT_NAMES), dtype=np.float64)
            high_level[self._wheel_indices] = (
                longitudinal_force_n * self.wheel_radius_m / len(LEG_PREFIXES)
            )
            total = np.clip(
                leg_pd + support + wheel_velocity + high_level,
                -JOINT_TORQUE_LIMIT,
                JOINT_TORQUE_LIMIT,
            )
            contact_force = np.zeros((len(LEG_PREFIXES), 3), dtype=np.float64)
            heading_rotation = _heading_rotation(float(state.base_rpy[2]))
            contact_force[:, 2] = upward_forces
            contact_force += (
                heading_rotation @ np.asarray((longitudinal_force_n, 0.0, 0.0))
            ) / len(LEG_PREFIXES)
            desired_upward, desired_moment = self._desired_support_wrench(
                state,
                command,
                vertical_force_offset_n,
            )
            desired_force = (
                heading_rotation @ np.asarray((longitudinal_force_n, 0.0, 0.0))
                + np.asarray((0.0, 0.0, desired_upward))
            )
            desired_wrench = np.concatenate((desired_force, desired_moment))
            achieved_wrench = np.concatenate(
                (
                    np.sum(contact_force, axis=0),
                    np.sum(np.cross(state.foot_offset_world, contact_force), axis=0),
                )
            )
            allocation_status = None
            allocation_wrench_tracking_status = None
            allocation_status_reason = "legacy"
            allocation_solve_ms = 0.0
            allocation_violation = 0.0
            wrench_reference = state.base_position
        else:
            desired_upward, desired_moment = self._desired_support_wrench(
                state,
                command,
                vertical_force_offset_n,
            )
            heading_rotation = _heading_rotation(float(state.base_rpy[2]))
            desired_force = (
                heading_rotation @ np.asarray((longitudinal_force_n, 0.0, 0.0))
                + np.asarray((0.0, 0.0, desired_upward))
            )
            allocation = self.contact_allocator.solve(
                D1ContactAllocationRequest(
                    state=state,
                    desired_force_world_n=desired_force,
                    desired_moment_world_nm=desired_moment,
                    committed_torque_nm=leg_pd + wheel_velocity,
                )
            )
            contact_force = allocation.contact_force_world_n
            normal_force = np.einsum(
                "ij,ij->i",
                contact_force,
                state.wheel_contact_normal,
            )
            normal_component = normal_force[:, None] * state.wheel_contact_normal
            support = np.zeros(len(D1_JOINT_NAMES), dtype=np.float64)
            for leg_index in range(len(LEG_PREFIXES)):
                joint_slice = slice(4 * leg_index, 4 * (leg_index + 1))
                support[joint_slice] = (
                    -state.wheel_contact_jacobian[leg_index].T
                    @ normal_component[leg_index]
                )
            high_level = allocation.contact_torque_nm - support
            upward_forces = normal_force
            total = allocation.command_torque_nm
            desired_wrench = allocation.desired_wrench_world
            achieved_wrench = allocation.achieved_wrench_world
            allocation_status = allocation.status
            allocation_wrench_tracking_status = allocation.wrench_tracking_status
            allocation_status_reason = allocation.status_reason
            allocation_solve_ms = allocation.solve_ms
            allocation_violation = allocation.max_constraint_violation
            wrench_reference = allocation.wrench_reference_position_world_m
        self._last = D1ControlBreakdown(
            leg_pd_nm=leg_pd.copy(),
            support_nm=support.copy(),
            wheel_velocity_nm=wheel_velocity.copy(),
            high_level_nm=high_level.copy(),
            total_nm=total.copy(),
            support_force_n=upward_forces.copy(),
            contact_force_world_n=contact_force.copy(),
            wrench_reference_position_world_m=wrench_reference.copy(),
            desired_wrench_world=desired_wrench.copy(),
            achieved_wrench_world=achieved_wrench.copy(),
            allocation_status=allocation_status,
            allocation_wrench_tracking_status=allocation_wrench_tracking_status,
            allocation_status_reason=allocation_status_reason,
            allocation_solve_ms=allocation_solve_ms,
            allocation_constraint_violation=allocation_violation,
        )
        return total
