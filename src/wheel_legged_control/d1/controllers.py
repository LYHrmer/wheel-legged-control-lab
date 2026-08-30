"""Low-level joint and virtual-model control for the full-body D1 plant."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    """Body-level command consumed by the low-level controller."""

    forward_velocity_mps: float = 0.0
    yaw_rate_rps: float = 0.0
    base_height_m: float = 0.455
    roll_rad: float = 0.0
    pitch_rad: float = 0.0


@dataclass(frozen=True)
class D1ControlBreakdown:
    """Auditable torque decomposition returned by :class:`D1VMCController`."""

    leg_pd_nm: np.ndarray
    support_nm: np.ndarray
    wheel_velocity_nm: np.ndarray
    high_level_nm: np.ndarray
    total_nm: np.ndarray
    support_force_n: np.ndarray


class D1VMCController:
    """Joint PD plus virtual-model support and wheel-speed control.

    The controller deliberately exposes two residual high-level channels:
    longitudinal ground force and total vertical support-force correction.
    LQR, MPC, or a learned residual can share those same bounded channels.
    """

    def __init__(self, plant: D1Plant) -> None:
        # Keep only nominal model constants.  Runtime feedback must arrive in a
        # D1StateEstimate so randomized MuJoCo truth cannot leak into control.
        self.total_mass_kg = plant.nominal_total_mass_kg
        self.gravity_mps2 = abs(float(plant.model.opt.gravity[2]))
        self.wheel_radius_m = float(plant.wheel_radius_m)
        self.leg_kp = np.tile((40.0, 40.0, 40.0, 0.0), len(LEG_PREFIXES))
        self.leg_kd = np.tile((1.5, 1.5, 1.5, 0.0), len(LEG_PREFIXES))
        self.wheel_velocity_gain = 0.55
        self.yaw_rate_gain = 5.0
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
        self._last = D1ControlBreakdown(
            *(np.zeros(len(D1_JOINT_NAMES), dtype=np.float64) for _ in range(5)),
            np.zeros(len(LEG_PREFIXES), dtype=np.float64),
        )

    @property
    def last_breakdown(self) -> D1ControlBreakdown:
        return self._last

    def reset(self) -> None:
        zeros = np.zeros(len(D1_JOINT_NAMES), dtype=np.float64)
        self._last = D1ControlBreakdown(
            zeros.copy(),
            zeros.copy(),
            zeros.copy(),
            zeros.copy(),
            zeros.copy(),
            np.zeros(len(LEG_PREFIXES), dtype=np.float64),
        )

    def _support_torque(
        self,
        state: D1StateEstimate,
        command: D1Command,
        vertical_force_offset_n: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        linear_velocity = state.base_linear_velocity_world
        roll, pitch, yaw = state.base_rpy
        heading_rotation = _heading_rotation(float(yaw))
        angular_velocity_heading = heading_rotation.T @ state.base_angular_velocity_world
        mass = self.total_mass_kg
        total_upward_force = (
            mass * self.gravity_mps2
            + self.height_kp * (command.base_height_m - state.base_position[2])
            - self.height_kd * linear_velocity[2]
            + vertical_force_offset_n
        )
        desired_roll_moment_heading = (
            self.roll_kp * (command.roll_rad - roll) - self.roll_kd * angular_velocity_heading[0]
        )
        desired_pitch_moment_heading = (
            self.pitch_kp * (command.pitch_rad - pitch)
            - self.pitch_kd * angular_velocity_heading[1]
        )
        desired_moment_world = heading_rotation @ np.asarray(
            (desired_roll_moment_heading, desired_pitch_moment_heading, 0.0),
            dtype=np.float64,
        )

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

        joint_position = state.joint_position
        joint_velocity = state.joint_velocity
        leg_pd = self.leg_kp * (NOMINAL_JOINT_POSITION - joint_position)
        leg_pd -= self.leg_kd * joint_velocity
        leg_pd[self._wheel_indices] = 0.0

        support, upward_forces = self._support_torque(
            state,
            command,
            vertical_force_offset_n,
        )

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

        high_level = np.zeros(len(D1_JOINT_NAMES), dtype=np.float64)
        high_level[self._wheel_indices] = (
            longitudinal_force_n * self.wheel_radius_m / len(LEG_PREFIXES)
        )
        total = np.clip(
            leg_pd + support + wheel_velocity + high_level,
            -JOINT_TORQUE_LIMIT,
            JOINT_TORQUE_LIMIT,
        )
        self._last = D1ControlBreakdown(
            leg_pd.copy(),
            support.copy(),
            wheel_velocity.copy(),
            high_level.copy(),
            total.copy(),
            upward_forces.copy(),
        )
        return total
