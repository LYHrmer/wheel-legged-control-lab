"""Low-level joint and virtual-model control for the full-body D1 plant."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .model import (
    D1_JOINT_NAMES,
    JOINT_TORQUE_LIMIT,
    LEG_PREFIXES,
    NOMINAL_JOINT_POSITION,
    D1Plant,
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
        self.plant = plant
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
        command: D1Command,
        vertical_force_offset_n: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        linear_velocity, angular_velocity = self.plant.base_velocity(local=False)
        roll, pitch, _ = self.plant.base_rpy
        mass = float(self.plant.model.body_mass.sum())
        total_upward_force = (
            mass * abs(float(self.plant.model.opt.gravity[2]))
            + self.height_kp * (command.base_height_m - self.plant.base_position[2])
            - self.height_kd * linear_velocity[2]
            + vertical_force_offset_n
        )
        desired_roll_moment = (
            self.roll_kp * (command.roll_rad - roll)
            - self.roll_kd * angular_velocity[0]
        )
        desired_pitch_moment = (
            self.pitch_kp * (command.pitch_rad - pitch)
            - self.pitch_kd * angular_velocity[1]
        )

        wheel_positions = np.asarray(
            [
                self.plant.data.xpos[
                    mujoco.mj_name2id(
                        self.plant.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        f"{leg}_foot",
                    )
                ]
                - self.plant.base_position
                for leg in LEG_PREFIXES
            ]
        )
        allocation = np.vstack(
            (
                np.ones(len(LEG_PREFIXES)),
                wheel_positions[:, 1],
                -wheel_positions[:, 0],
            )
        )
        desired_wrench = np.asarray(
            (total_upward_force, desired_roll_moment, desired_pitch_moment),
            dtype=np.float64,
        )
        upward_forces = np.linalg.lstsq(allocation, desired_wrench, rcond=None)[0]
        upward_forces = np.clip(upward_forces, 0.0, mass * 9.81 * 0.65)

        torque = np.zeros(len(D1_JOINT_NAMES), dtype=np.float64)
        for leg, force_n in zip(LEG_PREFIXES, upward_forces, strict=True):
            body_id = mujoco.mj_name2id(
                self.plant.model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot"
            )
            jacobian = np.zeros((3, self.plant.model.nv), dtype=np.float64)
            rotation_jacobian = np.zeros_like(jacobian)
            mujoco.mj_jacBodyCom(
                self.plant.model,
                self.plant.data,
                jacobian,
                rotation_jacobian,
                body_id,
            )
            indices = self._leg_indices[leg]
            foot_force = np.asarray((0.0, 0.0, -force_n), dtype=np.float64)
            torque[indices] += (
                jacobian[:, self.plant.dof_addresses[indices]].T @ foot_force
            )
        torque[self._wheel_indices] = 0.0
        return torque, upward_forces

    def compute(
        self,
        command: D1Command,
        *,
        longitudinal_force_n: float = 0.0,
        vertical_force_offset_n: float = 0.0,
    ) -> np.ndarray:
        """Compute all 16 commanded joint torques.

        ``longitudinal_force_n`` and ``vertical_force_offset_n`` are total-body
        effort corrections.  Positive longitudinal force drives forward;
        positive vertical correction asks the wheels to support more load.
        """

        joint_position = self.plant.joint_position
        joint_velocity = self.plant.joint_velocity
        leg_pd = self.leg_kp * (NOMINAL_JOINT_POSITION - joint_position)
        leg_pd -= self.leg_kd * joint_velocity
        leg_pd[self._wheel_indices] = 0.0

        support, upward_forces = self._support_torque(command, vertical_force_offset_n)

        wheel_velocity = np.zeros(len(D1_JOINT_NAMES), dtype=np.float64)
        for index in self._wheel_indices:
            body_id = int(self.plant.model.jnt_bodyid[self.plant.joint_ids[index]])
            lateral_position = self.plant.data.xpos[body_id, 1] - self.plant.base_position[1]
            target_linear_velocity = (
                command.forward_velocity_mps - command.yaw_rate_rps * lateral_position
            )
            target_wheel_velocity = target_linear_velocity / self.plant.wheel_radius_m
            wheel_velocity[index] = self.wheel_velocity_gain * (
                target_wheel_velocity - joint_velocity[index]
            )
        _, local_angular_velocity = self.plant.base_velocity(local=True)
        yaw_torque = float(
            np.clip(
                self.yaw_rate_gain
                * (command.yaw_rate_rps - local_angular_velocity[2]),
                -4.0,
                4.0,
            )
        )
        wheel_velocity[np.asarray((0, 8)) + 3] -= yaw_torque
        wheel_velocity[np.asarray((4, 12)) + 3] += yaw_torque

        high_level = np.zeros(len(D1_JOINT_NAMES), dtype=np.float64)
        high_level[self._wheel_indices] = (
            longitudinal_force_n * self.plant.wheel_radius_m / len(LEG_PREFIXES)
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
