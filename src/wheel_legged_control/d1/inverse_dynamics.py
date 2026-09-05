"""Task-space inverse dynamics on an independent nominal D1 model.

This opt-in controller does not read the running simulator's dynamics or contact
forces. Joint dry friction is unmodeled; nominal damping and armature are included.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import mujoco
import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation

from .controllers import D1Command
from .model import JOINT_TORQUE_LIMIT, NOMINAL_JOINT_POSITION, D1Plant
from .state_estimation import D1StateEstimate


@dataclass(frozen=True, slots=True)
class D1InverseDynamicsResult:
    status: str
    solver_status: str
    command_torque_nm: np.ndarray
    generalized_acceleration: np.ndarray
    contact_force_world_n: np.ndarray
    dynamics_residual_max: float
    constraint_violation_max: float
    solve_ms: float

    def __post_init__(self) -> None:
        for name in (
            "command_torque_nm", "generalized_acceleration", "contact_force_world_n"
        ):
            value = np.ascontiguousarray(getattr(self, name), dtype=np.float64)
            object.__setattr__(
                self, name, np.frombuffer(value.tobytes(), dtype=np.float64).reshape(value.shape)
            )


class D1InverseDynamicsController:
    """Compute 16 torques from a command and a control-state snapshot.

    ``last_result`` is None before the first compute and after reset. The input
    snapshot's legacy linear velocity refers to base inertial COM; it is shifted
    to the freejoint origin using the nominal inertial offset before calculation.
    """

    def __init__(
        self,
        *,
        control_dt: float = 0.01,
        torque_limit_nm: np.ndarray = JOINT_TORQUE_LIMIT,
    ) -> None:
        self._nominal = D1Plant(control_dt=control_dt)
        self.control_dt = self._nominal.control_dt
        self.torque_limit_nm = np.asarray(torque_limit_nm, dtype=np.float64).copy()
        if self.torque_limit_nm.shape != (16,) or not np.all(
            np.isfinite(self.torque_limit_nm) & (self.torque_limit_nm > 0.0)
        ):
            raise ValueError("torque_limit_nm must contain 16 positive finite limits")
        self._identity = np.eye(22)
        self._actuation = np.zeros((22, 16))
        self._actuation[self._nominal.dof_addresses, np.arange(16)] = 1.0
        self.last_result: D1InverseDynamicsResult | None = None

    def reset(self) -> None:
        self.last_result = None

    def _reconstruct(self, state: D1StateEstimate) -> tuple[np.ndarray, np.ndarray]:
        plant = self._nominal
        model, data = plant.model, plant.data
        data.qpos[:3] = state.base_position
        mujoco.mju_mat2Quat(data.qpos[3:7], state.base_rotation.ravel())
        data.qpos[plant.qpos_addresses] = state.joint_position
        com_offset = state.base_rotation @ model.body_ipos[plant.base_body_id]
        data.qvel[:3] = state.base_linear_velocity_world - np.cross(
            state.base_angular_velocity_world, com_offset
        )
        data.qvel[3:6] = state.base_angular_velocity_body
        data.qvel[plant.dof_addresses] = state.joint_velocity
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_fwdVelocity(model, data)
        # mj_fullM changed its Python signature; mj_mulM is stable across the
        # supported versions and already includes the actuator armature.
        columns = []
        for column in self._identity:
            result = np.empty(22)
            mujoco.mj_mulM(model, data, result, column)
            columns.append(result)
        return np.column_stack(columns), data.qfrc_bias - data.qfrc_passive

    def _task_cost(
        self, command: D1Command, state: D1StateEstimate
    ) -> tuple[np.ndarray, np.ndarray]:
        rotation = state.base_rotation
        yaw = float(state.base_rpy[2])
        forward = np.asarray((np.cos(yaw), np.sin(yaw), 0.0))
        velocity = self._nominal.data.qvel[:3]
        desired_linear = 3.0 * (command.forward_velocity_mps * forward - velocity)
        desired_linear[2] = (
            80.0 * (command.base_height_m - state.base_position[2]) - 18.0 * velocity[2]
        )
        desired_rotation = Rotation.from_euler(
            "xyz", (command.roll_rad, command.pitch_rad, yaw)
        ).as_matrix()
        rotation_error = Rotation.from_matrix(desired_rotation @ rotation.T).as_rotvec()
        desired_angular = (
            80.0 * rotation_error
            + 16.0 * (
                np.asarray((0.0, 0.0, command.yaw_rate_rps))
                - state.base_angular_velocity_world
            )
        )
        task_matrix = np.zeros((22, 50))
        task_matrix[:3, :3] = np.eye(3)
        task_matrix[3:6, 3:6] = rotation
        leg_indices = np.asarray([i for i in range(16) if i % 4 != 3])
        task_matrix[6:18, self._nominal.dof_addresses[leg_indices]] = np.eye(12)
        wheel_indices = np.asarray((3, 7, 11, 15))
        task_matrix[18:, self._nominal.dof_addresses[wheel_indices]] = np.eye(4)
        posture = (
            30.0 * (NOMINAL_JOINT_POSITION[leg_indices] - state.joint_position[leg_indices])
            - 6.0 * state.joint_velocity[leg_indices]
        )
        # Unconstrained airborne wheel accelerations otherwise act as reaction
        # wheels for the attitude task and can run to extreme speeds on landing.
        wheel_acceleration = 10.0 * (
            command.forward_velocity_mps / self._nominal.wheel_radius_m
            - state.joint_velocity[wheel_indices]
        )
        target = np.concatenate((desired_linear, desired_angular, posture, wheel_acceleration))
        # Acceleration tasks use 1 m/s² and 1 rad/s² reference scales; the
        # dimensionless weights express priorities, not a mixed-unit norm.
        weights = np.asarray((1.0, 1.0, 5.0, 10.0, 10.0, 2.0, *([0.05] * 16)))
        weighted = task_matrix * np.sqrt(weights)[:, None]
        target = target * np.sqrt(weights)
        regularization = np.concatenate((
            np.full(22, 1e-7),
            np.full(12, 1e-4 / (self._nominal.nominal_total_mass_kg * 9.81) ** 2),
            1e-3 / self.torque_limit_nm**2,
        ))
        return weighted.T @ weighted + np.diag(regularization), -weighted.T @ target

    def _contacts(
        self, state: D1StateEstimate
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        plant = self._nominal
        jacobians = np.zeros((4, 3, 22))
        frames = np.zeros((4, 3, 3))
        biases = np.zeros((4, 3))
        frame_rates = np.zeros((4, 3, 3))
        for wheel in np.flatnonzero(state.wheel_contact):
            normal = state.wheel_contact_normal[wheel]
            if np.linalg.norm(normal) < 1e-8:
                raise ValueError("active wheel contact normal must be nonzero")
            normal = normal / np.linalg.norm(normal)
            if normal[2] < 0.999:
                raise ValueError("contact normal outside the flat-ground prototype scope")
            axis = plant.data.xaxis[plant.joint_ids[4 * wheel + 3]]
            projection = normal - axis * (axis @ normal)
            if np.linalg.norm(projection) < 1e-6:
                raise ValueError("wheel axis is parallel to the contact normal")
            offset = -plant.wheel_radius_m * projection / np.linalg.norm(projection)
            point = plant.data.xpos[plant.wheel_body_ids_by_leg[wheel]] + offset
            rotational = np.zeros((3, 22))
            mujoco.mj_jac(
                plant.model, plant.data, jacobians[wheel], rotational,
                point, plant.wheel_body_ids_by_leg[wheel],
            )
            omega = rotational @ plant.data.qvel
            axis_dot = np.cross(omega, axis)
            projection_dot = -axis_dot * (axis @ normal) - axis * (axis_dot @ normal)
            radial = projection / np.linalg.norm(projection)
            offset_dot = (-plant.wheel_radius_m / np.linalg.norm(projection)) * (
                np.eye(3) - np.outer(radial, radial)
            ) @ projection_dot
            jacobian_dot = np.zeros((3, 22))
            mujoco.mj_jacDot(
                plant.model, plant.data, jacobian_dot, None,
                point, plant.wheel_body_ids_by_leg[wheel],
            )
            # mj_jacDot follows a body-fixed material point. The geometric rim
            # point migrates during rolling; remove the material centripetal term.
            biases[wheel] = jacobian_dot @ plant.data.qvel + np.cross(
                omega, offset_dot - np.cross(omega, offset)
            )
            rolling = np.cross(axis, normal)
            rolling_norm = np.linalg.norm(rolling)
            rolling /= rolling_norm
            frames[wheel] = np.stack((rolling, np.cross(normal, rolling), normal))
            rolling_dot = (np.eye(3) - np.outer(rolling, rolling)) @ np.cross(
                axis_dot, normal
            ) / rolling_norm
            frame_rates[wheel] = np.stack((
                rolling_dot, np.cross(normal, rolling_dot), np.zeros(3)
            ))
        return jacobians, frames, biases, frame_rates

    def compute(self, command: D1Command, state: D1StateEstimate) -> np.ndarray:
        self.last_result = None
        if not np.all(np.isfinite((
            command.forward_velocity_mps, command.yaw_rate_rps,
            command.base_height_m, command.roll_rad, command.pitch_rad,
        ))):
            raise ValueError("command values must be finite")
        if command.yaw_rate_rps != 0.0:
            raise ValueError("nonzero yaw commands are outside this straight-line prototype")
        started = perf_counter()
        mass, bias = self._reconstruct(state)
        hessian, gradient = self._task_cost(command, state)
        jacobians, frames, rolling_bias, frame_rates = self._contacts(state)
        dynamics = np.zeros((22, 50))
        dynamics[:, :22] = mass
        dynamics[:, 22:34] = -jacobians.reshape(12, 22).T
        dynamics[:, 34:] = -self._actuation
        bounds = np.zeros((28, 50))
        bounds[:, 22:] = np.eye(28)
        force_bound = np.repeat(np.where(state.wheel_contact, np.inf, 0.0), 3)
        lower = np.concatenate((-bias, -force_bound, -self.torque_limit_nm))
        upper = np.concatenate((-bias, force_bound, self.torque_limit_nm))
        rows = [dynamics, bounds]
        for wheel in np.flatnonzero(state.wheel_contact):
            frame = frames[wheel]
            # A conservative friction diamond, |ft| + |fl| <= mu fn.
            friction = np.asarray(((1, 1, -.55), (1, -1, -.55),
                                   (-1, 1, -.55), (-1, -1, -.55), (0, 0, 1)))
            force_rows = np.zeros((5, 50))
            force_rows[:, 22 + 3 * wheel:25 + 3 * wheel] = friction @ frame
            rows.append(force_rows)
            lower = np.concatenate((lower, [-np.inf] * 4 + [0.0]))
            upper = np.concatenate((upper, [0.0] * 4 + [
                .65 * self._nominal.nominal_total_mass_kg * 9.81
            ]))
            contact_rows = np.zeros((2, 50))
            contact_rows[:, :22] = frame[[0, 2]] @ jacobians[wheel]
            rows.append(contact_rows)
            velocity = jacobians[wheel] @ self._nominal.data.qvel
            target = frame @ (-rolling_bias[wheel] - 20.0 * velocity)
            target -= frame_rates[wheel] @ velocity
            lower = np.concatenate((lower, target[[0, 2]]))
            upper = np.concatenate((upper, target[[0, 2]]))
            # Lateral slip is penalized, not forbidden: four-wheel skid steering
            # cannot satisfy four hard lateral no-slip constraints in general.
            lateral = np.zeros(50)
            lateral[:22] = frame[1] @ jacobians[wheel]
            hessian += np.outer(lateral, lateral)
            gradient -= lateral * target[1]
        constraints = np.vstack(rows)
        solver = osqp.OSQP()
        solver.setup(
            P=sparse.csc_matrix(np.triu(hessian)), q=gradient,
            A=sparse.csc_matrix(constraints), l=lower, u=upper,
            verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=4000,
            polishing=True,
        )
        solution = solver.solve(raise_error=False)
        candidate = solution.x
        valid = (
            candidate is not None and np.all(np.isfinite(candidate))
            and solution.info.status_val in (1, 2)
        )
        if not valid:
            raise RuntimeError(f"inverse dynamics QP rejected: {solution.info.status}")
        # The plant must receive exactly the torque recorded in diagnostics.
        # Remove solver-tolerance overshoot, then audit the adjusted candidate.
        candidate[34:] = np.clip(candidate[34:], -self.torque_limit_nm, self.torque_limit_nm)
        value = constraints @ candidate
        violation = float(max(0.0, np.max(lower - value), np.max(value - upper)))
        if violation > 1e-4:
            raise RuntimeError(f"inverse dynamics QP constraint violation: {violation}")
        torque = candidate[34:].copy()
        self.last_result = D1InverseDynamicsResult(
            status="solved", solver_status=solution.info.status,
            command_torque_nm=torque,
            generalized_acceleration=candidate[:22],
            contact_force_world_n=candidate[22:34].reshape(4, 3),
            dynamics_residual_max=float(np.max(np.abs(dynamics @ candidate + bias))),
            constraint_violation_max=violation,
            solve_ms=1e3 * (perf_counter() - started),
        )
        return torque
