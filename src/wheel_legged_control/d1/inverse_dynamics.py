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


def _cross3(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Fixed-size cross product without NumPy's broadcasting/axis dispatch."""
    result = np.empty(3)
    mujoco.mju_cross(result, left, right)
    return result


def _sparse_template(mask: np.ndarray) -> tuple[sparse.csc_matrix, tuple[np.ndarray, np.ndarray]]:
    """Reserve structural entries, including coefficients that are zero today."""
    matrix = sparse.csc_matrix(mask, dtype=np.float64)
    columns = np.repeat(np.arange(mask.shape[1]), np.diff(matrix.indptr))
    return matrix, (matrix.indices.copy(), columns)


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
    ``warm_start=True`` is an experimental numerical configuration: its contact
    force allocation is history-dependent and is not equivalent to cold solves.
    """

    def __init__(
        self,
        *,
        control_dt: float = 0.01,
        torque_limit_nm: np.ndarray = JOINT_TORQUE_LIMIT,
        warm_start: bool = False,
    ) -> None:
        self._nominal = D1Plant(control_dt=control_dt)
        self.control_dt = self._nominal.control_dt
        self.torque_limit_nm = np.asarray(torque_limit_nm, dtype=np.float64).copy()
        if self.torque_limit_nm.shape != (16,) or not np.all(
            np.isfinite(self.torque_limit_nm) & (self.torque_limit_nm > 0.0)
        ):
            raise ValueError("torque_limit_nm must contain 16 positive finite limits")
        self._mass = np.empty((22, 22))
        try:
            mujoco.mj_fullM(self._nominal.model, self._nominal.data, self._mass)
            self._full_mass_uses_data = True
        except TypeError:  # MuJoCo 3.2.x Python binding
            self._full_mass_uses_data = False
        self._actuation = np.zeros((22, 16))
        self._actuation[self._nominal.dof_addresses, np.arange(16)] = 1.0
        self._bound_rows = np.zeros((28, 50))
        self._bound_rows[:, 22:] = np.eye(28)
        self._friction = np.asarray(((1, 1, -.55), (1, -1, -.55),
                                     (-1, 1, -.55), (-1, -1, -.55), (0, 0, 1)))
        self._normal_force_limit = .65 * self._nominal.nominal_total_mass_kg * 9.81
        self.last_result: D1InverseDynamicsResult | None = None
        # Reusing numerical history is opt-in because it changes weakly weighted
        # internal forces even when generalized accelerations stay close.
        self._solver_algebra = osqp.default_algebra()
        self._warm_start = warm_start
        self._solver: osqp.OSQP | None = None
        self._yaw_rate_integral = 0.0
        self._last_control_time: float | None = None
        self._leg_indices = np.asarray([i for i in range(16) if i % 4 != 3])
        self._wheel_indices = np.asarray((3, 7, 11, 15))
        self._wheel_body_ids = np.asarray(self._nominal.wheel_body_ids_by_leg)
        self._sqrt_weights = np.sqrt((1., 1., 5., 10., 10., 2., *([.05] * 16)))
        regularization = np.concatenate((
            np.full(22, 1e-7),
            np.full(12, 1e-4 / (self._nominal.nominal_total_mass_kg * 9.81) ** 2),
            np.zeros(16),
        ))
        self._cost_diagonal = np.diag(regularization)
        self._cost_diagonal[:3, :3] += np.diag(self._sqrt_weights[:3] ** 2)
        self._cost_diagonal[6:22, 6:22] += np.diag(self._sqrt_weights[6:] ** 2)

    def reset(self) -> None:
        self.last_result = None
        self._solver = None
        self._yaw_rate_integral = 0.0
        self._last_control_time = None

    def _solve(self, hessian, gradient, constraints, lower, upper, active):
        if not self._warm_start:
            solver = osqp.OSQP(algebra=self._solver_algebra)
            solver.setup(
                P=sparse.csc_matrix(np.triu(hessian)), q=gradient,
                A=sparse.csc_matrix(constraints), l=lower, u=upper,
                verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=4000, polishing=True,
            )
            return solver.solve(raise_error=False)

        # A numerical nonzero mask changes with pose: in one 600-step trace it
        # caused 514 setups. Reserve equation blocks, not today's nonzero values.
        signature = (tuple(active), tuple(self.torque_limit_nm))
        reuse = self._solver is not None and self._cached_signature == signature
        if not reuse:
            p_mask = np.eye(50, dtype=bool)
            p_mask[:22, :22] = np.triu(np.ones((22, 22), dtype=bool))
            a_mask = np.zeros(constraints.shape, dtype=bool)
            a_mask[:22, :22] = True
            a_mask[:22, 34:] = self._actuation != 0.
            a_mask[22:50] = self._bound_rows != 0.
            for block, wheel in enumerate(active):
                force = slice(22 + 3 * wheel, 25 + 3 * wheel)
                row = 50 + 7 * block
                a_mask[:22, force] = True
                a_mask[row:row + 5, force] = True
                a_mask[row + 5:row + 7, :22] = True
            self._p_matrix, self._p_coordinates = _sparse_template(p_mask)
            self._a_matrix, self._a_coordinates = _sparse_template(a_mask)
        self._p_matrix.data[:] = hessian[self._p_coordinates]
        self._a_matrix.data[:] = constraints[self._a_coordinates]
        if reuse:
            self._solver.update(
                Px=self._p_matrix.data, q=gradient, Ax=self._a_matrix.data, l=lower, u=upper,
            )
        else:
            self._solver = osqp.OSQP(algebra=self._solver_algebra)
            self._solver.setup(
                P=self._p_matrix, q=gradient, A=self._a_matrix, l=lower, u=upper,
                verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=4000, polishing=True,
            )
            self._cached_signature = signature
        return self._solver.solve(raise_error=False)

    def _reconstruct(self, state: D1StateEstimate) -> tuple[np.ndarray, np.ndarray]:
        plant = self._nominal
        model, data = plant.model, plant.data
        data.qpos[:3] = state.base_position
        mujoco.mju_mat2Quat(data.qpos[3:7], state.base_rotation.ravel())
        data.qpos[plant.qpos_addresses] = state.joint_position
        com_offset = state.base_rotation @ model.body_ipos[plant.base_body_id]
        data.qvel[:3] = state.base_linear_velocity_world - _cross3(
            state.base_angular_velocity_world, com_offset
        )
        data.qvel[3:6] = state.base_angular_velocity_body
        data.qvel[plant.dof_addresses] = state.joint_velocity
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_fwdVelocity(model, data)
        # The supported bindings differ; select the calling convention once.
        if self._full_mass_uses_data:
            mujoco.mj_fullM(model, data, self._mass)
        else:
            mujoco.mj_fullM(model, self._mass, data.qM)
        return self._mass, data.qfrc_bias - data.qfrc_passive

    def _task_cost(
        self, command: D1Command, state: D1StateEstimate
    ) -> tuple[np.ndarray, np.ndarray]:
        rotation = state.base_rotation
        yaw = float(state.base_rpy[2])
        forward = np.asarray((np.cos(yaw), np.sin(yaw), 0.0))
        velocity = self._nominal.data.qvel[:3]
        desired_linear = 3.0 * (command.forward_velocity_mps * forward - velocity)
        desired_linear[2] = (
            80.0 * (command.base_height_m - state.base_position[2])
            + 18.0 * (command.base_vertical_velocity_mps - velocity[2])
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
        # The ideal rim model misses lateral scrub resistance. Integrate only
        # commanded turns and their braking phase. Keep straight-only histories
        # unchanged, and do not accumulate during contact loss or clock jumps.
        dt = (0. if self._last_control_time is None
              else state.control_time_s - self._last_control_time)
        if state.wheel_ground_contacts < 2 or dt < 0. or dt > 2. * self.control_dt:
            self._yaw_rate_integral = 0.
        elif dt > 0. and (command.yaw_rate_rps != 0. or self._yaw_rate_integral != 0.):
            self._yaw_rate_integral = float(np.clip(
                self._yaw_rate_integral
                + dt * (command.yaw_rate_rps - state.base_angular_velocity_world[2]),
                -.1, .1,
            ))
        desired_angular[2] += 40. * self._yaw_rate_integral
        leg_indices, wheel_indices = self._leg_indices, self._wheel_indices
        posture = (
            30.0 * (NOMINAL_JOINT_POSITION[leg_indices] - state.joint_position[leg_indices])
            - 6.0 * state.joint_velocity[leg_indices]
        )
        # Unconstrained airborne wheel accelerations otherwise act as reaction
        # wheels for the attitude task and can run to extreme speeds on landing.
        wheel_offset = (
            self._nominal.data.xpos[self._wheel_body_ids] - state.base_position
        )
        lateral_offset = wheel_offset @ np.asarray((-np.sin(yaw), np.cos(yaw), 0.))
        wheel_speed_target = (
            command.forward_velocity_mps - command.yaw_rate_rps * lateral_offset
        ) / self._nominal.wheel_radius_m
        wheel_acceleration = 10.0 * (
            wheel_speed_target - state.joint_velocity[wheel_indices]
        )
        target = np.concatenate((desired_linear, desired_angular, posture, wheel_acceleration))
        # Same weighted least squares, assembled as its diagonal and 3x3
        # rotation blocks instead of multiplying a mostly-zero 22x50 matrix.
        target = target * self._sqrt_weights
        hessian = self._cost_diagonal.copy()
        # torque_limit_nm is an existing mutable configuration array. Keep the
        # objective consistent with the bounds even if a caller changes it.
        np.fill_diagonal(hessian[34:, 34:], 1e-3 / self.torque_limit_nm**2)
        weighted_rotation = rotation * self._sqrt_weights[3:6, None]
        hessian[3:6, 3:6] += weighted_rotation.T @ weighted_rotation
        gradient = np.zeros(50)
        gradient[:3] = -self._sqrt_weights[:3] * target[:3]
        gradient[3:6] = -weighted_rotation.T @ target[3:6]
        gradient[self._nominal.dof_addresses[leg_indices]] = -self._sqrt_weights[6:18] * target[6:18]
        gradient[self._nominal.dof_addresses[wheel_indices]] = -self._sqrt_weights[18:] * target[18:]
        return hessian, gradient

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
            if normal[2] < np.cos(np.deg2rad(15.)):
                raise ValueError("contact normal outside the gentle-slope (15 degree) scope")
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
            axis_dot = _cross3(omega, axis)
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
            biases[wheel] = jacobian_dot @ plant.data.qvel + _cross3(
                omega, offset_dot - _cross3(omega, offset)
            )
            rolling = _cross3(axis, normal)
            rolling_norm = np.linalg.norm(rolling)
            rolling /= rolling_norm
            frames[wheel] = np.stack((rolling, _cross3(normal, rolling), normal))
            rolling_dot = (np.eye(3) - np.outer(rolling, rolling)) @ _cross3(
                axis_dot, normal
            ) / rolling_norm
            frame_rates[wheel] = np.stack((
                rolling_dot, _cross3(normal, rolling_dot), np.zeros(3)
            ))
        return jacobians, frames, biases, frame_rates

    def compute(self, command: D1Command, state: D1StateEstimate) -> np.ndarray:
        self.last_result = None
        try:
            return self._compute(command, state)
        except (ValueError, RuntimeError):
            # No result or numerical history may survive a rejected command.
            self.reset()
            raise

    def _compute(self, command: D1Command, state: D1StateEstimate) -> np.ndarray:
        if not np.all(np.isfinite((
            command.forward_velocity_mps, command.yaw_rate_rps,
            command.base_height_m, command.roll_rad, command.pitch_rad,
            command.base_vertical_velocity_mps,
        ))):
            raise ValueError("command values must be finite")
        started = perf_counter()
        mass, bias = self._reconstruct(state)
        hessian, gradient = self._task_cost(command, state)
        jacobians, frames, rolling_bias, frame_rates = self._contacts(state)
        active = np.flatnonzero(state.wheel_contact)
        constraints = np.zeros((50 + 7 * len(active), 50))
        lower = np.zeros(len(constraints))
        upper = np.zeros(len(constraints))
        dynamics = constraints[:22]
        dynamics[:, :22] = mass
        dynamics[:, 22:34] = -jacobians.reshape(12, 22).T
        dynamics[:, 34:] = -self._actuation
        constraints[22:50] = self._bound_rows
        force_bound = np.repeat(np.where(state.wheel_contact, np.inf, 0.0), 3)
        lower[:22] = upper[:22] = -bias
        lower[22:34], upper[22:34] = -force_bound, force_bound
        lower[34:50], upper[34:50] = -self.torque_limit_nm, self.torque_limit_nm
        for block, wheel in enumerate(active):
            frame = frames[wheel]
            row = 50 + 7 * block
            # A conservative friction diamond, |ft| + |fl| <= mu fn.
            constraints[row:row + 5, 22 + 3 * wheel:25 + 3 * wheel] = self._friction @ frame
            lower[row:row + 4] = -np.inf
            upper[row + 4] = self._normal_force_limit
            constraints[row + 5:row + 7, :22] = frame[[0, 2]] @ jacobians[wheel]
            velocity = jacobians[wheel] @ self._nominal.data.qvel
            target = frame @ (-rolling_bias[wheel] - 20.0 * velocity)
            target -= frame_rates[wheel] @ velocity
            lower[row + 5:row + 7] = upper[row + 5:row + 7] = target[[0, 2]]
            # Lateral slip is penalized, not forbidden: four-wheel skid steering
            # cannot satisfy four hard lateral no-slip constraints in general.
            lateral = frame[1] @ jacobians[wheel]
            hessian[:22, :22] += np.outer(lateral, lateral)
            gradient[:22] -= lateral * target[1]
        solution = self._solve(hessian, gradient, constraints, lower, upper, active)
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
        self._last_control_time = state.control_time_s
        return torque
