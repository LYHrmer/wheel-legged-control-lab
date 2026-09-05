"""Contact-constrained wrench allocation for the D1 wheel-leg controller."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from typing import Protocol

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, minimize

from .model import D1_JOINT_NAMES, LEG_PREFIXES, D1Plant
from .state_estimation import D1StateEstimate

D1_CONTACT_ALLOCATION_MODES = ("legacy", "constrained")
D1_CONSTRAINED_FRICTION_COEFFICIENT = 0.55
D1_NORMAL_FORCE_LIMIT_WEIGHT_FRACTION = 0.65
D1_WRENCH_CHARACTERISTIC_LENGTH_M = 0.25
D1_SLSQP_MAX_ITERATIONS = 80
D1_SLSQP_FTOL = 1e-10
D1_CANDIDATE_CONSTRAINT_VIOLATION_TOL = 1e-7
D1_WRENCH_TRACKING_REL_TOL = 5e-3
D1_FORCE_TRACKING_ABS_TOL_N = 1.0
D1_MOMENT_TRACKING_ABS_TOL_NM = 0.5


def _readonly_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(shape)


def _max_torque_excess_ratio(command_nm: np.ndarray, limit_nm: np.ndarray) -> float:
    return float(
        np.max(np.maximum(np.abs(command_nm) - limit_nm, 0.0) / limit_nm)
    )


def _max_feasible_force_scale(
    committed_torque_nm: np.ndarray,
    contact_torque_nm: np.ndarray,
    torque_limit_nm: np.ndarray,
) -> float:
    lower = 0.0
    upper = 1.0
    for committed, contact, limit in zip(
        committed_torque_nm,
        contact_torque_nm,
        torque_limit_nm,
        strict=True,
    ):
        if abs(contact) <= 1e-12:
            if abs(committed) > limit:
                return 0.0
            continue
        endpoints = ((-limit - committed) / contact, (limit - committed) / contact)
        lower = max(lower, min(endpoints))
        upper = min(upper, max(endpoints))
        if lower > upper:
            return 0.0
    return float(np.clip(upper, 0.0, 1.0))


class D1AllocationStatus(str, Enum):
    """Observable outcome of one contact-allocation attempt."""

    CONVERGED = "converged"
    FEASIBLE_NONCONVERGED = "feasible_nonconverged"
    NO_CONTACT = "no_contact"
    FALLBACK = "fallback"
    LEGACY = "legacy"


class D1WrenchTrackingStatus(str, Enum):
    """Whether allocated force and moment meet engineering tolerances."""

    TRACKED = "tracked"
    LIMITED = "limited"


def _wrench_tracking_status(
    desired_wrench: np.ndarray,
    achieved_wrench: np.ndarray,
) -> D1WrenchTrackingStatus:
    error = achieved_wrench - desired_wrench
    force_tolerance_n = D1_FORCE_TRACKING_ABS_TOL_N + (
        D1_WRENCH_TRACKING_REL_TOL * np.linalg.norm(desired_wrench[:3])
    )
    moment_tolerance_nm = D1_MOMENT_TRACKING_ABS_TOL_NM + (
        D1_WRENCH_TRACKING_REL_TOL * np.linalg.norm(desired_wrench[3:])
    )
    if (
        np.linalg.norm(error[:3]) <= force_tolerance_n
        and np.linalg.norm(error[3:]) <= moment_tolerance_nm
    ):
        return D1WrenchTrackingStatus.TRACKED
    return D1WrenchTrackingStatus.LIMITED


@dataclass(frozen=True, slots=True)
class D1ContactAllocationRequest:
    """One desired world-frame wrench and the torque already committed to other tasks.

    Forces are ground-on-robot forces. Moments are measured about the visible
    base-link origin. Wrench ordering in results is ``[Fx, Fy, Fz, Mx, My, Mz]``.
    """

    state: D1StateEstimate
    desired_force_world_n: np.ndarray
    desired_moment_world_nm: np.ndarray
    committed_torque_nm: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "desired_force_world_n",
            _readonly_array(self.desired_force_world_n, (3,), "desired_force_world_n"),
        )
        object.__setattr__(
            self,
            "desired_moment_world_nm",
            _readonly_array(self.desired_moment_world_nm, (3,), "desired_moment_world_nm"),
        )
        object.__setattr__(
            self,
            "committed_torque_nm",
            _readonly_array(
                self.committed_torque_nm,
                (len(D1_JOINT_NAMES),),
                "committed_torque_nm",
            ),
        )


@dataclass(frozen=True, slots=True)
class D1ContactAllocationResult:
    """Auditable force, torque, wrench, and feasibility result."""

    status: D1AllocationStatus
    status_reason: str
    wrench_tracking_status: D1WrenchTrackingStatus
    contact_force_world_n: np.ndarray
    contact_torque_nm: np.ndarray
    command_torque_nm: np.ndarray
    wrench_reference_position_world_m: np.ndarray
    desired_wrench_world: np.ndarray
    achieved_wrench_world: np.ndarray
    max_constraint_violation: float
    iterations: int
    solve_ms: float
    warm_started: bool = False

    def __post_init__(self) -> None:
        if not self.status_reason:
            raise ValueError("status_reason must not be empty")
        specifications = {
            "contact_force_world_n": ((len(LEG_PREFIXES), 3), self.contact_force_world_n),
            "contact_torque_nm": ((len(D1_JOINT_NAMES),), self.contact_torque_nm),
            "command_torque_nm": ((len(D1_JOINT_NAMES),), self.command_torque_nm),
            "wrench_reference_position_world_m": (
                (3,),
                self.wrench_reference_position_world_m,
            ),
            "desired_wrench_world": ((6,), self.desired_wrench_world),
            "achieved_wrench_world": ((6,), self.achieved_wrench_world),
        }
        for name, (shape, value) in specifications.items():
            object.__setattr__(self, name, _readonly_array(value, shape, name))
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        if not isinstance(self.warm_started, (bool, np.bool_)):
            raise TypeError("warm_started must be a bool")
        diagnostics = np.asarray((self.max_constraint_violation, self.solve_ms))
        if not np.isfinite(diagnostics).all() or np.any(diagnostics < 0.0):
            raise ValueError("allocation diagnostics must be finite and non-negative")


class D1ContactAllocator(Protocol):
    """Small allocation interface shared by legacy and constrained adapters."""

    mode: str

    def reset(self) -> None: ...

    def set_torque_limits(self, torque_limit_nm: np.ndarray) -> None: ...

    def solve(self, request: D1ContactAllocationRequest) -> D1ContactAllocationResult: ...


class D1LegacyContactAllocator:
    """Adapter preserving the original unconstrained VMC force distribution."""

    mode = "legacy"

    def __init__(
        self,
        *,
        normal_force_limit_n: float,
        wheel_radius_m: float,
        torque_limit_nm: np.ndarray,
    ) -> None:
        if not np.isfinite(normal_force_limit_n) or normal_force_limit_n <= 0.0:
            raise ValueError("normal_force_limit_n must be finite and positive")
        if not np.isfinite(wheel_radius_m) or wheel_radius_m <= 0.0:
            raise ValueError("wheel_radius_m must be finite and positive")
        self.normal_force_limit_n = float(normal_force_limit_n)
        self.wheel_radius_m = float(wheel_radius_m)
        self.torque_limit_nm = _readonly_array(
            torque_limit_nm,
            (len(D1_JOINT_NAMES),),
            "torque_limit_nm",
        )
        if np.any(self.torque_limit_nm <= 0.0):
            raise ValueError("torque_limit_nm must be positive")

    def reset(self) -> None:
        """The legacy allocator has no history to clear."""

    def set_torque_limits(self, torque_limit_nm: np.ndarray) -> None:
        """Update the limits used for clipping and normalized diagnostics."""

        limits = _readonly_array(
            torque_limit_nm,
            (len(D1_JOINT_NAMES),),
            "torque_limit_nm",
        )
        if np.any(limits <= 0.0):
            raise ValueError("torque_limit_nm must be positive")
        self.torque_limit_nm = limits

    def solve(self, request: D1ContactAllocationRequest) -> D1ContactAllocationResult:
        """Reproduce four-wheel least squares, clipping, and equal wheel drive."""

        start = perf_counter()
        state = request.state
        desired_wrench = np.concatenate(
            (request.desired_force_world_n, request.desired_moment_world_nm)
        )
        offsets = state.foot_offset_world
        vertical_matrix = np.vstack(
            (np.ones(len(LEG_PREFIXES)), offsets[:, 1], -offsets[:, 0])
        )
        vertical_target = np.asarray(
            (
                request.desired_force_world_n[2],
                request.desired_moment_world_nm[0],
                request.desired_moment_world_nm[1],
            )
        )
        normal_force = np.linalg.lstsq(vertical_matrix, vertical_target, rcond=None)[0]
        normal_force = np.clip(normal_force, 0.0, self.normal_force_limit_n)

        contact_torque = np.zeros(len(D1_JOINT_NAMES), dtype=np.float64)
        for leg_index, force_n in enumerate(normal_force):
            joint_slice = slice(4 * leg_index, 4 * (leg_index + 1))
            contact_torque[joint_slice] = state.foot_jacobian[leg_index].T @ np.asarray(
                (0.0, 0.0, -force_n)
            )
        wheel_indices = np.asarray((3, 7, 11, 15), dtype=np.int32)
        contact_torque[wheel_indices] = 0.0

        forward_world = state.base_rotation[:, 0].copy()
        forward_world[2] = 0.0
        forward_world /= np.linalg.norm(forward_world)
        longitudinal_force = float(np.dot(request.desired_force_world_n, forward_world))
        contact_torque[wheel_indices] += (
            longitudinal_force * self.wheel_radius_m / len(LEG_PREFIXES)
        )
        unclipped_command = request.committed_torque_nm + contact_torque
        command = np.clip(unclipped_command, -self.torque_limit_nm, self.torque_limit_nm)

        contact_force = np.zeros((len(LEG_PREFIXES), 3), dtype=np.float64)
        contact_force[:, 2] = normal_force
        contact_force += (
            forward_world * longitudinal_force / len(LEG_PREFIXES)
        )[None, :]
        achieved_wrench = np.concatenate(
            (
                np.sum(contact_force, axis=0),
                np.sum(np.cross(offsets, contact_force), axis=0),
            )
        )
        max_violation = _max_torque_excess_ratio(
            unclipped_command,
            self.torque_limit_nm,
        )
        return D1ContactAllocationResult(
            status=D1AllocationStatus.LEGACY,
            status_reason="unconstrained_legacy",
            wrench_tracking_status=_wrench_tracking_status(
                desired_wrench,
                achieved_wrench,
            ),
            contact_force_world_n=contact_force,
            contact_torque_nm=contact_torque,
            command_torque_nm=command,
            wrench_reference_position_world_m=state.base_position,
            desired_wrench_world=desired_wrench,
            achieved_wrench_world=achieved_wrench,
            max_constraint_violation=max_violation,
            iterations=0,
            solve_ms=1e3 * (perf_counter() - start),
        )


class D1ConstrainedContactAllocator:
    """Allocate a desired body wrench over the currently active wheel contacts."""

    mode = "constrained"

    def __init__(
        self,
        *,
        friction_coefficient: float,
        normal_force_limit_n: float,
        torque_limit_nm: np.ndarray,
        wrench_characteristic_length_m: float = D1_WRENCH_CHARACTERISTIC_LENGTH_M,
    ) -> None:
        if not np.isfinite(friction_coefficient) or friction_coefficient <= 0.0:
            raise ValueError("friction_coefficient must be finite and positive")
        if not np.isfinite(normal_force_limit_n) or normal_force_limit_n <= 0.0:
            raise ValueError("normal_force_limit_n must be finite and positive")
        if (
            not np.isfinite(wrench_characteristic_length_m)
            or wrench_characteristic_length_m <= 0.0
        ):
            raise ValueError("wrench_characteristic_length_m must be finite and positive")
        self.friction_coefficient = float(friction_coefficient)
        self.normal_force_limit_n = float(normal_force_limit_n)
        self.wrench_characteristic_length_m = float(wrench_characteristic_length_m)
        self._wrench_weight = np.asarray(
            (1.0, 1.0, 1.0) + (1.0 / self.wrench_characteristic_length_m,) * 3,
            dtype=np.float64,
        )
        self.torque_limit_nm = _readonly_array(
            torque_limit_nm,
            (len(D1_JOINT_NAMES),),
            "torque_limit_nm",
        )
        if np.any(self.torque_limit_nm <= 0.0):
            raise ValueError("torque_limit_nm must be positive")
        self.reset()

    def reset(self) -> None:
        """Forget any warm-start state from the previous contact mode."""

        self._previous_contact_mask: np.ndarray | None = None
        self._previous_force_world_n = np.zeros((len(LEG_PREFIXES), 3), dtype=np.float64)

    def set_torque_limits(self, torque_limit_nm: np.ndarray) -> None:
        """Update actuator constraints and discard the old warm start."""

        limits = _readonly_array(
            torque_limit_nm,
            (len(D1_JOINT_NAMES),),
            "torque_limit_nm",
        )
        if np.any(limits <= 0.0):
            raise ValueError("torque_limit_nm must be positive")
        self.torque_limit_nm = limits
        self.reset()

    def solve(self, request: D1ContactAllocationRequest) -> D1ContactAllocationResult:
        """Return a constrained allocation with explicit status and diagnostics."""

        start = perf_counter()
        desired_wrench = np.concatenate(
            (request.desired_force_world_n, request.desired_moment_world_nm)
        )
        if not np.any(request.state.wheel_contact):
            self.reset()
            zeros_force = np.zeros((len(LEG_PREFIXES), 3), dtype=np.float64)
            zeros_torque = np.zeros(len(D1_JOINT_NAMES), dtype=np.float64)
            command = np.clip(
                request.committed_torque_nm,
                -self.torque_limit_nm,
                self.torque_limit_nm,
            )
            max_constraint_violation = _max_torque_excess_ratio(
                request.committed_torque_nm,
                self.torque_limit_nm,
            )
            return D1ContactAllocationResult(
                status=D1AllocationStatus.NO_CONTACT,
                status_reason="no_active_contacts",
                wrench_tracking_status=_wrench_tracking_status(
                    desired_wrench,
                    np.zeros(6, dtype=np.float64),
                ),
                contact_force_world_n=zeros_force,
                contact_torque_nm=zeros_torque,
                command_torque_nm=command,
                wrench_reference_position_world_m=request.state.base_position,
                desired_wrench_world=desired_wrench,
                achieved_wrench_world=np.zeros(6, dtype=np.float64),
                max_constraint_violation=max_constraint_violation,
                iterations=0,
                solve_ms=1e3 * (perf_counter() - start),
            )
        state = request.state
        active_indices = np.flatnonzero(state.wheel_contact)
        active_normal_norms = np.linalg.norm(state.wheel_contact_normal[active_indices], axis=1)
        if not np.allclose(active_normal_norms, 1.0, atol=1e-8, rtol=1e-8):
            raise ValueError("active contact normals must be unit vectors")
        wheel_separation = (
            state.foot_position[active_indices] - state.wheel_contact_point[active_indices]
        )
        if np.any(
            np.einsum(
                "ij,ij->i",
                state.wheel_contact_normal[active_indices],
                wheel_separation,
            )
            <= 0.0
        ):
            raise ValueError("active contact normals must point from the terrain toward the robot")
        contact_frames = np.empty((active_indices.size, 3, 3), dtype=np.float64)
        forward_world = state.base_rotation[:, 0]
        for local_index, leg_index in enumerate(active_indices):
            normal = state.wheel_contact_normal[leg_index]
            rolling = forward_world - normal * np.dot(normal, forward_world)
            rolling_norm = np.linalg.norm(rolling)
            if rolling_norm < 1e-8:
                raise ValueError("base forward direction cannot be parallel to a contact normal")
            rolling /= rolling_norm
            lateral = np.cross(normal, rolling)
            contact_frames[local_index] = np.column_stack((rolling, lateral, normal))

        variable_count = 3 * active_indices.size
        wrench_matrix = np.zeros((6, variable_count), dtype=np.float64)
        torque_matrix = np.zeros((len(D1_JOINT_NAMES), variable_count), dtype=np.float64)
        for local_index, leg_index in enumerate(active_indices):
            columns = slice(3 * local_index, 3 * (local_index + 1))
            frame = contact_frames[local_index]
            offset = state.wheel_contact_point[leg_index] - state.base_position
            wrench_matrix[:3, columns] = frame
            wrench_matrix[3:, columns] = np.column_stack(
                tuple(np.cross(offset, frame[:, axis]) for axis in range(3))
            )
            joint_slice = slice(4 * leg_index, 4 * (leg_index + 1))
            torque_matrix[joint_slice, columns] = (
                -state.wheel_contact_jacobian[leg_index].T @ frame
            )

        lower = np.tile(
            (-self.friction_coefficient * self.normal_force_limit_n,) * 2 + (0.0,),
            active_indices.size,
        )
        upper = np.tile(
            (self.friction_coefficient * self.normal_force_limit_n,) * 2
            + (self.normal_force_limit_n,),
            active_indices.size,
        )
        initial = np.empty(variable_count, dtype=np.float64)
        force_share = request.desired_force_world_n / active_indices.size
        warm_started = bool(
            self._previous_contact_mask is not None
            and np.array_equal(self._previous_contact_mask, state.wheel_contact)
        )
        for local_index, frame in enumerate(contact_frames):
            leg_index = active_indices[local_index]
            source_force = (
                self._previous_force_world_n[leg_index] if warm_started else force_share
            )
            local_force = frame.T @ source_force
            normal_force = np.clip(local_force[2], 0.0, self.normal_force_limit_n)
            tangential_limit = self.friction_coefficient * normal_force
            tangential_force = local_force[:2].copy()
            tangential_l1 = float(np.sum(np.abs(tangential_force)))
            if tangential_l1 > tangential_limit and tangential_l1 > 0.0:
                tangential_force *= tangential_limit / tangential_l1
            initial[3 * local_index : 3 * (local_index + 1)] = (
                tangential_force[0],
                tangential_force[1],
                normal_force,
            )

        friction_matrix = np.zeros((4 * active_indices.size, variable_count))
        for local_index in range(active_indices.size):
            first = 3 * local_index
            row = 4 * local_index
            friction_matrix[row, first] = 1.0
            friction_matrix[row, first + 1] = 1.0
            friction_matrix[row, first + 2] = -self.friction_coefficient
            friction_matrix[row + 1, first] = 1.0
            friction_matrix[row + 1, first + 1] = -1.0
            friction_matrix[row + 1, first + 2] = -self.friction_coefficient
            friction_matrix[row + 2, first] = -1.0
            friction_matrix[row + 2, first + 1] = 1.0
            friction_matrix[row + 2, first + 2] = -self.friction_coefficient
            friction_matrix[row + 3, first] = -1.0
            friction_matrix[row + 3, first + 1] = -1.0
            friction_matrix[row + 3, first + 2] = -self.friction_coefficient

        regularization = 1e-10

        def objective(local_forces: np.ndarray) -> tuple[float, np.ndarray]:
            error = wrench_matrix @ local_forces - desired_wrench
            weighted_error = self._wrench_weight * error
            value = 0.5 * (
                weighted_error @ weighted_error
                + regularization * (local_forces @ local_forces)
            )
            gradient = (
                wrench_matrix.T @ (self._wrench_weight * weighted_error)
                + regularization * local_forces
            )
            return float(value), gradient

        constraints = (
            LinearConstraint(friction_matrix, -np.inf, 0.0),
            LinearConstraint(
                torque_matrix,
                -self.torque_limit_nm - request.committed_torque_nm,
                self.torque_limit_nm - request.committed_torque_nm,
            ),
        )
        solution = minimize(
            objective,
            initial,
            method="SLSQP",
            jac=True,
            bounds=Bounds(lower, upper),
            constraints=constraints,
            options={
                "maxiter": D1_SLSQP_MAX_ITERATIONS,
                "ftol": D1_SLSQP_FTOL,
            },
        )
        candidate = np.asarray(solution.x, dtype=np.float64)
        candidate_is_finite = bool(
            candidate.shape == (variable_count,) and np.isfinite(candidate).all()
        )
        if candidate_is_finite:
            lower_violation = np.max(lower - candidate)
            upper_violation = np.max(candidate - upper)
            friction_violation = np.max(friction_matrix @ candidate)
            candidate_command = request.committed_torque_nm + torque_matrix @ candidate
            candidate_force_violation_ratio = float(
                max(0.0, lower_violation, upper_violation, friction_violation)
                / self.normal_force_limit_n
            )
            candidate_torque_violation_ratio = _max_torque_excess_ratio(
                candidate_command,
                self.torque_limit_nm,
            )
            candidate_violation_ratio = max(
                candidate_force_violation_ratio,
                candidate_torque_violation_ratio,
            )
            candidate_objective, _ = objective(candidate)
        else:
            candidate_violation_ratio = float("inf")
            candidate_objective = float("inf")
        accepted = bool(
            candidate_is_finite
            and np.isfinite(candidate_objective)
            and candidate_violation_ratio
            <= D1_CANDIDATE_CONSTRAINT_VIOLATION_TOL
        )

        if accepted:
            local_forces = candidate
            if bool(solution.success):
                status = D1AllocationStatus.CONVERGED
                status_reason = "slsqp_converged"
            else:
                status = D1AllocationStatus.FEASIBLE_NONCONVERGED
                status_reason = f"slsqp_status_{getattr(solution, 'status', 'unknown')}"
        else:
            normal_wrench = np.column_stack(
                [wrench_matrix[:, 3 * index + 2] for index in range(active_indices.size)]
            )
            fallback = lsq_linear(
                self._wrench_weight[:, None] * normal_wrench,
                self._wrench_weight * desired_wrench,
                bounds=(0.0, self.normal_force_limit_n),
                method="bvls",
                tol=1e-10,
            )
            local_forces = np.zeros(variable_count, dtype=np.float64)
            local_forces[2::3] = fallback.x
            force_scale = _max_feasible_force_scale(
                request.committed_torque_nm,
                torque_matrix @ local_forces,
                self.torque_limit_nm,
            )
            local_forces *= force_scale
            status = D1AllocationStatus.FALLBACK
            status_reason = "solver_rejected"

        contact_force = np.zeros((len(LEG_PREFIXES), 3), dtype=np.float64)
        for local_index, leg_index in enumerate(active_indices):
            columns = slice(3 * local_index, 3 * (local_index + 1))
            contact_force[leg_index] = contact_frames[local_index] @ local_forces[columns]

        contact_torque = torque_matrix @ local_forces
        unclipped_command = request.committed_torque_nm + contact_torque
        command = np.clip(unclipped_command, -self.torque_limit_nm, self.torque_limit_nm)
        achieved_wrench = wrench_matrix @ local_forces
        wrench_tracking_status = _wrench_tracking_status(
            desired_wrench,
            achieved_wrench,
        )

        local_friction_violation_n = float(np.max(friction_matrix @ local_forces))
        local_bound_violation_n = max(
            float(np.max(lower - local_forces)),
            float(np.max(local_forces - upper)),
        )
        force_violation_ratio = max(
            0.0,
            local_friction_violation_n,
            local_bound_violation_n,
        ) / self.normal_force_limit_n
        torque_violation_ratio = _max_torque_excess_ratio(
            unclipped_command,
            self.torque_limit_nm,
        )
        max_constraint_violation = float(
            max(force_violation_ratio, torque_violation_ratio)
        )
        self._previous_contact_mask = state.wheel_contact.copy()
        self._previous_force_world_n = contact_force.copy()
        solve_ms = 1e3 * (perf_counter() - start)
        return D1ContactAllocationResult(
            status=status,
            status_reason=status_reason,
            wrench_tracking_status=wrench_tracking_status,
            contact_force_world_n=contact_force,
            contact_torque_nm=contact_torque,
            command_torque_nm=command,
            wrench_reference_position_world_m=state.base_position,
            desired_wrench_world=desired_wrench,
            achieved_wrench_world=achieved_wrench,
            max_constraint_violation=max_constraint_violation,
            iterations=int(getattr(solution, "nit", 0) or 0),
            solve_ms=solve_ms,
            warm_started=warm_started,
        )


def make_d1_contact_allocator(
    plant: D1Plant,
    mode: str,
) -> D1ContactAllocator:
    """Build a named allocation adapter without exposing its solver backend."""

    if mode not in D1_CONTACT_ALLOCATION_MODES:
        raise ValueError(f"contact_allocation must be one of {D1_CONTACT_ALLOCATION_MODES}")
    normal_force_limit_n = (
        plant.nominal_total_mass_kg
        * abs(float(plant.model.opt.gravity[2]))
        * D1_NORMAL_FORCE_LIMIT_WEIGHT_FRACTION
    )
    if mode == "legacy":
        return D1LegacyContactAllocator(
            normal_force_limit_n=normal_force_limit_n,
            wheel_radius_m=plant.wheel_radius_m,
            torque_limit_nm=plant.actuator_torque_limit_nm,
        )
    return D1ConstrainedContactAllocator(
        friction_coefficient=D1_CONSTRAINED_FRICTION_COEFFICIENT,
        normal_force_limit_n=normal_force_limit_n,
        torque_limit_nm=plant.actuator_torque_limit_nm,
    )
