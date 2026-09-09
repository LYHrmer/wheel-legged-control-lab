"""Hierarchical D1 controllers: contact VMC inside, LQR or MPC outside."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import block_diag, solve_discrete_are
from scipy.optimize import minimize

from .contact_allocation import D1ContactAllocator
from .control_context import D1ControllerMemory, D1ControlProposal, D1ForceBaseline
from .controllers import D1Command, D1VMCController
from .linear_model import D1SagittalLinearModel, identify_sagittal_model
from .model import D1Plant
from .state_estimation import D1StateEstimate

D1_OUTER_Q = np.diag((0.2, 400.0, 20.0, 30.0))
D1_OUTER_R = np.asarray(((1e-8,),), dtype=np.float64)
D1_CONSTRAINED_OUTER_R = np.asarray(((1e-6,),), dtype=np.float64)
D1_LONGITUDINAL_FORCE_LIMIT_N = 180.0
D1_VERTICAL_RESIDUAL_LIMIT_N = 120.0
D1_VERTICAL_FEEDFORWARD_LIMIT_N = 500.0


@dataclass(frozen=True)
class _PreparedControl:
    command: D1Command
    state: D1StateEstimate
    requested_feedforward_n: float
    proposal: D1ControlProposal
    solution: np.ndarray | None = None
    solve_ms: float = 0.0
    iterations: int = 0


def _default_outer_r(allocation_mode: str) -> np.ndarray:
    if allocation_mode == "legacy":
        return D1_OUTER_R.copy()
    if allocation_mode == "constrained":
        return D1_CONSTRAINED_OUTER_R.copy()
    raise ValueError(f"unsupported contact-allocation mode: {allocation_mode!r}")


class _D1HierarchicalController:
    """Shared command tracking and low-level torque composition."""

    baseline_name = "abstract"

    def __init__(
        self,
        plant: D1Plant,
        linear_model: D1SagittalLinearModel | None = None,
        *,
        contact_allocator: D1ContactAllocator | None = None,
    ):
        self.control_dt = plant.control_dt
        allocation_mode = "legacy" if contact_allocator is None else contact_allocator.mode
        self.allocation_mode = allocation_mode
        self.linear_model = (
            identify_sagittal_model(plant.control_dt, allocation_mode)
            if linear_model is None
            else linear_model
        )
        self.low_level = D1VMCController(plant, contact_allocator=contact_allocator)
        self.low_level.wheel_velocity_gain = 0.0
        self._distance_m = 0.0
        self._distance_reference_m = 0.0
        self.last_longitudinal_force_n = 0.0
        self.last_vertical_residual_n = 0.0
        self.last_solve_ms = 0.0
        self.longitudinal_force_limit_n = D1_LONGITUDINAL_FORCE_LIMIT_N

    def reset(self) -> None:
        self.low_level.reset()
        self._distance_m = 0.0
        self._distance_reference_m = 0.0
        self.last_longitudinal_force_n = 0.0
        self.last_vertical_residual_n = 0.0
        self.last_solve_ms = 0.0
        self._prepared_control: _PreparedControl | None = None
        self._consumed_proposal_key: tuple[int, float] | None = None

    @property
    def control_memory(self) -> D1ControllerMemory:
        """Read-only current integrals, before the next compute call."""
        return D1ControllerMemory(self._distance_m, self._distance_reference_m)

    def _pending_preview(self, command, state, feedforward) -> _PreparedControl | None:
        if not np.isfinite(feedforward):
            raise ValueError("vertical feedforward must be finite")
        key = (state.sequence, state.control_time_s)
        if key == self._consumed_proposal_key:
            raise RuntimeError("this control-tick proposal has already been consumed")
        pending = self._prepared_control
        if pending is not None and (
            pending.state is not state
            or pending.command != command
            or pending.requested_feedforward_n != feedforward
            or pending.proposal.memory != self.control_memory
        ):
            raise RuntimeError(
                "pending proposal is stale for this state, command or controller memory"
            )
        return pending

    def _consume_preview(self, command, state, feedforward) -> _PreparedControl | None:
        pending = self._pending_preview(command, state, feedforward)
        if pending is not None:
            self._prepared_control = None
            self._consumed_proposal_key = (state.sequence, state.control_time_s)
        return pending

    def _preview_tracking(self, command, state) -> tuple[np.ndarray, np.ndarray]:
        """The exact next tracking state/reference, without advancing integrals."""
        linear_velocity, angular_velocity = state.base_velocity(local=True)
        distance = self._distance_m + float(linear_velocity[0]) * self.control_dt
        distance_reference = float(
            np.clip(
                self._distance_reference_m + command.forward_velocity_mps * self.control_dt,
                distance - 0.55,
                distance + 0.55,
            )
        )
        control_state = np.asarray(
            (distance, state.base_rpy[1], linear_velocity[0], angular_velocity[1]), dtype=np.float64
        )
        reference = np.asarray(
            (
                distance_reference,
                self.linear_model.operating_state[1] + command.pitch_rad,
                command.forward_velocity_mps,
                0.0,
            ),
            dtype=np.float64,
        )
        return control_state, reference

    def _force_proposal(self, command, state, force, feedforward) -> D1ControlProposal:
        if command.base_vertical_velocity_mps != 0.0:
            raise ValueError(
                "vertical velocity feedforward requires the inverse dynamics controller"
            )
        # Same current nominal upward request as VMC, before allocation and RL.
        support = (
            self.low_level.total_mass_kg * self.low_level.gravity_mps2
            + self.low_level.height_kp * (command.base_height_m - state.base_position[2])
            - self.low_level.height_kd * state.base_linear_velocity_world[2]
        )
        return D1ControlProposal(
            tick=state.sequence,
            control_time_s=state.control_time_s,
            baseline=D1ForceBaseline(
                longitudinal_force_n=float(force),
                support_vertical_force_n=float(support),
                vertical_feedforward_force_n=float(
                    np.clip(
                        feedforward,
                        -D1_VERTICAL_FEEDFORWARD_LIMIT_N,
                        D1_VERTICAL_FEEDFORWARD_LIMIT_N,
                    )
                ),
            ),
            memory=self.control_memory,
        )

    def _control_state(self, state: D1StateEstimate) -> np.ndarray:
        linear_velocity, angular_velocity = state.base_velocity(local=True)
        self._distance_m += float(linear_velocity[0]) * self.control_dt
        return np.asarray(
            (
                self._distance_m,
                state.base_rpy[1],
                linear_velocity[0],
                angular_velocity[1],
            ),
            dtype=np.float64,
        )

    def _advance_position_reference(self, velocity_mps: float) -> float:
        self._distance_reference_m += velocity_mps * self.control_dt
        self._distance_reference_m = float(
            np.clip(
                self._distance_reference_m,
                self._distance_m - 0.55,
                self._distance_m + 0.55,
            )
        )
        return self._distance_reference_m

    def _compose_torque(
        self,
        state: D1StateEstimate,
        command: D1Command,
        baseline_force_n: float,
        residual_force_n: np.ndarray | None,
        vertical_feedforward_force_n: float,
    ) -> np.ndarray:
        residual = (
            np.zeros(2, dtype=np.float64)
            if residual_force_n is None
            else np.asarray(residual_force_n, dtype=np.float64)
        )
        if residual.shape != (2,):
            raise ValueError("D1 residual force must have shape (2,)")
        longitudinal = float(
            np.clip(
                baseline_force_n + residual[0],
                -self.longitudinal_force_limit_n,
                self.longitudinal_force_limit_n,
            )
        )
        residual_vertical = float(
            np.clip(residual[1], -D1_VERTICAL_RESIDUAL_LIMIT_N, D1_VERTICAL_RESIDUAL_LIMIT_N)
        )
        feedforward_vertical = float(
            np.clip(
                vertical_feedforward_force_n,
                -D1_VERTICAL_FEEDFORWARD_LIMIT_N,
                D1_VERTICAL_FEEDFORWARD_LIMIT_N,
            )
        )
        vertical = residual_vertical + feedforward_vertical
        inner_command = D1Command(
            forward_velocity_mps=0.0,
            yaw_rate_rps=command.yaw_rate_rps,
            base_height_m=command.base_height_m,
            roll_rad=command.roll_rad,
            pitch_rad=command.pitch_rad,
            base_vertical_velocity_mps=command.base_vertical_velocity_mps,
        )
        self.last_longitudinal_force_n = longitudinal
        self.last_vertical_residual_n = vertical
        return self.low_level.compute(
            inner_command,
            state,
            longitudinal_force_n=longitudinal,
            vertical_force_offset_n=vertical,
        )


class D1LQRVMCController(_D1HierarchicalController):
    """Infinite-horizon LQR on the identified sagittal D1/VMC dynamics."""

    baseline_name = "D1 LQR+VMC"

    def __init__(
        self,
        plant: D1Plant,
        linear_model: D1SagittalLinearModel | None = None,
        q: np.ndarray | None = None,
        r: np.ndarray | None = None,
        *,
        contact_allocator: D1ContactAllocator | None = None,
    ) -> None:
        super().__init__(plant, linear_model, contact_allocator=contact_allocator)
        self.q = D1_OUTER_Q.copy() if q is None else np.asarray(q, dtype=np.float64)
        self.r = (
            _default_outer_r(self.allocation_mode) if r is None else np.asarray(r, dtype=np.float64)
        )
        riccati = solve_discrete_are(
            self.linear_model.a,
            self.linear_model.b,
            self.q,
            self.r,
        )
        self.gain = np.linalg.solve(
            self.r + self.linear_model.b.T @ riccati @ self.linear_model.b,
            self.linear_model.b.T @ riccati @ self.linear_model.a,
        )
        self.closed_loop_eigenvalues = np.linalg.eigvals(
            self.linear_model.a - self.linear_model.b @ self.gain
        )
        self.reset()

    def preview_baseline(
        self,
        command: D1Command,
        state: D1StateEstimate,
        *,
        vertical_feedforward_force_n: float = 0.0,
    ) -> D1ControlProposal:
        """Prepare current baseline once; read/preview never advances integrals.

        Repeated previews require the identical immutable state publication and
        command. Compute consumes this proposal exactly once. Reset discards it.
        """
        pending = self._pending_preview(command, state, vertical_feedforward_force_n)
        if pending is None:
            control_state, reference = self._preview_tracking(command, state)
            force = -float((self.gain @ (control_state - reference)).item())
            proposal = self._force_proposal(command, state, force, vertical_feedforward_force_n)
            pending = _PreparedControl(command, state, vertical_feedforward_force_n, proposal)
            self._prepared_control = pending
        return pending.proposal

    def compute(
        self,
        command: D1Command,
        state: D1StateEstimate,
        residual_force_n: np.ndarray | None = None,
        *,
        vertical_feedforward_force_n: float = 0.0,
    ) -> np.ndarray:
        pending = self._consume_preview(command, state, vertical_feedforward_force_n)
        control_state = self._control_state(state)
        position_reference = self._advance_position_reference(command.forward_velocity_mps)
        reference = np.asarray(
            (
                position_reference,
                self.linear_model.operating_state[1] + command.pitch_rad,
                command.forward_velocity_mps,
                0.0,
            ),
            dtype=np.float64,
        )
        force = (
            -float((self.gain @ (control_state - reference)).item())
            if pending is None
            else pending.proposal.baseline.longitudinal_force_n
        )
        return self._compose_torque(
            state,
            command,
            force,
            residual_force_n,
            vertical_feedforward_force_n,
        )


class D1MPCVMCController(_D1HierarchicalController):
    """Bounded receding-horizon MPC on the same identified D1 model."""

    baseline_name = "D1 MPC+VMC"

    def __init__(
        self,
        plant: D1Plant,
        linear_model: D1SagittalLinearModel | None = None,
        horizon: int = 20,
        q: np.ndarray | None = None,
        r: np.ndarray | None = None,
        *,
        contact_allocator: D1ContactAllocator | None = None,
    ) -> None:
        if horizon < 2:
            raise ValueError("horizon must be at least 2")
        super().__init__(plant, linear_model, contact_allocator=contact_allocator)
        self.horizon = horizon
        self.q = D1_OUTER_Q.copy() if q is None else np.asarray(q, dtype=np.float64)
        self.r = (
            _default_outer_r(self.allocation_mode) if r is None else np.asarray(r, dtype=np.float64)
        )
        self.terminal_q = solve_discrete_are(
            self.linear_model.a,
            self.linear_model.b,
            self.q,
            self.r,
        )
        self._sx, self._su = self._prediction_matrices()
        self._qbar = block_diag(*([self.q] * (horizon - 1)), self.terminal_q)
        self._rbar = np.kron(np.eye(horizon), self.r)
        self._hessian = 2.0 * (self._su.T @ self._qbar @ self._su + self._rbar)
        self._solution = np.zeros(horizon, dtype=np.float64)
        self.last_iterations = 0
        self.reset()

    def _prediction_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        state_size = self.linear_model.a.shape[0]
        sx = np.zeros((self.horizon * state_size, state_size), dtype=np.float64)
        su = np.zeros((self.horizon * state_size, self.horizon), dtype=np.float64)
        for row in range(self.horizon):
            sx[row * state_size : (row + 1) * state_size] = np.linalg.matrix_power(
                self.linear_model.a, row + 1
            )
            for column in range(row + 1):
                su[
                    row * state_size : (row + 1) * state_size,
                    column : column + 1,
                ] = np.linalg.matrix_power(self.linear_model.a, row - column) @ self.linear_model.b
        return sx, su

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "_solution"):
            self._solution[:] = 0.0
        self.last_iterations = 0

    def _tracking_reference(self, command: D1Command) -> np.ndarray:
        position_reference = self._advance_position_reference(command.forward_velocity_mps)
        return np.asarray(
            (
                position_reference,
                self.linear_model.operating_state[1] + command.pitch_rad,
                command.forward_velocity_mps,
                0.0,
            ),
            dtype=np.float64,
        )

    def preview_baseline(
        self,
        command: D1Command,
        state: D1StateEstimate,
        *,
        vertical_feedforward_force_n: float = 0.0,
    ) -> D1ControlProposal:
        """Solve once for this tick without advancing distance or warm start.

        The prepared solution is consumed by compute, avoiding a second QP
        solve. Timing covers the preparation solve; measure preview+compute
        wall time when benchmarking the full control decision.
        """
        pending = self._pending_preview(command, state, vertical_feedforward_force_n)
        if pending is None:
            control_state, reference = self._preview_tracking(command, state)
            gradient = 2.0 * self._su.T @ self._qbar @ (self._sx @ (control_state - reference))

            def objective(sequence):
                return (
                    0.5 * sequence @ self._hessian @ sequence + gradient @ sequence,
                    self._hessian @ sequence + gradient,
                )

            start = perf_counter()
            result = minimize(
                objective,
                self._solution.copy(),
                method="L-BFGS-B",
                jac=True,
                bounds=[(-self.longitudinal_force_limit_n, self.longitudinal_force_limit_n)]
                * self.horizon,
                options={"maxiter": 30, "ftol": 1e-8, "gtol": 1e-6},
            )
            solve_ms = 1e3 * (perf_counter() - start)
            solution = (
                result.x.copy()
                if result.success or np.isfinite(result.fun)
                else self._solution.copy()
            )
            proposal = self._force_proposal(
                command, state, float(solution[0]), vertical_feedforward_force_n
            )
            pending = _PreparedControl(
                command,
                state,
                vertical_feedforward_force_n,
                proposal,
                solution,
                solve_ms,
                int(result.nit),
            )
            self._prepared_control = pending
        return pending.proposal

    def compute(
        self,
        command: D1Command,
        state: D1StateEstimate,
        residual_force_n: np.ndarray | None = None,
        *,
        vertical_feedforward_force_n: float = 0.0,
    ) -> np.ndarray:
        pending = self._consume_preview(command, state, vertical_feedforward_force_n)
        if pending is not None:
            self._control_state(state)
            self._tracking_reference(command)
            self._solution[:] = pending.solution
            self.last_solve_ms = pending.solve_ms
            self.last_iterations = pending.iterations
            baseline_force = pending.proposal.baseline.longitudinal_force_n
            self._solution[:-1] = self._solution[1:]
            self._solution[-1] = self._solution[-2]
            return self._compose_torque(
                state, command, baseline_force, residual_force_n, vertical_feedforward_force_n
            )
        control_state = self._control_state(state)
        reference = self._tracking_reference(command)
        tracking_error = control_state - reference
        predicted_zero_input_error = self._sx @ tracking_error
        gradient = 2.0 * self._su.T @ self._qbar @ predicted_zero_input_error

        def objective(sequence: np.ndarray) -> tuple[float, np.ndarray]:
            value = 0.5 * sequence @ self._hessian @ sequence + gradient @ sequence
            derivative = self._hessian @ sequence + gradient
            return float(value), derivative

        start = perf_counter()
        result = minimize(
            objective,
            self._solution,
            method="L-BFGS-B",
            jac=True,
            bounds=[(-self.longitudinal_force_limit_n, self.longitudinal_force_limit_n)]
            * self.horizon,
            options={"maxiter": 30, "ftol": 1e-8, "gtol": 1e-6},
        )
        self.last_solve_ms = 1e3 * (perf_counter() - start)
        self.last_iterations = int(result.nit)
        if result.success or np.isfinite(result.fun):
            self._solution[:] = result.x
        baseline_force = float(self._solution[0])
        self._solution[:-1] = self._solution[1:]
        self._solution[-1] = self._solution[-2]
        return self._compose_torque(
            state,
            command,
            baseline_force,
            residual_force_n,
            vertical_feedforward_force_n,
        )
