"""Hierarchical D1 controllers: contact VMC inside, LQR or MPC outside."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.linalg import block_diag, solve_discrete_are
from scipy.optimize import minimize

from .controllers import D1Command, D1VMCController
from .linear_model import D1SagittalLinearModel, identify_sagittal_model
from .model import D1Plant
from .state_estimation import D1StateEstimate

D1_OUTER_Q = np.diag((0.2, 400.0, 20.0, 30.0))
D1_OUTER_R = np.asarray(((1e-8,),), dtype=np.float64)
D1_LONGITUDINAL_FORCE_LIMIT_N = 180.0
D1_VERTICAL_RESIDUAL_LIMIT_N = 120.0
D1_VERTICAL_FEEDFORWARD_LIMIT_N = 500.0


class _D1HierarchicalController:
    """Shared command tracking and low-level torque composition."""

    baseline_name = "abstract"

    def __init__(self, plant: D1Plant, linear_model: D1SagittalLinearModel | None = None):
        self.control_dt = plant.control_dt
        self.linear_model = (
            identify_sagittal_model(plant.control_dt) if linear_model is None else linear_model
        )
        self.low_level = D1VMCController(plant)
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
    ) -> None:
        super().__init__(plant, linear_model)
        self.q = D1_OUTER_Q.copy() if q is None else np.asarray(q, dtype=np.float64)
        self.r = D1_OUTER_R.copy() if r is None else np.asarray(r, dtype=np.float64)
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

    def compute(
        self,
        command: D1Command,
        state: D1StateEstimate,
        residual_force_n: np.ndarray | None = None,
        *,
        vertical_feedforward_force_n: float = 0.0,
    ) -> np.ndarray:
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
        force = -float((self.gain @ (control_state - reference)).item())
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
    ) -> None:
        if horizon < 2:
            raise ValueError("horizon must be at least 2")
        super().__init__(plant, linear_model)
        self.horizon = horizon
        self.q = D1_OUTER_Q.copy() if q is None else np.asarray(q, dtype=np.float64)
        self.r = D1_OUTER_R.copy() if r is None else np.asarray(r, dtype=np.float64)
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

    def compute(
        self,
        command: D1Command,
        state: D1StateEstimate,
        residual_force_n: np.ndarray | None = None,
        *,
        vertical_feedforward_force_n: float = 0.0,
    ) -> np.ndarray:
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
