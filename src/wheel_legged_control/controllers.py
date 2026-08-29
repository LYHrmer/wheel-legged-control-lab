"""LQR and constrained linear MPC baselines."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import block_diag, solve_discrete_are
from scipy.optimize import minimize

from .model import ACTION_SIZE, STATE_SIZE, WheelLeggedPlant


@dataclass(frozen=True)
class TrackingCommand:
    """Commanded forward speed and symmetric leg extension."""

    velocity_mps: float = 0.0
    leg_extension_m: float = 0.0


DEFAULT_Q = np.diag([0.4, 140.0, 90.0, 4.0, 14.0, 5.0])
DEFAULT_R = np.diag([0.025, 0.004])


class LQRController:
    """Infinite-horizon discrete LQR around the upright nominal pose."""

    def __init__(
        self,
        plant: WheelLeggedPlant,
        q: np.ndarray | None = None,
        r: np.ndarray | None = None,
    ) -> None:
        self.plant = plant
        self.a, self.b = plant.linearize()
        self.q = DEFAULT_Q.copy() if q is None else np.asarray(q, dtype=np.float64)
        self.r = DEFAULT_R.copy() if r is None else np.asarray(r, dtype=np.float64)
        riccati = solve_discrete_are(self.a, self.b, self.q, self.r)
        self.gain = np.linalg.solve(
            self.r + self.b.T @ riccati @ self.b,
            self.b.T @ riccati @ self.a,
        )
        self.closed_loop_eigenvalues = np.linalg.eigvals(self.a - self.b @ self.gain)
        self._x_reference = 0.0

    def reset(self, state: np.ndarray | None = None) -> None:
        self._x_reference = 0.0 if state is None else float(state[0])

    def reference(self, state: np.ndarray, command: TrackingCommand) -> np.ndarray:
        # A bounded moving position target removes steady velocity error without
        # allowing an unreachable position reference to wind up after a disturbance.
        self._x_reference += command.velocity_mps * self.plant.control_dt
        self._x_reference = float(
            np.clip(self._x_reference, state[0] - 0.45, state[0] + 0.45)
        )
        return np.array(
            [
                self._x_reference,
                0.0,
                command.leg_extension_m,
                command.velocity_mps,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

    def compute(self, state: np.ndarray, command: TrackingCommand) -> np.ndarray:
        reference = self.reference(state, command)
        delta = -self.gain @ (np.asarray(state) - reference)
        control = self.plant.equilibrium_control + delta
        return np.clip(control, self.plant.actuator_low, self.plant.actuator_high)


class LinearMPCController:
    """Receding-horizon linear MPC with actuator box constraints."""

    def __init__(
        self,
        plant: WheelLeggedPlant,
        horizon: int = 20,
        q: np.ndarray | None = None,
        r: np.ndarray | None = None,
    ) -> None:
        if horizon < 2:
            raise ValueError("horizon must be at least 2")
        self.plant = plant
        self.horizon = horizon
        self.a, self.b = plant.linearize()
        self.q = DEFAULT_Q.copy() if q is None else np.asarray(q, dtype=np.float64)
        self.r = DEFAULT_R.copy() if r is None else np.asarray(r, dtype=np.float64)
        self.terminal_q = solve_discrete_are(self.a, self.b, self.q, self.r)
        self._sx, self._su = self._prediction_matrices()
        self._qbar = block_diag(*([self.q] * (horizon - 1)), self.terminal_q)
        self._rbar = np.kron(np.eye(horizon), self.r)
        self._hessian = 2.0 * (self._su.T @ self._qbar @ self._su + self._rbar)
        self._solution = np.zeros(horizon * ACTION_SIZE, dtype=np.float64)
        self._x_reference = 0.0
        self.last_solve_ms = 0.0
        self.last_iterations = 0

    def _prediction_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        sx = np.zeros((self.horizon * STATE_SIZE, STATE_SIZE))
        su = np.zeros((self.horizon * STATE_SIZE, self.horizon * ACTION_SIZE))
        for row in range(self.horizon):
            sx[row * STATE_SIZE : (row + 1) * STATE_SIZE] = np.linalg.matrix_power(
                self.a, row + 1
            )
            for column in range(row + 1):
                block = np.linalg.matrix_power(self.a, row - column) @ self.b
                su[
                    row * STATE_SIZE : (row + 1) * STATE_SIZE,
                    column * ACTION_SIZE : (column + 1) * ACTION_SIZE,
                ] = block
        return sx, su

    def reset(self, state: np.ndarray | None = None) -> None:
        self._solution[:] = 0.0
        self._x_reference = 0.0 if state is None else float(state[0])
        self.last_solve_ms = 0.0
        self.last_iterations = 0

    def _target_trajectory(self, state: np.ndarray, command: TrackingCommand) -> np.ndarray:
        self._x_reference += command.velocity_mps * self.plant.control_dt
        self._x_reference = float(
            np.clip(self._x_reference, state[0] - 0.45, state[0] + 0.45)
        )
        targets = np.zeros((self.horizon, STATE_SIZE), dtype=np.float64)
        for index in range(self.horizon):
            targets[index] = (
                self._x_reference + index * self.plant.control_dt * command.velocity_mps,
                0.0,
                command.leg_extension_m,
                command.velocity_mps,
                0.0,
                0.0,
            )
        return targets.reshape(-1)

    def compute(self, state: np.ndarray, command: TrackingCommand) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        target = self._target_trajectory(state, command)
        prediction_error = self._sx @ state - target
        gradient_offset = 2.0 * self._su.T @ self._qbar @ prediction_error

        equilibrium = self.plant.equilibrium_control
        lower_one = self.plant.actuator_low - equilibrium
        upper_one = self.plant.actuator_high - equilibrium
        bounds = list(zip(np.tile(lower_one, self.horizon), np.tile(upper_one, self.horizon)))

        def objective(candidate: np.ndarray) -> float:
            return float(0.5 * candidate @ self._hessian @ candidate + gradient_offset @ candidate)

        def gradient(candidate: np.ndarray) -> np.ndarray:
            return self._hessian @ candidate + gradient_offset

        start = perf_counter()
        result = minimize(
            objective,
            self._solution,
            jac=gradient,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 35, "ftol": 1e-8, "gtol": 1e-6},
        )
        self.last_solve_ms = (perf_counter() - start) * 1e3
        self.last_iterations = int(result.nit)
        if np.all(np.isfinite(result.x)):
            self._solution = result.x

        delta = self._solution[:ACTION_SIZE].copy()
        self._solution[:-ACTION_SIZE] = self._solution[ACTION_SIZE:]
        self._solution[-ACTION_SIZE:] = self._solution[-2 * ACTION_SIZE : -ACTION_SIZE]
        control = equilibrium + delta
        return np.clip(control, self.plant.actuator_low, self.plant.actuator_high)
