"""One scalar load-disturbance observer on the registered DelayMPC82 skeleton.

The controller sees measured states and its own issued-force FIFO only. A
controller-side equilibrium view shifts the prediction, input penalty, bounds
and warm start consistently; the physical plant and nominal A/B never change.
This experimental mechanism has no performance or deployment qualification.
"""

from __future__ import annotations

from math import exp

import numpy as np

from .delay_mpc82 import DelayMPC82
from .model import STATE_SIZE

OBSERVER_TAU_S = 0.2
DISTURBANCE_LIMIT_N = 60.0
OBSERVER_EXTENSION_LIMIT_M = 0.115


class _ControllerPlantView83:
    """Snapshot the nominal controller interface; retain no live plant handle."""

    def __init__(self, plant):
        self.control_dt = float(plant.control_dt)
        self._a, self._b = (value.copy() for value in plant.linearize())
        self._equilibrium = plant.equilibrium_control.copy()
        self._low = plant.actuator_low.copy()
        self._high = plant.actuator_high.copy()

    def linearize(self):
        return self._a.copy(), self._b.copy()

    @property
    def equilibrium_control(self):
        return self._equilibrium.copy()

    @property
    def actuator_low(self):
        return self._low.copy()

    @property
    def actuator_high(self):
        return self._high.copy()


class OffsetMPC83(DelayMPC82):
    """Estimate one equivalent vertical force, with fixed filtering and guards.

    At compute k, the previous transition used the oldest absolute force from
    compute k-1's FIFO (or its issued force at delay zero). The innovation is
    always relative to the *nominal* equilibrium, independent of the estimate:

        r = y[k] - A@y[k-1] - B@(applied[k-1] - nominal_eq)
        raw_d = r[5] / B[5,1]
        d = clip((1-alpha)*d + alpha*raw_d, -60, 60)

    Both measured extension endpoints must lie strictly inside +/-0.115 m;
    otherwise the estimate freezes to avoid fitting joint-limit reaction force.
    The first compute has no transition observation. reset discards all history.

    estimated_eq = nominal_eq - [0,d] makes B@(u-estimated_eq) identical to
    B@(u-nominal_eq)+B[:,1]*d. The original optimizer consequently penalizes
    corrections around the inferred equilibrium, with unchanged absolute boxes.
    This is consistent for a constant, matched input disturbance; changing mass,
    nonlinear coupling, contact, noise and finite-horizon tracking remain risks.
    """

    def __init__(self, plant, *, delay_steps, horizon=20, q=None, r=None):
        view = _ControllerPlantView83(plant)
        if not np.isfinite(view.control_dt) or view.control_dt <= 0:
            raise ValueError("finite positive control timestep required")
        self._nominal_equilibrium = view.equilibrium_control
        self.observer_alpha = 1.0 - exp(-view.control_dt / OBSERVER_TAU_S)
        self.disturbance_force_n = 0.0
        super().__init__(view, delay_steps=delay_steps, horizon=horizon, q=q, r=r)
        if not np.isfinite(self.b[5, 1]) or self.b[5, 1] <= 0.0:
            raise ValueError("positive nominal vertical input gain B[5,1] required")

    @property
    def nominal_equilibrium_control(self):
        return self._nominal_equilibrium.copy()

    @property
    def estimated_equilibrium_control(self):
        return self.plant.equilibrium_control

    def reset(self, state=None):
        self.disturbance_force_n = 0.0
        self.plant._equilibrium = self._nominal_equilibrium.copy()
        self._previous_measurement = None
        self._expected_applied_control = None
        self.last_nominal_prediction = None
        self.last_residual = None
        self.last_transition_applied_control = None
        self.last_raw_disturbance_force_n = None
        self.observer_updated = False
        self.observer_freeze_reason = "no_previous_transition"
        super().reset(state)

    def _observe_disturbance(self, state):
        self.observer_updated = False
        if self._previous_measurement is None:
            return
        applied = self._expected_applied_control.copy()
        prediction = (
            self.a @ self._previous_measurement
            + self.b @ (applied - self._nominal_equilibrium)
        )
        residual = state - prediction
        raw = float(residual[5] / self.b[5, 1])
        if (
            not np.isfinite(prediction).all()
            or not np.isfinite(residual).all()
            or not np.isfinite(raw)
        ):
            raise FloatingPointError("nonfinite nominal-model disturbance innovation")
        self.last_transition_applied_control = applied
        self.last_nominal_prediction = prediction.copy()
        self.last_residual = residual.copy()
        self.last_raw_disturbance_force_n = raw
        if (
            abs(self._previous_measurement[2]) >= OBSERVER_EXTENSION_LIMIT_M
            or abs(state[2]) >= OBSERVER_EXTENSION_LIMIT_M
        ):
            self.observer_freeze_reason = "measured_extension_near_limit"
            return
        updated = (
            (1.0 - self.observer_alpha) * self.disturbance_force_n
            + self.observer_alpha * raw
        )
        if not np.isfinite(updated):
            raise FloatingPointError("nonfinite filtered disturbance")
        updated = float(np.clip(updated, -DISTURBANCE_LIMIT_N, DISTURBANCE_LIMIT_N))
        old_equilibrium = self.plant.equilibrium_control
        new_equilibrium = self._nominal_equilibrium.copy()
        new_equilibrium[1] -= updated
        if not np.array_equal(new_equilibrium, old_equilibrium):
            # Preserve the warm start's absolute planned forces under the new
            # coordinate origin. FIFO entries already store absolute forces.
            self._solution.reshape(-1, 2)[:] += old_equilibrium - new_equilibrium
        self.plant._equilibrium = new_equilibrium
        self.disturbance_force_n = updated
        self.observer_updated = True
        self.observer_freeze_reason = "updated"

    def compute(self, state, command):
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (STATE_SIZE,) or not np.isfinite(state).all():
            raise ValueError("finite physical6-state required")
        self._observe_disturbance(state)
        issued = super().compute(state, command)
        self._previous_measurement = state.copy()
        self._expected_applied_control = (
            self.last_pending_before[0].copy() if self.delay_steps else issued.copy()
        )
        return issued
