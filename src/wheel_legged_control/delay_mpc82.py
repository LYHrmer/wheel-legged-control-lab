"""Known input-delay prediction for the frozen planar MPC, without retuning.

Only the prediction state changes. The physical six-state cost/terminal weight,
issued-input R penalty, horizon, constraints, optimizer and warm start are inherited.
This module is an experimental candidate; the default controller is unchanged.
"""

from __future__ import annotations

import numpy as np

from .controllers import LinearMPCController, TrackingCommand
from .model import ACTION_SIZE, STATE_SIZE


def prediction_matrices(a, b, horizon, delay_steps):
    """Predict x[k+1:k+H+1] from [x,pending oldest,...] and H new inputs.

    Pending inputs and decision inputs are deviations from the same frozen
    equilibrium force. Only known delays0/2 are admitted by this candidate.
    """
    (a, b) = (np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    if (
        a.shape != (STATE_SIZE, STATE_SIZE)
        or b.shape != (STATE_SIZE, ACTION_SIZE)
        or (not np.isfinite(a).all())
        or (not np.isfinite(b).all())
    ):
        raise ValueError("finite original6x6 A and6x2 B required")
    if (
        type(horizon) is not int
        or horizon < 2
        or type(delay_steps) is not int
        or (delay_steps not in (0, 2))
    ):
        raise ValueError("horizon>=2 and known delay0/2 required")
    size = STATE_SIZE + delay_steps * ACTION_SIZE
    (aug_a, aug_b) = (np.zeros((size, size)), np.zeros((size, ACTION_SIZE)))
    aug_a[:STATE_SIZE, :STATE_SIZE] = a
    if delay_steps:
        aug_a[:STATE_SIZE, STATE_SIZE : STATE_SIZE + ACTION_SIZE] = b
        aug_a[STATE_SIZE : STATE_SIZE + ACTION_SIZE, STATE_SIZE + ACTION_SIZE :] = np.eye(
            ACTION_SIZE
        )
        aug_b[-ACTION_SIZE:] = np.eye(ACTION_SIZE)
    else:
        aug_b[:] = b
    sx = np.zeros((horizon * STATE_SIZE, size))
    su = np.zeros((horizon * STATE_SIZE, horizon * ACTION_SIZE))
    for row in range(horizon):
        part = slice(row * STATE_SIZE, (row + 1) * STATE_SIZE)
        sx[part] = np.linalg.matrix_power(aug_a, row + 1)[:STATE_SIZE]
        for column in range(row + 1):
            su[part, column * ACTION_SIZE : (column + 1) * ACTION_SIZE] = (
                np.linalg.matrix_power(aug_a, row - column) @ aug_b
            )[:STATE_SIZE]
    return (sx, su)


class DelayMPC82(LinearMPCController):
    """Original MPC with an explicit FIFO for zero-residual control issuance.

    compute is called once at reset and once after each physical transition,
    exactly as the frozen environment does. Its returned force is issued during
    the next env.step. The internal FIFO is advanced prospectively at compute;
    last_pending_before records the queue that the next env.step must consume.
    External residuals or repeated/skipped compute calls are outside this API.
    """

    def __init__(self, plant, *, delay_steps, horizon=20, q=None, r=None):
        if type(delay_steps) is not int or delay_steps not in (0, 2):
            raise ValueError("known delay must be0 or2")
        self.delay_steps = delay_steps
        super().__init__(plant, horizon=horizon, q=q, r=r)
        self._pending = np.empty((0, ACTION_SIZE))
        self.last_pending_before = self._pending.copy()
        self.last_pending_after = self._pending.copy()
        self.reset()

    def _prediction_matrices(self):
        if self.delay_steps == 0:
            return super()._prediction_matrices()
        return prediction_matrices(self.a, self.b, self.horizon, self.delay_steps)

    @property
    def pending_controls(self):
        return self._pending.copy()

    def reset(self, state=None):
        super().reset(state)
        self._pending = np.tile(self.plant.equilibrium_control, (self.delay_steps, 1))
        self.last_pending_before = self._pending.copy()
        self.last_pending_after = self._pending.copy()

    def compute(self, state, command: TrackingCommand):
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (STATE_SIZE,) or not np.isfinite(state).all():
            raise ValueError("finite physical6-state required")
        self.last_pending_before = self._pending.copy()
        augmented = (
            np.r_[state, (self._pending - self.plant.equilibrium_control).ravel()]
            if self.delay_steps
            else state
        )
        control = super().compute(augmented, command)
        if not np.isfinite(control).all():
            raise FloatingPointError("nonfinite issued control")
        if self.delay_steps:
            self._pending = np.vstack((self._pending[1:], control.copy()))
        self.last_pending_after = self._pending.copy()
        return control
