"""Transparent reward definition shared by training and teaching examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RewardBreakdown:
    """Weighted reward components returned for inspection and logging."""

    balance: float
    velocity_tracking: float
    height_tracking: float
    residual_penalty: float
    termination_penalty: float

    @property
    def total(self) -> float:
        return (
            self.balance
            + self.velocity_tracking
            + self.height_tracking
            + self.residual_penalty
            + self.termination_penalty
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self) | {"total": self.total}


def calculate_reward(
    state: np.ndarray,
    command_velocity_mps: float,
    command_leg_extension_m: float,
    normalized_action: np.ndarray,
    terminated: bool = False,
) -> RewardBreakdown:
    """Calculate every reward term without hiding weights inside the environment."""

    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(normalized_action, dtype=np.float64)
    if state.shape != (6,) or action.shape != (2,):
        raise ValueError("expected state shape (6,) and action shape (2,)")

    velocity_error = state[3] - command_velocity_mps
    height_error = state[2] - command_leg_extension_m
    return RewardBreakdown(
        balance=float(0.45 * np.exp(-18.0 * state[1] ** 2 - 0.35 * state[4] ** 2)),
        velocity_tracking=float(0.35 * np.exp(-1.8 * velocity_error**2)),
        height_tracking=float(
            0.20 * np.exp(-90.0 * height_error**2 - 0.08 * state[5] ** 2)
        ),
        residual_penalty=-0.04 * float(action @ action),
        termination_penalty=-10.0 if terminated else 0.0,
    )
