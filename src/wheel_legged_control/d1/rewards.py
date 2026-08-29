"""Inspectable reward terms for the full-body D1 residual policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .model import JOINT_POSITION_HIGH, JOINT_POSITION_LOW


@dataclass(frozen=True)
class D1RewardBreakdown:
    velocity_tracking: float
    upright: float
    height_tracking: float
    alive: float
    residual_effort: float
    residual_smoothness: float
    joint_limit_penalty: float
    contact_penalty: float
    termination_penalty: float

    @property
    def total(self) -> float:
        return float(sum(asdict(self).values()))

    def as_dict(self) -> dict[str, float]:
        return asdict(self) | {"total": self.total}


def calculate_d1_reward(
    *,
    forward_velocity_mps: float,
    roll_rad: float,
    pitch_rad: float,
    base_height_m: float,
    joint_position: np.ndarray,
    command_velocity_mps: float,
    command_height_m: float,
    normalized_action: np.ndarray,
    previous_normalized_action: np.ndarray,
    undesired_contacts: int,
    terminated: bool,
) -> D1RewardBreakdown:
    """Compute each weighted term without hidden environment-side bonuses."""

    joints = np.asarray(joint_position, dtype=np.float64)
    action = np.asarray(normalized_action, dtype=np.float64)
    previous = np.asarray(previous_normalized_action, dtype=np.float64)
    if joints.shape != (16,) or action.shape != (2,) or previous.shape != (2,):
        raise ValueError("expected 16 joint positions and two-dimensional residual actions")

    velocity_error = forward_velocity_mps - command_velocity_mps
    height_error = base_height_m - command_height_m
    finite_limits = np.isfinite(JOINT_POSITION_LOW)
    soft_low = JOINT_POSITION_LOW.copy()
    soft_high = JOINT_POSITION_HIGH.copy()
    centers = 0.5 * (soft_low[finite_limits] + soft_high[finite_limits])
    half_ranges = 0.45 * (soft_high[finite_limits] - soft_low[finite_limits])
    soft_low[finite_limits] = centers - half_ranges
    soft_high[finite_limits] = centers + half_ranges
    lower_violation = np.maximum(soft_low[finite_limits] - joints[finite_limits], 0.0)
    upper_violation = np.maximum(joints[finite_limits] - soft_high[finite_limits], 0.0)

    return D1RewardBreakdown(
        velocity_tracking=float(2.0 * np.exp(-((velocity_error / 0.35) ** 2))),
        upright=float(
            1.0 * np.exp(-((roll_rad / 0.22) ** 2) - ((pitch_rad / 0.22) ** 2))
        ),
        height_tracking=float(0.7 * np.exp(-((height_error / 0.045) ** 2))),
        alive=0.2,
        residual_effort=-0.04 * float(action @ action),
        residual_smoothness=-0.025 * float((action - previous) @ (action - previous)),
        joint_limit_penalty=-1.5
        * float(lower_violation @ lower_violation + upper_violation @ upper_violation),
        contact_penalty=-0.5 * float(undesired_contacts),
        termination_penalty=-10.0 if terminated else 0.0,
    )
