"""Pure, training-only C42 tracking-deficit transform.

This module neither imports a learning/simulation framework nor changes the
base nine-term interface. A future versioned training environment must call it
once after computing the original world-upright reward; evaluation stays on
the original reward. This helper alone does not implement a training entry.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

REWARD_SCHEMA42 = "d1-b22-training-tracking-deficit-alpha42-v1"
REWARD_TERMS42 = (
    "tracking_vx", "tracking_vy", "tracking_yaw", "height", "attitude",
    "actual_torque", "action_change", "leg_speed", "termination",
)
ALPHAS42 = (1, 8)


def _finite_number42(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not a boolean")
    try:
        finite = math.isfinite(value)
    except OverflowError as error:
        raise ValueError(f"{name} is not a finite float") from error
    if not finite:
        raise ValueError(f"{name} must be finite")


def apply_tracking_deficit42(
    original_terms: Mapping[str, Real], alpha: Real,
) -> dict[str, Real]:
    """Copy the original nine terms and scale only vx/yaw tracking deficits.

    The registered comparison is alpha=1 versus alpha=8. For original terms
    vx=2*exp(-qx), yaw=exp(-qy), the new terms are 2+alpha*(vx-2) and
    1+alpha*(yaw-1). All seven other values and the insertion order survive
    unchanged. Alpha=1 takes an identity branch to preserve the original
    floating-point values, including deficits too small for subtract/add.

    Inputs must be original reward terms, rather than already transformed
    terms. No rollout, policy action, observation, terminal flag or metadata
    is accepted or modified here.
    """
    _finite_number42(alpha, "alpha")
    if alpha not in ALPHAS42:
        raise ValueError("alpha must be the registered value 1 or 8")
    if not isinstance(original_terms, Mapping):
        raise TypeError("original_terms must be a mapping")
    if set(original_terms) != set(REWARD_TERMS42):
        raise ValueError("original reward must contain exactly the nine declared terms")
    for name, value in original_terms.items():
        _finite_number42(value, name)
    if not 0 <= original_terms["tracking_vx"] <= 2:
        raise ValueError("original tracking_vx must be in [0, 2]")
    if not 0 <= original_terms["tracking_yaw"] <= 1:
        raise ValueError("original tracking_yaw must be in [0, 1]")
    result = dict(original_terms)
    if alpha == 1:
        return result
    result["tracking_vx"] = 2. + float(alpha) * (float(original_terms["tracking_vx"]) - 2.)
    result["tracking_yaw"] = 1. + float(alpha) * (float(original_terms["tracking_yaw"]) - 1.)
    return result
