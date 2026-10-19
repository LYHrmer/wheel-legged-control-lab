"""Pure, training-only C46 tracking-shape transform.

The frozen nine-term interface is kept: only ``tracking_vx`` and ``tracking_yaw`` change value,
never their names or positions. The 42 deficit transform only multiplies the exponential deficit,
which stays flat for the small errors that dominate the evaluation (exp(-q) ~ 1 - q near q = 0).
This transform adds a logarithmic squared-error term:

    tracking_vx  = 2 + alpha*(2*exp(-qx) - 2) - beta*ln(1 + min(qx, cap))
    tracking_yaw = 1 + alpha*(exp(-qy) - 1) - beta*ln(1 + min(qy, cap))

with qx = ((vx - command_vx)/0.25)^2 and qy = ((yaw - command_yaw)/0.4)^2, the exact normalized
errors of the evaluation metric. Near zero the new term is linear in the evaluation SSE, so small
errors keep a gradient; its growth is logarithmic and capped at q=4, so the largest possible extra
penalty is 8*ln(5) ~ 12.9, the same order as the -10 termination term. At the rough-terrain error
of the completed runs (qx ~ 0.028) the tracking gradient rises by about 51% relative to the 42
transform alone. ``beta == 0`` is bit for bit the 42 transform. ``beta == 0`` is bit for bit the 42 transform. Evaluation stays on the
original reward.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

from b22_tracking_reward42 import REWARD_TERMS42, apply_tracking_deficit42

REWARD_SCHEMA46 = "d1-b22-training-tracking-log-sse-beta46-v1"
BETAS46 = (0, 8)
CAP46 = 4.0


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def normalized_errors46(
    forward_mps: float, yaw_rps: float, command_forward_mps: float, command_yaw_rps: float
) -> tuple[float, float]:
    """The evaluation metric's own normalized squared errors."""
    return (
        (_finite(forward_mps, "forward_mps") - _finite(command_forward_mps, "command_forward")) ** 2
        / 0.25**2,
        (_finite(yaw_rps, "yaw_rps") - _finite(command_yaw_rps, "command_yaw")) ** 2 / 0.4**2,
    )


def apply_tracking_log_sse46(
    original_terms: Mapping[str, Real],
    qx: float,
    qy: float,
    *,
    alpha: Real = 8,
    beta: Real = 8,
) -> dict[str, Real]:
    """Apply the registered alpha transform, then the logarithmic squared-error term.

    Inputs must be the original nine terms (not already transformed). ``beta == 0`` returns the
    exact alpha-transform result; ``alpha == 1`` keeps the original floating-point values there.
    """
    alpha_value, beta_value = _finite(alpha, "alpha"), _finite(beta, "beta")
    if beta_value not in BETAS46:
        raise ValueError("beta must be the registered value 0 or 8")
    qx_value, qy_value = _finite(qx, "qx"), _finite(qy, "qy")
    if qx_value < 0 or qy_value < 0:
        raise ValueError("normalized errors must be non-negative")
    result = apply_tracking_deficit42(original_terms, alpha_value)
    if beta_value == 0:
        return result
    vx = 2.0 * math.exp(-qx_value)
    yaw = math.exp(-qy_value)
    if not (
        math.isclose(original_terms["tracking_vx"], vx, rel_tol=0, abs_tol=1e-12)
        and math.isclose(original_terms["tracking_yaw"], yaw, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError("original tracking terms disagree with the supplied normalized errors")
    result["tracking_vx"] = result["tracking_vx"] - beta_value * math.log1p(min(qx_value, CAP46))
    result["tracking_yaw"] = result["tracking_yaw"] - beta_value * math.log1p(min(qy_value, CAP46))
    return result


__all__ = (
    "BETAS46",
    "CAP46",
    "REWARD_SCHEMA46",
    "REWARD_TERMS42",
    "apply_tracking_log_sse46",
    "normalized_errors46",
)
