"""C33 finite-jerk body translation reference; no plant or controller imports."""
from __future__ import annotations

import math


RAMP_FRACTION33 = 0.05  # fraction of the whole move duration
PEAK_ACCEL33 = 4.0 / (1.0 - 2.0 * RAMP_FRACTION33)
JERK33 = PEAK_ACCEL33 / RAMP_FRACTION33
ALPHAS33 = (0.85, 0.925, 1.0)
SHAPE33 = "symmetric_trapezoidal_acceleration_r0.05"


def curve33(time_s: float, duration_s: float) -> tuple[float, float, float]:
    """Position fraction, velocity/s and acceleration/s² of the same curve.

    Four equal ramps make acceleration continuous and jerk bounded. Clamping
    includes zero velocity and acceleration at both ends, even past the move.
    """
    duration = float(duration_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("body move duration must be positive and finite")
    u = float(time_s) / duration
    if not math.isfinite(u):
        raise ValueError("body move time must be finite")
    if u <= 0.0:
        return 0.0, 0.0, 0.0
    if u >= 1.0:
        return 1.0, 0.0, 0.0
    r = RAMP_FRACTION33
    knots = (0.0, r, 0.5-r, 0.5+r, 1.0-r, 1.0)
    signs = (1.0, -1.0, -1.0, 1.0, 1.0, -1.0)
    hinges = [max(0.0, u-knot) for knot in knots]
    position = (JERK33 / 6.0) * sum(s*h**3 for s, h in zip(signs, hinges))
    velocity = (JERK33 / (2.0*duration)) * sum(
        s*h*h for s, h in zip(signs, hinges))
    acceleration = (JERK33 / (duration*duration)) * sum(
        s*h for s, h in zip(signs, hinges))
    return position, velocity, acceleration


def duration33(span_xy_m, com_height_m: float, zmp_budget_m: float,
               alpha: float) -> float:
    """Spend at most alpha of the original horizontal acceleration budget."""
    if float(alpha) not in ALPHAS33:
        raise ValueError("C33 alpha must be one of the frozen pilot coefficients")
    if len(span_xy_m) != 2:
        raise ValueError("body span must have two horizontal components")
    x, y = (float(value) for value in span_xy_m)
    height, budget = float(com_height_m), float(zmp_budget_m)
    if (not all(math.isfinite(value) for value in (x, y, height, budget))
            or height <= 0.0 or budget <= 0.0):
        raise ValueError("body span, COM height and budget must be finite")
    distance = math.hypot(x, y)
    return max(0.30, math.sqrt(
        PEAK_ACCEL33 * distance * height / (9.81 * budget * alpha)))
