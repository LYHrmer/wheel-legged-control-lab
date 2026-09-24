"""Pure common wheel-P correction for the fixed base-COM-speed candidate."""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np

WHEELS = (3, 7, 11, 15)
WHEEL_KP = 2.2
WHEEL_RADIUS_M = 0.087


def finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in ("f", "i", "u"):
        raise TypeError(f"{label} must be a real numeric array with shape {shape}")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result


def finite_scalar(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def body_common_increment(wheel_omega_rad_s: Any, base_linear_velocity_body: Any,
                          *, raw_forward_mps: Any) -> dict[str, Any]:
    """Apply one exact wheel-only P increment, gated by the bound raw command."""
    omega = finite_array(wheel_omega_rad_s, (4,), "wheel_omega_rad_s")
    body_velocity = finite_array(base_linear_velocity_body, (3,),
                                 "base_linear_velocity_body")
    forward = finite_scalar(raw_forward_mps, "raw_forward_mps")
    active = forward != 0.0
    mean_omega = float(np.mean(omega))
    body_equivalent = float(body_velocity[0] / WHEEL_RADIUS_M)
    common_error = float(mean_omega - body_equivalent)
    scalar_delta = float(WHEEL_KP * common_error) if active else 0.0
    delta = np.zeros(16, dtype=np.float64)
    delta[list(WHEELS)] = scalar_delta
    if not (np.isfinite(mean_omega) and np.isfinite(body_equivalent)
            and np.isfinite(common_error) and np.isfinite(delta).all()):
        raise ValueError("common wheel-P arithmetic is nonfinite")
    return {
        "active": active,
        "mean_omega_rad_s": mean_omega,
        "body_equivalent_omega_rad_s": body_equivalent,
        "common_error_rad_s": common_error,
        "scalar_delta_nm": scalar_delta,
        "delta_torque_nm": delta,
    }
