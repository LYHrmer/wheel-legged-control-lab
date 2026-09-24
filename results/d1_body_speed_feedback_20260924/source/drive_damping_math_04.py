"""Pure fixed driving-phase longitudinal leg damping arithmetic.

The finite limits mirror the frozen D1 model arrays. The engine-facing
controller asserts exact equality to those original arrays before use.
"""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np

GAIN_NSPM = 126.4374005337902
WHEEL_INDICES = (3, 7, 11, 15)
TORQUE_LIMIT = np.tile((80.0, 80.0, 80.0, 12.0), 4)
VELOCITY_LIMIT = np.tile((20.0, 20.0, 20.0, 30.0), 4)
POSITION_LOW = np.tile((-0.785398, -1.8326, -2.775, -np.inf), 4)
POSITION_HIGH = np.tile((0.785398, 3.40339, -0.855, np.inf), 4)
for _array in (TORQUE_LIMIT, VELOCITY_LIMIT, POSITION_LOW, POSITION_HIGH):
    _array.setflags(write=False)


def _finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in ("f", "i", "u") or array.shape != shape:
        raise TypeError(f"{label} must be a real numeric array of shape {shape}")
    checked = np.asarray(array, dtype=np.float64)
    if not np.isfinite(checked).all():
        raise ValueError(f"{label} must be finite")
    return checked


def _finite_scalar(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real numeric scalar")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def drive_increment(rotation: Any, jacobian: Any, joint_velocity: Any,
                    *, raw_forward_mps: Any) -> dict[str, Any]:
    """Return the old Jx arithmetic gated by the consumed raw forward command."""
    forward = _finite_scalar(raw_forward_mps, "raw_forward_mps")
    matrix = _finite_array(rotation, (3, 3), "rotation")
    feet = _finite_array(jacobian, (4, 3, 4), "jacobian")
    velocity = _finite_array(joint_velocity, (16,), "joint_velocity")
    projected = np.zeros((4, 3))
    relative = np.zeros(4)
    delta = np.zeros(16)
    forward_axis = matrix[:, 0]
    active = forward != 0.0
    for leg in range(4):
        jx = forward_axis @ feet[leg, :, :3]
        projected[leg] = jx
        relative[leg] = float(jx @ velocity[4 * leg : 4 * leg + 3])
        if active:
            delta[4 * leg : 4 * leg + 3] = -GAIN_NSPM * jx * relative[leg]
    delta[list(WHEEL_INDICES)] = 0.0
    if not (np.isfinite(projected).all() and np.isfinite(relative).all()
            and np.isfinite(delta).all()):
        raise ValueError("driving leg damping arithmetic produced nonfinite values")
    return {
        "active": active,
        "projected_jx": projected,
        "relative_forward_mps": relative,
        "delta_torque_nm": delta,
        "raw_joint_power_w": float(np.sum(delta * velocity)),
    }


def protect_requested(requested_torque_nm: Any, joint_position: Any,
                      joint_velocity: Any) -> tuple[np.ndarray, np.ndarray]:
    """Repeat the frozen clip, then outward position and speed suppression."""
    requested = _finite_array(requested_torque_nm, (16,), "requested_torque_nm")
    position = _finite_array(joint_position, (16,), "joint_position")
    velocity = _finite_array(joint_velocity, (16,), "joint_velocity")
    safe = np.clip(requested, -TORQUE_LIMIT, TORQUE_LIMIT)
    upper_outward = (position >= POSITION_HIGH) & (safe > 0)
    lower_outward = (position <= POSITION_LOW) & (safe < 0)
    speed_outward = (np.abs(velocity) >= VELOCITY_LIMIT) & (safe * velocity > 0)
    safe[upper_outward | lower_outward | speed_outward] = 0.0
    if not np.isfinite(safe).all():
        raise ValueError("protected driving torque is nonfinite")
    return safe, safe != requested
