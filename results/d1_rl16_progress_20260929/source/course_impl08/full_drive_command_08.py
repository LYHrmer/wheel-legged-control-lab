"""Pure, versioned requests for a new full-drive training and GUI task.

The requested envelope is a construction bound, not a tracking qualification.
The caller must separately apply an evidence-backed capability gate before
putting a request into an actual control decision. This module has no engine,
controller, or model import.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

COMMAND_SCHEMA = "d1-full-drive-vx-vy-yaw-clearance-jump-v1"
MAX_FORWARD_REQUEST_MPS = 1.6
MAX_LATERAL_REQUEST_MPS = 0.12
MAX_YAW_REQUEST_RPS = 0.6
MIN_CLEARANCE_M = 0.38
MAX_CLEARANCE_M = 0.53


def _real(value: Real, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class FullDriveCommand:
    """Raw operator/training request; each field retains its physical unit."""

    forward_velocity_mps: float = 0.0
    lateral_velocity_mps: float = 0.0
    yaw_rate_rps: float = 0.0
    clearance_m: float = 0.455
    jump_requested: bool = False

    def __post_init__(self) -> None:
        for name in ("forward_velocity_mps", "lateral_velocity_mps", "yaw_rate_rps",
                     "clearance_m"):
            object.__setattr__(self, name, _real(getattr(self, name), name))
        if abs(self.forward_velocity_mps) > MAX_FORWARD_REQUEST_MPS:
            raise ValueError("forward request exceeds the new construction envelope")
        if abs(self.lateral_velocity_mps) > MAX_LATERAL_REQUEST_MPS:
            raise ValueError("lateral request exceeds the new construction envelope")
        if abs(self.yaw_rate_rps) > MAX_YAW_REQUEST_RPS:
            raise ValueError("yaw request exceeds the new construction envelope")
        if not MIN_CLEARANCE_M <= self.clearance_m <= MAX_CLEARANCE_M:
            raise ValueError("clearance request exceeds the new construction envelope")
        if type(self.jump_requested) is not bool:
            raise TypeError("jump_requested must be a bool token")


@dataclass(frozen=True, slots=True)
class QualifiedCommandCaps:
    """Externally justified subset of the construction envelope."""

    forward_mps: float
    reverse_mps: float
    lateral_mps: float
    yaw_rps: float
    jump_enabled: bool

    def __post_init__(self) -> None:
        for name, maximum in (("forward_mps", MAX_FORWARD_REQUEST_MPS),
                              ("reverse_mps", MAX_FORWARD_REQUEST_MPS),
                              ("lateral_mps", MAX_LATERAL_REQUEST_MPS),
                              ("yaw_rps", MAX_YAW_REQUEST_RPS)):
            value = _real(getattr(self, name), name)
            if not 0.0 <= value <= maximum:
                raise ValueError(f"{name} must be within the construction envelope")
            object.__setattr__(self, name, value)
        if type(self.jump_enabled) is not bool:
            raise TypeError("jump_enabled must be bool")

    def check(self, request: FullDriveCommand) -> None:
        """Reject unsupported requests, preserving the raw operator record."""
        if not isinstance(request, FullDriveCommand):
            raise TypeError("capability gate requires FullDriveCommand")
        if (request.forward_velocity_mps > self.forward_mps
                or -request.forward_velocity_mps > self.reverse_mps
                or abs(request.lateral_velocity_mps) > self.lateral_mps
                or abs(request.yaw_rate_rps) > self.yaw_rps
                or request.jump_requested and not self.jump_enabled):
            raise ValueError("request exceeds the separately qualified capability")
