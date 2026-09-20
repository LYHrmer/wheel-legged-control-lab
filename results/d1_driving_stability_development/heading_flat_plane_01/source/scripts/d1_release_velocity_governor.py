"""Diagnostic-only release-velocity reference shaping for the D1 stop experiment.

This module serves a bounded, fixed-deceleration forward-velocity reference tail
when the raw user forward request becomes exactly zero.  It is a single-variable
diagnostic instrument for the proposed stopping experiment, not a validated or
proven stopping fix, and it must not be presented as an accepted control
improvement.  The original user intent (``raw_command``) and the shaped
reference (``served_command``) are kept strictly distinct.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from wheel_legged_control.d1.control_loop import D1MotionCommand

RELEASE_GOVERNOR_SCHEMA = "d1-release-velocity-governor-v1"

__all__ = [
    "RELEASE_GOVERNOR_SCHEMA",
    "ReleaseVelocityGovernor",
    "ReleaseVelocityRecord",
]


@dataclass(frozen=True, slots=True)
class ReleaseVelocityRecord:
    """Immutable evidence for one governor decision tick."""

    schema: str
    simulation_time_s: float
    actual_dt_s: float
    raw_command: D1MotionCommand
    served_command: D1MotionCommand
    release_active: bool


def _reject_bool(value: object) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("booleans are not accepted as numeric scalars")


def _as_finite_real(value: object, what: str) -> float:
    _reject_bool(value)
    if not isinstance(value, Real):
        raise TypeError(f"{what} must be a real scalar, got {type(value)!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite")
    return number


class ReleaseVelocityGovernor:
    """Shape only the forward-velocity reference after an exact zero request."""

    def __init__(self, deceleration_mps2: float | None = 0.5) -> None:
        if deceleration_mps2 is None:
            self._deceleration_mps2: float | None = None
        else:
            value = _as_finite_real(deceleration_mps2, "deceleration_mps2")
            if value <= 0.0:
                raise ValueError("deceleration_mps2 must be positive")
            self._deceleration_mps2 = value
        self.reset()

    @property
    def deceleration_mps2(self) -> float | None:
        """Configured release deceleration in m/s^2, or ``None`` for exact bypass."""
        return self._deceleration_mps2

    def reset(self) -> None:
        """Clear all history; the next call behaves like a first call."""
        self._last_record: ReleaseVelocityRecord | None = None
        self._last_time_s: float | None = None
        self._last_served_vx: float = 0.0

    def advance(
        self, raw_command: D1MotionCommand, simulation_time_s: float
    ) -> ReleaseVelocityRecord:
        """Validate inputs and return the record for this decision tick."""
        # --- validation only; no state is touched before it fully succeeds ---
        if not isinstance(raw_command, D1MotionCommand):
            raise TypeError(
                f"raw_command must be a D1MotionCommand, got {type(raw_command)!r}"
            )
        # Re-run the command's own field validation even if the frozen instance
        # was constructed by bypassing __init__/__post_init__.
        D1MotionCommand.__post_init__(raw_command)

        time_s = _as_finite_real(simulation_time_s, "simulation_time_s")
        if time_s < 0.0:
            raise ValueError("simulation_time_s must be non-negative")

        if self._last_time_s is not None:
            if time_s < self._last_time_s:
                raise ValueError(
                    "simulation_time_s must not move backwards "
                    f"({time_s!r} < {self._last_time_s!r})"
                )
            if time_s == self._last_time_s:
                assert self._last_record is not None
                if raw_command == self._last_record.raw_command:
                    return self._last_record
                raise ValueError(
                    "conflicting raw_command served at an already recorded "
                    f"simulation_time_s {time_s!r}"
                )
            dt_s = time_s - self._last_time_s
        else:
            dt_s = 0.0

        # --- decision ---
        served_command = raw_command
        release_active = False
        raw_vx = raw_command.forward_velocity_mps

        if self._deceleration_mps2 is not None and dt_s > 0.0 and raw_vx == 0.0:
            tail_vx = self._release_tail(dt_s)
            if tail_vx != raw_vx:
                served_command = dataclasses.replace(
                    raw_command, forward_velocity_mps=tail_vx
                )
                release_active = True

        record = ReleaseVelocityRecord(
            schema=RELEASE_GOVERNOR_SCHEMA,
            simulation_time_s=time_s,
            actual_dt_s=dt_s,
            raw_command=raw_command,
            served_command=served_command,
            release_active=release_active,
        )
        self._last_record = record
        self._last_time_s = time_s
        self._last_served_vx = float(served_command.forward_velocity_mps)
        return record

    def _release_tail(self, dt_s: float) -> float:
        previous = self._last_served_vx
        assert self._deceleration_mps2 is not None
        magnitude = abs(previous) - self._deceleration_mps2 * dt_s
        if magnitude <= 0.0:
            return 0.0
        return magnitude if previous > 0.0 else -magnitude
