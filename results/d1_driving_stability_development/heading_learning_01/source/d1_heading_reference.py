"""Simulation-time heading reference and heading feedback for the D1 wheel-legged robot.

This module deliberately separates two concerns that are easy to entangle in a
GUI teleoperation loop:

1. ``heading_feedback`` is a *pure* function.  Given a desired heading, the
   measured heading and the measured yaw rate it returns the yaw-rate command
   that servos the base onto that heading.  It holds no memory, never reads a
   plant, a clock, the filesystem or any global state, and never advances
   physics.  With ``user_yaw_rate_rps=0.0`` and the default gains
   (``kp=2.0``, ``kd=0.4``, ``limit_rps=1.0``) it reproduces exactly the
   arithmetic of the existing GUI heading hold.

2. ``HeadingReference`` owns the *explicit* integration of an operator yaw-rate
   command into a heading target, using the simulation clock supplied by the
   caller.  It never samples the measured yaw, so the reference cannot silently
   drift along with the actual base attitude: heading error stays observable.

Sample timing (left-hold / zero-order hold on the previous sample)
-----------------------------------------------------------------
``advance(rate, t)`` first integrates the rate that was *held* since the last
published timestamp over the elapsed interval, and only then stores ``rate`` as
the value held for the *next* interval.  The rate handed to a call therefore
takes effect after that call, exactly like a command latched at the start of a
control period.  Consequences:

* the first ``advance`` at the reset timestamp is legal and integrates nothing;
* the first ``advance`` at a later timestamp integrates the post-reset held rate
  of zero over that gap;
* replaying the identical ``(rate, timestamp)`` pair is idempotent (a repeated
  frame must not integrate twice), while a *different* rate at an already
  published timestamp is rejected as an inconsistent republication;
* backward simulation time is rejected -- use ``reset`` to start a new episode.

GUI use versus yaw-rate use
---------------------------
A GUI that owns an absolute heading knob calls ``heading_feedback`` directly
with its own target.  A GUI that owns a yaw-rate stick calls ``advance`` once
per simulation step with the stick value and the current simulation time, then
feeds ``state.heading_rad`` as ``reference_yaw_rad`` into ``heading_feedback``
together with the same stick value as ``user_yaw_rate_rps`` (feed-forward).
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

HEADING_REFERENCE_SCHEMA = "d1-simulation-time-heading-reference-v1"

__all__ = [
    "HEADING_REFERENCE_SCHEMA",
    "HeadingFeedback",
    "HeadingReference",
    "HeadingReferenceState",
    "heading_feedback",
    "wrap_angle",
]

_MAX_USER_YAW_RATE_RPS = 1.0


def _finite(name: str, value: object) -> float:
    """Validate ``value`` as a finite real scalar and return it as ``float``."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar, not bool")
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real scalar, got {type(value).__name__}")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:  # e.g. unbounded integers
        raise ValueError(f"{name} is not representable as a finite float") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _finite_result(name: str, value: float) -> float:
    """Reject an arithmetic result that overflowed to NaN/infinity."""
    if not math.isfinite(value):
        raise ValueError(f"{name} overflowed to a non-finite value")
    return value


def wrap_angle(value: float) -> float:
    """Wrap ``value`` (radians) into ``[-pi, pi]`` via ``atan2(sin, cos)``.

    Rejects bools, non-real values, NaN and infinities.
    """
    angle = _finite("angle", value)
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class HeadingFeedback:
    """Result of one heading feedback evaluation (no state, no side effects)."""

    heading_error_rad: float
    servo_yaw_rate_rps: float
    unclipped_yaw_rate_rps: float
    saturated: bool


def heading_feedback(
    reference_yaw_rad: float,
    measured_yaw_rad: float,
    measured_yaw_rate_rps: float,
    user_yaw_rate_rps: float = 0.0,
    *,
    kp: float = 2.0,
    kd: float = 0.4,
    limit_rps: float = 1.0,
) -> HeadingFeedback:
    """Yaw-rate command that drives ``measured_yaw_rad`` onto ``reference_yaw_rad``.

    ``error = wrap(reference - measured)`` and
    ``raw = user + kp * error - kd * (measured_rate - user)``, clipped
    symmetrically to ``limit_rps``.  The derivative term is taken against the
    operator feed-forward rate so that a commanded turn is not damped away.

    Gains must be finite and non-negative, ``limit_rps`` finite and positive.
    This function changes no target and touches no plant state.
    """
    reference = _finite("reference_yaw_rad", reference_yaw_rad)
    measured = _finite("measured_yaw_rad", measured_yaw_rad)
    measured_rate = _finite("measured_yaw_rate_rps", measured_yaw_rate_rps)
    user_rate = _finite("user_yaw_rate_rps", user_yaw_rate_rps)
    gain_p = _finite("kp", kp)
    gain_d = _finite("kd", kd)
    limit = _finite("limit_rps", limit_rps)
    if gain_p < 0.0:
        raise ValueError(f"kp must be non-negative, got {gain_p!r}")
    if gain_d < 0.0:
        raise ValueError(f"kd must be non-negative, got {gain_d!r}")
    if not limit > 0.0:
        raise ValueError(f"limit_rps must be positive, got {limit!r}")

    error = _finite_result("heading error", math.atan2(
        math.sin(reference - measured), math.cos(reference - measured)
    ))
    proportional = _finite_result("proportional term", gain_p * error)
    damping = _finite_result("damping term", gain_d * (measured_rate - user_rate))
    raw = _finite_result("unclipped yaw rate", user_rate + proportional - damping)
    servo = max(-limit, min(limit, raw))
    return HeadingFeedback(
        heading_error_rad=error,
        servo_yaw_rate_rps=servo,
        unclipped_yaw_rate_rps=raw,
        saturated=bool(servo != raw),
    )


@dataclass(frozen=True)
class HeadingReferenceState:
    """Immutable snapshot of the heading reference after a committed update.

    ``user_yaw_rate_rps`` is the rate now *held* for the next interval, not the
    rate that produced ``heading_rad``.
    """

    simulation_time_s: float
    heading_rad: float
    user_yaw_rate_rps: float


class HeadingReference:
    """Integrates an operator yaw-rate command into a heading target.

    The instance starts uninitialised; ``reset`` must be called before
    ``advance`` or ``state``.  No wall clock is read: every timestamp comes from
    the caller's simulation clock.
    """

    def __init__(self) -> None:
        self._state: HeadingReferenceState | None = None
        self._held_rate_rps: float = 0.0
        self._published_rate_rps: float | None = None

    @property
    def state(self) -> HeadingReferenceState:
        """Last committed state; raises ``RuntimeError`` before ``reset``."""
        if self._state is None:
            raise RuntimeError("HeadingReference.reset() must be called first")
        return self._state

    def reset(self, yaw_rad: float, simulation_time_s: float = 0.0) -> HeadingReferenceState:
        """Start a new episode at ``yaw_rad`` and ``simulation_time_s``.

        Validates before mutating, wraps the heading, clears the stored command
        and publication history, and sets the held rate to zero.
        """
        heading = wrap_angle(yaw_rad)
        time_s = _finite("simulation_time_s", simulation_time_s)
        if time_s < 0.0:
            raise ValueError(f"simulation_time_s must be non-negative, got {time_s!r}")
        self._held_rate_rps = 0.0
        self._published_rate_rps = None
        self._state = HeadingReferenceState(
            simulation_time_s=time_s, heading_rad=heading, user_yaw_rate_rps=0.0
        )
        return self._state

    def advance(self, user_yaw_rate_rps: float, simulation_time_s: float) -> HeadingReferenceState:
        """Integrate the held rate up to ``simulation_time_s``, then hold ``user_yaw_rate_rps``.

        The rate must be finite within ``[-1, 1]`` rad/s and the time finite,
        non-negative and not earlier than the last published timestamp.  A
        repeated ``(rate, timestamp)`` pair returns the existing state without
        integrating again; a different rate at a published timestamp is an
        error.  Nothing is mutated unless every check and the arithmetic pass.
        """
        previous = self.state
        rate = _finite("user_yaw_rate_rps", user_yaw_rate_rps)
        if not -_MAX_USER_YAW_RATE_RPS <= rate <= _MAX_USER_YAW_RATE_RPS:
            raise ValueError(
                f"user_yaw_rate_rps must lie within "
                f"[-{_MAX_USER_YAW_RATE_RPS}, {_MAX_USER_YAW_RATE_RPS}], got {rate!r}"
            )
        time_s = _finite("simulation_time_s", simulation_time_s)
        if time_s < 0.0:
            raise ValueError(f"simulation_time_s must be non-negative, got {time_s!r}")
        if time_s < previous.simulation_time_s:
            raise ValueError(
                f"simulation time must not move backwards: {time_s!r} < "
                f"{previous.simulation_time_s!r}; call reset() for a new episode"
            )
        if time_s == previous.simulation_time_s and self._published_rate_rps is not None:
            if rate == self._published_rate_rps:
                return previous
            raise ValueError(
                f"simulation time {time_s!r} was already published with yaw rate "
                f"{self._published_rate_rps!r}; cannot republish with {rate!r}"
            )

        elapsed = _finite_result("elapsed simulation time", time_s - previous.simulation_time_s)
        delta = _finite_result("heading increment", self._held_rate_rps * elapsed)
        heading = _finite_result("heading", previous.heading_rad + delta)
        heading = math.atan2(math.sin(heading), math.cos(heading))

        self._held_rate_rps = rate
        self._published_rate_rps = rate
        self._state = HeadingReferenceState(
            simulation_time_s=time_s, heading_rad=heading, user_yaw_rate_rps=rate
        )
        return self._state
