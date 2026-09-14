"""Experimental course brake reference re-anchoring; legacy remains the default.

Context
-------
The default GUI drives the course controller (LQR + VMC); there is no PPO/RL path
here.  That controller integrates travelled ``distance_m`` and a commanded
``distance_reference_m``.  When the operator releases W/S (or presses X, or the
window loses focus) the requested forward velocity becomes exactly zero, but the
already-accumulated reference stays ahead of (or behind) the measured distance.
The position term of the legacy controller then keeps pushing until that retained
error is consumed, which shows up as post-release creep.

What this helper does
---------------------
On the falling edge "previous *actual* eligible legacy command was nonzero -> the
current eligible request is exactly zero" it performs a single, one-time
assignment ``controller._distance_reference_m = control_memory.distance_m``,
clearing the *retained reference error* only.  Measured locally with unchanged
physics and unchanged gains:

* flat, 0.5 m/s: reference-distance = 0.226327 m; 4 s forward drift 0.274233 m ->
  0.148027 m with re-anchoring;
* ramp, 0.4: reference-distance = -0.151833 m; reverse drift -0.140819 m ->
  -0.056776 m, last-1 s |vx| 0.058507 -> 0.034806 m/s.

Paired pre-release states and torque were bitwise identical.

Scope and honest limits
-----------------------
This is a retained-reference *correction*, not a complete stopping guarantee.
The GUI requires ``--brake-reference-mode release_reanchor_experimental`` to use
this helper. One uphill release increased retreat from its peak from 0.009 m to
0.200 m; neither full nor half correction was selected as the default.
Residual drift from body momentum, contact compliance and the velocity/attitude
terms of the legacy controller remains; the candidate reduces it but does not
eliminate it.  No hardware validation and no reinforcement-learning improvement
is claimed.  The helper never reads or writes physical state, never integrates,
never steps the model, never teleports, never applies body forces, and never
calls ``controller.reset()``.  The only controller member it may ever assign is
``controller._distance_reference_m``.

Primary implementation: Claude Opus; integrated and validated locally.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

#: Exact-zero tolerance for velocity commands (m/s).
ZERO_COMMAND_EPS = 1.0e-12


def _finite_float(value: Any, name: str) -> float:
    """Return ``value`` as float, rejecting bool, non-real and non-finite input."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


class CourseBrakeReference:
    """Two-phase edge detector that re-anchors the retained distance reference."""

    __slots__ = ("_event_count", "_previous_command_nonzero")

    def __init__(self) -> None:
        self._previous_command_nonzero = False
        self._event_count = 0

    def reset(self) -> None:
        """Clear the edge context and event counter after an external zone reset."""
        self._previous_command_nonzero = False
        self._event_count = 0

    @property
    def event_count(self) -> int:
        return self._event_count

    def prepare(
        self,
        controller: Any,
        requested_forward_mps: Any,
        *,
        eligible: Any,
    ) -> dict[str, Any]:
        """Called before the legacy ``teleop.compute`` of the current tick.

        Does *not* record the current request as "previous command": guards may
        still alter it, so :meth:`observe_applied` supplies the actual value after
        the real computation.  The stored previous context is consumed here, so
        two ``prepare`` calls before an ``observe_applied`` cannot trigger twice.
        """
        is_eligible = _require_bool(eligible, "eligible")
        requested = _finite_float(requested_forward_mps, "requested_forward_mps")

        previous_nonzero = self._previous_command_nonzero
        if not is_eligible or not previous_nonzero or abs(requested) > ZERO_COMMAND_EPS:
            # Side-owned / jump / recovery ticks leave no latent edge.
            self._previous_command_nonzero = False
            return self._diagnostics(False, None, None)

        memory = controller.control_memory
        distance_m = _finite_float(memory.distance_m, "control_memory.distance_m")
        reference_m = _finite_float(
            memory.distance_reference_m, "control_memory.distance_reference_m"
        )
        if getattr(controller, "_prepared_control", None) is not None:
            raise RuntimeError(
                "CourseBrakeReference.prepare: controller._prepared_control is not "
                "None; that proposal predates the brake edge and would be stale. "
                "Resolve or clear it upstream before re-anchoring."
            )

        previous_error_m = reference_m - distance_m
        if not math.isfinite(previous_error_m):
            raise ValueError("distance reference error must be finite")
        controller._distance_reference_m = distance_m  # sole permitted mutation
        # Consume only after validation and the assignment have succeeded.
        self._previous_command_nonzero = False
        self._event_count += 1
        return self._diagnostics(True, previous_error_m, 0.0)

    def observe_applied(
        self,
        forward_velocity_mps: Any,
        *,
        eligible: Any,
    ) -> None:
        """Record the *actual* legacy command of this tick, after its computation."""
        is_eligible = _require_bool(eligible, "eligible")
        applied = _finite_float(forward_velocity_mps, "forward_velocity_mps")
        if not is_eligible:
            self._previous_command_nonzero = False
            return
        self._previous_command_nonzero = abs(applied) > ZERO_COMMAND_EPS

    def _diagnostics(
        self,
        reanchored: bool,
        previous_error_m: float | None,
        new_error_m: float | None,
    ) -> dict[str, Any]:
        return {
            "brake_reference_reanchored": reanchored,
            "brake_reference_previous_error_m": previous_error_m,
            "brake_reference_new_error_m": new_error_m,
            "brake_reference_event_count": self._event_count,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"{type(self).__name__}(previous_command_nonzero="
            f"{self._previous_command_nonzero!r}, event_count={self._event_count})"
        )
