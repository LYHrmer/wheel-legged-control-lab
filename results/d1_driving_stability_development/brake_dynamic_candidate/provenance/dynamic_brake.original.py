"""Work-only helper: dynamic-brake distance-reference hold for a robot controller.

EXPERIMENTAL, REFERENCE-ONLY HYPOTHESIS
---------------------------------------
This module implements a *hypothesis* about how a distance-tracking controller's
internal distance *reference* could be nudged while a command release (a
transition from an accepted nonzero forward command to a requested zero forward
command) is being coasted out.  It only ever proposes/writes a new value for
``controller._distance_reference_m``.  It does **not** and **cannot** claim to
produce any real braking force, any real motor torque, or any guaranteed
deceleration of the physical machine.  ``DECELERATION_MPS2`` is nothing but a
bookkeeping constant used to bound how far the reference is allowed to sit ahead
of (or behind) the measured travelled distance; it is not a promise about plant
dynamics.  Nothing here reads or writes plant, torque, or controller state, and
``controller.reset`` is never called.

CONVENTIONS
-----------
* ``requested_forward_mps`` / ``actual_forward_mps`` are commanded forward
  velocities in the *body-forward* sign convention.
* ``measured_forward_mps`` is the measured body-forward speed expressed in the
  same convention and the same axis that the controller integrates into
  ``control_memory.distance_m``, so ``measured_v**2 / (2 * a)`` is directly
  comparable with a distance error in metres.
* The control period is the fixed ``CONTROL_DT`` (seconds); low-speed dwell is
  counted in whole control ticks, never in wall-clock time.

USAGE ORDER (per control tick, decided entirely by the caller)
--------------------------------------------------------------
1. ``prepare(controller, requested_v, measured_v, eligible=...)`` *before* the
   real controller compute; it may write the distance reference.
2. ``observe_applied(actual_v, eligible=...)`` *after* the compute, with the
   ACTUAL accepted command.

The caller is responsible for passing ``eligible=False`` for side/jump/recovery
gaps and any other discontinuity, which is what prevents a false "release" from
being synthesised across such a gap.
"""

from __future__ import annotations

import math
import numbers

__all__ = ["DynamicBrakeHold"]


class DynamicBrakeHold:
    """Reference-only dynamic-brake / hold bookkeeping (see module docstring)."""

    DECELERATION_MPS2 = 0.5
    LOW_SPEED_MPS = 0.03
    LOW_SPEED_TICKS = 30
    CONTROL_DT = 0.01
    ZERO_COMMAND_EPS = 1e-12

    def __init__(self) -> None:
        self.release_count = 0
        self.hold_count = 0
        self._phase = "idle"
        self._prev_nonzero = False
        self._low_ticks = 0
        self._hold_reference_m = None

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Clear everything, including the lifetime event counters."""
        self.release_count = 0
        self.hold_count = 0
        self._clear_transient()

    def _clear_transient(self) -> None:
        """Clear phase, command edge, dwell and hold state (not counters)."""
        self._phase = "idle"
        self._prev_nonzero = False
        self._low_ticks = 0
        self._hold_reference_m = None

    @property
    def phase(self) -> str:
        return self._phase

    # ------------------------------------------------------------ validation

    @staticmethod
    def _check_eligible(value) -> bool:
        if value is True or value is False:
            return value
        raise TypeError("eligible must be exactly True or False")

    @staticmethod
    def _check_scalar(value, name: str) -> float:
        if isinstance(value, bool):
            raise TypeError("%s must be a real scalar, not bool" % name)
        if not isinstance(value, numbers.Real):
            raise TypeError("%s must be a real scalar, got %r" % (name, type(value).__name__))
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("%s must be finite, got %r" % (name, value))
        return result

    @staticmethod
    def _check_result(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("%s is not finite: %r" % (name, value))
        return result

    def _read_memory(self, controller):
        memory = controller.control_memory
        distance = self._check_scalar(getattr(memory, "distance_m"), "control_memory.distance_m")
        reference = self._check_scalar(
            getattr(memory, "distance_reference_m"), "control_memory.distance_reference_m"
        )
        return distance, reference

    @staticmethod
    def _check_no_pending(controller) -> None:
        if controller._prepared_control is not None:
            raise RuntimeError(
                "controller._prepared_control must be None before the distance "
                "reference is changed"
            )

    # ---------------------------------------------------------- diagnostics

    def _diagnostics(
        self,
        *,
        measured_forward_mps: float,
        release_started: bool = False,
        hold_entered: bool = False,
        reference_written: bool = False,
        old_error_m=None,
        new_error_m=None,
        bound_m=None,
    ) -> dict:
        return {
            "phase": self._phase,
            "release_started": release_started,
            "release_count": self.release_count,
            "hold_entered": hold_entered,
            "hold_count": self.hold_count,
            "reference_written": reference_written,
            "old_error_m": old_error_m,
            "new_error_m": new_error_m,
            "bound_m": bound_m,
            "measured_forward_mps": measured_forward_mps,
            "low_speed_ticks": self._low_ticks,
            "hold_reference_m": self._hold_reference_m,
        }

    # -------------------------------------------------------------- prepare

    def prepare(self, controller, requested_forward_mps, measured_forward_mps, *, eligible) -> dict:
        """Maybe nudge ``controller._distance_reference_m``; return diagnostics."""
        eligible = self._check_eligible(eligible)
        requested = self._check_scalar(requested_forward_mps, "requested_forward_mps")
        measured = self._check_scalar(measured_forward_mps, "measured_forward_mps")

        zero_request = abs(requested) <= self.ZERO_COMMAND_EPS

        if not eligible or not zero_request:
            # Ineligible or a fresh nonzero request: drop all transient state so a
            # later genuine release can start a new cycle.  No memory is read and
            # no reference is written.
            self._clear_transient()
            return self._diagnostics(measured_forward_mps=measured)

        # Active tick: validate the controller memory and any pending proposal
        # *before* consuming the command edge, touching counters, or assigning.
        distance, reference = self._read_memory(controller)
        self._check_no_pending(controller)

        release_started = False
        if self._phase == "idle":
            if not self._prev_nonzero:
                # Initial standing zero (or already-cancelled cycle): do nothing.
                return self._diagnostics(measured_forward_mps=measured)
            self._prev_nonzero = False  # edge consumed
            self._phase = "brake"
            self.release_count += 1
            release_started = True

        # Dwell bookkeeping (consecutive low-speed prepare ticks).
        if abs(measured) <= self.LOW_SPEED_MPS:
            self._low_ticks += 1
        else:
            self._low_ticks = 0

        bound = self._check_result(
            (measured * measured) / (2.0 * self.DECELERATION_MPS2), "bound_m"
        )
        old_error = self._check_result(reference - distance, "old_error_m")
        new_error = self._check_result(
            self._clamp(old_error, -bound, bound), "new_error_m"
        )

        if self._phase == "hold":
            # Diagnostics only; the reference stays frozen until a new cycle.
            return self._diagnostics(
                measured_forward_mps=measured,
                release_started=release_started,
                old_error_m=old_error,
                new_error_m=new_error,
                bound_m=bound,
            )

        hold_entered = False
        if self._low_ticks >= self.LOW_SPEED_TICKS:
            hold_entered = True
            new_error = 0.0
            new_reference = self._check_result(distance, "new_reference_m")
        else:
            new_reference = self._check_result(distance + new_error, "new_reference_m")

        controller._distance_reference_m = new_reference

        if hold_entered:
            self._phase = "hold"
            self.hold_count += 1
            self._hold_reference_m = new_reference

        return self._diagnostics(
            measured_forward_mps=measured,
            release_started=release_started,
            hold_entered=hold_entered,
            reference_written=True,
            old_error_m=old_error,
            new_error_m=new_error,
            bound_m=bound,
        )

    # ------------------------------------------------------- observe_applied

    def observe_applied(self, actual_forward_mps, *, eligible) -> None:
        """Record the ACTUAL accepted forward command for this tick."""
        eligible = self._check_eligible(eligible)
        actual = self._check_scalar(actual_forward_mps, "actual_forward_mps")

        if not eligible:
            self._clear_transient()
            return

        if abs(actual) > self.ZERO_COMMAND_EPS:
            # A real command cancels any brake/hold phase and arms the edge.
            self._phase = "idle"
            self._low_ticks = 0
            self._hold_reference_m = None
            self._prev_nonzero = True
        else:
            # Zero command: no edge next tick, but an active cycle survives.
            self._prev_nonzero = False

    # ---------------------------------------------------------------- helper

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        if value < low:
            return low
        if value > high:
            return high
        return value
