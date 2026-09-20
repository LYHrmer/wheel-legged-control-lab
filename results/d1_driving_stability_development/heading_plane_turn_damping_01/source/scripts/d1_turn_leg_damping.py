"""Raw-turn-gated longitudinal leg damping on top of the frozen D1 controller.

This module is a *diagnostic candidate only*.  It adds one bounded joint-space
damping term to the frozen wheel/leg torque law on exactly those execution ticks
whose **raw** command is a pure turn (forward exactly zero, yaw nonzero), so that
a paired physical comparison can falsify it.  Nothing here is an accepted fix for
the turn/heading shortfall, no reliable turn is claimed, and no driving goal is
claimed to be completed.

Boundaries kept deliberately tight:

* the frozen controller/model/state-estimation sources are imported, never edited;
  neither candidate controller is inherited -- only ``D1WheelLegController`` is;
* the fixed coefficient is imported from ``scripts.d1_stop_leg_damping`` purely as
  a number; there is no selectable gain, no yaw cap, no tuning branch;
* ``super().compute`` is called exactly once per call, leaving the original
  targets, the wheel PI update with its antiwindup, the support/attitude terms
  and the yaw servo untouched;
* the damping torque is added to ``requested_torque_nm`` (the *unclipped* request)
  and the original safety sequence (rated-torque clip, then outward position and
  outward speed suppression) is repeated in the original order;
* the four wheel channels receive exactly positive zero increment, and no wheel
  request or wheel PI value is rewritten;
* there is **no** center wheel-speed compensation, **no** stop latch, no tick or
  case trigger, no threshold, no filter, no history, no latch of any kind.

``bind_raw_command`` records the raw command actually issued for the upcoming
execution tick.  It is *not* a mode latch: the gate is recomputed from the bound
raw scalars on every call, and binding a new command after a ``compute`` never
changes the stored ``last_damping`` record.

``joint_power_w`` is the raw sampled-instant algebra ``delta @ qdot`` (analytically
``-b * sum(u_i**2)`` before safety clipping).  It is reported unclamped and is
**not** integrated work, **not** a discrete-time passivity certificate and **not**
a global stability statement.

Failure semantics: every validation that can fail *before* ``super().compute``
runs leaves controller memory and the diagnostic record unchanged.  A diagnostic
error raised *after* the parent law has run may leave the wheel PI integral
already advanced; there is no rollback and no second parent call, so the same
physical state must not be retried -- reset instead.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from numbers import Real

import numpy as np

from scripts.d1_stop_leg_damping import STOP_LEG_DAMPING_NSPM
from wheel_legged_control.d1.model import (
    JOINT_POSITION_HIGH,
    JOINT_POSITION_LOW,
    JOINT_TORQUE_LIMIT,
    JOINT_VELOCITY_LIMIT,
)
from wheel_legged_control.d1.wheel_leg_controller import WHEEL_INDICES, D1WheelLegController

# --- fixed public contract ------------------------------------------------
#: Unique control schema so no checkpoint recorded against another law loads here.
TURN_LEG_DAMPING_SCHEMA = "d1-raw-turn-leg-longitudinal-damping-v1"
#: The one fixed longitudinal coefficient, N*s/m, reused verbatim (126.4374005337902).
#: It is the nominal longitudinal modal value; it is not a yaw critical damping.
TURN_LEG_DAMPING_NSPM = float(STOP_LEG_DAMPING_NSPM)
#: Binding/state time agreement tolerance, seconds.
TIME_MATCH_TOLERANCE_S = 1e-10


def _readonly(values, shape: tuple[int, ...], label: str) -> np.ndarray:
    """Return a finite, contiguous, byte-backed read-only copy of ``values``."""
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(shape)


_ZERO4 = _readonly(np.zeros(4), (4,), "zero4")
_ZERO16 = _readonly(np.zeros(16), (16,), "zero16")


def _finite_scalar(value, label: str) -> float:
    """Reject bool/non-real/nonfinite values and return a plain float."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class TurnRawCommandBinding:
    """Immutable raw command bound to one upcoming execution tick.

    Field names match the raw command exactly. This object carries no mode and no
    memory: it only states which raw scalars the next ``compute`` will execute.
    """

    forward_velocity_mps: float
    yaw_rate_rps: float
    control_time_s: float

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(
                self, field.name, _finite_scalar(getattr(self, field.name), field.name)
            )


@dataclass(frozen=True)
class TurnLegDampingRecord:
    """Immutable per-call diagnostic record; arrays are read-only byte-backed copies.

    Recorded for every ``compute`` call, including disabled and inactive ones,
    where the damping quantities are exactly zero and base/total requests match.
    ``joint_power_w`` is the unclamped sampled-instant algebra ``delta @ qdot``;
    it is not integrated work and carries no passivity or stability claim.
    The raw fields and ``control_time_s`` always describe the binding executed by
    this ``compute``, never a later preview or a later binding.
    """

    schema: str
    enabled: bool
    active: bool
    control_time_s: float
    raw_forward_mps: float
    raw_yaw_rate_rps: float
    servo_forward_mps: float
    servo_yaw_rate_rps: float
    damping_nspm: float
    leg_relative_forward_mps: np.ndarray  # u_i, shape (4,)
    delta_torque_nm: np.ndarray  # shape (16,), wheels exactly +0.0
    base_leg_pd_nm: np.ndarray  # shape (16,)
    base_requested_torque_nm: np.ndarray  # shape (16,)
    total_requested_torque_nm: np.ndarray  # shape (16,)
    safe_torque_nm: np.ndarray  # shape (16,)
    joint_power_w: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", str(self.schema))
        for name in ("enabled", "active"):
            object.__setattr__(self, name, bool(getattr(self, name)))
        for name in (
            "control_time_s",
            "raw_forward_mps",
            "raw_yaw_rate_rps",
            "servo_forward_mps",
            "servo_yaw_rate_rps",
            "damping_nspm",
            "joint_power_w",
        ):
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))
        object.__setattr__(
            self,
            "leg_relative_forward_mps",
            _readonly(self.leg_relative_forward_mps, (4,), "leg_relative_forward_mps"),
        )
        for name in (
            "delta_torque_nm",
            "base_leg_pd_nm",
            "base_requested_torque_nm",
            "total_requested_torque_nm",
            "safe_torque_nm",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), (16,), name))
        wheels = self.delta_torque_nm[WHEEL_INDICES]
        if np.any(wheels != 0.0) or np.any(np.signbit(wheels)):
            raise ValueError("wheel damping increments must be exactly positive zero")


class TurnLegDampingController(D1WheelLegController):
    """Frozen wheel/leg law plus one fixed raw-turn longitudinal leg damping term.

    The gate is exactly ``enabled and raw.forward_velocity_mps == 0.0 and
    raw.yaw_rate_rps != 0.0``, evaluated on the bound raw command of the current
    execution tick. Signed zeros are treated by ordinary float comparison, so a
    raw yaw of ``-0.0`` stays inactive. A nonzero heading-servo yaw never opens
    the gate when the raw yaw is zero, and there is no latch, threshold, filter or
    history: every tick is judged independently.

    This is a zero-residual diagnostic: any nonzero, non-real or wrongly shaped
    action is rejected before the parent law runs and before any PI update, in
    both enabled and disabled modes, so a policy checkpoint cannot be smuggled in.
    """

    def __init__(self, *, enabled: bool = True, **kwargs) -> None:
        if not isinstance(enabled, (bool, np.bool_)):
            # Numeric 0/1 is rejected: the switch must be an explicit boolean.
            raise TypeError("enabled must be a bool")
        super().__init__(**kwargs)
        self._enabled = bool(enabled)
        if self._enabled:
            # Distinct schema only when the added law is live; the disabled branch
            # keeps the parent's control_schema bit-for-bit, and action_schema is
            # never touched in either branch.
            self.control_schema = TURN_LEG_DAMPING_SCHEMA
        self._raw_binding: TurnRawCommandBinding | None = None
        self.last_damping: TurnLegDampingRecord | None = None

    # --- read-only configuration -----------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def damping_nspm(self) -> float:
        return TURN_LEG_DAMPING_NSPM

    @property
    def raw_binding(self) -> TurnRawCommandBinding | None:
        return self._raw_binding

    def bind_raw_command(
        self, *, forward_velocity_mps, yaw_rate_rps, control_time_s
    ) -> TurnRawCommandBinding:
        """Bind the raw command of the next execution tick (not a mode latch)."""
        binding = TurnRawCommandBinding(forward_velocity_mps, yaw_rate_rps, control_time_s)
        self._raw_binding = binding
        return binding

    def reset(self) -> None:
        """Clear the binding and the record, then the parent memory/diagnostics."""
        self._raw_binding = None
        self.last_damping = None
        super().reset()

    # --- validation -------------------------------------------------------
    @staticmethod
    def _checked_zero_action(action) -> np.ndarray:
        array = np.asarray(action)
        if isinstance(action, (bool, np.bool_)) or array.dtype.kind not in ("f", "i", "u"):
            raise TypeError("action must be a real numeric array of eight zeros")
        values = np.ascontiguousarray(array, dtype=np.float64)
        if values.shape != (8,):
            raise ValueError("action must have shape (8,)")
        if not np.isfinite(values).all():
            raise ValueError("action must contain eight finite values")
        if np.any(values != 0.0):
            raise ValueError("turn leg damping diagnostic accepts only exactly zero actions")
        return values

    def _require_binding(self, command, state) -> TurnRawCommandBinding:
        binding = self._raw_binding
        if binding is None:
            raise RuntimeError(
                "enabled turn leg damping requires bind_raw_command for the current tick"
            )
        time = _finite_scalar(state.control_time_s, "state.control_time_s")
        if abs(binding.control_time_s - time) > TIME_MATCH_TOLERANCE_S:
            raise ValueError("bound raw control_time_s does not match the state control time")
        forward = _finite_scalar(command.forward_velocity_mps, "command.forward_velocity_mps")
        if forward != binding.forward_velocity_mps:
            raise ValueError("servo forward velocity must equal the bound raw forward velocity")
        return binding

    def _gate_active(self) -> bool:
        binding = self._raw_binding
        return bool(
            self._enabled
            and binding is not None
            and binding.forward_velocity_mps == 0.0
            and binding.yaw_rate_rps != 0.0
        )

    # --- preview ----------------------------------------------------------
    def nominal_targets(self, command, state, *, ground_height_m: float = 0.0):
        """Pure preview: the parent targets object, unchanged, in every mode.

        No compensation is computed and neither the record nor the wheel PI is
        touched here; only the binding agreement is checked when enabled.
        """
        if self._enabled:
            self._require_binding(command, state)
        return super().nominal_targets(command, state, ground_height_m=ground_height_m)

    # --- control ----------------------------------------------------------
    def compute(self, command, state, action, *, ground_height_m: float = 0.0) -> np.ndarray:
        # 1. All pre-parent validation: nothing below mutates PI or the record.
        checked = self._checked_zero_action(action)
        binding = self._raw_binding
        if self._enabled:
            binding = self._require_binding(command, state)
        active = self._gate_active()
        servo_forward = _finite_scalar(command.forward_velocity_mps, "forward_velocity_mps")
        servo_yaw = _finite_scalar(command.yaw_rate_rps, "yaw_rate_rps")
        if binding is None:
            record_time = _finite_scalar(state.control_time_s, "state.control_time_s")
            raw_forward, raw_yaw = servo_forward, servo_yaw
        else:
            record_time = binding.control_time_s
            raw_forward, raw_yaw = binding.forward_velocity_mps, binding.yaw_rate_rps

        # 2. The frozen law runs exactly once, keeping its wheel PI update and
        #    antiwindup, its targets, its support and attitude terms untouched.
        #    Any error raised from here on may leave the PI advanced: do not
        #    retry the same physical state, reset instead.
        base_torque = super().compute(command, state, checked, ground_height_m=ground_height_m)
        base = self.last_result

        if not active:
            # Disabled or non-turn raw tick: the parent result object and the
            # returned torque object are passed through untouched, so logs stay
            # bit-for-bit identical to bypass.
            self.last_damping = TurnLegDampingRecord(
                schema=TURN_LEG_DAMPING_SCHEMA,
                enabled=self._enabled,
                active=False,
                control_time_s=record_time,
                raw_forward_mps=raw_forward,
                raw_yaw_rate_rps=raw_yaw,
                servo_forward_mps=servo_forward,
                servo_yaw_rate_rps=servo_yaw,
                damping_nspm=TURN_LEG_DAMPING_NSPM,
                leg_relative_forward_mps=_ZERO4,
                delta_torque_nm=_ZERO16,
                base_leg_pd_nm=base.leg_pd_nm,
                base_requested_torque_nm=base.requested_torque_nm,
                total_requested_torque_nm=base.requested_torque_nm,
                safe_torque_nm=base.torque_nm,
                joint_power_w=0.0,
            )
            return base_torque

        # 3. Longitudinal joint-relative damping on the three leg joints per leg.
        #    body_x_world is the body x axis including tilt, not a horizontal
        #    heading projection.
        body_x_world = np.asarray(state.base_rotation, dtype=np.float64)[:, 0]
        jacobian = np.asarray(state.foot_jacobian, dtype=np.float64)
        velocity = np.asarray(state.joint_velocity, dtype=np.float64)
        position = np.asarray(state.joint_position, dtype=np.float64)
        u = np.zeros(4)
        delta = np.zeros(16)
        for leg in range(4):
            jx = body_x_world @ jacobian[leg, :, :3]
            u[leg] = float(jx @ velocity[4 * leg : 4 * leg + 3])
            delta[4 * leg : 4 * leg + 3] = -TURN_LEG_DAMPING_NSPM * jx * u[leg]
        delta[WHEEL_INDICES] = 0.0  # wheel channels strictly untouched
        if not np.isfinite(delta).all() or not np.isfinite(u).all():
            raise ValueError("turn leg damping produced nonfinite values")

        # 4. Add to the unclipped request, then repeat the original safety order.
        combined = base.requested_torque_nm + delta
        safe = np.clip(combined, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
        upper_outward = (position >= JOINT_POSITION_HIGH) & (safe > 0)
        lower_outward = (position <= JOINT_POSITION_LOW) & (safe < 0)
        speed_outward = (np.abs(velocity) >= JOINT_VELOCITY_LIMIT) & (safe * velocity > 0)
        safe[upper_outward | lower_outward | speed_outward] = 0.0
        if not np.isfinite(combined).all() or not np.isfinite(safe).all():
            raise ValueError("turn leg damping produced nonfinite torques")

        # 5. Keep the requested = leg_pd + support + wheel identity by folding
        #    delta into leg_pd_nm; the base terms stay in the record.
        result = dataclasses.replace(
            base,
            leg_pd_nm=base.leg_pd_nm + delta,
            requested_torque_nm=combined,
            torque_nm=safe,
            torque_limited=safe != combined,
        )
        record = TurnLegDampingRecord(
            schema=TURN_LEG_DAMPING_SCHEMA,
            enabled=True,
            active=True,
            control_time_s=record_time,
            raw_forward_mps=raw_forward,
            raw_yaw_rate_rps=raw_yaw,
            servo_forward_mps=servo_forward,
            servo_yaw_rate_rps=servo_yaw,
            damping_nspm=TURN_LEG_DAMPING_NSPM,
            leg_relative_forward_mps=u,
            delta_torque_nm=delta,
            base_leg_pd_nm=base.leg_pd_nm,
            base_requested_torque_nm=base.requested_torque_nm,
            total_requested_torque_nm=combined,
            safe_torque_nm=safe,
            # Sampled-instant algebra of the added term only, -b * sum(u_i**2)
            # before safety clipping; reported unclamped, no passivity claim.
            joint_power_w=float(delta @ velocity),
        )

        # 6. Commit only after the full computation and the record succeeded.
        self.last_result = result
        self.last_damping = record
        return safe.copy()


__all__ = [
    "TIME_MATCH_TOLERANCE_S",
    "TURN_LEG_DAMPING_NSPM",
    "TURN_LEG_DAMPING_SCHEMA",
    "TurnLegDampingController",
    "TurnLegDampingRecord",
    "TurnRawCommandBinding",
]
