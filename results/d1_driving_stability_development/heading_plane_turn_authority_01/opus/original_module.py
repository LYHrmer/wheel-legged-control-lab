"""Raw-turn restoration of the ORIGINAL unclipped body-yaw feedback authority.

Diagnostic candidate only.  On exactly those execution ticks whose **raw** command
is a pure stationary turn (``forward == 0`` and ``yaw != 0``) this controller
replaces the frozen inner ``+/-0.6`` yaw clip by the original unclipped formula

    r_eff = servo_yaw + yaw_feedback_gain * (servo_yaw - base_angular_velocity_body[2])

and then keeps the ORIGINAL geometric wheel target ``(servo_forward - r_eff*y_i)/0.087``
with the ORIGINAL final ``+/-30 rad/s`` wheel clip.  Nothing else changes: gain 4,
wheel PI 2.2/3 with the +/-4 integral antiwindup, the 12 N*m wheel / 80 N*m leg rated
clips, the outward position/speed suppression, the leg PD/support terms and the outer
``+/-1`` action scaling are the parent's and are called exactly once per tick.

Deliberate non-scope: no other cap value, no headroom limiter, no gain/dt/duration
choice, no PI variant, no turn/stop damping law, no wheel-center ``+u/R``
compensation, no latch, no threshold, no history, no tick or case trigger.
``TurnRawCommandBinding`` is imported from ``scripts.d1_turn_leg_damping`` solely as
the already validated immutable raw-input dataclass; its damping law is not used and
neither candidate controller is inherited.

Honest expectations:

* ``r_eff`` magnitudes well above 0.6 rad/s are reported as such; the first ticks are
  expected to SATURATE the wheel targets at +/-30 rad/s and the unprotected wheel
  request is expected to exceed the 12 N*m rated clip.  Both are recorded explicitly
  (``wheel_target_clipped``, ``wheel_request_nm`` vs ``wheel_torque_nm``) and are not
  hidden or pre-limited.
* the reported ``effective_yaw_request_rps`` is a *request*, not a measured body yaw
  rate, and it does not weaken the final wheel-speed/torque protections.
* this module does not establish reliable turning and claims no task completion.

Failure semantics: every validation that can fail before ``super().compute`` leaves
the parent memory and the diagnostic record untouched.  An error raised after the
parent law has run may leave the wheel PI integral already advanced; there is no
rollback and no second parent call, so the same physical state must not be retried --
call ``reset`` instead.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from numbers import Real

import numpy as np

from scripts.d1_turn_leg_damping import TurnRawCommandBinding
from wheel_legged_control.d1.wheel_leg_controller import WHEEL_INDICES, D1WheelLegController

#: Unique control schema: no checkpoint recorded against another law loads here.
TURN_YAW_AUTHORITY_SCHEMA = "d1-raw-turn-original-yaw-feedback-authority-v1"
#: Binding/state time agreement tolerance, seconds.
TIME_MATCH_TOLERANCE_S = 1e-10
#: Original wheel rolling radius and original final wheel-speed clip.
WHEEL_RADIUS_M = 0.087
WHEEL_SPEED_LIMIT_RAD_S = 30.0


def _finite_scalar(value, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _readonly(values, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(shape)


def _readonly_bool(values, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=bool))
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    return np.frombuffer(array.tobytes(), dtype=bool).reshape(shape)


@dataclass(frozen=True)
class TurnYawAuthorityRecord:
    """Immutable per-``compute`` diagnostic; arrays are byte-backed read-only copies.

    ``original_*`` fields always report the frozen law for the same tick, so the
    masked feedback and its restored value can be compared directly.  Wheel
    request/torque memory comes from the parent result of this one call; no second
    PI update is performed.
    """

    schema: str
    enabled: bool
    active: bool
    control_time_s: float
    raw_forward_mps: float
    raw_yaw_rate_rps: float
    servo_forward_mps: float
    servo_yaw_rate_rps: float
    body_yaw_rate_rps: float
    original_unclipped_yaw_rps: float
    original_capped_yaw_rps: float
    effective_yaw_request_rps: float
    inner_cap_applied: bool
    inner_cap_occupied: bool
    original_inner_cap_would_be_occupied: bool
    base_wheel_target_rad_s: np.ndarray  # capped yaw, same current geometry
    unclipped_wheel_target_rad_s: np.ndarray  # before the original +/-30 clip
    wheel_target_rad_s: np.ndarray  # exactly the parent's executed targets
    wheel_target_clipped: np.ndarray  # bool (4,)
    wheel_integral_before_nm: np.ndarray
    wheel_integral_after_nm: np.ndarray
    wheel_request_nm: np.ndarray  # unprotected request
    wheel_torque_nm: np.ndarray  # final protected torque

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", str(self.schema))
        for name in (
            "enabled",
            "active",
            "inner_cap_applied",
            "inner_cap_occupied",
            "original_inner_cap_would_be_occupied",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        for name in (
            "control_time_s",
            "raw_forward_mps",
            "raw_yaw_rate_rps",
            "servo_forward_mps",
            "servo_yaw_rate_rps",
            "body_yaw_rate_rps",
            "original_unclipped_yaw_rps",
            "original_capped_yaw_rps",
            "effective_yaw_request_rps",
        ):
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))
        for field in fields(self):
            name = field.name
            if name == "wheel_target_clipped":
                object.__setattr__(self, name, _readonly_bool(self.wheel_target_clipped, (4,), name))
            elif field.type is np.ndarray or name.endswith(("_rad_s", "_nm")):
                object.__setattr__(self, name, _readonly(getattr(self, name), (4,), name))


class TurnYawAuthorityController(D1WheelLegController):
    """Frozen wheel/leg law with the original unclipped yaw feedback on raw turns.

    The gate is exactly ``enabled and raw.forward_velocity_mps == 0.0 and
    raw.yaw_rate_rps != 0.0`` on the bound raw command of the current tick.  Signed
    zeros compare as ordinary floats, and a nonzero heading-servo yaw never opens the
    gate when the raw yaw is zero.  There is no latch and no state beyond the binding
    and the last record.  Zero-residual diagnostic: any nonzero, non-real or wrongly
    shaped action is rejected before the parent law and before any PI update.
    """

    def __init__(self, *, enabled: bool = True, **kwargs) -> None:
        if not isinstance(enabled, (bool, np.bool_)):
            raise TypeError("enabled must be a bool")
        super().__init__(**kwargs)
        self._enabled = bool(enabled)
        if self._enabled:
            # Distinct schema only when the restored authority is live; the disabled
            # branch keeps the parent's control_schema bit-for-bit.
            self.control_schema = TURN_YAW_AUTHORITY_SCHEMA
        self._raw_binding: TurnRawCommandBinding | None = None
        self._last_authority: TurnYawAuthorityRecord | None = None

    # --- read-only configuration -----------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def raw_binding(self) -> TurnRawCommandBinding | None:
        return self._raw_binding

    @property
    def last_authority(self) -> TurnYawAuthorityRecord | None:
        return self._last_authority

    def bind_raw_command(
        self, *, forward_velocity_mps, yaw_rate_rps, control_time_s
    ) -> TurnRawCommandBinding:
        """Bind the raw command of the next execution tick (not a mode latch)."""
        binding = TurnRawCommandBinding(forward_velocity_mps, yaw_rate_rps, control_time_s)
        self._raw_binding = binding
        return binding

    def reset(self) -> None:
        self._raw_binding = None
        self._last_authority = None
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
            raise ValueError("turn yaw authority diagnostic accepts only exactly zero actions")
        return values

    def _require_binding(self, command, state) -> TurnRawCommandBinding:
        binding = self._raw_binding
        if binding is None:
            raise RuntimeError(
                "enabled turn yaw authority requires bind_raw_command for the current tick"
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

    # --- pure geometry / original inner formula ---------------------------
    def _yaw_geometry(self, command, state) -> tuple[np.ndarray, float, float]:
        """Return (lateral wheel offsets, original unclipped r0, original capped r0).

        Uses the SERVO yaw and the measured body yaw rate exactly as the frozen law
        does; the raw yaw is a gate input only.
        """
        yaw = _finite_scalar(state.base_rpy[2], "state.base_rpy[2]")
        lateral_world = np.asarray((-np.sin(yaw), np.cos(yaw), 0.0))
        lateral = np.asarray(state.foot_offset_world, dtype=np.float64) @ lateral_world
        if lateral.shape != (4,) or not np.isfinite(lateral).all():
            raise ValueError("lateral wheel offsets must be four finite values")
        servo_yaw = _finite_scalar(command.yaw_rate_rps, "command.yaw_rate_rps")
        body_yaw = _finite_scalar(
            state.base_angular_velocity_body[2], "state.base_angular_velocity_body[2]"
        )
        r0 = servo_yaw + self.yaw_feedback_gain * (servo_yaw - body_yaw)
        if not np.isfinite(r0):
            raise ValueError("original unclipped yaw feedback must be finite")
        capped = float(np.clip(r0, -self.yaw_request_limit_rps, self.yaw_request_limit_rps))
        return lateral, float(r0), capped

    @staticmethod
    def _wheel_targets(forward: float, yaw_request: float, lateral: np.ndarray) -> np.ndarray:
        """Original geometric wheel target, unclipped."""
        return (forward - yaw_request * lateral) / WHEEL_RADIUS_M

    # --- preview ----------------------------------------------------------
    def nominal_targets(self, command, state, *, ground_height_m: float = 0.0):
        """Pure preview: parent targets, with only the two yaw/wheel fields replaced.

        Inactive ticks return the parent OBJECT itself.  Nothing here records a
        diagnostic, advances the wheel PI, mutates ``yaw_request_limit_rps`` or any
        gain, and the parent is called exactly once.
        """
        if self._enabled:
            self._require_binding(command, state)
        nominal = super().nominal_targets(command, state, ground_height_m=ground_height_m)
        if not self._gate_active():
            return nominal
        lateral, r0, _ = self._yaw_geometry(command, state)
        forward = _finite_scalar(command.forward_velocity_mps, "command.forward_velocity_mps")
        wheels = np.clip(
            self._wheel_targets(forward, r0, lateral),
            -WHEEL_SPEED_LIMIT_RAD_S,
            WHEEL_SPEED_LIMIT_RAD_S,
        )
        if not np.isfinite(wheels).all():
            raise ValueError("restored wheel targets must be finite")
        # Only these two fields change; every leg/extension field is the parent's.
        return dataclasses.replace(
            nominal, nominal_wheel_speed_rad_s=wheels, effective_yaw_request_rps=r0
        )

    # --- control ----------------------------------------------------------
    def compute(self, command, state, action, *, ground_height_m: float = 0.0) -> np.ndarray:
        # 1. Pre-parent validation; nothing below touches PI or the record.
        checked = self._checked_zero_action(action)
        binding = self._raw_binding
        if self._enabled:
            binding = self._require_binding(command, state)
        active = self._gate_active()
        servo_forward = _finite_scalar(command.forward_velocity_mps, "forward_velocity_mps")
        servo_yaw = _finite_scalar(command.yaw_rate_rps, "yaw_rate_rps")
        body_yaw = _finite_scalar(
            state.base_angular_velocity_body[2], "state.base_angular_velocity_body[2]"
        )
        if binding is None:
            record_time = _finite_scalar(state.control_time_s, "state.control_time_s")
            raw_forward, raw_yaw = servo_forward, servo_yaw
        else:
            record_time = binding.control_time_s
            raw_forward, raw_yaw = binding.forward_velocity_mps, binding.yaw_rate_rps
        lateral, r0, capped = self._yaw_geometry(command, state)

        # 2. The frozen law runs EXACTLY once.  Dynamic dispatch into the pure
        #    nominal_targets above is the only control change; the PI update, its
        #    antiwindup, the rated clips, the outward protections and the leg torque
        #    are the parent's.  Any error below may leave the PI advanced: do not
        #    retry the same physical state, reset instead.
        torque = super().compute(command, state, checked, ground_height_m=ground_height_m)
        result = self.last_result

        # 3. Diagnostics read back from the parent result only.
        effective = _finite_scalar(result.effective_yaw_request_rps, "effective_yaw_request_rps")
        expected = r0 if active else capped
        if effective != expected:
            raise ValueError("parent effective yaw request does not match the intended authority")
        unclipped = self._wheel_targets(servo_forward, effective, lateral)
        base_target = np.clip(
            self._wheel_targets(servo_forward, capped, lateral),
            -WHEEL_SPEED_LIMIT_RAD_S,
            WHEEL_SPEED_LIMIT_RAD_S,
        )
        final = np.asarray(result.wheel_speed_target_rad_s, dtype=np.float64)
        if not np.array_equal(
            final, np.clip(unclipped, -WHEEL_SPEED_LIMIT_RAD_S, WHEEL_SPEED_LIMIT_RAD_S)
        ):
            raise ValueError("parent wheel targets do not match the recorded geometry")

        record = TurnYawAuthorityRecord(
            schema=TURN_YAW_AUTHORITY_SCHEMA,
            enabled=self._enabled,
            active=active,
            control_time_s=record_time,
            raw_forward_mps=raw_forward,
            raw_yaw_rate_rps=raw_yaw,
            servo_forward_mps=servo_forward,
            servo_yaw_rate_rps=servo_yaw,
            body_yaw_rate_rps=body_yaw,
            original_unclipped_yaw_rps=r0,
            original_capped_yaw_rps=capped,
            effective_yaw_request_rps=effective,
            inner_cap_applied=not active,
            inner_cap_occupied=(not active) and capped != r0,
            original_inner_cap_would_be_occupied=capped != r0,
            base_wheel_target_rad_s=base_target,
            unclipped_wheel_target_rad_s=unclipped,
            wheel_target_rad_s=final,
            # Expected to be True on the first restored ticks; reported, not hidden.
            wheel_target_clipped=final != unclipped,
            wheel_integral_before_nm=result.memory_before.wheel_integral_nm,
            wheel_integral_after_nm=result.memory_after.wheel_integral_nm,
            wheel_request_nm=np.asarray(result.requested_torque_nm, dtype=np.float64)[
                WHEEL_INDICES
            ],
            wheel_torque_nm=np.asarray(result.torque_nm, dtype=np.float64)[WHEEL_INDICES],
        )
        # 4. The parent result object is retained as-is; only the record is stored,
        #    and a later bind/preview cannot change it.
        self._last_authority = record
        return torque


__all__ = [
    "TIME_MATCH_TOLERANCE_S",
    "TURN_YAW_AUTHORITY_SCHEMA",
    "TurnRawCommandBinding",
    "TurnYawAuthorityController",
    "TurnYawAuthorityRecord",
    "WHEEL_RADIUS_M",
    "WHEEL_SPEED_LIMIT_RAD_S",
]
