"""One bounded turn-only wheel-CENTER longitudinal compensation diagnostic.

Scope and honesty
-----------------
This module adds a single algebraic candidate on top of the frozen
``D1WheelLegController``: while the *raw* user command is a pure stationary turn
(exactly zero forward, nonzero yaw rate), each leg's nominal wheel-speed target
gains the measured leg-center longitudinal velocity divided by the wheel radius,
with a **fixed coefficient of one**.  There is no adjustable compensation gain,
no filter, no ramp, no latch, no clock/tick/case-name trigger and no history.

The formula is a rolling approximation of the wheel *center*:

    r * omega_i  ~=  v_forward - r_eff * y_i + u_i

It is therefore only a wheel-center longitudinal correction.  It deliberately
omits wheel-carrier angular motion, contact-point Jacobian differences,
roll/pitch coupling and slip.  It is **not** an exact contact-point inverse
kinematic solution, **not** passive damping (the added wheel torque has no fixed
sign against wheel velocity, so no passivity or discrete-stability argument is
available), and **not** a promise of improved heading tracking or stability.
Offline evidence only makes it a falsifiable candidate on the zero-residual
native flat plane; it is a development candidate, not a proven control fix.

Authority
---------
The parent controller's safety remains authoritative.  Nothing here reimplements
or relaxes the wheel PI law, its antiwindup, support allocation, leg PD, residual
scaling, torque/position/speed protections or the ``+-30 rad/s`` wheel-target
clip (the clip constant below mirrors the parent's existing clip; it is not a new
safety limit).  ``compute`` calls ``super().compute`` exactly once and returns the
parent's torque result unchanged; dynamic dispatch into this module's pure
``nominal_targets`` is the only control change.

Yaw-cap distinction
-------------------
Because the compensation adds a *differential* wheel target, the yaw rate implied
by the corrected left/right wheel targets can exceed ``0.6 rad/s`` even though the
parent's effective body-yaw request stays clipped to ``+-0.6 rad/s``.  The record
stores both quantities separately, and ``target_equivalent_yaw_rps`` is only a
least-squares geometric fit of the wheel *targets*; it is not measured body yaw
rate and must not be reported as body motion.

The module reads nothing but the published immutable ``D1StateEstimate`` and the
servo ``D1Command``: no plant, model, truth, ground-contact or terrain access, no
filesystem or network use, and no imports beyond the standard library and NumPy.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from numbers import Real
from typing import Any

import numpy as np

from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.state_estimation import D1StateEstimate
from wheel_legged_control.d1.wheel_leg_controller import D1WheelLegController

__all__ = [
    "TURN_CENTER_COMPENSATION_SCHEMA",
    "WHEEL_RADIUS_M",
    "TurnCenterRawCommandBinding",
    "TurnCenterCompensationRecord",
    "TurnCenterCompensationController",
    "bind_raw_command_source",
]

#: Unique control schema for the enabled candidate law.  Recorded policies and
#: checkpoints from the frozen law therefore cannot silently load against it.
TURN_CENTER_COMPENSATION_SCHEMA = "d1-turn-wheel-center-velocity-compensation-v1"

#: Nominal wheel radius, identical to the frozen parent's rolling constant.
WHEEL_RADIUS_M = 0.087

# Mirrors the parent's existing nominal wheel-speed clip; not an added limit.
_WHEEL_SPEED_LIMIT_RAD_S = 30.0

# Absolute tolerance used only to confirm that the bound raw command belongs to
# the same control decision as the published state.  It is not a gate epsilon.
_TIME_MATCH_ATOL_S = 1e-12

# Minimum lateral-offset spread accepted by the geometric yaw fit.
_MIN_LATERAL_SPREAD = 1e-12

_LEGS = 4


def _validate_finite_scalar(name: str, value: Any) -> float:
    """Return ``value`` as a finite float, rejecting bools and non-real types."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not a boolean")
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _frozen_float_vector(name: str, values: Any) -> np.ndarray:
    """Copy a length-four real vector into immutable, finite float storage."""

    array = np.asarray(values)
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be a real numeric array")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must not be complex")
    array = np.array(array, dtype=np.float64, copy=True)
    if array.shape != (_LEGS,):
        raise ValueError(f"{name} must have shape ({_LEGS},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(_LEGS)


def _frozen_bool_vector(name: str, values: Any) -> np.ndarray:
    """Copy a length-four boolean vector into immutable storage."""

    array = np.asarray(values)
    if array.dtype != np.bool_:
        raise TypeError(f"{name} must be a boolean array")
    if array.shape != (_LEGS,):
        raise ValueError(f"{name} must have shape ({_LEGS},), got {array.shape}")
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=np.bool_).reshape(_LEGS)


@dataclasses.dataclass(frozen=True, slots=True)
class TurnCenterRawCommandBinding:
    """Immutable raw user command for the control decision being prepared.

    This is *not* the servo command and *not* a control-mode latch: it is the
    stateless raw input of one decision.  The gate reads only these raw values,
    so a raw yaw of zero keeps the candidate inactive even when the heading servo
    requests a large yaw rate.
    """

    forward_velocity_mps: float
    yaw_rate_rps: float
    control_time_s: float

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            object.__setattr__(
                self,
                field.name,
                _validate_finite_scalar(field.name, getattr(self, field.name)),
            )

    @property
    def is_stationary_turn(self) -> bool:
        """True iff forward is exactly zero (either signed zero) and yaw is not."""

        return self.forward_velocity_mps == 0.0 and self.yaw_rate_rps != 0.0


@dataclasses.dataclass(frozen=True, slots=True)
class TurnCenterCompensationRecord:
    """Execution-time snapshot of one compute tick's compensation algebra.

    Arrays are immutable finite copies.  ``target_equivalent_yaw_rps`` is a
    geometric least-squares fit of the recorded wheel targets, not body motion;
    it may exceed the parent's ``effective_yaw_request_rps`` cap of ``0.6``.
    """

    schema: str
    enabled: bool
    active: bool
    control_time_s: float
    raw_forward_mps: float
    raw_yaw_rate_rps: float
    servo_forward_mps: float
    servo_yaw_rate_rps: float
    leg_center_forward_mps: np.ndarray
    correction_unclipped_rad_s: np.ndarray
    base_unclipped_rad_s: np.ndarray
    corrected_unclipped_rad_s: np.ndarray
    base_target_rad_s: np.ndarray
    corrected_target_rad_s: np.ndarray
    target_clipped: np.ndarray
    effective_yaw_request_rps: float
    target_equivalent_yaw_rps: float

    def __post_init__(self) -> None:
        if not isinstance(self.schema, str):
            raise TypeError("schema must be a string")
        for name in ("enabled", "active"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
        for name in (
            "control_time_s",
            "raw_forward_mps",
            "raw_yaw_rate_rps",
            "servo_forward_mps",
            "servo_yaw_rate_rps",
            "effective_yaw_request_rps",
            "target_equivalent_yaw_rps",
        ):
            object.__setattr__(self, name, _validate_finite_scalar(name, getattr(self, name)))
        for name in (
            "leg_center_forward_mps",
            "correction_unclipped_rad_s",
            "base_unclipped_rad_s",
            "corrected_unclipped_rad_s",
            "base_target_rad_s",
            "corrected_target_rad_s",
        ):
            object.__setattr__(self, name, _frozen_float_vector(name, getattr(self, name)))
        object.__setattr__(
            self, "target_clipped", _frozen_bool_vector("target_clipped", self.target_clipped)
        )
        if not self.active and self.correction_unclipped_rad_s.any():
            raise ValueError("inactive records must carry an exactly zero correction")


class TurnCenterCompensationController(D1WheelLegController):
    """Frozen wheel/leg controller plus one pure stationary-turn wheel-center term.

    Gate (no clock, tick index, case name, timer, threshold or latch involved):

        active  <=>  enabled and raw_forward == 0.0 and raw_yaw != 0.0

    Both signed zeros count as zero forward, and any nonzero forward disables the
    term.  The raw values come from ``bind_raw_command`` (the external command
    callback's own output for the decision being prepared), never from the servo
    command, so a nonzero servo yaw with a zero raw yaw stays inactive.

    When active, only ``nominal_wheel_speed_rad_s`` changes; nominal leg targets,
    leg extension, the effective yaw request and every other field of the parent
    targets are preserved by ``dataclasses.replace``.  ``nominal_targets`` stays
    pure: preview mutates no integral, gate, latch, record or result state and is
    idempotent.  The parent's clipping order is preserved by applying exactly one
    final clip to the summed unclipped target; nothing is added after a clip.
    """

    def __init__(self, *, enabled: bool = True, **controller_kwargs: Any) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        super().__init__(**controller_kwargs)
        self._enabled = enabled
        if enabled:
            # Distinct law => distinct schema.  A disabled instance keeps the
            # parent's schema (including the parent's configured-gain variants).
            self.control_schema = TURN_CENTER_COMPENSATION_SCHEMA
        self._raw_binding: TurnCenterRawCommandBinding | None = None
        self._last_compensation: TurnCenterCompensationRecord | None = None

    # ---------------------------------------------------------------- state --

    @property
    def enabled(self) -> bool:
        """True when the candidate law (and its schema) is selected."""

        return self._enabled

    @property
    def raw_binding(self) -> TurnCenterRawCommandBinding | None:
        """Raw command bound for the decision currently being prepared."""

        return self._raw_binding

    @property
    def last_compensation(self) -> TurnCenterCompensationRecord | None:
        """Record of the most recent ``compute``; ``None`` initially and after reset."""

        return self._last_compensation

    def reset(self) -> None:
        self._raw_binding = None
        self._last_compensation = None
        super().reset()

    def bind_raw_command(
        self,
        *,
        forward_velocity_mps: Any,
        yaw_rate_rps: Any,
        control_time_s: Any,
    ) -> TurnCenterRawCommandBinding:
        """Bind one immutable raw command/time for the next preview and compute.

        A later binding simply overwrites this one; ``last_compensation`` always
        stores the binding that the executed ``compute`` actually used, so logs
        must read the record rather than this live property.
        """

        binding = TurnCenterRawCommandBinding(
            forward_velocity_mps=_validate_finite_scalar(
                "forward_velocity_mps", forward_velocity_mps
            ),
            yaw_rate_rps=_validate_finite_scalar("yaw_rate_rps", yaw_rate_rps),
            control_time_s=_validate_finite_scalar("control_time_s", control_time_s),
        )
        self._raw_binding = binding
        return binding

    # ------------------------------------------------------------ validation --

    @staticmethod
    def _validate_action(action: Any) -> np.ndarray:
        """Return the validated residual action: finite real ``(8,)`` exact zeros."""

        array = np.asarray(action)
        if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
            raise TypeError("action must be a real numeric array of eight values")
        if np.issubdtype(array.dtype, np.complexfloating):
            raise TypeError("action must not be complex")
        values = np.array(array, dtype=np.float64, copy=True)
        if values.shape != (8,):
            raise ValueError(f"action must have shape (8,), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("action must contain only finite values")
        if values.any():
            raise ValueError("this diagnostic only accepts an exactly zero eight-vector action")
        return values

    def _gate(self, command: D1Command, state: D1StateEstimate) -> bool:
        """Validate the enabled seam and return the pure raw-command gate.

        Pure: reads only the immutable binding, command and state.  A disabled
        controller delegates transparently and requires no binding at all.
        """

        if not self._enabled:
            return False
        if not isinstance(state, D1StateEstimate):
            raise TypeError("state must be a D1StateEstimate")
        binding = self._raw_binding
        if binding is None:
            raise RuntimeError(
                "enabled turn-center compensation requires a raw command binding "
                "for the current control decision"
            )
        state_time = _validate_finite_scalar("state.control_time_s", state.control_time_s)
        if abs(state_time - binding.control_time_s) > _TIME_MATCH_ATOL_S:
            raise RuntimeError(
                "bound raw command time does not match the published state time; "
                "refusing to use stale raw-command data"
            )
        servo_forward = _validate_finite_scalar(
            "command.forward_velocity_mps", command.forward_velocity_mps
        )
        _validate_finite_scalar("command.yaw_rate_rps", command.yaw_rate_rps)
        if servo_forward != binding.forward_velocity_mps:
            raise RuntimeError(
                "servo forward velocity must equal the bound raw forward velocity; "
                "the heading servo may only shape the yaw request"
            )
        return binding.is_stationary_turn

    # -------------------------------------------------------------- geometry --

    @staticmethod
    def _horizontal_geometry(state: D1StateEstimate) -> tuple[np.ndarray, np.ndarray]:
        """Return lateral offsets ``y`` and leg-center longitudinal speeds ``u``.

        Both use the same horizontal heading frame as the parent's yaw term:
        ``h = (cos psi, sin psi, 0)`` and ``lateral = (-sin psi, cos psi, 0)``.
        ``u`` is a measured leg-center velocity only; it contains no contact,
        carrier-rotation or slip model.
        """

        yaw = float(state.base_rpy[2])
        heading = np.asarray((np.cos(yaw), np.sin(yaw), 0.0))
        lateral = np.asarray((-np.sin(yaw), np.cos(yaw), 0.0))
        lateral_offsets = state.foot_offset_world @ lateral
        leg_center_forward = np.empty(_LEGS, dtype=np.float64)
        for leg in range(_LEGS):
            longitudinal_rows = heading @ state.foot_jacobian[leg, :, :3]
            leg_center_forward[leg] = longitudinal_rows @ state.joint_velocity[
                4 * leg : 4 * leg + 3
            ]
        if not (np.isfinite(lateral_offsets).all() and np.isfinite(leg_center_forward).all()):
            raise ValueError("state geometry and joint velocities must be finite")
        return lateral_offsets, leg_center_forward

    # --------------------------------------------------------- control law ---

    def nominal_targets(
        self, command: D1Command, state: D1StateEstimate, *, ground_height_m: float = 0.0
    ):
        """Parent nominal targets, with wheel-center compensation when gated.

        Inactive returns the parent's object itself, untouched and unwrapped.
        """

        active = self._gate(command, state)
        parent_targets = super().nominal_targets(command, state, ground_height_m=ground_height_m)
        if not active:
            return parent_targets
        lateral_offsets, leg_center_forward = self._horizontal_geometry(state)
        base_unclipped = (
            float(command.forward_velocity_mps)
            - float(parent_targets.effective_yaw_request_rps) * lateral_offsets
        ) / WHEEL_RADIUS_M
        corrected_unclipped = base_unclipped + leg_center_forward / WHEEL_RADIUS_M
        wheels = np.clip(
            corrected_unclipped, -_WHEEL_SPEED_LIMIT_RAD_S, _WHEEL_SPEED_LIMIT_RAD_S
        )
        return dataclasses.replace(parent_targets, nominal_wheel_speed_rad_s=wheels)

    def compute(
        self,
        command: D1Command,
        state: D1StateEstimate,
        action: np.ndarray,
        *,
        ground_height_m: float = 0.0,
    ) -> np.ndarray:
        """Run the frozen parent compute once and archive the execution record.

        All rejection happens before the parent runs, so no memory, integral or
        record state changes on invalid input.  The parent's torque result is
        returned exactly as produced.
        """

        validated_action = self._validate_action(action)
        binding = self._raw_binding
        active = self._gate(command, state)
        torque = super().compute(
            command, state, validated_action, ground_height_m=ground_height_m
        )
        self._last_compensation = self._build_record(command, state, active, binding)
        return torque

    # ---------------------------------------------------------- diagnostics --

    def _build_record(
        self,
        command: D1Command,
        state: D1StateEstimate,
        active: bool,
        binding: TurnCenterRawCommandBinding | None,
    ) -> TurnCenterCompensationRecord:
        """Recompute the algebra from the same pre-step state and parent result.

        The parent's ``compute``/``nominal_targets`` are not called again here.
        """

        result = self.last_result
        if result is None:
            raise RuntimeError("parent compute did not publish a result to archive")
        effective_yaw = _validate_finite_scalar(
            "effective_yaw_request_rps", result.effective_yaw_request_rps
        )
        servo_forward = _validate_finite_scalar(
            "command.forward_velocity_mps", command.forward_velocity_mps
        )
        servo_yaw = _validate_finite_scalar("command.yaw_rate_rps", command.yaw_rate_rps)

        lateral_offsets, leg_center_forward = self._horizontal_geometry(state)
        base_unclipped = (servo_forward - effective_yaw * lateral_offsets) / WHEEL_RADIUS_M
        if active:
            correction_unclipped = leg_center_forward / WHEEL_RADIUS_M
        else:
            correction_unclipped = np.zeros(_LEGS, dtype=np.float64)
        corrected_unclipped = base_unclipped + correction_unclipped
        base_target = np.clip(
            base_unclipped, -_WHEEL_SPEED_LIMIT_RAD_S, _WHEEL_SPEED_LIMIT_RAD_S
        )
        corrected_target = np.clip(
            corrected_unclipped, -_WHEEL_SPEED_LIMIT_RAD_S, _WHEEL_SPEED_LIMIT_RAD_S
        )
        executed_target = np.asarray(result.nominal_wheel_speed_rad_s, dtype=np.float64)
        if not np.array_equal(corrected_target, executed_target):
            raise RuntimeError(
                "recomputed nominal wheel target disagrees with the executed target; "
                "the diagnostic and the control law are out of sync"
            )

        if self._enabled:
            if binding is None:  # pragma: no cover - the gate already rejects this
                raise RuntimeError("enabled compute must archive its raw command binding")
            raw_forward = binding.forward_velocity_mps
            raw_yaw = binding.yaw_rate_rps
            control_time_s = binding.control_time_s
        else:
            # Disabled/unbound diagnostic binding: report the command actually
            # seen and the published state time, with the gate inactive.
            raw_forward = servo_forward
            raw_yaw = servo_yaw
            control_time_s = _validate_finite_scalar(
                "state.control_time_s", state.control_time_s
            )

        return TurnCenterCompensationRecord(
            schema=self.control_schema,
            enabled=self._enabled,
            active=bool(active),
            control_time_s=control_time_s,
            raw_forward_mps=raw_forward,
            raw_yaw_rate_rps=raw_yaw,
            servo_forward_mps=servo_forward,
            servo_yaw_rate_rps=servo_yaw,
            leg_center_forward_mps=leg_center_forward,
            correction_unclipped_rad_s=correction_unclipped,
            base_unclipped_rad_s=base_unclipped,
            corrected_unclipped_rad_s=corrected_unclipped,
            base_target_rad_s=base_target,
            corrected_target_rad_s=corrected_target,
            target_clipped=corrected_target != corrected_unclipped,
            effective_yaw_request_rps=effective_yaw,
            target_equivalent_yaw_rps=_target_equivalent_yaw_rps(
                lateral_offsets, corrected_target
            ),
        )


def _target_equivalent_yaw_rps(
    lateral_offsets: np.ndarray, corrected_target_rad_s: np.ndarray
) -> float:
    """Least-squares yaw implied by the wheel *targets*; a geometric fit only.

    This is not body yaw rate and is not limited by the parent's ``0.6 rad/s``
    effective yaw cap: leg motion can make the differential target imply more.
    """

    centered = lateral_offsets - float(np.mean(lateral_offsets))
    spread = float(centered @ centered)
    if not np.isfinite(spread) or spread < _MIN_LATERAL_SPREAD:
        raise ValueError("degenerate lateral offset spread; the yaw fit is undefined")
    numerator = float(centered @ (WHEEL_RADIUS_M * corrected_target_rad_s))
    equivalent = -numerator / spread
    if not np.isfinite(equivalent):
        raise ValueError("target equivalent yaw must be finite")
    return equivalent


def bind_raw_command_source(
    controller: TurnCenterCompensationController,
    command_source: Callable[..., Any],
    control_time_source: Callable[[], Any],
) -> Callable[..., Any]:
    """Wrap an existing external command callback with one raw-command binding.

    The wrapper invokes the original callback exactly once per call, binds that
    call's raw forward/yaw and the caller-supplied control time onto the
    controller, and returns the original command object unchanged (same
    identity, no copy, no clamping, no substitution).  It queries no other
    source, caches nothing and computes no reference of its own; the binding is
    a stateless input to the decision being prepared, not a control-mode latch.

    ``control_time_source`` must return the control time of that same decision.
    The bound time is used only to confirm decision consistency; it never gates.
    """

    if not callable(command_source):
        raise TypeError("command_source must be callable")
    if not callable(control_time_source):
        raise TypeError("control_time_source must be callable")
    if not isinstance(controller, TurnCenterCompensationController):
        raise TypeError("controller must be a TurnCenterCompensationController")

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        command = command_source(*args, **kwargs)
        for attribute in ("forward_velocity_mps", "yaw_rate_rps"):
            if not hasattr(command, attribute):
                raise TypeError(f"raw command must expose {attribute}")
        controller.bind_raw_command(
            forward_velocity_mps=command.forward_velocity_mps,
            yaw_rate_rps=command.yaw_rate_rps,
            control_time_s=control_time_source(),
        )
        return command

    return wrapped
