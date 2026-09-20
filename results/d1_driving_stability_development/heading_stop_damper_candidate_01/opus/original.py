"""Stop-mode longitudinal leg-velocity damping on top of the frozen D1 controller.

This module is a *diagnostic candidate only*: it adds one bounded joint-space
damping term to the frozen wheel/leg torque law after a commanded stop, so a
paired physical comparison can falsify it.  Nothing here is an accepted fix for
the late-phase leg/body rebound, and no discrete closed-loop passivity claim is
made -- only the algebraic sampled-instant inequality sum(delta * qdot) <= 0.

Boundaries kept deliberately tight:

* the frozen controller/model/state-estimation sources are imported, never edited;
* ``super().compute`` is called exactly once per call, keeping the original wheel
  PI update, targets, support, attitude feedback and yaw servo untouched;
* the damping torque is added to ``requested_torque_nm`` (the *unclipped* request)
  and the original safety sequence (rated-torque clip, outward position/speed
  suppression) is then repeated in the original order;
* the four wheel channels receive exactly zero increment;
* no ground truth, no plant/model access, no body-velocity feedback, no contact
  gating, no integral reset, no target shaping, no gain selection, no tick/case
  triggers, and a single fixed gain with no tuning branch.

The latch that enables the damper is advanced solely inside ``compute``, so a
pure preview (``nominal_targets``) never consumes a command twice.  A ``compute``
call that is not followed by a physical execution therefore leaves the latch
advanced, and the environment must be reset before retrying, exactly as the
existing control loop already requires.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
from dataclasses import dataclass

import numpy as np

# --- frozen parent import -------------------------------------------------
# The frozen controller module is imported as data; its location may differ per
# checkout, so a small candidate list (overridable by environment) is tried.
_MODULE_CANDIDATES: tuple[str, ...] = (
    os.environ.get("D1_WHEEL_LEG_CONTROL_MODULE", ""),
    "d1.wheel_leg_control",
    "d1.wheel_leg_controller",
    "d1.wheel_leg",
    "wheel_legged_control_lab.d1.wheel_leg_control",
    "wheel_legged_control_lab.d1.wheel_leg_controller",
    "src.d1.wheel_leg_control",
)


def _load_parent_module():
    """Return the frozen module that defines D1WheelLegController."""
    errors: list[str] = []
    for name in _MODULE_CANDIDATES:
        if not name:
            continue
        try:
            module = importlib.import_module(name)
        except ImportError as error:  # keep searching; report all failures once
            errors.append(f"{name}: {error}")
            continue
        if hasattr(module, "D1WheelLegController"):
            return module
        errors.append(f"{name}: no D1WheelLegController")
    raise ImportError(
        "could not import the frozen D1WheelLegController module; set "
        "D1_WHEEL_LEG_CONTROL_MODULE. Tried -> " + "; ".join(errors)
    )


_PARENT = _load_parent_module()
D1WheelLegController = _PARENT.D1WheelLegController
JOINT_TORQUE_LIMIT = _PARENT.JOINT_TORQUE_LIMIT
JOINT_POSITION_LOW = _PARENT.JOINT_POSITION_LOW
JOINT_POSITION_HIGH = _PARENT.JOINT_POSITION_HIGH
JOINT_VELOCITY_LIMIT = _PARENT.JOINT_VELOCITY_LIMIT
WHEEL_INDICES = np.asarray(_PARENT.WHEEL_INDICES)

# --- fixed public contract ------------------------------------------------
#: Unique control schema so no checkpoint recorded against another law loads here.
STOP_LEG_DAMPING_SCHEMA = "d1-stop-leg-longitudinal-damping-v1"
#: Per-leg longitudinal damping, N*s/m. Derived once from the nominal-pose modal
#: reduction (M_eff, K_eff, B_old, B_critical) documented in the diagnosis
#: contract; it is never re-selected from physical outcomes.
STOP_LEG_DAMPING_NSPM = 126.4374005337902

_ZERO16 = np.zeros(16)
_ZERO4 = np.zeros(4)


def _readonly(values: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    """Copy to a finite, contiguous, read-only array of the expected shape."""
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64)).copy()
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class StopLegDampingRecord:
    """Immutable per-call diagnostic record; arrays are copied and read-only.

    Recorded for every ``compute`` call, including disabled and inactive ones,
    where the damping quantities are exactly zero and base/total requests match.
    ``joint_power_w`` is the raw algebraic sum(delta * qdot); it is never clamped
    and is expected to be non-positive up to floating-point rounding.
    """

    schema: str
    enabled: bool
    active: bool
    previous_forward_mps: float | None
    forward_command_mps: float
    damping_nspm: float
    leg_relative_forward_mps: np.ndarray  # u_i, shape (4,)
    delta_torque_nm: np.ndarray  # shape (16,), wheels exactly zero
    base_leg_pd_nm: np.ndarray  # shape (16,)
    base_requested_torque_nm: np.ndarray  # shape (16,)
    total_requested_torque_nm: np.ndarray  # shape (16,)
    safe_torque_nm: np.ndarray  # shape (16,)
    joint_power_w: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", str(self.schema))
        for name in ("enabled", "active"):
            object.__setattr__(self, name, bool(getattr(self, name)))
        previous = self.previous_forward_mps
        if previous is not None:
            previous = float(previous)
            if not np.isfinite(previous):
                raise ValueError("previous_forward_mps must be finite or None")
            object.__setattr__(self, "previous_forward_mps", previous)
        for name in ("forward_command_mps", "damping_nspm", "joint_power_w"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
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
        if np.any(self.delta_torque_nm[WHEEL_INDICES] != 0.0):
            raise ValueError("wheel damping increments must be exactly zero")


class StopLegDampingController(D1WheelLegController):
    """Frozen wheel/leg law plus one fixed post-stop leg damping term.

    Activation is a pure command latch, never a clock, tick index or case name:
    inactive after ``reset``; armed only when the executed forward command goes
    from nonzero to exactly zero on adjacent compute calls; held while the zero
    command continues; closed immediately by any nonzero forward command. An
    initial zero-velocity settling phase therefore never activates the damper.

    This is a zero-residual diagnostic: any nonzero action is rejected before the
    parent law runs and before any latch state moves, in both enabled and
    disabled modes, so a policy checkpoint cannot be smuggled through it.
    """

    def __init__(self, *, enabled: bool = True, **kwargs) -> None:
        if not isinstance(enabled, (bool, np.bool_)):
            # Numeric 0/1 is rejected: the switch must be an explicit boolean.
            raise TypeError("enabled must be a bool")
        super().__init__(**kwargs)
        self._enabled = bool(enabled)
        if self._enabled:
            # Distinct schema only when the added law is live; the disabled
            # branch keeps the parent's control_schema bit-for-bit, and
            # action_schema is never touched in either branch.
            self.control_schema = STOP_LEG_DAMPING_SCHEMA
        self._previous_forward_mps: float | None = None
        self._stop_latched = False
        self.last_damping: StopLegDampingRecord | None = None

    # --- read-only configuration -----------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def damping_nspm(self) -> float:
        return STOP_LEG_DAMPING_NSPM

    @property
    def stop_damping_active(self) -> bool:
        return self._enabled and self._stop_latched

    @property
    def previous_forward_mps(self) -> float | None:
        return self._previous_forward_mps

    def reset(self) -> None:
        """Clear the latch, the previous command and the record, then the parent."""
        self._previous_forward_mps = None
        self._stop_latched = False
        self.last_damping = None
        super().reset()

    # --- control ----------------------------------------------------------
    def compute(
        self,
        command,
        state,
        action: np.ndarray,
        *,
        ground_height_m: float = 0.0,
    ) -> np.ndarray:
        # 1. Zero-residual gate, before the parent law and before any latch move.
        checked = np.asarray(action, dtype=np.float64)
        if checked.shape != (8,) or not np.isfinite(checked).all():
            raise ValueError("action must contain eight finite values")
        if np.any(checked != 0.0):
            raise ValueError(
                "stop leg damping diagnostic accepts only exactly zero residual actions"
            )
        forward = float(command.forward_velocity_mps)
        if not np.isfinite(forward):
            raise ValueError("command forward_velocity_mps must be finite")

        # 2. Latch decision from the command pair only (not yet committed).
        previous = self._previous_forward_mps
        if forward != 0.0:
            latched = False  # any new nonzero command closes the latch at once
        elif previous is not None and previous != 0.0:
            latched = True  # nonzero -> exact zero transition arms the damper
        else:
            latched = self._stop_latched  # continued zero holds the current latch
        active = self._enabled and latched

        # 3. The frozen law runs exactly once, keeping its wheel PI update.
        base_torque = super().compute(command, state, action, ground_height_m=ground_height_m)
        base_result = self.last_result

        if not active:
            # Disabled or pre-stop: parent result and returned object are passed
            # through untouched so logs stay bit-for-bit identical to bypass.
            record = StopLegDampingRecord(
                schema=self.control_schema,
                enabled=self._enabled,
                active=False,
                previous_forward_mps=previous,
                forward_command_mps=forward,
                damping_nspm=STOP_LEG_DAMPING_NSPM,
                leg_relative_forward_mps=_ZERO4,
                delta_torque_nm=_ZERO16,
                base_leg_pd_nm=base_result.leg_pd_nm,
                base_requested_torque_nm=base_result.requested_torque_nm,
                total_requested_torque_nm=base_result.requested_torque_nm,
                safe_torque_nm=base_result.torque_nm,
                joint_power_w=0.0,
            )
            self.last_damping = record
            self._previous_forward_mps = forward
            self._stop_latched = latched
            return base_torque

        # 4. Longitudinal joint-relative damping, three leg joints per leg.
        rotation = np.asarray(state.base_rotation, dtype=np.float64)
        jacobian = np.asarray(state.foot_jacobian, dtype=np.float64)
        velocity = np.asarray(state.joint_velocity, dtype=np.float64)
        forward_axis = rotation[:, 0]  # body x expressed in world
        u = np.zeros(4)
        delta = np.zeros(16)
        for leg in range(4):
            # World linear Jacobian of the wheel center w.r.t. the 3 leg joints,
            # projected on the body forward axis.
            jx = forward_axis @ jacobian[leg, :, :3]
            u[leg] = float(jx @ velocity[4 * leg : 4 * leg + 3])
            delta[4 * leg : 4 * leg + 3] = -STOP_LEG_DAMPING_NSPM * jx * u[leg]
        delta[WHEEL_INDICES] = 0.0  # wheel channels strictly untouched
        if not np.isfinite(delta).all() or not np.isfinite(u).all():
            raise ValueError("stop leg damping produced nonfinite values")

        # 5. Add to the unclipped request, then repeat the original safety order.
        combined = base_result.requested_torque_nm + delta
        safe = np.clip(combined, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
        position = np.asarray(state.joint_position, dtype=np.float64)
        upper_outward = (position >= JOINT_POSITION_HIGH) & (safe > 0)
        lower_outward = (position <= JOINT_POSITION_LOW) & (safe < 0)
        speed_outward = (np.abs(velocity) >= JOINT_VELOCITY_LIMIT) & (safe * velocity > 0)
        safe[upper_outward | lower_outward | speed_outward] = 0.0
        if not np.isfinite(combined).all() or not np.isfinite(safe).all():
            raise ValueError("stop leg damping produced nonfinite torques")

        # 6. Keep the requested = leg_feedback + support + wheel identity by
        # folding delta into leg_pd_nm, with the base terms kept in the record.
        result = dataclasses.replace(
            base_result,
            leg_pd_nm=base_result.leg_pd_nm + delta,
            requested_torque_nm=combined,
            torque_nm=safe,
            torque_limited=safe != combined,
        )
        record = StopLegDampingRecord(
            schema=self.control_schema,
            enabled=True,
            active=True,
            previous_forward_mps=previous,
            forward_command_mps=forward,
            damping_nspm=STOP_LEG_DAMPING_NSPM,
            leg_relative_forward_mps=u,
            delta_torque_nm=delta,
            base_leg_pd_nm=base_result.leg_pd_nm,
            base_requested_torque_nm=base_result.requested_torque_nm,
            total_requested_torque_nm=combined,
            safe_torque_nm=safe,
            # Algebraic sampled-instant joint power of the added term only,
            # -b * sum(u_i^2) analytically; reported unclamped.
            joint_power_w=float(np.sum(delta * velocity)),
        )

        # 7. Commit only after the full computation and record succeeded.
        self.last_result = result
        self.last_damping = record
        self._previous_forward_mps = forward
        self._stop_latched = latched
        return safe.copy()


__all__ = [
    "STOP_LEG_DAMPING_SCHEMA",
    "STOP_LEG_DAMPING_NSPM",
    "StopLegDampingRecord",
    "StopLegDampingController",
]
