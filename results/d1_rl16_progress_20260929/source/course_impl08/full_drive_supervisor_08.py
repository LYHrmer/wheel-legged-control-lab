"""Pure one-tick torque-owner arbitration for a future course controller.

This module has no engine, controller, model, or actuation import. It decides
*which* producer may compute a tick; the physical adapter must call exactly
that producer once and then let D1ControlLoop perform the sole plant.step.
No capability is inferred from an operator key: eligibility is an explicit
validated input, and the learned straight residual is limited to its current
qualified raw speeds and a separately qualified terrain region.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

Owner = Literal["straight06_rl", "course_wheel_yaw", "side_gait", "jump", "safe_stop"]
OWNERS: tuple[Owner, ...] = (
    "straight06_rl", "course_wheel_yaw", "side_gait", "jump", "safe_stop",
)
LOCKED_OWNERS = ("side_gait", "jump")
QUALIFIED_RL_SPEEDS_MPS = (0.20, 0.25)
SUPERVISOR_SCHEMA = "d1-full-drive-exclusive-torque-owner-v1"


def _real(value: Real, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class OwnerIntent:
    tick: int
    control_time_s: float
    raw_forward_mps: float
    raw_yaw_rate_rps: float
    side_direction: int = 0
    jump_pressed: bool = False
    cancel_pressed: bool = False
    fallen: bool = False
    rl_region_qualified: bool = False
    side_start_eligible: bool = False
    jump_start_eligible: bool = False

    def __post_init__(self) -> None:
        if type(self.tick) is not int or self.tick < 0:
            raise ValueError("tick must be a non-negative integer")
        for name in ("control_time_s", "raw_forward_mps", "raw_yaw_rate_rps"):
            object.__setattr__(self, name, _real(getattr(self, name), name))
        if self.control_time_s < 0.0:
            raise ValueError("control time must be non-negative")
        if type(self.side_direction) is not int or self.side_direction not in (-1, 0, 1):
            raise ValueError("side direction must be -1, 0, or +1")
        for name in ("jump_pressed", "cancel_pressed", "fallen", "rl_region_qualified",
                     "side_start_eligible", "jump_start_eligible"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")


@dataclass(frozen=True)
class OwnerDecision:
    schema: str
    tick: int
    control_time_s: float
    owner: Owner
    previous_owner: Owner
    reason: str
    new_owner_entry: bool
    cancel_current: bool
    learned_residual_enabled: bool
    raw_forward_mps: float
    raw_yaw_rate_rps: float


class ExclusiveTorqueOwner:
    """Select and hold exactly one producer across preview/compute of a tick.

    For side gait and jump, the selected producer remains in charge until it
    reports a safe completed handoff. X/cancel asks that same owner to land or
    finish; it never immediately swaps torque ownership in flight. After an
    exception, `fail()` latches the machine until the caller's real env reset.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.owner: Owner = "safe_stop"
        self.locked_owner: Owner | None = None
        self._pending: OwnerDecision | None = None
        self._last_key: tuple[int, float] | None = None
        self._failed = False

    @property
    def failed(self) -> bool:
        return self._failed

    def preview(self, intent: OwnerIntent) -> OwnerDecision:
        if not isinstance(intent, OwnerIntent):
            raise TypeError("owner preview requires OwnerIntent")
        if self._failed:
            raise RuntimeError("owner fault is latched until physical reset")
        key = (intent.tick, intent.control_time_s)
        if self._pending is not None:
            if key != (self._pending.tick, self._pending.control_time_s):
                raise RuntimeError("previous owner decision has not been consumed")
            return self._pending
        if self._last_key is not None and (intent.tick <= self._last_key[0]
                                           or intent.control_time_s <= self._last_key[1]):
            raise RuntimeError("owner preview cannot replay or reverse a physical tick")

        previous = self.owner
        if self.locked_owner is not None:
            owner = self.locked_owner
            reason = "locked_owner_until_safe_landing"
        elif intent.fallen or intent.cancel_pressed:
            owner, reason = "safe_stop", "fallen" if intent.fallen else "operator_cancel"
        elif intent.side_direction:
            if intent.side_start_eligible:
                owner, reason = "side_gait", "guarded_side_start"
            else:
                owner, reason = "safe_stop", "side_request_rejected_by_eligibility"
        elif intent.jump_pressed:
            if intent.jump_start_eligible:
                owner, reason = "jump", "guarded_jump_start"
            else:
                owner, reason = "safe_stop", "jump_request_rejected_by_eligibility"
        elif intent.raw_forward_mps == 0.0 and intent.raw_yaw_rate_rps == 0.0:
            owner, reason = "safe_stop", "zero_motion_request"
        elif (intent.raw_forward_mps in QUALIFIED_RL_SPEEDS_MPS
              and intent.raw_yaw_rate_rps == 0.0 and intent.rl_region_qualified):
            owner, reason = "straight06_rl", "qualified_straight_speed_and_region"
        else:
            owner, reason = "course_wheel_yaw", "nonqualified_for_learned_straight_residual"
        decision = OwnerDecision(
            SUPERVISOR_SCHEMA, intent.tick, intent.control_time_s, owner,
            previous, reason, owner != previous, bool(intent.cancel_pressed),
            owner == "straight06_rl", intent.raw_forward_mps, intent.raw_yaw_rate_rps,
        )
        self._pending = decision
        return decision

    def commit(self, decision: OwnerDecision, *, owner_still_active: bool,
               safe_handoff: bool) -> None:
        """Commit one actual producer call; completion is a measured outcome."""
        if self._failed or decision is not self._pending:
            raise RuntimeError("owner commit does not match the pending physical tick")
        if type(owner_still_active) is not bool or type(safe_handoff) is not bool:
            raise TypeError("owner outcome flags must be bool")
        if decision.owner in LOCKED_OWNERS:
            if owner_still_active:
                if safe_handoff:
                    raise ValueError("an active skill cannot claim safe handoff")
                self.locked_owner = decision.owner
            else:
                if not safe_handoff:
                    raise ValueError("skill ownership cannot end without a safe handoff")
                self.locked_owner = None
        elif owner_still_active or not safe_handoff:
            raise ValueError("ordinary owner must finish its tick without a held lock")
        self.owner = decision.owner
        self._last_key = (decision.tick, decision.control_time_s)
        self._pending = None

    def fail(self) -> None:
        """A partial controller or native step must never be retried."""
        self._failed = True
        self._pending = None
