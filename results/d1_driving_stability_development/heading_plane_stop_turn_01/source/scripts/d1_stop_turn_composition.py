"""Composition of the qualified post-stop leg damping and the qualified raw-turn
restoration of the ORIGINAL unclipped body-yaw feedback authority.

This module adds **no new control formula**.  It composes two already qualified
operations through a *single* parent chain:

    StopTurnCompositionController
        -> TurnYawAuthorityController   (raw stationary-turn yaw authority)
            -> D1WheelLegController     (frozen original law, wheel PI, protections)

``super().compute`` is called exactly once per call, so the original wheel PI
update, its antiwindup, the leg PD/support/attitude terms, the rated clips and the
outward position/speed suppression run exactly once, on the authority-selected
nominal target.  The fixed stop damping increment is then added to the *unclipped*
request and the original protection sequence is repeated in the original order.
Neither ``StopLegDampingController`` nor any second parent controller is
instantiated, inherited or shadow-executed; there is exactly one PI memory.

Boundaries kept deliberately tight:

* one boolean switch enables/disables both mechanisms together; there is no
  independently selectable mode, gain, cap, blend or threshold, and the damping
  gain is imported, never recalculated or tuned;
* the raw turn gate stays exactly ``raw.forward == 0.0 and raw.yaw != 0.0`` on the
  bound raw command of the current tick, independent of the stop latch;
* the stop latch is a pure command latch on the actually consumed
  ``command.forward_velocity_mps``: nonzero closes it, an exact zero after a
  nonzero arms it, a continued zero holds it; signed zero counts as zero and there
  is no epsilon, tick, case, clock, contact, truth, body-speed, raw-yaw or
  target-rate trigger.  Initial zeros never arm it;
* ``nominal_targets`` is inherited unchanged and stays pure: a preview never
  advances the latch, the PI or the raw binding;
* the four wheel channels receive exactly zero increment, so the inherited
  ``last_authority`` wheel request/torque snapshot remains valid for the executed
  parent law; it is not a record of the final leg torque -- ``last_result`` and
  ``last_damping`` supply that.

Overlap is not resolved away: during a drive -> stop -> turn sequence the stop
latch stays armed while forward remains exactly zero, so both mechanisms can be
active on the same tick.  Neither one is suppressed, the turn gate is not widened,
yaw does not unlatch the stop, and the gain does not change.  The two original
turn cases begin with zero forward and therefore never arm stop damping; original
stops and forward yaw disturbances leave the authority gate inactive.

Honest diagnostics: ``joint_power_w`` is the sampled algebraic ``delta @ qdot`` of
the added increment only.  It is not a global discrete passivity claim, not the
actually protected incremental dissipation (the clip/suppression sequence is
reapplied after the sum) and not native-substep work.  No model, plant or ground
truth is accessed here.

Failure semantics: every validation that can fail before ``super().compute``
leaves the inherited memory, the own latch and the records untouched.  An error
raised after the inherited law has run may leave the wheel PI and the authority
record already advanced while the own latch is still uncommitted; the error is
propagated as-is.  There is no rollback, no retry, no compensating second compute
and no silent reset -- the execution contract terminates, the partial evidence is
preserved, and a new ``reset`` is required before the next run.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from scripts.d1_stop_leg_damping import (
    STOP_LEG_DAMPING_NSPM,
    STOP_LEG_DAMPING_SCHEMA,
    StopLegDampingRecord,
)
from scripts.d1_turn_yaw_authority import TurnYawAuthorityController
from wheel_legged_control.d1.model import (
    JOINT_POSITION_HIGH,
    JOINT_POSITION_LOW,
    JOINT_TORQUE_LIMIT,
    JOINT_VELOCITY_LIMIT,
)
from wheel_legged_control.d1.wheel_leg_controller import WHEEL_INDICES

#: Unique control schema for the composition: no checkpoint recorded against
#: another law -- including either single mechanism -- loads here.
STOP_TURN_COMPOSITION_SCHEMA = "d1-stop-damping-original-turn-feedback-v1"

_ZERO4 = np.zeros(4)
_ZERO16 = np.zeros(16)


class StopTurnCompositionController(TurnYawAuthorityController):
    """Frozen law + restored raw-turn yaw authority + fixed post-stop leg damping.

    Single inheritance: the raw-turn authority parent is the only control parent,
    and it in turn calls the frozen original controller.  The inherited
    ``bind_raw_command``, ``raw_binding``, ``last_authority``, ``enabled``,
    ``_checked_zero_action``, ``_require_binding`` and the pure ``nominal_targets``
    are used unchanged; no nominal-target override exists or is needed.

    Zero-residual diagnostic: any nonzero, non-real, boolean, string, complex or
    wrongly shaped action is rejected before the inherited law runs and before any
    own state moves, in both enabled and disabled modes, so a policy checkpoint
    cannot be smuggled through it.  A stale or mismatched raw binding is rejected
    the same way while enabled.
    """

    def __init__(self, *, enabled: bool = True, **kwargs) -> None:
        # The parent performs the strict boolean validation of ``enabled``; it is
        # not re-implemented or relaxed here.
        super().__init__(enabled=enabled, **kwargs)
        if self.enabled:
            # Distinct schema only when the composition is live; the disabled
            # branch keeps the original frozen parent's schema and dynamics, and
            # the action schema is never touched in either branch.
            self.control_schema = STOP_TURN_COMPOSITION_SCHEMA
        self._previous_forward_mps: float | None = None
        self._stop_latched = False
        self.last_damping: StopLegDampingRecord | None = None

    # --- read-only configuration -----------------------------------------
    @property
    def damping_nspm(self) -> float:
        """The single fixed imported gain, N*s/m; never re-selected or tuned."""
        return STOP_LEG_DAMPING_NSPM

    @property
    def stop_damping_active(self) -> bool:
        return self._enabled and self._stop_latched

    @property
    def previous_forward_mps(self) -> float | None:
        return self._previous_forward_mps

    # --- lifecycle --------------------------------------------------------
    def reset(self) -> None:
        """Clear the own latch/previous command/record, then the whole parent chain.

        ``super().reset()`` is called exactly once and clears the raw binding, the
        authority record and the original PI memory.  Nothing calls this at command
        boundaries: the latch is command driven, not reset driven.
        """
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
        # 1. Pre-parent validation only; no own state is committed below until the
        #    full result and record have been built.
        checked = self._checked_zero_action(action)
        forward = float(command.forward_velocity_mps)
        if not np.isfinite(forward):
            raise ValueError("command forward_velocity_mps must be finite")
        if self._enabled:
            # Raw binding remains mandatory, same-time checked, and must preserve
            # servo/raw forward equality.  The inherited compute performs the very
            # same check; doing it here keeps the own latch untouched on failure.
            self._require_binding(command, state)

        # 2. Local, uncommitted stop-latch decision from the actually consumed
        #    forward command and the previous successful compute only.  The raw
        #    binding and any preview never advance this state.
        previous = self._previous_forward_mps
        if forward != 0.0:
            latched = False  # any nonzero forward command closes the latch at once
        elif previous is not None and previous != 0.0:
            latched = True  # nonzero -> exact (possibly signed) zero arms the damper
        else:
            latched = self._stop_latched  # continued zero holds the current latch
        active = self._enabled and latched

        # 3. The single parent chain runs EXACTLY once: composition -> authority ->
        #    original controller.  The original PI update, its antiwindup, the leg
        #    targets, support and attitude terms and the protections are the
        #    parent's, evaluated with the authority-selected nominal target.  Any
        #    error below may leave that memory advanced: do not retry the same
        #    physical state, reset instead.
        base_torque = super().compute(command, state, checked, ground_height_m=ground_height_m)
        base_result = self.last_result

        if not active:
            # Disabled, pre-stop or reopened: identical to the qualified authority
            # candidate, bit-for-bit.  The parent result object and the returned
            # torque object are passed through untouched.
            record = StopLegDampingRecord(
                schema=STOP_LEG_DAMPING_SCHEMA,
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
        #    Arithmetic and operation order copied exactly from the qualified stop
        #    module, including the full body x axis (not a horizontal heading).
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

        # 5. Add to the unclipped request, then repeat the original safety order
        #    with the original arrays and comparisons.  No torque is added after.
        combined = base_result.requested_torque_nm + delta
        safe = np.clip(combined, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
        position = np.asarray(state.joint_position, dtype=np.float64)
        upper_outward = (position >= JOINT_POSITION_HIGH) & (safe > 0)
        lower_outward = (position <= JOINT_POSITION_LOW) & (safe < 0)
        speed_outward = (np.abs(velocity) >= JOINT_VELOCITY_LIMIT) & (safe * velocity > 0)
        safe[upper_outward | lower_outward | speed_outward] = 0.0
        if not np.isfinite(combined).all() or not np.isfinite(safe).all():
            raise ValueError("stop leg damping produced nonfinite torques")

        # 6. Keep the requested = leg_feedback + support + wheel identity by folding
        #    delta into leg_pd_nm.  Every other field -- wheel request, wheel memory
        #    before/after, effective yaw request, targets -- stays the parent's.
        result = dataclasses.replace(
            base_result,
            leg_pd_nm=base_result.leg_pd_nm + delta,
            requested_torque_nm=combined,
            torque_nm=safe,
            torque_limited=safe != combined,
        )
        record = StopLegDampingRecord(
            schema=STOP_LEG_DAMPING_SCHEMA,  # the original stop mechanism's record
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
            # Sampled algebraic joint power of the added term only, -b*sum(u_i^2)
            # analytically; reported unclamped and claiming nothing further.
            joint_power_w=float(np.sum(delta * velocity)),
        )

        # 7. Commit only after the full computation and record construction
        #    succeeded.  ``last_authority`` deliberately keeps the executed parent
        #    snapshot: delta has zero wheel entries and the identical protection is
        #    reapplied componentwise, so its wheel request/torque stay valid.
        self.last_result = result
        self.last_damping = record
        self._previous_forward_mps = forward
        self._stop_latched = latched
        return safe.copy()


__all__ = [
    "STOP_TURN_COMPOSITION_SCHEMA",
    "StopTurnCompositionController",
]
