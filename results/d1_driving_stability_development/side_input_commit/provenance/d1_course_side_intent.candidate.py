"""Pure A/D single-press side-step commit intent for the D1 course runner.

This module owns *only* the operator input semantics of the commit-cycle side
step (``side_input_mode=commit_cycle_v1``): a short tap of A or D while idle
requests exactly one complete side-step cycle, holding the key repeats at the
cycle completion boundary, releasing stops the repeat while the already
committed cycle keeps running, and X (or focus loss / input timeout / fall
recovery) is an explicit cancel channel.

Nothing here touches a plant, a controller, a GUI, the filesystem, a clock or a
random generator. No torque, pose, gait phase or terrain judgement is computed:
the real ``SideStepController`` / ``FastSideStepController`` guards (four wheel
contacts, contact plane near world z=0, attitude, linear and angular speed)
remain the final physical authority, and ``CourseSideStepDrive`` remains the
single torque owner and the single ``plant.step`` caller.

The layer deliberately keeps no action queue and no second gait state machine.
Its whole memory is a release barrier plus the previous ``controller_failed``
value, so a release can never be delayed behind several queued clicks.
"""

from __future__ import annotations

from dataclasses import dataclass

_DIRECTIONS = (-1, 0, 1)


@dataclass(frozen=True)
class SideStepIntentDecision:
    """Immutable result of resolving one input packet.

    Attributes:
        drive_direction: Direction to hand to ``CourseSideStepDrive.step``.
            While a cycle is active without cancellation, this mirrors the
            controller's own ``active_direction`` so release or reversal cannot
            switch the running cycle. Cancellation returns zero. While idle it
            is a *request* to start, never a claim that a cycle has started.
        cancel_current: True when the caller must ask the side controller to
            cancel, so it can run its own safe abort_land / abort_hold path.
            Handing control back to legacy is still gated on the controller.
        finish_current_cycle: UI/log semantics only. True when a cycle is
            active, no cancel is requested this tick, and either the held
            direction differs from the active direction or new starts are
            inhibited. It never changes the current torque owner.
        repeat_direction: Direction that may be committed again at the next
            completion boundary, or 0 when the repeat has stopped. Always 0
            while ``start_inhibited`` is set, so no mode-forbidden repeat plan
            is ever displayed.
        blocked_until_release: True while the release barrier holds; new press
            and held requests are discarded until a real all-keys-released
            input packet re-arms the layer.
        reason: One of ``idle``, ``start_requested``, ``continuing_held``,
            ``finish_current_cycle``, ``reverse_after_cycle``,
            ``cancel_requested``, ``controller_failed``,
            ``blocked_until_release``, ``inhibited``, ``conflicting_keys``.
            The caller records its own ``cancel_reason`` string verbatim.
    """

    drive_direction: int
    cancel_current: bool
    finish_current_cycle: bool
    repeat_direction: int
    blocked_until_release: bool
    reason: str


class CourseSideStepIntent:
    """Resolve raw A/D keyboard truth into one committed side-step intent.

    A freshly constructed instance has no release barrier, so the very first
    user key press works normally. ``reset`` (after an explicit zone / R reset)
    installs a barrier instead, because a synthesized ``side_direction == 0``
    from a reset command must not be mistaken for a physical key release.
    """

    def __init__(self) -> None:
        self._blocked_until_release = False
        self._previous_controller_failed = False

    def reset(self) -> None:
        """Clear intent memory and require a real release before restarting.

        Intent memory only: this never touches the simulation, the drive
        adapter or the side controller, and it never fabricates a completion.
        """
        self._blocked_until_release = True
        self._previous_controller_failed = False

    @staticmethod
    def _checked_direction(name: str, value: object) -> int:
        """Return ``value`` as a -1/0/+1 direction, rejecting bools."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a non-bool int of -1, 0 or +1")
        if value not in _DIRECTIONS:
            raise ValueError(f"{name} must be -1, 0 or +1")
        return int(value)

    @staticmethod
    def _checked_bool(name: str, value: object) -> bool:
        """Return ``value`` unchanged, requiring a genuine bool."""
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool")
        return value

    def resolve(
        self,
        held_direction: int,
        *,
        keys_released: bool,
        active: bool,
        active_direction: int = 0,
        press_direction: int = 0,
        cancel_reason: str | None = None,
        controller_failed: bool = False,
        start_inhibited: bool = False,
    ) -> SideStepIntentDecision:
        """Resolve one input packet into a :class:`SideStepIntentDecision`.

        Args:
            held_direction: Raw A/D held difference sampled this poll, -1/0/+1.
            keys_released: True only when *both* A and D are actually released.
                Two keys held together cancel out to ``held_direction == 0``
                with ``keys_released=False``; that conflict stops the repeat and
                must never be treated as the release that re-arms the layer.
            active: Whether the side controller really is running a cycle. This
                must come from the controller, never from key expectations.
            active_direction: The running cycle's own direction; required to be
                non-zero when ``active`` is True.
            press_direction: One newly consumed PRESS edge for this poll, for
                taps whose PRESS and RELEASE land in the same poll so the held
                sample is already 0. Key REPEAT produces no press, an already
                consumed event must not be passed again, and only the last of
                several same-poll presses is kept. It is part of this input
                packet only; it is never queued for later.
            cancel_reason: None, or a non-empty explanation for an explicit
                cancel (X held, focus loss, input timeout, fallen/recovery).
            controller_failed: ``bool(side.failure)``. A sticky old failure may
                stay True after ``done``; only the False->True edge inhibits,
                so the layer does not re-lock every tick.
            start_inhibited: True while a mode forbids new cycles (accepted or
                running jump, recovery, other exclusive guards).

        Returns:
            The decision for this tick. ``drive_direction`` is a request; the
            side controller's physical guards still decide whether the cycle
            begins, and a guard-rejected tap leaves no delayed request behind.

        Raises:
            TypeError: On a wrong argument type (including bools passed as
                directions, or non-bools passed as flags).
            ValueError: On an out-of-range direction, an empty
                ``cancel_reason``, ``keys_released`` with a non-zero
                ``held_direction``, or ``active`` without a direction.

        All arguments are validated before any internal memory changes, so an
        invalid call leaves the release barrier and failure edge untouched.
        """
        held = self._checked_direction("held_direction", held_direction)
        active_dir = self._checked_direction("active_direction", active_direction)
        press = self._checked_direction("press_direction", press_direction)
        released = self._checked_bool("keys_released", keys_released)
        is_active = self._checked_bool("active", active)
        failed = self._checked_bool("controller_failed", controller_failed)
        inhibited = self._checked_bool("start_inhibited", start_inhibited)
        if cancel_reason is not None:
            if not isinstance(cancel_reason, str):
                raise TypeError("cancel_reason must be None or a non-empty str")
            if not cancel_reason:
                raise ValueError("cancel_reason must be None or a non-empty str")
        if released and held != 0:
            raise ValueError("keys_released=True requires held_direction == 0")
        if is_active and active_dir == 0:
            raise ValueError("active=True requires a non-zero active_direction")

        # Validation finished; only now is internal memory allowed to change.
        failure_edge = failed and not self._previous_controller_failed
        self._previous_controller_failed = failed
        conflicting_keys = held == 0 and not released

        # 1/2. An explicit cancel, or a new failure edge, outranks every key.
        if cancel_reason is not None or failure_edge:
            self._blocked_until_release = True
            return SideStepIntentDecision(
                drive_direction=0,
                cancel_current=is_active,
                finish_current_cycle=False,
                repeat_direction=0,
                blocked_until_release=True,
                reason="cancel_requested" if cancel_reason is not None else "controller_failed",
            )

        # 3. The barrier discards new press/held requests. Exactly one later
        # cancel-free, mode-permitting, truly released packet re-arms it, and
        # that packet only re-arms: it never also commits the tap.
        if self._blocked_until_release:
            if released and not inhibited:
                self._blocked_until_release = False
            still_blocked = self._blocked_until_release
            return SideStepIntentDecision(
                drive_direction=0,
                cancel_current=is_active,
                finish_current_cycle=False,
                repeat_direction=0,
                blocked_until_release=still_blocked,
                reason="blocked_until_release" if (still_blocked or is_active) else "idle",
            )

        # 4. A committed cycle keeps its own direction and its own owner.
        if is_active:
            repeat = 0 if (inhibited or conflicting_keys) else held
            finish = inhibited or held != active_dir
            if inhibited:
                reason = "inhibited"
            elif conflicting_keys:
                reason = "conflicting_keys"
            elif repeat == 0:
                reason = "finish_current_cycle"
            elif repeat == active_dir:
                reason = "continuing_held"
            else:
                reason = "reverse_after_cycle"
            return SideStepIntentDecision(
                drive_direction=active_dir,
                cancel_current=False,
                finish_current_cycle=finish,
                repeat_direction=repeat,
                blocked_until_release=False,
                reason=reason,
            )

        # 5. Idle under a mode guard: no cache, and a barrier so a key still
        # held when the jump or recovery ends cannot auto-trigger a cycle.
        if inhibited:
            self._blocked_until_release = True
            return SideStepIntentDecision(
                drive_direction=0,
                cancel_current=False,
                finish_current_cycle=False,
                repeat_direction=0,
                blocked_until_release=True,
                reason="inhibited",
            )

        # 6-8. Idle and armed: held keys request, a same-poll tap requests once,
        # conflicting keys request nothing, and nothing is carried over.
        if conflicting_keys:
            return SideStepIntentDecision(
                drive_direction=0,
                cancel_current=False,
                finish_current_cycle=False,
                repeat_direction=0,
                blocked_until_release=False,
                reason="conflicting_keys",
            )
        request = held if held != 0 else press
        return SideStepIntentDecision(
            drive_direction=request,
            cancel_current=False,
            finish_current_cycle=False,
            repeat_direction=0,
            blocked_until_release=False,
            reason="start_requested" if request != 0 else "idle",
        )
