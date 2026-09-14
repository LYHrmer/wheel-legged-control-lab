"""Contract-level user input sequences; no GUI, controller or physics exists here."""
import dataclasses

import pytest

from scripts.d1_course_side_intent import CourseSideStepIntent


def resolve(intent, held=0, **kwargs):
    defaults = {'keys_released': held == 0, 'active': False}
    defaults.update(kwargs)
    return intent.resolve(held, **defaults)


def test_short_tap_requests_once_and_guard_rejection_never_queues_it():
    intent = CourseSideStepIntent()
    tap = resolve(intent, press_direction=1)
    assert tap.drive_direction == 1 and tap.reason == 'start_requested'
    assert not tap.cancel_current and not tap.blocked_until_release
    assert resolve(intent).drive_direction == 0  # physical start could have rejected the tap
    assert resolve(intent).drive_direction == 0


@pytest.mark.parametrize('direction', [-1, 1])
def test_release_finishes_the_active_cycle_without_requesting_another(direction):
    intent = CourseSideStepIntent()
    assert resolve(intent, direction).drive_direction == direction
    for _ in range(4):
        active = resolve(intent, direction, active=True, active_direction=direction)
        assert active.drive_direction == direction and active.repeat_direction == direction
        assert not active.cancel_current and not active.finish_current_cycle
    for _ in range(4):
        released = resolve(intent, active=True, active_direction=direction)
        assert released.drive_direction == direction and released.repeat_direction == 0
        assert not released.cancel_current and released.finish_current_cycle
    assert resolve(intent).drive_direction == 0


def test_reversal_only_changes_direction_at_actual_cycle_boundary():
    intent = CourseSideStepIntent()
    resolve(intent, 1)
    opposite = resolve(intent, -1, active=True, active_direction=1, press_direction=-1)
    assert opposite.drive_direction == 1 and opposite.repeat_direction == -1
    assert opposite.finish_current_cycle and not opposite.cancel_current
    assert opposite.reason == 'reverse_after_cycle'
    assert resolve(intent, -1).drive_direction == -1
    assert resolve(intent, active=True, active_direction=-1).drive_direction == -1
    assert resolve(intent).drive_direction == 0


def test_busy_taps_do_not_survive_after_release_and_cycle_completion():
    intent = CourseSideStepIntent()
    for _ in range(4):
        result = resolve(intent, active=True, active_direction=1, press_direction=-1)
        assert result.drive_direction == 1 and result.repeat_direction == 0
        assert result.finish_current_cycle and not result.cancel_current
    assert resolve(intent).drive_direction == 0


@pytest.mark.parametrize('reason', ['x', 'focus_lost', 'input_timeout', 'fallen', 'recovery'])
def test_explicit_cancel_requires_a_later_real_release_before_restart(reason):
    intent = CourseSideStepIntent()
    result = resolve(intent, active=True, active_direction=1, cancel_reason=reason)
    assert result.drive_direction == 0 and result.cancel_current and result.blocked_until_release
    assert result.reason == 'cancel_requested'
    blocked = resolve(intent, 1, active=True, active_direction=1, press_direction=1)
    assert blocked.drive_direction == 0 and blocked.cancel_current
    assert blocked.repeat_direction == 0 and blocked.blocked_until_release
    conflict = resolve(intent, keys_released=False, press_direction=-1)
    assert conflict.drive_direction == 0 and conflict.blocked_until_release
    rearm = resolve(intent, press_direction=1)
    assert rearm.drive_direction == 0 and not rearm.blocked_until_release
    assert resolve(intent, press_direction=1).drive_direction == 1


def test_conflicting_keys_do_not_cancel_a_committed_cycle_or_select_a_new_direction():
    intent = CourseSideStepIntent()
    assert resolve(intent, keys_released=False, press_direction=1).drive_direction == 0
    conflict = resolve(intent, keys_released=False, active=True, active_direction=1,
                       press_direction=-1)
    assert conflict.drive_direction == 1 and not conflict.cancel_current
    assert conflict.finish_current_cycle and conflict.repeat_direction == 0


def test_sticky_controller_failure_unlocks_on_release_and_rearms_on_successful_start():
    intent = CourseSideStepIntent()
    failure = resolve(intent, 1, active=True, active_direction=1, controller_failed=True)
    assert failure.drive_direction == 0 and failure.cancel_current
    assert failure.blocked_until_release and failure.reason == 'controller_failed'
    rearm = resolve(intent, controller_failed=True)
    assert rearm.drive_direction == 0 and not rearm.blocked_until_release
    assert resolve(intent, 1, controller_failed=True).drive_direction == 1
    resolve(intent, 1, active=True, active_direction=1, controller_failed=False)
    new_failure = resolve(intent, 1, active=True, active_direction=1, controller_failed=True)
    assert new_failure.cancel_current and new_failure.blocked_until_release


def test_inhibited_taps_are_discarded_and_held_keys_cannot_start_after_the_mode_ends():
    intent = CourseSideStepIntent()
    assert resolve(intent, press_direction=1, start_inhibited=True).blocked_until_release
    assert resolve(intent, start_inhibited=True).blocked_until_release
    assert resolve(intent, 1).drive_direction == 0
    cleared = resolve(intent, press_direction=1)
    assert not cleared.blocked_until_release and cleared.drive_direction == 0
    assert resolve(intent, press_direction=-1).drive_direction == -1


def test_mode_inhibition_alone_does_not_abort_an_active_cycle():
    intent = CourseSideStepIntent()
    active = resolve(intent, 1, active=True, active_direction=1, start_inhibited=True)
    assert active.drive_direction == 1 and not active.cancel_current
    assert active.repeat_direction == 0 and active.finish_current_cycle
    after = resolve(intent, 1, start_inhibited=True)
    assert after.drive_direction == 0 and after.blocked_until_release


def test_reset_needs_real_release_and_does_not_consume_a_tap_in_the_rearming_packet():
    intent = CourseSideStepIntent()
    intent.reset()
    assert resolve(intent, 1, press_direction=1).blocked_until_release
    assert resolve(intent, keys_released=False).blocked_until_release
    released = resolve(intent, press_direction=-1)
    assert released.drive_direction == 0 and not released.blocked_until_release
    assert resolve(intent, press_direction=-1).drive_direction == -1


@pytest.mark.parametrize('changes', [
    {'held': True}, {'held': 1.0}, {'held': 2},
    {'press_direction': False}, {'press_direction': -2},
    {'active_direction': True}, {'active_direction': 1.0},
    {'active': True, 'active_direction': 0},
    {'held': 1, 'keys_released': True}, {'keys_released': 1},
    {'active': 0}, {'controller_failed': 1}, {'start_inhibited': 0},
    {'cancel_reason': ''}, {'cancel_reason': False}, {'cancel_reason': 3},
])
def test_invalid_input_is_atomic_even_when_it_also_contains_a_new_failure(changes):
    intent = CourseSideStepIntent()
    resolve(intent, cancel_reason='x')
    before = vars(intent).copy()
    args = {'controller_failed': True, **changes}
    with pytest.raises((TypeError, ValueError)):
        resolve(intent, **args)
    assert vars(intent) == before


def test_return_record_is_frozen():
    decision = resolve(CourseSideStepIntent())
    assert dataclasses.is_dataclass(decision)
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.drive_direction = 1


def test_nonempty_cancel_reason_is_not_silently_restricted_beyond_the_contract():
    decision = resolve(CourseSideStepIntent(), cancel_reason=' ')
    assert decision.drive_direction == 0 and decision.blocked_until_release
    assert decision.reason == 'cancel_requested'
