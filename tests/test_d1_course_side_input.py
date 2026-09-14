"""Observable keyboard packets and runner boundaries for committed side cycles."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.d1_course_controls import CourseKeyboardCommands
from scripts.d1_course_side_intent import CourseSideStepIntent
from scripts.run_d1_course_drive import DEFAULT_SIDE_INPUT_MODE, resolve_course_side_input


def controls():
    clock = [0.0]
    commands = CourseKeyboardCommands(clock=lambda: clock[0])
    commands.side_input_mode = "commit_cycle_v1"
    commands.reset(0.0, (0.0, 0.0))
    return commands, clock


def poll(commands, clock, text="", *, focused=True):
    clock[0] += 0.01
    commands.update_pressed(set(map(ord, text)), focused=focused)
    return commands(clock[0])


def drive(*, active=False, direction=0, failed=None, jump=False, recovery=False):
    state = SimpleNamespace(has_fallen=lambda: recovery, base_position=np.array([0., 0., .455]))
    return SimpleNamespace(active=active, side=SimpleNamespace(direction=direction, failure=failed),
                           teleop=SimpleNamespace(_state=state, _jump_requested=jump, _jump_step=None),
                           _blocked_until_release=False)


def resolve(commands, adapter, intent, mode="commit_cycle_v1"):
    return resolve_course_side_input(commands, adapter, intent, mode)


def test_same_poll_tap_is_consumed_once_and_blocks_drive_commands_for_that_packet():
    commands, clock = controls()
    commands.handle_key_event(ord("A"), 1)
    commands.handle_key_event(ord("A"), 0)
    command = poll(commands, clock, "WQT")
    assert command.forward_velocity_mps == command.yaw_rate_rps == 0
    assert command.clearance_m == .455
    assert commands.side_direction == commands.raw_side_direction == 0
    decision, packet = resolve(commands, drive(), CourseSideStepIntent())
    assert packet["press_direction"] == 1 and packet["keys_released"]
    assert decision.drive_direction == 1
    assert commands.consume_side_input()["press_direction"] == 0


def test_repeat_and_poll_do_not_duplicate_press_and_conflicting_keys_do_not_pick_a_side():
    commands, clock = controls()
    commands.handle_key_event(ord("A"), 1)
    poll(commands, clock, "A")
    assert commands.consume_side_input()["press_direction"] == 1
    for action in (2, 1):
        commands.handle_key_event(ord("A"), action)
        poll(commands, clock, "A")
        assert commands.consume_side_input()["press_direction"] == 0
    commands.handle_key_event(ord("D"), 1)
    poll(commands, clock, "AD")
    decision, packet = resolve(commands, drive(), CourseSideStepIntent())
    assert packet["held_direction"] == 0 and not packet["keys_released"]
    assert decision.drive_direction == 0 and decision.reason == "conflicting_keys"


def test_poll_only_input_has_a_single_press_edge_and_held_repeat_intent():
    commands, clock = controls()
    poll(commands, clock, "D")
    assert commands.consume_side_input()["press_direction"] == -1
    poll(commands, clock, "D")
    packet = commands.consume_side_input()
    assert packet["press_direction"] == 0 and packet["held_direction"] == -1


@pytest.mark.parametrize("cancel", ["x_tap", "focus", "timeout", "fallen"])
def test_cancellation_preserves_raw_input_and_cannot_restart_until_real_release(cancel):
    commands, clock = controls()
    intent = CourseSideStepIntent()
    poll(commands, clock, "A")
    resolve(commands, drive(), intent)
    commands.set_side_active(True)
    if cancel == "x_tap":
        commands.handle_key_event(ord("X"), 1)
        commands.handle_key_event(ord("X"), 0)
        poll(commands, clock, "A")
    elif cancel == "focus":
        poll(commands, clock, "A", focused=False)
    elif cancel == "timeout":
        clock[0] += 1.0
        commands(clock[0])
    else:
        commands.set_state(yaw_rad=0., position_xy=(0., 0.), fallen=True,
                           simulation_time_s=clock[0])
        poll(commands, clock, "A")
    decision, packet = resolve(commands, drive(active=True, direction=1), intent)
    assert packet["cancel_reason"] is not None and packet["held_direction"] == 1
    assert decision.cancel_current and decision.drive_direction == 0
    commands.set_state(yaw_rad=0., position_xy=(0., 0.), fallen=False,
                       simulation_time_s=clock[0])
    poll(commands, clock, "A")
    assert resolve(commands, drive(), intent)[0].blocked_until_release
    poll(commands, clock)
    assert not resolve(commands, drive(), intent)[0].blocked_until_release
    poll(commands, clock, "A")
    assert resolve(commands, drive(), intent)[0].drive_direction == 1


def test_reset_does_not_synthesize_release_from_a_previous_poll():
    commands, clock = controls()
    intent = CourseSideStepIntent()
    poll(commands, clock)
    commands.reset(0., (0., 0.))
    intent.reset()
    decision, packet = resolve(commands, drive(), intent)
    assert not packet["sampled_after_reset"] and not packet["keys_released"]
    assert decision.blocked_until_release
    poll(commands, clock, "A")
    assert resolve(commands, drive(), intent)[0].blocked_until_release
    poll(commands, clock)
    assert not resolve(commands, drive(), intent)[0].blocked_until_release


def test_jump_discards_tap_and_release_after_jump_cannot_replay_it():
    commands, clock = controls()
    intent = CourseSideStepIntent()
    commands.handle_key_event(ord("A"), 1)
    commands.handle_key_event(ord("A"), 0)
    poll(commands, clock)
    decision, packet = resolve(commands, drive(jump=True), intent)
    assert decision.drive_direction == 0 and packet["start_inhibited"]
    poll(commands, clock)
    assert resolve(commands, drive(), intent)[0].drive_direction == 0
    poll(commands, clock)
    assert resolve(commands, drive(), intent)[0].drive_direction == 0


def test_release_is_full_cycle_only_in_selected_mode_and_motion_waits_for_handoff():
    commands, clock = controls()
    commands.set_side_active(True)
    command = poll(commands, clock, "WQT")
    assert command.forward_velocity_mps == command.yaw_rate_rps == 0
    assert command.clearance_m == .455
    adapter = drive(active=True, direction=1)
    commit, _ = resolve(commands, adapter, CourseSideStepIntent())
    legacy, _ = resolve(commands, adapter, CourseSideStepIntent(), "release_abort")
    assert commit.drive_direction == 1 and commit.finish_current_cycle and not commit.cancel_current
    assert legacy.drive_direction == 0 and legacy.cancel_current


def test_reset_packet_is_never_a_release_even_for_a_direction_only_adapter():
    intent = CourseSideStepIntent()
    intent.reset()
    generic = SimpleNamespace(side_direction=0)
    first, packet = resolve_course_side_input(
        generic, drive(), intent, "commit_cycle_v1", reset_this_tick=True)
    assert not packet["keys_released"] and not packet["sampled_after_reset"]
    assert first.blocked_until_release
    later, _ = resolve_course_side_input(generic, drive(), intent, "commit_cycle_v1")
    assert not later.blocked_until_release and later.drive_direction == 0


def test_default_is_the_physically_validated_committed_mode():
    assert DEFAULT_SIDE_INPUT_MODE == "commit_cycle_v1"
