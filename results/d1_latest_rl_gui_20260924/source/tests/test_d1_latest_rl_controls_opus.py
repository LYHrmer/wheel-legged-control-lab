"""Bounded pure tests for the latest straight-driving RL keyboard intent.

Only the pure controls module is loaded; no engine, model or physics import.
Event orders below mirror the real viewer in scripts/d1_keyboard_viewer.py:
  * every key event calls handle_key_event(int(key), int(action))   (line 85)
  * an Escape PRESS additionally calls update_pressed({256})        (line 90)
  * focus loss calls update_pressed(set(), focused=False)           (line 96)
  * each control tick polls states via update_pressed(pressed, focused=...)
    with the candidate set taken from commands.key_codes            (line 117)
Expected speeds are written as independent literals, not read from the module.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

W, ONE, TWO, R, X, SPACE, ESC = ord("W"), ord("1"), ord("2"), ord("R"), ord("X"), 32, 256
RELEASE, PRESS, REPEAT = 0, 1, 2
SLOW_MPS, FAST_MPS = 0.20, 0.25

_SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "d1_latest_rl_controls.py"
_SPEC = importlib.util.spec_from_file_location("d1_latest_rl_controls_under_test", _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
LatestRLCommands = _MODULE.LatestRLCommands


def test_hold_w_and_gear_switching_use_the_two_fixed_speeds():
    commands = LatestRLCommands()
    # Nothing is requested before the first focused poll, even with W down.
    commands.handle_key_event(W, PRESS)
    assert commands.raw_forward_mps == pytest.approx(0.0)

    commands.update_pressed({W})
    assert commands.gear == 1
    assert commands.selected_speed_mps == pytest.approx(SLOW_MPS)
    assert commands.raw_forward_mps == pytest.approx(SLOW_MPS)

    commands.handle_key_event(TWO, PRESS)
    commands.update_pressed({W, TWO})
    assert commands.gear == 2
    assert commands.raw_forward_mps == pytest.approx(FAST_MPS)

    commands.handle_key_event(TWO, RELEASE)
    commands.handle_key_event(ONE, PRESS)
    commands.update_pressed({W, ONE})
    assert commands.raw_forward_mps == pytest.approx(SLOW_MPS)

    # Releasing W stops the request while the selected gear survives.
    commands.handle_key_event(W, RELEASE)
    commands.update_pressed({ONE})
    assert commands.raw_forward_mps == pytest.approx(0.0)
    assert commands.selected_speed_mps == pytest.approx(SLOW_MPS)
    assert commands.stopped is False


def test_callback_then_poll_does_not_double_trigger_gear_or_reset():
    commands = LatestRLCommands()
    commands.handle_key_event(TWO, PRESS)
    commands.update_pressed({W, TWO})
    assert commands.gear == 2
    assert commands.raw_forward_mps == pytest.approx(FAST_MPS)
    assert commands.consume_reset_request() is False

    # A gear key still held across later ticks must not re-apply or flip a gear.
    commands.update_pressed({W, TWO})
    commands.update_pressed({W, ONE})
    assert commands.gear == 2
    assert commands.selected_speed_mps == pytest.approx(FAST_MPS)

    commands.handle_key_event(R, PRESS)
    commands.update_pressed({R})
    assert commands.consume_reset_request() is True
    assert commands.consume_reset_request() is False

    # Polling a still-held R never enqueues a second reset.
    commands.update_pressed({R})
    commands.update_pressed({R})
    assert commands.consume_reset_request() is False


def test_key_repeat_never_enqueues_extra_reset():
    commands = LatestRLCommands()
    commands.handle_key_event(R, PRESS)
    for _ in range(4):
        commands.handle_key_event(R, REPEAT)
    assert commands.consume_reset_request() is True
    assert commands.consume_reset_request() is False

    for _ in range(4):
        commands.handle_key_event(R, REPEAT)
    assert commands.consume_reset_request() is False

    # A real release re-arms the edge; a repeat alone never does.
    commands.handle_key_event(R, RELEASE)
    commands.handle_key_event(R, PRESS)
    assert commands.consume_reset_request() is True

    commands.handle_key_event(TWO, PRESS)
    commands.handle_key_event(TWO, REPEAT)
    commands.handle_key_event(ONE, REPEAT)
    assert commands.gear == 2


def test_focus_loss_clears_motion_and_requires_fresh_w_release():
    commands = LatestRLCommands()
    commands.handle_key_event(TWO, PRESS)
    commands.handle_key_event(W, PRESS)
    commands.update_pressed({W, TWO})
    assert commands.raw_forward_mps == pytest.approx(FAST_MPS)

    commands.handle_key_event(R, PRESS)
    commands.update_pressed(set(), focused=False)
    state = commands.snapshot()
    assert state["focused"] is False
    assert state["held_forward"] is False
    assert state["blocked_until_w_release"] is True
    assert commands.raw_forward_mps == pytest.approx(0.0)
    # Focus loss drops the queued reset with the rest of the stale intent.
    assert commands.consume_reset_request() is False

    # Regaining focus with W physically still down must not resume driving.
    commands.update_pressed({W, TWO})
    assert commands.raw_forward_mps == pytest.approx(0.0)
    commands.update_pressed({TWO})
    assert commands.snapshot()["blocked_until_w_release"] is False
    assert commands.raw_forward_mps == pytest.approx(0.0)

    # The gear survives the focus cycle; a fresh hold drives again.
    commands.update_pressed({W, TWO})
    assert commands.raw_forward_mps == pytest.approx(FAST_MPS)


def test_stop_request_blocks_held_w_until_a_poll_observes_release():
    commands = LatestRLCommands()
    for stop_key in (SPACE, X):
        commands.handle_key_event(TWO, PRESS)
        commands.handle_key_event(W, PRESS)
        commands.update_pressed({W, TWO})
        assert commands.raw_forward_mps == pytest.approx(FAST_MPS)

        commands.handle_key_event(stop_key, PRESS)
        assert commands.raw_forward_mps == pytest.approx(0.0)
        commands.update_pressed({W, TWO, stop_key})
        assert commands.raw_forward_mps == pytest.approx(0.0)

        # Releasing only the stop key leaves the held W cancelled.
        commands.handle_key_event(stop_key, RELEASE)
        commands.update_pressed({W, TWO})
        assert commands.raw_forward_mps == pytest.approx(0.0)

        # The W release edge is recognised by the poll, not by the callback:
        # without an intervening poll showing W up the hold stays cancelled.
        commands.handle_key_event(W, RELEASE)
        commands.update_pressed({W, TWO})
        assert commands.raw_forward_mps == pytest.approx(0.0)

        commands.update_pressed({TWO})
        commands.handle_key_event(W, PRESS)
        commands.update_pressed({W, TWO})
        assert commands.raw_forward_mps == pytest.approx(FAST_MPS)
        assert commands.stopped is False

        commands.handle_key_event(W, RELEASE)
        commands.handle_key_event(TWO, RELEASE)
        commands.update_pressed(set())


def test_reset_edges_and_reset_for_new_segment_clear_speed_and_hold():
    commands = LatestRLCommands()
    commands.handle_key_event(TWO, PRESS)
    commands.handle_key_event(W, PRESS)
    commands.update_pressed({W, TWO})
    assert commands.raw_forward_mps == pytest.approx(FAST_MPS)

    commands.handle_key_event(R, PRESS)
    assert commands.snapshot()["reset_pending"] is True
    assert commands.consume_reset_request() is True
    commands.handle_key_event(R, RELEASE)
    commands.handle_key_event(R, PRESS)

    commands.reset_for_new_segment()
    # A pending request is owned by the runner's reset, not replayed after it.
    assert commands.consume_reset_request() is False
    assert commands.gear == 1
    assert commands.selected_speed_mps == pytest.approx(SLOW_MPS)
    assert commands.raw_forward_mps == pytest.approx(0.0)

    # W held through the reset stays blocked until an actual release is polled.
    commands.update_pressed({W})
    assert commands.raw_forward_mps == pytest.approx(0.0)
    commands.update_pressed(set())
    commands.handle_key_event(W, PRESS)
    commands.update_pressed({W})
    assert commands.raw_forward_mps == pytest.approx(SLOW_MPS)


def test_escape_is_sticky_across_empty_poll_focus_cycle_and_segment_reset():
    commands = LatestRLCommands()
    commands.handle_key_event(W, PRESS)
    commands.update_pressed({W})
    assert commands.raw_forward_mps == pytest.approx(SLOW_MPS)

    # Viewer Escape path: callback edge plus the explicit polled Escape state.
    commands.handle_key_event(ESC, PRESS)
    commands.update_pressed({ESC})
    assert commands.stopped is True
    assert commands.raw_forward_mps == pytest.approx(0.0)

    commands.update_pressed(set())
    assert commands.stopped is True

    commands.handle_key_event(ESC, RELEASE)
    commands.handle_key_event(W, RELEASE)
    commands.handle_key_event(W, PRESS)
    commands.update_pressed({W})
    assert commands.stopped is True
    assert commands.raw_forward_mps == pytest.approx(0.0)

    commands.update_pressed(set(), focused=False)
    commands.update_pressed(set())
    commands.reset_for_new_segment()
    commands.update_pressed({W})
    state = commands.snapshot()
    assert state["stopped"] is True
    assert state["raw_forward_mps"] == pytest.approx(0.0)


def test_legacy_keys_cannot_steer_and_malformed_inputs_are_rejected():
    commands = LatestRLCommands()
    commands.handle_key_event(W, PRESS)
    commands.update_pressed({W})
    assert commands.raw_forward_mps == pytest.approx(SLOW_MPS)

    # Reverse / turn / body-height / jump / policy-digit keys are not owned here.
    for legacy in (ord("S"), ord("A"), ord("D"), ord("F"), ord("J"),
                   ord("0"), ord("3"), 265, 340):
        commands.handle_key_event(legacy, PRESS)
        commands.handle_key_event(legacy, REPEAT)
        commands.handle_key_event(legacy, RELEASE)
    state = commands.snapshot()
    assert state["gear"] == 1
    assert state["stopped"] is False
    assert state["blocked_until_w_release"] is False
    assert state["reset_pending"] is False
    assert commands.raw_forward_mps == pytest.approx(SLOW_MPS)

    # A polled set may only contain keys this controller declares.
    with pytest.raises(ValueError):
        commands.update_pressed({ord("S")})
    with pytest.raises(ValueError):
        commands.update_pressed({W, 265})
    with pytest.raises(ValueError):
        commands.handle_key_event(W, 3)

    with pytest.raises(TypeError):
        commands.update_pressed(frozenset({W}))
    with pytest.raises(TypeError):
        commands.update_pressed([W])
    with pytest.raises(TypeError):
        commands.update_pressed({float(W)})
    with pytest.raises(TypeError):
        commands.update_pressed({W}, focused=1)
    with pytest.raises(TypeError):
        commands.handle_key_event(float(W), PRESS)
    with pytest.raises(TypeError):
        commands.handle_key_event("W", PRESS)
    with pytest.raises(TypeError):
        commands.handle_key_event(W, True)

    # Rejected input must leave the accepted driving intent untouched.
    assert commands.gear == 1
    assert commands.raw_forward_mps == pytest.approx(SLOW_MPS)
