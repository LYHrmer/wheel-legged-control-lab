"""Causal reference timing and heading feedback used by real driving commands."""
import math

import pytest

from scripts.d1_heading_reference import HeadingReference, heading_feedback


def test_reference_integrates_the_executed_interval_not_the_new_command():
    reference = HeadingReference()
    reference.reset(0.2, 10.0)
    reference.advance(0.6, 10.0)
    state = reference.advance(-0.4, 10.5)
    assert state.heading_rad == pytest.approx(0.5)
    state = reference.advance(0.0, 11.0)
    assert state.heading_rad == pytest.approx(0.3)
    assert reference.advance(0.0, 13.0).heading_rad == pytest.approx(0.3)


def test_preview_is_idempotent_and_conflicting_tick_is_rejected_atomically():
    reference = HeadingReference()
    reference.reset(0.0)
    reference.advance(0.6, 0.0)
    state = reference.advance(0.0, 0.5)
    assert reference.advance(0.0, 0.5) == state
    for rate, time in ((0.5, 0.5), (0.0, 0.49), (math.nan, 0.6), (1.01, 0.6)):
        with pytest.raises((ValueError, TypeError)):
            reference.advance(rate, time)
        assert reference.state == state
    assert reference.advance(0.0, 1.0).heading_rad == pytest.approx(0.3)


def test_reset_discards_old_motion_and_delayed_first_input_does_not_rewrite_history():
    reference = HeadingReference()
    with pytest.raises(RuntimeError):
        reference.advance(0.0, 0.0)
    with pytest.raises(RuntimeError):
        _ = reference.state
    reference.reset(0.0)
    reference.advance(1.0, 0.0)
    reference.advance(1.0, 1.0)
    reference.reset(-0.5, 0.0)
    assert reference.advance(0.4, 2.0).heading_rad == pytest.approx(-0.5)
    assert reference.advance(0.0, 3.0).heading_rad == pytest.approx(-0.1)


def test_wrap_boundary_and_rotated_world_have_the_same_control():
    angle = math.radians
    expected = heading_feedback(angle(-179), angle(179), 0.1)
    assert expected.heading_error_rad == pytest.approx(angle(2))
    assert expected.servo_yaw_rate_rps > 0.0
    for shift in (-4.0, 0.7, 4.0):
        actual = heading_feedback(angle(-179)+shift, angle(179)+shift, 0.1)
        assert actual.servo_yaw_rate_rps == pytest.approx(expected.servo_yaw_rate_rps)


def test_drift_is_corrected_without_moving_the_held_target():
    reference = HeadingReference()
    reference.reset(0.0)
    reference.advance(0.0, 0.0)
    reference.advance(0.0, 30.0)
    correction = heading_feedback(reference.state.heading_rad, 0.5, 0.03)
    assert correction.servo_yaw_rate_rps == -1.0
    assert correction.saturated
    assert reference.state.heading_rad == 0.0


def test_feedforward_keeps_an_accurately_followed_turn_and_gui_zero_rate_behavior():
    assert heading_feedback(0.4, 0.4, 0.2, 0.2).servo_yaw_rate_rps == pytest.approx(0.2)
    feedback = heading_feedback(0.3, 0.1, 0.15)
    assert feedback.servo_yaw_rate_rps == pytest.approx(2.0*0.2-0.4*0.15)
    assert not feedback.saturated


@pytest.mark.parametrize('invalid', [True, False, math.nan, math.inf, -math.inf, '0.1'])
def test_nonfinite_or_nonnumeric_input_cannot_enter_feedback(invalid):
    with pytest.raises((ValueError, TypeError)):
        heading_feedback(invalid, 0.0, 0.0)
    reference = HeadingReference()
    reference.reset(0.0)
    state = reference.state
    with pytest.raises((ValueError, TypeError)):
        reference.reset(invalid)
    assert reference.state == state


def test_invalid_gains_and_reference_overflow_do_not_change_committed_state():
    for kwargs in ({'kp': -1.0}, {'kd': -1.0}, {'limit_rps': 0.0}):
        with pytest.raises(ValueError):
            heading_feedback(0.0, 0.0, 0.0, **kwargs)
    with pytest.raises(ValueError):
        heading_feedback(0.0, 0.0, 1e308, kd=1e308)
