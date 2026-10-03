"""Pure saved-arithmetic fixtures; root alone runs checks."""
from __future__ import annotations

import math
import numpy as np
import pytest

from contact_math35 import (native_resultant35, pair_support35, all_four_support35,
                            static_contact_margin35)
from task_math35 import commanded35, continuous_metrics35, hermite_velocity35
from reference_math35 import landing35, two_swing35, verify_foot_reference35
from stop_cost_math35 import stop_metrics35, torque_cost35


def _contact(wheel: int, x: float, load: float, couple: float = 0.) -> dict:
    frame = [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]
    return {'geom1': 7, 'geom2': 10+wheel, 'efc_address': 0,
            'frame_world': frame, 'contact_force_local': [load, 0., 0., couple, 0., 0.],
            'position_world_m': [x, 0., 0.], 'robot_wheel_index': wheel,
            'force_on_robot_world_n': [0., 0., load],
            'normal_load_on_robot_n': load}


def test_pair_support_uses_two_wheel_sum_and_local_contact_moment():
    native = {'contacts': [_contact(0, 1., 4., 1.), _contact(3, -1., 2.)]}
    result = native_resultant35(native, {7}, [0., 0., 0.])
    assert np.allclose(result['world_force_sum_n'], [0., 0., 6.])
    # r×F contributes -4 and +2 about Y; the first local couple contributes Z=1.
    assert np.allclose(result['world_moment_about_com_sum_nm'], [0., -2., 1.])
    assert result['positive_wheel_support_mask'] == (True, False, False, True)
    assert pair_support35(result, (0, 3), 1.)['passed'] is True
    assert pair_support35(result, (1, 2), 1.)['passed'] is False
    assert all_four_support35(result, 1.)['passed'] is False
    native['contacts'][1]['normal_load_on_robot_n'] = 3.
    with pytest.raises(AssertionError):
        native_resultant35(native, {7}, [0., 0., 0.])


def test_all_four_stop_margin_does_not_require_arbitrary_pair_half_mg():
    contacts = [_contact(i, x, 2.) for i, x in enumerate((-.2, .2, -.2, .2))]
    for contact, y in zip(contacts, (.1, .1, -.1, -.1)):
        contact['position_world_m'][1] = y
    result = native_resultant35({'contacts': contacts}, {7}, [0., 0., 0.])
    assert all_four_support35(result, 1.)['passed'] is True
    assert pair_support35(result, (0, 3), 1.)['passed'] is False
    margin = static_contact_margin35(result, [0., 0., 0.])
    assert math.isclose(margin['static_margin_m'], .1, abs_tol=1e-12)


def test_commanded_symmetric_ramp_integrates_exact_six_second_goal():
    for direction in (-1, 1):
        assert commanded35(0., direction, 600) == (0., 0., 0.)
        for t in (.25, 6., 6.25):
            x, v, a = commanded35(t, direction, 600)
            assert math.isclose(x, direction*.025*(t-.125 if t <= 6. else 6.), abs_tol=1e-12)
            assert math.isclose(a, 0., abs_tol=1e-12)
            assert math.isclose(v, direction*(.025 if t <= 6. else 0.), abs_tol=1e-12)
        assert math.isclose(commanded35(6.25, direction, 600)[0],
                            direction*.150, abs_tol=1e-12)


def test_mid_ramp_cancel_preserves_velocity_acceleration_and_jerk():
    _, v, a, j = hermite_velocity35(.11, .25, 0., 0., 0., .025)
    x0, v0, a0, j0 = hermite_velocity35(0., .25, v, a, j, 0.)
    _, vend, aend, jend = hermite_velocity35(.25, .25, v, a, j, 0.)
    assert x0 == 0.
    assert np.allclose([v0, a0, j0], [v, a, j], atol=1e-12, rtol=0)
    assert np.allclose([vend, aend, jend], [0., 0., 0.], atol=1e-12, rtol=0)


def test_actual_steady_window_uses_origin_pre_post_states():
    for direction in (-1, 1):
        q = np.zeros((626, 23))
        q[:, 3] = 1.
        q[:, 1] = direction*np.arange(626)*.025*.01
        metrics = continuous_metrics35(q, direction)
        assert math.isclose(metrics['mean_signed_origin_vy_mps'], .025, abs_tol=1e-12)
        assert math.isclose(metrics['rms_steady_tracking_error_mps'], 0., abs_tol=1e-12)
        assert metrics['fraction_steady_signed_vy_ge_010'] == 1.
        assert metrics['steady_control_interval_half_open'] == [100, 500]
        assert math.isclose(metrics['commanded_complete_goal_m'], direction*.150, abs_tol=1e-12)


def test_two_swing_uses_common_duration_and_radial_projection():
    landing = landing35([0., 0.], [0., 0.], [.1, 0.], [0., 0.], [0., 0.])
    assert landing['saturated'] is True
    assert np.allclose(landing['projected_target_xy_m'], [.06, 0.])
    start = [[0., 0.]]*4
    target = [[.04, 0.], [0., 0.], [0., 0.], [0., .02]]
    pair = two_swing35((0, 3), .2, start, target)
    assert math.isclose(pair['duration_s'], 1.875*.04/.16, abs_tol=1e-12)
    assert np.allclose(pair['swing_by_leg'][0]['position_xy_m'], [.04*pair['swing_by_leg'][3]['position_xy_m'][1]/.02, 0.])


def test_stop_tail_and_null_cost_for_cancel():
    q = np.zeros((261, 23))
    q[:, 3] = 1.
    v = np.zeros((261, 22))
    margins = np.full(260, .01)
    loaded = np.full(260, True)
    pending = np.full(260, False)
    proof = stop_metrics35(q, v, stop_request_control=20,
                           handoff_control=260,
                           support_margin_by_control=margins,
                           all_four_loaded_by_control=loaded,
                           pending_swing_by_control=pending)
    assert proof['passed'] is True
    assert torque_cost35(full_scene_mean=.04, full_scene_integral_s=.02,
                         kind='cancel', commanded_distance_m=None,
                         c33_same_direction_integral_s=None)['distance_efficiency_gate'] is None
    v[-1, 0] = .02
    assert stop_metrics35(q, v, stop_request_control=20,
                          handoff_control=260, support_margin_by_control=margins,
                          all_four_loaded_by_control=loaded,
                          pending_swing_by_control=pending)['passed'] is False


def test_vertical_pair_reference_and_fixed_stance_anchors():
    anchors = np.array([[.2,.1,.1],[.2,-.1,.1],[-.2,.1,.1],[-.2,-.1,.1]])
    target = anchors.copy()
    target[[0,3], 0] += .02
    foot = anchors.copy()
    speed = np.zeros((4,3))
    foot[[0,3], 2] += .035*.5
    speed[[0,3], 2] = .035*1.875/.12
    ref = dict(phase='lift', pair=[0,3], feet_world_m=foot.tolist(),
               feet_velocity_world_mps=speed.tolist(),
               horizontal_origin_world_m=anchors.tolist(),
               horizontal_target_world_m=target.tolist(), phase_elapsed_s=.06)
    sensed = {'foot_world_m': anchors.tolist()}
    assert verify_foot_reference35(ref, None, anchors, sensed, None) is None
    lower = dict(ref, phase='lower')
    lowered = target.copy()
    lowered[[0,3], 2] += .035-.038*.5
    down = np.zeros((4,3))
    down[[0,3], 2] = -.038*1.875/.12
    lower['feet_world_m'],lower['feet_velocity_world_mps'] = lowered.tolist(),down.tolist()
    assert verify_foot_reference35(lower, ref, anchors, sensed, None) is None
    corrupt = dict(lower, feet_velocity_world_mps=np.asarray(down).copy().tolist())
    corrupt['feet_velocity_world_mps'][0][2] = 0.
    with pytest.raises(AssertionError):
        verify_foot_reference35(corrupt, ref, anchors, sensed, None)
    shifted = anchors.copy()
    shifted[1,0] += .001
    with pytest.raises(AssertionError):
        verify_foot_reference35(ref, None, shifted, sensed, None)
