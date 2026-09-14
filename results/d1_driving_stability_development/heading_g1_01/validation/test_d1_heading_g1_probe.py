"""Boundary and cumulative-path checks for the fixed command-domain probe."""
import numpy as np
import pytest

from scripts.probe_d1_heading_g1 import (
    command_at_tick,
    command_source,
    endpoint_window,
    gates_and_metrics,
    prescribed_wrench,
)


def command(target=.25, yaw=None):
    return {'forward_target_mps': target, 'settle_ticks': 50, 'ramp_ticks': 50,
            'stop_tick': 400, 'yaw_pulse': yaw, 'height_m': .455}


@pytest.mark.parametrize('sign', (-1., 1.))
def test_ramp_and_release_are_same_exact_ticks_for_both_directions(sign):
    spec = command(sign*.25)
    for tick, expected in ((49, 0.), (50, 0.), (51, .005), (99, .245),
                           (100, .25), (399, .25), (400, 0.), (401, 0.)):
        assert command_at_tick(spec, tick).forward_velocity_mps == pytest.approx(sign*expected)


@pytest.mark.parametrize('rate', (-.6, .6))
def test_turn_pulse_and_physics_clock_rounding(rate):
    spec = command(0., {'start_tick': 200, 'end_tick_exclusive': 250,
                        'user_yaw_rate_rps': rate})
    source = command_source(spec)
    for tick in (199, 200, 249, 250):
        expected = rate if 200 <= tick < 250 else 0.
        assert source(tick*.01+1e-13).yaw_rate_rps == expected
    assert sum(command_at_tick(spec, tick).yaw_rate_rps*.01 for tick in range(800)) == pytest.approx(np.sign(rate)*.3)


def test_wrench_interval_excludes_end_tick():
    case = {'external_wrench': {'start_tick': 600, 'end_tick_exclusive': 620,
                                'force_xyz_n': [0., 0., 0.], 'torque_xyz_nm': [0., 0., -.5]}}
    assert prescribed_wrench(case, 599)[5] == 0.
    assert prescribed_wrench(case, 600)[5] == -.5
    assert prescribed_wrench(case, 619)[5] == -.5
    assert prescribed_wrench(case, 620)[5] == 0.


def test_endpoint_window_excludes_late_endpoint_and_refuses_missing_sample():
    rows = [{'endpoint_tick': tick} for tick in range(1, 801)]
    selected = endpoint_window(rows, 500, 800)
    assert len(selected) == 300
    assert selected[0]['endpoint_tick'] == 500 and selected[-1]['endpoint_tick'] == 799
    assert endpoint_window(rows[:798], 500, 800) is None


def test_stop_gate_uses_cumulative_path_even_when_net_displacement_is_zero():
    rows = []
    for endpoint in range(1, 801):
        rows.append({'endpoint_tick': endpoint, 'heading_task': {
            'actual_dt_s': .01, 'user_yaw_rate_error_after': 0.,
            'reference_heading_before': 0., 'reference_heading_after': 0.},
            'user_forward_error_mps': 0., 'metrics': {'height_error_m': 0.,
                'undesired_ground_contacts': 0, 'roll_error_rad': 0., 'pitch_error_rad': 0.},
            'heading_error_rad': 0., 'cross_track_after_m': 0., 'reward': 0.,
            'servo_reward': 0., 'reward_terms': {'heading_goal': 0.},
            'applied_action': [0.]*8, 'terminated': False, 'truncated': endpoint == 800,
            'terminal_reason': 'time_limit' if endpoint == 800 else None,
            'actual_roll_pitch_rad': [0., 0.], 'body_forward_mps': 0.})
    positions = np.zeros((801, 3))
    positions[501:700:2, 0] = .0004
    assert positions[700, 0]-positions[500, 0] == 0.
    case = {'max_transitions': 800, 'gate_groups': ['all_cases', 'stop_cases'], 'command': command()}
    thresholds = {'all_cases': {'nonwheel_contact_steps_max': 0, 'max_abs_actual_roll_pitch_rad': .2,
        'height_rmse_m_max': .015, 'heading_peak_rad_max': .1},
        'stop_cases': {'speed_window_start_tick': 500, 'speed_window_end_tick_exclusive': 800,
            'stationary_path_window_start_tick': 500, 'stationary_path_window_end_tick_exclusive': 700,
            'release_tick': 400, 'max_abs_body_vx_late_mps': .03,
            'cumulative_planar_path_after_first_second_m_max': .05,
            'heading_rmse_rad_max': .06, 'velocity_rmse_mps_max': .05}}
    metrics, gates = gates_and_metrics(case, rows, positions, thresholds)
    assert metrics['late_stop_cumulative_planar_path_m'] == pytest.approx(.08)
    assert gates['checks']['late_stop_planar_path'] is False
    metrics, gates = gates_and_metrics(case, rows[:650], positions[:651], thresholds)
    assert metrics['late_stop_cumulative_planar_path_m'] is None
    assert gates['checks']['late_stop_planar_path'] is False
