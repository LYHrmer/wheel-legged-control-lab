"""Pure tests for ``analyze_saved_side27`` helpers on tiny synthetic saved-like data.

No engine, model, simulator or project import; every test calls the real helper
functions in the module under test rather than re-deriving their arithmetic.
Run with: python -m unittest test_saved_speed27 -v
"""
from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    'analyze_saved_side27', Path(__file__).resolve().parent/'analyze_saved_side27.py')
m = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(m)


def yaw_quaternion(yaw):
    """Saved-style (w, x, y, z) quaternion of a pure yaw rotation."""
    return np.asarray([math.cos(yaw/2.), 0., 0., math.sin(yaw/2.)])


def rotate(points, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(points, dtype=float)@np.asarray([[c, s], [-s, c]])


class TestBodyFrame(unittest.TestCase):
    """Body-frame projection must not depend on the world heading of the record."""

    PATH = np.asarray([[1.2, -.4], [1.21, -.38], [1.18, -.35], [1.23, -.33], [1.25, -.36]])

    def test_rotation_invariance_and_direction_sign(self):
        yaw = .37
        base_lateral, base_longitudinal = m.body_frame_track(
            self.PATH, self.PATH[0], m.yaw_from_quaternion(yaw_quaternion(yaw)), 1)

        for angle in (-1.9, .0, .8, 2.7):
            rotated = rotate(self.PATH, angle)
            turned_yaw = m.yaw_from_quaternion(yaw_quaternion(yaw + angle))
            lateral, longitudinal = m.body_frame_track(rotated, rotated[0], turned_yaw, 1)
            np.testing.assert_allclose(lateral, base_lateral, atol=1e-12, rtol=0.)
            np.testing.assert_allclose(longitudinal, base_longitudinal, atol=1e-12, rtol=0.)

        # The commanded direction only flips the lateral sign, never the geometry.
        flipped_lateral, flipped_longitudinal = m.body_frame_track(
            self.PATH, self.PATH[0], m.yaw_from_quaternion(yaw_quaternion(yaw)), -1)
        np.testing.assert_allclose(flipped_lateral, -base_lateral, atol=0., rtol=0.)
        np.testing.assert_allclose(flipped_longitudinal, base_longitudinal, atol=0., rtol=0.)

        # Lateral/longitudinal are an orthonormal decomposition of the planar move.
        delta = self.PATH - self.PATH[0]
        np.testing.assert_allclose(base_lateral**2 + base_longitudinal**2,
                                   (delta**2).sum(axis=1), atol=1e-15, rtol=0.)
        with self.assertRaises(m.DataError):
            m.body_frame_track(self.PATH, self.PATH[0], yaw, 0)


class TestSpeedSeparation(unittest.TestCase):
    """Net displacement rate, path-length rate and peak sample rate are distinct."""

    def setUp(self):
        self.times = np.linspace(0., 1., 101)
        # Net 10 mm with a large oscillation on top: net speed must not absorb it.
        self.lateral = .01*self.times + .02*np.sin(4.*np.pi*self.times)
        self.longitudinal = .004*self.times

    def test_net_speed_separated_from_peak_and_path(self):
        result = m.net_and_path_speeds(self.times, self.lateral, self.longitudinal)
        duration = float(self.times[-1] - self.times[0])
        net = float(self.lateral[-1] - self.lateral[0])

        self.assertEqual(result['duration_s'], duration)
        self.assertEqual(result['net_lateral_m'], net)
        self.assertEqual(result['net_signed_lateral_speed_mps'], net/duration)
        self.assertAlmostEqual(result['net_longitudinal_drift_m'], .004, places=12)
        self.assertAlmostEqual(result['net_longitudinal_drift_speed_mps'], .004, places=12)

        # Oscillation inflates path and peak rates by more than an order of magnitude.
        self.assertGreater(result['peak_abs_lateral_rate_mps'],
                           10.*abs(result['net_signed_lateral_speed_mps']))
        self.assertGreater(result['path_length_m'], 5.*abs(net))
        self.assertGreater(result['mean_path_speed_mps'],
                           abs(result['net_signed_lateral_speed_mps']))
        self.assertEqual(result['mean_path_speed_mps'], result['path_length_m']/duration)
        self.assertAlmostEqual(result['path_to_net_ratio'], result['path_length_m']/abs(net),
                               places=12)

        # Full-cycle extrema are reported, not used to redefine the net value.
        self.assertGreater(result['max_lateral_m'], net)
        self.assertLess(result['min_lateral_m'], 0.)
        self.assertGreaterEqual(result['time_of_max_lateral_s'], 0.)
        self.assertLessEqual(result['time_of_max_lateral_s'], duration)

    def test_initial_reverse_travel_and_crossing(self):
        times = np.round(np.arange(10)*.01, 10)
        lateral = np.asarray([0., -.002, -.004, -.003, -.001, 0., .003, .006, .009, .012])
        result = m.reversal_metrics(times, lateral)

        self.assertEqual(result['sample_count'], 10)
        self.assertEqual(result['negative_sample_count'], 4)
        self.assertEqual(result['first_negative_sample_offset'], 1)
        self.assertAlmostEqual(result['first_negative_time_s'], .01, places=12)
        self.assertAlmostEqual(result['negative_time_s'], .04, places=12)
        self.assertAlmostEqual(result['negative_fraction_of_cycle'], .04/.09, places=12)
        self.assertAlmostEqual(result['initial_negative_excursion_m'], .004, places=12)
        self.assertTrue(result['recrossed_start_position'])
        self.assertAlmostEqual(result['time_to_recross_start_s'], .05, places=12)
        self.assertAlmostEqual(result['fraction_of_cycle_before_recrossing'], .05/.09, places=12)

        forward_only = m.reversal_metrics(times, np.arange(10)*.001)
        self.assertEqual(forward_only['negative_sample_count'], 0)
        self.assertFalse(forward_only['recrossed_start_position'])
        self.assertIsNone(forward_only['initial_negative_excursion_m'])
        self.assertEqual(forward_only['negative_time_s'], 0.)


class TestSavedDataRejection(unittest.TestCase):
    """Discontinuous, split or reset-crossing saved data must be refused, not patched."""

    def test_discontinuous_or_split_saved_data_rejected(self):
        self.assertAlmostEqual(m.uniform_dt([0., .01, .02, .03]), .01, places=12)
        for bad in ([0., .01, .03], [0., .01, .01], [0., .01, .005]):
            with self.assertRaises(m.DataError):
                m.uniform_dt(bad)
        with self.assertRaises(m.DataError):
            m.uniform_dt([0., .02, .04])            # uniform, but not the saved control dt
        self.assertAlmostEqual(m.uniform_dt([0., .02, .04], nominal=None), .02, places=12)

        # A tick whose end time is not the next tick's start time is a gap.
        with self.assertRaises(m.DataError):
            m.contiguous_tick_times([0., .01, .03], [.01, .02, .04])
        self.assertAlmostEqual(
            m.contiguous_tick_times([0., .01, .02], [.01, .02, .03]), .01, places=12)

        self.assertEqual(m.sole_contiguous_interval([False, True, True, False]), (1, 3))
        with self.assertRaises(m.DataError):
            m.sole_contiguous_interval([False, True, True, False, True])
        with self.assertRaises(m.DataError):
            m.sole_contiguous_interval([False, False])

        self.assertTrue(m.validate_state_chain([5, 6, 7], [6, 7, 8], 5, 8))
        with self.assertRaises(m.DataError):
            m.validate_state_chain([5, 7, 8], [6, 8, 9], 5, 9)   # one state skipped
        with self.assertRaises(m.DataError):
            m.validate_state_chain([5, 6, 7], [6, 7, 8], 4, 8)   # endpoints disagree

    def test_cross_reset_rejected_and_retention_window(self):
        segment = [0, 0, 0, 0, 1, 1]
        self.assertEqual(m.require_single_segment(segment, 0, 3), 0)
        with self.assertRaises(m.DataError):
            m.require_single_segment(segment, 2, 5)

        times = np.round(np.arange(8)*.01, 10)
        lateral = np.asarray([0., .01, .02, .018, .015, .019, .03, .04])
        window = m.handoff_retention(times, lateral, 2, observation_s=.03)
        self.assertEqual(window['observation_end_sample'], 5)
        self.assertAlmostEqual(window['retained_lateral_m'], .019, places=12)
        self.assertAlmostEqual(window['retained_fraction'], .019/.02, places=12)
        self.assertAlmostEqual(window['max_rollback_m'], .005, places=12)

        # A record whose clock restarts cannot supply the observation endpoint.
        reset_times = np.asarray([0., .01, .02, 0., .01, .02, .03, .04])
        with self.assertRaises(m.DataError):
            m.handoff_retention(reset_times, lateral, 2, observation_s=.03)

        # An equal timestamp *before* the cycle end must never be selected.
        duplicate_times = np.asarray([.05, .01, .02, .03, .04, .05, .06, .07])
        tail = m.handoff_retention(duplicate_times, lateral, 1, observation_s=.04)
        self.assertEqual(tail['observation_end_sample'], 5)


class TestNumericSafety(unittest.TestCase):
    """Non-finite and unparsable saved cells must fail before reaching the JSON report."""

    def test_nonfinite_and_unparsable_values_rejected(self):
        self.assertEqual(m.json_safe_number('x', 1.5), 1.5)
        for bad in (float('nan'), float('inf'), float('-inf')):
            with self.assertRaises(m.DataError):
                m.json_safe_number('x', bad)

        with self.assertRaises(m.DataError):
            m.ensure_finite('qpos', [0., float('nan')])
        with self.assertRaises(m.DataError):
            m.body_frame_track([[0., 0.], [float('nan'), .1]], [0., 0.], .3, 1)
        with self.assertRaises(m.DataError):
            m.yaw_series([[1., 0., 0., float('inf')]])

        np.testing.assert_allclose(m.parse_floats('col', ['1', '2.5', '-0.0']),
                                   [1., 2.5, 0.], atol=0., rtol=0.)
        for bad in (['0.0', ''], ['0.0', 'legacy'], ['0.0', None], ['nan']):
            with self.assertRaises(m.DataError):
                m.parse_floats('col', bad)


class TestPhaseLabelAccounting(unittest.TestCase):
    """Saved phase labels are summed exactly and validated against the interval duration."""

    BEFORE = [0., .01, .02, .03]
    AFTER = [.01, .02, .03, .04]
    PHASE_BEFORE = ['shift', 'shift', 'unload', 'lift']
    PHASE_AFTER = ['shift', 'unload', 'lift', 'lift']

    def test_durations_sum_and_segments(self):
        steps = [a - b for a, b in zip(self.AFTER, self.BEFORE)]
        before = m.label_durations(self.PHASE_BEFORE, steps, .04)
        self.assertTrue(before['sums_to_interval_duration'])
        self.assertAlmostEqual(before['durations_s']['shift'], .02, places=12)
        self.assertEqual(before['tick_counts'], {'shift': 2, 'unload': 1, 'lift': 1})

        # phase_after is a different saved label stream, so its totals differ by a tick.
        after = m.label_durations(self.PHASE_AFTER, steps, .04)
        self.assertEqual(after['tick_counts'], {'shift': 1, 'unload': 1, 'lift': 2})
        self.assertNotAlmostEqual(after['durations_s']['shift'],
                                  before['durations_s']['shift'], places=12)

        with self.assertRaises(m.DataError):
            m.label_durations(self.PHASE_BEFORE, steps, .05)
        with self.assertRaises(m.DataError):
            m.label_durations(self.PHASE_BEFORE, [.01, .01, float('nan'), .01], .04)
        with self.assertRaises(m.DataError):
            m.label_durations(self.PHASE_BEFORE, steps[:3], .03)

    def test_contiguous_phase_before_segments(self):
        segments = m.contiguous_label_segments(self.PHASE_BEFORE, self.BEFORE, self.AFTER,
                                               tick_offset=200)
        self.assertEqual([s['phase'] for s in segments], ['shift', 'unload', 'lift'])
        self.assertEqual([s['start_tick'] for s in segments], [200, 202, 203])
        self.assertEqual([s['end_tick'] for s in segments], [202, 203, 204])
        self.assertEqual([s['tick_count'] for s in segments], [2, 1, 1])
        self.assertAlmostEqual(segments[0]['duration_s'], .02, places=12)
        self.assertEqual([s['next_phase'] for s in segments], ['unload', 'lift', None])
        self.assertAlmostEqual(sum(s['duration_s'] for s in segments), .04, places=12)

        # Repeated non-adjacent labels stay separate segments.
        repeated = m.contiguous_label_segments(['shift', 'unload', 'shift'],
                                               [0., .01, .02], [.01, .02, .03])
        self.assertEqual([s['phase'] for s in repeated], ['shift', 'unload', 'shift'])
        with self.assertRaises(m.DataError):
            m.contiguous_label_segments(['shift'], [0., .01], [.01, .02])


if __name__ == '__main__':
    unittest.main()
