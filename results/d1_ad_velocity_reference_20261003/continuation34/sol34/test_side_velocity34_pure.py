"""Pure C34 fixtures. No plant, model, scratch, policy, or physics execution."""
import unittest
from types import SimpleNamespace

import numpy as np

from body_curve33 import curve33
from side_velocity34 import (compensated_wrench34, install_side34,
                             reference_velocity34)


def tick(elapsed, duration=0.8):
    return dict(source_phase='shift', elapsed_s=elapsed, duration_s=duration,
                body_span_m=[0.04, -0.02, 0.0])


class ReferenceForceFixture(unittest.TestCase):
    def test_body_tick_uses_the_same_s_curve_derivative(self):
        for elapsed in (0.01, 0.04, 0.2, 0.4, 0.7, 0.79):
            got, active = reference_velocity34(
                tick(elapsed), phase='shift', failure=None, cancelled=False)
            expected = np.array(tick(elapsed)['body_span_m'])*curve33(elapsed, 0.8)[1]
            self.assertTrue(active)
            np.testing.assert_allclose(got, expected, atol=1e-14, rtol=0)

    def test_zero_at_endpoint_transition_failure_and_cancel(self):
        cases = (
            (tick(0), 'shift', None, False),
            (tick(.8), 'shift', None, False),
            (tick(1.0), 'shift', None, False),
            (tick(.2), 'unload', None, False),
            (tick(.2), 'shift', 'support_shift_timeout', False),
            (tick(.2), 'shift', None, True),
            (None, 'recenter', None, False),
        )
        for source, phase, failure, cancelled in cases:
            with self.subTest(phase=phase, failure=failure, cancelled=cancelled):
                value, active = reference_velocity34(
                    source, phase=phase, failure=failure, cancelled=cancelled)
                self.assertFalse(active)
                np.testing.assert_array_equal(value, np.zeros(3))

    def test_only_xy_force_changes_and_beta_scales_exactly(self):
        base = np.array([1., -2., 3., 4., 5., 6.])
        results = {}
        for beta in (0., .5, 1.):
            old, velocity, delta, desired, active = compensated_wrench34(
                base, tick(.2), phase='shift', failure=None, cancelled=False,
                beta=beta, mass=10., body_kd=9.)
            self.assertTrue(active)
            np.testing.assert_array_equal(old, base)
            np.testing.assert_array_equal(delta[2:], np.zeros(4))
            np.testing.assert_allclose(delta[:2], 90.*beta*velocity[:2], atol=1e-14)
            np.testing.assert_allclose(desired, base+delta, atol=1e-14)
            results[beta] = delta
        np.testing.assert_array_equal(results[0.], np.zeros(6))
        np.testing.assert_allclose(results[.5], .5*results[1.], atol=1e-14)

    def test_beta_zero_and_inactive_keep_fast_wrench_bytes(self):
        base = np.array([-0., +0., 1., -2., 3., -4.])
        for beta, phase in ((0., 'shift'), (1., 'unload')):
            _, _, delta, desired, _ = compensated_wrench34(
                base, tick(.2), phase=phase, failure=None, cancelled=False,
                beta=beta, mass=10., body_kd=9.)
            self.assertEqual(desired.tobytes(), base.tobytes())
            np.testing.assert_array_equal(delta, np.zeros(6))


class FakeSide:
    def __init__(self):
        self.phase = 'idle'
        self.side_arm = 'teacher'
        self._body_alpha33 = 1.
        self.mass = 10.
        self.cfg = SimpleNamespace(body_kd=9.)
        self.failure = None
        self.fixed_cancelled = False
        self._skill30_cancel_requested = False
        self._body_reference_tick33 = None
        self.last_allocation = None
        self.started_distance = None

    def start(self, direction, distance_m=.03):
        self.started_distance = distance_m
        return direction == 1

    def _allocate(self, points, com, wrench, weights):
        self.last_allocation = {
            'original_wrench': np.asarray(wrench).copy(),
            'desired_wrench': np.asarray(wrench).copy()}
        return np.zeros((4, 3))


class AdapterFixture(unittest.TestCase):
    def test_start_bridge_and_final_allocation_evidence(self):
        side = install_side34(FakeSide(), beta=.5, configured_distance_m=.04)
        self.assertTrue(side.start(1, distance_m=.03))
        self.assertEqual(side.started_distance, .04)
        self.assertEqual(side._last_start_distance34['incoming_hybrid_distance_m'], .03)
        self.assertEqual(side._last_start_distance34['forwarded_fast_distance_m'], .04)
        with self.assertRaises(RuntimeError):
            side.start(1, distance_m=.04)
        side.phase = 'shift'
        side._body_reference_tick33 = tick(.2)
        base = np.arange(6., dtype=float)
        side._allocate(np.zeros((4, 3)), np.zeros(3), base, np.ones(4))
        evidence = side.last_allocation
        np.testing.assert_array_equal(evidence['original_wrench'], base)
        np.testing.assert_array_equal(evidence['c34_base_wrench'], base)
        np.testing.assert_allclose(evidence['desired_wrench'],
                                   evidence['c34_desired_wrench'])
        np.testing.assert_allclose(evidence['desired_wrench'],
                                   base+evidence['c34_wrench_delta_n'])
        self.assertTrue(evidence['c34_velocity_reference_active'])
        self.assertEqual(evidence['c34_beta'], .5)
        with self.assertRaises(RuntimeError):
            side.set_case34(beta=1., configured_distance_m=.03)

    def test_cancel_preserves_old_allocator_input(self):
        side = install_side34(FakeSide(), beta=1., configured_distance_m=.03)
        side.phase = 'recenter'
        side._body_reference_tick33 = dict(tick(.2), source_phase='recenter')
        side.fixed_cancelled = True
        base = np.arange(6., dtype=float)
        side._allocate(np.zeros((4, 3)), np.zeros(3), base, np.ones(4))
        np.testing.assert_array_equal(side.last_allocation['desired_wrench'], base)
        np.testing.assert_array_equal(side.last_allocation['c34_wrench_delta_n'],
                                      np.zeros(6))


if __name__ == '__main__':
    unittest.main()
