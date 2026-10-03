"""Pure regressions for full-scene origins and the independent C34 force tick."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

C = Path(__file__).resolve().parents[1]
W = C.parent
for path in ('continuation26/root26', 'continuation24', 'continuation27/read27',
             'continuation30/read30', 'continuation30/pair_read30',
             'continuation31/read31', 'continuation33/read33'):
    sys.path.insert(0, str(W / path))

from force_read34 import c34_velocity34
from reward_read34 import actual_distance_components34
from reference_read33 import integrated_curve33
from reader26 import FORBIDDEN, NoExecution


class SavedReaderPure34(unittest.TestCase):
    def test_full_scene_can_begin_before_side_origin(self):
        q = np.zeros((301, 23))
        q[:, 3] = 1.
        q[:, :2] = [2., 3.]
        q[201:, 1] += .04
        result = actual_distance_components34(
            {'qpos': q}, 0, 300, 200, 1, .04)
        self.assertAlmostEqual(result['progress_potential_difference'], 1.)
        self.assertAlmostEqual(result['backtrack_normalized'], 0.)
        self.assertAlmostEqual(result['lateral_goal_error_square_integral_m2_s'],
                               .01 * 200 * .04**2)
        self.assertAlmostEqual(result['lateral_goal_error_normalized_square_integral_s'], 2.)

    def test_heading_and_rightward_goal_change_real_distance_error(self):
        q = np.zeros((3, 23))
        q[:, 3] = math.cos(math.pi / 4)
        q[:, 6] = math.sin(math.pi / 4)
        q[:, :2] = [1., 2.]
        q[-1, 0] += .04
        result = actual_distance_components34(
            {'qpos': q}, 0, 2, 1, -1, .04)
        self.assertAlmostEqual(result['progress_potential_difference'], 1.)
        self.assertAlmostEqual(result['longitudinal_normalized_square_integral_s'], 0.)
        self.assertAlmostEqual(result['lateral_goal_error_square_integral_m2_s'],
                               .01 * .04**2)

    def test_velocity_uses_current_tick_and_masks_phase_cancel_end(self):
        before = dict(phase='shift', body_to=[.1, -.2, 0.],
            body_from=[0., 0., 0.], phase_time_s=.49, shift_time_s=1.)
        after = dict(phase='shift', failure=None)
        actual, active = c34_velocity34(before, after, cancelled=False)
        self.assertTrue(active)
        expected = np.array([.1, -.2, 0.]) * integrated_curve33(.50, 1.)[1]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        for b, a, cancel in ((before, after, True),
                (before, dict(phase='unload', failure=None), False),
                (before, dict(phase='shift', failure='cancelled'), False),
                (dict(before, phase_time_s=.99), after, False)):
            actual, active = c34_velocity34(b, a, cancelled=cancel)
            self.assertFalse(active)
            np.testing.assert_array_equal(actual, np.zeros(3))

    def test_saved_reader_engine_and_model_import_guard_is_active(self):
        self.assertTrue(any(isinstance(finder, NoExecution) for finder in sys.meta_path))
        self.assertFalse(any(name.partition('.')[0] in FORBIDDEN for name in sys.modules))


if __name__ == '__main__':
    unittest.main()
