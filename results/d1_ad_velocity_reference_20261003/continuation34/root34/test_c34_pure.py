"""Pure contract fixtures; root owns execution and its finite check budget."""
from __future__ import annotations

import unittest

import numpy as np

from macro30 import components30
from macro31 import canonical_components31
from macro34 import distance_components34
from selection34 import best34, select34, selected_stage_b_cases34
from task31 import GATES31 as ORIGINAL_GATES
from task34 import gates34, target34


def pair(distance, beta, time, cost, *, complete=True, eligible=True):
    return dict(distance_m=distance, beta=beta, max_side_duration_s=time,
                mean_full_tau2_mean=cost, complete=complete, eligible=eligible,
                case_ids=['left', 'right'])


class C34PureContract(unittest.TestCase):
    def test_d30_equivalence_and_true_d40_goal_progress(self):
        self.assertEqual(gates34(.03), ORIGINAL_GATES)
        self.assertEqual(gates34(.04)['signed_lateral_m'], (.035, .05))
        self.assertEqual(gates34(.04)['final_xy_error_lt_m'], .012)
        np.testing.assert_array_equal(target34([1.,2.], 0., 1, .03), [1.,2.03])
        np.testing.assert_array_equal(target34([1.,2.], 0., -1, .04), [1.,1.96])
        q = np.zeros((2,23), dtype=float)
        q[:,3] = 1.
        q[1,1] = .04
        actual30 = canonical_components31(components30(q, np.zeros((1,5,16)),
            np.ones(16), origin_xy=[0.,0.], yaw0=0., direction=1), (0,1))
        recalculated30 = distance_components34(q, origin_xy=[0.,0.],
            yaw0=0., direction=1, distance_m=.03)
        for key,value in recalculated30.items():
            self.assertAlmostEqual(value, actual30[key], places=14)
        recalculated40 = distance_components34(q, origin_xy=[0.,0.],
            yaw0=0., direction=1, distance_m=.04)
        self.assertAlmostEqual(recalculated40['progress_potential_difference'], 1.)
        self.assertAlmostEqual(recalculated40['lateral_goal_error_square_integral_m2_s'], 0.)
        self.assertNotEqual(recalculated30['lateral_goal_error_square_integral_m2_s'], 0.)

    def test_rank_worst_then_cost_then_lower_beta(self):
        candidates = [pair(.03,0.,9.5,.02), pair(.03,.5,9.4,.03),
                      pair(.03,1.,9.4,.025)]
        self.assertEqual(best34(candidates)['beta'], 1.)
        candidates[-1]['mean_full_tau2_mean'] = .03
        self.assertEqual(best34(candidates)['beta'], .5)

    def test_controlled_failure_and_incomplete_40_fall_back_to_eligible_30(self):
        candidates = [pair(.03,0.,9.37,.02),
                      pair(.03,.5,9.1,.021,eligible=False),
                      pair(.04,0.,10.,.022,complete=False)]
        self.assertEqual(select34(candidates)['beta'], 0.)
        self.assertEqual(select34(candidates)['distance_m'], .03)
        candidates[2]['complete'] = True
        candidates[2]['eligible'] = True
        self.assertEqual(select34(candidates)['distance_m'], .04)

    def test_stage_b_skips_duplicate_beta_zero(self):
        self.assertEqual(selected_stage_b_cases34(0.), [])
        names = [case['case_id'] for case in selected_stage_b_cases34(.5)]
        self.assertEqual(names, ['d040_beta050_left','d040_beta050_right'])


if __name__ == '__main__':
    unittest.main()
