"""New C35 geometry and budget boundary checks; no simulation."""
import unittest
from task35 import actual_margin35
from runtime35 import static_bounds35


class Root35PureTest(unittest.TestCase):
    def test_stop_contact_hull_rejects_outside_and_degenerate_support(self):
        e = dict(positive_load_weighted_wheel_contact_world_m=[
            [.2,.2,0],[.2,-.2,0],[-.2,.2,0],[-.2,-.2,0]],
            whole_com_world_m=[0,0,.45])
        self.assertAlmostEqual(actual_margin35(e), .2)
        e['whole_com_world_m']=[.25,0,.45]
        self.assertLess(actual_margin35(e), 0)
        e['positive_load_weighted_wheel_contact_world_m']=[[i,0,0] for i in range(4)]
        self.assertLess(actual_margin35(e), 0)

    def test_global_static_limits_follow_reserved_controls_and_starts(self):
        self.assertEqual(static_bounds35(6000,5),dict(copy=12020,forward=1536320,
            jacBody=1152240,jac=24000,fullM=6000,objectVelocity=12005))
        self.assertEqual(static_bounds35(0,0),dict(copy=0,forward=0,
            jacBody=0,jac=0,fullM=0,objectVelocity=0))


if __name__ == '__main__':
    unittest.main()
