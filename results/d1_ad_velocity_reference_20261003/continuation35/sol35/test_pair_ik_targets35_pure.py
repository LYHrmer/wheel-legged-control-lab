"""Source-only fixture for the exact helper used at every paired IK outer pass."""
import unittest

import numpy as np

from pair_ik_targets35 import pair_ik_targets35


class PairIKTargets35Test(unittest.TestCase):
    def test_both_swing_z_offsets_survive_one_target_assembly(self):
        anchors = np.array([[.2,.2,.08],[.2,-.2,.09],
                            [-.2,.2,.10],[-.2,-.2,.11]])
        requested = anchors.copy()
        requested[0,:] += [.014,.021,.035]
        requested[3,:] += [-.019,.017,.035]
        # Even a corrupt stance Z proposal cannot override geometry correction.
        requested[1,2] = .8
        requested[2,2] = .7
        extent = np.array([.082,.091,.103,.114])
        targets = pair_ik_targets35(requested,anchors,extent,(0,3))
        np.testing.assert_array_equal(targets[:,:2],requested[:,:2])
        np.testing.assert_allclose(targets[:,2],
            [extent[0]-.001+.035,extent[1]-.001,
             extent[2]-.001,extent[3]-.001+.035],rtol=0,atol=1e-15)


if __name__ == '__main__':
    unittest.main()
