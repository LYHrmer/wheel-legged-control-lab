"""Pure tests for the C38 native-reader extension. No MuJoCo, no physics, no model."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# read35.py performs the one-time sys.path injection every module below depends on
# (continuation30/read30, continuation26/root26, etc.); importing it here, exactly as
# read38.py does at the top of its own module, is how those become importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'continuation35'/'read35'))
import read35  # noqa: F401,E402

from contact_math35 import all_four_support35, native_phase_support35, pair_support35  # noqa: E402
from core36 import PREUNLOAD_ELAPSED36  # noqa: E402
from native_read38 import (  # noqa: E402
    PREUNLOAD_ELAPSED38, native_phase_support38, preunload_window38,
)

MASS = 48.146865260000006


def result(mask, loads):
    return {'positive_wheel_support_mask': mask, 'positive_wheel_normal_load_sum_n': loads}


EVEN = result([True]*4, [120.]*4)
PAIR_ONLY = result([False, True, True, False], [0., 120., 120., 0.])


class BoundaryIdentity(unittest.TestCase):
    def test_constant_is_imported_not_retyped(self):
        """Guards against the two files silently drifting apart."""
        self.assertEqual(PREUNLOAD_ELAPSED38, PREUNLOAD_ELAPSED36)

    def test_window_matches_core36_at_every_sampled_point(self):
        from core36 import Core36
        core = Core36()
        for elapsed in (0., .01, .029999, .03, .030001, .04, .06):
            core.phase, core.elapsed = 'transfer', elapsed
            self.assertEqual(preunload_window38('transfer', elapsed), core.preunload36(),
                             f'elapsed {elapsed}')
        for elapsed in (0., .03, .06):
            core.phase, core.elapsed = 'transfer_restore', elapsed
            self.assertEqual(preunload_window38('transfer_restore', elapsed),
                             core.preunload36(), f'elapsed {elapsed}')


class GateSelection(unittest.TestCase):
    def test_preunload_transfer_uses_the_stance_pair_gate(self):
        proof = native_phase_support38(PAIR_ONLY, 'transfer', PREUNLOAD_ELAPSED38,
                                       (0, 3), MASS)
        self.assertEqual(proof['gate'], 'stance_pair_preunload38')
        self.assertTrue(proof['passed'])

    def test_preunload_transfer_still_fails_if_a_stance_wheel_is_lost(self):
        broken = result([False, False, True, False], [0., 0., 120., 0.])
        proof = native_phase_support38(broken, 'transfer', PREUNLOAD_ELAPSED38, (0, 3), MASS)
        self.assertFalse(proof['passed'])

    def test_before_the_boundary_the_frozen_all_four_gate_still_applies(self):
        proof = native_phase_support38(PAIR_ONLY, 'transfer', PREUNLOAD_ELAPSED38-.01,
                                       (0, 3), MASS)
        self.assertEqual(proof['gate'], 'all_four')
        self.assertFalse(proof['passed'])
        proof_even = native_phase_support38(EVEN, 'transfer', PREUNLOAD_ELAPSED38-.01,
                                            (0, 3), MASS)
        self.assertTrue(proof_even['passed'])

    def test_transfer_restore_uses_the_stance_pair_gate_throughout(self):
        stance_03 = result([True, False, False, True], [120., 0., 0., 120.])
        for elapsed in (0., .03, .06):
            proof = native_phase_support38(stance_03, 'transfer_restore', elapsed, (1, 2),
                                           MASS)
            self.assertEqual(proof['gate'], 'stance_pair_preunload38')
            self.assertTrue(proof['passed'])

    def test_other_phases_agree_exactly_with_the_frozen_selector(self):
        for phase in ('load', 'settle', 'idle', 'done'):
            new = native_phase_support38(EVEN, phase, .10, None, MASS)
            old = native_phase_support35(EVEN, phase, None, MASS)
            self.assertEqual(new['gate'], old['gate'])
            self.assertEqual(new['passed'], old['passed'])
        for phase in ('lift', 'horizontal', 'lower', 'probe', 'dwell'):
            air = result([True, False, False, True], [120., 0., 0., 120.])
            new = native_phase_support38(air, phase, .10, (1, 2), MASS)
            old = native_phase_support35(air, phase, (1, 2), MASS)
            self.assertEqual(new['gate'], old['gate'])
            self.assertEqual(new['passed'], old['passed'])

    def test_agrees_with_the_frozen_building_blocks_directly(self):
        self.assertEqual(native_phase_support38(PAIR_ONLY, 'transfer', PREUNLOAD_ELAPSED38,
                                                (0, 3), MASS)['passed'],
                         pair_support35(PAIR_ONLY, (1, 2), MASS)['passed'])
        self.assertEqual(native_phase_support38(EVEN, 'transfer', 0., (0, 3), MASS)['passed'],
                         all_four_support35(EVEN, MASS)['passed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
