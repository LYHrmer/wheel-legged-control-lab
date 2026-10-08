"""Pure tests for the C38 record semantics. No MuJoCo, no physics, no model."""
from __future__ import annotations

import unittest

from adapter35 import PairController35
from adapter38 import PairController38
from core35 import Reference35
from core36 import PREUNLOAD_ELAPSED36, swing_weight36


def reference(phase, elapsed, pair=(0, 3), feet_velocity=None):
    return Reference35(
        phase=phase, pair=pair, body_xy_m=(0., 0.), body_vxy_mps=(0., 0.),
        body_axy_mps2=(0., 0.), body_jerk_xy_mps3=(0., 0.),
        ramp_coefficients_mps=(0.,)*6, ramp_elapsed_s=.25,
        feet_world_m=((0., 0., 0.),)*4,
        feet_velocity_world_mps=feet_velocity or ((0., 0., 0.),)*4,
        force_weights=(1., 1., 1., 1.), foot_target_clipped=(False,)*4,
        landing_raw_xy_m=(None,)*4, landing_projected_xy_m=(None,)*4,
        phase_elapsed_s=elapsed, restore_start_weight=None, stop_reason=None,
        ready_for_handoff=False, safe_abort_required=False, fault=None)


class Bare(PairController38):
    """Exercise the record method alone, without a plant."""

    def __init__(self):
        self.previous_horizontal_nonzero = False
        self.previous_horizontal_pair = None


class Frozen(PairController35):
    def __init__(self):
        self.previous_horizontal_nonzero = False
        self.previous_horizontal_pair = None


class RecordSemantics(unittest.TestCase):
    def test_preunload_declares_the_regime_without_touching_swing_legs(self):
        row = Bare()._reference_record35(reference('transfer', PREUNLOAD_ELAPSED36))
        self.assertEqual(row['support_mode'], 'pair')
        self.assertEqual(row['stance_legs'], [1, 2])
        self.assertEqual(row['swing_legs'], [], 'swing_legs must stay at the frozen value')
        self.assertTrue(row['support_gate_required'])
        self.assertTrue(row['preunload36'])
        self.assertEqual(row['preunload_pair36'], [0, 3])
        self.assertAlmostEqual(row['commanded_swing_weight36'],
                               swing_weight36(PREUNLOAD_ELAPSED36), 15)

    def test_the_reader_cross_check_now_holds_in_transfer(self):
        """fullM calls are 0 in transfer, so bool(swing_legs) must also be 0."""
        for elapsed in (0., .02, PREUNLOAD_ELAPSED36, .05, .06):
            row = Bare()._reference_record35(reference('transfer', elapsed))
            self.assertEqual(int(bool(row['swing_legs'])), 0, f'elapsed {elapsed}')
        row = Bare()._reference_record35(reference('transfer_restore', .03))
        self.assertEqual(int(bool(row['swing_legs'])), 0)

    def test_air_phases_keep_the_frozen_declaration(self):
        for phase in ('lift', 'horizontal', 'lower', 'probe', 'dwell'):
            new = Bare()._reference_record35(reference(phase, .05))
            old = Frozen()._reference_record35(reference(phase, .05))
            self.assertEqual(new['support_mode'], old['support_mode'])
            self.assertEqual(new['swing_legs'], old['swing_legs'])
            self.assertEqual(new['stance_legs'], old['stance_legs'])
            self.assertEqual(int(bool(new['swing_legs'])), 1, phase)

    def test_outside_the_window_the_record_is_the_frozen_record(self):
        for phase, elapsed in (('transfer', .02), ('load', .03), ('settle', .10)):
            new = Bare()._reference_record35(reference(phase, elapsed))
            old = Frozen()._reference_record35(reference(phase, elapsed))
            self.assertEqual(new, old, f'{phase} at {elapsed}')

    def test_horizontal_active_is_untouched_by_the_declaration(self):
        moving = ((0., .1, 0.), (0., 0., 0.), (0., 0., 0.), (0., .1, 0.))
        new = Bare()._reference_record35(reference('transfer', .05, feet_velocity=moving))
        self.assertFalse(new['horizontal_active'])
        self.assertFalse(new['horizontal_reference_velocity_nonzero'])

    def test_second_diagonal_declares_the_other_stance_pair(self):
        row = Bare()._reference_record35(reference('transfer', .05, pair=(1, 2)))
        self.assertEqual(row['stance_legs'], [0, 3])
        self.assertEqual(row['swing_legs'], [])
        self.assertEqual(row['preunload_pair36'], [1, 2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
