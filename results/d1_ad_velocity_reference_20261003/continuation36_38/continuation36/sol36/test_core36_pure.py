"""Pure tests for the C36 pre-unload semantics. No MuJoCo, no physics, no model."""
from __future__ import annotations

import unittest
from dataclasses import replace

from core35 import PAIRS35, Command35, Core35, Reference35, Sensed35, smooth35
from core36 import PREUNLOAD_ELAPSED36, PREUNLOAD_WEIGHT36, Core36, swing_weight36

MASS = 48.146865260000006
HALF_WEIGHT = .5*MASS*9.81


def sensed(loads, *, control=0):
    """Minimal Sensed35 carrying only what the support gates read."""
    return Sensed35(
        control_index=control, body_xy_m=(0., 0.), body_com_vxy_mps=(0., 0.),
        body_origin_vxyz_mps=(0., 0., 0.), yaw_rad=0.,
        foot_world_m=((.19, .22, .0875), (.19, -.22, .0875),
                      (-.19, .22, .0875), (-.19, -.22, .0875)),
        wheel_contact_points_world_m=((.19, .22, 0.), (.19, -.22, 0.),
                                      (-.19, .22, 0.), (-.19, -.22, 0.)),
        actual_whole_wheel_min_z_m=(0., 0., 0., 0.),
        previous_five_contact_free=(False,)*4, previous_five_pair_loaded=(True, True),
        foot_contact=tuple(value > 0 for value in loads),
        normal_load_n=tuple(loads), mass_kg=MASS, whole_com_z_m=.3876, base_z_m=.4524,
        static_support_margin_m=.02, world_angular_speed_rps=0.,
        native_safety_ok=True)


class PreunloadBoundary(unittest.TestCase):
    def test_weight_boundary_is_exact(self):
        self.assertAlmostEqual(swing_weight36(PREUNLOAD_ELAPSED36), PREUNLOAD_WEIGHT36, 15)
        self.assertAlmostEqual(swing_weight36(0.), 1., 15)
        self.assertAlmostEqual(swing_weight36(.06), 0., 15)

    def test_weight_matches_the_frozen_ramp(self):
        for elapsed in (0., .01, .02, .03, .04, .05, .06):
            self.assertAlmostEqual(swing_weight36(elapsed), 1.-smooth35(elapsed/.06), 15)


class SupportRegime(unittest.TestCase):
    def setUp(self):
        self.core = Core36()
        self.core.pair_index = 0                       # swing pair (0, 3)

    def even(self):
        return sensed([HALF_WEIGHT/2]*4)

    def pair_only(self):
        loads = [0., HALF_WEIGHT/2, HALF_WEIGHT/2, 0.]  # stance pair (1, 2) carries it
        return sensed(loads)

    def test_before_the_boundary_all_four_is_required(self):
        self.core.phase, self.core.elapsed = 'transfer', PREUNLOAD_ELAPSED36-.01
        self.assertFalse(self.core.preunload36())
        self.assertTrue(self.core._all_loaded(self.even()))
        self.assertFalse(self.core._all_loaded(self.pair_only()))

    def test_at_and_after_the_boundary_the_stance_pair_suffices(self):
        for elapsed in (PREUNLOAD_ELAPSED36, PREUNLOAD_ELAPSED36+.01, .06):
            self.core.phase, self.core.elapsed = 'transfer', elapsed
            self.assertTrue(self.core.preunload36())
            self.assertTrue(self.core._all_loaded(self.pair_only()))
            self.assertTrue(self.core._all_loaded(self.even()))

    def test_the_stance_pair_gate_is_not_weaker_than_the_frozen_air_gate(self):
        self.core.phase, self.core.elapsed = 'transfer', PREUNLOAD_ELAPSED36
        swing = PAIRS35[self.core.pair_index]
        # losing a STANCE wheel must still fail, exactly as the frozen air regime requires
        for lost in (1, 2):
            loads = [0., HALF_WEIGHT/2, HALF_WEIGHT/2, 0.]
            loads[lost] = 0.
            self.assertFalse(self.core._all_loaded(sensed(loads)))
        # an insufficient stance-pair sum must fail even with both wheels touching
        self.assertFalse(self.core._all_loaded(sensed([0., 1., 1., 0.])))
        # and the frozen helper agrees on the same evidence
        self.assertTrue(Core35._stance_loaded(self.pair_only(), swing))

    def test_other_phases_keep_the_inherited_all_four_requirement(self):
        for phase in ('load', 'settle', 'lift', 'horizontal', 'lower', 'probe', 'dwell'):
            self.core.phase, self.core.elapsed = phase, .10
            self.assertFalse(self.core.preunload36())
            self.assertFalse(self.core._all_loaded(self.pair_only()))
            self.assertTrue(self.core._all_loaded(self.even()))

    def test_transfer_restore_uses_the_stance_pair_throughout(self):
        for elapsed in (0., .03, .06):
            self.core.phase, self.core.elapsed = 'transfer_restore', elapsed
            self.assertTrue(self.core.preunload36())
            self.assertTrue(self.core._all_loaded(self.pair_only()))

    def test_second_pair_index_checks_the_other_diagonal(self):
        self.core.pair_index = 1                       # swing pair (1, 2)
        self.core.phase, self.core.elapsed = 'transfer', PREUNLOAD_ELAPSED36
        self.assertTrue(self.core._all_loaded(sensed([HALF_WEIGHT/2, 0., 0., HALF_WEIGHT/2])))
        self.assertFalse(self.core._all_loaded(sensed([0., HALF_WEIGHT/2, HALF_WEIGHT/2, 0.])))


class DeclaredRegime(unittest.TestCase):
    """The adapter must declare the same boundary the core enforces."""

    def reference(self, phase, elapsed, pair=(0, 3)):
        return Reference35(
            phase=phase, pair=pair, body_xy_m=(0., 0.), body_vxy_mps=(0., 0.),
            body_axy_mps2=(0., 0.), body_jerk_xy_mps3=(0., 0.),
            ramp_coefficients_mps=(0.,)*6, ramp_elapsed_s=.25,
            feet_world_m=((0., 0., 0.),)*4, feet_velocity_world_mps=((0., 0., 0.),)*4,
            force_weights=(1., 1., 1., 1.), foot_target_clipped=(False,)*4,
            landing_raw_xy_m=(None,)*4, landing_projected_xy_m=(None,)*4,
            phase_elapsed_s=elapsed, restore_start_weight=None, stop_reason=None,
            ready_for_handoff=False, safe_abort_required=False, fault=None)

    def test_adapter_and_core_switch_together(self):
        from adapter36 import PairController36
        core = Core36()
        core.pair_index = 0
        for phase in ('transfer', 'transfer_restore', 'load', 'settle'):
            for elapsed in (0., .02, .03, .05, .06):
                core.phase, core.elapsed = phase, elapsed
                declared = PairController36.preunload36(self.reference(phase, elapsed))
                self.assertEqual(declared, core.preunload36(),
                                 f'{phase} at {elapsed}')

    def test_reference_is_untouched_by_the_mirror(self):
        row = self.reference('transfer', .05)
        self.assertEqual(replace(row, phase_elapsed_s=.05), row)


if __name__ == '__main__':
    unittest.main(verbosity=2)
