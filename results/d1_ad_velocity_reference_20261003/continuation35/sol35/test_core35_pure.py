"""Source-only C35 fixtures; only root may execute these checks."""
import math
import unittest

from core35 import (Command35, Core35, Sensed35, hermite_velocity35,
                    landing_xy35, velocity_integral35, velocity_sample35)


def sensed(index):
    feet = ((.2,.2,.09),(.2,-.2,.09),(-.2,.2,.09),(-.2,-.2,.09))
    return Sensed35(control_index=index,body_xy_m=(0.,0.),
        body_com_vxy_mps=(0.,0.),body_origin_vxyz_mps=(0.,0.,0.),
        yaw_rad=0.,foot_world_m=feet,
        wheel_contact_points_world_m=tuple((x,y,0.) for x,y,_ in feet),
        actual_whole_wheel_min_z_m=(0.,)*4,
        previous_five_contact_free=(False,)*4,
        previous_five_pair_loaded=(True,True),foot_contact=(True,)*4,
        normal_load_n=(3.,)*4,mass_kg=1.,whole_com_z_m=.45,
        base_z_m=.4,static_support_margin_m=.02,
        world_angular_speed_rps=0.,native_safety_ok=True)


class Core35PureTest(unittest.TestCase):
    def test_symmetric_velocity_pulse_integrates_exactly(self):
        up = hermite_velocity35(0.,0.,0.,.025)
        down = hermite_velocity35(.025,0.,0.,0.)
        displacement = (velocity_integral35(up,0.,.25)
            +.025*5.75+velocity_integral35(down,0.,.25))
        self.assertAlmostEqual(displacement,.15,places=12)
        self.assertAlmostEqual(velocity_sample35(up,.25)[0],.025,places=12)
        self.assertAlmostEqual(velocity_sample35(down,.25)[0],0.,places=12)

    def test_mid_ramp_cancel_preserves_velocity_acceleration_jerk(self):
        up = hermite_velocity35(0.,0.,0.,.025)
        before = velocity_sample35(up,.07)
        stop = hermite_velocity35(*before,0.)
        after = velocity_sample35(stop,0.)
        for a,b in zip(before,after):
            self.assertAlmostEqual(a,b,places=12)
        ending = velocity_sample35(stop,.25)
        for value in ending:
            self.assertAlmostEqual(value,0.,places=12)

    def test_landing_radial_cap_keeps_raw_and_projected(self):
        raw,projected,clipped = landing_xy35((.1,.2),(0.,0.),
            (0.,0.),(0.,0.),(0.,0.))
        self.assertTrue(clipped)
        self.assertEqual(raw,(.1,.2))
        self.assertAlmostEqual(math.hypot(*projected),.06,places=12)

    def test_release_in_transfer_does_not_admit_new_pair(self):
        core = Core35()
        held = Command35(.025,True,0,20,'none')
        released = Command35(0.,False,1,21,'release')
        first = core.step(held,sensed(0))
        second = core.step(released,sensed(1))
        self.assertEqual(first.phase,'transfer')
        self.assertEqual(second.phase,'transfer_restore')
        self.assertAlmostEqual(second.force_weights[0],first.force_weights[0])
        for index in range(2,9):
            next_ref = core.step(Command35(0.,False,index,index+20,'release'),
                                 sensed(index))
        self.assertEqual(next_ref.phase,'settle')
        self.assertEqual(core.completed_pair_exchanges,0)
        self.assertEqual(second.stop_reason,'release')

    def test_ttl_boundary_and_reversal_lock_stop_before_liftoff(self):
        stale = Command35(.025,True,0,100,'none')
        never_started = Core35()
        self.assertEqual(never_started.step(stale,sensed(20)).phase,'idle')

        active = Core35()
        active.step(stale,sensed(0))
        reverse = Command35(-.025,True,1,21,'none')
        result = active.step(reverse,sensed(1))
        self.assertEqual(result.phase,'transfer_restore')
        self.assertEqual(result.stop_reason,'reverse_wait')
        self.assertEqual(active.ramp_target,0.)
        self.assertEqual(active.completed_pair_exchanges,0)

        expiring = Core35()
        for index in range(21):
            result = expiring.step(stale,sensed(index))
        self.assertEqual(result.stop_reason,'expired')
        self.assertEqual(expiring.ramp_target,0.)
        self.assertTrue(expiring.stop_requested)


if __name__ == '__main__':
    unittest.main()
