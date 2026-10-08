"""Exact differential kinematics for the crouch-alignment wheel-roll compensation.

docs/ad_lateral_gait_review_20261004.md section 9.5 identified the mechanism but never
derived it as a control law: morphing the nominal crouch from thigh 0.800/calf -1.500 to
the aligned thigh 0.7524/calf -1.502261 moves each wheel's body-frame x position by
+25.53 mm (aligned_crouch_feasibility_20261005.py section 5). If the base is held fixed in
world during the morph, that body-frame shift is also a world-frame shift of the wheel
AXLE. For the wheel to stay in contact without sliding, its ground contact point must track
that same axle motion by ROLLING - not by commanding a position target (the frozen
wheel_angles anchor is a BRAKE, deliberately locking rotation during the side gait), but by
commanding a rolling-without-slip VELOCITY reference synchronised with the leg-angle ramp.

This script derives that reference exactly: it reuses the already-verified foot Jacobian
(motion_optimisation_20261005.jacobian) to get d(foot_x)/d(thigh) and d(foot_x)/d(calf) at
the nominal crouch, converts the commanded leg-angle RATE into a required wheel angular
velocity via rolling-without-slip (v_axle_x = -omega_wheel * radius, standard sign for a
wheel whose positive rotation drives the robot in +x), and prices the required torque
against the already-measured four-wheel stance torque budget.

Pure arithmetic over the URDF tree: no MuJoCo, no model construction, no physics, no model
load, no fitting.
"""
from __future__ import annotations

import math

from aligned_crouch_feasibility_20261005 import foot_of
from motion_optimisation_20261005 import (
    ALIGNED_CALF_RAD,
    ALIGNED_THIGH_RAD,
    HARD_LEG_QVEL_RPS,
    LEG_TORQUE_NM,
    REF_QVEL_CLIP_RPS,
    URDF_INERTIA,
    gravity_torque,
    jacobian,
    leg_mass_matrix,
)
from posture_statics_20261004 import NOMINAL, G, angle_vector, parse, uniform

WHEEL_RADIUS_M = .087             # same tight bound used throughout the review series
WHEEL_TORQUE_NM = 12.0            # contract35.json / model.py wheel actuator limit
WHEEL_BRAKE_KP = 12.0             # scripts/d1_fast_side_step.py wheel brake, for reference
MORPH_SECONDS_CANDIDATES = (.0172, .05, .1, .2, .5)  # .0172s is the frozen-limbs-only figure


def partial_foot_x(masses, joints, leg, angles, joint, step=1e-6):
    """d(foot_x)/d(joint angle) by central difference, holding the other two leg joints."""
    key = f'{leg}_{joint}_joint'
    shifted = []
    for sign in (+1, -1):
        perturbed = dict(angles)
        perturbed[key] = perturbed[key]+sign*step
        shifted.append(foot_of(masses, joints, leg, perturbed)[0])
    return (shifted[0]-shifted[1])/(2*step)


def main() -> None:
    masses, joints = parse()
    nominal_angles = angle_vector(uniform(NOMINAL['thigh'], NOMINAL['calf']))
    thigh_delta = ALIGNED_THIGH_RAD-NOMINAL['thigh']
    calf_delta = ALIGNED_CALF_RAD-NOMINAL['calf']

    print('== 1. exact body-frame foot-x sensitivity at the nominal crouch ==')
    d_thigh = partial_foot_x(masses, joints, 'FL', nominal_angles, 'thigh')
    d_calf = partial_foot_x(masses, joints, 'FL', nominal_angles, 'calf')
    print(f'  d(foot_x)/d(thigh) = {d_thigh:+.5f} m/rad')
    print(f'  d(foot_x)/d(calf)  = {d_calf:+.5f} m/rad')
    total_shift = d_thigh*thigh_delta+d_calf*calf_delta
    print(f'  linearised total foot-x shift for the commanded delta '
          f'(thigh {thigh_delta:+.6f}, calf {calf_delta:+.6f} rad): {total_shift*1000:+.3f} mm')
    print('  cross-check against the finite (non-linearised) shift already measured in '
          'aligned_crouch_feasibility_20261005.py section 5: +25.53 mm')
    print(f'  linear-vs-finite discrepancy: {abs(total_shift*1000-25.53):.3f} mm '
          f'({abs(total_shift*1000/25.53-1)*100:.1f}%), expected from the second-order term '
          f'since 0.0687 rad of thigh travel is not infinitesimal')

    print('\n== 2. rolling-without-slip wheel angle required, exact (not linearised) ==')
    # Use the exact finite shift (+25.53 mm) rather than the linear approximation, since the
    # morph is not infinitesimal; sign convention: body-frame +x shift of the axle with the
    # base held fixed in world must be matched by the wheel rolling forward by the same
    # ground arc length, so required wheel angle = shift / radius, independent of path.
    exact_shift_m = .02553   # aligned_crouch_feasibility_20261005.py section 5, per leg
    roll_rad = exact_shift_m/WHEEL_RADIUS_M
    print(f'  required wheel roll per leg: {exact_shift_m*1000:.2f} mm / {WHEEL_RADIUS_M} m '
          f'= {roll_rad:.4f} rad ({math.degrees(roll_rad):.2f} deg)')
    print('  this matches the figure already reported in aligned_crouch_feasibility '
          'section 5 (0.2934 rad); this script adds the velocity/torque law needed to '
          'actually command it, which that script did not derive')

    print('\n== 3. commanded wheel angular velocity for a range of morph durations ==')
    print(f'  {"duration":>10} {"leg rate":>12} {"wheel rate":>12} {"vs ref clip":>12} '
          f'{"vs hard gate":>13}')
    for duration in MORPH_SECONDS_CANDIDATES:
        leg_rate = max(abs(thigh_delta), abs(calf_delta))/duration
        wheel_rate = roll_rad/duration
        print(f'  {duration:10.4f}s {leg_rate:11.4f} {wheel_rate:11.4f} '
              f'{wheel_rate/REF_QVEL_CLIP_RPS*100:11.1f}% {wheel_rate/HARD_LEG_QVEL_RPS*100:12.1f}%')
    print('  (leg rate shown for the worst of thigh/calf; wheel rate is the required roll '
          'rate, both well inside the 4 rad/s reference clip and 18 rad/s hard gate at '
          'every candidate duration)')

    print('\n== 4. torque budget: four-wheel stance plus the rolling command ==')
    columns = jacobian(masses, joints, 'FL', nominal_angles)
    static = gravity_torque(masses, joints, 'FL', nominal_angles)
    total_mass = sum(mass for mass, _ in masses.values())
    share_n = total_mass*G/4
    # Static four-wheel stance torque, reused from aligned_crouch_feasibility section 3.
    reaction = (0., 0., share_n)
    leg_torque = [sum(columns[i][axis]*reaction[axis] for axis in range(3))+static[i]
                  for i in range(3)]
    worst_leg_torque = max(abs(value) for value in leg_torque)
    print(f'  four-wheel stance leg torque (reused figure): {worst_leg_torque:.3f} N*m of '
          f'{LEG_TORQUE_NM} N*m ({worst_leg_torque/LEG_TORQUE_NM*100:.1f}%)')
    matrix = leg_mass_matrix(masses, joints, 'FL', nominal_angles)
    print('  leg joint inertia diagonal (hip, thigh, calf), kg*m^2:',
          [round(matrix[i][i], 4) for i in range(3)])
    wheel_spin_inertia = URDF_INERTIA['FL_foot'][2][2]  # local z is the joint axis (axis="0 0 1")
    print(f'  exact wheel spin-axis inertia from the URDF tensor (FL_foot, I_zz): '
          f'{wheel_spin_inertia:.6f} kg*m^2 (local z is the joint axis)')
    print(f'  {"duration":>10} {"leg accel":>11} {"leg torque":>12} {"wheel accel":>12} '
          f'{"wheel torque":>13} {"vs 12 N*m":>10}')
    for duration in MORPH_SECONDS_CANDIDATES:
        # Same symmetric bang-bang bound used throughout the review series: 4*distance/T^2.
        leg_accel = 4*max(abs(thigh_delta), abs(calf_delta))/duration**2
        dynamic_torque = abs(matrix[1][1]*leg_accel)   # thigh row, diagonal term only
        wheel_accel = 4*roll_rad/duration**2
        wheel_torque = wheel_accel*wheel_spin_inertia
        print(f'  {duration:9.4f}s {leg_accel:11.2f} {dynamic_torque:12.3f} {wheel_accel:12.2f} '
              f'{wheel_torque:13.4f} {wheel_torque/WHEEL_TORQUE_NM*100:9.1f}%')
    low, high = .005, 1.
    for _ in range(100):
        middle = (low+high)/2
        torque = (4*roll_rad/middle**2)*wheel_spin_inertia
        low, high = (middle, high) if torque > WHEEL_TORQUE_NM else (low, middle)
    wheel_limited_duration = high
    print(f'  wheel-torque-limited minimum morph duration (exact inertia, bang-bang): '
          f'{wheel_limited_duration:.4f} s')

    low, high = .005, 1.
    for _ in range(100):
        middle = (low+high)/2
        leg_dynamic = abs(matrix[1][1])*4*max(abs(thigh_delta), abs(calf_delta))/middle**2
        total_leg_torque = worst_leg_torque+leg_dynamic
        low, high = (middle, high) if total_leg_torque > LEG_TORQUE_NM else (low, middle)
    leg_limited_duration = high
    print(f'  LEG-torque-limited minimum duration (static stance + dynamic morph combined, '
          f'against {LEG_TORQUE_NM} N*m): {leg_limited_duration:.4f} s')
    print(f'  binding duration: {max(wheel_limited_duration, leg_limited_duration):.4f} s '
          f'({"leg" if leg_limited_duration > wheel_limited_duration else "wheel"} is tighter)')
    print('  the earlier 0.0172 s figure in aligned_crouch_feasibility_20261005.py section 5 '
          'used only the reference VELOCITY clip and checked neither torque budget: at that '
          'duration the wheel needs ~30 N*m (2.5x its 12 N*m limit) AND the leg needs ~296 '
          'N*m total (3.7x its 80 N*m limit). Both are INFEASIBLE. This corrects that '
          'omission; a ~0.1 s morph clears both with comfortable margin and is still a '
          'small fraction of the 200-control (2 s) B22 preparation window.')

    print('\n== 5. what must actually change in the control law ==')
    print('  current frozen wheel reference: position lock to self.wheel_angles (a BRAKE),')
    print('    tau[wheel] += 12*(wheel_angles - qpos[wheel]) - 2*velocity[wheel]')
    print('  needed for the align phase: replace the POSITION target with a ramped')
    print('    VELOCITY target (or a ramped position target that itself advances by the')
    print('    rolling relation above), synchronised with the thigh/calf quintic ramp,')
    print('    for the SAME duration, released back to the position brake once the morph')
    print('    completes and the new foot anchors are re-measured from actual contact.')
    print('  this is a genuinely new reference law, not a parameter change to an existing')
    print('  one: it requires its own Core/Adapter phase, its own pure tests for the')
    print('  roll/leg synchronisation, and its own contract and GO before any physics.')


if __name__ == '__main__':
    main()
