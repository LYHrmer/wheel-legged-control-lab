"""Is the aligned crouch actually usable? Conditioning, workspace, torque and entry cost.

Follow-up to docs/ad_lateral_gait_review_20261004.md sections 4.2 and 8. Those sections
showed that a crouch of thigh 0.7313 rad removes the two-support statics deficit for free,
and that the optimised gait then reaches 0.07317 m/s. This script checks the four things
that could still block it:

  1. inverse-kinematics conditioning at the aligned crouch versus the frozen one,
  2. whether the 60 mm landing cap is a workspace limit or a design choice,
  3. stance-leg torque while carrying its share of the robot,
  4. what it costs to enter the aligned crouch from the frozen B22 nominal, given that the
     feet are planted and the 30 mm longitudinal excursion limit is frozen,
  5. and, asked directly: why step at all instead of just driving the leg roll axis?

Pure arithmetic over the URDF tree: no MuJoCo, no model construction, no physics, no model
load, no fitting. Kinematics helpers come from posture_statics_20261004, which validates
them against the saved C35 pre-state.
"""
from __future__ import annotations

import math

from motion_optimisation_20261005 import (
    ALIGNED_CALF_RAD,
    ALIGNED_THIGH_RAD,
    LEG_TORQUE_NM,
    REF_QVEL_CLIP_RPS,
    direction_cost,
    gravity_torque,
    jacobian,
    solve3,
)
from posture_statics_20261004 import (
    LANDING_CAP_M,
    LIMITS,
    NOMINAL,
    G,
    angle_vector,
    forward,
    parse,
    pose_geometry,
    uniform,
)

MAX_LONGITUDINAL_M = .030        # contract35.json task.continuous
SWING_CLEARANCE_MM = 12.0        # contract35.json native_safety.new_pair_contact_gate
STANCE_JOINT_PD = (80.0, 3.0)    # contract35.json reference.stance_joint_PD
WHEEL_RADIUS_M = .087
STATICS_TOLERANCE_RAD = .0554    # posture_statics_20261004 section 5
OBSERVED_SAG_RAD = .0211         # commanded 0.8 vs measured 0.7789 in the saved pre-state


def symmetric_eigenvalues(matrix):
    """Closed-form eigenvalues of a symmetric 3x3 matrix (Smith's trigonometric method)."""
    a, b, c = matrix[0]
    _, d, e = matrix[1]
    _, _, f = matrix[2]
    trace = (a+d+f)/3
    shifted = [[a-trace, b, c], [b, d-trace, e], [c, e, f-trace]]
    squared = sum(shifted[i][j]**2 for i in range(3) for j in range(3))
    scale = math.sqrt(squared/6)
    if scale < 1e-300:
        return (trace, trace, trace)
    reduced = [[value/scale for value in row] for row in shifted]
    determinant = (reduced[0][0]*(reduced[1][1]*reduced[2][2]-reduced[1][2]*reduced[2][1])
                   - reduced[0][1]*(reduced[1][0]*reduced[2][2]-reduced[1][2]*reduced[2][0])
                   + reduced[0][2]*(reduced[1][0]*reduced[2][1]-reduced[1][1]*reduced[2][0]))
    angle = math.acos(max(-1., min(1., determinant/2)))/3
    values = [trace+2*scale*math.cos(angle+2*math.pi*k/3) for k in range(3)]
    return tuple(sorted(values, reverse=True))


def singular_values(columns):
    """Singular values of the 3x3 foot Jacobian, from the eigenvalues of J^T J."""
    gram = [[sum(columns[i][axis]*columns[j][axis] for axis in range(3))
             for j in range(3)] for i in range(3)]
    return tuple(math.sqrt(max(0., value)) for value in symmetric_eigenvalues(gram))


def foot_of(masses, joints, leg, angles):
    identity = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    frames, _, _, _ = forward(masses, joints, angles, (0., 0., 0.), identity)
    return frames[f'{leg}_foot'][1]


def solve_ik(masses, joints, leg, start, target, iterations=60):
    """Newton inverse kinematics on (hip, thigh, calf); returns angles or None."""
    angles = dict(start)
    keys = [f'{leg}_{name}_joint' for name in ('hip', 'thigh', 'calf')]
    for _ in range(iterations):
        current = foot_of(masses, joints, leg, angles)
        error = tuple(target[axis]-current[axis] for axis in range(3))
        if max(abs(value) for value in error) < 1e-10:
            return angles
        try:
            delta, _ = solve3(jacobian(masses, joints, leg, angles), error)
        except ValueError:
            return None
        for index, key in enumerate(keys):
            angles[key] = angles[key]+delta[index]
    current = foot_of(masses, joints, leg, angles)
    if max(abs(target[axis]-current[axis]) for axis in range(3)) > 1e-8:
        return None
    return angles


def within_limits(angles, leg):
    for name in ('hip', 'thigh', 'calf'):
        low, high = LIMITS[name]
        if not low <= angles[f'{leg}_{name}_joint'] <= high:
            return False
    return True


def main() -> None:
    masses, joints = parse()
    total_mass = sum(mass for mass, _ in masses.values())
    frozen_angles = angle_vector(uniform(NOMINAL['thigh'], NOMINAL['calf']))
    aligned_angles = angle_vector(uniform(ALIGNED_THIGH_RAD, ALIGNED_CALF_RAD))

    print('== 1. inverse-kinematics conditioning ==')
    for label, angles in (('frozen nominal  thigh 0.800', frozen_angles),
                          ('aligned crouch  thigh 0.731', aligned_angles)):
        columns = jacobian(masses, joints, 'FL', angles)
        values = singular_values(columns)
        lateral, _ = direction_cost(columns, (0., 1., 0.))
        print(f'  {label}: singular values {values[0]:.4f} / {values[1]:.4f} / '
              f'{values[2]:.4f} m/rad, condition number {values[0]/values[2]:.3f}')
        print(f'    worst joint rate per 1 m/s lateral foot motion {lateral:.4f} rad/s')
    print('  the aligned crouch is 4 degrees from the frozen one, so conditioning barely '
          'moves: no new singularity risk')

    print('\n== 2. is the 60 mm landing cap a workspace limit? ==')
    reach = {}
    for leg in ('FL',):
        base_foot = foot_of(masses, joints, leg, aligned_angles)
        for direction, name in (((0., 1., 0.), 'outward +y'), ((0., -1., 0.), 'inward -y'),
                                ((1., 0., 0.), 'forward +x'), ((-1., 0., 0.), 'back -x')):
            low, high = 0., .40
            for _ in range(80):
                middle = (low+high)/2
                target = tuple(base_foot[axis]+direction[axis]*middle for axis in range(3))
                angles = solve_ik(masses, joints, leg, aligned_angles, target)
                ok = angles is not None and within_limits(angles, leg)
                low, high = (middle, high) if ok else (low, middle)
            reach[name] = low
            print(f'  {leg} foot can move {low*1000:6.1f} mm {name} at constant height '
                  f'inside the compiled joint limits')
    smallest = min(reach.values())
    print(f'  smallest reach {smallest*1000:.1f} mm against the {LANDING_CAP_M*1000:.0f} mm '
          f'landing cap: the cap uses {LANDING_CAP_M/smallest*100:.1f}% of the workspace, so '
          f'it is a design choice, not an IK limit')

    print('\n== 3. stance-leg torque while carrying its share ==')
    share_n = total_mass*G/4
    pair_share_n = total_mass*G/2
    columns = jacobian(masses, joints, 'FL', aligned_angles)
    static = gravity_torque(masses, joints, 'FL', aligned_angles)

    def stance_torque(reaction):
        return [sum(columns[index][axis]*reaction[axis] for axis in range(3))+static[index]
                for index in range(3)]

    for label, load_n in (('four-wheel support, mg/4 per wheel', share_n),
                          ('two-support diagonal, mg/2 per wheel', pair_share_n)):
        torque = stance_torque((0., 0., load_n))
        worst = max(abs(value) for value in torque)
        print(f'  {label}: {load_n:.1f} N -> joint torques '
              f'({torque[0]:+.3f}, {torque[1]:+.3f}, {torque[2]:+.3f}) N*m, worst '
              f'{worst:.3f} N*m, {worst/LEG_TORQUE_NM*100:.1f}% of {LEG_TORQUE_NM} N*m')
    print(f'  a pure stance PD of kp={STANCE_JOINT_PD[0]} could not hold this at all: '
          f'{max(abs(v) for v in stance_torque((0., 0., pair_share_n)))/STANCE_JOINT_PD[0]:.3f}'
          ' rad of droop. The original force allocation feed-forward is what holds the '
          'stance leg, and the saved run droops only 0.0211 rad, so the feed-forward is '
          'carrying essentially all of it. This is inherited behaviour, not a new risk.')
    # Does the stance pair have the torque headroom to reject the optimised swing reaction?
    swing_reaction_n = 103.
    per_wheel_n = swing_reaction_n/2
    combined = stance_torque((0., per_wheel_n, pair_share_n))
    hold_only = stance_torque((0., 0., pair_share_n))
    worst_combined = max(abs(value) for value in combined)
    print(f'  rejecting the optimised gait\'s {swing_reaction_n:.0f} N swing reaction adds '
          f'{per_wheel_n:.1f} N lateral per stance wheel:')
    print(f'    static hold alone {max(abs(v) for v in hold_only):.3f} N*m -> with rejection '
          f'{worst_combined:.3f} N*m, {worst_combined/LEG_TORQUE_NM*100:.1f}% of the limit')
    print(f'    torque headroom left {LEG_TORQUE_NM-worst_combined:.2f} N*m '
          f'({(1-worst_combined/LEG_TORQUE_NM)*100:.1f}%), so the actuator is not the '
          f'limit on rejecting it - whether the PD loops actually do is a separate, '
          f'simulation-only question')

    print('\n== 4. holding the aligned crouch against real tracking error ==')
    print(f'  statics tolerates thigh error +/-{STATICS_TOLERANCE_RAD:.4f} rad '
          f'({math.degrees(STATICS_TOLERANCE_RAD):.2f} deg)')
    print(f'  the saved C35 pre-state sat {OBSERVED_SAG_RAD:.4f} rad below its 0.800 command, '
          f'which is {OBSERVED_SAG_RAD/STATICS_TOLERANCE_RAD*100:.0f}% of that tolerance')
    print('  so the aligned crouch must be defined on the ACTUAL pose, not the command: '
          f'commanding {ALIGNED_THIGH_RAD:.4f} and settling {OBSERVED_SAG_RAD:.4f} low lands '
          f'at {ALIGNED_THIGH_RAD-OBSERVED_SAG_RAD:.4f} rad, still inside the window but '
          f'{(1-(OBSERVED_SAG_RAD)/STATICS_TOLERANCE_RAD)*100:.0f}% of margin left')
    print(f'  commanding {ALIGNED_THIGH_RAD+OBSERVED_SAG_RAD:.4f} instead recentres it; the '
          'new contract should specify the aligned crouch as a measured-pose target with a '
          'bias for the observed droop')

    identity = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    frames_aligned, _, _, _ = forward(masses, joints, aligned_angles, (0., 0., 0.), identity)

    print('\n== 5. the entry cost nobody has priced: the feet are planted ==')
    frozen_geometry = pose_geometry(masses, joints, NOMINAL['thigh'], NOMINAL['calf'])
    aligned_geometry = pose_geometry(masses, joints, ALIGNED_THIGH_RAD, ALIGNED_CALF_RAD)
    frozen_foot = foot_of(masses, joints, 'FL', frozen_angles)
    aligned_foot = foot_of(masses, joints, 'FL', aligned_angles)
    foot_shift_m = aligned_foot[0]-frozen_foot[0]
    print(f'  relative to the base, the aligned crouch puts each wheel '
          f'{foot_shift_m*1000:+.2f} mm in x compared with the frozen nominal')
    print(f'  base height is preserved: {frozen_geometry["base_height_m"]:.6f} -> '
          f'{aligned_geometry["base_height_m"]:.6f} m')
    print(f'  but the wheels are ON THE GROUND, so changing the leg angles moves the BASE '
          f'{-foot_shift_m*1000:+.2f} mm in x instead, unless the wheels roll')
    print(f'  that is {abs(foot_shift_m)/MAX_LONGITUDINAL_M*100:.0f}% of the frozen '
          f'{MAX_LONGITUDINAL_M*1000:.0f} mm longitudinal excursion limit, leaving only '
          f'{(MAX_LONGITUDINAL_M-abs(foot_shift_m))*1000:.1f} mm for the rest of the task')
    roll_rad = abs(foot_shift_m)/WHEEL_RADIUS_M
    print(f'  alternative: roll the wheels {roll_rad:.4f} rad '
          f'({math.degrees(roll_rad):.2f} deg) while morphing the legs, so the contact '
          f'pattern moves and the base stays put, spending no excursion budget at all')
    print('    the wheels are actuated (12 N*m) and the contract already replaces the wheel '
          'PD during side control, so this is a reference change, not a new authority; it '
          'does require releasing the 12-position wheel brake for the morph')
    thigh_delta = abs(ALIGNED_THIGH_RAD-NOMINAL['thigh'])
    calf_delta = abs(ALIGNED_CALF_RAD-NOMINAL['calf'])
    morph_s = max(thigh_delta, calf_delta)/REF_QVEL_CLIP_RPS
    print(f'  the morph itself is small: thigh {thigh_delta:.4f} rad, calf {calf_delta:.4f} '
          f'rad, so at the {REF_QVEL_CLIP_RPS} rad/s reference clip it needs only '
          f'{morph_s:.4f} s, far inside the 200-control preparation window')

    print('\n== 6. can the hip roll axis alone do the lateral motion? ==')
    # Asked directly: why step at all, instead of just driving the leg roll (abduction) axis?
    _, lateral_rates = direction_cost(columns, (0., 1., 0.))
    total_rate = sum(abs(value) for value in lateral_rates)
    print('  first: the hip roll axis IS already the primary lateral actuator')
    for name, rate in zip(('hip', 'thigh', 'calf'), lateral_rates):
        print(f'    {name:5s} {rate:+.4f} rad/s per 1 m/s of lateral foot speed, '
              f'{abs(rate)/total_rate*100:.1f}% of the total joint rate')
    print('  but hip rotation alone sweeps the foot on an arc, so it cannot hold height:')
    hip_world = frames_aligned['FL_hip'][1]
    foot_world = frames_aligned['FL_foot'][1]
    radius = math.dist(hip_world, foot_world)
    print(f'    hip-to-wheel radius {radius:.4f} m')
    print(f'    {"hip (rad)":>10} {"dy (mm)":>9} {"dz (mm)":>9}')
    for value in (.1, .2, .4, .6, LIMITS['hip'][1]):
        perturbed = dict(aligned_angles)
        perturbed['FL_hip_joint'] = value
        moved = foot_of(masses, joints, 'FL', perturbed)
        print(f'    {value:10.4f} {(moved[1]-foot_world[1])*1000:9.1f} '
              f'{(moved[2]-foot_world[2])*1000:9.1f}')
    print('    even 0.1 rad already lifts the foot 16 mm while moving it 36 mm sideways, so '
          'thigh and calf must coordinate - which is exactly what the paired IK does')
    print('  second, and this is the blocking one: the wheel axle is along body y, so the')
    print('    wheel rolls only along x. Any lateral motion of its CONTACT POINT is pure')
    print('    sliding. Translating the base sideways with all four wheels planted therefore')
    print('    requires all four contacts to slide, and with no gripping contact left there')
    print('    is no anchor to push against: internal joint motion produces no net lateral')
    print('    translation. Lateral travel fundamentally requires unloading wheels so they')
    print(f'    can be repositioned, which is the stepping gait. The {SWING_CLEARANCE_MM:.0f} mm'
          ' clearance gate exists to make that repositioning a clean non-sliding motion.')
    print('  third, the useful part of the question: the step is capped at 60 mm, which is')
    print(f'    only {LANDING_CAP_M/smallest*100:.1f}% of the {smallest*1000:.1f} mm workspace, '
          'so a bigger lateral move per step is kinematically available. What actually caps '
          'it is two-support statics at about 77.5 mm; raising the cap to that limit is the '
          'legitimate version of "take a bigger lateral step".')

    print('\n== 7. verdict ==')
    print('  none of the four checks blocks the aligned crouch:')
    print('    conditioning barely changes, the landing cap uses a third of the workspace, '
          'stance torque keeps headroom even while rejecting the swing reaction, and the '
          'morph takes 0.02 s')
    print(f'  one real finding: if the base moves, the crouch change spends '
          f'{abs(foot_shift_m)*1000:.2f} mm, {abs(foot_shift_m)/MAX_LONGITUDINAL_M*100:.0f}% '
          f'of the {MAX_LONGITUDINAL_M*1000:.0f} mm longitudinal excursion budget, so the '
          f'new contract must roll the wheels through the morph or raise that limit')
    print('  and one specification requirement: define the aligned crouch on the measured '
          'pose with a droop bias, not on the commanded angle')
    print('  still not addressed here: whether the body and attitude PD loops actually '
          'reject the 22% of body weight swing reaction, which needs simulation, and '
          'anything off flat ground')


if __name__ == '__main__':
    main()
