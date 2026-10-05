"""Motion optimisation for the two-support lateral gait: what is actually reachable?

Closes the open check from docs/ad_lateral_gait_review_20261004.md section 7. The review
showed the foot velocity profile is the only first-order lever left, and left the real
foot-speed bound as an open question behind a planar two-link estimate. This script
computes the exact foot Jacobian, leg mass matrix and gravity torque from the URDF, then
optimises the gait timing against them.

Pure arithmetic over the URDF tree and the frozen contract constants: no MuJoCo, no model
construction, no physics, no model load, no fitting. Kinematics helpers are reused from
posture_statics_20261004, which validates them against the saved C35 pre-state.

Hard limits are never relaxed: compiled joint ranges, leg |qvel| <= 18 rad/s, 80 N*m leg
and 12 N*m wheel actuators, 12 mm swing clearance, the 60 mm landing cap, the 30 mm
longitudinal excursion limit, 100 Hz control and no flight. Optimised are only reference
design parameters: the swing velocity profile, peak foot speed, lift height, step length
and the two all-four phase overlaps.
"""
from __future__ import annotations

import math

from posture_statics_20261004 import (
    LANDING_CAP_M,
    LEGS,
    PATCH_HALF_EXTENT_M,
    G,
    angle_vector,
    forward,
    parse,
    perpendicular_unit,
    pose_geometry,
    uniform,
)

# --- frozen hard limits (contract35.json, src/wheel_legged_control/d1/model.py) ---------
HARD_LEG_QVEL_RPS = 18.0          # native guard, terminates the campaign
LEG_TORQUE_NM = 80.0              # compiled actuator limit, never increased
SWING_CLEARANCE_M = .012          # whole-collision-shape minimum Z during horizontal motion
CONTROL_DT_S = .01                # 100 Hz control, 5 native substeps
# --- frozen reference clips: design parameters, but the frozen values are kept for ref --
REF_QVEL_CLIP_RPS = 4.0
REF_QACC_CLIP_RPS2 = 60.0
FROZEN_PEAK_FOOT_MPS = .16
# --- frozen phase durations that establish contact; treated as fixed, sensitivity shown -
TRANSFER_S, DWELL_S, LOAD_S = .06, .08, .06
FROZEN_LIFT_S, FROZEN_LOWER_S, FROZEN_LIFT_HEIGHT_M = .12, .12, .035
QUINTIC_PEAK_OVER_MEAN = 1.875
TARGET_FLOOR_MPS, TARGET_GOAL_MPS = .020, .040
ALIGNED_THIGH_RAD, ALIGNED_CALF_RAD = .731310, -1.502261
CLEARANCE_MARGIN_M = .003         # lift to clearance + margin instead of the frozen 35 mm


def jacobian(masses, joints, leg, angles, step=1e-7):
    """Exact-to-roundoff foot Jacobian d(foot world xyz)/d(hip, thigh, calf), central diff."""
    identity = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    columns = []
    for name in ('hip', 'thigh', 'calf'):
        key = f'{leg}_{name}_joint'
        shifted = []
        for sign in (+1, -1):
            perturbed = dict(angles)
            perturbed[key] = perturbed[key]+sign*step
            frames, _, _, _ = forward(masses, joints, perturbed, (0., 0., 0.), identity)
            shifted.append(frames[f'{leg}_foot'][1])
        columns.append(tuple((shifted[0][axis]-shifted[1][axis])/(2*step) for axis in range(3)))
    return columns                                     # columns[j][axis]


def solve3(columns, target):
    """Solve J q = target for a 3x3 J given as columns; Cramer's rule, no numpy."""
    def determinant(a, b, c):
        return (a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])
                + a[2]*(b[0]*c[1]-b[1]*c[0]))
    base = determinant(*columns)
    if abs(base) < 1e-12:
        raise ValueError('foot Jacobian is singular at this pose')
    out = []
    for index in range(3):
        replaced = list(columns)
        replaced[index] = target
        out.append(determinant(*replaced)/base)
    return tuple(out), base


def direction_cost(columns, direction):
    """Largest |joint rate| needed per unit foot speed along a unit direction."""
    rates, _ = solve3(columns, direction)
    return max(abs(rate) for rate in rates), rates


def leg_mass_matrix(masses, joints, leg, angles, step=1e-6):
    """3x3 mass matrix of the isolated swing chain (hip, thigh, calf) about its own joints.

    Translational part from finite-difference COM Jacobians, rotational part from the
    URDF inertia tensors rotated into the world frame. This is the swing leg's own
    inertia with the base held fixed: it ignores base coupling, so it is an estimate of
    the torque a swing acceleration needs, not the full coupled inertia the controller
    uses. Flagged as such in the report.
    """
    identity = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    chain = [f'{leg}_hip', f'{leg}_thigh', f'{leg}_calf', f'{leg}_foot']
    keys = [f'{leg}_{name}_joint' for name in ('hip', 'thigh', 'calf')]
    frames0, _, _, _ = forward(masses, joints, angles, (0., 0., 0.), identity)

    def com_world(frames, link):
        rotation, position = frames[link]
        local = masses[link][1]
        return tuple(position[axis]+sum(rotation[axis][k]*local[k] for k in range(3))
                     for axis in range(3))

    linear, angular = {}, {}
    for index, key in enumerate(keys):
        shifted = []
        for sign in (+1, -1):
            perturbed = dict(angles)
            perturbed[key] = perturbed[key]+sign*step
            frames, _, _, _ = forward(masses, joints, perturbed, (0., 0., 0.), identity)
            shifted.append(frames)
        for link in chain:
            a, b = com_world(shifted[0], link), com_world(shifted[1], link)
            linear.setdefault(link, [None]*3)[index] = tuple(
                (a[axis]-b[axis])/(2*step) for axis in range(3))
            # angular velocity Jacobian column: the joint axis in world, for links at or
            # below this joint; zero for links above it.
            below = chain.index(link) >= index
            axis_world = (0., 0., 0.)
            if below:
                joint = next(row for row in joints if row['name'] == key)
                rotation = frames0[joint['parent']][1] and frames0[joint['parent']][0]
                local = matmul_rpy(rotation, joint['rpy'])
                axis_world = tuple(sum(local[axis][k]*joint['axis'][k] for k in range(3))
                                   for axis in range(3))
            angular.setdefault(link, [None]*3)[index] = axis_world
    matrix = [[0.]*3 for _ in range(3)]
    for link in chain:
        mass = masses[link][0]
        inertia_world = world_inertia(masses, frames0, link, joints)
        for i in range(3):
            for j in range(3):
                matrix[i][j] += mass*sum(linear[link][i][a]*linear[link][j][a]
                                         for a in range(3))
                matrix[i][j] += sum(angular[link][i][a]*inertia_world[a][b]
                                    * angular[link][j][b]
                                    for a in range(3) for b in range(3))
    return matrix


def matmul_rpy(rotation, rpy):
    from posture_statics_20261004 import matmul, rpy_matrix
    return matmul(rotation, rpy_matrix(*rpy))


def world_inertia(masses, frames, link, joints):
    """Rotate the URDF inertia tensor of one link into the world frame."""
    del masses, joints
    tensor = URDF_INERTIA[link]
    rotation = frames[link][0]
    product = [[sum(rotation[i][k]*tensor[k][m] for k in range(3)) for m in range(3)]
               for i in range(3)]
    return [[sum(product[i][m]*rotation[j][m] for m in range(3)) for j in range(3)]
            for i in range(3)]


def read_inertia():
    """URDF inertia tensors about each link's own COM, in the link frame."""
    import xml.etree.ElementTree as ET

    from posture_statics_20261004 import URDF
    out = {}
    for link in ET.parse(URDF).getroot().findall('link'):
        inertial = link.find('inertial')
        if inertial is None:
            continue
        values = inertial.find('inertia').attrib
        ixx, ixy, ixz = (float(values[key]) for key in ('ixx', 'ixy', 'ixz'))
        iyy, iyz, izz = (float(values[key]) for key in ('iyy', 'iyz', 'izz'))
        out[link.get('name')] = [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]
    return out


URDF_INERTIA = read_inertia()


def gravity_torque(masses, joints, leg, angles, step=1e-6):
    """Static joint torque holding the swing chain against gravity, from dU/dq."""
    identity = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    chain = [f'{leg}_hip', f'{leg}_thigh', f'{leg}_calf', f'{leg}_foot']
    out = []
    for name in ('hip', 'thigh', 'calf'):
        key = f'{leg}_{name}_joint'
        heights = []
        for sign in (+1, -1):
            perturbed = dict(angles)
            perturbed[key] = perturbed[key]+sign*step
            frames, _, _, _ = forward(masses, joints, perturbed, (0., 0., 0.), identity)
            total = 0.
            for link in chain:
                rotation, position = frames[link]
                local = masses[link][1]
                z = position[2]+sum(rotation[2][k]*local[k] for k in range(3))
                total += masses[link][0]*G*z
            heights.append(total)
        out.append((heights[0]-heights[1])/(2*step))
    return tuple(out)


def minimum_time_s(distance_m, speed_cap_mps, accel_cap_mps2):
    """Minimum traverse time under symmetric speed and acceleration caps."""
    if distance_m <= 0:
        return 0.
    if not math.isfinite(accel_cap_mps2):
        return distance_m/speed_cap_mps
    if not math.isfinite(speed_cap_mps):
        return 2*math.sqrt(distance_m/accel_cap_mps2)
    if distance_m <= speed_cap_mps**2/accel_cap_mps2:
        return 2*math.sqrt(distance_m/accel_cap_mps2)          # triangular
    return distance_m/speed_cap_mps+speed_cap_mps/accel_cap_mps2   # trapezoidal


def quintic_time_s(distance_m, peak_speed_mps_value):
    """Duration of the frozen quintic smoothstep for a given peak reference speed."""
    return QUINTIC_PEAK_OVER_MEAN*distance_m/peak_speed_mps_value


def quantise_s(seconds):
    """Phases are consumed one 0.01 s control at a time, so round up to whole controls."""
    return math.ceil(seconds/CONTROL_DT_S-1e-9)*CONTROL_DT_S


def peak_speed_mps(distance_m, speed_cap_mps, accel_cap_mps2):
    if not math.isfinite(accel_cap_mps2):
        return speed_cap_mps
    if distance_m <= speed_cap_mps**2/accel_cap_mps2:
        return math.sqrt(distance_m*accel_cap_mps2)
    return speed_cap_mps


def gait(step_m, *, horizontal_s, lift_s, lower_s, overlap_s=0., probe_s=0., wait_s=0.):
    """One pair's swing duration from explicit, control-quantised phase durations."""
    phases = {'transfer': quantise_s(TRANSFER_S), 'lift': quantise_s(lift_s),
              'horizontal': quantise_s(horizontal_s), 'lower': quantise_s(lower_s),
              'probe': quantise_s(probe_s), 'dwell': quantise_s(DWELL_S),
              'load': quantise_s(LOAD_S), 'wait': quantise_s(wait_s)}
    overlap = quantise_s(overlap_s)
    seconds = sum(phases.values())-overlap
    air_s = phases['lift']+phases['horizontal']+phases['lower']+phases['probe']+phases['dwell']
    speed = step_m/(2*seconds)
    drift_m = LATERAL_PROJECTION*speed*air_s/2        # mid-swing-centred peak drift
    return {'pair_s': seconds, 'cycle_s': 2*seconds, 'speed_mps': speed, 'air_s': air_s,
            'phases': phases, 'overlap_s': overlap, 'drift_m': drift_m,
            'statics_ok': drift_m <= PATCH_HALF_EXTENT_M,
            'statics_margin': 1-drift_m/PATCH_HALF_EXTENT_M}


def largest_step_m(cap_m, build):
    """Largest step whose mid-swing drift still fits the contact-patch moment capacity."""
    if build(cap_m)['statics_ok']:
        return cap_m
    low, high = .002, cap_m
    for _ in range(200):
        middle = (low+high)/2
        low, high = (middle, high) if build(middle)['statics_ok'] else (low, middle)
    return low


# measured projection of a pure lateral body motion onto the support-diagonal normal
LOAD_WEIGHTED_M = ((-7.830793881720057, -4.46319798308953),
                   (-7.830768337013101, -4.924156600585739),
                   (-8.205612854620423, -4.475573401433002),
                   (-8.205617395689757, -4.9365798791605044))
LATERAL_PROJECTION = abs(perpendicular_unit(LOAD_WEIGHTED_M[1], LOAD_WEIGHTED_M[2])[0][1])


def main() -> None:
    masses, joints = parse()
    angles = angle_vector(uniform(ALIGNED_THIGH_RAD, ALIGNED_CALF_RAD))
    geometry = pose_geometry(masses, joints, ALIGNED_THIGH_RAD, ALIGNED_CALF_RAD)

    print('== 1. exact foot Jacobian at the aligned crouch ==')
    print(f'  pose: hip 0, thigh {ALIGNED_THIGH_RAD:.6f}, calf {ALIGNED_CALF_RAD:.6f} rad, '
          f'base height {geometry["base_height_m"]:.6f} m')
    costs = {}
    for leg in LEGS:
        columns = jacobian(masses, joints, leg, angles)
        lateral_cost, lateral_rates = direction_cost(columns, (0., 1., 0.))
        vertical_cost, vertical_rates = direction_cost(columns, (0., 0., 1.))
        longitudinal_cost, _ = direction_cost(columns, (1., 0., 0.))
        costs[leg] = {'lateral': lateral_cost, 'vertical': vertical_cost}
        print(f'  {leg}: per 1 m/s of foot motion the worst joint rate is '
              f'{lateral_cost:7.4f} rad/s lateral, {vertical_cost:7.4f} rad/s vertical, '
              f'{longitudinal_cost:7.4f} rad/s longitudinal')
        print(f'       lateral joint rates (hip, thigh, calf) = '
              f'({lateral_rates[0]:+.4f}, {lateral_rates[1]:+.4f}, {lateral_rates[2]:+.4f})'
              f'   vertical = ({vertical_rates[0]:+.4f}, {vertical_rates[1]:+.4f}, '
              f'{vertical_rates[2]:+.4f})')
    worst_lateral = max(costs[leg]['lateral'] for leg in LEGS)
    worst_vertical = max(costs[leg]['vertical'] for leg in LEGS)

    print('\n== 2. foot speed and acceleration envelopes ==')
    envelopes = {}
    for label, qvel, qacc in (('frozen reference clips', REF_QVEL_CLIP_RPS, REF_QACC_CLIP_RPS2),
                              ('native hard gate only', HARD_LEG_QVEL_RPS, None)):
        lateral_speed = qvel/worst_lateral
        vertical_speed = qvel/worst_vertical
        lateral_accel = None if qacc is None else qacc/worst_lateral
        vertical_accel = None if qacc is None else qacc/worst_vertical
        envelopes[label] = (lateral_speed, lateral_accel, vertical_speed, vertical_accel)
        print(f'  {label}: |qvel|<={qvel} rad/s -> lateral foot {lateral_speed:.4f} m/s, '
              f'vertical {vertical_speed:.4f} m/s')
        if qacc is not None:
            print(f'    |qacc|<={qacc} rad/s^2 -> lateral foot {lateral_accel:.4f} m/s^2, '
                  f'vertical {vertical_accel:.4f} m/s^2')
    print(f'  the frozen peak reference foot speed {FROZEN_PEAK_FOOT_MPS} m/s uses '
          f'{FROZEN_PEAK_FOOT_MPS/envelopes["frozen reference clips"][0]*100:.1f}% of the '
          f'reference clip envelope and '
          f'{FROZEN_PEAK_FOOT_MPS/envelopes["native hard gate only"][0]*100:.1f}% of the '
          f'native hard gate')

    print('\n== 3. does actuator torque allow it? ==')
    matrix = leg_mass_matrix(masses, joints, 'FL', angles)
    static = gravity_torque(masses, joints, 'FL', angles)
    columns = jacobian(masses, joints, 'FL', angles)
    print('  isolated swing-chain mass matrix (hip, thigh, calf), kg*m^2:')
    for row in matrix:
        print('    '+'  '.join(f'{value:+10.6f}' for value in row))
    print(f'  gravity hold torque {tuple(round(value, 4) for value in static)} N*m, '
          f'worst |tau_g| {max(abs(value) for value in static):.4f} N*m of {LEG_TORQUE_NM}')
    for label, accel in (('frozen clip equivalent',
                          envelopes['frozen reference clips'][1]),
                         ('10x that acceleration', envelopes['frozen reference clips'][1]*10)):
        rates, _ = solve3(columns, (0., accel, 0.))
        torque = [sum(matrix[i][j]*rates[j] for j in range(3))+static[i] for i in range(3)]
        worst = max(abs(value) for value in torque)
        print(f'  {label}: foot lateral accel {accel:.4f} m/s^2 -> worst joint torque '
              f'{worst:.3f} N*m, {worst/LEG_TORQUE_NM*100:.1f}% of the 80 N*m limit')
    print('  caveat: this is the swing chain\'s own inertia with the base held fixed, not '
          'the coupled 6x6 inertia the controller uses; it bounds the swing term only')

    print('\n== 4. the frozen quintic is 1.875x slower than a minimum-time traverse ==')
    for step_m in (.030, LANDING_CAP_M):
        quintic = quintic_time_s(step_m, FROZEN_PEAK_FOOT_MPS)
        fastest = minimum_time_s(step_m, FROZEN_PEAK_FOOT_MPS, math.inf)
        print(f'  step {step_m*1000:4.1f} mm at the same 0.16 m/s peak: quintic '
              f'{quintic:.4f} s vs minimum-time {fastest:.4f} s ({quintic/fastest:.3f}x)')
    print('  the frozen quintic also never uses its acceleration budget: a profile that '
          'does is strictly faster at the same peak speed')

    print('\n== 5. staged optimisation from the frozen baseline ==')
    frozen_vertical_peak = QUINTIC_PEAK_OVER_MEAN*FROZEN_LIFT_HEIGHT_M/FROZEN_LIFT_S
    short_lift_m = SWING_CLEARANCE_M+CLEARANCE_MARGIN_M
    lateral_accel = REF_QACC_CLIP_RPS2/worst_lateral
    lateral_speed = REF_QVEL_CLIP_RPS/worst_lateral
    vertical_accel = REF_QACC_CLIP_RPS2/worst_vertical
    vertical_speed = REF_QVEL_CLIP_RPS/worst_vertical
    print(f'  frozen quintic vertical peak = 1.875*{FROZEN_LIFT_HEIGHT_M}/{FROZEN_LIFT_S} = '
          f'{frozen_vertical_peak:.4f} m/s, inside the {vertical_speed:.4f} m/s envelope')

    def phases_of(horizontal_s, lift_s, overlap_s=0.):
        return {'horizontal_s': horizontal_s, 'lift_s': lift_s, 'lower_s': lift_s,
                'overlap_s': overlap_s}

    short_lift_s = quintic_time_s(short_lift_m, frozen_vertical_peak)
    fast_lift_s = minimum_time_s(short_lift_m, frozen_vertical_peak, vertical_accel)
    fastest_lift_s = minimum_time_s(short_lift_m, vertical_speed, vertical_accel)
    # Stage 0: the frozen gait, step limited by the frozen 0.975 s landing lead (see
    # posture_statics_20261004 section 8: statics caps the step at 46.7 mm there).
    stages = [
        (('frozen baseline: quintic, 0.16 m/s peak, 35 mm lift, no overlap, step capped '
          'by the 0.975 s landing lead'), .0467,
         {'horizontal_s': quintic_time_s(.0467, FROZEN_PEAK_FOOT_MPS),
          'lift_s': FROZEN_LIFT_S, 'lower_s': FROZEN_LOWER_S}),
        ('fix the landing lead so the 60 mm cap is reachable', LANDING_CAP_M,
         {'horizontal_s': quintic_time_s(LANDING_CAP_M, FROZEN_PEAK_FOOT_MPS),
          'lift_s': FROZEN_LIFT_S, 'lower_s': FROZEN_LOWER_S}),
        ((f'lift only to clearance + 3 mm ({short_lift_m*1000:.0f} mm) at the same '
          'vertical peak'), LANDING_CAP_M,
         phases_of(quintic_time_s(LANDING_CAP_M, FROZEN_PEAK_FOOT_MPS), short_lift_s)),
        ('overlap load with the next pair transfer', LANDING_CAP_M,
         phases_of(quintic_time_s(LANDING_CAP_M, FROZEN_PEAK_FOOT_MPS), short_lift_s,
                   LOAD_S)),
        ('minimum-time profiles at the frozen 0.16 m/s peak', LANDING_CAP_M,
         phases_of(minimum_time_s(LANDING_CAP_M, FROZEN_PEAK_FOOT_MPS, lateral_accel),
                   fast_lift_s, LOAD_S)),
        ('raise the peak foot speed to the 4 rad/s reference-clip envelope', LANDING_CAP_M,
         phases_of(minimum_time_s(LANDING_CAP_M, lateral_speed, lateral_accel),
                   fastest_lift_s, LOAD_S)),
    ]
    baseline = None
    previous = None
    optimised = None
    for label, step_m, config in stages:
        result = gait(step_m, **config)
        if baseline is None:
            baseline, previous = result['speed_mps'], result['speed_mps']
            print(f'  {label}\n    step {step_m*1000:4.1f} mm, T_pair {result["pair_s"]:.3f} s'
                  f' -> {result["speed_mps"]:.5f} m/s   statics margin '
                  f'{result["statics_margin"]*100:+.0f}%')
            continue
        print(f'  + {label}')
        print(f'    step {step_m*1000:4.1f} mm, T_pair {result["pair_s"]:.3f} s -> '
              f'{result["speed_mps"]:.5f} m/s  ({(result["speed_mps"]/previous-1)*100:+5.1f}%, '
              f'cumulative {(result["speed_mps"]/baseline-1)*100:+6.1f}%)   statics margin '
              f'{result["statics_margin"]*100:+.0f}% '
              f'{"OK" if result["statics_ok"] else "FAILS"}')
        previous = result['speed_mps']
        optimised = (step_m, config, result)

    print('\n== 6. the optimised gait in detail, every hard limit kept ==')
    step_m, config, result = optimised
    phases = result['phases']
    print(f'  step {step_m*1000:.1f} mm, T_pair {result["pair_s"]:.3f} s, cycle '
          f'{result["cycle_s"]:.3f} s, lateral speed {result["speed_mps"]:.5f} m/s')
    print('  control-quantised phases: '+', '.join(
        f'{name} {value*100:.0f}' for name, value in phases.items() if value > 0)
        + f' (overlap -{result["overlap_s"]*100:.0f} controls of 0.01 s)')
    horizontal_peak = peak_speed_mps(step_m, lateral_speed, lateral_accel)
    rates, _ = solve3(columns, (0., horizontal_peak, 0.))
    accel_rates, _ = solve3(columns, (0., lateral_accel, 0.))
    torque = max(abs(sum(matrix[i][j]*accel_rates[j] for j in range(3))+static[i])
                 for i in range(3))
    print(f'  peak foot lateral speed {horizontal_peak:.4f} m/s -> worst joint rate '
          f'{max(abs(rate) for rate in rates):.4f} rad/s of the {REF_QVEL_CLIP_RPS} clip and '
          f'the {HARD_LEG_QVEL_RPS} hard gate')
    print(f'  peak swing torque {torque:.2f} N*m, {torque/LEG_TORQUE_NM*100:.1f}% of the '
          f'{LEG_TORQUE_NM} N*m actuator limit')
    print(f'  mid-swing COM drift {result["drift_m"]*1000:.2f} mm of the '
          f'{PATCH_HALF_EXTENT_M*1000:.2f} mm patch capacity, margin '
          f'{result["statics_margin"]*100:.0f}%')
    swing_mass = sum(masses[f'FL_{part}'][0] for part in ('thigh', 'calf', 'foot'))
    body_weight_n = sum(mass for mass, _ in masses.values())*G
    reaction_n = 2*swing_mass*lateral_accel/2      # two legs, COM accelerates ~half the foot
    print(f'  swing reaction on the body: 2 x {swing_mass:.3f} kg of moving links, roughly '
          f'{reaction_n:.0f} N lateral at peak, {reaction_n/body_weight_n*100:.0f}% of body '
          f'weight - a first-order disturbance this analysis cannot clear')

    print('\n== 7. sensitivity ==')
    for wait_s in (0., .05, .1, .2, .4):
        row = gait(step_m, **{**config, 'wait_s': wait_s})
        print(f'  contact-admission wait {wait_s*100:3.0f} controls/pair -> '
              f'{row["speed_mps"]:.5f} m/s  '
              f'{"meets" if row["speed_mps"] >= TARGET_GOAL_MPS else "below"} the '
              f'{TARGET_GOAL_MPS} goal')
    for cap_m in (LANDING_CAP_M, .080, .090):
        def build(value, base=config):
            adjusted = {**base,
                        'horizontal_s': minimum_time_s(value, lateral_speed, lateral_accel)}
            return gait(value, **adjusted)
        best = largest_step_m(cap_m, build)
        row = build(best)
        print(f'  landing cap {cap_m*1000:.0f} mm -> step {best*1000:5.1f} mm '
              f'({"cap" if best >= cap_m-1e-9 else "statics"} binds), '
              f'{row["speed_mps"]:.5f} m/s')

    print('\n== 8. verdict ==')
    print(f'  with every frozen hard limit kept and only reference design parameters '
          f'changed, the optimised gait reaches {result["speed_mps"]:.5f} m/s')
    print(f'    that is {result["speed_mps"]/baseline:.2f}x the frozen baseline '
          f'{baseline:.5f} m/s, {result["speed_mps"]/TARGET_FLOOR_MPS:.1f}x the '
          f'{TARGET_FLOOR_MPS} m/s engineering floor and '
          f'{result["speed_mps"]/TARGET_GOAL_MPS:.2f}x the {TARGET_GOAL_MPS} m/s '
          f'development goal')
    print('  the binding constraint moves to the 60 mm landing cap; foot actuators, joint '
          'rates and torque all retain large margin')
    print('  this overturns the earlier conclusion that the development goal is '
          'structurally unreachable: that conclusion assumed the frozen quintic shape and '
          'the frozen 0.16 m/s peak, both of which are reference choices, not limits')
    print('  NOT verified here: closed-loop body and attitude rejection of the swing '
          'reaction above, IK conditioning at the aligned crouch, contact-gate admission '
          'at the 15 mm lift, behaviour on anything but flat ground, and the coupled 6x6 '
          'inertia. Every number is a reference-level bound, not a simulation.')


if __name__ == '__main__':
    main()
