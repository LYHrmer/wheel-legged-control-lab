"""Pure URDF kinematics/mass check: can posture align the COM with the support diagonals?

Follow-up to docs/ad_lateral_gait_review_20261004.md section 7. The review found the
whole-body COM sits 21.66 mm forward of the wheel contact rectangle centre, so neither
diagonal passes through it and diagonal-pair support is statically infeasible without a
per-pair body shift that costs 0.5096 s. This script asks whether a different nominal
crouch removes that offset instead, which would restore the speed the shift consumes.

Pure arithmetic over the URDF tree: no MuJoCo, no model construction, no physics, no
model load, no fitting. Validated against the saved pre-state of the single C35 run.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

URDF = Path(__file__).resolve().parents[3]/'src/wheel_legged_control/d1/assets/urdf/robot.urdf'
LEGS = ('FL', 'FR', 'RL', 'RR')
JOINTS = ('hip', 'thigh', 'calf', 'foot')
WHEEL_RADIUS_M = .087

# --- saved pre-state of the single C35 side control (native_contact_failure_0000.json) --
BASE_XYZ = (-7.9992424705267835, -4.6999753872327155, 0.46076393270250626)
BASE_QUAT = (0.9999997843134548, -0.000191637347074307, 0.0006020380046925411,
             -0.0001794391620904804)
MEASURED_Q = {  # hip, thigh, calf, foot per leg, in model.py D1_JOINT_NAMES order
    'FL': (-0.00725027841290074, 0.7810389955212396, -1.4617454672860377, -0.0006597232090642642),
    'FR': (0.008228237422574766, 0.7796747622497585, -1.4601464577152419, -0.0003725008667202607),
    'RL': (-0.007979649873122038, 0.7788028014315835, -1.4585954229356748, -0.00037791537249894276),
    'RR': (0.008369396457390736, 0.7790003954005231, -1.4596074535394516, 0.00026295271398291397),
}
MEASURED_COM_M = (-7.996515049003027, -4.699912570342095, 0.38758374348429603)
# Three different contact geometries the record keeps, per clarifications35_03.
MEASURED_FEET_M = ((-7.830859687849413, -4.482554610319743, 0.08751091345622383),
                   (-7.830839722873551, -4.9174429837761355, 0.08751103278022476),
                   (-8.205742334335339, -4.482696239552716, 0.08758220934590663),
                   (-8.20572236213381, -4.917259807287801, 0.0875824096849118))
MEASURED_RIM_CONTACT_M = ((-7.830779743774686, -4.463196398561754),
                          (-7.830752426854562, -4.9367834637916825),
                          (-8.20556744122596, -4.463397910908332),
                          (-8.205603917946572, -4.936580236718218))
# positive-load-weighted wheel contacts at native 1019, the last substep with all four loaded
MEASURED_LOAD_WEIGHTED_M = ((-7.830793881720057, -4.46319798308953),
                            (-7.830768337013101, -4.924156600585739),
                            (-8.205612854620423, -4.475573401433002),
                            (-8.205617395689757, -4.9365798791605044))
PATCH_HALF_EXTENT_M = .01242   # measured perpendicular half-extent of a saved wheel patch
# compiled limited joint ranges (src/wheel_legged_control/d1/model.py)
LIMITS = {'hip': (-0.785398, 0.785398), 'thigh': (-1.8326, 3.40339), 'calf': (-2.775, -0.855)}
NOMINAL = {'hip': 0., 'thigh': .8, 'calf': -1.5, 'foot': 0.}

# --- frozen C35 timing parameters, to re-price the result (contract35.json) ------------
OVERHEAD_S, K_S_PER_M, LANDING_CAP_M = .44, 11.71875, .060
QUINTIC_PEAK_ACCEL, REFERENCE_ZMP_M, G = 10/math.sqrt(3), .015, 9.81
LOWER_PLUS_DWELL_PLUS_LOAD_S = .12+.08+.06
TRANSFER_S, LIFT_S, LOWER_S = .06, .12, .12
FROZEN_LEAD_S = .975            # sol35/core35.py landing_xy35, latched once at lift -> horizontal
MAX_LONGITUDINAL_M = .030       # contract35.json task.continuous.max_abs_longitudinal_excursion_m


def matmul(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def apply(rotation, vector):
    return tuple(sum(rotation[i][k]*vector[k] for k in range(3)) for i in range(3))


def rpy_matrix(roll, pitch, yaw):
    """URDF fixed-axis convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
            [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
            [-sp, cp*sr, cp*cr]]


def axis_matrix(axis, angle):
    norm = math.sqrt(sum(value*value for value in axis))
    x, y, z = (value/norm for value in axis)
    c, s, t = math.cos(angle), math.sin(angle), 1-math.cos(angle)
    return [[t*x*x+c, t*x*y-s*z, t*x*z+s*y],
            [t*x*y+s*z, t*y*y+c, t*y*z-s*x],
            [t*x*z-s*y, t*y*z+s*x, t*z*z+c]]


def quaternion_matrix(w, x, y, z):
    return [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]


def parse():
    root = ET.parse(URDF).getroot()
    masses = {}
    for link in root.findall('link'):
        inertial = link.find('inertial')
        if inertial is None:
            continue
        origin = inertial.find('origin')
        masses[link.get('name')] = (
            float(inertial.find('mass').get('value')),
            tuple(float(v) for v in origin.get('xyz').split()))
    joints = []
    for joint in root.findall('joint'):
        origin = joint.find('origin')
        axis = joint.find('axis')
        joints.append({
            'name': joint.get('name'), 'type': joint.get('type'),
            'parent': joint.find('parent').get('link'),
            'child': joint.find('child').get('link'),
            'xyz': tuple(float(v) for v in origin.get('xyz').split()),
            'rpy': tuple(float(v) for v in origin.get('rpy').split()),
            'axis': None if axis is None or joint.get('type') == 'fixed'
                    else tuple(float(v) for v in axis.get('xyz').split())})
    return masses, joints


def forward(masses, joints, angles, base_xyz, base_rotation):
    """Return (link frames, whole COM, wheel-body centres)."""
    frames = {'base_link': (base_rotation, tuple(base_xyz))}
    pending = list(joints)
    while pending:
        progressed = False
        for joint in list(pending):
            if joint['parent'] not in frames:
                continue
            rotation, position = frames[joint['parent']]
            local = matmul(rotation, rpy_matrix(*joint['rpy']))
            offset = apply(rotation, joint['xyz'])
            if joint['axis'] is not None:
                local = matmul(local, axis_matrix(joint['axis'], angles[joint['name']]))
            frames[joint['child']] = (
                local, tuple(position[i]+offset[i] for i in range(3)))
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise RuntimeError('URDF tree is not connected')
    total, weighted = 0., [0., 0., 0.]
    for name, (mass, local_com) in masses.items():
        rotation, position = frames[name]
        world = apply(rotation, local_com)
        total += mass
        for i in range(3):
            weighted[i] += mass*(position[i]+world[i])
    com = tuple(value/total for value in weighted)
    wheels = tuple(frames[f'{leg}_foot'][1] for leg in LEGS)
    return frames, total, com, wheels


def angle_vector(per_leg):
    return {f'{leg}_{joint}_joint': per_leg[leg][index]
            for leg in LEGS for index, joint in enumerate(JOINTS)}


def uniform(thigh, calf, hip=0., foot=0.):
    return {leg: (hip, thigh, calf, foot) for leg in LEGS}


def perpendicular_unit(a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    length = math.hypot(dx, dy)
    return (-dy/length, dx/length), length


def diagonal_offset_m(com, contacts, support):
    anchor, far = contacts[support[0]], contacts[support[1]]
    unit, _ = perpendicular_unit(anchor, far)
    return abs((com[0]-anchor[0])*unit[0]+(com[1]-anchor[1])*unit[1])


def pose_geometry(masses, joints, thigh, calf):
    """Flat upright base; report COM/contact offsets and base height for one crouch."""
    identity = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    _, total, com, wheels = forward(masses, joints, angle_vector(uniform(thigh, calf)),
                                    (0., 0., 0.), identity)
    # Place the base so the lowest wheel bottom touches z=0, then read the base height.
    drop = min(wheel[2] for wheel in wheels)-WHEEL_RADIUS_M
    base_z = -drop
    contacts = tuple((wheel[0], wheel[1]) for wheel in wheels)
    centre = tuple(sum(contact[axis] for contact in contacts)/4 for axis in (0, 1))
    return {'total_mass_kg': total, 'base_height_m': base_z,
            'com_xy_m': (com[0], com[1]), 'contacts_xy_m': contacts,
            'contact_centre_xy_m': centre,
            'com_forward_of_centre_m': com[0]-centre[0],
            'com_lateral_of_centre_m': com[1]-centre[1],
            'foot_behind_hip_m': None,
            'diagonal_offsets_m': (diagonal_offset_m(com, contacts, (1, 2)),
                                   diagonal_offset_m(com, contacts, (0, 3)))}


def solve_aligned(masses, joints, target_height_m, bracket=(.2, 2.2)):
    """Find (thigh, calf) that zeroes the COM/contact x-offset at the given base height.

    Two scalar equations, two unknowns. For each candidate thigh angle the calf angle is
    chosen to hold the base height, then the residual x-offset is bisected to zero.
    """
    def calf_for_height(thigh):
        low, high = LIMITS['calf']
        for _ in range(200):
            middle = (low+high)/2
            height = pose_geometry(masses, joints, thigh, middle)['base_height_m']
            if height < target_height_m:
                low = middle          # straighter calf raises the base
            else:
                high = middle
        return (low+high)/2

    def residual(thigh):
        calf = calf_for_height(thigh)
        return pose_geometry(masses, joints, thigh, calf)['com_forward_of_centre_m'], calf

    low, high = bracket
    low_value, _ = residual(low)
    high_value, _ = residual(high)
    if low_value*high_value > 0:
        return None, low_value, high_value
    for _ in range(120):
        middle = (low+high)/2
        value, _ = residual(middle)
        if value*low_value > 0:
            low, low_value = middle, value
        else:
            high, high_value = middle, value
    thigh = (low+high)/2
    calf = calf_for_height(thigh)
    return (thigh, calf), low_value, high_value


def stepping_speed(extra_s):
    return LANDING_CAP_M/(2*(OVERHEAD_S+K_S_PER_M*LANDING_CAP_M+extra_s))


def shift_seconds(distance_m):
    return math.sqrt(QUINTIC_PEAK_ACCEL*distance_m/(REFERENCE_ZMP_M*G/MEASURED_COM_M[2]))


def main() -> None:
    masses, joints = parse()

    print('== 1. validate the URDF tree against the saved C35 pre-state ==')
    rotation = quaternion_matrix(*BASE_QUAT)
    _, total, com, wheels = forward(masses, joints, angle_vector(MEASURED_Q),
                                    BASE_XYZ, rotation)
    print(f'  total mass          computed {total:.8f} kg   saved 48.14686526 kg'
          f'   delta {abs(total-48.14686526):.2e}')
    for axis, name in enumerate('xyz'):
        print(f'  whole COM {name}         computed {com[axis]:+.6f} m   '
              f'saved {MEASURED_COM_M[axis]:+.6f} m   delta '
              f'{(com[axis]-MEASURED_COM_M[axis])*1000:+.3f} mm')
    print('  wheel-body centres against the controller\'s own feet_world_m:')
    for index, leg in enumerate(LEGS):
        got, want = wheels[index], MEASURED_FEET_M[index]
        print(f'    {leg} delta ({(got[0]-want[0])*1000:+.3f}, {(got[1]-want[1])*1000:+.3f}, '
              f'{(got[2]-want[2])*1000:+.3f}) mm')
    print('  saved rim contact slots are 40 mm apart, the cylinder length; their midpoint '
          'is the wheel plane, which is why rim points are not wheel centres')

    print('\n== 2. is the static offset robust to the contact definition? ==')
    weight = total*G
    for name, points in (('rim contact representative (used in the review)',
                          MEASURED_RIM_CONTACT_M),
                         ('wheel body centre (feet_world_m)',
                          tuple(point[:2] for point in MEASURED_FEET_M)),
                         ('positive-load-weighted (contract force geometry)',
                          MEASURED_LOAD_WEIGHTED_M)):
        centre = tuple(sum(point[axis] for point in points)/4 for axis in (0, 1))
        offsets = (diagonal_offset_m(MEASURED_COM_M, points, (1, 2)),
                   diagonal_offset_m(MEASURED_COM_M, points, (0, 3)))
        print(f'  {name}')
        print(f'    COM forward of contact centre {(MEASURED_COM_M[0]-centre[0])*1000:+6.2f} mm'
              f'   diagonal offsets {offsets[0]*1000:5.2f} / {offsets[1]*1000:5.2f} mm')
        print(f'    required moment {weight*offsets[0]:5.2f} / {weight*offsets[1]:5.2f} N*m'
              f'   vs patch capacity {weight*PATCH_HALF_EXTENT_M:5.2f} N*m'
              f'   deficit {max(offsets)/PATCH_HALF_EXTENT_M:.2f}x')

    print('\n== 3. where the forward offset comes from ==')
    identity = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    frames, _, _, _ = forward(masses, joints, angle_vector(uniform(.8, -1.5)),
                              (0., 0., 0.), identity)
    for leg in LEGS:
        hip, foot = frames[f'{leg}_hip'][1], frames[f'{leg}_foot'][1]
        print(f'  {leg}: hip x {hip[0]:+.4f} m, wheel x {foot[0]:+.4f} m, '
              f'wheel is {(hip[0]-foot[0])*1000:.1f} mm behind its hip')
    nominal = pose_geometry(masses, joints, .8, -1.5)
    hip_centre = sum(frames[f'{leg}_hip'][1][0] for leg in LEGS)/4
    behind = hip_centre-nominal['contact_centre_xy_m'][0]
    print(f'  hip rectangle centre is {hip_centre*1000:+.1f} mm of base_link and the wheels sit '
          f'{behind*1000:.1f} mm behind it, so the contact centre lands at '
          f'{nominal["contact_centre_xy_m"][0]*1000:+.1f} mm')
    print(f'  nominal crouch: base height {nominal["base_height_m"]:.6f} m, COM '
          f'{nominal["com_forward_of_centre_m"]*1000:+.2f} mm forward of the contact centre, '
          f'diagonal offsets {nominal["diagonal_offsets_m"][0]*1000:.2f} / '
          f'{nominal["diagonal_offsets_m"][1]*1000:.2f} mm')

    print('\n== 4. can a different crouch align the COM with both diagonals? ==')
    for target in (nominal['base_height_m'], .455, BASE_XYZ[2]):
        solution, low_value, high_value = solve_aligned(masses, joints, target)
        print(f'  target base height {target:.6f} m: residual at thigh 0.2 rad '
              f'{low_value*1000:+.2f} mm, at 2.2 rad {high_value*1000:+.2f} mm')
        if solution is None:
            print('    NO sign change in the scanned range: posture alone cannot align them')
            continue
        thigh, calf = solution
        aligned = pose_geometry(masses, joints, thigh, calf)
        limits_ok = all(LIMITS[name][0] <= value <= LIMITS[name][1]
                        for name, value in (('thigh', thigh), ('calf', calf)))
        print(f'    SOLUTION thigh {thigh:+.6f} rad, calf {calf:+.6f} rad  '
              f'(nominal {NOMINAL["thigh"]:+.3f} / {NOMINAL["calf"]:+.3f})')
        print(f'      base height {aligned["base_height_m"]:.6f} m, COM offset '
              f'{aligned["com_forward_of_centre_m"]*1000:+.4f} mm, diagonal offsets '
              f'{aligned["diagonal_offsets_m"][0]*1000:.4f} / '
              f'{aligned["diagonal_offsets_m"][1]*1000:.4f} mm')
        print(f'      compiled joint limits: {"all OK" if limits_ok else "VIOLATED"}'
              f'   stance y span '
              f'{abs(aligned["contacts_xy_m"][0][1]-aligned["contacts_xy_m"][1][1])*1000:.1f} mm'
              f' (nominal '
              f'{abs(nominal["contacts_xy_m"][0][1]-nominal["contacts_xy_m"][1][1])*1000:.1f} mm)')

    print('\n== 5. sensitivity: how precisely must the aligned crouch be held? ==')
    target = nominal['base_height_m']
    print(f'  {"thigh":>8} {"calf":>10} {"height":>9} {"COM-centre x":>13} {"diag off":>10} '
          f'{"static":>8}')
    for thigh in (.4, .6, .68, .70, .7313, .76, .8, .9, 1.0):
        low, high = LIMITS['calf']
        for _ in range(200):
            middle = (low+high)/2
            if pose_geometry(masses, joints, thigh, middle)['base_height_m'] < target:
                low = middle
            else:
                high = middle
        calf = (low+high)/2
        row = pose_geometry(masses, joints, thigh, calf)
        feasible = max(row['diagonal_offsets_m']) <= PATCH_HALF_EXTENT_M
        print(f'  {thigh:8.4f} {calf:10.5f} {row["base_height_m"]:9.5f} '
              f'{row["com_forward_of_centre_m"]*1000:+13.2f} '
              f'{row["diagonal_offsets_m"][0]*1000:10.2f} '
              f'{"OK" if feasible else "FAILS":>8}')
    near = [(thigh, pose_geometry(masses, joints, thigh,
                                  _calf_for_height(masses, joints, thigh, target)))
            for thigh in (.72, .74)]
    slope = ((near[1][1]['com_forward_of_centre_m']-near[0][1]['com_forward_of_centre_m'])
             / (near[1][0]-near[0][0]))
    tolerance = PATCH_HALF_EXTENT_M/abs(slope)/(max(nominal['diagonal_offsets_m'])
                                                / abs(nominal['com_forward_of_centre_m']))
    print(f'  local sensitivity {slope*1000:.1f} mm of COM x offset per rad of thigh')
    print(f'  static feasibility tolerates thigh error up to +/-{tolerance:.4f} rad '
          f'({math.degrees(tolerance):.2f} deg) before the patch capacity is exceeded')

    print('\n== 6. what this does to the speed budget ==')
    nominal_offset = max(nominal['diagonal_offsets_m'])
    shift = shift_seconds(nominal_offset)
    print(f'  nominal crouch needs a {nominal_offset*1000:.2f} mm per-pair body shift, '
          f'{shift:.4f} s')
    print(f'    not overlapped -> {stepping_speed(shift):.5f} m/s')
    print(f'    overlapped with lower+dwell+load -> '
          f'{stepping_speed(max(0., shift-LOWER_PLUS_DWELL_PLUS_LOAD_S)):.5f} m/s')
    solution, _, _ = solve_aligned(masses, joints, nominal['base_height_m'])
    if solution is not None:
        aligned = pose_geometry(masses, joints, *solution)
        print(f'  aligned crouch residual offset {max(aligned["diagonal_offsets_m"])*1000:.4f} mm'
              f' -> no per-pair shift needed')
        print(f'    speed stays at the frozen ideal ceiling {stepping_speed(0.):.5f} m/s')
        print('    the 0.0262 m/s ceiling and its asymptote v_peak/3.75 are unchanged; '
              'posture removes the statics penalty, not the timing limit')

    print('\n== 7. does the continuous lateral body motion undo the alignment? ==')
    # Stance anchors are frozen during a swing, so the body translating at the commanded
    # speed drags the COM off the fixed support diagonal.
    anchor, far = MEASURED_LOAD_WEIGHTED_M[1], MEASURED_LOAD_WEIGHTED_M[2]
    unit, _ = perpendicular_unit(anchor, far)
    lateral_projection = abs(unit[1])      # how much a pure +y body motion tilts off-diagonal
    pair_seconds = OVERHEAD_S+K_S_PER_M*LANDING_CAP_M
    travel_m = LANDING_CAP_M/2             # body lateral travel during one pair swing
    print(f'  pure +y body motion projects onto the diagonal normal with factor '
          f'{lateral_projection:.4f}')
    print(f'  one pair swing lasts {pair_seconds:.4f} s and the body travels '
          f'{travel_m*1000:.2f} mm laterally at the 60 mm step')
    for label, peak in (('start aligned, drift one full swing',
                         lateral_projection*travel_m),
                        ('bias so the COM crosses the diagonal at mid swing',
                         lateral_projection*travel_m/2)):
        verdict = 'OK' if peak <= PATCH_HALF_EXTENT_M else 'EXCEEDS CAPACITY'
        print(f'    {label:48s} peak offset {peak*1000:6.2f} mm  '
              f'vs {PATCH_HALF_EXTENT_M*1000:.2f} mm  {verdict}')
    statics_step_cap = 2*2*PATCH_HALF_EXTENT_M/lateral_projection
    print(f'  with the mid-swing bias, statics allows steps up to '
          f'{statics_step_cap*1000:.1f} mm, looser than the {LANDING_CAP_M*1000:.0f} mm '
          f'landing cap: statics stops being the binding constraint')
    print(f'  margin at the 60 mm landing cap: '
          f'{(1-lateral_projection*travel_m/2/PATCH_HALF_EXTENT_M)*100:.0f}%')

    print('\n== 8. what the sealed landing law actually does to that phase ==')
    # sol35/core35.py landing_xy35 is latched once, at the lift -> horizontal transition:
    #   raw = body_ref + layout_xy + .975*v_ref + .08*(vCOM - v_ref)
    # so the effective lead from the start of the pair's transfer is LATCH_S + LEAD_S.
    latch_s = TRANSFER_S+LIFT_S
    effective_lead_s = latch_s+FROZEN_LEAD_S
    for step_m in (.030, .0282, .040, LANDING_CAP_M):
        pair_s = OVERHEAD_S+max(.08, K_S_PER_M*step_m)
        speed = step_m/(2*pair_s)
        start = speed*(pair_s-effective_lead_s)
        end = speed*(2*pair_s-effective_lead_s)
        peak = max(abs(start), abs(end))*lateral_projection
        verdict = 'OK' if peak <= PATCH_HALF_EXTENT_M else 'EXCEEDS'
        print(f'  step {step_m*1000:5.1f} mm (v {speed:.5f} m/s, T_pair {pair_s:.4f} s): '
              f'COM lateral offset from the support pair runs '
              f'{start*1000:+7.2f} -> {end*1000:+7.2f} mm, peak perpendicular '
              f'{peak*1000:5.2f} mm  {verdict}')
    # Centring condition: midpoint of the start/end offsets is zero -> lead = 1.5 * T_pair.
    balanced_step_m = (effective_lead_s/1.5-OVERHEAD_S)/K_S_PER_M
    balanced_pair_s = OVERHEAD_S+K_S_PER_M*balanced_step_m
    print(f'  the frozen constants centre the drift only at lead = 1.5*T_pair, i.e. '
          f'T_pair {balanced_pair_s:.4f} s')
    print(f'    that is a {balanced_step_m*1000:.1f} mm step and only '
          f'{balanced_step_m/(2*balanced_pair_s):.5f} m/s')
    pair_s = OVERHEAD_S+K_S_PER_M*LANDING_CAP_M
    print(f'  at the 60 mm cap the lead constant would have to be '
          f'{1.5*pair_s-latch_s:.4f} s instead of {FROZEN_LEAD_S:.3f} s '
          f'(total lead {1.5*pair_s:.4f} s = 1.5*T_pair)')
    speed = LANDING_CAP_M/(2*pair_s)
    peak = speed*pair_s/2*lateral_projection
    print(f'    then the offset runs {-speed*pair_s/2*1000:+.2f} -> '
          f'{speed*pair_s/2*1000:+.2f} mm, peak perpendicular {peak*1000:.2f} mm, '
          f'margin {(1-peak/PATCH_HALF_EXTENT_M)*100:.0f}%')
    print('  note: T_pair depends on the projected step, which depends on the lead, so a '
          'corrected law needs a one-step fixed point or the previous pair period')
    low, high = .005, LANDING_CAP_M
    for _ in range(200):
        middle = (low+high)/2
        pair_s = OVERHEAD_S+max(.08, K_S_PER_M*middle)
        speed = middle/(2*pair_s)
        peak = speed*(2*pair_s-effective_lead_s)*lateral_projection
        low, high = (middle, high) if peak < PATCH_HALF_EXTENT_M else (low, middle)
    pair_s = OVERHEAD_S+K_S_PER_M*low
    print(f'  KEEPING the frozen 0.975 s lead, statics caps the step at {low*1000:.1f} mm '
          f'and the speed at {low/(2*pair_s):.5f} m/s')
    print('    that is a third binding constraint, tighter than both the 60 mm landing cap '
          'and the 0.02624 m/s timing ceiling')

    print('\n== 9. can longitudinal acceleration substitute for the posture fix? ==')
    # A body x-acceleration creates a d'Alembert moment m*a*h*|diagonal_y| about the support
    # line. Check it against the 30 mm longitudinal excursion limit of the frozen task.
    normal_unit, _ = perpendicular_unit(MEASURED_LOAD_WEIGHTED_M[1], MEASURED_LOAD_WEIGHTED_M[2])
    lever = MEASURED_COM_M[2]*abs(normal_unit[0])
    needed_nm = weight*diagonal_offset_m(MEASURED_COM_M, MEASURED_RIM_CONTACT_M, (1, 2))
    needed_accel = needed_nm/(total*lever)
    air_seconds = LIFT_S+K_S_PER_M*LANDING_CAP_M+LOWER_S+.08
    excursion_m = .5*needed_accel*air_seconds**2
    affordable_accel = 2*MAX_LONGITUDINAL_M/air_seconds**2
    print(f'  moment arm of a body x-acceleration about the diagonal: {lever:.4f} m')
    print(f'  to cover {needed_nm:.2f} N*m needs {needed_accel:.4f} m/s^2 held for the whole '
          f'{air_seconds:.4f} s air phase')
    print(f'    that would travel {excursion_m*1000:.1f} mm, against the frozen '
          f'{MAX_LONGITUDINAL_M*1000:.0f} mm longitudinal excursion limit')
    print(f'  the limit only affords {affordable_accel:.4f} m/s^2 -> '
          f'{total*lever*affordable_accel:.2f} N*m, '
          f'{total*lever*affordable_accel/needed_nm*100:.1f}% of what is needed')
    print('  so wheel-driven longitudinal acceleration cannot replace the posture fix; '
          'the required moment has one sign for the whole swing and cannot be oscillated')


def _calf_for_height(masses, joints, thigh, target_height_m):
    low, high = LIMITS['calf']
    for _ in range(200):
        middle = (low+high)/2
        if pose_geometry(masses, joints, thigh, middle)['base_height_m'] < target_height_m:
            low = middle
        else:
            high = middle
    return (low+high)/2


if __name__ == '__main__':
    main()
