"""Self-collision and wheel-interference check for the corrected lateral gait.

Closes the last pure item left open by docs/ad_lateral_gait_review_20261004.md section 7:
the workspace scan there verified only compiled joint ranges and forward-kinematic residual,
not whether the legs or wheels actually interfere at the aligned crouch or along a wider
lateral step.

Method: every one of the 42 URDF collision primitives is represented as an oriented
bounding box in the body frame (a cylinder becomes its tight box, which is strictly larger,
so a separated pair of boxes guarantees separated real shapes). Pairwise separation uses the
separating-axis theorem over all 15 candidate axes; the maximum positive projected gap is a
valid lower bound on the true distance, and a non-positive result means the boxes overlap.

Pure arithmetic over the URDF tree: no MuJoCo, no model construction, no physics, no model
load, no fitting. Kinematics come from posture_statics_20261004, validated there against
the saved C35 pre-state.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from aligned_crouch_feasibility_20261005 import foot_of, solve_ik, within_limits
from motion_optimisation_20261005 import ALIGNED_CALF_RAD, ALIGNED_THIGH_RAD
from posture_statics_20261004 import (
    URDF,
    angle_vector,
    forward,
    matmul,
    parse,
    rpy_matrix,
    uniform,
)

LEGS = ('FL', 'FR', 'RL', 'RR')
FRONT_BASE = ('F_base_link', 'F_funtion_link', 'F_docking_link')
REAR_BASE = ('R_base_link', 'R_funtion_link', 'R_docking_link')
BASE_GROUP = ('base_link', 'imu_link')+FRONT_BASE+REAR_BASE
STATICS_STEP_M = .0775           # two-support statics limit, gait_review section 4.3
SHORT_LIFT_M = .015              # optimised lift height, clearance + 3 mm
SWING_PAIRS = (('FL', 'RR'), ('FR', 'RL'))


def read_collisions():
    """Collision primitives as (link, local_offset, local_rotation, half_extents)."""
    out = []
    for link in ET.parse(URDF).getroot().findall('link'):
        for collision in link.findall('collision'):
            origin = collision.find('origin')
            offset = ((0., 0., 0.) if origin is None or origin.get('xyz') is None
                      else tuple(float(v) for v in origin.get('xyz').split()))
            rpy = ((0., 0., 0.) if origin is None or origin.get('rpy') is None
                   else tuple(float(v) for v in origin.get('rpy').split()))
            shape = next(iter(collision.find('geometry')))
            if shape.tag == 'box':
                size = tuple(float(v) for v in shape.get('size').split())
                half = tuple(value/2 for value in size)
                kind = 'box'
            elif shape.tag == 'cylinder':
                radius, length = float(shape.get('radius')), float(shape.get('length'))
                half = (radius, radius, length/2)      # tight box, strictly larger
                kind = 'cylinder'
            else:
                raise ValueError('unsupported collision shape '+shape.tag)
            out.append({'link': link.get('name'), 'offset': offset,
                        'rotation': rpy_matrix(*rpy), 'half': half, 'kind': kind})
    return out


def place(primitives, frames):
    """Express each primitive as a world-frame oriented box."""
    placed = []
    for item in primitives:
        rotation, position = frames[item['link']]
        axes = matmul(rotation, item['rotation'])
        shifted = tuple(position[axis]+sum(rotation[axis][k]*item['offset'][k]
                                           for k in range(3)) for axis in range(3))
        placed.append({'link': item['link'], 'kind': item['kind'], 'centre': shifted,
                       'axes': [tuple(axes[row][column] for row in range(3))
                                for column in range(3)], 'half': item['half']})
    return placed


def separation(first, second):
    """SAT lower bound on the distance between two oriented boxes; <=0 means overlap."""
    delta = tuple(second['centre'][axis]-first['centre'][axis] for axis in range(3))
    candidates = list(first['axes'])+list(second['axes'])
    for one in first['axes']:
        for other in second['axes']:
            cross = (one[1]*other[2]-one[2]*other[1],
                     one[2]*other[0]-one[0]*other[2],
                     one[0]*other[1]-one[1]*other[0])
            norm = math.sqrt(sum(value*value for value in cross))
            if norm > 1e-9:
                candidates.append(tuple(value/norm for value in cross))
    best = -math.inf
    for axis in candidates:
        centre_gap = abs(sum(delta[k]*axis[k] for k in range(3)))
        reach = sum(first['half'][index]*abs(sum(first['axes'][index][k]*axis[k]
                                                for k in range(3))) for index in range(3))
        reach += sum(second['half'][index]*abs(sum(second['axes'][index][k]*axis[k]
                                                  for k in range(3))) for index in range(3))
        best = max(best, centre_gap-reach)
    return best


def leg_of(link):
    return link[:2] if link[:2] in LEGS else None


def exempt(first, second):
    """Pairs whose proximity is structural, not a collision."""
    one, other = leg_of(first), leg_of(second)
    if one is not None and one == other:
        return 'same leg'
    if first in BASE_GROUP and second in BASE_GROUP:
        return 'base group'
    for leg, group in (('FL', FRONT_BASE), ('FR', FRONT_BASE),
                       ('RL', REAR_BASE), ('RR', REAR_BASE)):
        attached = (f'{leg}_hip', f'{leg}_thigh')
        if (first in attached and second in group) or (second in attached and first in group):
            return 'leg attachment'
        if (first in attached and second in ('base_link', 'imu_link')) or \
           (second in attached and first in ('base_link', 'imu_link')):
            return 'leg attachment'
    return None


def configure(masses, joints, offsets):
    """Joint angles for a set of per-leg foot offsets from the aligned crouch."""
    angles = angle_vector(uniform(ALIGNED_THIGH_RAD, ALIGNED_CALF_RAD))
    for leg, offset in offsets.items():
        base_foot = foot_of(masses, joints, leg, angles)
        target = tuple(base_foot[axis]+offset[axis] for axis in range(3))
        solved = solve_ik(masses, joints, leg, angles, target)
        if solved is None or not within_limits(solved, leg):
            return None
        for name in ('hip', 'thigh', 'calf'):
            angles[f'{leg}_{name}_joint'] = solved[f'{leg}_{name}_joint']
    return angles


def worst_pairs(masses, joints, primitives, angles, limit=6):
    identity = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    frames, _, _, _ = forward(masses, joints, angles, (0., 0., 0.), identity)
    placed = place(primitives, frames)
    rows = []
    for index, first in enumerate(placed):
        for second in placed[index+1:]:
            if exempt(first['link'], second['link']):
                continue
            rows.append((separation(first, second), first['link'], first['kind'],
                         second['link'], second['kind']))
    rows.sort(key=lambda row: row[0])
    return rows[:limit], len(rows)


def main() -> None:
    masses, joints = parse()
    primitives = read_collisions()
    print('== 1. collision model ==')
    print(f'  {len(primitives)} URDF collision primitives, '
          f'{sum(1 for p in primitives if p["kind"] == "cylinder")} cylinders replaced by '
          f'their tight bounding boxes (strictly larger, so separation is conservative)')
    base = angle_vector(uniform(ALIGNED_THIGH_RAD, ALIGNED_CALF_RAD))
    _, pair_count = worst_pairs(masses, joints, primitives, base)
    print(f'  {pair_count} checked pairs after exempting same-leg, base-group and '
          f'leg-attachment proximity')

    print('\n== 2. aligned crouch, all feet nominal ==')
    rows, _ = worst_pairs(masses, joints, primitives, base)
    for gap, one, kind_one, other, kind_other in rows:
        print(f'  {gap*1000:8.2f} mm   {one} ({kind_one})  vs  {other} ({kind_other})')

    print('\n== 3. along a lateral swing of the diagonal pair ==')
    print('  swing pair lifted 15 mm and displaced laterally; stance pair held')
    overall = (math.inf, None)
    for pair in SWING_PAIRS:
        for sign, label in ((+1, 'outward +y'), (-1, 'inward -y')):
            worst = (math.inf, None)
            blocked = None
            for fraction in range(11):
                shift = sign*STATICS_STEP_M*fraction/10
                offsets = {leg: (0., shift, SHORT_LIFT_M) for leg in pair}
                angles = configure(masses, joints, offsets)
                if angles is None:
                    blocked = shift
                    break
                rows, _ = worst_pairs(masses, joints, primitives, angles, limit=1)
                if rows[0][0] < worst[0]:
                    worst = (rows[0][0], (shift, rows[0]))
            if blocked is not None:
                print(f'  pair {pair} {label}: IK or joint limit blocked at '
                      f'{blocked*1000:.1f} mm')
                continue
            shift, row = worst[1]
            print(f'  pair {pair} {label}: minimum {worst[0]*1000:7.2f} mm at '
                  f'{shift*1000:+6.1f} mm  ({row[1]} vs {row[3]})')
            if worst[0] < overall[0]:
                overall = (worst[0], (pair, label, shift, row))

    print('\n== 4. where does it actually interfere? ==')
    print('  pushing the inward step past the statics limit to find the geometric edge')
    for pair in SWING_PAIRS:
        first_touch = None
        limit_block = None
        last_gap = None
        for millimetres in range(0, 301, 5):
            offsets = {leg: (0., -millimetres/1000, SHORT_LIFT_M) for leg in pair}
            angles = configure(masses, joints, offsets)
            if angles is None:
                limit_block = millimetres
                break
            rows, _ = worst_pairs(masses, joints, primitives, angles, limit=1)
            last_gap = (millimetres, rows[0])
            if rows[0][0] <= 0:
                first_touch = (millimetres, rows[0])
                break
        if first_touch is not None:
            shift, row = first_touch
            print(f'  pair {pair} inward: first interference at {shift} mm '
                  f'({row[1]} vs {row[3]})')
        elif limit_block is not None:
            shift, row = last_gap
            print(f'  pair {pair} inward: joint limit or IK stops it at {limit_block} mm, '
                  f'still {row[0]*1000:.2f} mm clear at {shift} mm ({row[1]} vs {row[3]})')
        else:
            shift, row = last_gap
            print(f'  pair {pair} inward: no interference out to {shift} mm, '
                  f'{row[0]*1000:.2f} mm clear ({row[1]} vs {row[3]})')
    print(f'  so the binding limit on step size stays two-support statics at '
          f'{STATICS_STEP_M*1000:.1f} mm, not geometry')

    print('\n== 5. verdict ==')
    gap, detail = overall
    pair, label, shift, row = detail
    print(f'  worst separation anywhere on the scanned envelope: {gap*1000:.2f} mm')
    print(f'    pair {pair} {label} at {shift*1000:+.1f} mm, {row[1]} vs {row[3]}')
    if gap > 0:
        print('  no self-collision: every checked pair stays separated across the full '
              f'+/-{STATICS_STEP_M*1000:.1f} mm statics-limited step at the 15 mm lift')
    else:
        print('  SELF-COLLISION: the scanned envelope is not clear')
    print('  conservative in three ways: cylinders are replaced by larger boxes, the SAT '
          'bound under-reports the true distance, and both swing feet are displaced to the '
          'same extreme simultaneously')
    print('  not covered: mesh-level geometry (the URDF collision primitives are what the '
          'compiled model uses), terrain contact, and the base-mounted 175 mm guide rollers '
          'are included as modelled')


if __name__ == '__main__':
    main()
