"""Independent saved compiled-shape arithmetic; no production skill import."""
from __future__ import annotations

import numpy as np
from kinematics24 import reconstruct24, collision_bounds24


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def wheel_envelope30(kinematics, qpos, qvel, wheel_map):
    """All collision shapes on each actual wheel body, not a planned rim point.

    The frozen support function uses the actual local geom rotation and sizes.
    For a cylinder its z radius is r*sqrt(1-axis_z**2)+halfheight*abs(axis_z).
    Unknown collision primitives and missing wheel envelopes fail closed.
    """
    mapping = {int(body): int(leg) for body, leg in wheel_map.items()}
    need(set(mapping.values()) == set(range(4)), 'actual four wheel-body mapping missing')
    state = reconstruct24(kinematics, qpos, qvel)
    bounds = collision_bounds24(kinematics, state, mapping)
    wheels = [row for row in bounds if row['wheel_index'] is not None]
    low = np.full(4, np.inf)
    for row in wheels:
        leg = row['wheel_index']
        low[leg] = min(low[leg], row['minimum_world_m'][2])
    need(np.isfinite(low).all(), 'nonfinite or missing wheel collision minimum')
    return {'wheel_shape_min_z_m': low.tolist(), 'wheel_collision_bounds': wheels,
            'geometry_scope': 'actual compiled collision primitive support functions in world z; excludes geom margin inflation'}


def effective_ground_contact30(native, leg):
    return any(c['robot_wheel_index'] == leg and c['terrain_family'] is not None
               and c['efc_address'] >= 0 for c in native['contacts'])


def prior_native_gate30(five, control_index, leg):
    """Only the immediately preceding completed normal interval can authorize motion."""
    need(type(control_index) is int and control_index >= 1
         and type(leg) is int and 0 <= leg < 4 and len(five) == 5,
         'horizontal admission lacks an entire prior normal interval')
    indices = [r['native_index'] for r in five]
    need(indices == list(range(5*(control_index-1), 5*control_index)),
         'horizontal admission used a stale/future native interval')
    flags = [effective_ground_contact30(r, leg) for r in five]
    return {'prior_control_index': control_index-1, 'native_indices': indices,
            'wheel_index': leg, 'effective_ground_contact': flags,
            'five_native_contact_free': not any(flags)}


def horizontal_native_gate30(five, control_index, leg, kinematics, wheel_map):
    need(len(five) == 5 and [r['native_index'] for r in five]
         == list(range(5*control_index, 5*(control_index+1))),
         'horizontal activity does not identify its exact actual normal interval')
    heights = [wheel_envelope30(kinematics, r['after']['qpos'], r['after']['qvel'], wheel_map)
               ['wheel_shape_min_z_m'][leg] for r in five]
    flags = [effective_ground_contact30(r, leg) for r in five]
    return {'control_index': control_index, 'wheel_index': leg,
            'native_indices': [r['native_index'] for r in five],
            'actual_wheel_shape_min_z_m': heights,
            'effective_ground_contact': flags,
            'all_recorded_native_above_12mm': all(h > .012 for h in heights),
            'all_recorded_native_contact_free': not any(flags),
            'continuous_between_native_samples_not_claimed': True}
