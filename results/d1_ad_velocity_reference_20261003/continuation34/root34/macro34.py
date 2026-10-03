"""Actual-D C34 macro components on the unchanged C31 event/native chain."""
from __future__ import annotations

import numpy as np

from macro31 import build_macros31


def distance_components34(qpos, *, origin_xy, yaw0, direction, distance_m):
    q = np.asarray(qpos, dtype=float)
    if (distance_m not in (.03, .04) or direction not in (-1, 1)
            or q.ndim != 2 or q.shape[1] != 23 or len(q) < 2
            or not np.isfinite(q).all()):
        raise ValueError('C34 actual-D macro endpoint array differs')
    c, s = np.cos(yaw0), np.sin(yaw0)
    xy = q[:, :2]-np.asarray(origin_xy, dtype=float)
    longitudinal = xy @ np.array([c, s])
    lateral = direction*(xy @ np.array([-s, c]))
    distance = float(distance_m)
    return {
        'progress_potential_difference': float(np.minimum(lateral[-1]/distance, 1.)-
            np.minimum(lateral[0]/distance, 1.)),
        'backtrack_normalized': float(np.maximum(-np.diff(lateral), 0.).sum()/distance),
        'longitudinal_normalized_square_integral_s':
            .01*float(np.dot(longitudinal[1:], longitudinal[1:]))/distance**2,
        'lateral_goal_error_square_integral_m2_s':
            .01*float(np.sum((lateral[1:]-distance)**2)),
        'lateral_goal_error_normalized_square_integral_s':
            .01*float(np.sum((lateral[1:]-distance)**2))/distance**2,
    }


def build_macros34(states, rows, events, *, torque_limits, origin_xy, yaw0,
                   direction, end_reason, global_control_offset=0, distance_m):
    result = build_macros31(states, rows, events, torque_limits=torque_limits,
        origin_xy=origin_xy, yaw0=yaw0, direction=direction,
        end_reason=end_reason, global_control_offset=global_control_offset)
    qpos = np.asarray(states['qpos'], dtype=float)
    for transition in result['transitions']:
        begin, end = transition['control_range']
        components = transition['reward_components']
        if distance_m == .04:
            components.update(distance_components34(qpos[begin:end+1],
                origin_xy=origin_xy, yaw0=yaw0, direction=direction,
                distance_m=distance_m))
        components['target_distance_m'] = distance_m
        components['source_component_schema'] = 'd1-c34-actual-D-macros-v1'
        transition['schema'] = 'd1-c34-actual-event-transition-v1'
        transition['legacy_30mm_diagnostics'] = transition['original_reward_components30']
        transition['training'] = False
    full = result['full_case_components']
    if distance_m == .04:
        full.update(distance_components34(qpos, origin_xy=origin_xy, yaw0=yaw0,
            direction=direction, distance_m=distance_m))
    full['target_distance_m'] = distance_m
    full['source_component_schema'] = 'd1-c34-actual-D-macros-v1'
    result['legacy_30mm_full_case_diagnostics'] = result['original_full_case_components30']
    result['schema'] = 'd1-c34-actual-event-macros-v1'
    result['target_distance_m'] = distance_m
    result['training'] = False
    return result
