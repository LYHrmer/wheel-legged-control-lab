"""Independent actual-D macro proof with complete old native cost accounting."""
from __future__ import annotations

import copy
import math
import numpy as np

from reader26 import require, close
from reward_read30 import verify_macro_receipts30, yaw_of

MAP = dict(progress='progress_potential_difference',
    backtrack='backtrack_normalized', elapsed_s='elapsed_s',
    longitudinal_square_m2_s='longitudinal_square_integral_m2_s',
    longitudinal_square_normalized_s='longitudinal_normalized_square_integral_s',
    lateral_goal_square_m2_s='lateral_goal_error_square_integral_m2_s',
    lateral_goal_square_normalized_s='lateral_goal_error_normalized_square_integral_s',
    yaw_square_rad2_s='yaw_error_square_integral_rad2_s',
    yaw_square_normalized_s='yaw_error_normalized_square_integral_s',
    actual_torque_square_normalized_s='torque_normalized_square_integral_s')
D_FIELDS = ('progress_potential_difference', 'backtrack_normalized',
            'longitudinal_normalized_square_integral_s',
            'lateral_goal_error_square_integral_m2_s',
            'lateral_goal_error_normalized_square_integral_s')


def actual_distance_components34(states, begin, end, side_start, direction,
                                 distance_m):
    """Recompute D-sensitive integrands from every physical state endpoint."""
    q = np.asarray(states['qpos'], dtype=float)
    require(distance_m in (.03, .04) and direction in (-1, 1)
            and 0 <= side_start < len(q) and 0 <= begin < end < len(q),
            'C34 macro actual-D range/target differs')
    yaw = yaw_of(q[side_start, 3:7])
    direction_xy = direction*np.array([-math.sin(yaw), math.cos(yaw)])
    along = np.array([math.cos(yaw), math.sin(yaw)])
    displacement = q[begin:end+1, :2]-q[side_start, :2]
    signed = displacement@direction_xy
    longitudinal = displacement@along
    potential = np.minimum(signed/distance_m, 1.)
    error = signed[1:]-distance_m
    return dict(progress_potential_difference=float(potential[-1]-potential[0]),
        backtrack_normalized=float(np.sum(np.maximum(-np.diff(signed), 0.))/distance_m),
        longitudinal_normalized_square_integral_s=float(.01*np.sum(longitudinal[1:]**2)/distance_m**2),
        lateral_goal_error_square_integral_m2_s=float(.01*np.sum(error**2)),
        lateral_goal_error_normalized_square_integral_s=float(.01*np.sum(error**2)/distance_m**2))


def canonical34(saved, computed, interval, distance_m):
    expected = {key: computed[key] for key in MAP.values()}
    expected.update(control_range=interval, elapsed_controls=interval[1]-interval[0],
        reward_weights_defined=False, weighted_scalar_reward=None,
        torque_cost_is_energy=False,
        source_component_schema='d1-c34-actual-D-macros-v1',
        target_distance_m=distance_m)
    require(set(saved) == set(expected), 'C34 actual-D canonical component keys differ')
    for key, value in expected.items():
        require(saved[key] == value if value is None or isinstance(value, (str, bool, list))
                else close(saved[key], value),
                'C34 actual-D canonical component differs: '+key)


def verify_macro_receipts34(episode, events, saved, direction, actuator_ids,
                            limits_nm, teacher_reason, *, distance_m):
    require(saved['schema'] == 'd1-c34-actual-event-macros-v1'
            and saved['target_distance_m'] == distance_m
            and saved['training'] is False,
            'C34 actual-D macro archive header differs')
    transitions = saved['transitions']
    legacy = []
    for row in transitions:
        require(row['schema'] == 'd1-c34-actual-event-transition-v1'
                and row['training'] is False
                and row['legacy_30mm_diagnostics'] == row['original_reward_components30'],
                'C34 actual-D transition or legacy diagnostic differs')
        old = copy.deepcopy(row)
        old['schema'] = 'd1-c30-event-transition-v1'
        old['reward_components'] = row['original_reward_components30']
        legacy.append(old)
    prior = verify_macro_receipts30(episode, events, legacy, direction,
                                     actuator_ids, limits_nm, teacher_reason)
    require(saved['legacy_30mm_full_case_diagnostics'] ==
            saved['original_full_case_components30'],
            'C34 legacy full-case diagnostic differs')
    side_start = prior['macro_control_ranges'][0][0]
    result = []
    for row, old, interval in zip(transitions, prior['macro_components'],
                                  prior['macro_control_ranges']):
        modified = dict(old)
        modified.update(actual_distance_components34(
            episode['states'], *interval, side_start, direction, distance_m))
        canonical34(row['reward_components'], modified, interval, distance_m)
        result.append(modified)
    full = dict(prior['full_scene_components'])
    full.update(actual_distance_components34(episode['states'], 0,
                episode['controls'], side_start, direction, distance_m))
    canonical34(saved['full_case_components'], full,
                [0, episode['controls']], distance_m)
    require(set(saved['original_full_case_components30']) == set(MAP)
            and all(close(saved['original_full_case_components30'][key],
                          prior['full_scene_components'][value])
                    for key, value in MAP.items()),
            'C34 legacy full-case raw/native components differ')
    return dict(prior, macro_components=result, full_scene_components=full,
                actual_target_distance_m=distance_m,
                legacy_30mm_diagnostics_verified=True)
