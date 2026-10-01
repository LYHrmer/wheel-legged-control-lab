"""Independent C30 saved macro components, with no training weights or policy calls."""
from __future__ import annotations

import math
import numpy as np

LENGTH_M = .03
YAW_RAD = .12
DT_S = .01


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def yaw_of(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=float)
    need(abs(w*w+x*x+y*y+z*z-1.) <= 1e-6, 'macro state quaternion is not unit length')
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def components30(states, native, begin, end, side_start, direction, actuator_ids, torque_limits_nm):
    """Integrate the exact [begin,end) controls using actual endpoints and all 5T.

    Position/yaw integrands use each control's post endpoint. Potential change
    begins at the first pre endpoint. Signed progress saturates only above the
    fixed 30 mm goal; negative positions remain negative. Native torque cost is
    a normalized square cost, not energy. No weighted scalar reward is defined.
    """
    qpos = np.asarray(states['qpos'], dtype=float)
    times = np.asarray(states['time'], dtype=float)
    n = len(qpos)-1
    need(type(begin) is int and type(end) is int and type(side_start) is int
         and 0 <= begin < end <= n and 0 <= side_start <= n
         and type(direction) is int and direction in (-1, 1)
         and qpos.shape == (n+1, 23) and times.shape == (n+1,)
         and np.isfinite(qpos).all() and np.isfinite(times).all(), 'invalid actual macro range/state')
    need(len(native) == 5*n and abs(times[end]-times[begin]-(end-begin)*DT_S) <= 1e-9,
         'macro duration or native count differs from actual controls')
    origin = qpos[side_start, :2]
    yaw0 = yaw_of(qpos[side_start, 3:7])
    forward = np.array([math.cos(yaw0), math.sin(yaw0)])
    left = np.array([-math.sin(yaw0), math.cos(yaw0)])
    positions = qpos[begin:end+1, :2]-origin
    signed = direction*(positions@left)
    longitudinal = positions@forward
    phi = np.minimum(signed/LENGTH_M, 1.)
    dy = np.diff(signed)
    dyaw = np.array([yaw_of(q)-yaw0 for q in qpos[begin+1:end+1, 3:7]])
    dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))
    goal_error = signed[1:]-LENGTH_M
    limits = np.asarray(torque_limits_nm, dtype=float)
    ids = np.asarray(actuator_ids, dtype=int)
    need(ids.shape == limits.shape == (16,) and len(set(ids.tolist())) == 16
         and np.all(limits > 0) and np.isfinite(limits).all(), 'invalid actual actuator normalization')
    subset = native[5*begin:5*end]
    need([r['native_index'] for r in subset] == list(range(5*begin, 5*end)),
         'macro torque integration skipped/repeated an actual native return')
    torques = np.stack([np.asarray(r['after']['ctrl'], dtype=float)[ids] for r in subset])
    need(torques.shape == (5*(end-begin), 16) and np.isfinite(torques).all(), 'invalid actual native torque')
    control_cost = np.mean((torques.reshape(end-begin, 5, 16)/limits)**2, axis=(1, 2))
    longitudinal_square = DT_S*float(np.sum(longitudinal[1:]**2))
    lateral_square = DT_S*float(np.sum(goal_error**2))
    yaw_square = DT_S*float(np.sum(dyaw**2))
    progress_sum = float(np.diff(phi).sum())
    progress_delta = float(phi[-1]-phi[0])
    need(abs(progress_sum-progress_delta) <= 1e-12, 'progress potential does not telescope')
    return {'control_range': [begin, end], 'elapsed_controls': end-begin,
            'elapsed_s': (end-begin)*DT_S, 'actual_native_steps': 5*(end-begin),
            'direction': direction, 'reference_side_start_control': side_start,
            'potential_start': float(phi[0]), 'potential_end': float(phi[-1]),
            'progress_potential_difference': progress_delta,
            'progress_sum_of_control_differences': progress_sum,
            'signed_net_displacement_m': float(signed[-1]-signed[0]),
            'signed_position_start_m': float(signed[0]), 'signed_position_end_m': float(signed[-1]),
            'backtrack_m': float(np.maximum(-dy, 0.).sum()),
            'backtrack_normalized': float(np.maximum(-dy, 0.).sum()/LENGTH_M),
            'longitudinal_square_integral_m2_s': longitudinal_square,
            'longitudinal_normalized_square_integral_s': longitudinal_square/LENGTH_M**2,
            'lateral_goal_error_square_integral_m2_s': lateral_square,
            'lateral_goal_error_normalized_square_integral_s': lateral_square/LENGTH_M**2,
            'yaw_error_square_integral_rad2_s': yaw_square,
            'yaw_error_normalized_square_integral_s': yaw_square/YAW_RAD**2,
            'torque_normalized_square_integral_s': DT_S*float(control_cost.sum()),
            'torque_normalized_square_mean': float(control_cost.mean()),
            'reward_weights_defined': False, 'weighted_scalar_reward': None,
            'torque_cost_is_energy': False}


def macro_partition30(events, run_end, side_start):
    """Derive half-open macro intervals from the actual latch event indices."""
    need(1 <= len(events) <= 4 and type(run_end) is int,
         'skill must save its actual one-to-four leg decisions')
    starts = [event['control_index'] for event in events]
    need(starts[0] == side_start and all(type(i) is int for i in starts)
         and starts == sorted(set(starts)) and starts[-1] < run_end,
         'macro latch is not a distinct executed control boundary')
    need([event['macro_index'] for event in events] == list(range(len(events))),
         'macro decision index skipped or repeated')
    return list(zip(starts, starts[1:]+[run_end]))


def verify_macro_receipts30(episode, latches, macros, direction, actuator_ids, limits_nm,
                            expected_terminal_reason):
    """Compare actual transitions to independently integrated saved controls.

    The caller derives terminal reason from the actual finish/failure and
    retention boundary; this function never calls the production macro writer.
    """
    states, rows, native = episode['states'], episode['rows'], episode['native']
    side = [i for i, r in enumerate(rows) if r['actor'] == 'traditional_side']
    need(bool(side), 'macro receipts require an actual admitted side action')
    ranges = macro_partition30(latches, len(rows), side[0])
    need(len(macros) == len(ranges), 'macro count is not actual latch-event count')
    reason_set = {'success_after_retention', 'safe_cancel', 'physical_failure',
                  'budget_truncated', 'software_failure'}
    need(expected_terminal_reason in reason_set, 'invalid independently derived terminal reason')
    key_map = {'progress': 'progress_potential_difference', 'backtrack': 'backtrack_normalized',
               'elapsed_s': 'elapsed_s',
               'longitudinal_square_m2_s': 'longitudinal_square_integral_m2_s',
               'longitudinal_square_normalized_s': 'longitudinal_normalized_square_integral_s',
               'lateral_goal_square_m2_s': 'lateral_goal_error_square_integral_m2_s',
               'lateral_goal_square_normalized_s': 'lateral_goal_error_normalized_square_integral_s',
               'yaw_square_rad2_s': 'yaw_error_square_integral_rad2_s',
               'yaw_square_normalized_s': 'yaw_error_normalized_square_integral_s',
               'actual_torque_square_normalized_s': 'torque_normalized_square_integral_s'}
    outputs = []
    for index, ((begin, end), event, saved) in enumerate(zip(ranges, latches, macros)):
        reason = 'next_leg' if index+1 < len(ranges) else expected_terminal_reason
        need(saved['schema'] == 'd1-c30-event-transition-v1'
             and saved['macro_index'] == index and saved['control_start'] == begin
             and saved['control_end'] == end and saved['elapsed_controls'] == end-begin
             and saved['terminal_reason'] == reason
             and saved['terminated'] is (reason in ('success_after_retention', 'safe_cancel', 'physical_failure'))
             and saved['truncated'] is (reason in ('budget_truncated', 'software_failure'))
             and saved['reward_weights_defined'] is False and saved['weighted_reward'] is None,
             'saved macro range/termination differs from actual boundaries')
        for key in ('proposed_action', 'projected_action', 'consumed_action'):
            action = np.asarray(saved[key], dtype=float)
            need(action.shape == (3,) and np.isfinite(action).all()
                 and np.array_equal(action, np.asarray(event[key]))
                 and np.all(np.abs(action[:2]) <= .012+1e-15) and 0 <= action[2] <= 1,
                 'macro action is not its actual finite bounded leg latch')
        need(saved['projection_reason'] == event['reason'], 'macro projection reason changed')
        for name, boundary, mi in (('observation', begin, index), ('next_observation', end, index+1)):
            obs = saved[name]
            need(obs['schema'] == 'd1-c30-event-observation-raw-v1'
                 and obs['control_boundary'] == boundary and obs['macro_index'] == mi
                 and obs['direction'] == direction and obs['source'] == 'actual_control_endpoint'
                 and obs['learning_normalization_defined'] is False
                 and obs['time_s'] == float(states['time'][boundary])
                 and np.array_equal(obs['qpos'], states['qpos'][boundary])
                 and np.array_equal(obs['qvel'], states['qvel'][boundary]),
                 'event observation is not its actual saved endpoint')
        calculated = components30(states, native, begin, end, side[0], direction, actuator_ids, limits_nm)
        actual = saved['reward_components']
        need(set(actual) == set(key_map) and all(math.isfinite(actual[k])
             and math.isclose(actual[k], calculated[v], rel_tol=1e-10, abs_tol=1e-11)
             for k, v in key_map.items()), 'macro reward components differ from independent native/state sum')
        outputs.append(calculated)
    # Preparation remains outside macros but inside full-scene accounting.
    full = components30(states, native, 0, len(rows), side[0], direction, actuator_ids, limits_nm)
    return {'passed': True, 'actual_macro_transitions': len(outputs),
            'macro_control_ranges': [list(r) for r in ranges],
            'preparation_controls_outside_macros': side[0],
            'macro_components': outputs, 'full_scene_components': full,
            'weighted_training_reward_defined': False, '100Hz_controls_not_relabelled_as_macro_samples': True}
