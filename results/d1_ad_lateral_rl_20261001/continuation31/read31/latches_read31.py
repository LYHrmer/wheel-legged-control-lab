"""Pure saved C31 latent/action/54D/IK projection joins at real leg boundaries.

The caller first loads the frozen cold-reader helpers and independently checks
the episode's original global ledger and local physical-state view.
"""
from __future__ import annotations

import math
import numpy as np

from reader26 import require, close
from side_math27 import compiled_side_geometry, support_margin
from math_read31 import observation54, action3, gaussian_log_probability, linear_outputs


def verify_joint_map31(saved, construction, binding):
    actual = construction['geometry_binding']
    kin = construction['side_kinematic_binding']['kinematics24']
    limits = construction['skill30_joint_limits']
    ids = np.asarray(actual['joint_ids'], dtype=int)
    qa, va = np.asarray(actual['joint_qpos_addresses']), np.asarray(actual['joint_dof_addresses'])
    act = np.asarray(actual['actuator_ids'], dtype=int)
    trnid, trntype = np.asarray(actual['actuator_trnid']), np.asarray(actual['actuator_trntype'])
    require(np.array_equal(qa, np.asarray(kin['jnt_qposadr'])[ids])
            and np.array_equal(va, np.asarray(kin['jnt_dofadr'])[ids])
            and np.array_equal(trnid[act, 0], ids) and np.all(trntype[act] == 0),
            '54D source actuator/joint/free-state map differs from compiled robot')
    for key, expected in (('joint_ids', ids), ('joint_qpos_addresses', qa), ('joint_dof_addresses', va),
                          ('actuator_ids', act), ('actuator_joint_ids', trnid[act, 0]),
                          ('jnt_range', np.asarray(limits['jnt_range'])[ids]),
                          ('jnt_limited', np.asarray(limits['jnt_limited'])[ids])):
        require(np.array_equal(saved[key], expected), 'saved observation compiled map differs: '+key)
    wheel_bodies = [int(body) for leg in range(4) for body, owner in binding['wheel_map'].items() if owner == leg]
    require(saved['wheel_body_ids_by_leg'] == wheel_bodies, '54D leg order differs from actual wheel ancestry')


def verify_latches31(episode, events, construction, binding, *, cycle_index, direction, mode,
                     rollout_parameters=None):
    rows, states = episode['rows'], episode['states']
    kin = construction['side_kinematic_binding']['kinematics24']
    limits = construction['skill30_joint_limits']
    order = [2, 0, 3, 1] if direction == 1 else [3, 1, 2, 0]
    require(mode in ('zero', 'fixed', 'learned', 'train') and direction in (-1, 1)
            and [e for row in rows for e in row['macro_events30']] == events
            and 1 <= len(events) <= 4 and events[0]['control_index'] == 200,
            'C31 real latch set/start differs')
    previous_action = np.zeros(3)
    summaries = []
    for mi, event in enumerate(events):
        i = event['control_index']
        row = rows[i]
        record = row['info']['controller_record']
        before, after = record['diagnostic_before'], record['diagnostic_after']
        latch = event['skill31_latch']
        source = latch['observation31']
        require(event['schema'] == 'd1-c30-lateral-reference-skill-v1'
                and latch['schema'] == 'd1-c31-event-lateral-latch-v1'
                and event['macro_index'] == source['macro_index'] == mi
                and event['leg_index'] == source['leg_index'] == order[mi]
                and event['latch_control_index'] == latch['local_control_index'] == i
                and latch['cycle_index'] == cycle_index and latch['mode'] == mode
                and source['direction'] == direction and row['actor'] == 'traditional_side',
                'C31 actual cycle/local-control/leg identity differs')
        require(np.array_equal(source['pre_qpos23'], states['qpos'][i])
                and np.array_equal(source['pre_qvel22'], states['qvel'][i])
                and close(source['initial_xy_m'], states['qpos'][200, :2])
                and close(source['initial_yaw_rad'], before['start_yaw_rad'])
                and source['completed_side_controls'] == i-200
                and close(source['side_elapsed_s'], .01*(i-200))
                and np.array_equal(source['previous_consumed_action'], previous_action),
                'latch observation used future/old-episode state or reference action')
        verify_joint_map31(source['compiled_joint_map31'], construction, binding)
        observation = observation54(source)
        require(close(source['raw54'], observation['raw54'])
                and close(source['normalized54'], observation['normalized54'])
                and np.array_equal(source['clip_mask54'], observation['clip_mask54']), '54D observation arithmetic differs')
        original_from, original_to = (np.asarray(event[key], dtype=float)
                                     for key in ('teacher_body_from_m', 'teacher_body_to_m'))
        require(close(original_from, before['pose']) and close(event['consumed_body_from_m'], original_from),
                'teacher original waypoint source differs')
        for key in ('teacher_body_from_m', 'teacher_body_to_m', 'teacher_shift_time_s', 'plan_com_height_m'):
            require(close(source[key], event[key]), 'policy observed a different teacher plan')
        yaw = float(event['initial_body_yaw_rad'])
        require(close(yaw, before['start_yaw_rad']) and close(yaw, source['initial_yaw_rad']), 'initial heading differs')
        if mode in ('train', 'learned'):
            require(rollout_parameters is not None, 'actual event policy parameters are required')
            mean, value = linear_outputs(rollout_parameters, observation['normalized54'][None, :])
            require(close(latch['old_mean_z3'], mean[0]) and latch['latent_std_z3'] == [.35]*3
                    and close(latch['old_log_prob'], gaussian_log_probability(latch['latent_z3'], mean[0]))
                    and latch['latch_callback_calls_cumulative'] == mi+1, 'sampled latent/model call differs')
            if mode == 'learned':
                require(np.array_equal(latch['latent_z3'], latch['old_mean_z3']) and latch['old_value'] is None,
                        'evaluation did not use deterministic actor-only mean')
            else:
                require(close(latch['old_value'], value[0]), 'old rollout value differs')
            receipt = latch['policy_receipt']
            require(receipt['mode'] == mode and receipt['cycle_index'] == cycle_index
                    and receipt['local_control_index'] == i, 'policy receipt references another boundary')
        else:
            require(all(latch[key] is None for key in ('latent_z3', 'old_mean_z3', 'latent_std_z3',
                         'old_log_prob', 'old_value', 'policy_receipt'))
                    and latch['latch_callback_calls_cumulative'] == 0, 'zero/fixed invoked lateral policy')
        actions = action3(original_from, original_to, yaw, latch['latent_z3'], mode)
        proposed, projected = actions['proposed'], actions['projected']
        rotate = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        if mode == 'zero':
            consumed = np.zeros(3)
            reason = 'teacher_exact_zero'
            require(event['extra_plan_candidate_qpos23'] is None and event['extra_plan_ik_error_m'] is None,
                    'zero branch performed extra plan IK')
        else:
            candidate = original_to.copy()
            candidate[:2] += rotate@projected[:2]
            consumed = np.array([0., 0., projected[2]])
            if np.linalg.norm(candidate[:2]-original_from[:2]) > .120:
                reason = 'max_shift_rejected_xy'
                require(event['extra_plan_candidate_qpos23'] is None, 'span rejection performed extra IK')
            else:
                q = np.asarray(event['extra_plan_candidate_qpos23'], dtype=float)
                require(q.shape == (23,) and np.isfinite(q).all() and close(q[:3], candidate)
                        and close(q[3:7], [math.cos(yaw/2), 0., 0., math.sin(yaw/2)]), 'IK candidate qpos differs')
                geometry = compiled_side_geometry(kin, q, np.zeros(22), binding['wheel_map'])
                stance = [leg for leg in range(4) if leg != order[mi]]
                margin = support_margin(geometry['contact_points_m'][stance, :2], geometry['whole_com_m'][:2])
                require(close(geometry['contact_points_m'], event['extra_plan_contact_points_m'], atol=2e-9)
                        and close(geometry['whole_com_m'], event['extra_plan_com_m'], atol=2e-9)
                        and close(margin, event['predicted_tripod_static_margin_m'], atol=2e-9), 'IK geometry proof differs')
                ids = np.asarray(construction['geometry_binding']['joint_ids'], dtype=int)
                qa = np.asarray(binding['qpos_addresses'], dtype=int)
                bounds = np.asarray(limits['jnt_range'])[ids]
                limited = np.asarray(limits['jnt_limited'], dtype=bool)[ids]
                inside = np.all(q[qa][limited] >= bounds[limited, 0]-1e-12) and np.all(q[qa][limited] <= bounds[limited, 1]+1e-12)
                error = event['extra_plan_ik_error_m']
                accepted = math.isfinite(error) and error <= .008 and margin >= .028 and inside
                reason = 'body_xy_projected_and_consumed' if accepted else 'extra_plan_IK_or_tripod_margin_rejected_xy'
                if accepted:
                    consumed = projected.copy()
        for key, expected in (('proposed_action', proposed), ('projected_action', projected), ('consumed_action', consumed)):
            require(close(event[key], expected) and close(latch[key], expected), 'latent/projection/consumed action differs')
        require(event['reason'] == latch['projection_reason'] == reason, 'projection reason differs')
        expected_to = original_to.copy()
        expected_to[:2] += rotate@consumed[:2]
        require(close(event['consumed_body_to_m'], expected_to) and close(after['body_to'], expected_to),
                'actual reference did not consume the latched action')
        geometry = compiled_side_geometry(kin, states['qpos'][i], states['qvel'][i], binding['wheel_map'])
        height = .45 if mi == 0 else float(np.clip(geometry['whole_com_m'][2]-geometry['contact_points_m'][:, 2].min(), .10, 1.))
        require(close(event['plan_com_height_m'], height), 'teacher acceleration used another COM height')
        def duration(target):
            return max(.30, math.sqrt(max(1e-9, 5.7735*np.linalg.norm(target[:2]-original_from[:2])*height/(9.81*.015))))
        require(close(event['teacher_shift_time_s'], duration(original_to))
                and close(event['consumed_shift_time_s'], duration(expected_to))
                and close(after['shift_time_s'], event['consumed_shift_time_s']), 'original timing bound changed')
        previous_action = consumed
        summaries.append(dict(macro_index=mi, local_control_index=i, consumed_action=consumed.tolist(),
                              projection_reason=reason, observation_reconstructed=True))
    return summaries
